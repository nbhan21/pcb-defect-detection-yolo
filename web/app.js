import { csvForDetections, formatApiError, createRequestGate } from './helpers.mjs';

const $ = (id) => document.getElementById(id);
const CLASS_NAMES = ['mouse_bite', 'spur', 'missing_hole', 'short', 'open_circuit', 'spurious_copper'];
const MAX_BYTES = 10 * 1024 * 1024;
const MAX_PIXELS = 25_000_000;
const gate = createRequestGate();
const state = { file: null, originalUrl: null, resultUrls: [], previewReady: false, modelReady: false, modelCheckpoint: null, busy: false, controller: null, confidenceTouched: false, healthBusy: false };

function message(text, kind = '') {
  $('result-message-text').textContent = text;
  $('result-message').className = `result-message ${kind}`;
}

function summary(counts = null) {
  $('class-summary').replaceChildren(...CLASS_NAMES.map((name) => {
    const item = document.createElement('div');
    const count = counts == null ? null : Number(counts[name] ?? 0);
    item.className = `class-item${count > 0 ? ' has-findings' : ''}`;
    const label = document.createElement('span'); label.textContent = name;
    const value = document.createElement('strong'); value.textContent = count == null ? '—' : String(count);
    item.append(label, value);
    return item;
  }));
}

function switchView(view) {
  const result = view === 'result' && !$('tab-result').disabled;
  $('pane-original').hidden = result;
  $('pane-result').hidden = !result;
  $('tab-original').setAttribute('aria-selected', String(!result));
  $('tab-result').setAttribute('aria-selected', String(result));
  $('tab-original').tabIndex = result ? -1 : 0;
  $('tab-result').tabIndex = result ? 0 : -1;
}

function syncControls() {
  $('inspect-button').disabled = !state.file || !state.previewReady || !state.modelReady || state.busy;
  $('inspect-button-label').textContent = state.busy ? 'SEDANG MEMPROSES' : 'JALANKAN INSPEKSI';
  $('inspect-button-icon').textContent = state.busy ? '…' : '↗';
  $('processing-overlay').hidden = !state.busy;
  $('preview-stage').setAttribute('aria-busy', String(state.busy));
  $('control-hint').textContent = state.busy ? 'Tunggu hingga proses selesai. Hasil menggunakan gambar dan confidence saat dikirim.' : !state.modelReady ? 'Model belum siap. Periksa status model di atas.' : !state.file ? 'Pilih gambar PCB polos untuk memulai.' : !state.previewReady ? 'Menyiapkan preview gambar…' : 'Siap diproses. Gambar dan hasil tidak disimpan sebagai riwayat.';
}

function clearResults(text = 'Jalankan inspeksi untuk meninjau gambar yang dipilih.') {
  gate.invalidate();
  state.controller?.abort();
  state.controller = null;
  state.busy = false;
  state.resultUrls.forEach((url) => URL.revokeObjectURL(url));
  state.resultUrls = [];
  $('result-image').removeAttribute('src');
  $('tab-result').disabled = true;
  switchView('original');
  for (const id of ['download-png', 'download-csv']) {
    $(id).removeAttribute('href'); $(id).removeAttribute('download');
    $(id).setAttribute('aria-disabled', 'true'); $(id).tabIndex = -1;
  }
  for (const id of ['total-detections', 'inference-time', 'applied-confidence']) $(id).textContent = '—';
  $('detections-body').replaceChildren();
  const row = document.createElement('tr'); const cell = document.createElement('td');
  cell.colSpan = 4; cell.className = 'empty-table'; cell.textContent = 'Belum ada hasil inspeksi.'; row.append(cell); $('detections-body').append(row);
  $('view-label').textContent = state.file ? 'GAMBAR BELUM DIPROSES' : 'MENUNGGU GAMBAR';
  summary(); message(text); syncControls();
}

function clearFile() {
  state.file = null; state.previewReady = false;
  if (state.originalUrl) URL.revokeObjectURL(state.originalUrl);
  state.originalUrl = null;
  $('image-file').value = ''; $('original-image').removeAttribute('src'); $('original-image').hidden = true;
  $('empty-original').hidden = false; $('file-details').hidden = true; $('image-dimensions').textContent = '— × — PX';
  clearResults('Hasil deteksi akan muncul setelah inspeksi dijalankan.');
}

function chooseFile(file) {
  clearFile();
  if (!file) return;
  if (file.size === 0) { message('File kosong. Pilih gambar JPEG atau PNG yang valid.', 'error'); return; }
  if (file.size > MAX_BYTES) { message('Ukuran file melebihi 10 MiB. Pilih gambar yang lebih kecil.', 'error'); return; }
  state.file = file;
  state.originalUrl = URL.createObjectURL(file);
  const selectedUrl = state.originalUrl;
  $('file-name').textContent = file.name;
  $('file-name').title = file.name;
  $('file-meta').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MiB`;
  $('file-details').hidden = false;
  $('original-image').onload = () => {
    if (state.originalUrl !== selectedUrl) return;
    const { naturalWidth: width, naturalHeight: height } = $('original-image');
    if (width * height > MAX_PIXELS) { clearFile(); message('Resolusi gambar melebihi 25 megapiksel. Perkecil gambar dahulu.', 'error'); return; }
    state.previewReady = true;
    $('original-image').hidden = false; $('empty-original').hidden = true;
    $('image-dimensions').textContent = `${width} × ${height} PX`;
    $('file-meta').textContent += ` · ${width} × ${height} px`;
    $('view-label').textContent = 'GAMBAR BELUM DIPROSES';
    message('Gambar siap. Jalankan inspeksi untuk melihat kandidat cacat.'); syncControls();
  };
  $('original-image').onerror = () => { if (state.originalUrl === selectedUrl) { clearFile(); message('Gambar tidak dapat dibaca. Pilih file JPEG atau PNG yang valid.', 'error'); } };
  $('original-image').src = selectedUrl;
  syncControls();
}

async function checkHealth() {
  if (state.healthBusy) return;
  state.healthBusy = true; $('refresh-health').disabled = true;
  try {
    const response = await fetch('/api/health', { cache: 'no-store' });
    const data = await response.json();
    const wasReady = state.modelReady;
    const previousCheckpoint = state.modelCheckpoint;
    state.modelReady = response.ok && data.ready === true;
    state.modelCheckpoint = state.modelReady ? data.model?.checkpoint_sha256 ?? null : null;
    if (previousCheckpoint && state.modelCheckpoint && previousCheckpoint !== state.modelCheckpoint) {
      clearResults('Checkpoint berubah. Jalankan inspeksi kembali untuk memperbarui hasil.');
    }
    $('model-pill-text').textContent = state.modelReady ? 'MODEL SIAP' : 'MODEL BELUM SIAP';
    $('model-pill').className = `status-pill${state.modelReady ? ' ready' : ''}`;
    $('model-notice').className = `model-notice ${state.modelReady ? 'ready' : 'warning'}`;
    $('model-notice-title').textContent = state.modelReady ? 'Checkpoint PCB siap digunakan' : 'Model belum siap untuk inspeksi';
    $('model-notice-description').textContent = state.modelReady ? 'Pilih gambar PCB polos yang menyerupai dataset untuk memulai inspeksi.' : formatApiError(data, 'Checkpoint PCB belum tersedia. Muat checkpoint dan jalankan ulang aplikasi.');
    $('model-name').textContent = data.model?.name || 'Belum tersedia';
    $('model-device').textContent = state.modelReady && data.device ? String(data.device).toUpperCase() : '—';
    $('model-checkpoint').textContent = data.model?.checkpoint_sha256 ? String(data.model.checkpoint_sha256).slice(0, 12) : '—';
    $('model-checkpoint').title = data.model?.checkpoint_sha256 || '';
    if (!state.confidenceTouched && Number.isFinite(data.default_confidence) && data.default_confidence >= 0.05 && data.default_confidence <= 0.95) {
      $('confidence').value = data.default_confidence;
      $('confidence-value').textContent = `${Math.round(Number($('confidence').value) * 100)}%`;
    }
    if (wasReady && !state.modelReady) clearResults('Model belum siap. Periksa status sebelum menjalankan inspeksi kembali.');
  } catch {
    state.modelReady = false;
    state.modelCheckpoint = null;
    $('model-pill-text').textContent = 'KONEKSI TERPUTUS'; $('model-pill').className = 'status-pill';
    $('model-notice').className = 'model-notice warning';
    $('model-notice-title').textContent = 'Status model tidak dapat diperiksa';
    $('model-notice-description').textContent = 'Pastikan aplikasi lokal berjalan, lalu pilih Cek model.';
    $('model-name').textContent = 'Belum terhubung'; $('model-device').textContent = '—'; $('model-checkpoint').textContent = '—';
    clearResults('Koneksi ke aplikasi terputus. Periksa status model dan coba lagi.');
  } finally { state.healthBusy = false; $('refresh-health').disabled = false; syncControls(); }
}

function renderDetections(detections) {
  $('detections-body').replaceChildren();
  if (!detections.length) {
    const row = document.createElement('tr'); const cell = document.createElement('td'); cell.colSpan = 4; cell.className = 'empty-table';
    cell.textContent = 'Tidak ada kandidat cacat yang memenuhi batas confidence.'; row.append(cell); $('detections-body').append(row); return;
  }
  const rows = detections.map((item, index) => {
    const row = document.createElement('tr');
    const number = document.createElement('td'); number.textContent = String(index + 1).padStart(2, '0');
    const label = document.createElement('td'); label.textContent = item.class_name;
    const confidence = document.createElement('td');
    const meter = document.createElement('div'); meter.className = 'confidence-cell';
    const percentage = document.createElement('span'); percentage.textContent = `${(item.confidence * 100).toFixed(1)}%`;
    const track = document.createElement('span'); track.className = 'confidence-track'; track.setAttribute('aria-hidden', 'true');
    const fill = document.createElement('span'); fill.className = 'confidence-fill'; fill.style.display = 'block'; fill.style.width = `${Math.max(0, Math.min(100, item.confidence * 100))}%`;
    track.append(fill); meter.append(percentage, track); confidence.append(meter);
    const coords = document.createElement('td'); coords.textContent = `[${item.bbox_xyxy.map((value) => Number(value).toFixed(1)).join(', ')}]`;
    row.append(number, label, confidence, coords); return row;
  });
  $('detections-body').replaceChildren(...rows);
}

async function inspect(event) {
  event.preventDefault();
  if (!state.file || !state.previewReady || !state.modelReady || state.busy) return;
  clearResults('Inspeksi sedang berjalan.');
  const token = gate.next();
  const file = state.file;
  const confidence = Number($('confidence').value);
  state.controller = new AbortController();
  const controller = state.controller;
  state.busy = true; $('view-label').textContent = 'INSPEKSI BERJALAN'; syncControls();
  const form = new FormData(); form.append('image', file); form.append('confidence', String(confidence));
  try {
    const response = await fetch('/api/inspect', { method: 'POST', body: form, signal: controller.signal });
    let data;
    try { data = await response.json(); } catch { throw new Error('Respons aplikasi tidak dapat dibaca. Silakan coba lagi.'); }
    if (!gate.isCurrent(token)) return;
    if (!response.ok) {
      if (response.status === 503) { state.modelReady = false; void checkHealth(); }
      throw new Error(formatApiError(data));
    }
    if (!Array.isArray(data.detections) || typeof data.annotated_image !== 'string' || !data.annotated_image.startsWith('data:image/png;base64,')) throw new Error('Format hasil inspeksi belum sesuai. Periksa aplikasi dan coba lagi.');
    if (data.model?.checkpoint_sha256 !== state.modelCheckpoint) {
      void checkHealth();
      throw new Error('Checkpoint berubah. Cek model dan jalankan inspeksi kembali.');
    }
    const imageBlob = await (await fetch(data.annotated_image, { signal: controller.signal })).blob();
    if (!gate.isCurrent(token)) return;
    const imageUrl = URL.createObjectURL(imageBlob);
    const csvUrl = URL.createObjectURL(new Blob(['\ufeff', csvForDetections(data.detections)], { type: 'text/csv;charset=utf-8' }));
    state.resultUrls = [imageUrl, csvUrl];
    $('result-image').src = imageUrl;
    $('tab-result').disabled = false; switchView('result');
    const filename = file.name.replace(/\.[^.]+$/, '').replace(/[^\w.-]/g, '_').slice(0, 80) || 'pcb';
    for (const [id, url, extension] of [['download-png', imageUrl, 'png'], ['download-csv', csvUrl, 'csv']]) {
      $(id).href = url; $(id).download = `${filename}-inspection.${extension}`; $(id).setAttribute('aria-disabled', 'false'); $(id).tabIndex = 0;
    }
    const counts = data.counts || Object.fromEntries(CLASS_NAMES.map((name) => [name, data.detections.filter((item) => item.class_name === name).length]));
    summary(counts); renderDetections(data.detections);
    $('total-detections').textContent = String(data.total_detections ?? data.detections.length);
    $('inference-time').textContent = Number.isFinite(data.inference_ms) ? data.inference_ms.toFixed(1) : '—';
    $('applied-confidence').textContent = `${Math.round(Number(data.confidence_threshold ?? confidence) * 100)}%`;
    if (data.image?.width && data.image?.height) $('image-dimensions').textContent = `${data.image.width} × ${data.image.height} PX`;
    $('view-label').textContent = 'INSPEKSI SELESAI';
    message(data.detections.length ? `${data.detections.length} kandidat cacat terdeteksi. Periksa kotak dan detail temuan di bawah.` : 'Tidak ada cacat terdeteksi pada pengaturan ini.', 'success');
  } catch (error) {
    if (!gate.isCurrent(token) || error.name === 'AbortError') return;
    $('view-label').textContent = 'INSPEKSI GAGAL';
    message(error instanceof TypeError ? 'Tidak dapat terhubung ke aplikasi. Periksa koneksi lokal dan coba lagi.' : error.message || 'Inspeksi gagal. Silakan coba lagi.', 'error');
  } finally {
    if (gate.isCurrent(token)) { state.busy = false; state.controller = null; syncControls(); }
  }
}

$('image-file').addEventListener('change', (event) => chooseFile(event.target.files?.[0]));
$('remove-file').addEventListener('click', clearFile);
$('inspect-form').addEventListener('submit', inspect);
$('refresh-health').addEventListener('click', checkHealth);
$('confidence').addEventListener('input', () => {
  state.confidenceTouched = true;
  $('confidence-value').textContent = `${Math.round(Number($('confidence').value) * 100)}%`;
  clearResults(state.file ? 'Confidence diubah. Jalankan inspeksi kembali untuk memperbarui hasil.' : 'Pilih gambar PCB polos untuk memulai inspeksi.');
});
$('tab-original').addEventListener('click', () => switchView('original'));
$('tab-result').addEventListener('click', () => switchView('result'));
for (const tab of [$('tab-original'), $('tab-result')]) tab.addEventListener('keydown', (event) => {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key) || $('tab-result').disabled) return;
  event.preventDefault();
  const target = event.key === 'Home' || (event.key !== 'End' && tab === $('tab-result')) ? 'original' : 'result';
  switchView(target); $(`tab-${target}`).focus();
});
for (const id of ['download-png', 'download-csv']) $(id).addEventListener('click', (event) => { if ($(id).getAttribute('aria-disabled') === 'true') event.preventDefault(); });
const dropZone = $('drop-zone');
for (const name of ['dragenter', 'dragover']) dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.add('dragging'); });
for (const name of ['dragleave', 'drop']) dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.remove('dragging'); });
dropZone.addEventListener('drop', (event) => {
  const files = event.dataTransfer?.files;
  if (!files?.length) return;
  if (files.length > 1) { message('Pilih satu gambar per inspeksi.', 'error'); return; }
  chooseFile(files[0]);
});
document.addEventListener('dragover', (event) => event.preventDefault());
document.addEventListener('drop', (event) => event.preventDefault());
window.addEventListener('pagehide', () => {
  state.controller?.abort();
  if (state.originalUrl) URL.revokeObjectURL(state.originalUrl);
  state.resultUrls.forEach((url) => URL.revokeObjectURL(url));
});
summary(); syncControls(); void checkHealth();
