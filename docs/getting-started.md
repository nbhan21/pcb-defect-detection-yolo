# Panduan Praktik: dari Dataset sampai Website

Panduan ini menunjukkan urutan menjalankan proyek, file Python yang bekerja di balik setiap perintah, serta hasil yang perlu diperiksa. Jalankan perintah dari PowerShell pada root proyek di D:.

## 1. Memilih cara mulai

**Pada mesin ini, environment, dataset prepared, dan model terlatih sudah tersedia.** Untuk memakai website, langsung ke bagian 2. Tidak perlu setup, prepare, atau training ulang setiap kali membuka website.

Bagian 4 sampai 8 menjelaskan reproduksi dari awal pada folder proyek yang belum berisi hasil, atau untuk mempelajari tahapnya satu per satu. Jangan menjalankan semuanya sekaligus pada hasil yang sudah ada: prepare menolak output yang sudah terisi, sedangkan training/evaluasi menulis laporan hasil terbaru.

Urutan belajar utama:

```text
setup.ps1
    ↓
dataset raw + pretrained yolo11n.pt
    ↓
prepare-data.ps1 → scripts/prepare_data.py → pcb_inspector/data.py
    ↓
train.ps1        → scripts/train.py
    ↓
evaluate.ps1     → scripts/evaluate.py
    ↓
models/pcb-best.pt + models/model-metadata.json
    ↓
run.ps1         → Uvicorn → pcb_inspector/api.py → inference.py
    ↓
http://127.0.0.1:8000 → web/index.html + app.js + styles.css
```

## 2. Membuka website dengan model yang sudah ada

```powershell
Set-Location 'D:\!PROJECT\!persiapan_maganghub\projects\comvis_yolo'
.\run.ps1
```

Biarkan terminal tersebut terbuka. Setelah muncul `Uvicorn running on http://127.0.0.1:8000`, buka alamat itu di browser. Dokumentasi endpoint interaktif tersedia di `http://127.0.0.1:8000/docs`.

Saat panduan ini dibuat, backend pada port 8000 masih dijalankan melalui proses background oleh agen. Jika website itu masih dapat dibuka, gunakan langsung. Untuk melihat proses server dari terminal milik sendiri tanpa konflik dengan port tersebut, gunakan:

```powershell
.\run.ps1 -Port 8001
```

Lalu buka `http://127.0.0.1:8001`. Itu menjalankan server tambahan; hentikan dengan **Ctrl+C di terminal yang menjalankannya** setelah selesai. Jika muncul error port 8000 sudah dipakai, jangan menjalankan training sebagai solusi: port tersebut memang sedang digunakan server lain.

Launcher memakai `.venv\Scripts\python.exe` secara langsung. Tidak perlu mengetik `activate`, memasang Anaconda, atau menjalankan Node untuk membuka website.

Pilih gambar PCB polos, periksa preview, lalu tekan **JALANKAN INSPEKSI**. Slider memakai default 60% dari validation; perubahan slider hanya mengubah inference. Hasil bisa diunduh sebagai PNG/CSV. Folder ekspor mengikuti pengaturan unduhan browser: arahkan ke D: secara terpisah dari pengaturan server.

Jangan membuka `web/index.html` dengan double-click. Halaman harus diakses melalui alamat HTTP FastAPI agar `/api/health` dan `/api/inspect` tersedia.

## 3. Peta file yang perlu dipelajari

| Tahap | Perintah utama | File Python yang bekerja | Hasil |
|---|---|---|---|
| Environment | `setup.ps1` | Installer uv dan Python di D: | `.venv/`, dependensi |
| Prepare | `prepare-data.ps1` | `scripts/prepare_data.py`, `pcb_inspector/data.py` | `data/prepared/`, manifest, YAML, audit |
| Training | `train.ps1` | `scripts/train.py`, helper `pcb_inspector/training.py` | Checkpoint, kurva, CSV, konfigurasi |
| Evaluasi/persiapan model web | `evaluate.ps1` | `scripts/evaluate.py`, helper `training.py` | Metrik, model dan metadata web |
| Server | `run.ps1` | Modul `pcb_inspector.api:app` di `api.py` | HTTP server FastAPI |
| Prediksi dari web | Tombol inspeksi | `pcb_inspector/inference.py` | Bounding boxes dan PNG hasil |

File `.ps1` adalah launcher PowerShell: memilih Python proyek, mengatur lokasi cache/temp ke D:, lalu menjalankan Python. `scripts/project-env.ps1` dimuat oleh launcher, bukan tahap training tersendiri. `constants.py`, `data.py`, `training.py`, dan `inference.py` adalah modul pendukung; tidak perlu dijalankan satu per satu.

## 4. Menyiapkan environment dari awal

Hanya diperlukan jika `.venv` belum tersedia atau sedang menyiapkan clone baru:

```powershell
Set-Location 'D:\!PROJECT\!persiapan_maganghub\projects\comvis_yolo'
.\setup.ps1
```

Default memakai base Python dan uv yang sudah tersedia di D: pada workspace ini. Untuk mesin lain, gunakan executable yang memang tersedia:

```powershell
.\setup.ps1 -BasePython 'D:\Tools\Python312\python.exe' -UvPath 'D:\Tools\uv.exe'
```

Path contoh mesin lain harus disesuaikan. Setup membuat venv khusus proyek, memasang PyTorch CUDA dan dependensi, lalu menampilkan apakah CUDA tersedia. Environment/cache/temp proyek diarahkan ke D: oleh `scripts/project-env.ps1`; Windows tetap menggunakan file sistem/executable yang sudah ada di C:.

## 5. Dataset raw dan prepare data

Pada mesin ini ZIP dan hasil ekstraksinya sudah berada di `data/raw/`. Untuk clone baru, simpan ZIP dataset Kaggle di D: dengan susunan berikut:

```text
data/raw/pcb-defect-dataset.zip
data/raw/pcb-defect-dataset/
    data.yaml
    train/images/ dan train/labels/
    val/images/ dan val/labels/
    test/images/ dan test/labels/
```

ZIP yang digunakan proyek ini berisi direktori `pcb-defect-dataset/`. Untuk ekstraksi pertama, setelah ZIP ditempatkan pada lokasi di atas:

```powershell
Expand-Archive -LiteralPath '.\data\raw\pcb-defect-dataset.zip' -DestinationPath '.\data\raw'
.\prepare-data.ps1
```

`scripts/prepare_data.py` memanggil fungsi persiapan pada `pcb_inspector/data.py`. Proses memeriksa pasangan gambar/label dan koordinat, memperbaiki pasangan nama `_256`/`_600` pada data turunan, lalu membuat split berdasarkan family sumber agar rotasi/varian family yang sama tidak tersebar ke train dan test.

Periksa hasil:

- `data/prepared/dataset.yaml`: lokasi gambar dan urutan enam kelas untuk YOLO.
- `data/prepared/manifest.json`: catatan sumber, family, hash dan split.
- `reports/data-audit.json`: jumlah gambar/anotasi dan hasil audit.
- `reports/visuals/annotation-preview.png`: cek visual posisi kotak anotasi.

Pada hasil yang sudah tersedia: 5.856 train, 2.112 validation, dan 2.700 test. Jika prepare menolak karena `manifest.json` sudah ada, jangan hapus hasil lama hanya untuk membuka website; lanjutkan memakai data prepared tersebut.

## 6. Pretrained model dan training

Untuk setup baru, pretrained weights harus ada di `models/base/yolo11n.pt`. Unduh sekali jika belum tersedia:

```powershell
New-Item -ItemType Directory -Path '.\models\base' -Force
Invoke-WebRequest -Uri 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt' -OutFile '.\models\base\yolo11n.pt'
```

Lakukan smoke run pada setup baru untuk memeriksa GPU/pipeline dengan subset kecil:

```powershell
.\train.ps1 --smoke --batch 4
```

Kemudian training utama:

```powershell
.\train.ps1 --batch 4 --epochs 40 --time-hours 1 --name baseline
```

`scripts/train.py` membaca manifest dan YAML, memuat pretrained YOLO11n, lalu memperbarui bobot pada dataset PCB. Validation dijalankan setiap epoch untuk memilih checkpoint terbaik. Test tidak dipakai untuk update bobot atau pemilihan checkpoint.

Batasnya adalah maksimum 40 epoch, early stopping patience 8, atau budget satu jam yang diperiksa pada batas epoch. Waktu dapat melewati budget hingga satu epoch dan final validation. Run sebelumnya berhenti pada 12 epoch, checkpoint terbaik pada epoch 4; ini hasil aktual, bukan jumlah yang dijamin untuk setiap run.

Untuk eksperimen baru, gunakan nama berbeda, misalnya `--name belajar-01`. Itu membedakan direktori run. Namun `reports/training-config.json` tetap ditulis sebagai laporan training terbaru; simpan salinan laporan/model lama di folder D: terpisah jika ingin mempertahankan perbandingan eksperimen.

## 7. Melihat hasil training

Untuk run awal bernama baseline, lihat:

| File | Gunanya |
|---|---|
| `runs/baseline/results.csv` | Loss dan metrik validation per epoch |
| `runs/baseline/results.png` | Kurva training/validation |
| `runs/baseline/confusion_matrix.png` | Gambaran prediksi kelas versus anotasi |
| `runs/baseline/weights/best.pt` | Checkpoint terbaik berdasarkan validation |
| `runs/baseline/weights/last.pt` | Checkpoint terakhir, belum tentu terbaik |
| `reports/training-config.json` | Konfigurasi, hardware, durasi dan path/hash checkpoint terbaik |

Nama direktori dapat berbeda pada run berikutnya. Path yang dipakai tahap evaluasi dibaca dari `best_checkpoint` dalam `reports/training-config.json`, bukan ditebak dari nama folder.

## 8. Evaluasi dan menyiapkan model untuk website

```powershell
.\evaluate.ps1 --batch 4
```

`scripts/evaluate.py` membaca checkpoint terbaik yang tercatat, memastikan hash-nya cocok, memilih confidence dari validation, lalu mengevaluasi test setelah pilihan tersebut dibekukan. Jalankan test untuk evaluasi akhir eksperimen, bukan untuk menyetel parameter berulang-ulang.

Hasil yang diperiksa:

- `reports/evaluation.json`: metrik agregat/per kelas, confidence terpilih dan hash checkpoint.
- `reports/visuals/test-errors.png` dan `test-detections.png`: contoh kegagalan dan keberhasilan.
- `models/pcb-best.pt`: salinan checkpoint yang disiapkan untuk aplikasi.
- `models/model-metadata.json`: urutan kelas, hash checkpoint, confidence default dan informasi evaluasi.

**Tidak ada tahap konversi ONNX/TensorRT dalam MVP ini.** Model sudah tersimpan sebagai checkpoint `.pt` dan dipakai langsung oleh Ultralytics. Tahap yang disebut menyiapkan/mengekspor model untuk web di sini adalah menyalin checkpoint terpilih beserta metadata-nya melalui `evaluate.ps1`.

`reports/error-analysis.md` berisi tinjauan manual eksperimen pertama. File Markdown itu tidak otomatis ditulis ulang oleh evaluator; jika membuat eksperimen baru, tinjau ulang sesuai hasil barunya.

## 9. Bagaimana model terhubung ke website

Jalankan `run.ps1` setelah model dan metadata tersedia. Launcher menjalankan:

```text
.venv/Scripts/python.exe -m uvicorn pcb_inspector.api:app
```

Saat startup, lifespan pada `pcb_inspector/api.py` membuat `PCBDetector` dari `pcb_inspector/inference.py`. Detector membaca dua file pada `models/`, memeriksa hash dan nama kelas, lalu memuat model sekali.

Saat pengguna mengunggah gambar:

1. `web/app.js` mengirim file dan confidence ke `POST /api/inspect`.
2. `api.py` memvalidasi input dan meneruskan pekerjaan ke detector.
3. `inference.py` menjalankan model, menyusun kotak/kelas/skor, dan membuat PNG hasil.
4. FastAPI mengembalikan JSON; JavaScript menampilkan hasil dan membuat ekspor PNG/CSV.

`GET /api/health` memberi status kesiapan, nama/hash model dan device aktual. Jika weights/metadata hilang atau tidak cocok, aplikasi menampilkan model belum siap; tidak beralih diam-diam ke model COCO.

Jika weights di `models/` diganti setelah evaluasi baru, **restart proses server** agar model baru dimuat. Refresh halaman browser saja tidak memuat ulang model di backend. Selanjutnya jalankan inspeksi baru; hasil lama dibatalkan ketika perubahan checkpoint terdeteksi.

## 10. Menjalankan Python secara langsung untuk belajar

Launcher lebih praktis karena selalu memuat environment proyek. Jika ingin melihat entry point Python secara eksplisit, ekuivalennya dari root proyek adalah:

```powershell
. .\scripts\project-env.ps1

# Pilih satu tahap yang ingin dijalankan, bukan semuanya sekaligus:
& $ProjectPython -B scripts/prepare_data.py
& $ProjectPython -B scripts/train.py --smoke --batch 4
& $ProjectPython -B scripts/train.py --batch 4 --epochs 40 --time-hours 1 --name baseline
& $ProjectPython -B scripts/evaluate.py --batch 4

# Untuk server, tetap gunakan launcher agar port/device diatur konsisten:
.\run.ps1
```

Jangan menjalankan `python pcb_inspector/api.py` dan berharap server otomatis terbuka: file itu mendefinisikan aplikasi yang kemudian dijalankan Uvicorn. Detail konsep, tensor ke JSON, serta confidence dijelaskan dalam [learning guide](learning-guide.md).

## 11. Posisi tes dan folder pendukung

Tes kode sudah dipindahkan ke `archive/verification/`. Tes tersebut tidak menjadi prasyarat menjalankan alur di atas. Baca README pada folder arsip bila ingin menjalankannya kembali.

`data/prepared/test/` adalah **split dataset untuk evaluasi model**, berbeda dari file tes kode. Folder dataset itu tetap dibutuhkan evaluator dan tidak diarsipkan.

Untuk belajar, fokus pada launcher root, `scripts/`, `pcb_inspector/`, `web/`, serta hasil pada `runs/`, `models/`, dan `reports/`. `.local/` menyimpan cache, temporary files dan tooling lokal; `.venv/` menyimpan dependensi. Keduanya tidak perlu dibuka satu per satu untuk mengikuti alur proyek.
