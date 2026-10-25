# Status Implementasi

Verifikasi akhir 25 Oktober 2026. MVP selesai diimplementasikan dan diuji dengan checkpoint final nyata.

Panduan menjalankan tahap demi tahap tersedia di [getting-started.md](../docs/getting-started.md), sedangkan learning-guide menjelaskan konsep dan alur kode. Tes kode beserta konfigurasi pytest diarsipkan ke `archive/verification/`; 50 tes Python dan 7 tes frontend tetap lulus setelah pemindahan. Arsip tidak perlu dijalankan untuk memakai pipeline atau website.

- FastAPI + HTML/CSS/JavaScript berjalan lokal, satu worker. Upload JPEG/PNG, slider confidence, gambar hasil, kotak/kelas/confidence, ringkasan, serta ekspor PNG/CSV tersedia.
- Data: 5.856 train / 2.112 validation / 2.700 test; enam kelas. Family/hash tidak melewati split. Semua 10.668 gambar prepared cocok dengan hash manifest. Label `_256` dipasangkan dengan `_600` pada data turunan; raw dipertahankan.
- Environment: venv khusus, 55 paket kompatibel dan requirements.txt dibekukan. Torch 2.5.1+cu121, CUDA 12.1, RTX 3050 Laptop 4 GB, driver 537.70. Runtime/cache/temp/data/weights proyek berada di D; Windows tetap membaca executable/file sistem yang sudah ada di C.
- Smoke batch 4 selesai tanpa OOM. Baseline berhenti melalui early stopping pada 12 epoch; checkpoint terbaik epoch 4. Durasi wrapper 2.913,766 detik (48,6 menit). Peak PyTorch allocated 648.290.816 byte, reserved 817.889.280 byte; ini bukan total alokasi driver/display.
- Confidence default 0,60 dipilih dari validation (F1 85,48%). Test dipakai setelah pilihan checkpoint/confidence dibekukan: mAP50 65,80%, mAP50–95 26,52%, precision 68,64%, recall 61,57%, F1 64,91%.
- Checkpoint final `models/pcb-best.pt` dan `models/model-metadata.json` tersedia lokal; SHA256 checkpoint cocok dengan training/evaluation/service. Nama kelas tervalidasi. Tidak ada fallback ke COCO atau deteksi buatan.
- Suite kode: 50 tes Python dan 7 tes frontend lulus. Satu upstream deprecation warning Starlette/AnyIO tetap ada.
- Browser memverifikasi not-ready, upload, hasil nol deteksi (checkpoint smoke nyata), hasil positif (checkpoint sementara dan final), MIME menyesatkan pada JPEG valid, file bukan gambar, pergantian confidence dan checkpoint yang membatalkan hasil/download lama. Layout 390 piksel tidak meluap horizontal; tabel dapat digeser dalam kontainernya.
- Model final digunakan dari browser pada CUDA. Satu input validation 600×600: cold inference 5.247,8 ms termasuk warmup; satu pengulangan 82,4 ms. Ini bukan benchmark distribusi latency.
- Ekspor: mekanisme unduh CSV browser diperiksa, PNG payload 600×600 valid, tautan PNG menunjuk blob yang sama dengan gambar hasil. CSV prediksi final dari helper browser yang sama diperiksa terhadap response nyata dan ditulis langsung ke D. Screenshot/PNG/CSV uji tersedia lokal di reports/browser dan tidak masuk Git.
- Repo public: https://github.com/nbhan21/pcb-defect-detection-yolo. Source, dokumentasi Markdown, konfigurasi, dan laporan agregat dipublikasikan; dataset, checkpoint, cache, serta gambar turunan tetap lokal.

## Batas yang masih berlaku

Generalisasi masih terbatas: validation mAP50 87,11% lebih tinggi daripada test 65,80%. `short` dan `spurious_copper` masih lemah. Seluruh gambar dataset memiliki anotasi cacat, sehingga false-alarm rate pada PCB normal belum diukur. Hasil tidak menjadi keputusan lolos/gagal QC.

Lokasi ekspor browser mengikuti pengaturan browser, bukan TEMP server. Satu CSV uji 59 byte sempat disimpan IAB ke Downloads bawaan pada C dan segera dipindah ke D. Tidak ada dataset/checkpoint/cache training di C. Atur folder unduhan browser ke D untuk ekspor selanjutnya; launcher aplikasi tidak dapat mengubah preferensi browser tersebut.

Lihat README, reports/evaluation.json, reports/error-analysis.md, dan reports/runtime-checks.json untuk hasil serta batas interpretasinya. Jalankan kembali aplikasi dari root proyek dengan `./run.ps1`.
