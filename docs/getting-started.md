# Panduan praktik: dataset sampai website

Panduan ini sengaja memakai **satu perintah Python per tahap**. Jalankan sendiri satu tahap, baca kode dan outputnya, lalu lanjut ketika hasilnya sudah masuk akal. Contoh menggunakan nama eksperimen `belajar-01`; semua file hasil eksperimen berada di `experiments/belajar-01/` pada drive proyek. Baseline yang sudah ada di `data/prepared/`, `runs/`, `reports/`, dan `models/` tidak ditimpa.

## 0. Peta alur

| Tahap | Baca kode | Hasil utama |
|---|---|---|
| Setup awal | `setup.ps1`, `pcb_inspector/workflow.py` | `.venv/` dan `.local/` |
| 1. Prepare data | `scripts/prepare_data.py`, `pcb_inspector/data.py` | `prepared/dataset.yaml`, manifest, preview label |
| 2. Train | `scripts/train.py`, `pcb_inspector/training.py` | `training/weights/best.pt`, kurva dan laporan |
| 3. Evaluate | `scripts/evaluate.py` | validation, test, contoh kesalahan |
| 4. Predict | `scripts/predict.py` | PNG berbingkai dan JSON satu gambar |
| 5. Prepare model | `scripts/prepare_model.py` | `.pt` dan metadata untuk website |
| 6. Serve | `scripts/serve.py`, `pcb_inspector/api.py` | FastAPI dan halaman web lokal |

`pcb_inspector/workflow.py` hanya mengatur path eksperimen dan cache/temp proyek. Enam file pada `scripts/` adalah perintah yang lu jalankan. Fungsi pendukung di `pcb_inspector/` dipanggil oleh script, bukan perintah tambahan. Tiap perintah dijalankan dari **root proyek**. Tidak diperlukan Anaconda, aktivasi venv, atau launcher `.ps1` per tahap.

## Setup sekali di D:

Pada mesin ini, `.venv`, dataset raw, dan checkpoint pretrained telah tersedia di proyek. **Tidak perlu mengulang setup atau download** untuk mencoba alur manual. Untuk clone baru, siapkan Python 3.12 dan `uv.exe` di D:, lalu jalankan:

```powershell
Set-Location 'D:\!PROJECT\!persiapan_maganghub\projects\comvis_yolo'
.\setup.ps1 -BasePython 'D:\Tools\Python312\python.exe' -UvPath 'D:\Tools\uv.exe'
```

Ganti kedua path contoh dengan lokasi executable yang benar. Script setup membuat `.venv` di proyek, memasang PyTorch CUDA dan dependensi sesuai `requirements.txt`, serta memeriksa CUDA. Pengaturan baseline memakai driver 537.70 dan RTX 3050 Laptop 4 GB; mesin lain perlu versi PyTorch/CUDA yang sesuai. Script setup tidak memasang driver GPU. Jika `.venv` sudah ada, tahap ini boleh dilewati.

Semua entry point Python mengatur `TEMP`, cache Torch/uv/pip, konfigurasi Ultralytics, dan cache lain di `.local/` pada proyek **sebelum** mengimpor library ML. Gunakan `.venv\Scripts\python.exe -B` dari D:. `-B` mencegah penulisan bytecode Python. Windows masih dapat membaca file sistem yang ada di C:, sementara file kerja proyek diarahkan ke D:.

### Dataset dan bobot awal

Download [PCB Defect dataset dari Kaggle](https://www.kaggle.com/datasets/norbertelter/pcb-defect-dataset/data). Paket ini **sudah memiliki file label YOLO**; lu tidak perlu menganotasi seluruh gambar di Roboflow. Atur folder unduhan browser ke D:, lalu simpan ZIP di `data/raw/pcb-defect-dataset.zip`. Untuk mengekstraknya pertama kali:

```powershell
Expand-Archive -LiteralPath '.\data\raw\pcb-defect-dataset.zip' -DestinationPath '.\data\raw'
```

Pastikan hasil ekstraksi memiliki `data/raw/pcb-defect-dataset/data.yaml` dan folder `train`, `val`, `test`, masing-masing dengan `images/` dan `labels/`. Simpan pretrained [Ultralytics `yolo11n.pt`](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) di `models/base/yolo11n.pt`. Pada mesin ini keduanya sudah tersedia. Dataset dan checkpoint tetap lokal karena ketentuan redistribusi dataset belum jelas.

## 1. Prepare data

Baca `scripts/prepare_data.py` lalu fungsi `prepare_dataset` di `pcb_inspector/data.py`. Script memeriksa gambar, label, class ID, koordinat bounding box, dan pasangan nama file. Sebagian label dari sumber memakai akhiran `_256` sedangkan gambar pasangannya `_600`; script memasangkannya pada salinan turunan, tanpa membuat anotasi baru. Gambar yang berasal dari family nama sama ditempatkan dalam split yang sama untuk mengurangi kebocoran antara train, validation, dan test.

```powershell
& .\.venv\Scripts\python.exe -B -m scripts.prepare_data --experiment belajar-01
```

Periksa `experiments/belajar-01/prepared/dataset.yaml`, `manifest.json`, `preparation/data-audit.json`, dan `preparation/visuals/annotation-preview.png`. Preview memperlihatkan beberapa bounding box dari label sumber; lihat apakah kotaknya menandai cacat yang dimaksud. Dataset awal berisi 10.668 gambar dan 21.664 box; run dengan seed yang sama diharapkan menghasilkan 5.856 train, 2.112 validation, dan 2.700 test. Jika output eksperimen sudah ada, preparation akan menolak menimpanya: pilih nama eksperimen baru daripada menghapus baseline.

## 2. Train model

Baca `training_options()` dan `train_model()` di `scripts/train.py`. Intinya sama dengan contoh standar Ultralytics: `model = YOLO("yolo11n.pt")`, lalu `model.train(...)` dengan `dataset.yaml` dari tahap 1. Checkpoint terbaik dipilih dari validation selama training. Split test tidak dipakai untuk memperbarui bobot.

```powershell
& .\.venv\Scripts\python.exe -B -m scripts.train --experiment belajar-01 --batch 4 --epochs 40 --time-hours 1
```

Run ini memakai batch 4, batas maksimum 40 epoch, early stopping patience 8, dan anggaran satu jam yang diperiksa pada akhir epoch. Anggaran waktu bisa terlewati oleh satu epoch dan final validation. Jika GPU kehabisan memori, jangan menimpa folder training yang sudah dibuat; periksa error dan gunakan eksperimen baru dengan batch 2 atau 1. Tidak ada langkah smoke training dalam alur ini.

Periksa `experiments/belajar-01/training/results.csv` untuk loss/metrik per epoch, `results.png` untuk kurva, `confusion_matrix.png` untuk kekeliruan kelas, serta `weights/best.pt` dan `weights/last.pt`. Laporan `experiments/belajar-01/training-report.json` mencatat konfigurasi, hardware, durasi, dan hash checkpoint terbaik. `best.pt` dan `last.pt` bisa berbeda; tahap selanjutnya memakai `best.pt` yang tercatat.

## 3. Evaluate model

Baca `scripts/evaluate.py`. Script mengambil checkpoint terbaik dari laporan training, memeriksa hash dan kelasnya, lalu memilih batas confidence pada **validation**. Setelah pilihan itu tetap, script menghitung metrik pada **test**. Jangan mengubah training atau threshold berdasarkan test yang sama berulang kali lalu menganggap skor test itu evaluasi baru yang independen.

```powershell
& .\.venv\Scripts\python.exe -B -m scripts.evaluate --experiment belajar-01 --batch 4
```

Periksa `experiments/belajar-01/evaluation/evaluation.json` untuk confidence terpilih, precision, recall, F1, mAP50, mAP50–95, dan rincian per kelas. Folder `evaluation/visuals/` berisi contoh prediksi yang berhasil dan yang banyak salah. Nilai precision/recall/F1 pada satu threshold berbeda dari mAP yang dihitung dari kurva skor. Gambar evaluasi berasal dari dataset, jadi visual tersebut tetap lokal.

## 4. Prediksi satu gambar melalui Python

Tahap ini memperlihatkan `YOLO(best.pt).predict(...)` secara langsung sebelum beralih ke website. Contoh gambar berikut ada pada split validation dari dataset yang saat ini digunakan. Lu juga dapat menggantinya dengan file JPG/PNG PCB polos lain di D:.

```powershell
& .\.venv\Scripts\python.exe -B -m scripts.predict --experiment belajar-01 --image '.\experiments\belajar-01\prepared\val\images\l_light_07_missing_hole_01_1_600.jpg'
```

Periksa `experiments/belajar-01/predictions/<nama-gambar>-prediction.png` untuk gambar dengan kotak, serta file `.json` dengan kelas, confidence, dan koordinat `[x1, y1, x2, y2]`. Prediksi memakai confidence yang dipilih dari validation pada tahap 3. Hasil satu gambar bukan ukuran akurasi keseluruhan model.

## 5. Siapkan model untuk website

Ini adalah tahap deployment model pada proyek ini: salin checkpoint `.pt` terpilih dan tulis metadata yang berisi hash, urutan kelas, dan confidence. **Tidak ada konversi ONNX/TensorRT.** Script memastikan evaluasi, checkpoint, dan manifest berasal dari eksperimen yang sama.

```powershell
& .\.venv\Scripts\python.exe -B -m scripts.prepare_model --experiment belajar-01
```

Periksa `experiments/belajar-01/deployment/pcb-best.pt` dan `model-metadata.json`. Folder `models/` baseline lama tetap utuh. Tahap ini tidak menjalankan training atau inference.

## 6. Buka website FastAPI

```powershell
& .\.venv\Scripts\python.exe -B -m scripts.serve --experiment belajar-01 --port 8000
```

Biarkan terminal terbuka selama memakai website, lalu buka [http://127.0.0.1:8000](http://127.0.0.1:8000). Hentikan server dengan **Ctrl+C** di terminal itu. Jika port 8000 sudah digunakan, pilih `--port 8001` dan buka alamat port 8001. Untuk membuka website memakai model baseline lama yang sudah tersedia lokal, jalankan `scripts.serve` tanpa `--experiment`.

Halaman harus dibuka melalui alamat FastAPI, bukan dengan membuka `web/index.html` langsung. `GET /api/health` menunjukkan model aktif dan device; `POST /api/inspect` menerima gambar serta confidence. Saat startup, `pcb_inspector/api.py` memuat `PCBDetector` dari `pcb_inspector/inference.py` sekali. Jika file model diganti ketika server masih aktif, hentikan dan jalankan server lagi. API interaktif tersedia di `/docs`.

Di website, pilih satu gambar PCB polos, jalankan inspeksi, tinjau kotak dan tabel deteksi, lalu unduh PNG/CSV bila perlu. Browser menentukan folder unduhan secara terpisah dari cache backend; arahkan folder unduhannya ke D:. Model belajar dari gambar dan label saat **training**, bukan ketika tombol inspeksi ditekan.

## Catatan ketika membaca hasil

- Dataset ini menggunakan PCB polos dan anotasi cacat visual. Model tidak memeriksa koneksi listrik, solder, atau komponen rakitan.
- Semua gambar train/validation/test pada dataset ini berlabel cacat; kemampuan menolak PCB normal belum diukur.
- Baseline lama menghasilkan validation mAP50 87,11% dan test mAP50 65,80%. Run `belajar-01` milik lu dapat memberi angka berbeda; selalu lihat laporan eksperimennya sendiri.
- Jika sebuah tahap gagal, baca pesan error dan file yang sudah dibuat di folder eksperimen sebelum mengulang. Jangan menghapus folder baseline untuk memperbaiki eksperimen baru.
