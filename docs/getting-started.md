# Panduan praktik: dataset sampai website

Jalankan **satu perintah Python per tahap** dari root proyek. Baca skripnya, jalankan sendiri, lalu periksa hasilnya sebelum lanjut. Contoh memakai eksperimen `belajar-01`. Semua path proyek pada mesin ini berada di drive D.

| Tahap | Skrip | Hasil utama |
|---|---|---|
| 1. Koreksi dataset | `scripts/prepare_data.py` | `data/kaggle-corrected/dataset.yaml` dan `audit.json` |
| 2. Training | `scripts/train.py` | `experiments/belajar-01/training/weights/best.pt` |
| 3. Evaluasi | `scripts/evaluate.py` | Metrik dan plot validation/test |
| 4. Prediksi | `scripts/predict.py` | PNG dan JSON satu gambar |
| 5. Siapkan model | `scripts/prepare_model.py` | `.pt` dan metadata untuk FastAPI |
| 6. Website | `scripts/serve.py` | FastAPI lokal di `127.0.0.1:8000` |

## Setup awal di D

Pada mesin proyek ini `.venv`, dataset raw hasil ekstraksi, dan pretrained YOLO11n sudah tersedia. Setup tidak perlu diulang. Jika menyiapkan clone baru, tempatkan Python 3.12 dan `uv.exe` di D, lalu jalankan `setup.ps1` satu kali:

```powershell
Set-Location 'D:\!PROJECT\!persiapan_maganghub\projects\comvis_yolo'
.\setup.ps1 -BasePython 'D:\Tools\Python312\python.exe' -UvPath 'D:\Tools\uv.exe'
```

Ganti kedua path executable sesuai lokasi sebenarnya. Tidak diperlukan Anaconda atau aktivasi venv. Gunakan `.venv\Scripts\python.exe -B` untuk setiap tahap; `-B` mencegah bytecode Python. `pcb_inspector/workflow.py` mengarahkan file sementara dan cache proyek ke `.local/` di D sebelum library ML dimuat.

Siapkan [dataset PCB dari Kaggle](https://www.kaggle.com/datasets/norbertelter/pcb-defect-dataset/data) yang **sudah diekstrak** pada `data/raw/pcb-defect-dataset/`. Folder itu harus berisi `train`, `val`, dan `test`, masing-masing dengan `images/` dan `labels/`. ZIP tidak dipakai oleh skrip; jika ZIP masih ada, ia hanya arsip lokal. Simpan pretrained `yolo11n.pt` di `models/base/yolo11n.pt`. Paket Kaggle sudah memiliki anotasi YOLO; lu tidak perlu membuat label dari nol.

## 1. Siapkan salinan dataset

Baca fungsi `prepare_dataset()` di `scripts/prepare_data.py`. Sebagian gambar bernama `..._600.jpg` memiliki label `..._256.txt`. Skrip mengaudit semua pasangan dan menyalin label tersebut dengan nama `..._600.txt` pada dataset hasil. **Isi gambar dan angka anotasi tidak diubah; pembagian train/val/test Kaggle juga tidak diubah.** Jika ada gambar tanpa label, label tanpa gambar, atau dua label yang mungkin cocok, skrip berhenti dengan error.

```powershell
& .\.venv\Scripts\python.exe -B -m scripts.prepare_data
```

Periksa `data/kaggle-corrected/audit.json` dan `dataset.yaml`. Pada dataset yang kita unduh, jumlah raw adalah **8.534 train, 1.066 val, 1.068 test**; nama label yang perlu disamakan masing-masing **2.164, 264, 239**. Salinan dibuat sekali dan dipakai oleh semua eksperimen. Raw, ZIP, serta `data/prepared/` dari baseline sebelumnya tidak ditimpa. Skrip menolak folder output yang sudah ada. Penyalinan memakai folder sementara di D; jika proses terhenti paksa, folder sementara `.kaggle-corrected-*` mungkin tersisa di `data/`, tetapi folder hasil final belum dibuat dan tahap ini bisa dijalankan lagi.

## 2. Training

Baca `scripts/train.py`. Panggilan intinya adalah `model = YOLO('.../yolo11n.pt')`, lalu `model.train(data='.../data/kaggle-corrected/dataset.yaml', ...)`. Ultralytics memakai train untuk memperbarui bobot dan val untuk memilih checkpoint terbaik. Test tidak dipakai saat training.

```powershell
& .\.venv\Scripts\python.exe -B -m scripts.train --experiment belajar-01 --epochs 40 --batch 4
```

Periksa `experiments/belajar-01/training/results.csv`, `results.png`, plot lain yang dihasilkan Ultralytics, dan `weights/best.pt`. Batch 4 adalah titik awal untuk GPU 4 GB pada mesin ini; bila kehabisan VRAM, gunakan eksperimen baru dengan `--batch 2` atau `--batch 1`. Training default memakai `--device 0`; gunakan `--device cpu` hanya jika memang ingin training di CPU. `patience=8` pada `model.train(...)` mengaktifkan early stopping, jadi training bisa berhenti sebelum mencapai `--epochs` jika hasil validation tidak membaik.

## 3. Evaluasi

`scripts/evaluate.py` memuat `best.pt` lalu memanggil `model.val()` untuk `split='val'` dan `split='test'`. Precision, recall, mAP50, dan mAP50–95 berasal dari Ultralytics, bukan rumus pencocokan box buatan proyek.

```powershell
& .\.venv\Scripts\python.exe -B -m scripts.evaluate --experiment belajar-01 --batch 4
```

Periksa `experiments/belajar-01/evaluation/evaluation.json`, serta plot di `evaluation/val/` dan `evaluation/test/`. Nilai precision/recall ringkasan dari validation tidak harus sama dengan hasil prediksi menggunakan confidence 0,25; Ultralytics menghitung metrik evaluasi dari kurva prediksi. Jangan berulang kali memilih model atau parameter berdasarkan skor test yang sama lalu menyebutnya pengujian independen.

## 4. Prediksi satu gambar

`scripts/predict.py` memperlihatkan `YOLO(best.pt).predict(...)`. Langkah ini **bisa dijalankan segera setelah training**, walaupun tahap evaluasi belum dijalankan.

```powershell
& .\.venv\Scripts\python.exe -B -m scripts.predict --experiment belajar-01 --image '.\data\kaggle-corrected\val\images\light_01_mouse_bite_13_1_600.jpg'
```

Ganti `--image` dengan JPG/PNG PCB polos lain bila perlu. Tambahkan `--conf 0.4` untuk mengubah ambang prediksi. Hasilnya ada di `experiments/belajar-01/predictions/` sebagai PNG berbingkai dan JSON kelas, confidence, serta koordinat `[x1, y1, x2, y2]`. Satu gambar bukan ukuran akurasi keseluruhan.

## 5. Siapkan model untuk website

```powershell
& .\.venv\Scripts\python.exe -B -m scripts.prepare_model --experiment belajar-01
```

Periksa `experiments/belajar-01/deployment/pcb-best.pt` dan `model-metadata.json`. Tahap ini menyalin `.pt` dan menulis nama enam kelas serta confidence default 0,25. Ini adalah penyiapan model untuk FastAPI, **bukan konversi ONNX/TensorRT**. Evaluasi tidak menjadi prasyarat teknis untuk menyalin model, walau sebaiknya hasil evaluasi diperiksa sebelum menggunakan website sebagai demo.

## 6. Buka website FastAPI

```powershell
& .\.venv\Scripts\python.exe -B -m scripts.serve --experiment belajar-01 --port 8000
```

Biarkan terminal terbuka lalu buka [http://127.0.0.1:8000](http://127.0.0.1:8000). Hentikan dengan **Ctrl+C**. Jika port 8000 terpakai, pilih `--port 8001` dan buka alamat dengan port itu. Halaman harus dibuka melalui FastAPI, bukan file `web/index.html` langsung. `GET /api/health` menunjukkan kesiapan model; dokumentasi API interaktif tersedia di `/docs`.

Unggah satu gambar PCB polos, periksa kotak dan tabel deteksi, lalu unduh PNG/CSV bila perlu. Atur folder unduhan browser ke D secara terpisah. Model tidak training saat tombol inspeksi ditekan. Server memuat checkpoint sekali saat startup, jadi jalankan ulang server bila model deployment diganti.

## Batas interpretasi hasil

Split raw Kaggle mencampur kelompok nama `light_##` di train, val, dan test. Hasil test menunjukkan performa pada pembagian Kaggle tersebut, bukan bukti kemampuan pada kelompok sumber yang benar-benar baru. Seluruh gambar evaluasi memiliki cacat beranotasi; kemampuan menolak PCB normal belum diukur. Model juga tidak memeriksa sambungan listrik, solder, atau komponen rakitan. File `reports/` berisi hasil baseline lama dari metode pembagian berbeda dan tidak setara dengan eksperimen ini.
