# PCB Defect Detection with YOLO11n

Proyek pembelajaran untuk mendeteksi cacat visual pada **PCB polos**. Model pretrained YOLO11n dilatih ulang pada enam kelas cacat, lalu digunakan oleh website lokal FastAPI untuk menampilkan bounding box, ringkasan deteksi, dan unduhan PNG/CSV.

Dataset latihan diperoleh dari [PCB Defect dataset di Kaggle](https://www.kaggle.com/datasets/norbertelter/pcb-defect-dataset/data), yang diunggah akun `norbertelter`. Paketnya sudah memuat gambar dan anotasi YOLO, sehingga tidak perlu memberi label dari nol. Dataset ini berkaitan dengan penelitian [Ding dkk., *TDD-Net*](https://doi.org/10.1049/trit.2019.0019). Kredit pengunggah Kaggle tidak berarti pengunggah tersebut dipastikan membuat anotasi aslinya.

## Yang dideteksi

| Kelas model | Cacat visual |
|---|---|
| `mouse_bite` | Tepi jalur tembaga terkikis atau berlekuk |
| `spur` | Tonjolan kecil pada jalur tembaga |
| `missing_hole` | Lubang pada pad tidak terbentuk |
| `short` | Jalur yang seharusnya terpisah tampak tersambung |
| `open_circuit` | Jalur tembaga tampak terputus |
| `spurious_copper` | Sisa tembaga di area yang tidak semestinya |

## Jalankan bertahap

Panduan lengkap, termasuk input dan output setiap perintah: **[Panduan praktik](docs/getting-started.md)**. Penjelasan konsep dan alur kode: [Learning guide](docs/learning-guide.md).

Jalankan perintah dari root repo di PowerShell. Tempatkan proyek, Python, uv, `.venv`, dataset, model, cache, dan output pada drive D: bila mengikuti batasan komputer proyek ini. `setup.ps1` hanya dipakai untuk instalasi awal; ganti contoh path Python dan uv dengan lokasi executable yang benar di drive proyek.

```powershell
.\setup.ps1 -BasePython 'D:\Tools\Python312\python.exe' -UvPath 'D:\Tools\uv.exe'
```

Jika `.venv` proyek sudah tersedia, setup tidak perlu diulang. Simpan dan ekstrak ZIP Kaggle menjadi `data/raw/pcb-defect-dataset/`, lalu tempatkan pretrained `yolo11n.pt` di `models/base/`. Setelah itu, **jalankan satu tahap, periksa hasilnya, baru lanjut**:

```powershell
& .\.venv\Scripts\python.exe -B -m scripts.prepare_data --experiment belajar-01
& .\.venv\Scripts\python.exe -B -m scripts.train --experiment belajar-01 --batch 4 --epochs 40 --time-hours 1
& .\.venv\Scripts\python.exe -B -m scripts.evaluate --experiment belajar-01 --batch 4
& .\.venv\Scripts\python.exe -B -m scripts.predict --experiment belajar-01 --image '.\experiments\belajar-01\prepared\val\images\l_light_07_missing_hole_01_1_600.jpg'
& .\.venv\Scripts\python.exe -B -m scripts.prepare_model --experiment belajar-01
& .\.venv\Scripts\python.exe -B -m scripts.serve --experiment belajar-01
```

Script Python mengarahkan file kerja framework ke `.local/` pada proyek sebelum memuat Torch/Ultralytics. Semua hasil run baru berada di `experiments/belajar-01/` dan tidak menimpa baseline lama. Nama eksperimen lain dapat digunakan untuk percobaan berikutnya. Training memakai GPU CUDA; batch 4 adalah pengaturan awal untuk RTX 3050 Laptop 4 GB yang digunakan pada baseline, bukan jaminan cocok untuk perangkat lain. Tidak ada training atau persiapan data yang dijalankan otomatis ketika server dimulai.

Setelah server menampilkan alamatnya, buka [http://127.0.0.1:8000](http://127.0.0.1:8000). Dokumentasi API ada di `/docs`. Server dapat memakai baseline lokal lama dengan menjalankan `scripts.serve` **tanpa** `--experiment`. Pada clone baru tanpa checkpoint, status model akan belum siap sampai tahap persiapan model selesai. Atur folder unduhan browser ke D: bila PNG/CSV juga harus tersimpan di sana.

## Baseline sebelumnya

Eksperimen awal yang sudah dijalankan menghasilkan **test mAP50 65,80%** dan **mAP50–95 26,52%** pada 2.700 gambar test dari family sumber yang dipisah dari train dan validation. Confidence 0,60 dipilih dari validation sebelum test dihitung. [Metrik rinci](reports/evaluation.json), [audit data](reports/data-audit.json), dan [analisis kesalahan](reports/error-analysis.md) adalah hasil **baseline lama**, bukan hasil otomatis dari run `belajar-01`.

Selisih validation mAP50 87,11% dan test mAP50 65,80% menunjukkan keterbatasan generalisasi baseline. Seluruh gambar evaluasi memiliki anotasi cacat; false-alarm rate pada PCB normal belum diukur. Hasil website perlu diperiksa manusia dan tidak memastikan kelayakan listrik atau mutu PCB produksi.

## Data, model, dan lisensi

ZIP dataset, gambar, checkpoint `.pt`, dan hasil prediksi disimpan lokal serta diabaikan Git. Halaman Kaggle mencantumkan lisensi **Unknown** saat diperiksa; periksa ketentuan sumber sebelum menyebarkan dataset atau model turunan. [Ultralytics juga memiliki ketentuan lisensi sendiri](https://www.ultralytics.com/license).

Tahap `scripts.prepare_model` menyiapkan checkpoint `.pt` beserta metadata untuk FastAPI. Proyek ini tidak mengonversi model ke ONNX. Website hanya memproses satu gambar JPEG/PNG per request; tidak ada akun, database, atau riwayat inspeksi.

Antarmuka mengikuti gaya [industrial-brutalist-ui](https://github.com/Leonxlnx/taste-skill-classics) dengan palet biru terang.
