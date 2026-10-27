# PCB Defect Detection with YOLO11n

Proyek deteksi enam jenis cacat visual pada **PCB polos**. Model YOLO11n dilatih dengan dataset beranotasi, lalu digunakan oleh website lokal FastAPI untuk menampilkan bounding box dan mengunduh hasil PNG/CSV.

Dataset berasal dari [PCB Defect dataset di Kaggle](https://www.kaggle.com/datasets/norbertelter/pcb-defect-dataset/data). Paketnya sudah berisi label YOLO. Enam kelasnya adalah `mouse_bite`, `spur`, `missing_hole`, `short`, `open_circuit`, dan `spurious_copper`.

## Jalankan bertahap

Ikuti [panduan dari dataset sampai website](docs/getting-started.md) untuk setup dan penjelasan hasil setiap tahap. Jalankan perintah berikut **satu per satu** dari root proyek setelah environment, folder dataset raw, dan pretrained `models/base/yolo11n.pt` tersedia:

```powershell
& .\.venv\Scripts\python.exe -B -m scripts.prepare_data
& .\.venv\Scripts\python.exe -B -m scripts.train --experiment belajar-01 --epochs 40 --batch 4
& .\.venv\Scripts\python.exe -B -m scripts.evaluate --experiment belajar-01 --batch 4
& .\.venv\Scripts\python.exe -B -m scripts.predict --experiment belajar-01 --image '.\data\kaggle-corrected\val\images\light_01_mouse_bite_13_1_600.jpg'
& .\.venv\Scripts\python.exe -B -m scripts.prepare_model --experiment belajar-01
& .\.venv\Scripts\python.exe -B -m scripts.serve --experiment belajar-01
```

Setelah server berjalan, buka [http://127.0.0.1:8000](http://127.0.0.1:8000). Dokumentasi API tersedia di `/docs`.

`prepare_data.py` membuat **satu salinan** pada `data/kaggle-corrected/`: nama label yang berbeda disamakan dengan nama gambar, sementara isi file dan split train/val/test bawaan Kaggle dipertahankan. Folder `data/raw/` tidak diubah. Hasil model tiap percobaan berada di `experiments/<nama>/`. Data, checkpoint, cache, dan hasil lokal tidak masuk Git.

Proyek ini memakai Python `venv`, bukan Anaconda. Pada mesin proyek, semua file kerja, termasuk `.venv`, data, model, cache, dan output, ditempatkan di drive D. `setup.ps1` hanya untuk setup awal; tahap berikutnya memakai perintah Python langsung.

## Membaca hasil

Metrik evaluasi berasal dari `model.val()` Ultralytics pada split validation dan test Kaggle. Kelompok gambar bernama `light_##` dapat muncul di lebih dari satu split bawaan, sehingga hasil test **tidak membuktikan** kemampuan pada kelompok sumber yang sepenuhnya baru. Dataset ini juga belum mengukur false alarm pada PCB normal. Prediksi website perlu ditinjau manusia dan tidak memastikan fungsi listrik PCB.

File dalam `reports/` berasal dari **eksperimen baseline sebelumnya dengan pembagian data berbeda**; angkanya tidak boleh dibandingkan langsung dengan eksperimen baru. Baca [learning guide](docs/learning-guide.md) untuk konsep dan alur kode.

Dataset dan checkpoint disimpan lokal. Periksa ketentuan sumber Kaggle dan [lisensi Ultralytics](https://www.ultralytics.com/license) sebelum mendistribusikan keduanya. Website hanya memproses satu gambar JPEG/PNG per permintaan; tidak ada akun atau riwayat inspeksi.
