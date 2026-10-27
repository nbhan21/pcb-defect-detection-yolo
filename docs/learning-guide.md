# Membaca alur PCB Defect Detection

Dokumen ini menjelaskan hubungan antar-tahap. Untuk perintah yang dijalankan satu per satu dan lokasi hasilnya, ikuti [panduan praktik](getting-started.md).

## Dari gambar ke label

Satu baris label YOLO berisi `class_id x_center y_center width height`. Empat koordinat dinormalisasi ke rentang 0–1. Contoh `4 0.5 0.5 0.1 0.04` berarti kelas `open_circuit` di sekitar tengah gambar. Pada gambar 600 × 600, lebar kotaknya 60 piksel dan tingginya 24 piksel. Ini hanya contoh format.

Ultralytics mencari label dengan nama dasar yang sama seperti gambar di folder `labels/` yang sejajar dengan `images/`. Pada dataset raw Kaggle, sebagian gambar `..._600.jpg` justru mempunyai label `..._256.txt`. Jika dipakai apa adanya, sebagian anotasi tidak akan ditemukan. `scripts/prepare_data.py` membuat salinan `data/kaggle-corrected/` dan memberi **nama salinan label** sesuai gambar. Byte gambar, isi label, serta penempatan train/val/test bawaan Kaggle tetap sama. Dataset raw tidak diedit, dan ZIP tidak dibaca oleh skrip.

Skrip mengaudit semua pasangan sebelum menyalin. Gambar tanpa label, label tanpa gambar, atau pasangan ambigu menghasilkan error. Ringkasan jumlahnya disimpan di `audit.json`; file `dataset.yaml` memberi Ultralytics lokasi ketiga split dan urutan enam kelas.

## Training dan checkpoint

`scripts/train.py` memuat pretrained YOLO11n, lalu menjalankan `model.train(...)` pada salinan data yang sudah cocok namanya. Bobot model diperbarui memakai gambar train dan anotasinya. Validation dipakai Ultralytics selama training untuk memilih `weights/best.pt`; test disimpan untuk pemeriksaan setelah training. Bobot pretrained adalah titik awal, sedangkan enam kelas PCB berasal dari label dataset ini.

Hasil setiap run berada di `experiments/<nama>/training/`. Periksa `results.csv` dan plot training untuk melihat perkembangan loss dan metrik. `best.pt` dapat berbeda dari `last.pt`: yang pertama dipilih berdasarkan validation, yang kedua berasal dari akhir training. Opsi seperti epoch, batch, dan device terlihat langsung pada pemanggilan `model.train(...)`, supaya perubahan konfigurasi mudah dipahami.

## Evaluasi dan prediksi

`scripts/evaluate.py` memuat `best.pt` dan memanggil `model.val(data=..., split='val')`, kemudian `split='test'`. Hasil utama adalah precision, recall, mAP50, dan mAP50–95 dari Ultralytics. mAP menilai peringkat confidence bersama kecocokan lokasi dan kelas; mAP50–95 memakai beberapa ambang IoU sehingga lebih ketat terhadap lokasi kotak. [Dokumentasi validasi Ultralytics](https://docs.ultralytics.com/modes/val) menjelaskan metrik tersebut.

`scripts/predict.py` memakai `YOLO(best.pt).predict(...)` untuk satu gambar. Tahap ini berdiri sendiri setelah training; ia tidak membaca laporan evaluasi. Confidence default 0,25 dapat diubah saat menjalankan skrip. Confidence lebih rendah cenderung menampilkan lebih banyak kandidat, tetapi dapat menambah false positive; confidence lebih tinggi dapat melewatkan cacat yang lemah. Hasil satu gambar berguna untuk inspeksi visual, bukan pengganti metrik seluruh dataset.

Metrik validation/test menggunakan prosedur bawaan Ultralytics, sehingga angka precision dan recall ringkasannya tidak harus sama dengan deteksi yang terlihat pada satu confidence prediksi. Seaborn, bila dipakai, berfungsi untuk visualisasi; metrik deteksi sudah dihitung oleh Ultralytics.

## Dari checkpoint ke website

`scripts/prepare_model.py` menyalin `best.pt` ke folder deployment eksperimen dan menulis metadata sederhana. File `.pt` langsung dipakai oleh FastAPI; tahap ini tidak mengekspor ONNX. Saat `scripts/serve.py` dijalankan, `pcb_inspector/inference.py` memuat model sekali dan memastikan nama enam kelas pada checkpoint sesuai kelas yang ditampilkan website.

Di browser, `web/app.js` mengirim gambar dan confidence ke `POST /api/inspect`. `pcb_inspector/api.py` memeriksa ukuran dan format gambar sebelum memanggil detector. Hasil YOLO diubah menjadi nama kelas, confidence, koordinat kotak, dan PNG anotasi. Website menampilkan hasil dan menyediakan unduhan PNG/CSV; tidak ada database atau riwayat inspeksi. Mengunggah gambar **tidak** melatih model lagi.

## Cara menafsirkan hasil proyek ini

Keenam jenis cacat yang dipelajari berasal dari PCB polos. Model tidak menilai fungsi listrik, solder, atau komponen terpasang. Dataset evaluasi tidak mengukur false alarm pada PCB normal.

Split train/val/test bawaan Kaggle kira-kira 80/10/10 dan mencampur kelompok nama `light_##` antar-split. Karena itu, angka test mencerminkan pembagian dataset tersebut; ia tidak otomatis menunjukkan performa pada kelompok sumber yang sepenuhnya baru. Gambar dari kelompok yang berbeda, pencahayaan lain, atau kamera lain perlu diuji tersendiri bila nanti proyek dipakai lebih luas. Hasil dalam `reports/` adalah baseline lama dari pembagian berbeda dan tidak boleh disamakan dengan laporan eksperimen baru.
