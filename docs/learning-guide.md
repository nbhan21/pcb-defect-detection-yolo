# Membaca Alur PCB Inspect

## Apa yang dipelajari model

Satu baris label YOLO memuat `class_id x_center y_center width height`. Empat koordinat adalah rasio terhadap ukuran gambar, dari 0 sampai 1. Misalnya `4 0.5 0.5 0.1 0.04` berarti cacat `open_circuit` dengan pusat di tengah gambar, lebar 10% dan tinggi 4% gambar. Nilai ini ilustrasi format, bukan output model atau anotasi yang telah diverifikasi.

Pada gambar 600 x 600, ilustrasi tersebut memiliki pusat (300,300), lebar 60, tinggi 24; sudut kotaknya adalah (270,288) dan (330,312). Web mengembalikan format sudut piksel `bbox_xyxy`, sehingga dapat langsung dibandingkan dengan gambar hasil.

## Persiapan data

`scripts/prepare_data.py` memanggil `pcb_inspector/data.py`. Fungsi persiapan memeriksa pasangan gambar/label, nilai koordinat dan class ID, lalu membuat data turunan. Perubahan nama `_256` ke pasangan `_600` tidak mengubah angka anotasi; kesesuaiannya diperiksa melalui gambar bounding boxes.

Split tidak mengikuti pembagian bawaan Kaggle. Nama sumber seperti `light_01` dipakai untuk mengelompokkan gambar dan rotasinya. Jika varian yang berkerabat masuk train dan test, evaluasi dapat memberi kesan kemampuan generalisasi yang terlalu baik. Manifest merekam family, hash file, sumber, dan split agar pilihan ini dapat diaudit.

## Training dan checkpoint

`scripts/train.py` memuat pretrained `yolo11n.pt` dan melanjutkan training pada enam kelas PCB. Backbone dan bagian model lain yang dapat dilatih ikut diperbarui. Bobot pretrained memberi titik awal untuk pola visual; label COCO tidak otomatis menjadi label cacat PCB.

Siklus training: prediksi gambar train, bandingkan dengan kelas/kotak anotasi, hitung loss, lalu optimizer memperbarui bobot. Validation mengukur hasil pada data yang tidak dipakai untuk update bobot. Checkpoint terbaik dipilih dengan validation. Smoke run hanya membuktikan pipeline dan penggunaan hardware bekerja; performanya tidak dianggap hasil akhir.

`scripts/evaluate.py` memilih confidence dari validation, kemudian mengevaluasi test setelah pilihan dibekukan. Precision menjawab berapa banyak temuan yang benar; recall menjawab berapa banyak cacat anotasi yang berhasil ditemukan. mAP memperhitungkan ranking confidence dan kecocokan lokasi; mAP50-95 lebih ketat terhadap posisi kotak daripada mAP50.

## Dari tombol web sampai prediksi

1. `web/app.js` membaca file pilihan dan mengirim `FormData` melalui `fetch('/api/inspect')`. File dan confidence menjadi input request.
2. `pcb_inspector/api.py` memeriksa kesiapan model, confidence, ukuran file dan format gambar. Pillow mendecode byte asli; nama ekstensi saja tidak dianggap bukti format.
3. Orientasi EXIF dinormalisasi, kemudian gambar RGB masuk ke `PCBDetector.inspect` di `pcb_inspector/inference.py`.
4. Model menjalankan inference. Hasilnya diubah menjadi class ID, nama kelas, confidence dan sudut bounding box pada ukuran gambar asli yang telah diorientasikan.
5. Backend menggambar kotak pada salinan gambar, membuat PNG dalam memori, dan mengembalikan JSON berisi prediksi dan PNG sebagai data URI.
6. Frontend menampilkan gambar hasil, menghitung tampilan berdasarkan data response, dan membentuk unduhan PNG/CSV. Tidak ada record inspeksi permanen di database.

Training tidak terjadi ketika tombol inspeksi ditekan. Web memuat checkpoint sekali saat startup, lalu menggunakannya untuk prediksi. Hash checkpoint dan mapping kelas diperiksa untuk mencegah weights/metadata yang salah dipakai sebagai detector PCB.

## Mengubah confidence

Batas confidence yang lebih rendah dapat menampilkan lebih banyak kandidat sekaligus menambah false positive. Batas lebih tinggi dapat menyembunyikan kandidat lemah sekaligus menambah cacat yang terlewat. Nilai default dipilih dari validation; menggeser slider di demo adalah eksplorasi inference, bukan training ulang.

Saat file atau confidence berubah, hasil lama dibatalkan agar unduhan tidak tertukar dengan input baru. Response dari request yang sudah kedaluwarsa juga diabaikan. Jika ada proses inference lain, API memberi 429 agar satu model tidak dipakai bersamaan di luar batas yang direncanakan.

## Membaca kesalahan

Periksa gambar dengan ground truth dan prediksi: cacat terlewat adalah false negative, kotak pada bagian normal adalah false positive, kelas yang tertukar adalah salah klasifikasi, dan kotak yang meleset dapat gagal pada batas IoU lebih ketat. Metrik agregat tidak menggantikan pemeriksaan contoh ini.

Perubahan layout PCB dan kondisi foto dapat mengubah hasil. Proyek membuktikan pipeline dan performa pada data evaluasi yang dicatat; pemeriksaan kelayakan PCB fisik memerlukan prosedur lain.

## Menelusuri satu inspeksi aktual

Pada gambar validation `l_light_07_missing_hole_01_1_600.jpg`, aplikasi final pada CUDA menghasilkan satu kandidat `missing_hole`, class ID 2, confidence sekitar 79,0%, dan kotak sekitar `[299.6, 163.4, 333.8, 197.2]`. Angka dibulatkan dari output nyata; ini satu contoh input, bukan gambaran akurasi seluruh dataset.

Buka `pcb_inspector/api.py`: `UploadFile` menerima file multipart dan `Form` menerima confidence. Route memeriksa kesiapan model dan input, lalu memanggil pekerjaan decode/inference melalui `run_in_threadpool`. Route async tetap dapat melayani health ketika pekerjaan yang memblokir berjalan di thread. Lock menolak inference kedua dengan 429; pengujian concurrency memeriksa perilaku ini.

Model dimuat sekali melalui lifespan aplikasi. `PCBDetector` mengubah output tensor menjadi dictionary Python yang berisi nama kelas, skor, koordinat dan PNG. FastAPI menyerialisasikannya menjadi JSON. `web/app.js` menerima JSON melalui fetch, membuat elemen tabel dan blob unduhan. Hash checkpoint pada health dan response dijaga konsisten; pergantian model membatalkan hasil lama agar metadata tidak tertukar.

Eksperimen final menghasilkan validation mAP50 87,11% dan test 65,80%. Gap ini menunjukkan mengapa web yang berhasil menjalankan model belum berarti model andal di semua board. Baca `reports/error-analysis.md` bersama visual lokal untuk memahami kesalahan model sebelum menafsirkan confidence.
