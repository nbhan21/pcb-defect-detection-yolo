# Arsip Verifikasi Kode

Tes dipindahkan dari folder `tests/` pada root untuk memisahkan pemeriksaan kode dari alur belajar utama. Tidak ada tes yang dihapus. Kode aplikasi tidak mengimpor folder ini; setup, prepare, training, evaluasi dan server dapat dijalankan tanpa menjalankan tes kode.

Ini berbeda dengan `data/prepared/test/`, yaitu data evaluasi model yang tetap dibutuhkan.

Untuk menjalankan tes kembali dari root proyek:

```powershell
Set-Location 'D:\!PROJECT\!persiapan_maganghub\projects\comvis_yolo'
. .\scripts\project-env.ps1
& $ProjectPython -B -m pytest -c archive/verification/pytest.ini archive/verification/tests -q
node --test archive/verification/tests/frontend.test.mjs
```

Node hanya diperlukan untuk tes helper frontend. Website tidak membutuhkan Node/npm build. Konfigurasi pytest di arsip menambahkan root proyek pada import path; import helper JavaScript sudah disesuaikan dengan lokasi arsip.

Tes mencakup integritas data/split, batas training, evaluasi helper, API/inference, ekspor CSV, dan pembatalan request lama. Fixture unit bukan pengganti pengukuran kualitas model; hasil evaluasi model berada di `reports/evaluation.json`.
