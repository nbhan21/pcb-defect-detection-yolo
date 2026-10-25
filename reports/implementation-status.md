# Status Implementasi

Pembaruan 25 Oktober 2026. Status ini diperbarui berdasarkan eksekusi aktual.

- Desain disetujui, implementation plan tersimpan, repo public telah dibuat.
- Data turunan selesai: 5.856 train / 2.112 validation / 2.700 test; 2.667 pasangan nama label diperbaiki pada salinan kerja.
- Pemeriksaan visual enam kelas dan varian rotasi selesai. Hash file identik tidak melewati split.
- Suite Python di venv proyek: 50 passed; helper frontend: 7 passed. Ada satu upstream deprecation warning dari Starlette/AnyIO.
- UI dan API telah diimplementasikan; browser memverifikasi model not-ready, file bukan gambar, dan layout lebar 390 piksel tanpa overflow horizontal. Alur model terlatih/download belum diuji.
- Review kode diselesaikan; perbaikan batas epoch/waktu, konsistensi sumber ZIP, MIME gambar, dan provenance checkpoint dilengkapi tes regresi yang relevan.
- Hash seluruh 10.668 gambar prepared cocok dengan manifest; lihat prepared-integrity.json.
- Pretrained weights YOLO11n tersedia lokal di D:.
- Environment lengkap terpasang, 55 paket kompatibel, requirements.txt dibekukan. Torch 2.5.1+cu121 menjalankan kernel CUDA pada RTX 3050 Laptop, runtime CUDA 12.1.
- Smoke satu epoch batch 4 selesai: peak allocated 658.549.760 byte, reserved 861.929.472 byte. Ini verifikasi pipeline, bukan model final.
- Training utama berjalan, maksimum 40 epoch dengan budget satu jam pada batas epoch; evaluasi test akhir belum dilakukan.
- Browser menjalankan checkpoint smoke nyata di CPU untuk memeriksa hasil nol deteksi, CSV header-only, perubahan confidence yang membatalkan unduhan lama, serta penolakan file bukan gambar. Belum diklaim sebagai verifikasi kualitas model final.
- Browser in-app memakai Downloads bawaan di C untuk CSV uji; file 59 byte itu segera dipindah ke reports/browser/downloads di D. Pengaturan folder unduhan browser tidak diwarisi dari launcher server; lihat README.

Tidak ada hasil model yang dibuat-buat. Dataset dan checkpoint tidak diunggah ke repo public.
