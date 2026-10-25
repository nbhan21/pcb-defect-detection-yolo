# Analisis Kesalahan Baseline PCB

Evaluasi akhir 25 Oktober 2026. Checkpoint dan confidence 0,60 dipilih menggunakan validation sebelum test. Tidak ada tuning ulang dari hasil test.

## Hasil per kelas pada test

Precision/recall/F1 memakai agregasi TP/FP/FN dengan IoU ≥ 0,50 pada confidence 0,60. mAP berasal dari kurva Ultralytics, bukan satu confidence saja.

| Kelas | Precision | Recall | F1 | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|
| mouse_bite | 61.90% | 70.15% | 65.77% | 73.15% | 32.17% |
| spur | 97.33% | 45.62% | 62.13% | 59.72% | 24.20% |
| missing_hole | 73.73% | 94.17% | 82.71% | 94.34% | 40.53% |
| short | 65.10% | 39.97% | 49.53% | 35.10% | 9.48% |
| open_circuit | 94.46% | 66.86% | 78.30% | 84.57% | 37.24% |
| spurious_copper | 44.90% | 50.00% | 47.31% | 47.92% | 15.52% |

## Temuan dari contoh visual

- Tulisan/silkscreen pada board, termasuk bagian tulisan SOUND dan 500S, menghasilkan false positive `spurious_copper` pada contoh kegagalan yang ditinjau. Kelas ini memiliki 454 FP dan 370 TP pada test.
- Beberapa `short` terlewat atau kotak prediksinya tidak cocok dengan kotak anotasi; pada contoh lain muncul prediksi di jalur normal. Recall test sekitar 40,0% dan mAP50–95 sekitar 9,5%, sehingga kelas ini belum dapat diandalkan.
- `spur` memiliki precision tinggi tetapi recall rendah pada operating point ini: banyak cacat anotasi belum ditemukan.
- `missing_hole` merupakan kelas terkuat dalam eksperimen ini, dengan mAP50 sekitar 94,3% dan recall sekitar 94,2%. Ini tidak menjamin semua lubang hilang pada board lain akan ditemukan.
- mAP50–95 agregat 26,5%, jauh di bawah mAP50 65,8%; kecocokan lokasi kotak pada kriteria lebih ketat masih lemah.

## Batas interpretasi

Validation mAP50 87,1% turun menjadi 65,8% pada test. Perbedaan pola board merupakan kemungkinan penjelasan yang konsisten dengan split per family dan contoh visual; eksperimen ini tidak mengisolasi sebab penurunan tersebut. Nama family juga bukan bukti bahwa setiap grup pasti berasal dari board fisik independen.

Manifest menunjukkan nol gambar dengan label kosong pada train, validation, maupun test. Evaluasi ini belum mengukur false-alarm rate pada PCB normal tanpa cacat. Prediksi kosong tidak berarti PCB lolos QC.

Visual lengkap tersedia lokal di `reports/visuals/test-errors.png` dan `reports/visuals/test-detections.png`. Oranye adalah anotasi dan biru adalah prediksi. Gambar tersebut tidak disertakan dalam Git karena memuat gambar sumber. Contoh sukses dipilih berdasarkan TP tinggi, dan contoh gagal berdasarkan jumlah FP+FN; keduanya bukan sampel representatif acak.

Semua angka merupakan hasil baseline yang dibekukan; tidak ada klaim akurasi produksi atau generalisasi ke PCB rakitan, solder, dan foto bebas dengan kondisi berbeda.
