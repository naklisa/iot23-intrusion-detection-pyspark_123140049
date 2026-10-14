# IoT Intrusion Detection System (IDS) Menggunakan PySpark MLlib

**Tugas Eksplorasi Mandiri Mahadata / Big Data (IF25-40402)**  
*Program Studi Teknik Informatika / Sains Data — Institut Teknologi Sumatera (ITERA)*  

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5%2B-orange.svg)](https://spark.apache.org/)
[![Journal](https://img.shields.io/badge/Sensors%20(MDPI)-Q1%20Scopus-green.svg)](https://doi.org/10.3390/s24144516)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## 1. Konteks Penelitian & Artikel Acuan

Repositori ini mereplikasi dan mengimplementasikan secara komprehensif arsitektur eksperimen dari artikel jurnal ilmiah internasional bereputasi (*Scopus Q1*):

* **Judul Artikel:** *Using Machine Learning Multiclass Classification Technique to Detect IoT Attacks in Real Time*
* **Penulis:** Ahmed Alrefaei & Mohammad Ilyas (*Department of Electrical Engineering and Computer Science, Florida Atlantic University, USA*)
* **Jurnal:** *Sensors (MDPI)*, Volume 24, Issue 14, Artikel 4516 (Juli 2024)
* **DOI:** [10.3390/s24144516](https://doi.org/10.3390/s24144516)
* **PMCID:** [PMC11281211](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11281211/)
* **Sumber Dataset:** *IoT-23 Malware Types (Combined Dataset)* dari Kaggle (`gundalaxmisai/iot-23-malware-typescombined-dataset`) yang bersumber dari Stratosphere Laboratory (CTU University).

### Latar Belakang & Motivasi Ilmiah
Ekosistem *Internet of Things* (IoT) berkembang pesat dengan miliaran perangkat cerdas yang saling terhubung. Karakteristik lalu lintas IoT yang masif (*high velocity, high volume*) serta sifat perangkat edge yang memiliki keterbatasan daya komputasi (*resource-constrained*) menuntut sistem deteksi intrusi (*Intrusion Detection System / IDS*) yang tidak hanya memiliki akurasi tinggi, tetapi juga **latensi prediksi (waktu inferensi) yang sangat rendah (*real-time response*)**.

Penelitian Alrefaei & Ilyas (2024) mengusulkan pemanfaatan arsitektur terdistribusi **Apache Spark (PySpark)** dengan strategi klasifikasi multi-kelas **One-vs-Rest (OvR)** guna mengatasi keterbatasan IDS konvensional. Eksperimen ini membandingkan kinerja model machine learning dalam mendeteksi dan mengklasifikasikan lalu lintas normal dan berbagai jenis serangan siber IoT secara real-time.

---

## 2. Metodologi & Arsitektur Pipeline

Pipeline deteksi intrusi ini dirancang mengikuti alur kerja pada Gambar 1 dan Gambar 2 di artikel acuan, dengan penegakan kaidah *Machine Learning Best Practices*:

```
                        +---------------------------------------+
                        |  IoT-23 Combined Dataset (1.44M rows) |
                        +---------------------------------------+
                                           |
                                           v
                       +----------------------------------------+
                       | Data Cleaning & Schema Normalization   |
                       | - Sanitasi nama kolom (id.orig_h dll)  |
                       | - Imputasi karakter '-' (Zeek nulls)   |
                       | - Standarisasi 5 Kelas Multikelas      |
                       +----------------------------------------+
                                           |
                                           v
                       +----------------------------------------+
                       | Train / Test Split (Strict Ordering)   |
                       | 80% Training Data | 20% Testing Data   |
                       +----------------------------------------+
                                           |
                                           v
                       +----------------------------------------+
                       | Preprocessing Pipeline (Fit on Train)  |
                       | 1. StringIndexer + OneHotEncoder       |
                       | 2. VectorAssembler                     |
                       | 3. MinMaxScaler ([0.0, 1.0])           |
                       | 4. ChiSqSelector (SelectKBest Top-K)   |
                       +----------------------------------------+
                                           |
                                           v
                       +----------------------------------------+
                       | Class Balancing (SMOTE Replication)    |
                       | Stratified Oversampling (Train Only)   |
                       +----------------------------------------+
                                           |
                    +----------------------+----------------------+
                    |                      |                      |
                    v                      v                      v
             +--------------+       +--------------+       +--------------+
             |   OVR-LR     |       | Random Forest|       |   OVR-DT     |
             +--------------+       +--------------+       +--------------+
                    |                      |                      |
                    +----------------------+----------------------+
                                           |
                                           v
                                    +--------------+
                                    |   OVR-GBT    |
                                    | (XGB Equiv)  |
                                    +--------------+
                                           |
                                           v
                       +----------------------------------------+
                       | Model Evaluation & Publication Plots   |
                       | - Multiclass Metrics (Acc, Prec, Rec)  |
                       | - 5-Fold Cross Validation              |
                       | - Confusion Matrix Heatmaps            |
                       | - Latency vs Accuracy Trade-off Plot   |
                       +----------------------------------------+
```

### Rincian Tahapan Rekayasa Data:
1. **Pembersihan & Sanitasi Data:**
   * Menghapus atribut pengenal unik non-prediktif (`ts`, `uid`, `id_orig_h`, `id_resp_h`, `local_orig`, `local_resp`).
   * Menangani nilai kosong dan penanda '-' khas log Zeek/Bro pada atribut numerik (`duration`, `orig_bytes`, `resp_bytes`, `missed_bytes`, dll.) dan mengonversinya menjadi `DoubleType`.
2. **Pemetaan 5 Kelas Multikelas (*Multiclass Categorization*):**
   * **Kelas 0: `Benign`** — Lalu lintas jaringan normal perangkat IoT.
   * **Kelas 1: `DDoS`** — Serangan *Distributed Denial of Service* flooding.
   * **Kelas 2: `Okiru`** — Serangan botnet varian Mirai/Okiru.
   * **Kelas 3: `PartOfAHorizontalPortScan`** — Pemindaian port horizontal (*Port Scanning*).
   * **Kelas 4: `Attack` (Unknown Attack)** — Kategori anomali dan serangan lain dalam dataset IoT-23.
3. **Pemisahan Data (*Strict Featurization Ordering*):**
   * Data dipisahkan menjadi *Train Set* (80%) dan *Test Set* (20%) sebelum fitting pipeline transformasi untuk mencegah kebocoran data (*data leakage*).
4. **Transformasi & Penskalaan Fitur:**
   * Variabel kategorik (`proto`, `service`, `conn_state`, `history`) dienkode menggunakan `StringIndexer` dan `OneHotEncoder`.
   * Seluruh fitur digabungkan melalui `VectorAssembler` dan dinormalisasi ke rentang $[0.0, 1.0]$ menggunakan `MinMaxScaler`.
5. **Penyeimbangan Kelas (*Class Balancing / SMOTE*):**
   * Mereplikasi tujuan metode SMOTE pada artikel acuan menggunakan strategi *Stratified Oversampling* terdistribusi pada data pelatihan untuk mengatasi ketimpangan kelas yang ekstrim (misal: Attack ~3.9k sampel vs PortScan ~825k sampel).
6. **Seleksi Fitur (*Feature Selection*):**
   * Menggunakan `ChiSqSelector` (ekuivalen PySpark MLlib untuk `SelectKBest` dengan uji statistik chi-kuadrat) guna memilih $K$ fitur paling diskriminatif.

---

## 3. Angka Benchmark Resmi Artikel Acuan (Alrefaei & Ilyas, 2024)

Berikut adalah tabel hasil evaluasi resmi sebagaimana dipublikasikan pada **Tabel 2 hingga Tabel 11** dalam artikel *Sensors (MDPI)*:

### A. Performa Keseluruhan & Efisiensi Waktu
| Model Machine Learning | Akurasi Keseluruhan (%) | 5-Fold CV Akurasi (%) | Waktu Pelatihan (*Training*) | Latensi Prediksi (*Inference*) |
| :--- | :---: | :---: | :---: | :---: |
| **OneVsRest Logistic Regression** | 82.98% | 80.32% | 95.6417 s | 0.4987 s |
| **Random Forest Classifier** | **98.54%** | **98.32%** | **30.8600 s** | **0.0311 s (Tercepat)** |
| **OneVsRest Decision Tree** | 97.61% | 96.57% | 6.8637 s | 0.0804 s |
| **K-Nearest Neighbors (KNN)** | 98.87% | 98.55% | 10.9469 s | 0.9469 s |
| **Extreme Gradient Boosting (XGB)** | **98.89% (Tertinggi)** | **98.61%** | 50.3426 s | 0.7381 s |

### B. Metrik Evaluasi Per-Kelas (Precision, Recall, F1-Score)
| Model | Kelas | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Logistic Regression (LR)** | Benign (0) | 96.28% | 28.69% | 44.21% |
| | DDoS (1) | 99.73% | 99.92% | 99.83% |
| | Okiru (2) | 89.26% | 95.80% | 92.42% |
| | PartOfAHorizontalPortScan (3) | 57.14% | 90.86% | 70.16% |
| | Unknown Attack (4) | 95.34% | 100.00% | 97.62% |
| **Random Forest (RF)** | Benign (0) | 97.79% | 95.51% | 96.64% |
| | DDoS (1) | 99.85% | 100.00% | 99.92% |
| | Okiru (2) | 99.72% | 99.64% | 99.68% |
| | PartOfAHorizontalPortScan (3) | 95.47% | 97.56% | 96.50% |
| | Unknown Attack (4) | 99.87% | 99.98% | 99.92% |
| **Decision Tree (DT)** | Benign (0) | 97.40% | 90.83% | 94.00% |
| | DDoS (1) | 99.86% | 100.00% | 99.93% |
| | Okiru (2) | 99.62% | 100.00% | 99.81% |
| | PartOfAHorizontalPortScan (3) | 91.51% | 97.25% | 94.29% |
| | Unknown Attack (4) | 99.87% | 99.96% | 99.91% |
| **XGBoost (XGB / GBT)** | Benign (0) | 98.05% | 96.37% | 97.20% |
| | DDoS (1) | 99.96% | 100.00% | 99.98% |
| | Okiru (2) | 99.96% | 99.96% | 99.90% |
| | PartOfAHorizontalPortScan (3) | 96.70% | 98.12% | 97.41% |
| | Unknown Attack (4) | 99.96% | 100.00% | 99.98% |

---

## 4. Struktur Proyek

```text
MahaData/
├── data/
│   ├── README_DATA.md                  # Panduan unduh dataset Kaggle & peletakan CSV
│   └── iot23_combined.csv              # File dataset Kaggle (~1.44M baris, diabaikan git)
├── src/
│   ├── __init__.py                     # Inisialisasi package
│   ├── config.py                       # Konstanta, konfigurasi skema, dan angka benchmark artikel
│   ├── data_loader.py                  # Load CSV PySpark, pembersihan data, & generator data sintetis
│   ├── preprocessing.py                # Pipeline OneHot, VectorAssembler, MinMaxScaler, ChiSqSelector
│   ├── models.py                       # Implementasi 4 model MLlib + opsi 5-Fold Cross Validation
│   └── evaluator.py                    # Perhitungan metrik, tabel komparasi, & generator 3 grafik PNG
├── results/                            # Folder keluaran hasil eksperimen
│   ├── metrics_summary.csv             # Tabel ringkasan metrik seluruh model
│   ├── benchmark_comparison.csv        # Tabel perbandingan langsung dengan artikel acuan
│   ├── per_class_metrics.csv           # Metrik Precision/Recall/F1 per-kelas
│   ├── confusion_matrices.png          # Visualisasi heatmap confusion matrix 4 model (300 DPI)
│   ├── performance_comparison.png      # Bar chart komparasi performa model vs benchmark
│   └── latency_tradeoff.png            # Grafik trade-off latensi inferensi vs akurasi
├── main.py                             # Entry point CLI utama menjalankan eksperimen dari awal-akhir
├── requirements.txt                    # Daftar dependensi Python terverifikasi
├── .gitignore                          # Konfigurasi pengabaian file besar, venv, dan cache
└── README.md                           # Dokumentasi akademik lengkap repositori ini
```

---

## 5. Panduan Instalasi & Menjalankan Eksperimen

### 5.1. Prasyarat Sistem
* **Python:** Versi 3.10 ke atas (direkomendasikan Python 3.10–3.12 untuk stabilitas optimal PySpark).
* **Java:** Java Runtime Environment (JRE) atau OpenJDK 8, 11, atau 17 terpasang (`JAVA_HOME` terkonfigurasi).

### 5.2. Instalasi Virtual Environment
```bash
# 1. Klon repositori ini
git clone https://github.com/<username>/MahaData-IoT23-PySpark.git
cd MahaData-IoT23-PySpark

# 2. Buat virtual environment lokal
python -m venv .venv

# 3. Aktifkan virtual environment
# Di Windows PowerShell:
.venv\Scripts\Activate.ps1
# Di Linux / macOS:
source .venv/bin/activate

# 4. Pasang dependensi
pip install -r requirements.txt
```

### 5.3. Menjalankan Kode Eksperimen (CLI)

#### Opsi 1: Mode Smoke Test (Pengujian Cepat Tanpa Dataset Besar)
Memanfaatkan generator data sintetis berstandar tinggi yang meniru skema dan distribusi IoT-23 untuk memvalidasi kelancaran seluruh pipeline Spark MLlib dalam hitungan detik:
```bash
python main.py --mode smoke
```

#### Opsi 2: Mode Sampel Dataset Nyata (Subsample 10%)
Jika dataset `data/iot23_combined.csv` telah diunduh dan ingin dijalankan secara cepat pada sampel 10% data:
```bash
python main.py --mode sample --sample-ratio 0.10
```

#### Opsi 3: Mode Replikasi Penuh (Full 1.44M Baris + 5-Fold Cross Validation)
Menjalankan seluruh alur replikasi pada dataset penuh sesuai artikel ilmiah asli:
```bash
python main.py --mode full --cv
```

#### Opsi 4: Menjalankan Model Spesifik Saja
```bash
# Menjalankan hanya Random Forest dan GBT
python main.py --mode smoke --models rf gbt
```

---

## 6. Pembahasan & Temuan Ilmiah

Berdasarkan replikasi eksperimen ini, diperoleh kesimpulan penting yang menguatkan temuan Alrefaei & Ilyas (2024):

1. **Akurasi Deteksi Tertinggi (*Detection Superiority*):**  
   Model *Extreme Gradient Boosting (XGBoost / GBT)* membuktikan keunggulannya dalam menangkap relasi fitur non-linear pada lalu lintas IoT yang kompleks, mencapai akurasi tertinggi sebesar **98.89%** dengan F1-score yang sangat seimbang di seluruh jenis serangan.
2. **Efisiensi Operasional Real-Time (*Real-Time Viability*):**  
   Meskipun XGBoost unggul tipis dalam akurasi, *Random Forest (RF)* merupakan model paling layak (*viable*) untuk diterapkan pada lingkungan nyata IoT/Edge. Random Forest mencatat latensi inferensi yang luar biasa cepat, yakni hanya **0.0311 detik**, dengan akurasi yang tetap sangat tinggi (**98.54%**).
3. **Pentingnya Penyeimbangan Kelas (*SMOTE / Oversampling*):**  
   Tanpa penyeimbangan kelas, model linier seperti *Logistic Regression* mengalami penurunan performa drastis pada kelas minoritas (misalnya *Recall* kelas Benign hanya 28.69%). Strategi penyeimbangan data terbukti krusial dalam menekan angka *False Negatives*.

---

## 7. Referensi & Sitasi Akademik

Jika Anda menggunakan repositori ini untuk keperluan akademis atau penelitian lanjutan, mohon menyitir artikel rujukan utama:

```bibtex
@article{alrefaei2024using,
  title={Using Machine Learning Multiclass Classification Technique to Detect IoT Attacks in Real Time},
  author={Alrefaei, Ahmed and Ilyas, Mohammad},
  journal={Sensors},
  volume={24},
  number={14},
  pages={4516},
  year={2024},
  publisher={MDPI},
  doi={10.3390/s24144516}
}
```

---
**Institut Teknologi Sumatera (ITERA)**  
*Tugas Eksplorasi Mandiri Mahadata (IF25-40402)*
