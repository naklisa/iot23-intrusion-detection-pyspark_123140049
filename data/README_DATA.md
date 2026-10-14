# Panduan Pengunduhan & Peletakan Dataset IoT-23 (Combined Dataset)

Dokumen ini memuat panduan lengkap pengunduhan dan penataan dataset untuk eksperimen replikasi artikel jurnal:
> **Ahmed Alrefaei & Mohammad Ilyas (2024)**  
> *"Using Machine Learning Multiclass Classification Technique to Detect IoT Attacks in Real Time"*  
> **Sensors (MDPI)**, Vol. 24, No. 14, Artikel 4516.  
> DOI: [10.3390/s24144516](https://doi.org/10.3390/s24144516)

---

## 1. Sumber Dataset Resmi
Dataset yang digunakan dalam penelitian ini merupakan varian gabungan (*combined subset*) dari **IoT-23 Dataset** (Stratosphere Laboratory, CTU University). Dataset ini diunggah dan dibagikan secara publik di Kaggle:

- **Nama Repositori Kaggle:** `gundalaxmisai/iot-23-malware-typescombined-dataset`
- **URL Kaggle:** [https://www.kaggle.com/datasets/gundalaxmisai/iot-23-malware-typescombined-dataset](https://www.kaggle.com/datasets/gundalaxmisai/iot-23-malware-typescombined-dataset)
- **Nama File Utama:** `iot23_combined.csv` (atau terkadang diekstrak sebagai `iot23_combined_new.csv` / `iot23_combined.csv`)
- **Total Baris:** Sekitar **1.444.674** baris rekaman lalu lintas jaringan IoT (*network connection flows*).

---

## 2. Cara Mengunduh Dataset

### Opsi A: Unduh Manual Melalui Browser
1. Buka tautan Kaggle di atas: [Kaggle IoT-23 Malware Types Dataset](https://www.kaggle.com/datasets/gundalaxmisai/iot-23-malware-typescombined-dataset).
2. Masuk (*log in*) menggunakan akun Kaggle Anda.
3. Klik tombol **Download (Archive)** di pojok kanan atas halaman dataset.
4. Ekstrak file arsip `.zip` yang telah diunduh.
5. Pindahkan file CSV hasil ekstrak ke dalam folder `data/` pada repositori ini dan pastikan diberi nama:
   ```text
   data/iot23_combined.csv
   ```

### Opsi B: Unduh Menggunakan Kaggle CLI
Jika Anda telah memasang Kaggle API CLI dan mengonfigurasi `kaggle.json`:
```bash
# Dari root direktori proyek ini:
kaggle datasets download -d gundalaxmisai/iot-23-malware-typescombined-dataset -p data/ --unzip
```
Jika nama file hasil ekstraksi adalah `iot23_combined_new.csv`, ubah namanya menjadi `iot23_combined.csv`:
```bash
# Di Windows PowerShell:
Rename-Item -Path "data/iot23_combined_new.csv" -NewName "iot23_combined.csv"

# Di Linux / macOS:
mv data/iot23_combined_new.csv data/iot23_combined.csv
```

---

## 3. Struktur Direktori Data
Pastikan struktur folder Anda terlihat seperti berikut:
```text
MahaData/
├── data/
│   ├── README_DATA.md          <-- Dokumen panduan ini
│   └── iot23_combined.csv      <-- File dataset berukuran ~700MB - 1GB (diabaikan oleh git)
├── src/
├── results/
└── main.py
```

---

## 4. Gambaran Skema dan Fitur Dataset
Dataset IoT-23 merekam alur koneksi jaringan (*flow logs*) yang dihasilkan oleh perangkat lunak analisis keamanan Zeek/Bro IDS. Atribut-atribut utama meliputi:

| Nama Kolom | Tipe Data | Deskripsi |
| :--- | :--- | :--- |
| `ts` | Float/Double | Timestamp saat koneksi dimulai |
| `uid` | String | Pengenal unik koneksi (*connection hash UID*) |
| `id.orig_h` | String | Alamat IP pengirim / sumber (*Originating IP*) |
| `id.orig_p` | Integer | Nomor port pengirim (*Originating Port*) |
| `id.resp_h` | String | Alamat IP penerima / tujuan (*Responding IP*) |
| `id.resp_p` | Integer | Nomor port penerima (*Responding Port*) |
| `proto` | String | Protokol transport (`tcp`, `udp`, `icmp`) |
| `service` | String | Layanan lapisan aplikasi yang terdeteksi (`http`, `dns`, `dhcp`, `-`) |
| `duration` | Float | Durasi koneksi jaringan dalam detik |
| `orig_bytes` | Integer/Float | Jumlah payload bytes yang dikirim pengirim |
| `resp_bytes` | Integer/Float | Jumlah payload bytes yang dikirim penerima |
| `conn_state` | String | Status koneksi TCP/IP (`S0`, `SF`, `REJ`, `RSTO`, dll.) |
| `history` | String | Riwayat state koneksi Zeek (`ShADF`, `D`, `Sr`, dll.) |
| `orig_pkts` | Integer | Jumlah paket yang dikirim pengirim |
| `orig_ip_bytes`| Integer | Total byte level-IP pengirim (termasuk header paket) |
| `resp_pkts` | Integer | Jumlah paket yang dikirim penerima |
| `resp_ip_bytes`| Integer | Total byte level-IP penerima (termasuk header paket) |
| `label` | String | Label kelas klasifikasi serangan / normal |

---

## 5. Pemetaan 5 Kelas Multiclass Target (Sesuai Artikel Acuan)
Sesuai dengan metodologi Alrefaei & Ilyas (2024), 5 kelas yang menjadi fokus klasifikasi adalah:

1. **Kelas 0: `Benign`** — Lalu lintas jaringan normal perangkat IoT rumah pintar.
2. **Kelas 1: `DDoS`** — Serangan *Distributed Denial of Service* (flooding traffic).
3. **Kelas 2: `Okiru`** — Serangan botnet varian Mirai/Okiru.
4. **Kelas 3: `PartOfAHorizontalPortScan`** — Serangan pemindaian port horizontal (*Port Scanning*).
5. **Kelas 4: `Attack` (Unknown Attack)** — Kategori serangan umum/anomali lainnya dalam dataset IoT-23.

---

## 6. Mode Tanpa Dataset (Smoke Test / Data Sintetis)
Jika Anda belum mengunduh dataset lengkap berukuran besar, sistem menyediakan generator data sintetis otomatis:
```bash
python main.py --mode smoke
```
Perintah ini akan secara otomatis membangkitkan data sintetis dengan skema yang 100% identik untuk memvalidasi kelancaran seluruh pipeline Spark MLlib tanpa perlu menunggu unduhan file 1.44M baris.
