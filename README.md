# Segmentasi Pelanggan dengan K-Means

Proyek ini mengelompokkan pelanggan retail berdasarkan profil dan perilaku belanja menggunakan **unsupervised learning**. Hasil segmentasi diterjemahkan menjadi persona pelanggan dan usulan strategi pemasaran untuk setiap kelompok.

Notebook utama: [`customer_segmentation_kmeans_fixed.ipynb`](customer_segmentation_kmeans_fixed.ipynb).

## Dataset

Notebook menggunakan dataset Kaggle dengan ID `vishakhdapat/customer-segmentation-clustering`, yang diunduh melalui `kagglehub`. File input yang dibaca adalah **`customer_segmentation.csv`**.

Berdasarkan output yang tersimpan dalam notebook:

- Data awal berisi **2.240 pelanggan dan 29 kolom**.
- Tanggal pendaftaran pelanggan berada pada periode **2012–2014**.
- Data mencakup profil pelanggan, pendapatan, pengeluaran per kategori produk, transaksi melalui berbagai kanal, dan respons kampanye.
- Setelah pembersihan, terdapat **2.212 pelanggan** yang digunakan dalam analisis.

Pembersihan mencakup penghapusan 24 baris dengan `Income` kosong, dua kolom konstan (`Z_CostContact` dan `Z_Revenue`), serta empat baris di luar kriteria `Year_Birth > 1900` dan `Income < 200000`.

## Alur Analisis

1. Memuat dan memeriksa dataset.
2. Membersihkan data dan mengubah tipe tanggal.
3. Membuat fitur seperti `Age`, `Customer_Days`, `Spent`, `Purchases`, `Children`, dan `Family_Size`.
4. Melakukan eksplorasi data melalui distribusi, korelasi, dan visualisasi perilaku belanja.
5. Menerapkan transformasi `log1p` pada fitur tertentu dan standardisasi dengan `StandardScaler`.
6. Menggunakan PCA dengan ambang variansi kumulatif minimal 80% sebagai skenario alternatif.
7. Membandingkan K-Means, Agglomerative Clustering, dan Gaussian Mixture pada data terstandardisasi serta data PCA untuk K = 2–8. DBSCAN diuji secara terpisah pada data PCA.
8. Memilih model, menguji stabilitas, dan menyusun profil serta rekomendasi per cluster.
9. Menyimpan model beserta komponen preprocessing.

Sembilan fitur pembentuk cluster adalah `Income`, `Age`, `Spent`, `Children`, `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, dan `NumWebVisitsMonth`.

## Hasil Utama

Model akhir adalah **K-Means dengan K = 3 pada data PCA**, menggunakan `random_state=42` dan `n_init=20`.

| Metrik | Nilai |
| --- | ---: |
| Silhouette Score | 0,351 |
| Calinski–Harabasz Index | 1621,066 |
| Davies–Bouldin Index | 1,087 |
| Adjusted Rand Index pada lima seed pembanding | 0,997–1,000 |

K = 2 memiliki Silhouette lebih tinggi (0,393), tetapi K = 3 dipilih untuk mendapatkan segmentasi pemasaran yang lebih terperinci. Angka di atas mengacu pada output tersimpan; hasil eksekusi ulang dapat sedikit berbeda akibat versi dataset atau pustaka.

| Cluster | Persona | Pelanggan | Rata-rata `Income` | Rata-rata `Spent` | Usulan strategi |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | Pelanggan Premium | 645 | 75.204,58 | 1.255,76 | Program loyalitas/VIP, produk premium, dan katalog eksklusif |
| 1 | Pelanggan Hemat | 878 | 32.246,61 | 70,57 | Paket keluarga terjangkau dan insentif konversi pembelian melalui website |
| 2 | Keluarga Pemburu Diskon | 689 | 55.317,01 | 684,12 | Bundling keluarga, promo diskon, dan flash sale online |

Nilai pendapatan dan belanja mengikuti satuan dataset. Rekomendasi tersebut merupakan interpretasi analisis dan belum diuji melalui eksperimen pemasaran.

## File yang Perlu Disertakan

Susunan berikut merupakan susunan berkas untuk pengumpulan. Pastikan notebook dan CSV benar-benar diunggah bersama README ini.

```text
.
├── README.md
├── customer_segmentation_kmeans_fixed.ipynb
├── customer_segmentation.csv          # Input untuk presentasi final
└── customer_segmentation_model.pkl    # Dihasilkan setelah notebook dijalankan
```

File `.pkl` dapat dibuat ulang dari notebook. CSV merupakan input mentah, bukan file model.

## Cara Menjalankan

### 1. Siapkan proyek dan Python

Unduh repository melalui **Code → Download ZIP**, kemudian ekstrak dan buka terminal di folder hasil ekstraksi. Alternatifnya, clone repository menggunakan URL repository ini.

Gunakan Python 3.12 untuk mendekati lingkungan notebook asal (metadata mencatat Python 3.12.10). Buat lingkungan virtual:

```bash
python -m venv .venv
```

Aktifkan pada Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Atau pada macOS/Linux:

```bash
source .venv/bin/activate
```

Jika perintah `python` tidak tersedia pada macOS/Linux, gunakan `python3` untuk membuat lingkungan virtual.

### 2. Instal dependensi

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib seaborn scikit-learn joblib kagglehub jupyterlab ipykernel
```

Versi pustaka asal tidak dicantumkan dalam notebook, sehingga instruksi ini tidak mengunci versi dependensi.

### 3. Buka dan jalankan notebook

Dari folder proyek, jalankan:

```bash
python -m jupyterlab
```

Buka `customer_segmentation_kmeans_fixed.ipynb`, pilih kernel Python dari lingkungan yang digunakan, lalu jalankan **Restart Kernel and Run All Cells**. Tunggu sampai seluruh proses selesai, pastikan tidak ada error, lalu simpan notebook agar tabel dan grafik tersimpan serta dapat dilihat melalui GitHub.

Secara bawaan, bagian **Load Data** menjalankan:

```python
path = kagglehub.dataset_download("vishakhdapat/customer-segmentation-clustering")
df = pd.read_csv(os.path.join(path, "customer_segmentation.csv"))
```

Pengunduhan pertama memerlukan koneksi internet. Jika Kaggle meminta autentikasi, selesaikan autentikasi pada lingkungan yang digunakan, atau gunakan CSV lokal sebagaimana dijelaskan di bawah.

## Menyiapkan File Input untuk Presentasi Final

Untuk memenuhi ketentuan file input, sertakan **`customer_segmentation.csv`** di repository dan siapkan notebook agar membacanya secara lokal sebelum presentasi.

1. Jalankan bagian **Load Data** dengan koneksi internet agar dataset tersedia.
2. Salin file asli dari lokasi unduhan ke folder yang sama dengan notebook. Jalankan kode berikut setelah variabel `path` tersedia, dengan direktori kerja berada di folder proyek:

   ```python
   from pathlib import Path
   import shutil

   source = Path(path) / "customer_segmentation.csv"
   target = Path.cwd() / "customer_segmentation.csv"
   if source.resolve() != target.resolve():
       shutil.copy2(source, target)
   print("File input presentasi:", target)
   ```

3. Ganti isi cell pengunduhan dan pembacaan data pada bagian **Load Data** dengan:

   ```python
   df = pd.read_csv("customer_segmentation.csv")
   print(df.shape)
   df.head()
   ```

4. Jalankan ulang seluruh cell dari kernel baru, lalu simpan notebook beserta outputnya.
5. Unggah notebook dan CSV tersebut ke repository bersama README.

Langkah penggantian cell di atas adalah penyesuaian untuk presentasi; notebook asal masih menggunakan unduhan KaggleHub. Setelah CSV lokal dan seluruh dependensi tersedia, proses analisis tidak memerlukan unduhan dataset saat presentasi.

## Output

Notebook menampilkan hasil pembersihan data, grafik eksplorasi, perbandingan model, evaluasi cluster, uji stabilitas, serta profil dan persona pelanggan.

Cell terakhir menghasilkan `customer_segmentation_model.pkl` pada direktori kerja. File tersebut berisi:

- `scaler`: objek `StandardScaler` yang sudah dilatih.
- `pca`: objek PCA yang sudah dilatih.
- `kmeans`: model K-Means akhir.
- `features`: urutan fitur input.
- `log_cols`: daftar fitur yang ditransformasi dengan `log1p`.

File model bukan pipeline lengkap untuk data mentah. Penggunaan pada data baru membutuhkan feature engineering, urutan fitur, dan transformasi log yang sama sebelum menerapkan scaler, PCA, dan model.

## Keterbatasan

Silhouette sekitar 0,35 menunjukkan pemisahan antar-cluster belum terlalu tegas. Dataset bersifat historis, sehingga penggunaan untuk keputusan bisnis saat ini memerlukan data yang lebih baru dan validasi tambahan. Nomor cluster merupakan label teknis; interpretasi segmen harus mengacu pada profilnya.

## Checklist Pengumpulan

- [ ] Notebook utama sudah diunggah ke repository.
- [ ] Seluruh cell sudah dijalankan tanpa error dan output tersimpan.
- [ ] Kode serta seluruh file yang diperlukan tersedia.
- [ ] File input `customer_segmentation.csv` sudah disertakan untuk presentasi final.
- [ ] Notebook untuk presentasi sudah diuji dengan file input lokal.
- [ ] Cara menjalankan pada README sudah dicoba pada lingkungan yang digunakan.
- [ ] Repository disetel publik, atau akun panitia sudah diberi akses.
