# suku-face-recognition

## 📌 Deskripsi Program
**suku-face-recognition** adalah sistem pengenalan wajah berbasis deep learning yang mampu melakukan identifikasi dan klasifikasinya. Sistem ini menggunakan pendekatan multi-task untuk melakukan berbagai tugas sekaligus, seperti pengenalan suku, deteksi ekspresi wajah, prediksi jenis kelamin dan umur, serta pencocokan wajah (face similarity).

---

## ✨ Fitur Utama
- 🔄 **Preprocessing Otomatis**: Resize, normalisasi, dan augmentasi gambar dilakukan secara otomatis.
- 👥 **Face Similarity**: Mencari wajah yang paling mirip dari dataset menggunakan embedding wajah.
- 🧑‍🤝‍🧑 **Deteksi Suku**: Mengklasifikasikan wajah ke dalam kategori suku seperti Jawa, Melayu, atau Sunda.
- 🙻 **Deteksi Gender & Umur**: Memprediksi jenis kelamin dan estimasi umur dari wajah.
- 😃 **Deteksi Ekspresi Wajah**: Mengenali ekspresi wajah seperti senang, serius, dan terkejut.

---

## ⚙️ Cara Setup Program

### 1. Clone Repository
```bash
git clone https://github.com/yanirahmawati234/suku-face-recognition.git
cd suku-face-recognition
```

### 2. Buat dan Aktifkan Virtual Environment (Opsional)
```bash
python -m venv venv
source venv/bin/activate        # Linux/MacOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🧠 Model

### 1. Download Model Terlatih
Unduh model terlatih dari link berikut:  
👉 [Download Model](https://drive.google.com/drive/folders/1Kdy6XbF7uR1jNUCf2VfvweixeyluJ-Mj?usp=sharing)

### 2. Simpan Model ke Folder `model/`
Letakkan file model hasil unduhan ke dalam folder `suku-face-recognition/model/`

---

## 👥 Anggota Kelompok
- Alya Gustiani Nur ‘Afifah — 231511035  
- Muhamad Ilham Fadillah S. — 231511053  
- Yani Rahmawati — 231511063

