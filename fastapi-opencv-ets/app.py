import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
from mtcnn import MTCNN
import tensorflow as tf

# Fungsi untuk deteksi wajah dan pemotongan gambar menggunakan MTCNN
def crop_face_with_mtcnn(image):
    detector = MTCNN()
    faces = detector.detect_faces(image)

    if len(faces) == 0:
        return None, "Tidak ada wajah terdeteksi."
    
    (x, y, w, h) = faces[0]['box']
    # Pastikan koordinat tidak negatif dan berada dalam batas gambar
    x, y = max(0, x), max(0, y)
    w, h = min(w, image.shape[1] - x), min(h, image.shape[0] - y)
    face_image = image[y:y+h, x:x+w]
    return face_image, None

# Fungsi augmentasi
def rotate_image(image, angle=15):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def flip_image(image):
    return cv2.flip(image, 1)

def add_gaussian_noise(image):
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def adjust_brightness_contrast(image, alpha=1.5, beta=50):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Fungsi untuk ekstraksi metadata dari nama file
def extract_metadata_from_filename(filename):
    # Remove prefix like 'cropped_' if present
    filename_clean = filename.replace("cropped_", "")
    parts = filename_clean.split('_')
    if len(parts) >= 3:
        ekspresi = parts[0]  # senyum atau serius
        sudut = parts[1]     # 45 atau profil
        if sudut == "45":
            sudut = "miring45"
        pencahayaan = parts[2].split('.')[0]  # indoor atau outdoor
    else:
        ekspresi = "unknown"
        sudut = "unknown"
        pencahayaan = "unknown"
    return ekspresi, sudut, pencahayaan

# Fungsi untuk memproses cropping
def process_cropping(image_path, nama, suku):
    image = cv2.imread(image_path)
    if image is None:
        return None, "Gagal membaca gambar."

    cropped_image, error = crop_face_with_mtcnn(image)
    if cropped_image is None:
        return None, error

    filename = os.path.basename(image_path)
    cropped_dir = os.path.join("dataset", "cropped", nama, suku)
    os.makedirs(cropped_dir, exist_ok=True)
    # Tambahkan awalan 'cropped_' pada nama file
    cropped_filename = f"cropped_{filename}"
    cropped_path = os.path.join(cropped_dir, cropped_filename)
    cv2.imwrite(cropped_path, cropped_image)

    ekspresi, sudut, pencahayaan = extract_metadata_from_filename(filename)
    csv_data = [{
        "path_gambar": cropped_path,
        "nama": nama,
        "suku": suku,
        "ekspresi": ekspresi,
        "sudut": sudut,
        "pencahayaan": pencahayaan
    }]

    csv_file = os.path.join("dataset", "metadata.csv")
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

    return cropped_image, None

# Fungsi untuk memproses augmentasi
def process_augmentation(image_path, nama, suku, augmentations):
    image = cv2.imread(image_path)
    if image is None:
        return None, "Gagal membaca gambar."

    filename = os.path.basename(image_path)
    # Remove 'cropped_' prefix for base name
    base_name = filename.replace("cropped_", "")
    augmented_dir = os.path.join("dataset", "augmented", nama, suku)
    os.makedirs(augmented_dir, exist_ok=True)

    ekspresi, sudut, pencahayaan = extract_metadata_from_filename(filename)
    csv_data = []
    for aug in augmentations:
        if aug == "rotate_15":
            aug_image = rotate_image(image, 15)
            aug_path = os.path.join(augmented_dir, f"rotate_{base_name}")
            cv2.imwrite(aug_path, aug_image)
            csv_data.append({
                "path_gambar": aug_path,
                "nama": nama,
                "suku": suku,
                "ekspresi": ekspresi,
                "sudut": "rotasi15",
                "pencahayaan": pencahayaan
            })
        elif aug == "rotate_minus_15":
            aug_image = rotate_image(image, -15)
            aug_path = os.path.join(augmented_dir, f"rotate_{base_name}")
            cv2.imwrite(aug_path, aug_image)
            csv_data.append({
                "path_gambar": aug_path,
                "nama": nama,
                "suku": suku,
                "ekspresi": ekspresi,
                "sudut": "rotasi-15",
                "pencahayaan": pencahayaan
            })
        elif aug == "flip":
            aug_image = flip_image(image)
            aug_path = os.path.join(augmented_dir, f"flip_{base_name}")
            cv2.imwrite(aug_path, aug_image)
            csv_data.append({
                "path_gambar": aug_path,
                "nama": nama,
                "suku": suku,
                "ekspresi": ekspresi,
                "sudut": "flip",
                "pencahayaan": pencahayaan
            })
        elif aug == "noise":
            aug_image = add_gaussian_noise(image)
            aug_path = os.path.join(augmented_dir, f"gaussiannoisy_{base_name}")
            cv2.imwrite(aug_path, aug_image)
            csv_data.append({
                "path_gambar": aug_path,
                "nama": nama,
                "suku": suku,
                "ekspresi": ekspresi,
                "sudut": "noise",
                "pencahayaan": pencahayaan
            })
        elif aug == "brightness":
            aug_image = adjust_brightness_contrast(image)
            aug_path = os.path.join(augmented_dir, f"brightness&contrast_{base_name}")
            cv2.imwrite(aug_path, aug_image)
            csv_data.append({
                "path_gambar": aug_path,
                "nama": nama,
                "suku": suku,
                "ekspresi": ekspresi,
                "sudut": "brightness",
                "pencahayaan": "adjusted"
            })

    if csv_data:
        csv_file = os.path.join("dataset", "metadata.csv")
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

    return image, None

# Streamlit bagian
st.title("JTK PCD 2024 - NAMA - NIM")
st.header("Proses Citra Digital: Cropping dan Augmentasi")

# Sidebar untuk navigasi
st.sidebar.title("Navigasi")
st.sidebar.markdown("[Home](#)")
st.sidebar.markdown("[Grayscale](#)")
st.sidebar.markdown("[Histogram](#)")
st.sidebar.markdown("[Histogram Equalization](#)")
st.sidebar.markdown("[Histogram Specification](#)")

# Bagian 1: Cropping
st.subheader("Langkah 1: Cropping Gambar")
st.write("Unggah gambar baru untuk dicrop, pilih folder tertentu untuk dicrop, atau proses semua gambar di folder original.")

# Opsi 1: Unggah gambar baru untuk cropping
st.subheader("Opsi 1: Unggah Gambar Baru")
col1, col2 = st.columns(2)
with col1:
    nama_crop = st.text_input("Masukkan Nama Subjek (Cropping)", value="Alfhi")
with col2:
    suku_crop = st.text_input("Masukkan Suku Subjek (Cropping)", value="Sunda")

uploaded_file = st.file_uploader("Pilih gambar untuk dicrop", type=["jpg", "jpeg", "png"])

if st.button("Proses Cropping Gambar (Unggah)"):
    if uploaded_file is not None and nama_crop and suku_crop:
        original_dir = os.path.join("dataset", "original", nama_crop, suku_crop)
        os.makedirs(original_dir, exist_ok=True)
        original_path = os.path.join(original_dir, uploaded_file.name)

        image = Image.open(uploaded_file)
        opencv_image = np.array(image)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(original_path, opencv_image)

        st.image(opencv_image, channels="BGR", caption="Gambar Asli", use_column_width=True)

        cropped_image, error = process_cropping(original_path, nama_crop, suku_crop)
        if cropped_image is not None:
            st.image(cropped_image, channels="BGR", caption="Gambar Setelah Crop", use_column_width=True)
            st.success(f"Gambar telah dicrop dan disimpan di: dataset/cropped/{nama_crop}/{suku_crop}/cropped_{uploaded_file.name}")
        else:
            st.error(error)
    else:
        st.error("Harap masukkan nama, suku, dan unggah gambar!")

# Opsi 2: Pilih folder tertentu untuk cropping
st.subheader("Opsi 2: Pilih Folder untuk Cropping")
original_base_dir = os.path.join("dataset", "original")
if not os.path.exists(original_base_dir):
    st.error("Folder dataset/original tidak ditemukan!")
else:
    nama_options = [d for d in os.listdir(original_base_dir) if os.path.isdir(os.path.join(original_base_dir, d))]
    if not nama_options:
        st.error("Tidak ada folder nama di dataset/original!")
    else:
        nama_crop_select = st.selectbox("Pilih Nama (Cropping)", nama_options)
        suku_dir = os.path.join(original_base_dir, nama_crop_select)
        suku_options = [d for d in os.listdir(suku_dir) if os.path.isdir(os.path.join(suku_dir, d))]
        if not suku_options:
            st.error(f"Tidak ada folder suku di dataset/original/{nama_crop_select}!")
        else:
            suku_crop_select = st.selectbox("Pilih Suku (Cropping)", suku_options)

            if st.button("Proses Cropping Folder Terpilih"):
                cropped_dir = os.path.join(original_base_dir, nama_crop_select, suku_crop_select)
                image_files = [f for f in os.listdir(cropped_dir) if f.endswith(('.jpg', '.jpeg', '.png')) and not any(suffix in f for suffix in ['rot15', 'rot-15', 'flip', 'noise', 'bright'])]

                for image_file in image_files:
                    image_path = os.path.join(cropped_dir, image_file)
                    cropped_image, error = process_cropping(image_path, nama_crop_select, suku_crop_select)
                    if cropped_image is not None:
                        st.write(f"Berhasil memproses cropping: {image_path}")
                    else:
                        st.write(f"Gagal memproses cropping {image_path}: {error}")

                st.success(f"Selesai memproses cropping untuk folder dataset/original/{nama_crop_select}/{suku_crop_select}!")

# Opsi 3: Proses semua gambar di folder original
st.subheader("Opsi 3: Proses Semua Gambar")
if st.button("Proses Cropping Semua Gambar di Folder Original"):
    if not os.path.exists(original_base_dir):
        st.error("Folder dataset/original tidak ditemukan!")
    else:
        for root, dirs, files in os.walk(original_base_dir):
            relative_path = os.path.relpath(root, original_base_dir)
            path_parts = relative_path.split(os.sep)
            if len(path_parts) != 2:
                continue

            nama, suku = path_parts
            image_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png')) and not any(suffix in f for suffix in ['rot15', 'rot-15', 'flip', 'noise', 'bright'])]

            for image_file in image_files:
                image_path = os.path.join(root, image_file)
                cropped_image, error = process_cropping(image_path, nama, suku)
                if cropped_image is not None:
                    st.write(f"Berhasil memproses cropping: {image_path}")
                else:
                    st.write(f"Gagal memproses cropping {image_path}: {error}")

        st.success("Selesai memproses cropping semua gambar di folder original!")

# Bagian 2: Augmentasi
st.subheader("Langkah 2: Augmentasi Gambar dari Folder Cropped")
st.write("Pilih folder dari cropped untuk di-augmentasi, lalu pilih jenis augmentasi.")

# Pilih folder dari cropped
cropped_base_dir = os.path.join("dataset", "cropped")
if not os.path.exists(cropped_base_dir):
    st.error("Folder dataset/cropped tidak ditemukan! Silakan lakukan cropping terlebih dahulu.")
else:
    nama_options = [d for d in os.listdir(cropped_base_dir) if os.path.isdir(os.path.join(cropped_base_dir, d))]
    if not nama_options:
        st.error("Tidak ada folder nama di dataset/cropped!")
    else:
        nama_aug = st.selectbox("Pilih Nama (Augmentasi)", nama_options)
        suku_dir = os.path.join(cropped_base_dir, nama_aug)
        suku_options = [d for d in os.listdir(suku_dir) if os.path.isdir(os.path.join(suku_dir, d))]
        if not suku_options:
            st.error(f"Tidak ada folder suku di dataset/cropped/{nama_aug}!")
        else:
            suku_aug = st.selectbox("Pilih Suku (Augmentasi)", suku_options)

            # Checkbox untuk memilih augmentasi
            st.subheader("Pilih Jenis Augmentasi")
            augmentations = []
            if st.checkbox("Rotasi +15 Derajat"):
                augmentations.append("rotate_15")
            if st.checkbox("Rotasi -15 Derajat"):
                augmentations.append("rotate_minus_15")
            if st.checkbox("Horizontal Flip"):
                augmentations.append("flip")
            if st.checkbox("Gaussian Noise"):
                augmentations.append("noise")
            if st.checkbox("Brightness Adjustment"):
                augmentations.append("brightness")

            # Tombol untuk memproses augmentasi
            if st.button("Proses Augmentasi"):
                if not augmentations:
                    st.error("Harap pilih setidaknya satu jenis augmentasi!")
                else:
                    cropped_dir = os.path.join(cropped_base_dir, nama_aug, suku_aug)
                    image_files = [f for f in os.listdir(cropped_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

                    for image_file in image_files:
                        image_path = os.path.join(cropped_dir, image_file)
                        aug_image, error = process_augmentation(image_path, nama_aug, suku_aug, augmentations)
                        if aug_image is not None:
                            st.write(f"Berhasil memproses augmentasi: {image_path}")
                        else:
                            st.write(f"Gagal memproses augmentasi {image_path}: {error}")

                    st.success(f"Selesai memproses augmentasi untuk folder dataset/cropped/{nama_aug}/{suku_aug}!")