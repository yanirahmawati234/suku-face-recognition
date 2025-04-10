import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
from mtcnn import MTCNN

# Fungsi untuk deteksi wajah dan pemotongan gambar

def crop_face(image):
    detector = MTCNN()
    faces = detector.detect_faces(image)

    if len(faces) == 0:
        return None, "Tidak ada wajah terdeteksi."
    
    (x, y, w, h) = faces[0]['box']
    face_image = image[y:y+h, x:x+w]
    return face_image, None

def rotate_image(image, angle=15):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def flip_image(image):
    flipped_image = cv2.flip(image, 1)
    return flipped_image

def adjust_brightness_contrast(image, alpha, beta):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def add_gaussian_noise(image):
    noise = np.random.normal(0, 5, image.shape).astype(np.uint8)  # Noise level rendah
    noisy_image = cv2.add(image, noise)
    return noisy_image

def process_image(image, filename, suku, nama, rotate_check, angle, flip_check, brightness_check, alpha, beta, noise_check):
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Memproses gambar dan memotong wajah
    face_image, error = crop_face(opencv_image)
    
    if face_image is None:
        return False, error
    
    # Membuat direktori untuk cropped image
    cropped_dir = os.path.join("dataset", "cropped", suku, nama)
    os.makedirs(cropped_dir, exist_ok=True)
    
    # Menyimpan gambar yang dicrop
    original_filename = os.path.splitext(filename)[0]
    cropped_filename = f"cropped_{original_filename}.jpg"
    cropped_path = os.path.join(cropped_dir, cropped_filename)
    cv2.imwrite(cropped_path, face_image)
    
    # Proses augmentasi jika ada yang dipilih
    augmented_images = {}
    
    if rotate_check:
        rotated_img = rotate_image(face_image, angle=angle)
        augmented_images[f"rotated_{angle}_{original_filename}.jpg"] = rotated_img
    
    if flip_check:
        flipped_img = flip_image(face_image)
        augmented_images[f"flipped_{original_filename}.jpg"] = flipped_img
    
    if brightness_check:
        adjusted_img = adjust_brightness_contrast(face_image, alpha=alpha, beta=beta)
        augmented_images[f"brightness_{alpha}_{beta}_{original_filename}.jpg"] = adjusted_img
    
    if noise_check:
        noisy_img = add_gaussian_noise(face_image)
        augmented_images[f"noisy_{original_filename}.jpg"] = noisy_img
    
    # Menyimpan gambar augmented
    if augmented_images:
        augmented_dir = os.path.join("dataset", "augmented", suku, nama)
        os.makedirs(augmented_dir, exist_ok=True)
        for filename, img in augmented_images.items():
            augmented_path = os.path.join(augmented_dir, filename)
            cv2.imwrite(augmented_path, img)
    
    return True, None

# Streamlit UI
st.title('Deteksi Wajah dan Pemotongan Gambar')

# Input untuk suku dan nama
suku = st.text_input("Masukkan nama suku (contoh: Jawa, Sunda, Batak, dll):", "")
nama = st.text_input("Masukkan nama subjek:", "")

# Pilih mode input (file atau folder)
input_mode = st.radio("Pilih mode input:", ("File Tunggal", "Folder"))

uploaded_files = []
if input_mode == "File Tunggal":
    uploaded_file = st.file_uploader("Pilih gambar untuk diproses", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        uploaded_files = [(uploaded_file.name, Image.open(uploaded_file))]
else:
    uploaded_folder = st.file_uploader("Pilih folder (unggah semua gambar sekaligus)", 
                                     type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_folder:
        uploaded_files = [(f.name, Image.open(f)) for f in uploaded_folder]

# Checkbox untuk augmentasi
st.subheader("Pilihan Augmentasi")
col1, col2 = st.columns(2)
with col1:
    rotate_check = st.checkbox("Rotasi")
    if rotate_check:
        angle = st.slider("Pilih sudut rotasi (derajat)", -180, 180, 15)
    flip_check = st.checkbox("Flip Horizontal")

with col2:
    brightness_check = st.checkbox("Brightness & Contrast")
    if brightness_check:
        alpha = st.slider("Tingkat kontras (alpha)", 0.5, 3.0, 1.0)
        beta = st.slider("Tingkat kecerahan (beta)", -50, 50, 0)
    noise_check = st.checkbox("Gaussian Noise (Sedikit)")

if st.button("Proses") and suku and nama and uploaded_files:
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)
    success_count = 0
    
    for i, (filename, image) in enumerate(uploaded_files):
        st.write(f"Memproses: {filename}")
        
        # Menampilkan gambar asli
        st.image(image, caption=f"Original: {filename}", use_column_width=True)
        
        # Proses gambar
        success, error = process_image(
            image, filename, suku, nama, 
            rotate_check, angle, flip_check, 
            brightness_check, alpha, beta, noise_check
        )
        
        if success:
            success_count += 1
            st.success(f"Berhasil memproses {filename}")
        else:
            st.error(f"Gagal memproses {filename}: {error}")
        
        progress_bar.progress((i + 1) / total_files)
    
    st.success(f"Proses selesai! {success_count}/{total_files} gambar berhasil diproses")
elif (not suku or not nama) and uploaded_files:
    st.warning("Harap masukkan nama suku dan nama subjek terlebih dahulu")