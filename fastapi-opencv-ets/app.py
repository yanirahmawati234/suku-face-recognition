import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib

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
    filename_clean = filename.replace("cropped_", "")
    parts = filename_clean.split('_')
    if len(parts) >= 3:
        ekspresi = parts[0]
        sudut = parts[1]
        if sudut == "45":
            sudut = "miring45"
        pencahayaan = parts[2].split('.')[0]
    else:
        ekspresi = "unknown"
        sudut = "unknown"
        pencahayaan = "unknown"
    return ekspresi, sudut, pencahayaan

# Fungsi untuk memproses cropping
def process_cropping(image_path, suku, nama):
    image = cv2.imread(image_path)
    if image is None:
        return None, "Gagal membaca gambar."

    cropped_image, error = crop_face_with_mtcnn(image)
    if cropped_image is None:
        return None, error

    filename = os.path.basename(image_path)
    cropped_dir = os.path.join("dataset", "cropped", suku, nama)
    os.makedirs(cropped_dir, exist_ok=True)
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
def process_augmentation(image_path, suku, nama, augmentations):
    image = cv2.imread(image_path)
    if image is None:
        return None, "Gagal membaca gambar."

    filename = os.path.basename(image_path)
    base_name = filename.replace("cropped_", "")
    augmented_dir = os.path.join("dataset", "augmented", suku, nama)
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

# Fungsi untuk membangun model CNN
def create_model(num_classes, learning_rate=0.001, dropout_rate=0.5):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Streamlit bagian
st.title("JTK PCD 2024 - NAMA - NIM")
st.header("Proses Citra Digital dan Deteksi Suku/Etnis")

# Sidebar untuk navigasi
st.sidebar.title("Navigasi")
menu = st.sidebar.selectbox("Pilih Menu", ["Preprocessing", "Deteksi Suku/Etnis", "Face Similarity"])

# Menu 1: Preprocessing
if menu == "Preprocessing":
    st.subheader("Langkah 1: Cropping Gambar")
    st.write("Unggah gambar baru untuk dicrop, pilih folder tertentu untuk dicrop, atau proses semua gambar di folder original.")

    # Opsi 1: Unggah gambar baru untuk cropping
    st.subheader("Opsi 1: Unggah Gambar Baru")
    col1, col2 = st.columns(2)
    with col1:
        suku_crop = st.text_input("Masukkan Suku Subjek (Cropping)", value="Sunda")
    with col2:
        nama_crop = st.text_input("Masukkan Nama Subjek (Cropping)", value="Alfhi")

    uploaded_file = st.file_uploader("Pilih gambar untuk dicrop", type=["jpg", "jpeg", "png"], key="crop_uploader")

    if st.button("Proses Cropping Gambar (Unggah)"):
        if uploaded_file is not None and suku_crop and nama_crop:
            original_dir = os.path.join("dataset", "original", suku_crop, nama_crop)
            os.makedirs(original_dir, exist_ok=True)
            original_path = os.path.join(original_dir, uploaded_file.name)

            image = Image.open(uploaded_file)
            opencv_image = np.array(image)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(original_path, opencv_image)

            st.image(opencv_image, channels="BGR", caption="Gambar Asli", use_column_width=True)

            cropped_image, error = process_cropping(original_path, suku_crop, nama_crop)
            if cropped_image is not None:
                st.image(cropped_image, channels="BGR", caption="Gambar Setelah Crop", use_column_width=True)
                st.success(f"Gambar telah dicrop dan disimpan di: dataset/cropped/{suku_crop}/{nama_crop}/cropped_{uploaded_file.name}")
            else:
                st.error(error)
        else:
            st.error("Harap masukkan suku, nama, dan unggah gambar!")

    # Opsi 2: Pilih folder tertentu untuk cropping
    st.subheader("Opsi 2: Pilih Folder untuk Cropping")
    original_base_dir = os.path.join("dataset", "original")
    if not os.path.exists(original_base_dir):
        st.error("Folder dataset/original tidak ditemukan!")
    else:
        suku_options = [d for d in os.listdir(original_base_dir) if os.path.isdir(os.path.join(original_base_dir, d))]
        if not suku_options:
            st.error("Tidak ada folder suku di dataset/original!")
        else:
            suku_crop_select = st.selectbox("Pilih Suku (Cropping)", suku_options)
            nama_dir = os.path.join(original_base_dir, suku_crop_select)
            nama_options = [d for d in os.listdir(nama_dir) if os.path.isdir(os.path.join(nama_dir, d))]
            if not nama_options:
                st.error(f"Tidak ada folder nama di dataset/original/{suku_crop_select}!")
            else:
                nama_crop_select = st.selectbox("Pilih Nama (Cropping)", nama_options)

                if st.button("Proses Cropping Folder Terpilih"):
                    cropped_dir = os.path.join(original_base_dir, suku_crop_select, nama_crop_select)
                    image_files = [f for f in os.listdir(cropped_dir) if f.endswith(('.jpg', '.jpeg', '.png')) and not any(suffix in f for suffix in ['rot15', 'rot-15', 'flip', 'noise', 'bright'])]

                    for image_file in image_files:
                        image_path = os.path.join(cropped_dir, image_file)
                        cropped_image, error = process_cropping(image_path, suku_crop_select, nama_crop_select)
                        if cropped_image is not None:
                            st.write(f"Berhasil memproses cropping: {image_path}")
                        else:
                            st.write(f"Gagal memproses cropping {image_path}: {error}")

                    st.success(f"Selesai memproses cropping untuk folder dataset/original/{suku_crop_select}/{nama_crop_select}!")

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

                suku, nama = path_parts
                image_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png')) and not any(suffix in f for suffix in ['rot15', 'rot-15', 'flip', 'noise', 'bright'])]

                for image_file in image_files:
                    image_path = os.path.join(root, image_file)
                    cropped_image, error = process_cropping(image_path, suku, nama)
                    if cropped_image is not None:
                        st.write(f"Berhasil memproses cropping: {image_path}")
                    else:
                        st.write(f"Gagal memproses cropping {image_path}: {error}")

            st.success("Selesai memproses cropping semua gambar di folder original!")

    # Bagian 2: Augmentasi
    st.subheader("Langkah 2: Augmentasi Gambar dari Folder Cropped")
    st.write("Pilih folder dari cropped untuk di-augmentasi, lalu pilih jenis augmentasi.")

    cropped_base_dir = os.path.join("dataset", "cropped")
    if not os.path.exists(cropped_base_dir):
        st.error("Folder dataset/cropped tidak ditemukan! Silakan lakukan cropping terlebih dahulu.")
    else:
        suku_options = [d for d in os.listdir(cropped_base_dir) if os.path.isdir(os.path.join(cropped_base_dir, d))]
        if not suku_options:
            st.error("Tidak ada folder suku di dataset/cropped!")
        else:
            suku_aug = st.selectbox("Pilih Suku (Augmentasi)", suku_options)
            nama_dir = os.path.join(cropped_base_dir, suku_aug)
            nama_options = [d for d in os.listdir(nama_dir) if os.path.isdir(os.path.join(nama_dir, d))]
            if not nama_options:
                st.error(f"Tidak ada folder nama di dataset/cropped/{suku_aug}!")
            else:
                nama_aug = st.selectbox("Pilih Nama (Augmentasi)", nama_options)

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

                if st.button("Proses Augmentasi"):
                    if not augmentations:
                        st.error("Harap pilih setidaknya satu jenis augmentasi!")
                    else:
                        cropped_dir = os.path.join(cropped_base_dir, suku_aug, nama_aug)
                        image_files = [f for f in os.listdir(cropped_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

                        for image_file in image_files:
                            image_path = os.path.join(cropped_dir, image_file)
                            aug_image, error = process_augmentation(image_path, suku_aug, nama_aug, augmentations)
                            if aug_image is not None:
                                st.write(f"Berhasil memproses augmentasi: {image_path}")
                            else:
                                st.write(f"Gagal memproses augmentasi {image_path}: {error}")

                        st.success(f"Selesai memproses augmentasi untuk folder dataset/cropped/{suku_aug}/{nama_aug}!")

# Menu 2: Deteksi Suku/Etnis
elif menu == "Deteksi Suku/Etnis":
    st.subheader("Tahap 3: Deteksi Suku/Etnis")

    # Load metadata
    csv_file = os.path.join("dataset", "metadata.csv")
    if not os.path.exists(csv_file):
        st.error("File metadata.csv tidak ditemukan! Silakan lakukan preprocessing terlebih dahulu.")
    else:
        df = pd.read_csv(csv_file)
        df = df[df['path_gambar'].str.contains("augmented")]
        if df.empty:
            st.error("Tidak ada data di folder augmented! Silakan lakukan augmentasi terlebih dahulu.")
        else:
            # Persiapkan data
            image_paths = df['path_gambar'].values
            suku_labels = df['suku'].values

            # Encode label suku
            label_encoder = LabelEncoder()
            suku_encoded = label_encoder.fit_transform(suku_labels)
            joblib.dump(label_encoder, "label_encoder.pkl")  # Simpan label encoder untuk prediksi

            # Load dan preprocess gambar
            images = []
            valid_labels = []
            for img_path, label in zip(image_paths, suku_encoded):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (150, 150))
                    img = img / 255.0
                    images.append(img)
                    valid_labels.append(label)
                else:
                    st.warning(f"Gagal memuat gambar: {img_path}")

            images = np.array(images)
            suku_encoded = np.array(valid_labels)

            if len(images) == 0:
                st.error("Tidak ada gambar yang dapat diproses!")
            else:
                st.write(f"Jumlah gambar yang berhasil dimuat: {len(images)}")
                st.write(f"Jumlah label: {len(suku_encoded)}")

                # Parameter untuk pelatihan
                learning_rates = [0.001, 0.01, 0.1]
                dropout_rates = [0.3, 0.5, 0.7]
                batch_sizes = [16, 32]
                epochs_list = [5, 10]

                # K-Fold Cross Validation
                kfold = KFold(n_splits=5, shuffle=True, random_state=42)

                # Opsi 1: Latih model
                if st.button("Latih Model dengan K-Fold Cross Validation"):
                    best_accuracy = 0
                    best_model = None
                    best_params = {}
                    results = []

                    for lr in learning_rates:
                        for dr in dropout_rates:
                            for batch_size in batch_sizes:
                                for epochs in epochs_list:
                                    st.write(f"Melatih dengan learning_rate={lr}, dropout_rate={dr}, batch_size={batch_size}, epochs={epochs}")
                                    fold_accuracies = []

                                    for fold, (train_idx, val_idx) in enumerate(kfold.split(images)):
                                        X_train, X_val = images[train_idx], images[val_idx]
                                        y_train, y_val = suku_encoded[train_idx], suku_encoded[val_idx]

                                        model = create_model(num_classes=len(np.unique(suku_labels)), learning_rate=lr, dropout_rate=dr)
                                        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(X_val, y_val))

                                        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
                                        fold_accuracies.append(val_accuracy)
                                        st.write(f"Fold {fold+1}: Akurasi = {val_accuracy:.4f}")

                                    mean_accuracy = np.mean(fold_accuracies)
                                    std_accuracy = np.std(fold_accuracies)
                                    st.write(f"Rata-rata akurasi: {mean_accuracy:.4f} (±{std_accuracy:.4f})")

                                    results.append({
                                        "learning_rate": lr,
                                        "dropout_rate": dr,
                                        "batch_size": batch_size,
                                        "epochs": epochs,
                                        "mean_accuracy": mean_accuracy,
                                        "std_accuracy": std_accuracy
                                    })

                                    if mean_accuracy > best_accuracy:
                                        best_accuracy = mean_accuracy
                                        best_model = model
                                        best_params = {"learning_rate": lr, "dropout_rate": dr, "batch_size": batch_size, "epochs": epochs}

                    # Simpan model terbaik
                    if best_model is not None:
                        best_model.save("best_model.h5")
                        st.success("Model terbaik disimpan sebagai best_model.h5")
                        st.write("Parameter terbaik:")
                        st.write(best_params)
                        st.write(f"Akurasi terbaik: {best_accuracy:.4f}")

                    # Tampilkan hasil dalam tabel
                    results_df = pd.DataFrame(results)
                    st.write("Detail Hasil Pelatihan:")
                    st.write(results_df)

                    # Plot hasil
                    plt.figure(figsize=(10, 6))
                    plt.plot(results_df['mean_accuracy'], label='Mean Accuracy')
                    plt.fill_between(range(len(results_df)),
                                     results_df['mean_accuracy'] - results_df['std_accuracy'],
                                     results_df['mean_accuracy'] + results_df['std_accuracy'],
                                     alpha=0.2)
                    plt.xlabel('Parameter Kombinasi')
                    plt.ylabel('Akurasi')
                    plt.title('Hasil Pelatihan dengan K-Fold Cross Validation')
                    plt.legend()
                    plt.savefig('training_results.png')
                    st.image('training_results.png', caption="Grafik Hasil Pelatihan", use_column_width=True)

                # Opsi 2: Prediksi suku/etnis dari gambar baru
                st.subheader("Prediksi Suku/Etnis dari Gambar Baru")
                uploaded_image = st.file_uploader("Unggah gambar untuk diprediksi", type=["jpg", "jpeg", "png"], key="predict_uploader")

                if uploaded_image is not None:
                    # Tampilkan gambar yang diunggah
                    image = Image.open(uploaded_image)
                    opencv_image = np.array(image)
                    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
                    st.image(opencv_image, channels="BGR", caption="Gambar Asli", use_column_width=True)

                    # Crop wajah
                    cropped_image, error = crop_face_with_mtcnn(opencv_image)
                    if cropped_image is None:
                        st.error(error)
                    else:
                        st.image(cropped_image, channels="BGR", caption="Gambar Setelah Crop", use_column_width=True)

                        # Preprocess gambar untuk prediksi
                        img = cv2.resize(cropped_image, (150, 150))
                        img = img / 255.0
                        img = np.expand_dims(img, axis=0)  # Tambahkan dimensi batch

                        # Muat model dan label encoder
                        try:
                            model = tf.keras.models.load_model("best_model.h5")
                            label_encoder = joblib.load("label_encoder.pkl")
                        except:
                            st.error("Model atau label encoder belum tersedia. Silakan latih model terlebih dahulu!")
                        else:
                            # Prediksi
                            prediction = model.predict(img)
                            predicted_class = np.argmax(prediction, axis=1)[0]
                            predicted_suku = label_encoder.inverse_transform([predicted_class])[0]

                            # Tampilkan hasil prediksi
                            st.success(f"Gambar ini diprediksi sebagai suku: **{predicted_suku}**")
                            st.write("Probabilitas untuk setiap suku:")
                            suku_classes = label_encoder.classes_
                            for suku, prob in zip(suku_classes, prediction[0]):
                                st.write(f"{suku}: {prob:.4f}")

# Menu 3: Face Similarity
elif menu == "Face Similarity":
    st.subheader("Face Similarity")
    st.write("Fitur ini belum diimplementasikan.")