import streamlit as st
import cv2
import numpy as np
import pandas as pd
import scripts.suku.suku_identification as backend
import scripts.ekspresi.ekspresi_identification as ekspresi 
import os
import tempfile
import matplotlib.pyplot as plt
import shutil

from PIL import Image
from scripts.preprocessing import crop_faces, preprocess_images, split_dataset
from scripts.facesimilarity.facesimilarity import extract_embedding, calculate_similarity, visualize_comparison
from scripts.gender.deteksi_gender_umur import analyze_deepface
from scripts.facesimilarity.facesimilarity_evaluation import (
    load_embeddings_from_dataset,
    plot_tsne,
    generate_pairs,
    evaluate_roc,
    plot_similarity_distribution,
    save_example_visuals
)

st.set_page_config(page_title="Face Processing App", layout="centered")
st.title("👤 Face Recognition App")

option = st.sidebar.radio("Pilih Mode", ["Preprocessing", "Face Similarity", "Deteksi Umur & Gender", "Deteksi Suku", "Deteksi Ekspresi"])

def save_uploaded_folder(uploaded_files, extract_dir):
    for uploaded_file in uploaded_files:
        file_path = os.path.join(extract_dir, uploaded_file.name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
    return extract_dir

if option == "Preprocessing":
    st.header("🛠 Preprocessing Wajah")
    task = st.selectbox("Pilih tindakan", ["Face Detection & Crop", "Augmentasi", "Split Dataset"])
    
    folder_path = st.text_input("Masukkan path folder")

    if folder_path:
        if task == "Face Detection & Crop":
            if st.button("Mulai Deteksi dan Crop"):
                crop_faces(folder_path)  # Adjusted to work with new output folder structure
                st.success("Deteksi wajah dan crop selesai.")

        elif task == "Augmentasi":
            if st.button("Mulai Augmentasi"):
                preprocess_images(folder_path)  # Adjusted to work with new output folder structure
                st.success("Proses augmentasi selesai.")

        elif task == "Split Dataset":
            if st.button("Mulai Split Dataset"):
                split_dataset(folder_path)  # Adjusted to work with new output folder structure
                st.success("Split dataset selesai.")


elif option == "Face Similarity":
    st.header("📊 Evaluasi Teknik Face Similarity")
    dataset_path = st.text_input("Masukkan path folder dataset")

    if dataset_path and st.button("Uji Teknik"):
        st.write("📥 Memuat data...")
        embeddings, labels, image_paths = load_embeddings_from_dataset(dataset_path)

        st.write("📈 Menjalankan t-SNE plot...")
        tsne_path = "tsne_plot.png"
        plt.figure()
        plot_tsne(embeddings, labels)
        plt.savefig(tsne_path)
        st.image(tsne_path, caption="t-SNE Visualization")

        st.write("📊 Evaluasi ROC dan metrik lainnya...")
        pairs, index_pairs = generate_pairs(embeddings, labels, image_paths)

        import io, sys
        buffer = io.StringIO()
        sys.stdout = buffer
        threshold = evaluate_roc(pairs)
        st.session_state["face_eval_done"] = True
        st.session_state["threshold"] = threshold
        sys.stdout = sys.__stdout__
        roc_output = buffer.getvalue()

        roc_path = "roc_curve.png"
        plt.savefig(roc_path)
        st.image(roc_path, caption="ROC Curve")

        st.text("\n" + roc_output)

        st.write("📊 Visualisasi distribusi skor similarity...")
        dist_path = "similarity_distribution.png"
        plt.figure()
        plot_similarity_distribution(pairs)
        plt.savefig(dist_path)
        st.image(dist_path, caption="Distribusi Skor Similarity")

        st.write("🖼️ Menyimpan contoh visualisasi TP, FP, FN, TN...")
        pairs_with_paths = [((emb1, emb2, label), (image_paths[i1], image_paths[i2]))
                            for (emb1, emb2, label), (i1, i2) in zip(pairs, index_pairs)]
        save_example_visuals(pairs_with_paths, threshold)

        for label in ["tp", "fp", "fn", "tn"]:
            image_path = os.path.join("examples", f"{label}.png")
            if os.path.exists(image_path):
                st.image(image_path, caption=label.upper())

        st.success("Evaluasi selesai dan visualisasi disimpan!")

if st.session_state.get("face_eval_done", False):
    st.header("🔍 Face Similarity Checker")
    img1 = st.file_uploader("Upload Gambar 1", type=["jpg", "jpeg", "png"], key="img1")
    img2 = st.file_uploader("Upload Gambar 2", type=["jpg", "jpeg", "png"], key="img2")

    if img1 and img2 and st.button("Cek Kesamaan"):
        temp1 = os.path.join(tempfile.gettempdir(), "img1.jpg")
        temp2 = os.path.join(tempfile.gettempdir(), "img2.jpg")

        with open(temp1, "wb") as f:
            f.write(img1.read())
        with open(temp2, "wb") as f:
            f.write(img2.read())

        emb1 = extract_embedding(temp1)
        emb2 = extract_embedding(temp2)

        if emb1 is not None and emb2 is not None:
            score = calculate_similarity(emb1, emb2)
            threshold = st.session_state["threshold"]
            match = "MATCH" if score >= threshold else "NOT MATCH"
            st.write(f"**Similarity Score:** {score:.4f}")
            st.write(f"**Status:** {match}")

            col1, col2 = st.columns(2)
            with col1:
                st.image(temp1, caption="Gambar 1", use_column_width=True)
            with col2:
                st.image(temp2, caption="Gambar 2", use_column_width=True)

            fig = plt.figure()
            visualize_comparison(temp1, temp2, score)
            st.pyplot(fig)
        else:
            st.error("Gagal mendeteksi wajah pada salah satu gambar.")

elif option == "Deteksi Umur & Gender":
    st.header("🧓 Deteksi Umur & Gender (DeepFace)")
    uploaded_img = st.file_uploader("Upload gambar wajah", type=["jpg", "jpeg", "png"], key="age_gender")

    if uploaded_img is not None:
        temp_img_path = os.path.join(tempfile.gettempdir(), uploaded_img.name)
        with open(temp_img_path, "wb") as f:
            f.write(uploaded_img.read())

        with st.spinner("Mendeteksi umur dan gender..."):
            result = analyze_deepface(temp_img_path)

        if result['time_sec'] != "ERROR":
            st.image(temp_img_path, caption="Gambar Wajah", use_column_width=True)
            st.markdown("### 🔍 Hasil Deteksi:")
            st.write(f"**Umur:** {result['age']} tahun")
            st.write(f"**Gender:** {result['gender']} (Confidence: {result['confidence']}%)")
            st.write(f"**Waktu Proses:** {result['time_sec']} detik")
        else:
            st.error("Gagal mendeteksi wajah. Silakan coba gambar lain.")

elif option == "Deteksi Suku":
    st.title("Deteksi Suku/Etnis")
    choice = st.radio("Pilih mode deteksi", ('Webcam', 'Upload Gambar'))
    labels = backend.get_labels()

    if choice == 'Upload Gambar':
        uploaded_file = st.file_uploader("Unggah gambar untuk prediksi", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang diunggah", use_column_width=True)

            image_cv = np.array(image.convert("RGB"))
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

            preprocessed_face, cropped_face = backend.preprocess_face(image_cv)

            if preprocessed_face is not None:
                st.subheader("Wajah yang Diproses")
                st.image(cropped_face, caption="Wajah yang Di-crop", use_column_width=True)

                preds = backend.ensemble_predict(preprocessed_face)
                class_idx = np.argmax(preds)
                confidence = preds[class_idx]

                st.success(f"Prediksi: {labels[class_idx]} ({confidence:.2f})")
                st.subheader("Distribusi Confidence Score:")

                df = pd.DataFrame({
                    'Etnis': labels,
                    'Confidence': preds
                })
                st.bar_chart(df.set_index('Etnis'))
            else:
                st.error("Tidak terdeteksi wajah pada gambar. Pastikan wajah terlihat jelas.")

    elif choice == 'Webcam':
        stframe = st.empty()
        run = st.checkbox('Nyalakan Kamera')

        if run:
            cap = cv2.VideoCapture(0)

            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Gagal mengambil frame dari webcam.")
                    break

                faces = backend.detect_faces(frame)
                preds_to_display = np.zeros(len(labels))

                if faces:
                    for face in faces:
                        x, y, w, h = face['box']
                        x, y = max(0, x), max(0, y)
                        face_crop = frame[y:y+h, x:x+w]

                        face_input = backend.preprocess_multiple_faces(face_crop)
                        preds = backend.ensemble_predict(face_input)
                        preds_to_display = preds

                        class_idx = np.argmax(preds)
                        confidence = preds[class_idx]

                        label_text = f"{labels[class_idx]} ({confidence:.2f})"
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                backend.draw_legend(frame, preds_to_display, labels)
                stframe.image(frame, channels="BGR")

            cap.release()
        else:
            st.warning("Kamera belum dinyalakan.")

elif option == "Deteksi Ekspresi":
    st.title("Deteksi Ekspresi")

    labels = ekspresi.get_labels() 
    uploaded_file = st.file_uploader("Unggah gambar untuk prediksi", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Membaca gambar yang diupload
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

        # Konversi gambar menjadi format yang bisa diproses
        image_cv = np.array(image.convert("RGB"))
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        # Preprocessing untuk deteksi wajah
        preprocessed_face, cropped_face = backend.preprocess_face(image_cv)

        if preprocessed_face is not None:
            st.subheader("Wajah yang Diproses")
            st.image(cropped_face, caption="Wajah yang Di-crop", use_column_width=True)

            # Melakukan prediksi ekspresi
            preds = ekspresi.predict(preprocessed_face)
            class_idx = np.argmax(preds)
            confidence = preds[class_idx]

            # Menampilkan hasil prediksi
            st.success(f"Prediksi Ekspresi: {labels[class_idx]} ({confidence:.2f})")
        else:
            st.error("Tidak terdeteksi wajah pada gambar. Pastikan wajah terlihat jelas.")

