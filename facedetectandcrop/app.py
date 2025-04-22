import streamlit as st
import os
import tempfile
from preprocessing import crop_faces, preprocess_images, split_dataset
from facesimilarity import extract_embedding, calculate_similarity, visualize_comparison
from deteksi_gender_umur import analyze_deepface
from facesimilarity_evaluation import (
    load_embeddings_from_dataset,
    plot_tsne,
    generate_pairs,
    evaluate_roc,
    plot_similarity_distribution,
    save_example_visuals
)
import matplotlib.pyplot as plt
import shutil

st.set_page_config(page_title="Face Processing App", layout="centered")
st.title("👤 Face Recognition App")

option = st.sidebar.radio("Pilih Mode", ["Preprocessing", "Face Similarity", "Deteksi Umur & Gender"])

def save_uploaded_folder(uploaded_files, extract_dir):
    for uploaded_file in uploaded_files:
        file_path = os.path.join(extract_dir, uploaded_file.name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
    return extract_dir

if option == "Preprocessing":
    st.header("🛠️ Preprocessing Wajah")
    task = st.selectbox("Pilih tindakan", ["Face Detection & Crop", "Augmentasi", "Split Dataset"])
    input_mode = st.radio("Pilih metode input", ["Input Path Folder", "Upload File"])

    if input_mode == "Input Path Folder":
        folder_path = st.text_input("Masukkan path folder")
    else:
        uploaded_files = st.file_uploader("Upload file atau folder (upload multiple files)", accept_multiple_files=True)
        folder_path = None
        if uploaded_files:
            tmpdir = tempfile.mkdtemp()
            folder_path = save_uploaded_folder(uploaded_files, tmpdir)

    if folder_path:
        if task == "Face Detection & Crop":
            if st.button("Mulai Deteksi dan Crop"):
                crop_faces(folder_path)
                st.success("Deteksi wajah dan crop selesai.")

        elif task == "Augmentasi":
            if st.button("Mulai Augmentasi"):
                preprocess_images(folder_path)
                st.success("Proses augmentasi selesai.")

        elif task == "Split Dataset":
            if st.button("Mulai Split Dataset"):
                split_dataset(folder_path)
                st.success("Split dataset selesai.")

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