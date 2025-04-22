import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import suku_identification as backend  # Ubah sesuai nama file back-end kamu kalau perlu

st.title("Deteksi Suku/Etnis Real-time dan Upload Gambar")
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

            preds = backend.predict(preprocessed_face)  # ✅ Ganti ke predict
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
                    preds = backend.predict(face_input)  # ✅ Ganti ke predict
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
