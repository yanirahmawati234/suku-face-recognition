import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from mtcnn import MTCNN
import pandas as pd

# Load model
vgg_model = load_model("model/best_vgg16.h5")
mobilenet_model = load_model("model/best_mobilenetv2.h5")
labels = ['Jawa', 'Melayu', 'Sunda']

# Inisialisasi MTCNN face detector
detector = MTCNN()

# Fungsi preprocessing
def rotate_image(image, angle=0):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def flip_image(image):
    return cv2.flip(image, 1)

def adjust_brightness_contrast(image, alpha=1.0, beta=30):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def preprocess_face(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)
    if faces:
        x, y, w, h = faces[0]['box']
        x, y = max(0, x), max(0, y)
        face_img = image[y:y+h, x:x+w]
        
        face_img = rotate_image(face_img, angle=0)
        face_img = flip_image(face_img)
        face_img = adjust_brightness_contrast(face_img, alpha=1.0, beta=20)

        face_img = cv2.resize(face_img, (224, 224))
        face_img = face_img.astype("float32") / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        return face_img
    else:
        return None

# Fungsi ensemble prediksi
def ensemble_predict(img_array, weight_vgg=0.4, weight_mobilenet=0.6):
    vgg_preds = vgg_model.predict(img_array)[0]
    mobilenet_preds = mobilenet_model.predict(img_array)[0]
    combined_preds = (weight_vgg * vgg_preds) + (weight_mobilenet * mobilenet_preds)
    return combined_preds

# Fungsi gambar legend di frame
def draw_legend(frame, preds, labels, start_x=10, start_y=10):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    line_height = 20

    # background box
    box_width = 200
    box_height = line_height * (len(labels) + 1) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (start_x, start_y), (start_x + box_width, start_y + box_height), (50, 50, 50), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # title
    cv2.putText(frame, "Confidence:", (start_x + 5, start_y + 20),
                font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    # per label
    for i, label in enumerate(labels):
        text = f"{label}: {preds[i]*100:.1f}%"
        y = start_y + 20 + (i+1)*line_height
        cv2.putText(frame, text, (start_x + 5, y),
                    font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

# Streamlit UI
st.title("Deteksi Suku/Etnis Real-time dan Upload Gambar")
choice = st.radio("Pilih mode deteksi", ('Webcam', 'Upload Gambar'))

if choice == 'Upload Gambar':
    uploaded_file = st.file_uploader("Unggah gambar untuk prediksi", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

        image_cv = np.array(image.convert("RGB"))
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        preprocessed_face = preprocess_face(image_cv)
        
        if preprocessed_face is not None:
            preds = ensemble_predict(preprocessed_face)
            class_idx = np.argmax(preds)
            confidence = preds[class_idx]

            st.success(f"Prediksi: {labels[class_idx]} ({confidence:.2f})")
            st.subheader("Distribusi Confidence Score:")

            # Tampilkan bar chart
            df = pd.DataFrame({
                'Etnis': labels,
                'Confidence': preds
            })
            st.bar_chart(df.set_index('Etnis'))
        else:
            st.error("Tidak terdeteksi wajah pada gambar.")

elif choice == 'Webcam':
    stframe = st.empty()
    run = st.checkbox('Nyalakan Kamera')

    cap = None

    if run:
        cap = cv2.VideoCapture(0)
    else:
        if cap:
            cap.release()

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Gagal mengambil frame dari webcam.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(frame_rgb)

        preds_to_display = np.zeros(len(labels))  # inisialisasi prediksi 0 semua

        if faces:
            for face in faces:
                x, y, w, h = face['box']
                x, y = max(0, x), max(0, y)
                face_crop = frame[y:y+h, x:x+w]

                preprocessed_face = preprocess_face(face_crop)
                if preprocessed_face is not None:
                    preds = ensemble_predict(preprocessed_face)
                    preds_to_display = preds  # update prediksi buat legend
                    class_idx = np.argmax(preds)
                    confidence = preds[class_idx]

                    label_text = f"{labels[class_idx]} ({confidence:.2f})"
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, label_text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Gambar legend confidence di kiri atas
        draw_legend(frame, preds_to_display, labels)

        stframe.image(frame, channels="BGR")

    if cap:
        cap.release()
