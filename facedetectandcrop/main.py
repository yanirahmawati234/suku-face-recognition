import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
from mtcnn import MTCNN

# def crop_face(image):
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     if len(faces) == 0:
#         return None, "Tidak ada wajah terdeteksi."
#     (x, y, w, h) = faces[0]
#     face_image = image[y:y+h, x:x+w]
#     return face_image, None

def crop_face_with_mtcnn(image):
    detector = MTCNN()
    faces = detector.detect_faces(image)

    if len(faces) == 0:
        return None, "Tidak ada wajah terdeteksi."
    
    (x, y, w, h) = faces[0]['box']
    face_image = image[y:y+h, x:x+w]
    return face_image, None

st.title('Deteksi Wajah dan Pemotongan Gambar')

suku_name = st.text_input("Masukkan Nama Suku")
person_name = st.text_input("Masukkan Nama Orang")

uploaded_file = st.file_uploader("Pilih gambar untuk diproses", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and suku_name and person_name:
    image = Image.open(uploaded_file)
    opencv_image = np.array(image)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
    st.image(opencv_image, channels="BGR", caption="Gambar Asli", use_column_width=True)

    # face_image, error = crop_face(opencv_image)
    face_image, error = crop_face_with_mtcnn(opencv_image)

    if face_image is not None:
        st.image(face_image, channels="BGR", caption="Gambar Setelah Crop", use_column_width=True)

        if st.button("Simpan Foto"):
            dataset_directory = 'dataset'
            original_directory = os.path.join(dataset_directory, 'original', suku_name, person_name)
            crop_directory = os.path.join(dataset_directory, 'crop', suku_name, person_name)

            if not os.path.exists(dataset_directory):
                os.makedirs(dataset_directory)

            if not os.path.exists(original_directory):
                os.makedirs(original_directory)
            
            if not os.path.exists(crop_directory):
                os.makedirs(crop_directory)

            original_path = os.path.join(original_directory, uploaded_file.name)
            with open(original_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            cropped_image_path = os.path.join(crop_directory, f"cropped_{uploaded_file.name}")
            cv2.imwrite(cropped_image_path, face_image)

            st.success(f"Gambar asli telah disimpan di: {original_path}")
            st.success(f"Gambar hasil crop telah disimpan di: {cropped_image_path}")
    else:
        st.error(error)
else:
    if uploaded_file is not None:
        st.error("Silakan isi Nama Suku dan Nama Orang untuk menyimpan gambar.")
