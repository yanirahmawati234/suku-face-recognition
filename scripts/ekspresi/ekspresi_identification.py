import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Memuat model ekspresi yang sudah dilatih
MODEL_PATH = 'model/ekspresi_mobilenetv2_model.h5'  
model = load_model(MODEL_PATH)

# Label ekspresi wajah yang digunakan pada model
labels = ['senyum', 'serius','terkejut']  

# Fungsi untuk memuat label ekspresi
def get_labels():
    return labels

# Fungsi untuk mendeteksi wajah dari gambar (menggunakan OpenCV)
def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    detected_faces = []
    for (x, y, w, h) in faces:
        detected_faces.append({'box': (x, y, w, h)})
    return detected_faces

# Fungsi untuk memproses gambar wajah sebelum prediksi
def preprocess_face(image):
    # Mengubah gambar ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Deteksi wajah menggunakan haarcascades OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None, None

    # Ambil wajah pertama (jika ada lebih dari satu wajah)
    (x, y, w, h) = faces[0]
    cropped_face = image[y:y+h, x:x+w]

    # Resize wajah untuk input ke model (misalnya 224x224)
    preprocessed_face = cv2.resize(cropped_face, (224, 224))
    preprocessed_face = preprocessed_face.astype("float32") / 255.0  # Normalisasi
    preprocessed_face = np.expand_dims(preprocessed_face, axis=0)  # Tambahkan dimensi batch
    return preprocessed_face, cropped_face

# Fungsi untuk memproses banyak wajah
def preprocess_multiple_faces(face_crop):
    preprocessed_face = cv2.resize(face_crop, (224, 224))
    preprocessed_face = preprocessed_face.astype("float32") / 255.0
    preprocessed_face = np.expand_dims(preprocessed_face, axis=0)
    return preprocessed_face

# Fungsi untuk memprediksi ekspresi wajah
def predict(image):
    preds = model.predict(image)  # Melakukan prediksi pada wajah yang sudah diproses
    return preds[0]

# Fungsi untuk menggambar legend atau hasil prediksi di frame
def draw_legend(frame, preds, labels):
    for i, label in enumerate(labels):
        cv2.putText(frame, f"{label}: {preds[i]:.2f}", (10, 30 + (i * 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
