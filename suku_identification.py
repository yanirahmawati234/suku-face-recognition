# face_detector_backend.py

import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN

# Load model
vgg_model = tf.keras.models.load_model('vgg16_finetuned_final.h5')
mobilenet_model = tf.keras.models.load_model('mobilenetv2_finetuned_final.h5')
labels = ['Jawa', 'Melayu', 'Sunda']

# Inisialisasi face detector
detector = MTCNN()

def get_labels():
    return labels

def preprocess_face(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)
    if faces:
        x, y, w, h = faces[0]['box']
        x, y = max(0, x), max(0, y)
        face_img = image[y:y+h, x:x+w]

        face_img_resized = cv2.resize(face_img, (224, 224))
        face_img_resized = face_img_resized.astype("float32") / 255.0
        face_img_resized = np.expand_dims(face_img_resized, axis=0)
        return face_img_resized, face_img
    else:
        return None, None

def ensemble_predict(img_array, weight_vgg=0.4, weight_mobilenet=0.6):
    vgg_preds = vgg_model.predict(img_array)[0]
    mobilenet_preds = mobilenet_model.predict(img_array)[0]
    combined_preds = (weight_vgg * vgg_preds) + (weight_mobilenet * mobilenet_preds)
    return combined_preds

def detect_faces(image):
    return detector.detect_faces(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def preprocess_multiple_faces(face_crop):
    resized = cv2.resize(face_crop, (224, 224))
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=0)

def draw_legend(frame, preds, labels, start_x=10, start_y=10):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    line_height = 20

    box_width = 200
    box_height = line_height * (len(labels) + 1) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (start_x, start_y), (start_x + box_width, start_y + box_height), (50, 50, 50), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, "Confidence:", (start_x + 5, start_y + 20), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    for i, label in enumerate(labels):
        text = f"{label}: {preds[i]*100:.1f}%"
        y = start_y + 20 + (i+1)*line_height
        cv2.putText(frame, text, (start_x + 5, y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
