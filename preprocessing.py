import 
import os
import numpy as np
from tqdm import tqdm

# ===== Helper: Buat Folder =====
def make_dirs(folder_list):
    for folder in folder_list:
        os.makedirs(folder, exist_ok=True)

# ===== Crop Wajah =====
def crop_faces(input_folder, output_folder, face_cascade_path='haarcascade_frontalface_default.xml'):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + face_cascade_path)

    for root, _, files in os.walk(input_folder):
        for img_name in tqdm(files, desc="Cropping wajah"):
            img_path = os.path.join(root, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                face = cv2.resize(face, (224, 224))

                # Simpan dengan struktur folder yang sama
                rel_path = os.path.relpath(root, input_folder)
                save_dir = os.path.join(output_folder, rel_path)
                os.makedirs(save_dir, exist_ok=True)
                output_path = os.path.join(save_dir, img_name)

                cv2.imwrite(output_path, face)
                break  # Ambil satu wajah saja

# ===== Augmentasi Functions =====
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(image, M, (w, h))

def flip_image(image):
    return cv2.flip(image, 1)

def adjust_brightness_contrast(image, alpha, beta):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def add_gaussian_noise(image):
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

# ===== Augmentasi Pipeline =====
def augment_images(input_folder, output_folder):
    for root, _, files in os.walk(input_folder):
        for img_name in tqdm(files, desc="Augmentasi gambar"):
            img_path = os.path.join(root, img_name)
            image = cv2.imread(img_path)

            if image is None:
                continue

            rel_path = os.path.relpath(root, input_folder)
            save_dir = os.path.join(output_folder, rel_path)
            os.makedirs(save_dir, exist_ok=True)

            base_name, ext = os.path.splitext(img_name)

            # Rotasi
            rotated = rotate_image(image, 15)
            cv2.imwrite(os.path.join(save_dir, f"{base_name}_rotated{ext}"), rotated)

            # Flip
            flipped = flip_image(image)
            cv2.imwrite(os.path.join(save_dir, f"{base_name}_flipped{ext}"), flipped)

            # Brightness dan Contrast (+20%)
            bright = adjust_brightness_contrast(image, alpha=1.2, beta=20)
            cv2.imwrite(os.path.join(save_dir, f"{base_name}_bright{ext}"), bright)

            # Brightness dan Contrast (-20%)
            dark = adjust_brightness_contrast(image, alpha=0.8, beta=-20)
            cv2.imwrite(os.path.join(save_dir, f"{base_name}_dark{ext}"), dark)

            # Gaussian Noise
            noisy = add_gaussian_noise(image)
            cv2.imwrite(os.path.join(save_dir, f"{base_name}_noisy{ext}"), noisy)

# ===== Main Program =====
if __name__ == "__main__":
    original_folder = 'dataset/original'
    cropped_folder = 'dataset/cropped'
    augmented_folder = 'dataset/augmented'

    # Buat folder output
    make_dirs([cropped_folder, augmented_folder])

    # Step 1: Crop wajah dulu
    crop_faces(original_folder, cropped_folder)

    # Step 2: Augmentasi hasil crop
    augment_images(cropped_folder, augmented_folder)

    print("Proses selesai! ✅")
