import os
import cv2
import numpy as np
import random
import shutil
from mtcnn import MTCNN

def rotate_image(image, angle=15):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def flip_image(image):
    return cv2.flip(image, 1)

def adjust_brightness_contrast(image, alpha=1.0, beta=30):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def add_gaussian_noise(image):
    noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def crop_faces(folder_path):
    detector = MTCNN()
    success_dir = "Dataset/Berhasil"
    failure_dir = "Dataset/Gagal"

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, file)
                img = cv2.imread(file_path)
                if img is None:
                    print(f"[!] Gagal membaca gambar: {file_path}")
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(img_rgb)

                if faces:
                    for i, face in enumerate(faces):
                        x, y, w, h = face['box']
                        x, y = max(0, x), max(0, y)
                        cropped_face = img[y:y+h, x:x+w]

                        relative_folder = os.path.relpath(root, "Dataset")
                        target_folder = os.path.join(success_dir, relative_folder)
                        os.makedirs(target_folder, exist_ok=True)

                        save_path = os.path.join(target_folder, f"{os.path.splitext(file)[0]}_face_{i+1}.jpg")
                        cv2.imwrite(save_path, cropped_face)

                    print(f"[✓] Wajah terdeteksi dan disimpan dari: {file_path}")
                else:
                    relative_folder = os.path.relpath(root, "Dataset")
                    target_folder = os.path.join(failure_dir, relative_folder)
                    os.makedirs(target_folder, exist_ok=True)

                    save_path = os.path.join(target_folder, file)
                    cv2.imwrite(save_path, img)

                    print(f"[X] Tidak ada wajah terdeteksi pada: {file_path}")

def preprocess_images(folder_path):
    output_base = "Augmented"
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, file)
                img = cv2.imread(file_path)
                if img is None:
                    print(f"[!] Gagal membaca gambar: {file_path}")
                    continue

                relative_folder = os.path.relpath(root, "Dataset/Berhasil")
                target_folder = os.path.join(output_base, relative_folder)
                os.makedirs(target_folder, exist_ok=True)

                base_name = os.path.splitext(file)[0]
                ext = os.path.splitext(file)[1]

                rotated = rotate_image(img)
                cv2.imwrite(os.path.join(target_folder, f"rotated_{base_name}{ext}"), rotated)

                flipped = flip_image(img)
                cv2.imwrite(os.path.join(target_folder, f"flipped_{base_name}{ext}"), flipped)

                bc_adjusted = adjust_brightness_contrast(img, alpha=1.0, beta=30)
                cv2.imwrite(os.path.join(target_folder, f"brightness&contrast_1.0_{base_name}{ext}"), bc_adjusted)

                noisy = add_gaussian_noise(img)
                cv2.imwrite(os.path.join(target_folder, f"noise_{base_name}{ext}"), noisy)

                print(f"[✓] Preprocessing selesai untuk: {file_path}")

def split_dataset(folder_path, output_base="DatasetSplit", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Total rasio harus 1.0"

    class_names = os.listdir(folder_path)
    for class_name in class_names:
        class_path = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)

        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        split_map = {
            "Training": images[:train_end],
            "Validation": images[train_end:val_end],
            "Testing": images[val_end:]
        }

        for split_name, split_images in split_map.items():
            target_dir = os.path.join(output_base, split_name, class_name)
            os.makedirs(target_dir, exist_ok=True)
            for img_name in split_images:
                src = os.path.join(class_path, img_name)
                dst = os.path.join(target_dir, img_name)
                shutil.copy2(src, dst)

        print(f"[✓] {class_name}: {len(images)} gambar dibagi ke Training, Validation, dan Testing.")

    print(f"\n[✓] Dataset berhasil dibagi dan disimpan ke folder '{output_base}'.")

def main():
    print("=== Program Crop & Preprocessing Gambar ===")
    print("1. Deteksi wajah dan Crop Gambar")
    print("2. Preprocessing gambar")
    print("3. Split dataset")
    pilihan = input("Masukkan pilihan (1/2/3): ")

    if pilihan == '1':
        folder = input("Masukkan folder yang ingin diproses (misal: Dataset/Jawa/Yasin): ")
        crop_faces(folder)

    elif pilihan == '2':
        folder = input("Masukkan folder hasil crop (misal: Dataset/Berhasil/Jawa/Yasin): ")
        preprocess_images(folder)

    elif pilihan == '3':
        folder = input("Masukkan folder yang ingin di-split (misal: Augmented/Jawa/Yasin): ")
        split_dataset(folder)

    else:
        print("Pilihan tidak valid.")

if __name__ == "__main__":
    main()
