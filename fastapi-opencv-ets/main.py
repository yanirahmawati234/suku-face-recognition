import cv2
import os
import pandas as pd
from preprocessing import preprocess_image

def test_preprocessing(input_path, output_dir, nama, suku, ekspresi, sudut, pencahayaan):
    """
    Menguji fungsi preprocessing pada gambar input dan menyimpan hasilnya.
    Juga memperbarui metadata.csv.
    """
    # Membaca gambar
    image = cv2.imread(input_path)
    if image is None:
        print(f"Gagal membaca gambar: {input_path}")
        return

    filename = os.path.basename(input_path)
    base_name = os.path.splitext(filename)[0]  # Misal: "senyum_45_indoor"

    # Pastikan direktori output ada (sama dengan direktori input)
    output_subdir = os.path.join(output_dir, "original", nama, suku)
    os.makedirs(output_subdir, exist_ok=True)

    # Cek apakah file sudah diproses (jika semua file hasil preprocessing sudah ada, lewati)
    rot15_output_path = os.path.join(output_subdir, f"{base_name}_rot15.jpg")
    rot_minus15_output_path = os.path.join(output_subdir, f"{base_name}_rot-15.jpg")
    flip_output_path = os.path.join(output_subdir, f"{base_name}_flip.jpg")
    if (os.path.exists(rot15_output_path) and 
        os.path.exists(rot_minus15_output_path) and 
        os.path.exists(flip_output_path)):
        print(f"Gambar {filename} sudah diproses sebelumnya, melewati...")
        return

    # Data untuk CSV
    csv_data = []

    # Simpan gambar asli
    original_output_path = os.path.join(output_subdir, filename)
    cv2.imwrite(original_output_path, image)
    csv_data.append({
        "path_gambar": original_output_path,
        "nama": nama,
        "suku": suku,
        "ekspresi": ekspresi,
        "sudut": sudut,
        "pencahayaan": pencahayaan
    })

    # Preprocessing: Rotasi +15°
    rot15_image = preprocess_image(image, apply_rotation=True, apply_flip=False, rotation_angle=15)
    cv2.imwrite(rot15_output_path, rot15_image)
    csv_data.append({
        "path_gambar": rot15_output_path,
        "nama": nama,
        "suku": suku,
        "ekspresi": ekspresi,
        "sudut": "rotasi15",
        "pencahayaan": pencahayaan
    })

    # Preprocessing: Rotasi -15°
    rot_minus15_image = preprocess_image(image, apply_rotation=True, apply_flip=False, rotation_angle=-15)
    cv2.imwrite(rot_minus15_output_path, rot_minus15_image)
    csv_data.append({
        "path_gambar": rot_minus15_output_path,
        "nama": nama,
        "suku": suku,
        "ekspresi": ekspresi,
        "sudut": "rotasi-15",
        "pencahayaan": pencahayaan
    })

    # Preprocessing: Flip
    flip_image_result = preprocess_image(image, apply_rotation=False, apply_flip=True)
    cv2.imwrite(flip_output_path, flip_image_result)
    csv_data.append({
        "path_gambar": flip_output_path,
        "nama": nama,
        "suku": suku,
        "ekspresi": ekspresi,
        "sudut": "flip",
        "pencahayaan": pencahayaan
    })

    # Simpan ke CSV di folder dataset/
    csv_file = os.path.join(output_dir, "metadata.csv")
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)
    print(f"Hasil preprocessing untuk {filename} disimpan di: {output_subdir}")
    print(f"Metadata diperbarui di: {csv_file}")

if __name__ == "__main__":
    # Konfigurasi untuk pengujian
    dataset_dir = "dataset"
    original_dir = os.path.join(dataset_dir, "original")

    # Pastikan folder dataset/original ada
    if not os.path.exists(original_dir):
        print(f"Folder {original_dir} tidak ditemukan!")
        exit()

    # Telusuri semua subfolder di dataset/original/
    test_cases = []
    for root, dirs, files in os.walk(original_dir):
        # Dapatkan nama subjek dan suku dari struktur folder
        # Contoh path: dataset/original/Alfhi/Sunda
        relative_path = os.path.relpath(root, original_dir)
        path_parts = relative_path.split(os.sep)
        if len(path_parts) != 2:
            continue  # Lewati jika struktur folder tidak sesuai (harus ada nama dan suku)

        nama, suku = path_parts  # Misal: nama = "Alfhi", suku = "Sunda"

        # Filter file gambar asli (abaikan file hasil preprocessing)
        image_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png')) and not any(suffix in f for suffix in ['rot15', 'rot-15', 'flip'])]

        # Buat test case untuk setiap file gambar
        for image_file in image_files:
            # Ekstrak metadata dari nama file (misal: senyum_45_indoor.jpg)
            parts = image_file.split('_')
            if len(parts) >= 3:
                ekspresi = parts[0]  # senyum atau serius
                sudut = parts[1]     # 45 atau profil
                if sudut == "45":
                    sudut = "miring45"
                pencahayaan = parts[2].split('.')[0]  # indoor atau outdoor
            else:
                ekspresi = "unknown"
                sudut = "unknown"
                pencahayaan = "unknown"

            test_cases.append({
                "input_path": os.path.join(root, image_file),
                "nama": nama,
                "suku": suku,
                "ekspresi": ekspresi,
                "sudut": sudut,
                "pencahayaan": pencahayaan
            })

    # Jalankan pengujian untuk setiap test case
    for test_case in test_cases:
        test_preprocessing(
            input_path=test_case["input_path"],
            output_dir=dataset_dir,
            nama=test_case["nama"],
            suku=test_case["suku"],
            ekspresi=test_case["ekspresi"],
            sudut=test_case["sudut"],
            pencahayaan=test_case["pencahayaan"]
        )

    print("Selesai memproses semua gambar di dataset!")