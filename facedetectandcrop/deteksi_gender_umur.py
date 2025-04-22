import os
import time
from deepface import DeepFace

def analyze_deepface(img_path):
    try:
        start = time.time()
        result = DeepFace.analyze(
            img_path=img_path,
            actions=['age', 'gender'],
            detector_backend='retinaface',
            enforce_detection=False
        )
        result = result[0] if isinstance(result, list) else result
        gender_dict = result['gender']
        gender = max(gender_dict, key=gender_dict.get)
        confidence = gender_dict[gender]
        age = result['age']
        elapsed = round(time.time() - start, 3)

        return {
            'source': 'DeepFace (retinaface)',
            'age': int(age),
            'gender': gender,
            'confidence': round(confidence, 2),
            'time_sec': elapsed
        }
    except Exception:
        return {
            'source': 'DeepFace (retinaface)',
            'age': 'N/A',
            'gender': 'N/A',
            'confidence': 'N/A',
            'time_sec': 'ERROR'
        }

def compare_model(img_path):
    print(f"[INFO] Analisis gambar: {img_path}\n")
    print(f"{'Model':<25} | {'Age':<5} | {'Gender':<6} | {'Conf%':<7} | {'Time(s)':<7}")
    print("-" * 60)

    result = analyze_deepface(img_path)
    print(f"{result['source']:<25} | {str(result['age']):<5} | {result['gender']:<6} | {str(result['confidence']):<7} | {result['time_sec']:<7}")

if __name__ == "__main__":
    img_path = "kemal1.jpg"
    if os.path.exists(img_path):
        compare_model(img_path)
    else:
        print(f"[ERROR] Gambar '{img_path}' tidak ditemukan.")
