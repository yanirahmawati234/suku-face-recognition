from flask import Flask, request, jsonify
import time
import os
from deepface import DeepFace

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze_deepface():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    temp_path = f"temp_{file.filename}"
    file.save(temp_path)

    try:
        start = time.time()
        result = DeepFace.analyze(
            img_path=temp_path,
            actions=['age', 'gender'],
            detector_backend='retinaface',
            enforce_detection=False
        )
        result = result[0] if isinstance(result, list) else result

        if 'gender' not in result or not result['gender']:
            raise ValueError("Gender tidak terdeteksi")

        gender_dict = result['gender']
        gender = max(gender_dict, key=gender_dict.get)
        confidence = gender_dict[gender]
        age = result['age']
        elapsed = round(time.time() - start, 3)

        os.remove(temp_path)

        return jsonify({
            "source": "DeepFace (retinaface)",
            "age": int(age),
            "gender": gender,
            "confidence": round(confidence, 2),
            "time_sec": elapsed
        })
    except Exception as e:
        return jsonify({
            "source": "DeepFace (retinaface)",
            "age": "N/A",
            "gender": "N/A",
            "confidence": "N/A",
            "time_sec": "ERROR",
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
