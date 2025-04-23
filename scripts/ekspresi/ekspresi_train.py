import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ===== Config =====
DATASET_TRAIN_DIR = 'dataset/split/train'
DATASET_VAL_DIR = 'dataset/split/val'
DATASET_TEST_DIR = 'dataset/split/test'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15  
def load_data(dataset_dir):
    X, y = [], []

    for suku_name in os.listdir(dataset_dir):
        suku_path = os.path.join(dataset_dir, suku_name)
        if not os.path.isdir(suku_path):
            continue
        
        for subject_name in os.listdir(suku_path):
            subject_path = os.path.join(suku_path, subject_name)
            if not os.path.isdir(subject_path):
                continue

            for filename in os.listdir(subject_path):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    img_path = os.path.join(subject_path, filename)
                    try:
                        img = load_img(img_path, target_size=IMG_SIZE)
                        img = img_to_array(img)
                        img = preprocess_input(img)  # penting!
                        X.append(img)

                        parts = filename.split('_')
                        if len(parts) > 1:
                            ekspresi = parts[1].lower()
                            y.append(ekspresi)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
    
    return np.array(X), np.array(y)

# ===== Load Data =====
print("Loading data...")
X_train, y_train = load_data(DATASET_TRAIN_DIR)
X_val, y_val = load_data(DATASET_VAL_DIR)
X_test, y_test = load_data(DATASET_TEST_DIR)

# ===== Label Encoding =====
label_encoder = LabelEncoder()
y_train_enc = to_categorical(label_encoder.fit_transform(y_train))
y_val_enc = to_categorical(label_encoder.transform(y_val))
y_test_enc = to_categorical(label_encoder.transform(y_test))
num_classes = y_train_enc.shape[1]

print(f"Kelas ekspresi: {label_encoder.classes_}")

# ===== Build Model with MobileNetV2 =====
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # bisa di-unfreeze sebagian jika ingin fine-tuning

inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs, outputs)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ===== Train =====
history = model.fit(
    X_train, y_train_enc,
    validation_data=(X_val, y_val_enc),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# ===== Evaluate =====
loss, acc = model.evaluate(X_test, y_test_enc)
print(f"Test Accuracy: {acc*100:.2f}%")

# ===== Confusion Matrix =====
y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test_enc, axis=1)
cm = confusion_matrix(y_true, y_pred_class)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# ===== Save Model =====
os.makedirs('model', exist_ok=True)
model.save('model/ekspresi_mobilenetv2_model.h5')
print("Model saved at model/ekspresi_mobilenetv2_model.h5")
