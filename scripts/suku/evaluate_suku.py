import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ===== 1. Configuration =====
DATASET_DIR = 'dataset/split'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ===== 2. Data Preparation =====
val_test_aug = ImageDataGenerator(rescale=1./255)

test_gen = val_test_aug.flow_from_directory(
    os.path.join(DATASET_DIR, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ===== 3. Load Models =====
vgg_model = tf.keras.models.load_model('model/vgg16_finetuned_final.h5')
mobilenet_model = tf.keras.models.load_model('model/mobilenetv2_finetuned_final.h5')

# ===== 4. Ensemble Evaluation =====
test_gen.reset()
pred_vgg = vgg_model.predict(test_gen)
pred_mobilenet = mobilenet_model.predict(test_gen)

ensemble_pred = (pred_vgg + pred_mobilenet) / 2
y_pred_ensemble = np.argmax(ensemble_pred, axis=1)
y_true = test_gen.classes
labels = list(test_gen.class_indices.keys())

# Save classification report to .txt file
report = classification_report(y_true, y_pred_ensemble, target_names=labels)
with open('classification_report.txt', 'w') as f:
    f.write(report)

# Print and plot confusion matrix
print("\nEnsemble Classification Report (VGG16 + MobileNetV2):")
print(report)

conf_mat = confusion_matrix(y_true, y_pred_ensemble)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix: VGG16 + MobileNetV2')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Confusion Matrix for VGG16
pred_vgg_classes = np.argmax(pred_vgg, axis=1)
conf_mat_vgg = confusion_matrix(y_true, pred_vgg_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat_vgg, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix: VGG16')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Confusion Matrix for MobileNetV2
pred_mobilenet_classes = np.argmax(pred_mobilenet, axis=1)
conf_mat_mobilenet = confusion_matrix(y_true, pred_mobilenet_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat_mobilenet, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix: MobileNetV2')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
