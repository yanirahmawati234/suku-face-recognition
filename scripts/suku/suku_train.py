import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import VGG16, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ===== 1. Configuration =====
DATASET_DIR = 'dataset_split'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
FINE_TUNE_EPOCHS = 10

# ===== 2. Data Preparation =====
train_aug = ImageDataGenerator(rescale=1./255)
val_test_aug = ImageDataGenerator(rescale=1./255)

train_gen = train_aug.flow_from_directory(
    os.path.join(DATASET_DIR, 'augmented'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)
val_gen = val_test_aug.flow_from_directory(
    os.path.join(DATASET_DIR, 'cropped/val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
test_gen = val_test_aug.flow_from_directory(
    os.path.join(DATASET_DIR, 'cropped/test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = train_gen.num_classes
class_weights = dict(enumerate(
    class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
))

# ===== 3. Model Builder =====
def create_model(base_model, num_classes):
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=predictions), base_model

vgg_model, vgg_base = create_model(VGG16(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,)), num_classes)
mobilenet_model, mobilenet_base = create_model(MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,)), num_classes)

# Compile models
for model in [vgg_model, mobilenet_model]:
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# ===== 4. Callbacks =====
def create_callbacks(name):
    return [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
        ModelCheckpoint(f'best_{name}.h5', monitor='val_accuracy', save_best_only=True)
    ]

callbacks_vgg = create_callbacks("vgg16")
callbacks_mobilenet = create_callbacks("mobilenetv2")

# ===== 5. Initial Training =====
history_vgg = vgg_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks_vgg
)
history_mobilenet = mobilenet_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks_mobilenet
)

# ===== 6. Fine-tuning =====
def fine_tune_model(model, base_model, fine_tune_at, history, callbacks):
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True
    model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    return model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=history.epoch[-1] + 1,
        class_weight=class_weights,
        callbacks=callbacks
    )

history_vgg_fine = fine_tune_model(vgg_model, vgg_base, 15, history_vgg, callbacks_vgg)
history_mobilenet_fine = fine_tune_model(mobilenet_model, mobilenet_base, 100, history_mobilenet, callbacks_mobilenet)

# ===== 7. Save Fine-tuned Models =====
vgg_model.save('model/vgg16_finetuned_final2.h5')
mobilenet_model.save('model/mobilenetv2_finetuned_final2.h5')
print("Fine-tuned models saved successfully!")

# ===== 8. Ensemble Evaluation with Confusion Matrix =====
vgg_model = tf.keras.models.load_model('best_vgg16.h5')
mobilenet_model = tf.keras.models.load_model('best_mobilenetv2.h5')

test_gen.reset()
pred_vgg = vgg_model.predict(test_gen)
pred_mobilenet = mobilenet_model.predict(test_gen)

ensemble_pred = (pred_vgg + pred_mobilenet) / 2
y_pred_ensemble = np.argmax(ensemble_pred, axis=1)
y_true = test_gen.classes
labels = list(test_gen.class_indices.keys())

print("\n📋 Ensemble Classification Report (VGG16 + MobileNetV2):")
print(classification_report(y_true, y_pred_ensemble, target_names=labels))

conf_mat = confusion_matrix(y_true, y_pred_ensemble)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix: VGG16 + MobileNetV2')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ===== 9. Plot Training History =====
def plot_history(histories, metrics=('accuracy', 'loss')):
    plt.figure(figsize=(14, 5))
    for i, metric in enumerate(metrics):
        plt.subplot(1, 2, i + 1)
        for name, history in histories:
            plt.plot(history.history[metric], label=f'{name} Train')
            plt.plot(history.history['val_' + metric], label=f'{name} Val')
        plt.title(f'{metric.capitalize()} Curve')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
    plt.tight_layout()
    plt.show()

plot_history([
    ('VGG16', history_vgg),
    ('VGG16 Fine', history_vgg_fine),
    ('MobileNetV2', history_mobilenet),
    ('MobileNetV2 Fine', history_mobilenet_fine),
])
