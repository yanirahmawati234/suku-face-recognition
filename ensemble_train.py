import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ===== 1. Configuration =====
DATASET_DIR = 'dataset/split'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
FINE_TUNE_EPOCHS = 10

# ===== 2. Data Preparation =====
train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)
val_test_aug = ImageDataGenerator(rescale=1./255)

train_gen = train_aug.flow_from_directory(
    os.path.join(DATASET_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)
val_gen = val_test_aug.flow_from_directory(
    os.path.join(DATASET_DIR, 'val'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
test_gen = val_test_aug.flow_from_directory(
    os.path.join(DATASET_DIR, 'test'),
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

# ===== 3. Model Definitions =====
def create_resnet_model(num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def create_vgg16_model(num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def create_efficientnet_model(num_classes):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

# Create models
resnet_model, resnet_base = create_resnet_model(num_classes)
vgg16_model, vgg16_base = create_vgg16_model(num_classes)
efficientnet_model, efficientnet_base = create_efficientnet_model(num_classes)

# Compile models
resnet_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
vgg16_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
efficientnet_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# ===== 4. Callbacks =====
callbacks_resnet = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
    ModelCheckpoint('best_resnet.h5', monitor='val_accuracy', save_best_only=True),
]
callbacks_vgg16 = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
    ModelCheckpoint('best_vgg16.h5', monitor='val_accuracy', save_best_only=True),
]
callbacks_efficientnet = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
    ModelCheckpoint('best_efficientnet.h5', monitor='val_accuracy', save_best_only=True),
]

# ===== 5. Initial Training =====
# Train ResNet50
history_resnet = resnet_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks_resnet
)

# Train VGG16
history_vgg16 = vgg16_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks_vgg16
)

# Train EfficientNetB0
history_efficientnet = efficientnet_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks_efficientnet
)

# ===== 6. Fine-tuning =====
# Fine-tune ResNet50
fine_tune_at_resnet = 140  # Unfreeze from this layer
for layer in resnet_base.layers[:fine_tune_at_resnet]:
    layer.trainable = False
for layer in resnet_base.layers[fine_tune_at_resnet:]:
    layer.trainable = True
resnet_model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
history_fine_resnet = resnet_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=history_resnet.epoch[-1] + 1,
    class_weight=class_weights,
    callbacks=callbacks_resnet
)

# Fine-tune VGG16
fine_tune_at_vgg16 = 15  # VGG16 has fewer layers, adjust accordingly
for layer in vgg16_base.layers[:fine_tune_at_vgg16]:
    layer.trainable = False
for layer in vgg16_base.layers[fine_tune_at_vgg16:]:
    layer.trainable = True
vgg16_model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
history_fine_vgg16 = vgg16_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=history_vgg16.epoch[-1] + 1,
    class_weight=class_weights,
    callbacks=callbacks_vgg16
)

# Fine-tune EfficientNetB0
fine_tune_at_efficientnet = 200  # EfficientNetB0 has more layers
for layer in efficientnet_base.layers[:fine_tune_at_efficientnet]:
    layer.trainable = False
for layer in efficientnet_base.layers[fine_tune_at_efficientnet:]:
    layer.trainable = True
efficientnet_model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
history_fine_efficientnet = efficientnet_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=history_efficientnet.epoch[-1] + 1,
    class_weight=class_weights,
    callbacks=callbacks_efficientnet
)

# ===== 7. Ensemble Prediction (Soft Voting) =====
# Load best models
resnet_model = tf.keras.models.load_model('best_resnet.h5')
vgg16_model = tf.keras.models.load_model('best_vgg16.h5')
efficientnet_model = tf.keras.models.load_model('best_efficientnet.h5')

# Get predictions from each model
test_gen.reset()
pred_resnet = resnet_model.predict(test_gen)
pred_vgg16 = vgg16_model.predict(test_gen)
pred_efficientnet = efficientnet_model.predict(test_gen)

# Soft voting: Average the probabilities
ensemble_pred = (pred_resnet + pred_vgg16 + pred_efficientnet) / 3
y_pred_ensemble = np.argmax(ensemble_pred, axis=1)
y_true = test_gen.classes
labels = list(test_gen.class_indices.keys())

# Evaluate ensemble
print("\n📋 Ensemble Classification Report:")
print(classification_report(y_true, y_pred_ensemble, target_names=labels))

conf_mat = confusion_matrix(y_true, y_pred_ensemble)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Ensemble Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ===== 8. Training Curves =====
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
    ('ResNet50', history_resnet),
    ('ResNet50 Fine', history_fine_resnet),
    ('VGG16', history_vgg16),
    ('VGG16 Fine', history_fine_vgg16),
    ('EfficientNetB0', history_efficientnet),
    ('EfficientNetB0 Fine', history_fine_efficientnet)
])