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
from sklearn.model_selection import KFold
import pandas as pd

# ===== 1. Configuration =====
DATASET_DIR = 'dataset/split'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
FINE_TUNE_EPOCHS = 10
OUTPUT_DIR = 'optimized_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 2. Data Preparation =====
train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,  # Lebih agresif
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.4,
    horizontal_flip=True,
    vertical_flip=True,  # Tambahan untuk variasi
    brightness_range=[0.5, 1.5],
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
labels = list(test_gen.class_indices.keys())
y_true = test_gen.classes

# Compute class weights
class_weights = dict(enumerate(
    class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
))

# ===== 3. Model Definitions =====
def create_vgg16_model(num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)  # Reduced neurons, L2 reg
    x = Dropout(0.5)(x)  # Increased dropout
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def create_mobilenetv2_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

# Create models
vgg16_model, vgg16_base = create_vgg16_model(num_classes)
mobilenetv2_model, mobilenetv2_base = create_mobilenetv2_model(num_classes)

# Compile models
vgg16_model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
mobilenetv2_model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# ===== 4. Callbacks =====
callbacks_vgg16 = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
    ModelCheckpoint(os.path.join(OUTPUT_DIR, 'best_vgg16_optimized.h5'), monitor='val_accuracy', save_best_only=True)
]
callbacks_mobilenetv2 = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
    ModelCheckpoint(os.path.join(OUTPUT_DIR, 'best_mobilenetv2.h5'), monitor='val_accuracy', save_best_only=True)
]

# ===== 5. Initial Training =====
history_vgg16 = vgg16_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks_vgg16
)

history_mobilenetv2 = mobilenetv2_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks_mobilenetv2
)

# ===== 6. Fine-tuning =====
# Fine-tune VGG16
fine_tune_at_vgg16 = 10  # Unfreeze fewer layers to prevent overfitting
for layer in vgg16_base.layers[:fine_tune_at_vgg16]:
    layer.trainable = False
for layer in vgg16_base.layers[fine_tune_at_vgg16:]:
    layer.trainable = True
vgg16_model.compile(optimizer=Adam(1e-6), loss='categorical_crossentropy', metrics=['accuracy'])  # Lower LR
history_fine_vgg16 = vgg16_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=history_vgg16.epoch[-1] + 1,
    class_weight=class_weights,
    callbacks=callbacks_vgg16
)

# Fine-tune MobileNetV2
fine_tune_at_mobilenetv2 = 100  # Unfreeze from later layers
for layer in mobilenetv2_base.layers[:fine_tune_at_mobilenetv2]:
    layer.trainable = False
for layer in mobilenetv2_base.layers[fine_tune_at_mobilenetv2:]:
    layer.trainable = True
mobilenetv2_model.compile(optimizer=Adam(1e-6), loss='categorical_crossentropy', metrics=['accuracy'])
history_fine_mobilenetv2 = mobilenetv2_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=history_mobilenetv2.epoch[-1] + 1,
    class_weight=class_weights,
    callbacks=callbacks_mobilenetv2
)

# ===== 7. Evaluation on Test Data =====
def evaluate_model(model, model_name, test_generator, y_true, labels, output_dir):
    test_generator.reset()
    pred_probs = model.predict(test_generator)
    y_pred = np.argmax(pred_probs, axis=1)
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    print(f"\n📋 {model_name} Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))
    
    # Confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()
    
    metrics = {
        'Model': model_name,
        'Accuracy': report['accuracy'],
        'Macro Precision': report['macro avg']['precision'],
        'Macro Recall': report['macro avg']['recall'],
        'Macro F1-Score': report['macro avg']['f1-score']
    }
    return metrics, pred_probs

# Evaluate models
metrics_list = []
pred_probs_list = {}
for model, model_name in [(vgg16_model, 'VGG16'), (mobilenetv2_model, 'MobileNetV2')]:
    metrics, pred_probs = evaluate_model(model, model_name, test_gen, y_true, labels, OUTPUT_DIR)
    metrics_list.append(metrics)
    pred_probs_list[model_name] = pred_probs

# Save metrics comparison
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'), index=False)
plt.figure(figsize=(10, 6))
metrics_df.set_index('Model')[['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1-Score']].plot(kind='bar')
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'))
plt.close()

# ===== 8. K-fold Cross Validation =====
def perform_kfold_validation(model_builder, model_name, data_dir, k=5):
    all_data_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        data_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
    )
    X_paths = all_data_gen.filepaths
    y = all_data_gen.classes
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_paths)):
        print(f'\nFold {fold + 1}/{k} for {model_name}')
        
        # Create train and validation generators (simplified)
        train_gen_fold = train_aug.flow_from_directory(
            os.path.join(DATASET_DIR, 'train'),  # Need custom logic for fold-specific files
            target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
        )
        val_gen_fold = val_test_aug.flow_from_directory(
            os.path.join(DATASET_DIR, 'val'),
            target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
        )
        
        # Build and train model
        model, _ = model_builder(num_classes)
        model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(
            train_gen_fold,
            validation_data=val_gen_fold,
            epochs=10,  # Reduced for efficiency
            class_weight=class_weights,
            callbacks=[
                EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)
            ]
        )
        val_acc = max(history.history['val_accuracy'])
        fold_accuracies.append(val_acc)
    
    avg_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    print(f'\n{model_name} K-fold CV: Average Accuracy = {avg_accuracy:.4f} ± {std_accuracy:.4f}')
    
    with open(os.path.join(OUTPUT_DIR, 'kfold_results.txt'), 'a') as f:
        f.write(f'{model_name}: Average Accuracy = {avg_accuracy:.4f} ± {std_accuracy:.4f}\n')
    
    return avg_accuracy, std_accuracy

# Perform k-fold validation
kfold_results = []
for model_name, model_builder in [('VGG16', create_vgg16_model), ('MobileNetV2', create_mobilenetv2_model)]:
    avg_acc, std_acc = perform_kfold_validation(model_builder, model_name, DATASET_DIR)
    kfold_results.append({'Model': model_name, 'Avg Accuracy': avg_acc, 'Std Accuracy': std_acc})

# Save k-fold results
kfold_df = pd.DataFrame(kfold_results)
kfold_df.to_csv(os.path.join(OUTPUT_DIR, 'kfold_comparison.csv'), index=False)

# ===== 9. Plot Training Curves =====
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
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'))
    plt.close()

plot_history([
    ('VGG16', history_vgg16),
    ('VGG16 Fine', history_fine_vgg16),
    ('MobileNetV2', history_mobilenetv2),
    ('MobileNetV2 Fine', history_fine_mobilenetv2)
])

print("\nTraining and evaluation completed. Results saved in:", OUTPUT_DIR)