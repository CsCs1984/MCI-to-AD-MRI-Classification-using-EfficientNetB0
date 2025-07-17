"""
MCI to AD MRI Classification using EfficientNetB0
Author: Samah Abuayeid
Description:
- Classifies MRI scans into 4 cognitive stages: EMCI, LMCI, MCI, AD
- Uses Transfer Learning (EfficientNetB0)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ==============================
# 1. PARAMETERS
# ==============================
DATA_DIR = "mci_to_ad_dataset/train_sorted"  # مسار البيانات
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# ==============================
# 2. DATA LOADING
# ==============================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

class_names = list(train_generator.class_indices.keys())
print(f"✅ Classes: {class_names}")

# ==============================
# 3. MODEL BUILDING
# ==============================
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # تجميد الطبقات الأساسية

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ==============================
# 4. TRAINING
# ==============================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# ==============================
# 5. VISUALIZE TRAINING
# ==============================
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss")

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()

# ==============================
# 6. EVALUATION
# ==============================
val_generator.reset()
preds = model.predict(val_generator)
y_pred = np.argmax(preds, axis=1)
y_true = val_generator.classes

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# ==============================
# 7. SAMPLE PREDICTIONS
# ==============================
x_batch, y_batch = next(val_generator)
sample_preds = model.predict(x_batch)
pred_labels = np.argmax(sample_preds, axis=1)
true_labels = np.argmax(y_batch, axis=1)

plt.figure(figsize=(14, 8))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(x_batch[i])
    plt.axis("off")
    true_class = class_names[true_labels[i]]
    pred_class = class_names[pred_labels[i]]
    color = "green" if true_class == pred_class else "red"
    plt.title(f"True: {true_class}\nPred: {pred_class}", color=color)

plt.tight_layout()
plt.savefig("sample_predictions.png")
plt.show()

# ==============================
# 8. SAVE MODEL
# ==============================
model.save("mci_to_ad_efficientnet.h5")
print("✅ Model saved as mci_to_ad_efficientnet.h5")
