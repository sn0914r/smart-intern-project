import os, json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
DATA_DIR = "Data/train"
labels = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

# Step 1: Collect valid image paths and labels
image_paths, image_labels = [], []
for label in labels:
    folder = os.path.join(DATA_DIR, label)
    for file in os.listdir(folder):
        if file.lower().endswith(('.jpg', '.png', '.jpeg', '.webp')):
            full_path = os.path.join(folder, file)
            try:
                with Image.open(full_path) as img:
                    img.verify()
                image_paths.append(full_path)
                image_labels.append(label)
            except Exception:
                print(f"❌ Skipped invalid image: {full_path}")

# Step 2: Build dataframe
df = pd.DataFrame({"path": image_paths, "label": image_labels})
print(f"✅ Total valid images: {len(df)} in {len(set(df['label']))} classes")

# Step 3: Check stratification
label_counts = df['label'].value_counts()
stratify = df['label'] if all(label_counts >= 2) else None
if stratify is None:
    print("⚠️ Stratify disabled due to low samples in some classes")

# Step 4: Train-validation split
train_df, val_df = train_test_split(df, test_size=0.33, stratify=stratify, random_state=42)

# Step 5: Synchronize validation classes with training
common_classes = sorted(train_df['label'].unique())
val_df = val_df[val_df['label'].isin(common_classes)].reset_index(drop=True)

# Step 6: Define generators
train_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, horizontal_flip=True)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_dataframe(
    train_df, x_col='path', y_col='label',
    classes=common_classes,
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=True
)

val_data = val_gen.flow_from_dataframe(
    val_df, x_col='path', y_col='label',
    classes=common_classes,
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

# Confirm classes
print("✅ Class indices:", train_data.class_indices)
print("✅ Training on:", train_data.samples, "samples")
print("✅ Validating on:", val_data.samples, "samples")

# Step 7: Build model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(train_data.class_indices), activation='softmax')
])

model.compile(optimizer=optimizers.Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 8: Train
os.makedirs("saved_models", exist_ok=True)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[early_stop]
)

# Step 9: Save model and label map
model.save("saved_models/model_cnn.h5")
json.dump(train_data.class_indices, open("saved_models/label_map.json", "w"))
print("✅ Model and label map saved in 'saved_models/'")