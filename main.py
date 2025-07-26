import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Set paths
CSV_PATH = "/Users/soumitbuddhala/Documents/eye_model/archive/train_1.csv"
IMAGE_DIR = "/Users/soumitbuddhala/Documents/eye_model/archive/train_images/train_images"

# Load the CSV file
df = pd.read_csv(CSV_PATH)

# Fix file paths (append .png to match actual filenames)
df["id_code"] = df["id_code"] + ".png"

# Convert labels to strings for categorical mode
df["diagnosis"] = df["diagnosis"].astype(str)

# Check for missing files
missing_files = [img for img in df["id_code"] if not os.path.exists(os.path.join(IMAGE_DIR, img))]
if missing_files:
    print("Warning: Some images are missing!", missing_files)

# Parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Train Generator
train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=IMAGE_DIR,
    x_col="id_code",
    y_col="diagnosis",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

# Validation Generator
val_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=IMAGE_DIR,
    x_col="id_code",
    y_col="diagnosis",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Define Model
base_model = EfficientNetB5(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(5, activation="softmax")(x)  # 5 classes (0 to 4)

model = Model(inputs=base_model.input, outputs=x)

# Compile Model
model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

# Train Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    verbose=1
)

# Plot Accuracy & Loss
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()
