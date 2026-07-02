"""
MEDICAL IMAGE CLASSIFICATION
==============================
Level: Basic to Intermediate
Goal : Classify medical images (e.g., chest X-rays) into two classes,
       such as "Normal" vs "Pneumonia" / "Abnormal", using a
       Convolutional Neural Network (CNN).

REQUIREMENTS
------------
This script uses TensorFlow/Keras. Install it first if needed:
    pip install tensorflow

NOTES ON DATA
-------------
Popular real datasets for this project:
  - Chest X-Ray Images (Pneumonia) - Kaggle
  - NIH Chest X-ray Dataset
  - ISIC Skin Cancer Images

Real image datasets are normally organized in folders like this:
    data/
      train/
        normal/
        abnormal/
      val/
        normal/
        abnormal/

To keep this script self-contained and runnable WITHOUT downloading
anything, we GENERATE synthetic grayscale "images" (random patterns
with a class-dependent signal) so you can see the full pipeline run
end-to-end. To use real data, replace the "Load / Generate Data"
section with an ImageDataGenerator pointed at your real folders
(example provided in the comments below).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------------------------------------------
# 1. LOAD / GENERATE DATA
# -----------------------------------------------------------------
IMG_SIZE = 64      # 64x64 pixels keeps this fast to train on a laptop/CPU
N_SAMPLES = 1200

rng = np.random.default_rng(42)

def make_synthetic_images(n, img_size, label):
    """
    Creates simple synthetic grayscale images.
    Class 0 ("normal")   -> mostly smooth/uniform noise
    Class 1 ("abnormal") -> noise plus a bright circular "lesion" patch,
                             loosely simulating an anomaly in a scan.
    """
    imgs = rng.normal(0.4, 0.1, size=(n, img_size, img_size, 1))
    if label == 1:
        for i in range(n):
            cx, cy = rng.integers(15, img_size - 15, size=2)
            radius = rng.integers(5, 12)
            yy, xx = np.ogrid[:img_size, :img_size]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
            imgs[i, mask, 0] += rng.uniform(0.3, 0.5)
    return np.clip(imgs, 0, 1)

normal_imgs = make_synthetic_images(N_SAMPLES // 2, IMG_SIZE, label=0)
abnormal_imgs = make_synthetic_images(N_SAMPLES // 2, IMG_SIZE, label=1)

X = np.concatenate([normal_imgs, abnormal_imgs], axis=0).astype("float32")
y = np.concatenate([
    np.zeros(N_SAMPLES // 2),
    np.ones(N_SAMPLES // 2),
]).astype("int32")

print("Image data shape:", X.shape)   # (samples, height, width, channels)
print("Labels shape:", y.shape)

# -----------------------------------------------------------------
#   HOW TO USE REAL IMAGES INSTEAD (example, commented out):
# -----------------------------------------------------------------
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
#
# train_gen = train_datagen.flow_from_directory(
#     "data/train",
#     target_size=(IMG_SIZE, IMG_SIZE),
#     color_mode="grayscale",
#     class_mode="binary",
#     subset="training",
# )
# val_gen = train_datagen.flow_from_directory(
#     "data/train",
#     target_size=(IMG_SIZE, IMG_SIZE),
#     color_mode="grayscale",
#     class_mode="binary",
#     subset="validation",
# )
# Then call model.fit(train_gen, validation_data=val_gen, epochs=10)

# -----------------------------------------------------------------
# 2. TRAIN / TEST SPLIT
# -----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------------------------------
# 3. BUILD THE CNN MODEL
# -----------------------------------------------------------------
# A CNN learns visual patterns (edges, shapes, textures) directly
# from pixel data, which is why it's the standard choice for image
# classification tasks.
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

    # First convolutional block: detects simple patterns (edges, blobs)
    layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # Second convolutional block: detects more complex combinations
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # Third convolutional block: even higher-level features
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # Flatten the 2D feature maps into a 1D vector for the dense layers
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),   # reduces overfitting by randomly disabling neurons

    # Single output neuron with sigmoid = binary classification (0 or 1)
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",   # standard loss for binary classification
    metrics=["accuracy"],
)

model.summary()

# -----------------------------------------------------------------
# 4. TRAIN THE MODEL
# -----------------------------------------------------------------
history = model.fit(
    X_train, y_train,
    validation_split=0.15,   # hold out part of training data to monitor overfitting
    epochs=10,
    batch_size=32,
    verbose=2,
)

# -----------------------------------------------------------------
# 5. EVALUATE ON THE TEST SET
# -----------------------------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.3f}")
print(f"Test loss: {test_loss:.3f}")

y_pred_probs = model.predict(X_test, verbose=0)
y_pred = (y_pred_probs > 0.5).astype(int).ravel()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Abnormal"], digits=3))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print("                Predicted Normal  Predicted Abnormal")
print(f"Actual Normal        {cm[0][0]:<17} {cm[0][1]}")
print(f"Actual Abnormal      {cm[1][0]:<17} {cm[1][1]}")

# -----------------------------------------------------------------
# 6. PREDICT ON A SINGLE NEW IMAGE (example)
# -----------------------------------------------------------------
sample_image = X_test[0:1]   # keep batch dimension
prediction = model.predict(sample_image, verbose=0)[0][0]
label = "Abnormal" if prediction > 0.5 else "Normal"
print(f"\nExample prediction: {label} (confidence: {prediction:.2%})")

# -----------------------------------------------------------------
# 7. SAVE THE TRAINED MODEL
# -----------------------------------------------------------------
model.save("medical_image_classifier.keras")
print("\nModel saved to medical_image_classifier.keras")

# -----------------------------------------------------------------
# 8. NEXT STEPS (ideas to extend this project)
# -----------------------------------------------------------------
# - Swap in a real dataset (e.g., Kaggle Chest X-Ray Pneumonia) using
#   the ImageDataGenerator example above.
# - Use data augmentation (rotation, zoom, flip) to improve generalization.
# - Try transfer learning with a pretrained model like MobileNetV2 or
#   ResNet50 for much higher accuracy with less training data.
# - Plot training/validation accuracy and loss curves (history.history)
#   to visually check for overfitting.
