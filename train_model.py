import os
import json
import tensorflow as tf

IMG_SIZE = 224
BATCH_SIZE = 16

EPOCHS_HEAD = 10         # train top layers
EPOCHS_FINE = 10         # fine-tune last layers

DATASET_PATH = "../dataset"
MODEL_OUT = "../model/waste_classifier.h5"
LABELS_OUT = "../model/class_names.json"

# IMPORTANT: explicit class order (must match folder names)
CLASS_NAMES = ["hazardous", "organic", "recyclable"]

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
MobileNetV2 = tf.keras.applications.MobileNetV2
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Data augmentation + MobileNetV2 preprocessing
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10,
    horizontal_flip=True,
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    classes=CLASS_NAMES,                 # fixes label index order [web:549]
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    classes=CLASS_NAMES,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

print("class_indices:", train_data.class_indices)

# Save class names for inference
os.makedirs(os.path.dirname(LABELS_OUT), exist_ok=True)
with open(LABELS_OUT, "w") as f:
    json.dump(CLASS_NAMES, f)

# Base model
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)  # keep BN in inference mode while frozen [web:557]
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_data, validation_data=val_data, epochs=EPOCHS_HEAD)

# -------- Fine-tuning (unfreeze last N layers) --------
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 30  # unfreeze last ~30 layers
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_data, validation_data=val_data, epochs=EPOCHS_FINE)

# Save final model
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
model.save(MODEL_OUT)
print("✅ Model trained and saved:", MODEL_OUT)
print("✅ Class names saved:", LABELS_OUT)
