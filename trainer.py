import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === Paths ===
TRAINING_DIR = "reviewed_images"
MODEL_OUTPUT_DIR = "assets"
MODEL_NAME_KERAS = "model.keras" # Changed from H5/TFLite to the new Keras format
LABELS_PATH = os.path.join(MODEL_OUTPUT_DIR, "labels.txt")

# === Parameters ===
IMG_SIZE = (224, 224)
BATCH_SIZE = 2 # Lowered batch size for fine-tuning with fewer images
EPOCHS = 10  # Increased epochs slightly for better fine-tuning

def train_model():
    print("Clearing session...")
    tf.keras.backend.clear_session()

    print("Preparing data...")
    # Using ImageDataGenerator for loading from directories
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2 # Use 20% of data for validation
    )

    # Check if the training directory exists and has subdirectories
    if not os.path.exists(TRAINING_DIR) or not any(os.scandir(TRAINING_DIR)):
        print(f"Error: Training directory '{TRAINING_DIR}' is empty or does not exist.")
        return

    try:
        train_generator = datagen.flow_from_directory(
            TRAINING_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="binary",
            classes=['normal', 'dangerous'], # Force 'normal' to be 0, 'dangerous' to be 1
            subset="training"
        )

        val_generator = datagen.flow_from_directory(
            TRAINING_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="binary",
            classes=['normal', 'dangerous'], # Force 'normal' to be 0, 'dangerous' to be 1
            subset="validation"
        )
    except ValueError as e:
        print(f"Error creating data generators: {e}")
        print("Please ensure you have classified at least one image for both 'normal' and 'dangerous'.")
        return

    print("Building model...")
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("Training model...")
    if val_generator.n > 0:
        model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=EPOCHS
        )
    else:
        print("Validation set is empty, training without validation.")
        model.fit(
            train_generator,
            epochs=EPOCHS
        )

    # === Save Keras model ===
    # We are now saving in the modern .keras format, which is a single file.
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    keras_path = os.path.join(MODEL_OUTPUT_DIR, MODEL_NAME_KERAS)
    model.save(keras_path)
    print(f"Model saved in Keras format: {keras_path}")

    # === Save labels.txt ===
    # Match the class indices found by flow_from_directory
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    with open(LABELS_PATH, "w") as f:
        for i in range(len(labels)):
            f.write(f"{i} {labels[i]}\n")
    print(f"Labels saved to: {LABELS_PATH}")

    print("Training pipeline complete.")

if __name__ == "__main__":
    train_model()
