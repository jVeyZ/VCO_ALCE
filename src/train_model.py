import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models # type: ignore
import numpy as np
import os
from pathlib import Path

# Constants
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "emnist_model.h5"

def load_data():
    print("Loading EMNIST data from tensorflow_datasets...")
    # Load 'balanced' split
    ds_train, ds_test = tfds.load('emnist/balanced', split=['train', 'test'], as_supervised=True, shuffle_files=True)
    
    def preprocess(image, label):
        # EMNIST images in TFDS are rotated 90 degrees and flipped.
        # We need to transpose them to match standard orientation if we want to visualize,
        # but for training, as long as inference does the same, it's fine.
        # However, our inference code (processV2.py) uses standard OpenCV images (upright).
        # So we MUST transpose the training images to match upright orientation.
        # EMNIST mapping: 
        # The images are flipped and rotated.
        # Usually: tf.transpose(image, perm=[1, 0, 2])
        
        image = tf.transpose(image, perm=[1, 0, 2])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    # Batch and prefetch
    ds_train = ds_train.map(preprocess).batch(64).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.map(preprocess).batch(64).prefetch(tf.data.AUTOTUNE)
    
    return ds_train, ds_test

def create_model():
    # We need to handle 47 classes for 'balanced'
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(47, activation='softmax') 
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(parents=True)
        
    ds_train, ds_test = load_data()
    
    model = create_model()
    
    print("Training model...")
    model.fit(ds_train, epochs=5, validation_data=ds_test)
    
    print(f"Saving model to {MODEL_PATH}...")
    model.save(str(MODEL_PATH))
    print("Done.")

if __name__ == "__main__":
    main()
