# AUGMENT=0 DROPOUT=0.3 L2_COEF=1e-4 EPOCHS=10 BATCH_SIZE=64 python src/train_model.py      

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models, regularizers # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Constants
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "emnist_model.h5"

def load_data(batch_size=64):
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

    # Simple augmentation pipeline (applied only to training)
    # Default AUGMENT=0 (off) because heavy augmentation reduced peak val accuracy in earlier experiments.
    AUGMENT = os.environ.get('AUGMENT', '0') not in ('0', 'false', 'False')
    if AUGMENT:
        print("Data augmentation: ON")
        data_augmentation = tf.keras.Sequential([
            layers.RandomRotation(0.08),
            layers.RandomTranslation(0.06, 0.06),
            layers.RandomZoom(0.05),
            layers.RandomContrast(0.08)
        ])

        def augment(image, label):
            image = data_augmentation(image)
            return image, label

        ds_train = ds_train.map(preprocess).map(augment).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        print("Data augmentation: OFF")
        ds_train = ds_train.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return ds_train, ds_test

def create_model():
    # Regularization hyperparameters (tunable via env)
    l2_coef = float(os.environ.get('L2_COEF', '1e-4'))
    dropout_rate = float(os.environ.get('DROPOUT', '0.3'))

    # We need to handle 47 classes for 'balanced'
    model = models.Sequential([
        layers.Input(shape=(28,28,1)),
        layers.Conv2D(32, (3, 3), activation=None, kernel_regularizer=regularizers.l2(l2_coef)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation=None, kernel_regularizer=regularizers.l2(l2_coef)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation=None, kernel_regularizer=regularizers.l2(l2_coef)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Flatten(),
        layers.Dense(128, activation=None, kernel_regularizer=regularizers.l2(l2_coef)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(47, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(parents=True)
        
    batch_size = int(os.environ.get('BATCH_SIZE', '64'))
    epochs = int(os.environ.get('EPOCHS', '50'))
    ds_train, ds_test = load_data(batch_size=batch_size)
    
    model = create_model()
    
    # Print hyperparameters for transparency
    augment_flag = os.environ.get('AUGMENT', '0') not in ('0', 'false', 'False')
    print(f"Config: AUGMENT={augment_flag}, DROPOUT={os.environ.get('DROPOUT', '0.3')}, L2_COEF={os.environ.get('L2_COEF', '1e-4')}, BATCH_SIZE={batch_size}, EPOCHS={epochs}")

    # Callbacks to prevent/mitigate overfitting and help training
    checkpoint_path = MODEL_DIR / "best_emnist_model.h5"
    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ckpt = ModelCheckpoint(str(checkpoint_path), monitor='val_loss', save_best_only=True, verbose=1)
    callbacks = [es, rlrop, ckpt]
    
    print("Training model...")
    history = model.fit(ds_train, epochs=epochs, validation_data=ds_test, callbacks=callbacks)
    
    # Save final model (best weights restored by EarlyStopping)
    print(f"Saving model to {MODEL_PATH}...")
    model.save(str(MODEL_PATH))
    print(f"Best checkpoint saved to {checkpoint_path}")
    
    # Plot training history
    try:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(history.history['accuracy'], label='train_acc')
        plt.plot(history.history['val_accuracy'], label='val_acc')
        plt.legend(); plt.title('Accuracy')
        plt.subplot(1,2,2)
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.legend(); plt.title('Loss')
        history_path = MODEL_DIR / "training_history.png"
        plt.savefig(str(history_path))
        print(f"Training plots saved to {history_path}")
    except Exception as e:
        print(f"Could not save training plot: {e}")

    # Confusion matrix on validation set (uses best checkpoint if available)
    try:
        # If a best checkpoint was saved, load it for evaluation
        if checkpoint_path.exists():
            try:
                eval_model = tf.keras.models.load_model(str(checkpoint_path))
                print(f"Loaded best checkpoint from {checkpoint_path} for evaluation")
            except Exception as e:
                print(f"Could not load checkpoint, using current model: {e}")
                eval_model = model
        else:
            eval_model = model

        # Collect predictions and true labels
        y_true = []
        y_pred = []
        for x_batch, y_batch in ds_test:
            preds = eval_model.predict(x_batch, verbose=0)
            y_pred.extend(np.argmax(preds, axis=1).tolist())
            y_true.extend(y_batch.numpy().tolist())
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Compute confusion matrix
        num_classes = 47
        cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy()

        # Plot and save confusion matrix
        plt.figure(figsize=(10,8))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Confusion matrix (validation)')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.colorbar()
        cm_path = MODEL_DIR / "confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(str(cm_path))
        print(f"Confusion matrix saved to {cm_path}")
    except Exception as e:
        print(f"Could not compute confusion matrix: {e}")

    print("Done.")

if __name__ == "__main__":
    main()