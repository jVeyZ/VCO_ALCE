import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
from tensorflow.keras import callbacks, layers, models, optimizers, regularizers  # type: ignore

# Training constants
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "emnist_model.h5"
TFDS_DATASET = "emnist/balanced"
NUM_CLASSES = 47
BATCH_SIZE = 256
SHUFFLE_BUFFER = 20_000
EPOCHS = 60
MIXUP_ALPHA = 0.3
AUTOTUNE = tf.data.AUTOTUNE
L2_FACTOR = 1e-4


def _sample_beta(alpha: float, shape):
    gamma1 = tf.random.gamma(shape, alpha, dtype=tf.float32)
    gamma2 = tf.random.gamma(shape, alpha, dtype=tf.float32)
    return gamma1 / (gamma1 + gamma2)


def _mixup_batch(images: tf.Tensor, labels: tf.Tensor):
    batch_size = tf.shape(images)[0]
    beta = _sample_beta(MIXUP_ALPHA, [batch_size, 1])
    beta = tf.maximum(beta, 1.0 - beta)

    beta_x = tf.reshape(beta, [batch_size, 1, 1, 1])
    beta_y = tf.reshape(beta, [batch_size, 1])

    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, indices)
    shuffled_labels = tf.gather(labels, indices)

    mixed_images = images * beta_x + shuffled_images * (1.0 - beta_x)
    mixed_labels = labels * beta_y + shuffled_labels * (1.0 - beta_y)
    return mixed_images, mixed_labels


def _preprocess(image, label):
    image = tf.transpose(image, perm=[1, 0, 2])  # fix EMNIST orientation
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


def load_data():
    print("Loading EMNIST data from tensorflow_datasets...")
    (ds_train, ds_test), ds_info = tfds.load(
        TFDS_DATASET,
        split=["train", "test"],
        as_supervised=True,
        shuffle_files=True,
        with_info=True,
    )

    train_examples = ds_info.splits["train"].num_examples

    train_ds = (
        ds_train
        .shuffle(SHUFFLE_BUFFER)
        .map(_preprocess, num_parallel_calls=AUTOTUNE)
        .cache()
        .batch(BATCH_SIZE, drop_remainder=True)
        .map(_mixup_batch, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    test_ds = (
        ds_test
        .map(_preprocess, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    steps_per_epoch = train_examples // BATCH_SIZE
    return train_ds, test_ds, steps_per_epoch


def _build_loss():
    try:
        return tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    except TypeError:
        print("[WARN] TensorFlow build lacks label_smoothing on CategoricalCrossentropy; disabling it.")
        return tf.keras.losses.CategoricalCrossentropy()


def _create_optimizer(steps_per_epoch: int):
    if steps_per_epoch <= 0:
        return optimizers.Adam(learning_rate=1e-3)

    schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3,
        first_decay_steps=steps_per_epoch * 5,
        t_mul=1.5,
        m_mul=0.9,
        alpha=1e-2,
    )
    return optimizers.Adam(learning_rate=schedule)


def _conv_block(filters: int, spatial_dropout: float, dropout: float):
    reg = regularizers.l2(L2_FACTOR)
    return [
        layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg),
        layers.BatchNormalization(momentum=0.95),
        layers.Activation("relu"),
        layers.Conv2D(filters, 3, padding="same", kernel_regularizer=reg),
        layers.BatchNormalization(momentum=0.95),
        layers.Activation("relu"),
        layers.SpatialDropout2D(spatial_dropout),
        layers.MaxPooling2D(),
        layers.Dropout(dropout),
    ]


def create_model(optimizer) -> tf.keras.Model:
    augmentation = tf.keras.Sequential(
        [
            layers.RandomRotation(0.2),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.15),
            layers.GaussianNoise(0.05),
        ],
        name="augmentation",
    )

    reg = regularizers.l2(L2_FACTOR)
    model_layers = [layers.Input(shape=(28, 28, 1)), augmentation]
    model_layers += _conv_block(64, spatial_dropout=0.1, dropout=0.25)
    model_layers += _conv_block(128, spatial_dropout=0.15, dropout=0.35)
    model_layers += _conv_block(256, spatial_dropout=0.2, dropout=0.45)
    model_layers += [
        layers.Conv2D(384, 3, padding="same", kernel_regularizer=reg),
        layers.BatchNormalization(momentum=0.95),
        layers.Activation("relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(512, activation="relu", kernel_regularizer=reg),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]

    model = models.Sequential(model_layers)
    model.compile(
        optimizer=optimizer,
        loss=_build_loss(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
        ],
    )
    return model


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, test_ds, steps_per_epoch = load_data()
    optimizer = _create_optimizer(steps_per_epoch)
    model = create_model(optimizer)

    print("Training model with enhanced regularization...")
    checkpoint = callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        restore_best_weights=True,
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.4,
        patience=4,
        min_lr=5e-6,
        verbose=1,
    )

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=test_ds,
        callbacks=[checkpoint, early_stop, reduce_lr],
    )

    best_val_acc = max(history.history.get("val_accuracy", [0.0]))
    print(f"Best validation accuracy: {best_val_acc:.4%}")

    print(f"Reloading best weights from {MODEL_PATH} and saving final model...")
    model.load_weights(str(MODEL_PATH))
    model.save(str(MODEL_PATH))

    eval_metrics = model.evaluate(test_ds, verbose=0)
    metric_summary = ", ".join(
        f"{name}={value:.4f}" for name, value in zip(model.metrics_names, eval_metrics)
    )
    print(f"Final evaluation on test split -> {metric_summary}")


if __name__ == "__main__":
    main()
