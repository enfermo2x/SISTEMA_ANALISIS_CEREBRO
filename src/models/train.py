import os
import tensorflow as tf
from src.utils.config import MODELS_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE

def train_model(model, train_ds, val_ds, epochs=EPOCHS, train_size=None):
    if train_size is None:
        train_size = BATCH_SIZE * 10
    steps_per_epoch = max(1, train_size // BATCH_SIZE)
    total_steps = steps_per_epoch * epochs

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=total_steps,
        alpha=0.01
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    checkpoint_path = os.path.join(MODELS_DIR, 'cnn_custom.keras')
    if os.path.exists(checkpoint_path):
        print(f"⚠️  Sobrescribiendo modelo existente: {checkpoint_path}")
        os.remove(checkpoint_path)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            min_delta=0.005,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    return history
