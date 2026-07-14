import sys, os, numpy as np, tensorflow as tf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.config import TRAINING_DIR, CLASSES, SEED
from src.data.loader import load_image_paths, load_and_preprocess_image
from src.data.preprocessing import encode_labels, split_dataset

AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = (128, 128)
BATCH_SIZE = 64

def build_small_model():
    inputs = tf.keras.Input(shape=(128, 128, 3))
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_med_model():
    inputs = tf.keras.Input(shape=(128, 128, 3))
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def augment(x, y):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.95, 1.05)
    return x, y

def make_ds(X, y, batch_size=BATCH_SIZE, do_aug=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if do_aug:
        ds = ds.shuffle(buffer_size=len(X), seed=SEED)
        ds = ds.batch(batch_size)
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds

print("Cargando imagenes (128x128)...")
train_paths, train_labels = load_image_paths(TRAINING_DIR)
X, y = [], []
for path, label in zip(train_paths, train_labels):
    img = load_and_preprocess_image(path, IMG_SIZE)
    if img is not None:
        X.append(img)
        y.append(label)
X = np.array(X)
y = encode_labels(np.array(y))
print(f"Cargadas: {X.shape}, clases: {np.bincount(y)}")

X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

train_ds = make_ds(X_train, y_train, do_aug=True)
val_ds = make_ds(X_val, y_val, do_aug=False)

for name, build_fn in [("small_no_bn", build_small_model), ("med_with_bn", build_med_model)]:
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    model = build_fn()
    model.summary()
    h = model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)
    print(f"\n>>> {name}: train_acc={h.history['accuracy'][-1]:.4f}, val_acc={h.history['val_accuracy'][-1]:.4f}")
