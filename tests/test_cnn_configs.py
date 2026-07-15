import sys
import os
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.config import TRAINING_DIR, CLASSES, IMG_SIZE, BATCH_SIZE, SEED
from src.data.loader import load_image_paths, load_and_preprocess_image
from src.data.preprocessing import encode_labels, split_dataset
from src.data.augmentation import get_tf_generators

AUTOTUNE = tf.data.AUTOTUNE

def build_arch_v0():
    """Original architecture (baseline)"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*IMG_SIZE, 1)),
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_arch_v1():
    """V1: 2 convs per block, more filters, less dropout"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*IMG_SIZE, 1)),
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-4),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_arch_v2():
    """V2: Higher LR, simpler arch, Flatten"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*IMG_SIZE, 1)),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_arch_v3():
    """V3: He init, kernel_regularizer=l2, more conservative"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*IMG_SIZE, 1)),
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same',
                                kernel_initializer='he_normal'),
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same',
                                kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same',
                                kernel_initializer='he_normal'),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same',
                                kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same',
                                kernel_initializer='he_normal'),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same',
                                kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same',
                                kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_arch_v4():
    """V4: No augmentation test (raw data only) to isolate the issue"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*IMG_SIZE, 1)),
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_arch_v5():
    """V5: SGD + Momentum (classic, often works well for CNNs from scratch)"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*IMG_SIZE, 1)),
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-2, momentum=0.9),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_arch_v6():
    """V6: AdamW with weight decay, 2 conv per block, higher capacity"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*IMG_SIZE, 1)),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=5e-4, weight_decay=1e-4),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def make_no_aug_dataset(X, y, batch_size=BATCH_SIZE):
    """Creates a simple dataset WITHOUT augmentation (for testing)"""
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(buffer_size=len(X), seed=SEED)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds

from src.models.cnn_custom import build_cnn_custom as build_arch_v7

ARCHITECTURES = {
    "v0_original_baseline": build_arch_v0,
    "v1_2conv_block_less_dropout": build_arch_v1,
    "v2_flatten_higher_lr": build_arch_v2,
    "v3_he_init_l2": build_arch_v3,
    "v4_no_dropout_test": build_arch_v4,
    "v5_sgd_momentum": build_arch_v5,
    "v6_adamw_high_capacity": build_arch_v6,
    "v7_improved_from_src": build_arch_v7,
}

def load_data(subset_frac=1.0):
    print("Cargando imágenes...")
    train_paths, train_labels = load_image_paths(TRAINING_DIR)
    X, y = [], []
    for path, label in zip(train_paths, train_labels):
        img = load_and_preprocess_image(path, IMG_SIZE)
        if img is not None:
            X.append(img)
            y.append(label)
    X = np.array(X)
    y = encode_labels(np.array(y))

    if subset_frac < 1.0:
        from sklearn.model_selection import train_test_split
        _, X, _, y = train_test_split(X, y, test_size=subset_frac, random_state=SEED, stratify=y)
        print(f"Usando subconjunto: {X.shape[0]} imágenes")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def test_architecture(name, build_fn, X_train, y_train, X_val, y_val, use_aug=True, epochs=1):
    print(f"\n{'='*60}")
    print(f"TESTEANDO: {name}")
    print(f"{'='*60}")

    model = build_fn()
    model.summary()

    if use_aug:
        from src.data.augmentation import create_train_dataset, create_val_dataset
        train_ds = create_train_dataset(X_train, y_train, BATCH_SIZE)
        val_ds = create_val_dataset(X_val, y_val, BATCH_SIZE)
    else:
        train_ds = make_no_aug_dataset(X_train, y_train, BATCH_SIZE)
        val_ds = make_no_aug_dataset(X_val, y_val, BATCH_SIZE)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    final_val_acc = history.history['val_accuracy'][-1]
    final_train_acc = history.history['accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]

    result = {
        'name': name,
        'train_acc': final_train_acc,
        'val_acc': final_val_acc,
        'val_loss': final_val_loss,
    }

    if final_val_acc >= 0.79:
        result['veredict'] = 'EXCELENTE (umbral superado)'
    elif final_val_acc >= 0.50:
        result['veredict'] = 'BUENO (aprendiendo bien)'
    elif final_val_acc >= 0.35:
        result['veredict'] = 'PROMETEDOR (aprendiendo)'
    elif final_val_acc > 0.26:
        result['veredict'] = 'MEJORABLE (ligero progreso)'
    else:
        result['veredict'] = 'MALO (no aprende)'

    print(f"\n>>> RESULTADO [{name}]:")
    print(f"    Train Acc: {final_train_acc:.4f}")
    print(f"    Val Acc:   {final_val_acc:.4f}")
    print(f"    Val Loss:  {final_val_loss:.4f}")
    print(f"    VEREDICTO: {result['veredict']}")

    return result

def run_fast_test():
    """Quick test with a small subset to find promising configs fast"""
    print("="*60)
    print("MODO: TEST RÁPIDO (subconjunto de datos)")
    print(f"TOTAL arquitecturas a testear: {len(ARCHITECTURES)}")
    print("="*60)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(subset_frac=0.3)
    results = []

    for name, build_fn in ARCHITECTURES.items():
        try:
            r = test_architecture(name, build_fn, X_train, y_train, X_val, y_val, use_aug=False, epochs=1)
            results.append(r)
        except Exception as e:
            print(f"\n[ERROR] {name} falló: {e}")
            import traceback
            traceback.print_exc()

    print("\n\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    results.sort(key=lambda x: x['val_acc'], reverse=True)
    for i, r in enumerate(results):
        print(f"{i+1}. {r['name']:40s} | val_acc={r['val_acc']:.4f} | {r['veredict']}")

    return results

def run_full_test():
    """Full test with all data"""
    print("="*60)
    print("MODO: TEST COMPLETO (todos los datos)")
    print(f"TOTAL arquitecturas a testear: {len(ARCHITECTURES)}")
    print("="*60)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(subset_frac=1.0)
    results = []

    for name, build_fn in ARCHITECTURES.items():
        try:
            r = test_architecture(name, build_fn, X_train, y_train, X_val, y_val, use_aug=True, epochs=1)
            results.append(r)
        except Exception as e:
            print(f"\n[ERROR] {name} falló: {e}")
            import traceback
            traceback.print_exc()

    print("\n\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    results.sort(key=lambda x: x['val_acc'], reverse=True)
    for i, r in enumerate(results):
        print(f"{i+1}. {r['name']:40s} | val_acc={r['val_acc']:.4f} | {r['veredict']}")

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test CNN architectures for brain tumor classification")
    parser.add_argument("--mode", choices=["fast", "full"], default="fast",
                        help="'fast' (30% datos, sin augment) o 'full' (100% datos, con augment)")
    parser.add_argument("--arch", type=str, default=None,
                        help="Nombre de arquitectura específica a testear (opcional)")
    args = parser.parse_args()

    if args.arch:
        if args.arch not in ARCHITECTURES:
            print(f"Error: '{args.arch}' no encontrada. Opciones: {list(ARCHITECTURES.keys())}")
            sys.exit(1)
        print(f"Testeando solo: {args.arch}")
        X_train, X_val, _, y_train, y_val, _ = load_data(subset_frac=0.3 if args.mode == "fast" else 1.0)
        use_aug = (args.mode == "full")
        test_architecture(args.arch, ARCHITECTURES[args.arch], X_train, y_train, X_val, y_val, use_aug=use_aug, epochs=1)
    elif args.mode == "fast":
        run_fast_test()
    else:
        run_full_test()
