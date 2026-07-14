import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils.config import IMG_SIZE, IMG_SIZE_150, SEED, VALIDATION_SPLIT, TEST_SPLIT, CLASSES, CLASS_MAP

def resize_image(img, target_size=IMG_SIZE):
    return cv2.resize(img, target_size)

def normalize_image(img):
    return img.astype(np.float32) / 255.0

def standardize_image(img, target_size=IMG_SIZE):
    img = resize_image(img, target_size)
    img = normalize_image(img)
    return img

def preprocess_pipeline(images, labels, target_size=IMG_SIZE):
    processed = []
    for img in images:
        processed.append(standardize_image(img, target_size))
    return np.array(processed), np.array(labels)

def encode_labels(labels):
    return np.array([CLASS_MAP[l] for l in labels])

def split_dataset(images, labels, val_split=VALIDATION_SPLIT, test_split=TEST_SPLIT, seed=SEED):
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=(val_split + test_split),
        random_state=seed, stratify=labels
    )
    val_ratio = val_split / (val_split + test_split)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio),
        random_state=seed, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def decode_predictions(preds):
    return [CLASSES[i] for i in np.argmax(preds, axis=1)]
