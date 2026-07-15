import os
import cv2
import numpy as np
from src.utils.config import TRAINING_DIR, TESTING_DIR, CLASSES, IMG_SIZE, SEED

def load_image_paths(data_dir):
    paths = []
    labels = []
    for cls in CLASSES:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(cls)
    return paths, labels

def count_images_per_class(data_dir):
    counts = {}
    for cls in CLASSES:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            counts[cls] = 0
            continue
        files = [f for f in os.listdir(cls_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        counts[cls] = len(files)
    return counts

def load_and_preprocess_image(path, target_size=IMG_SIZE):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.stack([img, img, img], axis=-1)
    return img

def load_dataset(data_dir, target_size=IMG_SIZE):
    paths, labels = load_image_paths(data_dir)
    images = []
    valid_labels = []
    for path, label in zip(paths, labels):
        img = load_and_preprocess_image(path, target_size)
        if img is not None:
            images.append(img)
            valid_labels.append(label)
    return np.array(images), np.array(valid_labels)

def check_corrupted_images(data_dir):
    corrupted = []
    for cls in CLASSES:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(cls_dir, fname)
            img = cv2.imread(path)
            if img is None:
                corrupted.append(path)
    return corrupted
