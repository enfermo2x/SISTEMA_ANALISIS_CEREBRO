import os
import cv2
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = (128, 128)

def load_keras_model(path):
    if os.path.isdir(path):
        tmp = path + "_tmp"
        os.rename(path, tmp)
        try:
            model = tf.keras.models.load_model(tmp)
        except Exception as e:
            os.rename(tmp, path)
            raise e
        os.rename(tmp, path)
        return model
    return tf.keras.models.load_model(path)

_model = None

def get_model():
    global _model
    if _model is None:
        model_path = os.path.join(MODELS_DIR, "cnn_custom.keras")
        _model = load_keras_model(model_path)
    return _model

def preprocess_image(image_bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def predict(image_bytes):
    model = get_model()
    img = preprocess_image(image_bytes)
    if img is None:
        return {"error": "No se pudo decodificar la imagen"}
    probs = model.predict(img, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "tipo_tumor": CLASSES[idx],
        "probabilidad": float(probs[idx]),
        "probabilidades": {cls: float(p) for cls, p in zip(CLASSES, probs)}
    }
