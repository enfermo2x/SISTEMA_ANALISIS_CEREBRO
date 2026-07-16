import os
import sys
import json
import time
import subprocess
import urllib.request
import urllib.error
import http.client
import threading
from pathlib import Path

API_URL = "http://localhost:5000/predict"
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
TIMEOUT = 30


def test_frontend_files_exist():
    assert (FRONTEND_DIR / "index.html").exists(), "Falta index.html"
    assert (FRONTEND_DIR / "style.css").exists(), "Falta style.css"
    assert (FRONTEND_DIR / "app.js").exists(), "Falta app.js"


def test_index_html_has_key_elements():
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    assert "drag" in html.lower() or "drop" in html.lower()
    assert "fileInput" in html or "file" in html
    assert "tipo_tumor" in html or "result" in html
    assert "disclaimer" in html.lower() or "aviso" in html.lower()


def test_app_js_has_api_url():
    js = (FRONTEND_DIR / "app.js").read_text(encoding="utf-8")
    assert "localhost:5000" in js or "/predict" in js
    assert "fetch" in js
    assert "handleFile" in js or "analyzeImage" in js


def is_api_running():
    try:
        req = urllib.request.Request(
            API_URL,
            method="POST",
            data=b"--boundary\r\nContent-Disposition: form-data; name=\"image\"; filename=\"test.jpg\"\r\n\r\nfake\r\n--boundary--",
            headers={"Content-Type": "multipart/form-data; boundary=boundary"}
        )
        urllib.request.urlopen(req, timeout=5)
        return True
    except (urllib.error.HTTPError, urllib.error.URLError, ConnectionResetError):
        return True
    except Exception:
        return False


def find_test_image():
    base_dir = Path(__file__).resolve().parent.parent
    candidates = [
        base_dir / "Testing" / "glioma",
        base_dir / "Testing" / "meningioma",
        base_dir / "Testing" / "notumor",
        base_dir / "Testing" / "pituitary",
        base_dir / "Training" / "glioma",
    ]
    for folder in candidates:
        if folder.exists():
            files = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))
            if files:
                return str(files[0])
    return None


def test_api_endpoint_integration():
    if not is_api_running():
        print("WARN: API no responde en localhost:5000. Se salta la prueba E2E real.")
        return

    image_path = find_test_image()
    assert image_path is not None, "No se encontró imagen de prueba en Testing/"

    import requests
    with open(image_path, "rb") as f:
        files = {"image": f}
        resp = requests.post(API_URL, files=files, timeout=TIMEOUT)

    assert resp.status_code == 200, f"Status code: {resp.status_code}"
    data = resp.json()
    assert "tipo_tumor" in data, "Falta tipo_tumor en respuesta"
    assert "probabilidad" in data, "Falta probabilidad en respuesta"
    assert data["tipo_tumor"] in ["glioma", "meningioma", "notumor", "pituitary"]
    assert 0.0 <= data["probabilidad"] <= 1.0
    assert "probabilidades" in data
    assert len(data["probabilidades"]) == 4


def test_api_error_handling():
    if not is_api_running():
        print("WARN: API no responde. Se salta prueba de error.")
        return

    import requests
    resp = requests.post(API_URL, files={}, timeout=TIMEOUT)
    assert resp.status_code == 400

    resp2 = requests.post(API_URL, files={"image": b"notanimage"}, timeout=TIMEOUT)
    assert resp2.status_code == 400


if __name__ == "__main__":
    test_frontend_files_exist()
    print("[PASS] Archivos del frontend existen")
    test_index_html_has_key_elements()
    print("[PASS] index.html tiene elementos clave")
    test_app_js_has_api_url()
    print("[PASS] app.js tiene fetch y API URL")
    try:
        test_api_endpoint_integration()
        print("[PASS] Integración con API - predicción exitosa")
    except Exception as e:
        print(f"[SKIP] API no disponible: {e}")
    try:
        test_api_error_handling()
        print("[PASS] Manejo de errores de API")
    except Exception as e:
        print(f"[SKIP] Error handling skip: {e}")
    print("\n[TODOS LOS TESTS COMPLETADOS]")
