from flask import Flask, request, jsonify
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from inference import predict

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    if "image" not in request.files:
        return jsonify({"error": "Campo 'image' requerido en multipart/form-data"}), 400
    file = request.files["image"]
    result = predict(file.read())
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
