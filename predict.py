#!/usr/bin/env python3
"""
Concrete Crack Detection - Prediction/Serving Script
ML Zoomcamp 2025 Capstone Project

Flask REST API for serving crack detection predictions using TFLite.
"""

import os
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "tflite_path": os.environ.get("TFLITE_PATH", "models/crack_detector.tflite"),
    "image_size": (224, 224),
    "threshold": float(os.environ.get("THRESHOLD", "0.5")),
}

# Initialize Flask app
app = Flask(__name__)

# Global model variable
interpreter = None


def load_tflite_model():
    """Load TFLite model."""
    global interpreter
    if interpreter is None:
        logger.info(f"Loading TFLite model from {CONFIG['tflite_path']}...")
        interpreter = tf.lite.Interpreter(model_path=CONFIG["tflite_path"])
        interpreter.allocate_tensors()
        logger.info("TFLite model loaded successfully!")
    return interpreter


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image for model input.

    Note: MobileNetV2 preprocessing is already included in the TFLite model,
    so we only need to resize and convert to the correct format here.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Preprocessed numpy array ready for inference (raw pixel values 0-255)
    """
    # Open image
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize
    image = image.resize(CONFIG["image_size"], Image.Resampling.LANCZOS)

    # Convert to array (keep pixel values in 0-255 range)
    img_array = np.array(image, dtype=np.float32)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # No preprocessing here - the TFLite model includes MobileNetV2 preprocessing layer
    return img_array


def predict_tflite(img_array: np.ndarray) -> float:
    """Make prediction using TFLite model."""
    interp = load_tflite_model()

    # Get input and output details
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    # Set input tensor
    interp.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interp.invoke()

    # Get prediction probability
    output = interp.get_tensor(output_details[0]['index'])
    return float(output[0][0])


def classify(probability: float, threshold: float = 0.5) -> dict:
    """
    Classify based on probability.

    Args:
        probability: Model output probability
        threshold: Classification threshold

    Returns:
        Classification result dictionary
    """
    is_crack = probability >= threshold

    return {
        "prediction": "crack" if is_crack else "no_crack",
        "probability": round(probability, 4),
        "confidence": round(probability if is_crack else 1 - probability, 4),
        "threshold": threshold,
    }


# API Routes

@app.route("/", methods=["GET"])
def home():
    """Health check and API info."""
    return jsonify({
        "service": "Concrete Crack Detection API",
        "version": "1.0.0",
        "status": "healthy",
        "model": "MobileNetV2 (TFLite)",
        "endpoints": {
            "/": "API info (GET)",
            "/health": "Health check (GET)",
            "/predict": "Make prediction (POST)",
        },
        "config": {
            "runtime": "TFLite",
            "input_size": CONFIG["image_size"],
            "threshold": CONFIG["threshold"],
        }
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Make crack detection prediction.

    Expects:
        - Image file in request.files["image"]
        - Optional: threshold in request.form["threshold"]

    Returns:
        JSON with prediction results
    """
    # Check for image
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Read image bytes
        image_bytes = file.read()

        # Preprocess
        img_array = preprocess_image(image_bytes)

        # Get threshold
        threshold = float(request.form.get("threshold", CONFIG["threshold"]))

        # Make prediction
        probability = predict_tflite(img_array)

        # Classify
        result = classify(probability, threshold)

        # Add metadata
        result["filename"] = file.filename
        result["model_type"] = "TFLite"

        logger.info(f"Prediction for {file.filename}: {result['prediction']} ({result['probability']:.4f})")

        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """
    Make predictions on multiple images.

    Expects:
        - Multiple image files in request.files
    """
    if not request.files:
        return jsonify({"error": "No images provided"}), 400

    results = []
    threshold = float(request.form.get("threshold", CONFIG["threshold"]))

    for key, file in request.files.items():
        try:
            image_bytes = file.read()
            img_array = preprocess_image(image_bytes)
            probability = predict_tflite(img_array)

            result = classify(probability, threshold)
            result["filename"] = file.filename
            results.append(result)

        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    return jsonify({"results": results, "count": len(results)})


# Error handlers

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad request", "message": str(e)}), 400


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error", "message": str(e)}), 500


if __name__ == "__main__":
    # Load model on startup
    if os.path.exists(CONFIG["tflite_path"]):
        load_tflite_model()
    else:
        logger.warning("No model file found! Please train and save a model first.")

    # Run server
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("DEBUG", "false").lower() == "true"

    logger.info(f"Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=debug)
