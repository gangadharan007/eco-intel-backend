import os
import logging
import traceback
import io

from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
from PIL import Image
import requests

# TensorFlow is heavy; keep import but avoid loading model at import time
import tensorflow as tf


# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------- App ----------------
app = Flask(__name__)

# Allow all origins (simple for hackathon demo). You can restrict later.
CORS(app)


# ---------------- Config ----------------
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "dfa03052a6ec019c943a7889fb20b8fe")

MODEL_PATH = os.environ.get("MODEL_PATH", "model/waste_classifier.h5")
CLASS_NAMES = ["hazardous", "organic", "recyclable"]

# Lazy-loaded model cache
_model = None
_model_load_error = None


# ---------------- Helpers ----------------
def get_db_connection():
    """
    Non-blocking DB getter for serverless:
    if db.py or MySQL env is missing, it won't crash the function.
    """
    try:
        from db import get_connection
        return get_connection()
    except Exception as e:
        logger.error(f"DB connection failed (non-critical): {str(e)}")
        return None


def get_model():
    """
    Lazy-load TF model only when /predict is called.
    This avoids crashing the function during cold start import.
    """
    global _model, _model_load_error

    # If previously failed, don't keep retrying every request (optional behavior)
    if _model is None and _model_load_error is not None:
        return None

    if _model is None:
        try:
            logger.info(f"Loading model from: {MODEL_PATH}")
            _model = tf.keras.models.load_model(MODEL_PATH)
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            _model_load_error = e
            logger.error(f"❌ Model load failed (non-critical): {str(e)}")
            _model = None
    return _model


def apply_rules(waste_type: str, confidence: float):
    if confidence < 60:
        return {
            "status": "low_confidence",
            "message": "Image is unclear. Please upload a clearer image."
        }
    if waste_type == "organic":
        return {
            "status": "compostable",
            "message": "Waste is organic and suitable for composting."
        }
    if waste_type == "recyclable":
        return {
            "status": "recyclable",
            "message": "Waste should be sent for recycling."
        }
    return {
        "status": "hazardous",
        "message": "Waste is hazardous. Do NOT compost. Dispose safely."
    }


def get_rule_explanation(waste_type: str):
    if waste_type == "organic":
        return [
            "Natural texture patterns detected",
            "Green or brown color tones identified",
            "Biodegradable material characteristics",
        ]
    if waste_type == "recyclable":
        return [
            "Uniform surface structure detected",
            "Man-made material patterns found",
            "Common recyclable object shapes identified",
        ]
    if waste_type == "hazardous":
        return [
            "Synthetic or chemical-like appearance",
            "Sharp or unsafe material indicators",
            "Not suitable for composting",
        ]
    return []


def fetch_weather(city: str):
    try:
        url = (
            "https://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={WEATHER_API_KEY}&units=metric"
        )
        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            return None

        data = res.json()
        main = data.get("main", {})
        weather_list = data.get("weather", [{}])

        return {
            "temperature": main.get("temp"),
            "rainfall": data.get("rain", {}).get("1h", 0),
            "humidity": main.get("humidity"),
            "icon": weather_list[0].get("icon"),
            "description": weather_list[0].get("description"),
        }
    except Exception as e:
        logger.error(f"fetch_weather error: {str(e)}")
        return None


# ---------------- DB Saves (non-blocking) ----------------
def save_waste_analysis(waste_type, confidence, status, message):
    try:
        conn = get_db_connection()
        if not conn:
            return
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO waste_analysis (waste_type, confidence, status, message)
            VALUES (%s, %s, %s, %s)
            """,
            (waste_type, confidence, status, message),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"save_waste_analysis error (non-critical): {str(e)}")


def save_crop_recommendation(location, temp, rain, soil, season, crops):
    try:
        conn = get_db_connection()
        if not conn:
            return
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO crop_recommendation (location, temperature, rainfall, soil, season, recommended_crops)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (location, temp, rain, soil, season, ", ".join(crops)),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"save_crop_recommendation error (non-critical): {str(e)}")


# ---------------- Routes ----------------
@app.route("/", methods=["GET"])
def home():
    try:
        return jsonify(
            {
                "status": "Eco Intel AI Backend LIVE!",
                "endpoints": [
                    "/predict",
                    "/crop-recommend",
                    "/api/carbon-footprint",
                    "/api/profit",
                ],
                "model_loaded": get_model() is not None,
            }
        )
    except Exception as e:
        logger.error(f"Home error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict_waste():
    try:
        model = get_model()
        if model is None:
            # Don’t crash; just tell frontend model isn’t available
            return jsonify({"error": "AI model not available on server"}), 503

        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]

        # Read bytes safely
        img_bytes = file.read()
        if not img_bytes:
            return jsonify({"error": "Empty image file"}), 400

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)
        predicted_class = CLASS_NAMES[int(np.argmax(prediction))]
        confidence = round(float(np.max(prediction)) * 100, 2)

        rule_result = apply_rules(predicted_class, confidence)
        explanation = get_rule_explanation(predicted_class)

        save_waste_analysis(
            predicted_class,
            confidence,
            rule_result["status"],
            rule_result["message"],
        )

        return jsonify(
            {
                "predicted_waste_type": predicted_class,
                "confidence": confidence,
                "explanation": explanation,
                "status": rule_result["status"],
                "message": rule_result["message"],
            }
        )
    except Exception as e:
        logger.error(f"Predict error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Prediction failed"}), 500


@app.route("/crop-recommend", methods=["POST"])
def crop_recommend():
    try:
        data = request.get_json(silent=True) or {}

        location = data.get("location")
        soil = data.get("soil")
        season = data.get("season")

        if not location or not soil or not season:
            return jsonify({"error": "Missing required fields"}), 400

        weather = fetch_weather(location)
        if not weather:
            return jsonify({"error": "Invalid location"}), 400

        temperature = weather["temperature"]
        rainfall = weather["rainfall"]

        crops = []
        explanation = []

        # Simple crop logic (keep your own rules if you want)
        if rainfall is not None and rainfall > 200 and season.lower() == "kharif":
            crops.append("Rice")
            explanation.append("High rainfall during Kharif season suits rice")
        elif temperature is not None and temperature > 25 and season.lower() == "rabi":
            crops.append("Wheat")
            explanation.append("Moderate temperature suits Rabi wheat")
        else:
            crops.append("Millets")
            explanation.append("Millets are climate-resilient crops")

        save_crop_recommendation(location, temperature, rainfall, soil, season, crops)

        return jsonify(
            {
                "location": location,
                "temperature": temperature,
                "rainfall": rainfall,
                "humidity": weather["humidity"],
                "weather_icon": weather["icon"],
                "weather_desc": weather["description"],
                "recommended_crops": crops,
                "explanation": explanation,
            }
        )
    except Exception as e:
        logger.error(f"Crop recommend error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Crop recommendation failed"}), 500


# ---------------- Blueprints ----------------
try:
    from module2_carbon import carbon_bp
    app.register_blueprint(carbon_bp, url_prefix="/api")
    logger.info("✅ Carbon blueprint registered")
except Exception as e:
    logger.error(f"❌ Carbon blueprint failed: {str(e)}")

try:
    from module3_profit import profit_bp
    app.register_blueprint(profit_bp, url_prefix="/api")
    logger.info("✅ Profit blueprint registered")
except Exception as e:
    logger.error(f"❌ Profit blueprint failed: {str(e)}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
