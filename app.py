import os
import logging
import traceback
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import requests
import tensorflow as tf
import mysql.connector
from mysql.connector import Error

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- App ----------------
app = Flask(__name__)
CORS(app)

# ---------------- Config ----------------
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "dfa03052a6ec019c943a7889fb20b8fe")
MODEL_PATH = os.environ.get("MODEL_PATH", "model/waste_classifier.h5")
CLASS_NAMES = ["hazardous", "organic", "recyclable"]

# Lazy-loaded model cache
_model = None
_model_load_error = None

# ---------------- FIXED DB Connection ----------------
def get_db_connection():
    """✅ FIXED: Direct MySQL connection for Vercel"""
    try:
        conn = mysql.connector.connect(
            host=os.environ.get("MYSQL_HOST"),
            port=int(os.environ.get("MYSQL_PORT", 3306)),
            user=os.environ.get("MYSQL_USER"),
            password=os.environ.get("MYSQL_PASSWORD"),
            database=os.environ.get("MYSQL_DATABASE"),
            autocommit=True,
        )
        return conn
    except Error as e:
        logger.error(f"DB Error: {e}")
        return None

# ✅ FIXED: Save to YOUR 3 TABLES
def save_carbon_footprint(carbon_value):
    try:
        conn = get_db_connection()
        if not conn: return
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO carbon_footprint (carbon_value) VALUES (%s)",
            (carbon_value,)
        )
        logger.info(f"✅ Saved carbon: {carbon_value}")
    except Exception as e:
        logger.error(f"Carbon save error: {e}")

def save_waste_data(waste_type, waste_amount):
    try:
        conn = get_db_connection()
        if not conn: return
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO waste_data (waste_type, waste_amount) VALUES (%s, %s)",
            (waste_type, waste_amount)
        )
        logger.info(f"✅ Saved waste: {waste_type}")
    except Exception as e:
        logger.error(f"Waste save error: {e}")

def save_profit_metrics(revenue, profit_margin):
    try:
        conn = get_db_connection()
        if not conn: return
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO profit_metrics (revenue, profit_margin) VALUES (%s, %s)",
            (revenue, profit_margin)
        )
        logger.info(f"✅ Saved profit: {revenue}")
    except Exception as e:
        logger.error(f"Profit save error: {e}")

def get_model():
    global _model, _model_load_error
    if _model is None and _model_load_error is not None:
        return None
    if _model is None:
        try:
            logger.info(f"Loading model from: {MODEL_PATH}")
            _model = tf.keras.models.load_model(MODEL_PATH)
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            _model_load_error = e
            logger.error(f"❌ Model load failed: {str(e)}")
            _model = None
    return _model

def apply_rules(waste_type: str, confidence: float):
    if confidence < 60:
        return {"status": "low_confidence", "message": "Image is unclear. Please upload a clearer image."}
    if waste_type == "organic":
        return {"status": "compostable", "message": "Waste is organic and suitable for composting."}
    if waste_type == "recyclable":
        return {"status": "recyclable", "message": "Waste should be sent for recycling."}
    return {"status": "hazardous", "message": "Waste is hazardous. Do NOT compost. Dispose safely."}

def get_rule_explanation(waste_type: str):
    if waste_type == "organic":
        return ["Natural texture patterns detected", "Green or brown color tones identified", "Biodegradable material characteristics"]
    if waste_type == "recyclable":
        return ["Uniform surface structure detected", "Man-made material patterns found", "Common recyclable object shapes identified"]
    if waste_type == "hazardous":
        return ["Synthetic or chemical-like appearance", "Sharp or unsafe material indicators", "Not suitable for composting"]
    return []

def fetch_weather(city: str):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
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

# ---------------- FIXED ROUTES ----------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "Eco Intel AI Backend LIVE!",
        "database": "Railway MySQL Connected ✅",
        "endpoints": ["/api/carbon", "/api/waste", "/api/profit", "/api/crop-recommend"],
        "model_loaded": get_model() is not None,
    })

@app.route("/api/carbon", methods=["POST"])
def carbon_footprint():
    try:
        data = request.get_json() or {}
        fertilizer = data.get("fertilizer", 0)
        diesel = data.get("diesel", 0)
        electricity = data.get("electricity", 0)

        # Calculate CO2
        fertilizer_co2 = fertilizer * 2.5 if fertilizer else 0
        diesel_co2 = diesel * 2.7 if diesel else 0
        electricity_co2 = electricity * 0.82 if electricity else 0
        total_co2 = round(fertilizer_co2 + diesel_co2 + electricity_co2, 2)

        status = "Low" if total_co2 < 50 else "Medium" if total_co2 < 100 else "High"
        suggestions = [
            "Switch to organic fertilizers",
            "Use solar-powered irrigation",
            "Optimize tractor routes to save diesel"
        ]

        # ✅ SAVE TO RAILWAY DB
        save_carbon_footprint(total_co2)

        return jsonify({
            "total_co2": total_co2,
            "status": status,
            "suggestions": suggestions,
            "fertilizer_co2": fertilizer_co2 if fertilizer else None,
            "diesel_co2": diesel_co2 if diesel else None,
            "electricity_co2": electricity_co2 if electricity else None,
        })
    except Exception as e:
        logger.error(f"Carbon error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/waste", methods=["POST"])
def predict_waste():
    try:
        model = get_model()
        if model is None:
            return jsonify({"error": "AI model not available"}), 503

        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]
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

        # ✅ SAVE TO RAILWAY DB (waste_amount = confidence as proxy)
        save_waste_data(predicted_class, confidence)

        return jsonify({
            "predicted_waste_type": predicted_class,
            "confidence": confidence,
            "explanation": explanation,
            "status": rule_result["status"],
            "message": rule_result["message"],
        })
    except Exception as e:
        logger.error(f"Waste error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Prediction failed"}), 500

@app.route("/api/profit", methods=["POST"])
def profit_estimator():
    try:
        data = request.get_json() or {}
        seedCost = data.get("seedCost", 0)
        fertilizerCost = data.get("fertilizerCost", 0)
        laborCost = data.get("laborCost", 0)
        waterCost = data.get("waterCost", 0)
        expectedIncome = data.get("expectedIncome", 0)

        total_cost = seedCost + fertilizerCost + laborCost + waterCost
        profit = expectedIncome - total_cost
        profit_margin = round((profit / expectedIncome * 100) if expectedIncome else 0, 2)
        status = "Profitable" if profit > 0 else "Loss"

        # ✅ SAVE TO RAILWAY DB
        save_profit_metrics(expectedIncome, profit_margin)

        return jsonify({
            "total_cost": round(total_cost, 2),
            "total_income": expectedIncome,
            "profit": round(profit, 2),
            "status": status,
        })
    except Exception as e:
        logger.error(f"Profit error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/crop-recommend", methods=["POST"])
def crop_recommend():
    try:
        data = request.get_json() or {}
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

        # Simple crop logic
        if rainfall > 200 and season.lower() == "kharif":
            crops = ["Rice", "Maize", "Cotton"]
            explanation = ["High rainfall during Kharif suits these crops"]
        elif temperature > 25 and season.lower() == "rabi":
            crops = ["Wheat", "Barley", "Mustard"]
            explanation = ["Moderate temperature suits Rabi crops"]
        else:
            crops = ["Millets", "Pulses", "Groundnut"]
            explanation = ["Climate-resilient crops for your conditions"]

        return jsonify({
            "temperature": temperature,
            "rainfall": rainfall,
            "humidity": weather["humidity"],
            "weather_icon": weather["icon"],
            "weather_desc": weather["description"],
            "recommended_crops": crops,
            "explanation": explanation,
        })
    except Exception as e:
        logger.error(f"Crop error: {str(e)}")
        return jsonify({"error": "Crop recommendation failed"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
