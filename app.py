import os
import logging
import traceback
import io

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import requests
import mysql.connector
from mysql.connector import Error

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- App ----------------
app = Flask(__name__)

# ✅ CORS: allow your frontend to call /api/*
CORS(
    app,
    resources={r"/api/*": {"origins": "https://eco-intel-frontend.vercel.app"}},
    supports_credentials=False,
)

# ---------------- Config ----------------
WEATHER_API_KEY = os.environ.get(
    "WEATHER_API_KEY", "dfa03052a6ec019c943a7889fb20b8fe"
)

# ---------------- DB Connection ----------------
def get_db_connection():
    """Direct MySQL connection for Vercel (Railway DB)."""
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

# ---------------- DB Save Helpers ----------------
def save_carbon_footprint(carbon_value: float):
    try:
        conn = get_db_connection()
        if not conn:
            return
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO carbon_footprint (carbon_value) VALUES (%s)",
            (carbon_value,),
        )
        logger.info(f"✅ Saved carbon: {carbon_value}")
    except Exception as e:
        logger.error(f"Carbon save error: {e}")

def save_waste_data(waste_type: str, waste_amount: float):
    try:
        conn = get_db_connection()
        if not conn:
            return
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO waste_data (waste_type, waste_amount) VALUES (%s, %s)",
            (waste_type, waste_amount),
        )
        logger.info(f"✅ Saved waste: {waste_type}")
    except Exception as e:
        logger.error(f"Waste save error: {e}")

def save_profit_metrics(revenue: float, profit_margin: float):
    try:
        conn = get_db_connection()
        if not conn:
            return
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO profit_metrics (revenue, profit_margin) VALUES (%s, %s)",
            (revenue, profit_margin),
        )
        logger.info(f"✅ Saved profit: {revenue}")
    except Exception as e:
        logger.error(f"Profit save error: {e}")

# ---------------- Weather Helper ----------------
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

# ---------------- Routes ----------------
@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "status": "Eco Intel AI Backend LIVE!",
            "database": "Railway MySQL Connected ✅",
            "endpoints": [
                "/api/carbon",
                "/api/waste",
                "/api/profit",
                "/api/crop-recommend",
            ],
            "ai_image_status": "Disabled on Vercel backend (TensorFlow too large).",
        }
    )

@app.route("/api/carbon", methods=["POST"])
def carbon_footprint():
    try:
        data = request.get_json() or {}
        fertilizer = data.get("fertilizer", 0)
        diesel = data.get("diesel", 0)
        electricity = data.get("electricity", 0)

        fertilizer_co2 = fertilizer * 2.5 if fertilizer else 0
        diesel_co2 = diesel * 2.7 if diesel else 0
        electricity_co2 = electricity * 0.82 if electricity else 0
        total_co2 = round(fertilizer_co2 + diesel_co2 + electricity_co2, 2)

        status = "Low" if total_co2 < 50 else "Medium" if total_co2 < 100 else "High"
        suggestions = [
            "Switch to organic fertilizers",
            "Use solar-powered irrigation",
            "Optimize tractor routes to save diesel",
        ]

        save_carbon_footprint(total_co2)

        return jsonify(
            {
                "total_co2": total_co2,
                "status": status,
                "suggestions": suggestions,
                "fertilizer_co2": fertilizer_co2 if fertilizer else None,
                "diesel_co2": diesel_co2 if diesel else None,
                "electricity_co2": electricity_co2 if electricity else None,
            }
        )
    except Exception as e:
        logger.error(f"Carbon error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/waste", methods=["POST"])
def predict_waste_disabled():
    """Placeholder: AI image classification disabled in Vercel backend."""
    return jsonify(
        {
            "error": (
                "Image AI is disabled on the Vercel backend because TensorFlow is too large. "
                "Run the local backend for full waste classification."
            )
        }
    ), 503

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
        profit_margin = round(
            (profit / expectedIncome * 100) if expectedIncome else 0, 2
        )
        status = "Profitable" if profit > 0 else "Loss"

        save_profit_metrics(expectedIncome, profit_margin)

        return jsonify(
            {
                "total_cost": round(total_cost, 2),
                "total_income": expectedIncome,
                "profit": round(profit, 2),
                "status": status,
            }
        )
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

        if rainfall > 200 and season.lower() == "kharif":
            crops = ["Rice", "Maize", "Cotton"]
            explanation = ["High rainfall during Kharif suits these crops"]
        elif temperature > 25 and season.lower() == "rabi":
            crops = ["Wheat", "Barley", "Mustard"]
            explanation = ["Moderate temperature suits Rabi crops"]
        else:
            crops = ["Millets", "Pulses", "Groundnut"]
            explanation = ["Climate-resilient crops for your conditions"]

        return jsonify(
            {
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
        logger.error(f"Crop error: {str(e)}")
        return jsonify({"error": "Crop recommendation failed"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
