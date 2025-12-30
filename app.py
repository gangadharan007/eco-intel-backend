import os
import logging
import traceback
from urllib.parse import urlparse
from datetime import date, timedelta

from flask import Flask, request, jsonify
from flask_cors import CORS
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
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "")
WASTE_AI_URL = os.environ.get("WASTE_AI_URL", "").rstrip("/")


# ---------------- DB Connection ----------------
def _conn_from_mysql_url(mysql_url: str):
    """
    Parse mysql://user:pass@host:port/db into mysql.connector.connect args.
    """
    u = urlparse(mysql_url)
    db_name = (u.path or "").lstrip("/")
    if not (u.hostname and u.username and db_name):
        raise ValueError("Invalid MYSQL_URL format. Expected mysql://user:pass@host:port/db")

    return mysql.connector.connect(
        host=u.hostname,
        port=u.port or 3306,
        user=u.username,
        password=u.password,
        database=db_name,
        autocommit=True,
    )


def get_db_connection():
    """Direct MySQL connection for Vercel (Railway DB)."""
    try:
        mysql_url = os.environ.get("MYSQL_URL")
        if mysql_url:
            return _conn_from_mysql_url(mysql_url)

        # Fallback to separate env vars
        return mysql.connector.connect(
            host=os.environ.get("MYSQL_HOST"),
            port=int(os.environ.get("MYSQL_PORT", 3306)),
            user=os.environ.get("MYSQL_USER"),
            password=os.environ.get("MYSQL_PASSWORD"),
            database=os.environ.get("MYSQL_DATABASE"),
            autocommit=True,
        )
    except Error as e:
        logger.error(f"DB Error: {e}")
        return None
    except Exception as e:
        logger.error(f"DB Parse/Error: {e}")
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
        cursor.close()
        conn.close()
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
        cursor.close()
        conn.close()
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
        cursor.close()
        conn.close()
        logger.info(f"✅ Saved profit: {revenue}")
    except Exception as e:
        logger.error(f"Profit save error: {e}")


def save_crop_recommendation(location: str, soil: str, season: str, avg_temp_30d: float, rain_30d: float, humidity: int, crops: list):
    """
    Optional DB save. Create the table if you want to store results:

    CREATE TABLE crop_recommendations (
      id INT AUTO_INCREMENT PRIMARY KEY,
      location VARCHAR(100),
      soil VARCHAR(30),
      season VARCHAR(20),
      avg_temp_30d FLOAT,
      rain_30d_mm FLOAT,
      humidity INT,
      crops_json TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    try:
        conn = get_db_connection()
        if not conn:
            return
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO crop_recommendations (location, soil, season, avg_temp_30d, rain_30d_mm, humidity, crops_json) VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (location, soil, season, avg_temp_30d, rain_30d, humidity, str(crops)),
        )
        cur.close()
        conn.close()
        logger.info("✅ Saved crop recommendation to DB")
    except Exception:
        # If table doesn't exist, ignore (keeps API working).
        pass


# ---------------- Weather + Geo Helpers ----------------
def fetch_weather(city: str):
    """
    OpenWeather current weather endpoint (temp, humidity, icon, description).
    Rain data here is only for last 1h/3h when it is raining. [web:1022]
    """
    try:
        if not WEATHER_API_KEY:
            return None

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

        rain = data.get("rain", {}) or {}
        rain_mm = rain.get("1h") or rain.get("3h") or 0

        return {
            "temperature": main.get("temp"),
            "rain_1h_or_3h": rain_mm,
            "humidity": main.get("humidity"),
            "icon": weather_list[0].get("icon"),
            "description": weather_list[0].get("description"),
        }
    except Exception as e:
        logger.error(f"fetch_weather error: {str(e)}")
        return None


def geocode_city(city: str):
    """
    Use OpenWeather geocoding: city -> lat/lon.
    """
    try:
        if not WEATHER_API_KEY:
            return None

        url = "https://api.openweathermap.org/geo/1.0/direct"
        params = {"q": city, "limit": 1, "appid": WEATHER_API_KEY}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        arr = r.json()
        if not arr:
            return None
        return {
            "lat": float(arr[0]["lat"]),
            "lon": float(arr[0]["lon"]),
            "name": arr[0].get("name") or city,
        }
    except Exception as e:
        logger.error(f"geocode_city error: {e}")
        return None


# ---------------- NASA POWER Climate Helper ----------------
def fetch_nasa_power_last30(lat: float, lon: float):
    """
    NASA POWER daily API: compute last 30 days total rainfall (PRECTOT) and average temp (T2M). [web:1052]
    Parameters reference. [web:1037]
    """
    try:
        end = date.today()
        start = end - timedelta(days=30)

        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start": start.strftime("%Y%m%d"),
            "end": end.strftime("%Y%m%d"),
            "community": "AG",
            "format": "JSON",
            "parameters": "T2M,PRECTOT",
        }

        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            logger.error(f"NASA POWER status {r.status_code}: {r.text[:200]}")
            return None

        j = r.json()
        props = (j.get("properties") or {})
        param = (props.get("parameter") or {})
        t2m = (param.get("T2M") or {})
        prec = (param.get("PRECTOT") or {})

        temps = []
        rains = []

        for _, v in t2m.items():
            if v is None:
                continue
            try:
                temps.append(float(v))
            except Exception:
                pass

        for _, v in prec.items():
            if v is None:
                continue
            try:
                rains.append(float(v))
            except Exception:
                pass

        if not temps or not rains:
            return None

        avg_temp = sum(temps) / len(temps)
        total_rain = sum(rains)

        return {
            "avg_temp_30d": round(avg_temp, 2),
            "rain_30d": round(total_rain, 2),
        }
    except Exception as e:
        logger.error(f"fetch_nasa_power_last30 error: {e}")
        return None


# ---------------- Crop Rules (extend this list) ----------------
# NOTE: For “mostly all crops”, later import FAO ECOCROP dataset and score all crops from it. [web:1036]
CROP_RULES = {
    "Rice": {
        "seasons": {"kharif"},
        "soils": {"clay", "loamy"},
        "t": (20, 35),
        "r30": (120, 500),
        "reason": "Warm conditions and higher water availability suit rice.",
    },
    "Maize": {
        "seasons": {"kharif", "rabi"},
        "soils": {"loamy", "sandy"},
        "t": (18, 34),
        "r30": (50, 250),
        "reason": "Maize performs well in warm weather with moderate rainfall.",
    },
    "Cotton": {
        "seasons": {"kharif"},
        "soils": {"loamy", "clay", "sandy"},
        "t": (20, 35),
        "r30": (30, 200),
        "reason": "Cotton prefers warm temperatures and moderate moisture.",
    },
    "Wheat": {
        "seasons": {"rabi"},
        "soils": {"loamy"},
        "t": (10, 25),
        "r30": (10, 120),
        "reason": "Cooler season with low-to-moderate rainfall suits wheat.",
    },
    "Barley": {
        "seasons": {"rabi"},
        "soils": {"loamy", "sandy"},
        "t": (7, 25),
        "r30": (10, 100),
        "reason": "Barley tolerates cooler conditions and lower rainfall.",
    },
    "Mustard": {
        "seasons": {"rabi"},
        "soils": {"loamy", "sandy"},
        "t": (10, 25),
        "r30": (5, 80),
        "reason": "Mustard suits cool weather and relatively dry conditions.",
    },
    "Groundnut": {
        "seasons": {"kharif", "summer"},
        "soils": {"sandy", "loamy"},
        "t": (20, 33),
        "r30": (20, 150),
        "reason": "Groundnut prefers warm weather and well-drained soils.",
    },
    "Sugarcane": {
        "seasons": {"kharif", "summer"},
        "soils": {"clay", "loamy"},
        "t": (20, 35),
        "r30": (80, 400),
        "reason": "Sugarcane needs warmth and higher water availability.",
    },
    "Millets": {
        "seasons": {"kharif", "summer"},
        "soils": {"sandy", "loamy"},
        "t": (20, 38),
        "r30": (5, 120),
        "reason": "Millets are resilient in heat and low rainfall.",
    },
    "Pulses": {
        "seasons": {"kharif", "rabi"},
        "soils": {"loamy", "sandy"},
        "t": (15, 32),
        "r30": (5, 120),
        "reason": "Many pulses fit moderate temperatures and low-to-moderate rainfall.",
    },
}


def score_crop(rule, t, r30, soil, season):
    if season not in rule["seasons"]:
        return None
    if soil not in rule["soils"]:
        return None

    tmin, tmax = rule["t"]
    rmin, rmax = rule["r30"]

    if t is None or r30 is None:
        return None
    if not (tmin <= t <= tmax):
        return None
    if not (rmin <= r30 <= rmax):
        return None

    tmid = (tmin + tmax) / 2.0
    rmid = (rmin + rmax) / 2.0

    score = 100.0
    score -= abs(t - tmid) * 3.0
    score -= abs(r30 - rmid) * 0.3
    return round(score, 2)


# ---------------- Routes ----------------
@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "status": "Eco Intel AI Backend LIVE!",
            "database": "Railway MySQL via mysql.connector ✅",
            "endpoints": [
                "/api/db-health",
                "/api/carbon",
                "/api/waste",
                "/api/profit",
                "/api/crop-recommend",
            ],
            "waste_ai_proxy_enabled": bool(WASTE_AI_URL),
        }
    )


@app.route("/api/db-health", methods=["GET"])
def db_health():
    conn = get_db_connection()
    if not conn:
        return jsonify({"ok": False, "error": "DB connection failed"}), 500
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1")
        row = cur.fetchone()
        cur.close()
        conn.close()
        return jsonify({"ok": row == (1,)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/carbon", methods=["POST"])
def carbon_footprint():
    try:
        data = request.get_json() or {}
        fertilizer = float(data.get("fertilizer", 0) or 0)
        diesel = float(data.get("diesel", 0) or 0)
        electricity = float(data.get("electricity", 0) or 0)

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
def waste_route():
    """
    Two modes:
    1) If WASTE_AI_URL is set: proxy image classification to Railway AI service (multipart/form-data 'image').
    2) Else: just store manual waste data if JSON provided.
    """
    if WASTE_AI_URL:
        if "image" not in request.files:
            return jsonify({"error": "No image provided. Send multipart/form-data with key 'image'."}), 400

        f = request.files["image"]
        if not f or f.filename == "":
            return jsonify({"error": "No file selected"}), 400

        try:
            files = {"image": (f.filename, f.stream, f.mimetype or "application/octet-stream")}
            r = requests.post(f"{WASTE_AI_URL}/predict", files=files, timeout=60)

            try:
                payload = r.json()
            except Exception:
                payload = {"error": "AI service returned non-JSON response", "raw": r.text}

            return jsonify(payload), r.status_code

        except Exception as e:
            logger.error("Waste proxy error: %s\n%s", str(e), traceback.format_exc())
            return jsonify({"error": "Waste classification proxy failed", "detail": str(e)}), 502

    try:
        data = request.get_json() or {}
        waste_type = data.get("waste_type")
        waste_amount = float(data.get("waste_amount", 0) or 0)

        if not waste_type:
            return jsonify({"error": "waste_type is required (or set WASTE_AI_URL and send an image)"}), 400

        save_waste_data(waste_type, waste_amount)
        return jsonify({"ok": True, "waste_type": waste_type, "waste_amount": waste_amount})
    except Exception as e:
        logger.error(f"Waste error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/profit", methods=["POST"])
def profit_estimator():
    try:
        data = request.get_json() or {}
        seedCost = float(data.get("seedCost", 0) or 0)
        fertilizerCost = float(data.get("fertilizerCost", 0) or 0)
        laborCost = float(data.get("laborCost", 0) or 0)
        waterCost = float(data.get("waterCost", 0) or 0)
        expectedIncome = float(data.get("expectedIncome", 0) or 0)

        total_cost = seedCost + fertilizerCost + laborCost + waterCost
        profit = expectedIncome - total_cost
        profit_margin = round((profit / expectedIncome * 100) if expectedIncome else 0, 2)
        status = "Profitable" if profit > 0 else "Loss"

        save_profit_metrics(expectedIncome, profit_margin)

        return jsonify(
            {
                "total_cost": round(total_cost, 2),
                "total_income": expectedIncome,
                "profit": round(profit, 2),
                "profit_margin": profit_margin,
                "status": status,
            }
        )
    except Exception as e:
        logger.error(f"Profit error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/crop-recommend", methods=["POST"])
def crop_recommend():
    """
    Improved crop recommender:
    - Current weather (OpenWeather) for UI [web:1022]
    - Geocode city -> lat/lon (OpenWeather geo)
    - NASA POWER daily -> last 30 days rainfall + avg temp [web:1052][web:1037]
    - Score crops based on rules (extendable; later replace with ECOCROP) [web:1036]
    """
    try:
        data = request.get_json() or {}
        location = (data.get("location") or "").strip()
        soil = (data.get("soil") or "").strip().lower()
        season = (data.get("season") or "").strip().lower()

        if not location or not soil or not season:
            return jsonify({"error": "Missing required fields"}), 400

        weather = fetch_weather(location)
        if not weather:
            return jsonify({"error": "Invalid location"}), 400

        geo = geocode_city(location)
        if not geo:
            return jsonify({"error": "Failed to geocode location"}), 400

        climate = fetch_nasa_power_last30(geo["lat"], geo["lon"])
        if not climate:
            return jsonify({"error": "Climate data fetch failed"}), 502

        avg_temp_30d = climate["avg_temp_30d"]
        rain_30d = climate["rain_30d"]

        scored = []
        for crop, rule in CROP_RULES.items():
            s = score_crop(rule, avg_temp_30d, rain_30d, soil, season)
            if s is not None:
                scored.append((s, crop, rule["reason"]))

        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored:
            # fallback (still returns something)
            crops = ["Millets", "Pulses", "Groundnut"]
            explanation = [
                "No exact match found for your soil/season with the current climate window.",
                "Fallback to climate-resilient crops; add more crops/rules or import ECOCROP for broader coverage.",
            ]
        else:
            top = scored[:6]
            crops = [c for _, c, _ in top]
            explanation = [f"{c}: {reason} (score={score})" for score, c, reason in top]

        # Optional DB save
        save_crop_recommendation(
            location=location,
            soil=soil,
            season=season,
            avg_temp_30d=avg_temp_30d,
            rain_30d=rain_30d,
            humidity=int(weather.get("humidity") or 0),
            crops=crops,
        )

        return jsonify(
            {
                # OpenWeather UI values
                "temperature": weather.get("temperature"),
                "humidity": weather.get("humidity"),
                "weather_icon": weather.get("icon"),
                "weather_desc": weather.get("description"),

                # Agro signal (more useful than 1h rain)
                "rainfall": rain_30d,         # 30-day total rainfall (mm)
                "avg_temp_30d": avg_temp_30d, # extra field (optional for frontend)

                "recommended_crops": crops,
                "explanation": explanation,
            }
        )

    except Exception as e:
        logger.error("Crop error: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"error": "Crop recommendation failed"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
