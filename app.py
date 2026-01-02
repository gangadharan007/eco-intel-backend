import os
import logging
import traceback
from urllib.parse import urlparse
from datetime import date, timedelta

import requests
import mysql.connector
from mysql.connector import Error
from flask import Flask, request, jsonify
from flask_cors import CORS

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- App ----------------
app = Flask(__name__)

# CORS for frontend -> backend /api/*
CORS(
    app,
    resources={r"/api/*": {"origins": "https://eco-intel-frontend.vercel.app"}},
    supports_credentials=False,
)

# ---------------- Config ----------------
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "").strip()
WASTE_AI_URL = os.environ.get("WASTE_AI_URL", "").rstrip("/").strip()

OPENWEATHER_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHER_GEO_URL = "https://api.openweathermap.org/geo/1.0/direct"
NASA_POWER_DAILY_POINT_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# ---------------- DB Connection ----------------
def _conn_from_mysql_url(mysql_url: str):
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
    try:
        mysql_url = os.environ.get("MYSQL_URL")
        if mysql_url:
            return _conn_from_mysql_url(mysql_url)

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
        cur = conn.cursor()
        cur.execute("INSERT INTO carbon_footprint (carbon_value) VALUES (%s)", (carbon_value,))
        cur.close()
        conn.close()
        logger.info("✅ Saved carbon: %s", carbon_value)
    except Exception as e:
        logger.error("Carbon save error: %s", e)

def save_waste_data(waste_type: str, waste_amount: float):
    try:
        conn = get_db_connection()
        if not conn:
            return
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO waste_data (waste_type, waste_amount) VALUES (%s, %s)",
            (waste_type, waste_amount),
        )
        cur.close()
        conn.close()
        logger.info("✅ Saved waste: %s", waste_type)
    except Exception as e:
        logger.error("Waste save error: %s", e)

def save_profit_metrics(revenue: float, profit_margin: float):
    try:
        conn = get_db_connection()
        if not conn:
            return
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO profit_metrics (revenue, profit_margin) VALUES (%s, %s)",
            (revenue, profit_margin),
        )
        cur.close()
        conn.close()
        logger.info("✅ Saved profit: %s", revenue)
    except Exception as e:
        logger.error("Profit save error: %s", e)

def save_crop_recommendation(location: str, soil: str, season: str, avg_temp_30d: float, rain_30d: float, humidity: int, crops: list):
    try:
        conn = get_db_connection()
        if not conn:
            return
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO crop_recommendations (location, soil, season, avg_temp_30d, rain_30d_mm, humidity, crops_json) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (location, soil, season, avg_temp_30d, rain_30d, humidity, str(crops)),
        )
        cur.close()
        conn.close()
        logger.info("✅ Saved crop recommendation to DB")
    except Exception:
        pass

# ---------------- Weather + Geo Helpers ----------------
def fetch_weather(city: str):
    if not WEATHER_API_KEY:
        return {"_error": "WEATHER_API_KEY not set on backend"}

    try:
        params = {"q": city, "appid": WEATHER_API_KEY, "units": "metric"}
        r = requests.get(OPENWEATHER_WEATHER_URL, params=params, timeout=12)
        if r.status_code != 200:
            print("OpenWeather weather failed", r.status_code, r.text[:300])  # shows in Vercel logs [web:1726]
            logger.error("OpenWeather /weather failed (%s): %s", r.status_code, r.text[:200])
            return {"_error": "OpenWeather weather failed", "status": r.status_code, "raw": r.text[:300]}

        data = r.json()
        main = data.get("main", {}) or {}
        weather_list = data.get("weather", [{}]) or [{}]
        rain = data.get("rain", {}) or {}
        rain_mm = rain.get("1h") or rain.get("3h") or 0

        return {
            "temperature": main.get("temp"),
            "humidity": main.get("humidity"),
            "icon": weather_list[0].get("icon"),
            "description": weather_list[0].get("description"),
            "rain_1h_or_3h": rain_mm,
        }
    except Exception as e:
        print("fetch_weather exception", str(e))
        logger.error("fetch_weather error: %s\n%s", str(e), traceback.format_exc())
        return {"_error": "fetch_weather exception", "detail": str(e)}

def geocode_city(city: str):
    if not WEATHER_API_KEY:
        return {"_error": "WEATHER_API_KEY not set on backend"}

    try:
        params = {"q": city, "limit": 1, "appid": WEATHER_API_KEY}
        r = requests.get(OPENWEATHER_GEO_URL, params=params, timeout=12)
        if r.status_code != 200:
            print("OpenWeather geocode failed", r.status_code, r.text[:300])
            logger.error("OpenWeather geocode failed (%s): %s", r.status_code, r.text[:200])
            return {"_error": "OpenWeather geocode failed", "status": r.status_code, "raw": r.text[:300]}

        arr = r.json()
        if not arr:
            return {"_error": "No geocode results"}

        return {
            "lat": float(arr[0]["lat"]),
            "lon": float(arr[0]["lon"]),
            "name": arr[0].get("name") or city,
            "country": arr[0].get("country"),
            "state": arr[0].get("state"),
        }
    except Exception as e:
        print("geocode_city exception", str(e))
        logger.error("geocode_city error: %s\n%s", str(e), traceback.format_exc())
        return {"_error": "geocode_city exception", "detail": str(e)}

# ---------------- NASA POWER Climate Helper ----------------
def fetch_nasa_power_last30(lat: float, lon: float):
    try:
        end = date.today()
        start = end - timedelta(days=30)

        params = {
            "latitude": lat,
            "longitude": lon,
            "start": start.strftime("%Y%m%d"),
            "end": end.strftime("%Y%m%d"),
            "community": "AG",
            "format": "JSON",
            "parameters": "T2M,PRECTOT",
        }

        r = requests.get(NASA_POWER_DAILY_POINT_URL, params=params, timeout=25)
        if r.status_code != 200:
            # Print to ensure it appears in Vercel logs [web:1726]
            print("NASA POWER failed", r.status_code, r.text[:300])
            logger.error("NASA POWER failed (%s): %s", r.status_code, r.text[:200])
            return {"_error": "NASA POWER failed", "status": r.status_code, "raw": r.text[:300]}

        j = r.json()
        props = (j.get("properties") or {})
        param = (props.get("parameter") or {})
        t2m = (param.get("T2M") or {})
        prec = (param.get("PRECTOT") or {})

        temps = [float(v) for v in t2m.values() if v is not None]
        rains = [float(v) for v in prec.values() if v is not None]

        if not temps or not rains:
            return {"_error": "NASA POWER missing data"}

        avg_temp = sum(temps) / len(temps)
        total_rain = sum(rains)

        return {"avg_temp_30d": round(avg_temp, 2), "rain_30d": round(total_rain, 2)}
    except Exception as e:
        print("NASA POWER exception", str(e))
        logger.error("fetch_nasa_power_last30 error: %s\n%s", str(e), traceback.format_exc())
        return {"_error": "fetch_nasa_power_last30 exception", "detail": str(e)}

# ---------------- Crop Rules ----------------
CROP_RULES = {
    "Rice": {"seasons": {"kharif"}, "soils": {"clay", "loamy"}, "t": (20, 35), "r30": (120, 500), "reason": "Warm + higher water availability suits rice."},
    "Maize": {"seasons": {"kharif", "rabi"}, "soils": {"loamy", "sandy"}, "t": (18, 34), "r30": (50, 250), "reason": "Warm weather with moderate rainfall suits maize."},
    "Cotton": {"seasons": {"kharif"}, "soils": {"loamy", "clay", "sandy"}, "t": (20, 35), "r30": (30, 200), "reason": "Warm temperatures and moderate moisture suit cotton."},
    "Wheat": {"seasons": {"rabi"}, "soils": {"loamy"}, "t": (10, 25), "r30": (10, 120), "reason": "Cooler season + low/moderate rainfall suits wheat."},
    "Barley": {"seasons": {"rabi"}, "soils": {"loamy", "sandy"}, "t": (7, 25), "r30": (10, 100), "reason": "Cooler conditions + lower rainfall suits barley."},
    "Mustard": {"seasons": {"rabi"}, "soils": {"loamy", "sandy"}, "t": (10, 25), "r30": (5, 80), "reason": "Cool + relatively dry conditions suit mustard."},
    "Groundnut": {"seasons": {"kharif", "summer"}, "soils": {"sandy", "loamy"}, "t": (20, 33), "r30": (20, 150), "reason": "Warm + well-drained soils suit groundnut."},
    "Sugarcane": {"seasons": {"kharif", "summer"}, "soils": {"clay", "loamy"}, "t": (20, 35), "r30": (80, 400), "reason": "Warm + higher water availability suits sugarcane."},
    "Millets": {"seasons": {"kharif", "summer"}, "soils": {"sandy", "loamy"}, "t": (20, 38), "r30": (5, 120), "reason": "Heat + low rainfall resilience suits millets."},
    "Pulses": {"seasons": {"kharif", "rabi"}, "soils": {"loamy", "sandy"}, "t": (15, 32), "r30": (5, 120), "reason": "Many pulses fit moderate temps + low/mod rainfall."},
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
@app.get("/")
def root():
    return jsonify(
        {
            "status": "Eco Intel AI Backend LIVE!",
            "waste_ai_proxy_enabled": bool(WASTE_AI_URL),
            "weather_key_configured": bool(WEATHER_API_KEY),
        }
    )

@app.get("/api/db-health")
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

@app.post("/api/carbon")
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
        logger.error("Carbon error: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.post("/api/waste")
def waste_route():
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
        logger.error("Waste error: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.post("/api/profit")
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
        logger.error("Profit error: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.post("/api/crop-recommend")
def crop_recommend():
    try:
        data = request.get_json() or {}
        location = (data.get("location") or "").strip()
        soil = (data.get("soil") or "").strip().lower()
        season = (data.get("season") or "").strip().lower()

        if not location or not soil or not season:
            return jsonify({"error": "Missing required fields: location, soil, season"}), 400

        weather = fetch_weather(location)
        if weather.get("_error"):
            return jsonify({"error": "Weather fetch failed", "detail": weather}), 502

        geo = geocode_city(location)
        if geo.get("_error"):
            return jsonify({"error": "Geocode failed", "detail": geo}), 502

        climate = fetch_nasa_power_last30(geo["lat"], geo["lon"])

        # ✅ Fallback (do NOT fail hard if NASA POWER is throttling/down) [web:1567]
        if climate.get("_error"):
            avg_temp_30d = weather.get("temperature")
            rain_30d = 0
            climate_source = "fallback_openweather"
            climate_detail = climate
        else:
            avg_temp_30d = climate["avg_temp_30d"]
            rain_30d = climate["rain_30d"]
            climate_source = "nasa_power"
            climate_detail = None

        scored = []
        for crop, rule in CROP_RULES.items():
            s = score_crop(rule, avg_temp_30d, rain_30d, soil, season)
            if s is not None:
                scored.append((s, crop, rule["reason"]))
        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored:
            crops = ["Millets", "Pulses", "Groundnut"]
            explanation = [
                "No exact match found for your soil/season with the current climate window.",
                "Fallback to climate-resilient crops; add more crops/rules for wider coverage.",
            ]
        else:
            top = scored[:6]
            crops = [c for _, c, _ in top]
            explanation = [f"{c}: {reason} (score={score})" for score, c, reason in top]

        save_crop_recommendation(
            location=location,
            soil=soil,
            season=season,
            avg_temp_30d=float(avg_temp_30d) if avg_temp_30d is not None else 0.0,
            rain_30d=float(rain_30d) if rain_30d is not None else 0.0,
            humidity=int(weather.get("humidity") or 0),
            crops=crops,
        )

        resp = {
            "location": location,
            "soil": soil,
            "season": season,

            "temperature": weather.get("temperature"),
            "humidity": weather.get("humidity"),
            "weather_icon": weather.get("icon"),
            "weather_desc": weather.get("description"),

            "rainfall": rain_30d,
            "avg_temp_30d": avg_temp_30d,

            "climate_source": climate_source,
            "recommended_crops": crops,
            "explanation": explanation,
        }

        if climate_detail:
            resp["climate_warning"] = climate_detail

        return jsonify(resp)

    except Exception as e:
        logger.error("Crop error: %s\n%s", str(e), traceback.format_exc())
        return jsonify({"error": "Crop recommendation failed", "detail": str(e)}), 500

# Debug helper: test OpenWeather quickly
@app.get("/api/weather-debug")
def weather_debug():
    city = (request.args.get("city") or "").strip()
    if not city:
        return jsonify({"error": "Pass ?city=CityName"}), 400
    return jsonify(
        {
            "weather_key_configured": bool(WEATHER_API_KEY),
            "city": city,
            "weather": fetch_weather(city),
            "geo": geocode_city(city),
        }
    )

# ✅ NEW: Debug helper for NASA POWER
@app.get("/api/climate-debug")
def climate_debug():
    lat = request.args.get("lat")
    lon = request.args.get("lon")
    if not lat or not lon:
        return jsonify({"error": "Pass ?lat=...&lon=..."}), 400

    out = fetch_nasa_power_last30(float(lat), float(lon))
    return jsonify(out), (502 if out.get("_error") else 200)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
