from flask import Blueprint, request, jsonify
import logging

logger = logging.getLogger(__name__)
carbon_bp = Blueprint("carbon", __name__)

def save_carbon_footprint(fertilizer, diesel, electricity, total_co2, status):
    """Save carbon footprint data to MySQL (Non-blocking for Vercel)"""
    try:
        from db import get_connection
        conn = get_connection()
        if conn is not None:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO carbon_footprint (fertilizer, diesel, electricity, total_co2, status)
                VALUES (%s, %s, %s, %s, %s)
            """, (fertilizer, diesel, electricity, total_co2, status))
            conn.commit()
            conn.close()
            logger.info("✅ Carbon data saved to DB")
        else:
            logger.info("⚠️ No DB connection - continuing without save")
    except Exception as e:
        logger.error(f"DB save failed (non-critical): {str(e)}")
        # Continue without crashing - Vercel serverless!

@carbon_bp.route("/carbon-footprint", methods=["POST"])
def calculate_carbon():
    data = request.get_json() or {}
    
    try:
        fertilizer = float(data.get("fertilizer", 0) or 0)
        diesel = float(data.get("diesel", 0) or 0)
        electricity = float(data.get("electricity", 0) or 0)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid numeric input"}), 400

    # Emission factors (kg CO2 per unit)
    co2_fertilizer = fertilizer * 1.3
    co2_diesel = diesel * 2.68
    co2_electricity = electricity * 0.82

    total_co2 = round(
        co2_fertilizer + co2_diesel + co2_electricity, 2
    )

    if total_co2 < 50:
        status = "Low"
    elif total_co2 < 150:
        status = "Medium"
    else:
        status = "High"

    # ✅ SAVE TO DATABASE (Non-blocking)
    save_carbon_footprint(fertilizer, diesel, electricity, total_co2, status)

    return jsonify({
        "fertilizer_co2": round(co2_fertilizer, 2),
        "diesel_co2": round(co2_diesel, 2),
        "electricity_co2": round(co2_electricity, 2),
        "total_co2": total_co2,
        "status": status,
        "suggestions": [
            "Use organic manure",
            "Adopt solar-powered pumps",
            "Reduce chemical fertilizer usage"
        ]
    })
