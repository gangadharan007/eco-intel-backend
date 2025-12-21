from flask import Blueprint, request, jsonify
from db import get_connection

profit_bp = Blueprint("profit", __name__)

def save_profit_estimator(total_cost, revenue, profit, risk):
    """Save profit estimation data to MySQL"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO profit_estimator (total_cost, revenue, profit, risk)
        VALUES (%s, %s, %s, %s)
    """, (total_cost, revenue, profit, risk))
    conn.commit()
    conn.close()

@profit_bp.route("/profit-estimator", methods=["POST"])
def calculate_profit():
    data = request.json or {}
    
    try:
        seed = float(data.get("seedCost", 0) or 0)
        fertilizer = float(data.get("fertilizerCost", 0) or 0)
        labor = float(data.get("laborCost", 0) or 0)
        water = float(data.get("waterCost", 0) or 0)
        expected_income = float(data.get("expectedIncome", 0) or 0)
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid numeric input"}), 400

    # Calculate totals
    total_cost = round(seed + fertilizer + labor + water, 2)
    profit = round(expected_income - total_cost, 2)
    
    # Risk assessment
    if profit > 10000:
        risk = "Low"
    elif profit > 0:
        risk = "Medium"
    else:
        risk = "High"
    
    status = "Profit" if profit > 0 else "Loss"

    # ✅ SAVE TO DATABASE (exact schema match)
    save_profit_estimator(total_cost, expected_income, profit, risk)

    return jsonify({
        "total_cost": total_cost,
        "total_income": expected_income,
        "profit": profit,
        "status": status
    })
