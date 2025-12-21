from flask import Flask, request, jsonify
from flask_cors import CORS
from db import get_connection
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import requests
import os

app = Flask(__name__)
CORS(app)

# ================= MODEL =================
MODEL_PATH = "model/waste_classifier.h5"  # Fixed path for Flask
model = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ["hazardous", "organic", "recyclable"]

# ================= WEATHER CONFIG =================
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "dfa03052a6ec019c943a7889fb20b8fe")

# ---------------- IMAGE PREPROCESSING ----------------
def preprocess_image(image_file):
    """Process image from Flask file upload"""
    img = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- RULE ENGINE ----------------
def apply_rules(waste_type, confidence):
    """Rule-based decision engine"""
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

    if waste_type == "hazardous":
        return {
            "status": "hazardous",
            "message": "Waste is hazardous. Do NOT compost. Dispose safely."
        }

    # Fallback (safety)
    return {
        "status": "unknown",
        "message": "Unable to classify waste confidently."
    }

# ---------------- EXPLANATION ENGINE ----------------
def get_rule_explanation(waste_type):
    """Explainable AI logic"""
    if waste_type == "organic":
        return [
            "Natural texture patterns detected",
            "Green or brown color tones identified",
            "Biodegradable material characteristics",
            "Suitable for composting and manure preparation"
        ]

    if waste_type == "recyclable":
        return [
            "Uniform surface structure detected",
            "Man-made material patterns found",
            "Common recyclable object shapes identified"
        ]

    if waste_type == "hazardous":
        return [
            "Synthetic or chemical-like appearance",
            "Sharp or unsafe material indicators",
            "Not suitable for composting or reuse"
        ]

    return []

# ---------------- MANURE GUIDANCE ----------------
def get_manure_guidance():
    """Composting and nutrient data for organic waste"""
    return [
        {
            "waste_name": "Organic Waste",
            "compost_method": "Aerated Pile Composting",
            "preparation_time": "45-60 days",
            "nutrients": "N:1.5-2.5%, P:1.0-1.8%, K:0.8-1.5%",
            "suitable_crops": "Tomato, Chilli, Brinjal, Onion, Leafy Greens"
        }
    ]

# ---------------- DB FUNCTIONS ----------------
def save_waste_analysis(waste_type, confidence, status, message):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO waste_analysis (waste_type, confidence, status, message)
        VALUES (%s, %s, %s, %s)
    """, (waste_type, confidence, status, message))
    conn.commit()
    conn.close()

def save_manure_guidance(waste_name, compost_method, preparation_time, nutrients, suitable_crops):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO manure_guidance (waste_name, compost_method, preparation_time, nutrients, suitable_crops)
        VALUES (%s, %s, %s, %s, %s)
    """, (waste_name, compost_method, preparation_time, nutrients, suitable_crops))
    conn.commit()
    conn.close()

# ================= WASTE CLASSIFICATION API =================
@app.route("/predict", methods=["POST"])
def predict_waste():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    try:
        # Reset file pointer
        file.seek(0)
        
        # Process image
        img = preprocess_image(file)
        
        # Predict
        prediction = model.predict(img, verbose=0)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = round(float(np.max(prediction)) * 100, 2)
        
        # Apply rules
        rule_result = apply_rules(predicted_class, confidence)
        explanation = get_rule_explanation(predicted_class)
        
        # Manure guidance for organic waste
        manure_guidance = None
        if predicted_class == "organic":
            manure_guidance = get_manure_guidance()
            # Save to manure_guidance table
            guidance = manure_guidance[0]
            save_manure_guidance(
                guidance["waste_name"],
                guidance["compost_method"],
                guidance["preparation_time"],
                guidance["nutrients"],
                guidance["suitable_crops"]
            )
        
        # Save to waste_analysis table
        save_waste_analysis(predicted_class, confidence, rule_result["status"], rule_result["message"])
        
        # Complete response
        response = {
            "predicted_waste_type": predicted_class,
            "confidence": confidence,
            "explanation": explanation,
            "status": rule_result["status"],
            "message": rule_result["message"],
            "manure_guidance": manure_guidance
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 500

# ================= HEALTH CHECK =================
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "classes": CLASS_NAMES,
        "message": "Waste AI API ready!"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
