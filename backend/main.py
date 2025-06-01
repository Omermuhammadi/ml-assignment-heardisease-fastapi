# backend/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
import warnings

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- Setup correct base paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "../frontend")
MODEL_PATH = os.path.join(BASE_DIR, "../ml/model.pkl")

# Define the FastAPI app
app = FastAPI(title="Heart Disease Prediction API")

# Load the model safely with error handling
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Add CORS middleware first
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for API input
class HeartData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# Prediction endpoint - define this BEFORE static files
@app.post("/predict")
def predict(data: HeartData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Create input array with proper feature names to avoid warnings
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        # Create DataFrame with proper column names
        input_df = pd.DataFrame([[
            data.age, data.sex, data.cp, data.trestbps, data.chol,
            data.fbs, data.restecg, data.thalach, data.exang,
            data.oldpeak, data.slope, data.ca, data.thal
        ]], columns=feature_names)
        
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else None
        
        result = "Presence of heart disease" if prediction == 1 else "No heart disease"
        
        response = {
            "prediction": int(prediction), 
            "result": result
        }
        
        if probability is not None:
            response["probability"] = {
                "no_disease": float(probability[0]),
                "disease": float(probability[1])
            }
        
        return response
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Mount static files AFTER API routes
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Root endpoint to serve frontend
@app.get("/")
def read_root():
    try:
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
    except Exception as e:
        raise HTTPException(status_code=404, detail="Frontend not found")