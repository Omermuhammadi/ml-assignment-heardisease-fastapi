# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# --- Setup correct base paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "../frontend")
MODEL_PATH = os.path.join(BASE_DIR, "../ml/model.pkl")

# Define the FastAPI app
app = FastAPI(title="Heart Disease Prediction API")

# Mount static files (frontend)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def read_root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# Load the model safely with full path
model = joblib.load(MODEL_PATH)

# Allow all origins for now (update in production!)
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

@app.post("/predict")
def predict(data: HeartData):
    input_data = np.array([[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal
    ]])
    prediction = model.predict(input_data)[0]
    result = "Presence of heart disease" if prediction == 1 else "No heart disease"
    return {"prediction": int(prediction), "result": result}
