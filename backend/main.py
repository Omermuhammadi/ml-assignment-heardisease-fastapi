# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Define the FastAPI app
app = FastAPI(title="Heart Disease Prediction API")

# Mount static files (frontend)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def read_root():
    return FileResponse("frontend/index.html")

# Load the saved model (use relative path for Docker compatibility)
model = joblib.load("ml/model.pkl")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://127.0.0.1:5500"] if you want to restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input format using Pydantic
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

@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API is running!"}

@app.post("/predict")
def predict(data: HeartData):
    input_data = np.array([
        [
            data.age, data.sex, data.cp, data.trestbps, data.chol,
            data.fbs, data.restecg, data.thalach, data.exang,
            data.oldpeak, data.slope, data.ca, data.thal
        ]
    ])

    prediction = model.predict(input_data)[0]
    result = "Presence of heart disease" if prediction == 1 else "No heart disease"
    return {"prediction": int(prediction), "result": result}
