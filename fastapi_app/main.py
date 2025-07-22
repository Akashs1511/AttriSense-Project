# fastapi_app/main.py

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib

# Load model once
model = joblib.load('attrisense_rf_model.pkl')

# Example schema: adjust fields to match your CSV columns
class EmployeeData(BaseModel):
    Age: int = Field(..., ge=18, le=100)
    JobSatisfaction: int = Field(..., ge=1, le=5)
    DistanceFromHome: int = Field(..., ge=0)
    # ... add all other features the model needs

app = FastAPI()

@app.post("/check_employee")
def check_employee(data: EmployeeData):
    # Prepare input in the right format (list of lists)
    features = [
        data.Age,
        data.JobSatisfaction,
        data.DistanceFromHome,
        # ... in correct order!
    ]
    prediction = model.predict([features])[0]
    risk = "High" if prediction == 1 else "Low"
    return {"attrition_risk": risk}
