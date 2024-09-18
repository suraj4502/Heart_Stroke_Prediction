from pydantic import BaseModel

# Input data format for FastAPI with human-readable strings
class StrokeInput(BaseModel):
    gender: str  # "Male", "Female", "Other"
    age: float
    hypertension: int  # 0 or 1
    heart_disease: int  # 0 or 1
    ever_married: str  # "Yes" or "No"
    work_type: str  # "Govt_job", "Never_worked", "Private", "Self-employed", "children"
    Residence_type: str  # "Rural" or "Urban"
    avg_glucose_level: float
    bmi: float
    smoking_status: str  # "Unknown", "formerly smoked", "never smoked", "smokes"