from fastapi import FastAPI
import joblib
import numpy as np
from data_model import StrokeInput

app = FastAPI()

rf_model = joblib.load('MODELS/stroke_prediction_model.pkl')
lr_model = joblib.load('MODELS/lr_model.pkl')
scaler = joblib.load('MODELS/scaler.pkl')

@app.get("/")
def read_root():
    return {"Hello": "Welcome to the heart Stroke predictor."}

# Mappings
gender_mapping = {'female': 0, 'male': 1, 'other': 2}
ever_married_mapping = {'no': 0, 'yes': 1}
work_type_mapping = {'govt_job': 0, 'never_worked': 1, 'private': 2, 'self-employed': 3, 'children': 4}
residence_type_mapping = {'rural': 0, 'urban': 1}
smoking_status_mapping = {'unknown': 0, 'formerly smoked': 1, 'never smoked': 2, 'smokes': 3}

def convert_input(data: StrokeInput):
    return [
        gender_mapping[data.gender],
        data.age,
        data.hypertension,
        data.heart_disease,
        ever_married_mapping[data.ever_married],
        work_type_mapping[data.work_type],
        residence_type_mapping[data.Residence_type],
        data.avg_glucose_level,
        data.bmi,
        smoking_status_mapping[data.smoking_status]
    ]
    
@app.post('/predict')
def predict_stroke(data: StrokeInput):
    input_data = np.array([convert_input(data)])
    
    #scaling
    input_data_scaled = scaler.transform(input_data)
    
    #prediction
    print("INPUT DATA::", input_data)
    rf_prediction = rf_model.predict(input_data)
    print("RF::",rf_prediction)
    
    lr_prediction = lr_model.predict(input_data)
    print("LR::", lr_prediction)
    
    return {'stroke_prediction_RF': int(rf_prediction[0]),
            'stroke_prediction_LR': int(lr_prediction[0])}
    