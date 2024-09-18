import streamlit as st
import time
import numpy as np
from utils.helpers import load_model_and_scaler, get_mappings
from utils.markdown import demographics_info

rf_model, lr_model, new_lr_model, scaler = load_model_and_scaler()

# Sidebar
st.sidebar.title("Model Selection")
model_option = st.sidebar.radio(
    "Choose a model for prediction:",
    options=["Random Forest", "Logistic Regression", "Logistic Regression_01"],
    index=1
)
model = {
    "Random Forest": rf_model,
    "Logistic Regression": lr_model,
    "Logistic Regression_01": new_lr_model
}.get(model_option, lr_model)  


st.title("Heart Stroke Predictor")
st.markdown("---")
st.write("This app helps predict the likelihood of a patient experiencing a stroke based on various health and demographic factors.")
with st.expander("Click to view demographic information"):
    st.markdown(demographics_info)

# st.markdown("---")

st.subheader("Use Our ML models to check if a patient may get a heart stroke or not.")

mappings = get_mappings()
with st.form(key="patient_form"):
    st.subheader("Please fill in the patient's demographic information:")
    
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", list(mappings['gender_mapping'].keys()))
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        hypertension = st.selectbox("Hypertension", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        heart_disease = st.selectbox("Heart Disease", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

    with col2:
        ever_married = st.selectbox("Ever Married", list(mappings['ever_married_mapping'].keys()))
        work_type = st.selectbox("Work Type", list(mappings['work_type_mapping'].keys()))
        residence_type = st.selectbox("Residence Type", list(mappings['residence_type_mapping'].keys()))
        smoking_status = st.selectbox("Smoking Status", list(mappings['smoking_status_mapping'].keys()))
        
    col3, col4 = st.columns(2)
    with col3:
        avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, step=0.1)
    with col4:
        bmi = st.number_input("BMI", min_value=0.0, step=0.1)

    submit_button = st.form_submit_button(label="Submit", use_container_width=True)

if submit_button:
    patient_data = {
        'gender': mappings['gender_mapping'][gender],
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': mappings['ever_married_mapping'][ever_married],
        'work_type': mappings['work_type_mapping'][work_type],
        'residence_type': mappings['residence_type_mapping'][residence_type],
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': mappings['smoking_status_mapping'][smoking_status]
    }

    input_data = np.array([list(patient_data.values())])
    input_data_scaled = scaler.transform(input_data)

    with st.spinner('Processing patient data...'):
        prediction = model.predict(input_data_scaled)
        time.sleep(2)

    if prediction[0] == 0:
        st.success("The patient is not likely to experience any heart strokes.")
    else:
        st.warning("The patient has a risk of experiencing a heart stroke.")