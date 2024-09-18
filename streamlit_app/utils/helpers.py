import joblib
import streamlit as st

@st.cache_data
def load_model_and_scaler():
    """
    Loads models and scaler, caching them for optimized performance.
    This avoids reloading models on every rerun.
    """
    try:
        rf_model = joblib.load('MODELS/stroke_prediction_model.pkl')
        lr_model = joblib.load('MODELS/lr_model.pkl')
        new_lr_model = joblib.load('MODELS/new_lr_model.pkl')
        scaler = joblib.load('MODELS/scaler.pkl')

        return rf_model, lr_model, new_lr_model, scaler
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        raise
    except Exception as e:
        st.error(f"An error occurred while loading models: {e}")
        raise

def get_mappings():
    return {
        'gender_mapping': {'Female': 0, 'Male': 1, 'Other': 2},
        'ever_married_mapping': {'No': 0, 'Yes': 1},
        'work_type_mapping': {'Govt job': 0, 'Never worked': 1, 'Private': 2, 'Self-employed': 3, 'children': 4},
        'residence_type_mapping': {'Rural': 0, 'Urban': 1},
        'smoking_status_mapping': {'Unknown': 0, 'formerly smoked': 1, 'never smoked': 2, 'smokes': 3}
    }