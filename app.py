import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.preprocessing import LabelEncoder


classification_model = joblib.load('migraine_trigger_model.pkl')
regression_model = joblib.load('migraine_trigger_regression_model.pkl')


label_encoders = {
    'Weather': LabelEncoder(),
    'Type': LabelEncoder()
}


st.title("Migraine Prediction App")


st.header("Input Features")
patient_id = st.number_input("Patient ID", min_value=1)
diet = st.number_input("Diet", min_value=0)
sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0)
vomit = st.number_input("Vomit (0 or 1)", min_value=0, max_value=1)
frequency = st.number_input("Frequency of migraines", min_value=0)
intensity = st.number_input("Intensity of pain (1-10)", min_value=1, max_value=10)
visual = st.number_input("Visual disturbances (0 or 1)", min_value=0, max_value=1)
location = st.number_input("Location of pain (0 or 1)", min_value=0, max_value=1)
weather = st.selectbox("Weather", options=["Sunny", "Rainy", "Cloudy", "Snowy"])
air_quality = st.number_input("Air Quality Index", min_value=0)
type_of_migraine = st.selectbox("Type of Migraine", options=["Type1", "Type2", "Type3"])


weather_encoded = label_encoders['Weather'].fit_transform([weather])[0]
type_encoded = label_encoders['Type'].fit_transform([type_of_migraine])[0]


input_data_classification = pd.DataFrame({
    'Patient_id': [patient_id],
    'Diet': [diet],
    'Sleep Duration': [sleep_duration],
    'Vomit': [vomit],
    'Frequency': [frequency],
    'Intensity': [intensity],
    'Visual': [visual],
    'Location': [location],
    'Weather': [weather_encoded],
    'Air-Q': [air_quality],
    'Type': [type_encoded],
})

input_data_regression = input_data_classification.copy()


if st.button("Predict"):
    
    classification_prediction = classification_model.predict(input_data_classification)[0]
    
    
    predicted_time_hours = regression_model.predict(input_data_regression)[0]
    
    
    predicted_time_minutes = predicted_time_hours * 60
    predicted_hours = int(predicted_time_hours)
    predicted_minutes_only = int(predicted_time_minutes)
    
    
    st.success(f"Predicted Triggered In: {classification_prediction}")
    st.success(f"Predicted Time Until Triggered: {predicted_hours} hours and {predicted_minutes_only % 60} minutes")