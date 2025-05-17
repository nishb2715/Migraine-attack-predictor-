import streamlit as st
import pandas as pd
import joblib

# Loading custom CSS
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #e0eafc, #cfdef3);
    font-family: 'Arial', sans-serif;
}
.stApp {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}
.header {
    text-align: center;
    color: #2c3e50;
    font-size: 36px;
    margin-bottom: 20px;
}
.subheader {
    color: #34495e;
    font-size: 24px;
    margin-top: 20px;
}
.card {
    background: white;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.stButton>button {
    background-color: #3498db;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    transition: background-color 0.3s;
}
.stButton>button:hover {
    background-color: #2980b9;
}
.result {
    background: #ecf0f1;
    border-radius: 10px;
    padding: 15px;
    margin-top: 20px;
    font-size: 18px;
    color: #2c3e50;
}
</style>
""", unsafe_allow_html=True)

# Loading models and encoders
try:
    classification_model = joblib.load('migraine_trigger_model.pkl')
    regression_model = joblib.load('migraine_trigger_regression_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
except FileNotFoundError:
    st.error("Model or encoder files not found. Please ensure they are in the correct directory.")
    st.stop()

# Setting page title
st.markdown('<div class="header">Migraine Prediction App</div>', unsafe_allow_html=True)

# Input section
st.markdown('<div class="subheader">Enter Patient Details</div>', unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        patient_name = st.text_input("Patient Name", value="John Doe")
        patient_id = st.number_input("Patient ID", min_value=1, step=1, value=1)
        diet = st.number_input("Diet (0 or 1)", min_value=0, max_value=1, step=1, value=0)
        sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=24.0, step=0.1, value=6.0)
        vomit_option = st.selectbox("Vomit", options=["No", "Yes"])
        vomit = 1 if vomit_option == "Yes" else 0
        frequency = st.number_input("Frequency of Migraines", min_value=0, max_value=10, step=1, value=1)
        intensity = st.number_input("Intensity of Pain (1-10)", min_value=1, max_value=10, step=1, value=1)
    
    with col2:
        visual = st.number_input("Visual Disturbances (0-4)", min_value=0, max_value=4, step=1, value=0)
        location_option = st.selectbox("Location of Pain", options=["N/A", "Back of head", "Left side", "Right side"])
        location = 0 if location_option == "N/A" else 1
        weather = st.selectbox("Weather", options=["Sunny", "Rainy", "Cloudy", "Humid", "Clear"])
        air_quality = st.number_input("Air Quality Index", min_value=0.0, max_value=100.0, step=0.1, value=10.0)
        type_of_migraine = st.selectbox("Type of Migraine", options=[
            "Typical aura with migraine", "Migraine without aura", 
            "Typical aura without migraine", "Other"
        ])
    st.markdown('</div>', unsafe_allow_html=True)

# Encoding categorical inputs
try:
    weather_encoded = label_encoders['Weather'].transform([weather])[0]
    type_encoded = label_encoders['Type'].transform([type_of_migraine])[0]
except ValueError as e:
    st.error(f"Error in encoding inputs: {e}")
    st.stop()

# Preparing input data
input_data = pd.DataFrame({
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

# Prediction button
if st.button("Predict Migraine"):
    try:
        # Classification prediction
        classification_prediction = classification_model.predict(input_data)[0]
        trigger_label = "Triggered" if classification_prediction == 1 else "Not Triggered"

        # Regression prediction
        predicted_time_hours = regression_model.predict(input_data)[0]
        predicted_hours = int(predicted_time_hours)
        predicted_minutes = int((predicted_time_hours - predicted_hours) * 60)

        # Displaying results
        st.markdown('<div class="result">', unsafe_allow_html=True)
        st.success(f"**Patient Name**: {patient_name}")
        st.success(f"**Predicted Trigger Status**: {trigger_label}")
        st.success(f"**Predicted Time Until Triggered**: {predicted_hours} hours and {predicted_minutes} minutes")
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error in prediction: {e}")
