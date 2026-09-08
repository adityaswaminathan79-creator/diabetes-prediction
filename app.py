"""
app.py
-------
This is the Streamlit web application. It loads the model + scaler
that train_model.py already saved, shows input boxes for a user,
and displays a prediction.

Run it from the terminal with:
    streamlit run app.py
"""

import pickle
import numpy as np
import streamlit as st

# ----------------------------------------------------------------------
# Load the trained model and scaler (created by train_model.py)
# ----------------------------------------------------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="Diabetes Prediction System", page_icon="🩺")
st.title("🩺 Diabetes Prediction System")
st.write(
    "Enter the patient's medical details below. The model will predict "
    "whether the patient is likely to have diabetes, based on patterns "
    "learned from the Pima Indians Diabetes dataset."
)

st.divider()

# ----------------------------------------------------------------------
# Input widgets — one per feature the model was trained on
# ----------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose level", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

# ----------------------------------------------------------------------
# Predict button
# ----------------------------------------------------------------------
if st.button("Predict"):
    # Build the input in the SAME column order the model was trained on
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                             skin_thickness, insulin, bmi, dpf, age]])

    # Scale it using the same scaler used during training
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]  # probability of class "1"

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ The model predicts **Diabetic** (confidence: {probability:.1%})")
    else:
        st.success(f"✅ The model predicts **Not Diabetic** (confidence: {(1-probability):.1%})")

    st.caption(
        "Disclaimer: This is a student ML project for learning purposes only, "
        "not a medical diagnostic tool."
    )
