import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessing tools
model = joblib.load("rf_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("House Price Prediction")
st.markdown("Enter the house features below to predict its price.")

# User inputs
area = st.slider("Area (in sq ft)", min_value=500, max_value=5000, value=1500, step=100)
bedrooms = st.selectbox("Number of Bedrooms", options=list(range(1, 11)))
bathrooms = st.selectbox("Number of Bathrooms", options=list(range(1, 11)))
floors = st.selectbox("Number of Floors", options=list(range(1, 4)))
year_built = st.slider("Year Built", min_value=1900, max_value=2023, value=2000)

location = st.selectbox("Location", options=label_encoders["Location"].classes_)
condition = st.selectbox("Condition", options=label_encoders["Condition"].classes_)
garage = st.selectbox("Garage", options=label_encoders["Garage"].classes_)

# Encode categorical
location_encoded = label_encoders["Location"].transform([location])[0]
condition_encoded = label_encoders["Condition"].transform([condition])[0]
garage_encoded = label_encoders["Garage"].transform([garage])[0]

# Scale numeric
numerical_features = pd.DataFrame([[area, bedrooms, bathrooms, floors, year_built]],
                                  columns=['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt'])
numerical_scaled = scaler.transform(numerical_features)[0]

# Final input
input_data = np.concatenate((numerical_scaled, [location_encoded, condition_encoded, garage_encoded])).reshape(1, -1)

# Predict
if st.button("Predict House Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Price: ${prediction:,.2f}")
