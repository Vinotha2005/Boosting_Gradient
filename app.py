# ==========================================
# app.py
# ==========================================

import streamlit as st
import pickle
import numpy as np

# ===============================
# Load trained model
# ===============================
with open("car_price_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="🚗 Car Price Prediction App", layout="wide")

st.title("🚗 Car Price Prediction using Gradient Boosting")
st.markdown("### Enter the car details below to predict its price")

# ===============================
# Input fields
# ===============================
col1, col2, col3 = st.columns(3)

with col1:
    year = st.number_input("Year", min_value=1990, max_value=2025, value=2018)
    kilometer = st.number_input("Kilometer Driven", min_value=0, max_value=500000, value=30000)
    fuel_type = st.number_input("Fuel Type (Encoded)", min_value=0, value=1)
    transmission = st.number_input("Transmission (Encoded)", min_value=0, value=1)
    engine = st.number_input("Engine (cc)", min_value=500, max_value=6000, value=2000)

with col2:
    max_power = st.number_input("Max Power (Encoded)", min_value=0, value=100)
    max_torque = st.number_input("Max Torque (Encoded)", min_value=0, value=200)
    drivetrain = st.number_input("Drivetrain (Encoded)", min_value=0, value=1)
    seating_capacity = st.number_input("Seating Capacity", min_value=2, max_value=10, value=5)
    fuel_tank_capacity = st.number_input("Fuel Tank Capacity", min_value=20, max_value=100, value=50)

with col3:
   with col3:
    length = st.number_input("Length (mm)", min_value=3000, max_value=6000, value=4500)
    width = st.number_input("Width (mm)", min_value=1000, max_value=2500, value=1800)
    height = st.number_input("Height (mm)", min_value=1000, max_value=2000, value=1500)
    make = st.number_input("Make (Encoded)", min_value=0, value=1)
    model_col = st.number_input("Model (Encoded)", min_value=0, value=1)
    location = st.number_input("Location (Encoded)", min_value=0, value=1)
    color = st.number_input("Color (Encoded)", min_value=0, value=1)
    owner = st.number_input("Owner (Encoded)", min_value=0, value=1)
    seller_type = st.number_input("Seller Type (Encoded)", min_value=0, value=1)

# ===============================
# Predict Button
# ===============================
if st.button("🔮 Predict Price"):
    try:
        input_data = np.array([[
    make, model_col, year, kilometer, fuel_type, transmission,
    location, color, owner, seller_type, engine, max_power, max_torque,
    drivetrain, length, width, height, seating_capacity, fuel_tank_capacity
]])

        prediction = model.predict(input_data)[0]
        st.success(f"💰 Estimated Car Price: ₹ {prediction:,.2f}")
    except Exception as e:
        st.error(f"❌ Error: {e}")
