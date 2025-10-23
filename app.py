# ==================================================
# app.py — Streamlit Car Price Prediction App
# ==================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==================================================
# Load model and encoder data
# ==================================================
with open("model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
encoders = model_data["encoders"]
feature_names = model_data["features"]

# ==================================================
# Page setup
# ==================================================
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")
st.title("🚗 Used Car Price Prediction App")
st.write("Enter or select the car details below to predict its **price** using a trained Gradient Boosting model.")

# ==================================================
# Load dataset to extract unique categorical values
# ==================================================
df = pd.read_csv("car details v4.csv")

# Identify categorical and numeric columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols = [col for col in feature_names if col not in cat_cols and col != "Price"]

# ==================================================
# Input form with user-friendly UI
# ==================================================
st.subheader("🔧 Car Feature Inputs")

cols = st.columns(3)
user_input = {}

for i, col in enumerate(feature_names):
    with cols[i % 3]:
        if col in cat_cols:
            # Dropdown for categorical columns
            options = sorted(df[col].dropna().astype(str).unique().tolist())
            selected = st.selectbox(f"{col}", options)
            user_input[col] = selected
        elif col == "Year":
            user_input[col] = st.number_input(f"{col}", min_value=1980, max_value=2025, value=2020)
        elif col == "Kilometer":
            user_input[col] = st.number_input(f"{col}", min_value=0, max_value=500000, value=50000)
        elif col in ["Length", "Width", "Height"]:
            user_input[col] = st.number_input(f"{col} (mm)", min_value=0.0, value=4000.0, step=10.0)
        elif col in ["Seating Capacity", "Fuel Tank Capacity"]:
            user_input[col] = st.number_input(f"{col}", min_value=0.0, value=5.0, step=1.0)
        else:
            user_input[col] = st.number_input(f"{col}", value=0.0)

# ==================================================
# Convert input to DataFrame
# ==================================================
if st.button("🔮 Predict Price"):
    try:
        input_df = pd.DataFrame([user_input])

        # Encode categorical columns
        for col in encoders.keys():
            if col in input_df.columns:
                le = encoders[col]
                val = input_df[col].iloc[0]
                if val not in le.classes_:
                    st.warning(f"⚠️ Unknown value '{val}' for {col}, replacing with 'Unknown'")
                    le.classes_ = np.append(le.classes_, "Unknown")
                    input_df[col] = le.transform(["Unknown"])
                else:
                    input_df[col] = le.transform([val])

        # Convert numeric columns
        for col in input_df.columns:
            if col not in encoders.keys():
                input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0)

        # Reorder columns
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Predict
        pred = model.predict(input_df)[0]

        st.success(f"💰 Predicted Car Price: ₹ {pred:,.2f}")

        st.write("### 🔍 Input Summary")
        st.dataframe(input_df)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
