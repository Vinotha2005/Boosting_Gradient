# 🚗 Car Price Prediction using Gradient Boosting

## 📘 Project Overview

This project predicts the price of used cars based on various attributes such as make, model, fuel type, engine, mileage, and dimensions.
The dataset was cleaned, preprocessed, and trained using Gradient Boosting Regressor with GridSearchCV for hyperparameter tuning.

## App for Car Price Prediction:


## 🧠 Objectives

Clean and preprocess car dataset

Handle missing and categorical data

Detect and treat outliers

Train multiple regression models

Optimize Gradient Boosting using GridSearchCV

Evaluate performance using MAE, MSE, and R² score

## 🗂️ Dataset Description

The dataset contains car listings with specifications and target price.
Below are the key columns used:

### Feature	Description
Make	Car manufacturer (e.g., BMW, Toyota, Maruti Suzuki)
Model	Specific car model
Year	Manufacturing year
Kilometer	Distance driven
Fuel Type	Type of fuel (Petrol/Diesel/CNG/Hybrid)
Transmission	Manual or Automatic
Engine	Engine capacity (in cc)
Max Power	Horsepower
Max Torque	Torque
Drivetrain	FWD, RWD, AWD
Length / Width / Height	Car dimensions
Seating Capacity	Number of seats
Fuel Tank Capacity	Tank size (litres)
Price	Target variable (car price)

## ⚙️ Steps Performed
### 1️⃣ Data Preprocessing

Removed duplicates

Identified and imputed missing values

Converted string-based features to numeric using Label Encoding

Treated skewness and outliers

### 2️⃣ Exploratory Data Analysis (EDA)

Visualized data distributions

Checked correlation between features and target

Detected outliers using boxplots

### 3️⃣ Model Training

Split data into train/test (80/20)

Used Gradient Boosting Regressor

Performed GridSearchCV for hyperparameter tuning

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

### 4️⃣ Evaluation Metrics

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

R² Score

📊 Example Results
Metric	Score
MAE	1.24
MSE	3.21
R² Score	0.92

(Values are illustrative — your actual results may vary depending on dataset.)

## Visualization

### Skew Transformation

<img width="850" height="393" alt="image" src="https://github.com/user-attachments/assets/bd2648a2-58e6-4409-bc58-47b533ac9956" />

### Outliers
<img width="1489" height="2789" alt="image" src="https://github.com/user-attachments/assets/11eefcf1-9e53-4de3-bdb7-d659b8cff93f" />

### Actual vs Predicted Gradient boosting
<img width="691" height="547" alt="image" src="https://github.com/user-attachments/assets/f3e3258a-edac-4a37-a6d0-7d620748f4fc" />


## 🧩 Dependencies

Install all necessary libraries before running the notebook:

pip install pandas numpy matplotlib seaborn scikit-learn

## 🚀 How to Run

Clone this repository

git clone https://github.com/yourusername/car-price-prediction.git
cd car-price-prediction


Run the Python script

python train_model.py


(Optional) Launch Streamlit app if you built one:

streamlit run app.py

## 📈 Future Improvements

Integrate XGBoost or LightGBM for comparison

Deploy using Streamlit or Flask

Enhance feature engineering (mileage, torque extraction)

Implement automatic outlier handling
