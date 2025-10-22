# ==========================================
# train_model.py
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# ===============================
# Step 1: Load dataset
# ===============================
df = pd.read_csv("car details v4.csv")

# ===============================
# Step 2: Handle missing values
# ===============================
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna("Unknown", inplace=True)

# ===============================
# Step 3: Encode categorical columns
# ===============================
cat_cols = df.select_dtypes(include=['object']).columns
encoder = LabelEncoder()

for col in cat_cols:
    df[col] = encoder.fit_transform(df[col].astype(str))

# ===============================
# Step 4: Split into features and target
# ===============================
X = df.drop(columns=['Price'])
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# Step 5: Train Gradient Boosting Model with GridSearch
# ===============================
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4]
}

gb = GradientBoostingRegressor(random_state=42)
grid = GridSearchCV(gb, param_grid, cv=5, scoring='r2', verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# ===============================
# Step 6: Evaluate the model
# ===============================
y_pred = best_model.predict(X_test)
print("\n✅ Model Evaluation Results:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
print("\nBest Parameters:", grid.best_params_)

# ===============================
# Step 7: Save the model
# ===============================
with open("car_price_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("\n✅ Model saved successfully as car_price_model.pkl")
