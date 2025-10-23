# ==================================================
# train_model.py — Train Gradient Boosting Model
# ==================================================

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==================================================
# Step 1 — Load the dataset
# ==================================================
df = pd.read_csv("car details v4.csv")

print("✅ Dataset Loaded Successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ==================================================
# Step 2 — Handle missing values
# ==================================================
print("\n🔍 Checking for missing values...")
print(df.isna().sum())

# Fill numeric NaNs with median and categorical with mode
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

print("✅ Missing values handled.")

# ==================================================
# Step 3 — Encode categorical columns
# ==================================================
encoders = {}
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

print("✅ All categorical columns converted to numeric.")

# ==================================================
# Step 4 — Split features and target
# ==================================================
X = df.drop(columns=['Price'])
y = df['Price']
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Data split completed.")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# ==================================================
# Step 5 — Gradient Boosting Model + Hyperparameter Tuning
# ==================================================
print("\n🚀 Training Gradient Boosting Model...")

param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 4],
    "subsample": [0.8, 1.0]
}

gb = GradientBoostingRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=gb,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\n✅ Best Hyperparameters:", grid_search.best_params_)

# ==================================================
# Step 6 — Evaluate model
# ==================================================
y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n📊 Model Performance:")
print(f"R² Score  : {r2:.4f}")
print(f"MAE       : {mae:.2f}")
print(f"RMSE      : {rmse:.2f}")

# ==================================================
# Step 7 — Save model, encoders, and feature names
# ==================================================
model_data = {
    "model": best_model,
    "encoders": encoders,
    "features": feature_names
}

with open("model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("\n💾 Model and encoders saved to model.pkl successfully!")
