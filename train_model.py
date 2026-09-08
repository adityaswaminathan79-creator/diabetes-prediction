"""
train_model.py
----------------
This script does the "offline" part of the ML workflow:
1. Load the dataset
2. Clean/preprocess it
3. Split into train/test sets
4. Train a model
5. Evaluate it
6. Save the trained model + scaler to disk (so the Streamlit app can reuse them)

Run this once from the terminal:  python train_model.py
"""

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ----------------------------------------------------------------------
# 1. LOAD THE DATA
# ----------------------------------------------------------------------
df = pd.read_csv("diabetes.csv")
print("Shape of dataset:", df.shape)
print(df.head())

# ----------------------------------------------------------------------
# 2. PREPROCESSING
# ----------------------------------------------------------------------
# In this dataset, a value of 0 in these medical columns is not a real
# measurement (you can't have 0 blood pressure) — it means "missing data"
# that was recorded as 0. We replace those 0s with NaN, then fill them
# with the median of that column, which is a common, simple strategy.
cols_with_invalid_zeros = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
]

for col in cols_with_invalid_zeros:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# Separate features (X) from the target/label (y)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split into training data (80%) and testing data (20%).
# random_state=42 just makes the split reproducible every time you run it.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling: puts all numeric columns on a similar scale.
# Many models (and especially distance-based ones) perform better this way.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------------------
# 3. TRAIN THE MODEL
# ----------------------------------------------------------------------
# RandomForestClassifier: an ensemble of decision trees. Good default
# choice for tabular data like this — robust and doesn't need much tuning.
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# ----------------------------------------------------------------------
# 4. EVALUATE THE MODEL
# ----------------------------------------------------------------------
y_pred = model.predict(X_test_scaled)

print("\n--- MODEL EVALUATION ---")
print("Accuracy :", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(precision_score(y_test, y_pred), 3))
print("Recall   :", round(recall_score(y_test, y_pred), 3))
print("F1 Score :", round(f1_score(y_test, y_pred), 3))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nFull Report:\n", classification_report(y_test, y_pred))

# Which features mattered most to the model? (nice to mention in a viva/demo)
importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nFeature importances:\n", importances.sort_values(ascending=False))

# ----------------------------------------------------------------------
# 5. SAVE THE MODEL + SCALER
# ----------------------------------------------------------------------
# We save both, because the Streamlit app must scale new user input
# using the SAME scaler that was fit on the training data.
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nSaved model.pkl and scaler.pkl — you're ready to run the Streamlit app.")
