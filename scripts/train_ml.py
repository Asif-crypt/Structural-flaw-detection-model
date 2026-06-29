import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
from xgboost import XGBClassifier, XGBRegressor

def load_data(csv_path="data/tabular/structural_data.csv"):
    df = pd.read_csv(csv_path)
    return df

def train_models(df):
    feature_cols = [
        "building_age", "corrosion_level", "crack_width",
        "crack_density", "moisture_content", "compressive_strength",
        "temperature", "humidity", "load_stress"
    ]

    X = df[feature_cols]

    # ----- 1. SHI Regressor (XGBoost) -----
    y_shi = df["shi"]
    X_train, X_test, y_train, y_test = train_test_split(X, y_shi, test_size=0.2, random_state=42)
    shi_model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    shi_model.fit(X_train, y_train)
    shi_pred = shi_model.predict(X_test)
    shi_mae = mean_absolute_error(y_test, shi_pred)
    print(f"[SHI Regressor] Test MAE: {shi_mae:.2f}")

    # ----- 2. Risk Classifier (XGBoost) -----
    risk_map = {"Safe": 0, "Moderate": 1, "Critical": 2}
    y_risk = df["risk_level"].map(risk_map)
    X_train, X_test, y_train, y_test = train_test_split(X, y_risk, test_size=0.2, random_state=42)
    risk_model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                               use_label_encoder=False, eval_metric="mlogloss", random_state=42)
    risk_model.fit(X_train, y_train)
    risk_pred = risk_model.predict(X_test)
    risk_acc = accuracy_score(y_test, risk_pred)
    print(f"[Risk Classifier] Test Accuracy: {risk_acc * 100:.1f}%")

    # ----- 3. RUL Regressor (Random Forest) -----
    y_rul = df["rul"]
    X_train, X_test, y_train, y_test = train_test_split(X, y_rul, test_size=0.2, random_state=42)
    rul_model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rul_model.fit(X_train, y_train)
    rul_pred = rul_model.predict(X_test)
    rul_mae = mean_absolute_error(y_test, rul_pred)
    print(f"[RUL Regressor] Test MAE: {rul_mae:.2f} years")

    return shi_model, risk_model, rul_model

def save_models(shi_model, risk_model, rul_model, models_dir="models/ml"):
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(shi_model, os.path.join(models_dir, "shi_regressor.pkl"))
    joblib.dump(risk_model, os.path.join(models_dir, "risk_classifier.pkl"))
    joblib.dump(rul_model, os.path.join(models_dir, "rul_regressor.pkl"))

    # Also save the feature column order and risk label map for use at inference time
    metadata = {
        "feature_cols": [
            "building_age", "corrosion_level", "crack_width",
            "crack_density", "moisture_content", "compressive_strength",
            "temperature", "humidity", "load_stress"
        ],
        "risk_labels": {0: "Safe", 1: "Moderate", 2: "Critical"}
    }
    joblib.dump(metadata, os.path.join(models_dir, "metadata.pkl"))
    print(f"Models saved to {models_dir}/")

if __name__ == "__main__":
    df = load_data()
    print(f"Loaded dataset: {len(df)} rows, columns: {list(df.columns)}")
    shi_model, risk_model, rul_model = train_models(df)
    save_models(shi_model, risk_model, rul_model)
    print("All ML models trained and saved successfully!")
