import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files
import matplotlib.pyplot as plt

def load_models_and_data():
    df = pd.read_csv("data/tabular/structural_data.csv")
    shi_model = joblib.load("models/ml/shi_regressor.pkl")
    risk_model = joblib.load("models/ml/risk_classifier.pkl")
    metadata = joblib.load("models/ml/metadata.pkl")
    feature_cols = metadata["feature_cols"]
    X = df[feature_cols]
    return df, X, shi_model, risk_model, feature_cols

def generate_shap_plots(output_dir="outputs/shap"):
    os.makedirs(output_dir, exist_ok=True)
    df, X, shi_model, risk_model, feature_cols = load_models_and_data()

    # ---------- SHAP for SHI Regressor ----------
    print("Computing SHAP values for SHI Regressor...")
    shi_explainer = shap.Explainer(shi_model, X)
    shi_shap_values = shi_explainer(X)

    # Summary Bar Plot
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shi_shap_values, show=False)
    plt.title("SHAP Feature Importance — Structural Health Index (SHI)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shi_shap_bar.png"), dpi=120)
    plt.close()
    print("Saved: shi_shap_bar.png")

    # Beeswarm Summary Plot
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shi_shap_values, show=False)
    plt.title("SHAP Beeswarm — SHI Regressor", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shi_shap_beeswarm.png"), dpi=120)
    plt.close()
    print("Saved: shi_shap_beeswarm.png")

    # Waterfall for single sample (first row)
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shi_shap_values[0], show=False)
    plt.title("SHAP Waterfall — Sample #0 SHI Prediction", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shi_shap_waterfall.png"), dpi=120)
    plt.close()
    print("Saved: shi_shap_waterfall.png")

    # ---------- SHAP for Risk Classifier ----------
    print("Computing SHAP values for Risk Classifier...")
    risk_explainer = shap.Explainer(risk_model, X)
    risk_shap_values = risk_explainer(X)

    # Bar summary (class 2 = Critical)
    plt.figure(figsize=(10, 6))
    shap.plots.bar(risk_shap_values[:, :, 2], show=False)
    plt.title("SHAP Feature Importance — Risk 'Critical' Class", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "risk_shap_bar.png"), dpi=120)
    plt.close()
    print("Saved: risk_shap_bar.png")

    print(f"\nAll SHAP plots saved to: {output_dir}")

if __name__ == "__main__":
    generate_shap_plots()
