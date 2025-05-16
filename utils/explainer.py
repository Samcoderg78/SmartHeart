import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from models.risk_model import get_model
import os

class RiskExplainer:
    def __init__(self, model_name=None):
        self.risk_model = get_model(model_name)
        self.explainer = None
        self.background = None
        self.shap_ready = False
        self.initialize_shap_explainer()

    def initialize_shap_explainer(self):
        """Initialize a SHAP explainer compatible with model type"""
        try:
            # 1. Prepare background data for explainer initialization.
            # Use a small amount of real user-like data if possible.
            features = self.risk_model.features
            n_bg = min(50, len(features)*4)
            background_data = None

            if os.path.exists('data/framingham.csv'):
                bg_df = pd.read_csv('data/framingham.csv').dropna().sample(n=n_bg, random_state=42)
                processed_rows = []
                for _, row in bg_df.iterrows():
                    user_data = {
                        'age': row['age'],
                        'gender': 'Male' if row['male'] == 1 else 'Female',
                        'total_cholesterol': row['totChol'],
                        'hdl_cholesterol': row['totChol'] * 0.25,
                        'systolic_bp': row['sysBP'],
                        'smoker': int(row['currentSmoker']),
                        'diabetes': int(row['diabetes']),
                        'bp_treatment': int(row['BPMeds']),
                        'bmi': row['BMI'],
                        'weight': 70,  # fill with median if not available
                        'height': 170,
                    }
                    processed_rows.append(self.risk_model.preprocess_user_data(user_data)[0])
                background_data = np.stack(processed_rows)
            else:
                background_data = np.zeros((n_bg, len(features)))  # fallback: zeros

            self.background = background_data

            # 2. Choose SHAP explainer type according to model
            from sklearn.ensemble import VotingClassifier, RandomForestClassifier
            from sklearn.linear_model import LogisticRegression

            model = self.risk_model.model
            # VotingClassifier or Logistics? Use KernelExplainer.
            if (isinstance(model, VotingClassifier)
                or isinstance(model, LogisticRegression)):
                self.explainer = shap.KernelExplainer(model.predict_proba, self.background)
            # XGBoost/RandomForest: use TreeExplainer
            elif hasattr(model, "feature_importances_") or "xgboost" in str(type(model)).lower() or "forest" in str(type(model)).lower():
                self.explainer = shap.TreeExplainer(model, self.background)
            else:
                self.explainer = shap.KernelExplainer(model.predict_proba, self.background)

            self.shap_ready = self.explainer is not None

        except Exception as e:
            print(f"[SHAP WARNING] Could not initialize explainer: {e}")
            self.explainer = None
            self.shap_ready = False

    def get_shap_values(self, user_data):
        if not self.shap_ready or self.explainer is None:
            raise RuntimeError("SHAP explainer is not initialized for this model type or environment.")
        X = self.risk_model.preprocess_user_data(user_data)
        # For Classification: shap_values is a list with an array for each class, choose class 1
        shap_vals = self.explainer.shap_values(X)
        if isinstance(shap_vals, list) and len(shap_vals) > 1:
            return shap_vals[1]
        return shap_vals

    def generate_shap_plot(self, user_data):
        X = self.risk_model.preprocess_user_data(user_data)
        shap_values = self.get_shap_values(user_data)
        feature_names = self.risk_model.features
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.tight_layout()
        return plt

    def simulate_intervention(self, user_data, interventions):
        original_risk = self.risk_model.predict_risk(user_data)
        modified_data = user_data.copy()
        for key, value in interventions.items():
            modified_data[key] = value
        new_risk = self.risk_model.predict_risk(modified_data)
        try:
            original_shap = self.get_shap_values(user_data)
            new_shap = self.get_shap_values(modified_data)
        except Exception:
            original_shap = new_shap = None
        risk_difference = new_risk['risk_percentage'] - original_risk['risk_percentage']
        return {
            'original_risk': original_risk,
            'new_risk': new_risk,
            'risk_difference': risk_difference,
            'original_shap': original_shap,
            'new_shap': new_shap
        }

# Singleton instance
_explainer_instance = None

def get_explainer(model_name=None):
    global _explainer_instance
    if _explainer_instance is None or (model_name and _explainer_instance.risk_model.model_name != model_name):
        _explainer_instance = RiskExplainer(model_name)
    return _explainer_instance