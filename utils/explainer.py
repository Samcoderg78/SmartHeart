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

    def generate_lifestyle_explanation(self, original_data, modified_data, language="en"):
        """
        Generate written explanations for lifestyle changes and their impact on heart disease risk.
        """
        original_risk = self.risk_model.predict_risk(original_data)
        new_risk = self.risk_model.predict_risk(modified_data)
        risk_change = new_risk['risk_percentage'] - original_risk['risk_percentage']
        
        explanations = []
        
        # Weight/BMI changes
        if abs(modified_data.get('weight', 0) - original_data.get('weight', 0)) > 1:
            weight_diff = modified_data.get('weight', 0) - original_data.get('weight', 0)
            bmi_diff = modified_data.get('bmi', 0) - original_data.get('bmi', 0)
            
            if weight_diff < 0:
                explanations.append(f"**Weight Loss Impact:** Reducing your weight by {abs(weight_diff):.1f} kg (BMI decrease of {abs(bmi_diff):.1f}) can significantly lower your heart disease risk. Excess weight puts strain on your heart and increases blood pressure.")
            else:
                explanations.append(f"**Weight Gain Impact:** Increasing your weight by {weight_diff:.1f} kg (BMI increase of {bmi_diff:.1f}) can raise your heart disease risk. Higher BMI is associated with increased cardiovascular risk factors.")
        
        # Smoking changes
        if original_data.get('smoker', 0) == 1 and modified_data.get('smoker', 0) == 0:
            explanations.append("**Quitting Smoking:** This is one of the most impactful changes you can make! Quitting smoking immediately begins to improve your cardiovascular health. Within just 1 year, your risk of heart disease drops by half compared to a smoker.")
        
        # Cholesterol changes
        chol_diff = modified_data.get('total_cholesterol', 0) - original_data.get('total_cholesterol', 0)
        if abs(chol_diff) > 5:
            if chol_diff < 0:
                explanations.append(f"**Cholesterol Reduction:** Lowering your total cholesterol by {abs(chol_diff):.0f} mg/dL can significantly reduce your heart disease risk. High cholesterol contributes to plaque buildup in arteries.")
            else:
                explanations.append(f"**Cholesterol Increase:** Raising your total cholesterol by {chol_diff:.0f} mg/dL may increase your heart disease risk. Elevated cholesterol levels are a major risk factor for cardiovascular disease.")
        
        # HDL changes
        hdl_diff = modified_data.get('hdl_cholesterol', 0) - original_data.get('hdl_cholesterol', 0)
        if abs(hdl_diff) > 2:
            if hdl_diff > 0:
                explanations.append(f"**HDL Improvement:** Increasing your HDL (good cholesterol) by {hdl_diff:.0f} mg/dL is excellent for heart health. HDL helps remove bad cholesterol from your arteries.")
            else:
                explanations.append(f"**HDL Decrease:** Lowering your HDL by {abs(hdl_diff):.0f} mg/dL may increase your heart disease risk. HDL is protective against heart disease.")
        
        # Blood pressure changes
        bp_diff = modified_data.get('systolic_bp', 0) - original_data.get('systolic_bp', 0)
        if abs(bp_diff) > 5:
            if bp_diff < 0:
                explanations.append(f"**Blood Pressure Reduction:** Lowering your systolic blood pressure by {abs(bp_diff):.0f} mmHg can significantly reduce your heart disease risk. High blood pressure damages blood vessels and increases heart strain.")
            else:
                explanations.append(f"**Blood Pressure Increase:** Raising your systolic blood pressure by {bp_diff:.0f} mmHg may increase your heart disease risk. Elevated blood pressure is a major cardiovascular risk factor.")
        
        # BP medication
        if original_data.get('bp_treatment', 0) == 0 and modified_data.get('bp_treatment', 0) == 1:
            explanations.append("**Blood Pressure Medication:** Starting BP medication can help control high blood pressure and reduce your heart disease risk. These medications work by relaxing blood vessels and reducing the heart's workload.")
        
        # Diabetes
        if original_data.get('diabetes', 0) == 0 and modified_data.get('diabetes', 0) == 1:
            explanations.append("**Diabetes Development:** Developing diabetes significantly increases your heart disease risk. Diabetes damages blood vessels and increases the likelihood of cardiovascular complications.")
        
        # Overall risk change explanation
        if abs(risk_change) > 0.5:
            if risk_change < 0:
                risk_explanation = f"**Overall Impact:** Your heart disease risk has decreased by {abs(risk_change):.1f} percentage points. These lifestyle changes are working in your favor!"
            else:
                risk_explanation = f"**Overall Impact:** Your heart disease risk has increased by {risk_change:.1f} percentage points. Consider focusing on positive lifestyle changes to reduce your risk."
        else:
            risk_explanation = "**Overall Impact:** Your heart disease risk has changed minimally. Small changes can add up over time - consistency is key!"
        
        explanations.append(risk_explanation)
        
        # Risk category explanation
        if new_risk['risk_category'] != original_risk['risk_category']:
            if new_risk['risk_percentage'] < original_risk['risk_percentage']:
                category_explanation = f"**Risk Category Improvement:** Your risk category has improved from '{original_risk['risk_category']}' to '{new_risk['risk_category']}'. Keep up the good work!"
            else:
                category_explanation = f"**Risk Category Change:** Your risk category has changed from '{original_risk['risk_category']}' to '{new_risk['risk_category']}'. Consider consulting with your healthcare provider."
            explanations.append(category_explanation)
        
        return explanations

# Singleton instance
_explainer_instance = None

def get_explainer(model_name=None):
    global _explainer_instance
    if _explainer_instance is None or (model_name and _explainer_instance.risk_model.model_name != model_name):
        _explainer_instance = RiskExplainer(model_name)
    return _explainer_instance