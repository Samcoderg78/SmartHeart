import pandas as pd
import numpy as np
import pickle
import os

# === Set the default model to load here ===
# Options: "ensemble" (Voting ensemble), "logreg" (Logistic Regression only)
DEFAULT_MODEL_NAME = "ensemble"

MODEL_PATHS = {
    "ensemble": "models/saved_models/ensemble_model.pkl",
    "logreg": "models/saved_models/logreg_model.pkl" # This works if you saved one separately
}

class HeartRiskModel:
    def __init__(self, model_name=DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.features = None
        self.feature_mapping = None
        self.imputer = None
        self.load_model()

    def load_model(self):
        model_path = MODEL_PATHS[self.model_name]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        print(f"Loading model from {model_path}")
        with open(model_path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.features = data["features"]
            self.feature_mapping = data.get("feature_mapping", {})
            self.imputer = data.get("imputer", None)
        print(f"Model ({self.model_name}) loaded successfully.")

    def preprocess_user_data(self, user_data: dict):
        """
        Accepts a dict of user input. Returns processed features ready for model prediction.
        Missing values will be imputed as in training.
        """
        data = pd.DataFrame([user_data])
        # Map/cast all relevant fields as float, except those already int
        for k in ['age', 'total_cholesterol', 'systolic_bp', 'bmi', 'hdl_cholesterol', 'weight', 'height']:
            if k in data.columns:
                data[k] = data[k].astype(float)
        # Map to expected model columns
        df = pd.DataFrame()
        df['age'] = data['age']
        df['male'] = [1 if user_data.get('gender', 'Male') == 'Male' else 0]
        df['totChol'] = data['total_cholesterol']
        df['sysBP'] = data['systolic_bp']
        df['currentSmoker'] = [int(user_data.get('smoker', 0))]
        df['diabetes'] = [int(user_data.get('diabetes', 0))]
        df['BPMeds'] = [int(user_data.get('bp_treatment', 0))]
        df['BMI'] = data['bmi']
        df['age_squared'] = df['age'] ** 2
        if 'hdl_cholesterol' in user_data:
            df['hdl_cholesterol'] = data['hdl_cholesterol']
        else:
            df['hdl_cholesterol'] = df['totChol'] * 0.25
        df['chol_hdl_ratio'] = df['totChol'] / np.maximum(df['hdl_cholesterol'], 1)
        # Select feature columns
        df = df[self.features]
        # Impute missing if needed
        if self.imputer is not None:
            df = pd.DataFrame(self.imputer.transform(df), columns=self.features)
        # Scale
        df_scaled = self.scaler.transform(df)
        return df_scaled

    def predict_risk(self, user_data):
        """
        Predicts CVD risk given dict of user data. Returns probability, percentage, risk class.
        """
        X = self.preprocess_user_data(user_data)
        proba = self.model.predict_proba(X)[0, 1] * 100  # Probability (%) for CHD=1
        risk_category = self.get_risk_category(proba)
        return {
            "risk_percentage": proba,
            "risk_category": risk_category
        }

    def get_risk_category(self, risk_percentage):
        # You can adjust thresholds as you like!
        if risk_percentage < 5:
            return "Low Risk"
        elif risk_percentage < 10:
            return "Moderate Risk"
        elif risk_percentage < 20:
            return "High Risk"
        else:
            return "Very High Risk"

# Singleton loader
_model_instance = None
def get_model(model_name=None):
    global _model_instance
    if _model_instance is None or (model_name and _model_instance.model_name != model_name):
        _model_instance = HeartRiskModel(model_name if model_name else DEFAULT_MODEL_NAME)
    return _model_instance