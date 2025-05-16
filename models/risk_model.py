import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import os

class HeartRiskModel:
    def __init__(self, model_path='models/saved_models/xgb_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.features = None
        self.feature_mapping = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            if os.path.exists(self.model_path):
                print(f"Loading model from {self.model_path}")
                with open(self.model_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.model = saved_data['model']
                    self.scaler = saved_data['scaler']
                    self.features = saved_data.get('features', None)
                    self.feature_mapping = saved_data.get('feature_mapping', {})
                    
                print("Model loaded successfully!")
                if 'metrics' in saved_data:
                    print(f"Model metrics: Accuracy={saved_data['metrics']['accuracy']:.4f}, AUC={saved_data['metrics']['auc']:.4f}")
            else:
                print(f"Warning: Model file not found at {self.model_path}. Using fallback model.")
                self._create_fallback_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Create a simple fallback model for demo purposes"""
        print("Creating fallback model and scaler")
        # Create a simple XGBoost model
        self.model = xgb.XGBClassifier(n_estimators=10)
        
        # Create and fit a scaler with dummy data
        self.scaler = StandardScaler()
        dummy_data = np.random.randn(100, 11)  # 11 features as expected by the model
        self.scaler.fit(dummy_data)
        
        # Default features (using original Framingham dataset column names)
        self.features = [
            'age', 'male', 'totChol', 'hdl_cholesterol', 'sysBP', 
            'currentSmoker', 'diabetes', 'BPMeds', 'BMI', 'age_squared', 'chol_hdl_ratio'
        ]
        
        # Default feature mapping
        self.feature_mapping = {
            'male': 'gender_numeric',
            'totChol': 'total_cholesterol',
            'sysBP': 'systolic_bp',
            'currentSmoker': 'smoker',
            'BPMeds': 'bp_treatment',
            'BMI': 'bmi'
        }
        
        # Set up model with dummy predict_proba method
        self.model.predict_proba = lambda x: np.array([[0.85, 0.15]] * len(x))
    
    def preprocess_data(self, user_data):
        """Preprocess the input data for the model"""
        # Create a DataFrame with the user data
        data = pd.DataFrame([user_data])
        
        # Map from UI field names to dataset field names
        data_for_model = pd.DataFrame()
        
        # Map gender to 'male' (1 for Male, 0 for Female)
        data_for_model['male'] = [1 if user_data['gender'] == 'Male' else 0]
        
        # Map other fields according to the feature mapping
        # Direct mappings
        data_for_model['age'] = data['age']
        data_for_model['diabetes'] = data['diabetes'].astype(int)
        
        # Mapped fields
        data_for_model['totChol'] = data['total_cholesterol']
        data_for_model['sysBP'] = data['systolic_bp']
        data_for_model['currentSmoker'] = data['smoker'].astype(int)
        data_for_model['BPMeds'] = data['bp_treatment'].astype(int)
        data_for_model['BMI'] = data['bmi']
        
        # Calculate derived features
        data_for_model['age_squared'] = data_for_model['age'] ** 2
        data_for_model['hdl_cholesterol'] = data_for_model['totChol'] * 0.25  # Estimate HDL
        data_for_model['chol_hdl_ratio'] = data_for_model['totChol'] / data_for_model['hdl_cholesterol']
        
        # Select features used by the model
        if self.features:
            X = data_for_model[self.features].values
        else:
            # Fallback feature list if not loaded with model
            X = data_for_model[[
                'age', 'male', 'totChol', 'hdl_cholesterol', 'sysBP', 
                'currentSmoker', 'diabetes', 'BPMeds', 'BMI', 'age_squared', 'chol_hdl_ratio'
            ]].values
        
        # Scale features
        if self.scaler:
            X = self.scaler.transform(X)
        
        return X
    
    def predict_risk(self, user_data):
        """Predict 10-year cardiovascular risk for a user"""
        X = self.preprocess_data(user_data)
        
        # Predict using XGBoost model
        risk_probability = self.model.predict_proba(X)[0, 1]  # Probability of class 1 (has CVD risk)
        
        # Convert to percentage
        risk_percentage = risk_probability * 100
        
        # Get risk category
        risk_category = self.get_risk_category(risk_percentage)
        
        return {
            'risk_percentage': risk_percentage,
            'risk_category': risk_category
        }
    
    def get_risk_category(self, risk_percentage):
        """Categorize risk level based on percentage"""
        if risk_percentage < 5:
            return "Low Risk"
        elif risk_percentage < 10:
            return "Moderate Risk"
        elif risk_percentage < 20:
            return "High Risk"
        else:
            return "Very High Risk"

# Singleton instance
_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = HeartRiskModel()
    return _model_instance
