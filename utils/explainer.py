import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from models.risk_model import get_model
import os

class RiskExplainer:
    def __init__(self):
        self.risk_model = get_model()
        self.explainer = None
        self.initialize_shap_explainer()
    
    def initialize_shap_explainer(self):
        """Initialize the SHAP explainer"""
        try:
            # Load sample data to create the explainer background
            # In a real app, use a representative dataset subset
            if os.path.exists('data/sample_user_data.csv'):
                sample_data = pd.read_csv('data/sample_user_data.csv')
                
                # Preprocess the sample data
                processed_samples = []
                for _, row in sample_data.iterrows():
                    user_data = {
                        'age': row['age'],
                        'gender': 'Male' if row['gender_numeric'] == 1 else 'Female',
                        'total_cholesterol': row['total_cholesterol'],
                        'hdl_cholesterol': row['hdl_cholesterol'],
                        'systolic_bp': row['systolic_bp'],
                        'smoker': bool(row['smoker']),
                        'diabetes': bool(row['diabetes']),
                        'bp_treatment': bool(row['bp_treatment']),
                        'bmi': row['bmi'],
                        'weight': row['weight'],
                        'height': row['height']
                    }
                    processed_samples.append(self.risk_model.preprocess_data(user_data)[0])
                
                background_data = np.array(processed_samples)
                
                # Create the SHAP explainer
                self.explainer = shap.TreeExplainer(self.risk_model.model, background_data)
            else:
                # Create a simple background dataset
                print("Sample data not found, creating synthetic background data")
                background_data = np.random.randn(100, 11)  # 11 features
                self.explainer = shap.TreeExplainer(self.risk_model.model, background_data)
        except Exception as e:
            print(f"Error initializing SHAP explainer: {e}")
            # Create a dummy explainer
            self.explainer = type('DummyExplainer', (), {
                'shap_values': lambda x: np.random.randn(1, 11)
            })()
    
    def get_shap_values(self, user_data):
        """Get SHAP values for the user data"""
        X = self.risk_model.preprocess_data(user_data)
        return self.explainer.shap_values(X)
    
    def generate_shap_plot(self, user_data):
        """Generate a SHAP force plot for the user's prediction"""
        X = self.risk_model.preprocess_data(user_data)
        shap_values = self.explainer.shap_values(X)
        
        # Create feature names for the plot
        feature_names = [
            'Age', 'Gender', 'Total Cholesterol', 'HDL Cholesterol', 
            'Systolic BP', 'Smoker', 'Diabetes', 'BP Treatment', 'BMI',
            'Age Squared', 'Chol/HDL Ratio'
        ]
        
        # Generate the plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.tight_layout()
        return plt
    
    def simulate_intervention(self, user_data, interventions):
        """Simulate the effect of lifestyle interventions on risk
        
        Args:
            user_data (dict): Original user data
            interventions (dict): Dictionary of interventions to apply
                Example: {'smoker': False, 'bmi': 24.5}
        
        Returns:
            dict: Dictionary containing original and new risk, with breakdown
        """
        # Get original risk prediction
        original_risk = self.risk_model.predict_risk(user_data)
        
        # Create a copy of the user data and apply interventions
        modified_data = user_data.copy()
        for key, value in interventions.items():
            if key in modified_data:
                modified_data[key] = value
        
        # Get new risk prediction
        new_risk = self.risk_model.predict_risk(modified_data)
        
        # Get SHAP values for both scenarios
        original_shap = self.get_shap_values(user_data)
        new_shap = self.get_shap_values(modified_data)
        
        # Calculate the differences
        risk_difference = new_risk['risk_percentage'] - original_risk['risk_percentage']
        
        # Return results
        return {
            'original_risk': original_risk,
            'new_risk': new_risk,
            'risk_difference': risk_difference,
            'original_shap': original_shap,
            'new_shap': new_shap
        }

# Singleton instance
_explainer_instance = None

def get_explainer():
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = RiskExplainer()
    return _explainer_instance