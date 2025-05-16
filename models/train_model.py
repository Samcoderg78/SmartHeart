import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def train_risk_model():
    """Train the XGBoost model for heart disease risk prediction"""
    print("Loading data...")
    
    # Load the Framingham dataset
    try:
        data = pd.read_csv('../data/framingham.csv')
        print(f"Loaded dataset with {data.shape[0]} rows and {data.shape[1]} columns")
    except FileNotFoundError:
        print("Error: Framingham dataset not found. Please download it from Kaggle and place in data/ folder.")
        return
    
    # Print dataset info
    print("\nDataset columns:")
    print(data.columns.tolist())
    print("\nSample data:")
    print(data.head())
    
    # Handle missing values
    print(f"\nMissing values before cleaning:\n{data.isnull().sum()}")
    data = data.dropna()
    print(f"Dataset shape after dropping missing values: {data.shape}")
    
    # Feature engineering
    print("\nPerforming feature engineering...")
    data['age_squared'] = data['age'] ** 2
    
    # The dataset doesn't have HDL cholesterol, so we'll estimate it
    # Typically, HDL is about 20-30% of total cholesterol
    data['hdl_cholesterol'] = data['totChol'] * 0.25  # Assuming HDL is 25% of total cholesterol
    data['chol_hdl_ratio'] = data['totChol'] / data['hdl_cholesterol']
    
    # Prepare features and target
    features = [
        'age', 'male', 'totChol', 'hdl_cholesterol', 'sysBP', 
        'currentSmoker', 'diabetes', 'BPMeds', 'BMI', 'age_squared', 'chol_hdl_ratio'
    ]
    
    X = data[features]
    y = data['TenYearCHD']  # 10-year risk of coronary heart disease
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost model
    print("\nTraining XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nModel Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Save model and scaler
    print("\nSaving model...")
    os.makedirs('models/saved_models', exist_ok=True)
    
    # Save feature names mapping to ensure consistent naming between training and prediction
    feature_mapping = {
        'male': 'gender_numeric',
        'totChol': 'total_cholesterol',
        'sysBP': 'systolic_bp',
        'currentSmoker': 'smoker',
        'BPMeds': 'bp_treatment',
        'BMI': 'bmi'
    }
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': features,
        'feature_mapping': feature_mapping,
        'metrics': {
            'accuracy': accuracy,
            'auc': auc
        }
    }
    
    with open('models/saved_models/xgb_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved successfully to models/saved_models/xgb_model.pkl")

if __name__ == "__main__":
    train_risk_model()
