import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb

def train_risk_model():
    # 1. Load data
    data_path = 'data/framingham.csv'
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    print("\nInitial dataset shape:", data.shape)
    print("Missing values per column:\n", data.isnull().sum())

    # 2. Impute missing values on only those columns you need
    #    These columns will be used in feature engineering or as final features!
    for col in ['totChol', 'BPMeds', 'BMI', 'cigsPerDay', 'heartRate', 'glucose']:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
    # Systolic BP shouldn't be missing. If so, fill for robustness
    data['sysBP'] = data['sysBP'].fillna(data['sysBP'].median())

    # 3. Feature engineering (add only engineered features you'll use)
    data['age_squared'] = data['age'] ** 2
    data['hdl_cholesterol'] = data['totChol'] * 0.25
    data['chol_hdl_ratio'] = data['totChol'] / np.maximum(data['hdl_cholesterol'], 1)

    # 4. Select features for ML -- ONLY these columns will be used by your app!
    features = [
        'age', 'male', 'totChol', 'hdl_cholesterol', 'sysBP',
        'currentSmoker', 'diabetes', 'BPMeds', 'BMI', 'age_squared', 'chol_hdl_ratio'
    ]
    # As a safety, fill any remaining missing values in these features only
    X = data[features]
    y = data['TenYearCHD'].astype(int)

    # 5. Split train/test set (stratify for balanced evaluation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Imputer/scaler (fit ONLY on the final features!)
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # 7. Define models
    logreg = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    n_negative, n_positive = np.bincount(y_train)
    scale_pos = n_negative / n_positive
    xgbc = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                             n_estimators=80, max_depth=4, random_state=42,
                             scale_pos_weight=scale_pos)
    # Ensemble
    ensemble = VotingClassifier(
        estimators=[('lr', logreg), ('rf', rf), ('xgb', xgbc)],
        voting='soft',
        n_jobs=-1
    )

    models = {
        'Ensemble (Voting)': ensemble,
        'Logistic Regression': logreg,
        'Random Forest': rf,
        'XGBoost': xgbc
    }

    for name, clf in models.items():
        # Fit individual models before ensemble to make sure everything is ready
        if name == "Ensemble (Voting)":
            logreg.fit(X_train_scaled, y_train)
            rf.fit(X_train_scaled, y_train)
            xgbc.fit(X_train_scaled, y_train)
        print(f"\nTraining {name}...")
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        print(f"\n{name} Results:")
        print("Accuracy: %.4f" % accuracy_score(y_test, y_pred))
        print("ROC AUC: %.4f" % roc_auc_score(y_test, y_prob))
        cr = classification_report(y_test, y_pred, digits=3, output_dict=True)
        print("Recall for positive class (CHD): %.3f" % cr['1']['recall'])
        print(classification_report(y_test, y_pred, digits=3))

    # ===== SAVE the ENSEMBLE MODEL with imputer, scaler, and feature order ====
    print("\nSaving best model ensemble to 'models/saved_models/ensemble_model.pkl'")
    os.makedirs('models/saved_models', exist_ok=True)
    feature_mapping = {
        'male': 'gender_numeric',
        'totChol': 'total_cholesterol',
        'sysBP': 'systolic_bp',
        'currentSmoker': 'smoker',
        'BPMeds': 'bp_treatment',
        'BMI': 'bmi'
    }
    model_data = {
        'model': ensemble,
        'scaler': scaler,
        'imputer': imputer,
        'features': features,  # ORDER MATTERS!
        'feature_mapping': feature_mapping
    }
    with open('models/saved_models/ensemble_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print("\nDone! Model and artifacts saved.")

if __name__ == "__main__":
    train_risk_model()