import pandas as pd
import numpy as np
import os

def generate_sample_data():
    """Generate sample user data for demonstration purposes"""
    np.random.seed(42)
    
    # Create 50 sample users
    n_samples = 50
    
    # Age between 30 and 75
    ages = np.random.randint(30, 76, size=n_samples)
    
    # Gender (approximately 50% male, 50% female)
    genders = np.random.choice([1, 0], size=n_samples)
    
    # Total cholesterol between 150 and 300
    total_chol = np.random.randint(150, 301, size=n_samples)
    
    # HDL between 30 and 80
    hdl_chol = np.random.randint(30, 81, size=n_samples)
    
    # Systolic BP between 100 and 180
    systolic_bp = np.random.randint(100, 181, size=n_samples)
    
    # Smoking status (approximately 25% smokers)
    smokers = np.random.choice([1, 0], size=n_samples, p=[0.25, 0.75])
    
    # Diabetes status (approximately 15% diabetic)
    diabetes = np.random.choice([1, 0], size=n_samples, p=[0.15, 0.85])
    
    # BP treatment (approximately 30% on treatment)
    bp_treatment = np.random.choice([1, 0], size=n_samples, p=[0.3, 0.7])
    
    # Weight between 50 and 110 kg
    weight = np.random.uniform(50, 110, size=n_samples)
    
    # Height between 150 and 190 cm
    height = np.random.uniform(150, 190, size=n_samples)
    
    # Calculate BMI
    bmi = weight / ((height/100) ** 2)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': ages,
        'gender_numeric': genders,
        'total_cholesterol': total_chol,
        'hdl_cholesterol': hdl_chol,
        'systolic_bp': systolic_bp,
        'smoker': smokers,
        'diabetes': diabetes,
        'bp_treatment': bp_treatment,
        'weight': weight,
        'height': height,
        'bmi': bmi
    })
    
    # Ensure directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    df.to_csv('data/sample_user_data.csv', index=False)
    print("Sample data generated and saved to data/sample_user_data.csv")

if __name__ == "__main__":
    generate_sample_data()
