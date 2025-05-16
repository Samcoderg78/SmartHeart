import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px
import io

class TimeSeriesRiskTracker:
    def __init__(self):
        """Initialize the time series risk tracker"""
        self.model = None
        self.scaler = None
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize a simple LightGBM model for trend prediction"""
        # For this demo, we'll create a simple model that predicts future risk
        # based on current measurements and their trends
        self.model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
        self.scaler = StandardScaler()
    
    def train_model(self, time_series_data):
        """Train the model on historical data
        
        Args:
            time_series_data (list): List of dictionaries with historical data
        """
        if len(time_series_data) < 3:
            return False
        
        # Convert to DataFrame
        df = pd.DataFrame(time_series_data)
        
        # Convert date strings to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date')
        
        # Create features (including trend features)
        features = []
        targets = []
        
        for i in range(2, len(df)):
            # Get current and previous data points
            current = df.iloc[i]
            prev1 = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            
            # Calculate trends
            chol_trend = current['total_cholesterol'] - prev1['total_cholesterol']
            hdl_trend = current['hdl_cholesterol'] - prev1['hdl_cholesterol']
            bp_trend = current['systolic_bp'] - prev1['systolic_bp']
            weight_trend = current['weight'] - prev1['weight']
            
            # Previous chol trend
            prev_chol_trend = prev1['total_cholesterol'] - prev2['total_cholesterol']
            
            # Create feature vector
            feature = [
                current['total_cholesterol'],
                current['hdl_cholesterol'],
                current['systolic_bp'],
                current['weight'],
                current['bmi'],
                chol_trend,
                hdl_trend,
                bp_trend,
                weight_trend,
                prev_chol_trend
            ]
            
            features.append(feature)
            targets.append(current['risk_percentage'])
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(targets)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        return True
    
    def predict_future_risk(self, time_series_data, periods=3):
        """Predict future risk based on historical data
        
        Args:
            time_series_data (list): List of dictionaries with historical data
            periods (int): Number of future periods to predict
            
        Returns:
            dict: Dictionary with predicted risk values and visualization
        """
        if len(time_series_data) < 3:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(time_series_data)
        
        # Convert date strings to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date')
        
        # Train model if needed
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            self.train_model(time_series_data)
        
        # Get the latest data points
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        
        # Calculate current trends
        chol_trend = latest['total_cholesterol'] - prev['total_cholesterol']
        hdl_trend = latest['hdl_cholesterol'] - prev['hdl_cholesterol']
        bp_trend = latest['systolic_bp'] - prev['systolic_bp']
        weight_trend = latest['weight'] - prev['weight']
        prev_chol_trend = prev['total_cholesterol'] - prev2['total_cholesterol']
        
        # Generate future dates
        last_date = df['date'].max()
        future_dates = [last_date + pd.Timedelta(days=30*i) for i in range(1, periods+1)]
        
        # Initialize predictions
        predictions = []
        
        # Current feature vector
        current_features = np.array([
            [latest['total_cholesterol'],
             latest['hdl_cholesterol'],
             latest['systolic_bp'],
             latest['weight'],
             latest['bmi'],
             chol_trend,
             hdl_trend,
             bp_trend,
             weight_trend,
             prev_chol_trend]
        ])
        
        # Scale features
        current_features_scaled = self.scaler.transform(current_features)
        
        # Initial prediction
        current_risk = self.model.predict(current_features_scaled)[0]
        
        # Generate future predictions
        total_chol = latest['total_cholesterol']
        hdl_chol = latest['hdl_cholesterol']
        systolic_bp = latest['systolic_bp']
        weight = latest['weight']
        bmi = latest['bmi']
        
        for i, future_date in enumerate(future_dates):
            # Update values based on trends (simplified)
            total_chol += chol_trend * 0.7  # Assume trend diminishes
            hdl_chol += hdl_trend * 0.7
            systolic_bp += bp_trend * 0.7
            weight += weight_trend * 0.7
            bmi = weight / ((latest['height']/100) ** 2)
            
            # Update trends (simplified)
            prev_chol_trend = chol_trend
            chol_trend *= 0.7
            hdl_trend *= 0.7
            bp_trend *= 0.7
            weight_trend *= 0.7
            
            # Create feature vector
            future_features = np.array([
                [total_chol,
                 hdl_chol,
                 systolic_bp,
                 weight,
                 bmi,
                 chol_trend,
                 hdl_trend,
                 bp_trend,
                 weight_trend,
                 prev_chol_trend]
            ])
            
            # Scale features
            future_features_scaled = self.scaler.transform(future_features)
            
            # Predict
            risk = self.model.predict(future_features_scaled)[0]
            
            # Add to predictions
            predictions.append({
                'date': future_date.strftime('%Y-%m-%d'),
                'risk_percentage': risk,
                'total_cholesterol': total_chol,
                'hdl_cholesterol': hdl_chol,
                'systolic_bp': systolic_bp,
                'weight': weight,
                'bmi': bmi
            })
        
        # Create combined data for visualization
        historical_data_for_viz = [{
            'date': row['date'].strftime('%Y-%m-%d'), 
            'risk_percentage': row['risk_percentage'],
            'type': 'Historical'
        } for _, row in df.iterrows()]
        
        prediction_data_for_viz = [{
            'date': pred['date'],
            'risk_percentage': pred['risk_percentage'],
            'type': 'Predicted'
        } for pred in predictions]
        
        combined_data = historical_data_for_viz + prediction_data_for_viz
        combined_df = pd.DataFrame(combined_data)
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        combined_df = combined_df.sort_values('date')
        
        # Generate plot
        fig = px.line(
            combined_df, 
            x='date', 
            y='risk_percentage', 
            color='type',
            markers=True,
            title='Heart Disease Risk Projection',
            color_discrete_map={'Historical': '#FF69B4', 'Predicted': '#9370DB'}
        )
        
        # Add risk zones
        fig.add_shape(
            type="rect", 
            x0=combined_df['date'].min(), x1=combined_df['date'].max(),
            y0=0, y1=5,
            fillcolor="rgba(46, 204, 113, 0.3)",  # Green, transparent
            line=dict(width=0),
            layer="below"
        )
        fig.add_shape(
            type="rect", 
            x0=combined_df['date'].min(), x1=combined_df['date'].max(),
            y0=5, y1=10,
            fillcolor="rgba(243, 156, 18, 0.3)",  # Orange, transparent
            line=dict(width=0),
            layer="below"
        )
        fig.add_shape(
            type="rect", 
            x0=combined_df['date'].min(), x1=combined_df['date'].max(),
            y0=10, y1=20,
            fillcolor="rgba(230, 126, 34, 0.3)",  # Dark orange, transparent
            line=dict(width=0),
            layer="below"
        )
        fig.add_shape(
            type="rect", 
            x0=combined_df['date'].min(), x1=combined_df['date'].max(),
            y0=20, y1=combined_df['risk_percentage'].max() * 1.1,
            fillcolor="rgba(231, 76, 60, 0.3)",  # Red, transparent
            line=dict(width=0),
            layer="below"
        )
        
        # Save plot to BytesIO
        img_bytes = io.BytesIO()
        fig.write_image(img_bytes, format="png")
        img_bytes.seek(0)
        
        # Create trend description
        first_risk = df.iloc[0]['risk_percentage']
        last_risk = df.iloc[-1]['risk_percentage']
        future_risk = predictions[-1]['risk_percentage']
        
        historical_change = last_risk - first_risk
        future_change = future_risk - last_risk
        
        if historical_change < 0:
            historical_trend = f"Your risk has decreased by {abs(historical_change):.1f}% over the recorded period."
        elif historical_change > 0:
            historical_trend = f"Your risk has increased by {historical_change:.1f}% over the recorded period."
        else:
            historical_trend = "Your risk has remained stable over the recorded period."
        
        if future_change < 0:
            future_trend = f"Based on current trends, your risk is projected to decrease by {abs(future_change):.1f}% in the next {periods} months."
        elif future_change > 0:
            future_trend = f"Based on current trends, your risk is projected to increase by {future_change:.1f}% in the next {periods} months."
        else:
            future_trend = f"Based on current trends, your risk is projected to remain stable in the next {periods} months."
        
        trend_description = f"{historical_trend} {future_trend}"
        
        return {
            'historical_data': df.to_dict('records'),
            'predictions': predictions,
            'trend_chart': img_bytes,
            'trend_description': trend_description,
            'risk_trend': {
                'historical_change': historical_change,
                'future_change': future_change
            }
        }

# Singleton instance
_tracker_instance = None

def get_time_series_tracker():
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = TimeSeriesRiskTracker()
    return _tracker_instance