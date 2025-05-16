import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from models.risk_model import get_model
from utils.translator import translate_text
import io
import base64

def render_time_series_tracker(user_data, language):
    st.header(translate_text("Time-Series Risk Tracker", language))
    st.write(translate_text(
        "Track how your heart disease risk changes over time, similar to a 'credit score for health'.",
        language
    ))
    
    # Initialize time series data in session state if not exists
    if 'time_series_data' not in st.session_state:
        # Generate some sample historical data
        dates = [(datetime.now() - timedelta(days=30*i)).strftime("%Y-%m-%d") for i in range(6, 0, -1)]
        
        # Simulate slightly different user data for each date
        historical_data = []
        model = get_model()
        
        base_risk = model.predict_risk(user_data)['risk_percentage']
        
        for i, date in enumerate(dates):
            # Create small variations in the parameters
            modified_data = user_data.copy()
            modified_data['total_cholesterol'] = max(100, user_data['total_cholesterol'] + np.random.randint(-20, 20))
            modified_data['hdl_cholesterol'] = max(20, user_data['hdl_cholesterol'] + np.random.randint(-5, 5))
            modified_data['systolic_bp'] = max(90, user_data['systolic_bp'] + np.random.randint(-10, 10))
            
            # Add slight weight change trend (decreasing over time)
            weight_change = (5 - i) * 0.5  # Simulates weight loss over time
            modified_data['weight'] = max(40, user_data['weight'] + weight_change)
            modified_data['bmi'] = modified_data['weight'] / ((user_data['height']/100) ** 2)
            
            # Calculate risk for this modified data
            risk = model.predict_risk(modified_data)
            
            # Store the data point
            historical_data.append({
                'date': date,
                'risk_percentage': risk['risk_percentage'],
                'risk_category': risk['risk_category'],
                'total_cholesterol': modified_data['total_cholesterol'],
                'hdl_cholesterol': modified_data['hdl_cholesterol'],
                'systolic_bp': modified_data['systolic_bp'],
                'weight': modified_data['weight'],
                'bmi': modified_data['bmi']
            })
        
        # Add current data point
        current_risk = model.predict_risk(user_data)
        historical_data.append({
            'date': datetime.now().strftime("%Y-%m-%d"),
            'risk_percentage': current_risk['risk_percentage'],
            'risk_category': current_risk['risk_category'],
            'total_cholesterol': user_data['total_cholesterol'],
            'hdl_cholesterol': user_data['hdl_cholesterol'],
            'systolic_bp': user_data['systolic_bp'],
            'weight': user_data['weight'],
            'bmi': user_data['bmi']
        })
        
        # Store in session state
        st.session_state.time_series_data = historical_data
    
    # Create DataFrame from historical data
    df = pd.DataFrame(st.session_state.time_series_data)
    
    # Create columns for chart and details
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(translate_text("Your Risk Score Trend", language))
        
        # Create risk trend chart with Plotly
        fig = px.line(
            df, 
            x='date', 
            y='risk_percentage',
            markers=True,
            title=translate_text('10-Year Heart Disease Risk Over Time', language),
            labels={'date': translate_text('Date', language), 
                   'risk_percentage': translate_text('Risk (%)', language)}
        )
        
        # Create color-coded horizontal zones
        fig.add_shape(
            type="rect", 
            x0=df['date'].min(), x1=df['date'].max(),
            y0=0, y1=5,
            fillcolor="rgba(46, 204, 113, 0.3)",  # Green, transparent
            line=dict(width=0),
            layer="below"
        )
        fig.add_shape(
            type="rect", 
            x0=df['date'].min(), x1=df['date'].max(),
            y0=5, y1=10,
            fillcolor="rgba(243, 156, 18, 0.3)",  # Orange, transparent
            line=dict(width=0),
            layer="below"
        )
        fig.add_shape(
            type="rect", 
            x0=df['date'].min(), x1=df['date'].max(),
            y0=10, y1=20,
            fillcolor="rgba(230, 126, 34, 0.3)",  # Dark orange, transparent
            line=dict(width=0),
            layer="below"
        )
        fig.add_shape(
            type="rect", 
            x0=df['date'].min(), x1=df['date'].max(),
            y0=20, y1=df['risk_percentage'].max() * 1.1,
            fillcolor="rgba(231, 76, 60, 0.3)",  # Red, transparent
            line=dict(width=0),
            layer="below"
        )
        
        # Customize line
        fig.update_traces(line=dict(color='#FF69B4', width=3))  # Sakura pink
        
        # Enhance layout
        fig.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=50, b=20),
            plot_bgcolor='rgba(255,255,255,0.9)'
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate trend
        first_risk = df.iloc[0]['risk_percentage']
        last_risk = df.iloc[-1]['risk_percentage']
        risk_change = last_risk - first_risk
        
        if risk_change < 0:
            st.success(f"{translate_text('Your risk has decreased by', language)} {abs(risk_change):.1f}% 🎉")
        elif risk_change > 0:
            st.error(f"{translate_text('Your risk has increased by', language)} {risk_change:.1f}% ⚠️")
        else:
            st.info(f"{translate_text('Your risk has remained stable', language)}")
    
    with col2:
        st.subheader(translate_text("Key Metrics", language))
        
        # Show latest risk score
        latest = df.iloc[-1]
        
        # Create risk score display like a credit score
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #ddd;">
            <h4 style="margin-bottom: 5px;">Heart Health Score</h4>
            <h2 style="color: #FF69B4; margin-top: 0; font-size: 42px;">
                {0:.0f}
            </h2>
            <p>{1}</p>
        </div>
        """.format(
            max(0, 100 - latest['risk_percentage']),  # Invert risk to make higher = better
            translate_text(latest['risk_category'], language)
        ), unsafe_allow_html=True)
        
        # Show other key metrics with trend indicators
        metrics = {
            'Total Cholesterol': {'value': latest['total_cholesterol'], 'unit': 'mg/dL', 'ideal': '< 200'},
            'HDL Cholesterol': {'value': latest['hdl_cholesterol'], 'unit': 'mg/dL', 'ideal': '> 40'},
            'Systolic BP': {'value': latest['systolic_bp'], 'unit': 'mmHg', 'ideal': '< 120'},
            'BMI': {'value': latest['bmi'], 'unit': 'kg/m²', 'ideal': '18.5-24.9'}
        }
        
        for name, data in metrics.items():
            # Calculate trend (first vs last value)
            first_val = df.iloc[0][name.lower().replace(' ', '_')]
            trend = data['value'] - first_val
            
            # Determine if the trend is good or bad
            if name == 'HDL Cholesterol':
                trend_good = trend > 0  # For HDL, higher is better
            else:
                trend_good = trend < 0  # For others, lower is better
            
            trend_icon = "↑" if trend > 0 else "↓" if trend < 0 else "→"
            trend_color = "green" if trend_good else "red" if trend != 0 else "gray"
            
            st.markdown(f"""
            <div style="margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 5px;">
                <div style="display: flex; justify-content: space-between;">
                    <span>{translate_text(name, language)}</span>
                    <span>{data['value']:.1f} {data['unit']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8em;">
                    <span>Ideal: {data['ideal']}</span>
                    <span style="color: {trend_color};">{trend_icon} {abs(trend):.1f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Add section for entering new measurements
    st.subheader(translate_text("Update Your Measurements", language))
    st.write(translate_text("Enter new health measurements to update your time-series data.", language))
    
    # Create form for new measurements
    with st.form(key="new_measurement_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            new_weight = st.number_input(translate_text("Weight (kg)", language), 
                                        min_value=40.0, max_value=200.0, value=float(user_data['weight']))
            new_systolic_bp = st.number_input(translate_text("Systolic BP (mmHg)", language), 
                                             min_value=90, max_value=200, value=int(user_data['systolic_bp']))
        
        with col2:
            new_total_chol = st.number_input(translate_text("Total Cholesterol (mg/dL)", language), 
                                            min_value=100, max_value=400, value=int(user_data['total_cholesterol']))
            new_hdl_chol = st.number_input(translate_text("HDL Cholesterol (mg/dL)", language), 
                                          min_value=20, max_value=100, value=int(user_data['hdl_cholesterol']))
        
        with col3:
            smoker_status = st.radio(translate_text("Currently Smoking", language), 
                                     [translate_text("Yes", language), translate_text("No", language)],
                                     index=0 if user_data['smoker'] else 1)
            measurement_date = st.date_input(translate_text("Measurement Date", language), datetime.now())
        
        submit_button = st.form_submit_button(label=translate_text("Save New Measurements", language))
        
        if submit_button:
            # Update user data with new measurements
            new_user_data = user_data.copy()
            new_user_data['weight'] = new_weight
            new_user_data['systolic_bp'] = new_systolic_bp
            new_user_data['total_cholesterol'] = new_total_chol
            new_user_data['hdl_cholesterol'] = new_hdl_chol
            new_user_data['smoker'] = True if smoker_status == translate_text("Yes", language) else False
            
            # Recalculate BMI
            new_user_data['bmi'] = new_weight / ((user_data['height']/100) ** 2)
            
            # Calculate new risk
            model = get_model()
            new_risk = model.predict_risk(new_user_data)
            
            # Add new data point
            new_data_point = {
                'date': measurement_date.strftime("%Y-%m-%d"),
                'risk_percentage': new_risk['risk_percentage'],
                'risk_category': new_risk['risk_category'],
                'total_cholesterol': new_total_chol,
                'hdl_cholesterol': new_hdl_chol,
                'systolic_bp': new_systolic_bp,
                'weight': new_weight,
                'bmi': new_user_data['bmi']
            }
            
            # Add to time series data
            st.session_state.time_series_data.append(new_data_point)
            
            # Sort by date
            st.session_state.time_series_data.sort(key=lambda x: datetime.strptime(x['date'], "%Y-%m-%d"))
            
            st.success(translate_text("New measurements saved successfully!", language))
            st.experimental_rerun()