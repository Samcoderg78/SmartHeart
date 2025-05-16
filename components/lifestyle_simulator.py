import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from utils.explainer import get_explainer
from utils.translator import translate_text
from models.risk_model import get_model

def render_lifestyle_simulator(user_data, language):
    st.header(translate_text("Lifestyle Intervention Simulator", language))
    st.write(translate_text(
        "Explore how lifestyle changes could impact your heart disease risk.",
        language
    ))
    
    # Get the model and explainer
    model = get_model()
    explainer = get_explainer()
    
    # Create columns for original risk and simulated risk
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(translate_text("Current Risk Factors", language))
        
        # Display current risk factors
        risk_factors = pd.DataFrame({
            'Factor': ['Age', 'BMI', 'Total Cholesterol', 'HDL Cholesterol', 
                     'Systolic BP', 'Smoking', 'Diabetes', 'BP Treatment'],
            'Current Value': [
                f"{user_data['age']} years",
                f"{user_data['bmi']:.1f} kg/m",
                f"{user_data['total_cholesterol']} mg/dL",
                f"{user_data['hdl_cholesterol']} mg/dL",
                f"{user_data['systolic_bp']} mmHg",
                "Yes" if user_data['smoker'] else "No",
                "Yes" if user_data['diabetes'] else "No",
                "Yes" if user_data['bp_treatment'] else "No"
            ]
        })
        
        st.table(risk_factors)
        
        # Display current risk - use the actual model prediction
        current_risk = model.predict_risk(user_data)
        st.subheader(translate_text("Current 10-Year Risk", language))
        st.markdown(f"<h1 style='color: {'red' if current_risk['risk_percentage'] > 10 else 'orange' if current_risk['risk_percentage'] > 5 else 'green'};'>{current_risk['risk_percentage']:.1f}%</h1>", unsafe_allow_html=True)
        st.markdown(f"**{translate_text('Risk Category', language)}:** {translate_text(current_risk['risk_category'], language)}")
    
    with col2:
        st.subheader(translate_text("Simulate Lifestyle Changes", language))
        st.write(translate_text("Adjust the factors below to see how they would affect your risk.", language))
        
        # Create intervention sliders
        interventions = {}
        
        # Weight/BMI intervention
        weight_change = st.slider(
            translate_text("Target Weight (kg)", language), 
            float(max(40, user_data['weight'] - 30)), 
            float(user_data['weight'] + 30), 
            float(user_data['weight']),
            key="sim_weight_slider",
            help="Adjust to see how weight loss or gain would affect your risk"
        )
        
        if weight_change != user_data['weight']:
            # Recalculate BMI
            new_bmi = weight_change / ((user_data['height']/100) ** 2)
            interventions['weight'] = weight_change
            interventions['bmi'] = new_bmi
        
        # Smoking intervention
        if user_data['smoker']:
            quit_smoking = st.checkbox(translate_text("Quit Smoking", language), 
                                     key="sim_smoking_check",
                                     help="Check to see how quitting smoking would affect your risk")
            if quit_smoking:
                interventions['smoker'] = False
        
        # Cholesterol interventions
        new_total_chol = st.slider(
            translate_text("Target Total Cholesterol (mg/dL)", language), 
            100, 400, user_data['total_cholesterol'],
            key="sim_chol_slider",
            help="Adjust to see how lowering cholesterol would affect your risk"
        )
        if new_total_chol != user_data['total_cholesterol']:
            interventions['total_cholesterol'] = new_total_chol
            
        new_hdl = st.slider(
            translate_text("Target HDL Cholesterol (mg/dL)", language), 
            20, 100, user_data['hdl_cholesterol'],
            key="sim_hdl_slider",
            help="Higher HDL is generally better for heart health"
        )
        if new_hdl != user_data['hdl_cholesterol']:
            interventions['hdl_cholesterol'] = new_hdl
        
        # Blood pressure intervention
        new_bp = st.slider(
            translate_text("Target Systolic Blood Pressure (mmHg)", language), 
            90, 200, user_data['systolic_bp'],
            key="sim_bp_slider",
            help="Adjust to see how blood pressure control would affect your risk"
        )
        if new_bp != user_data['systolic_bp']:
            interventions['systolic_bp'] = new_bp
        
        # BP medication intervention
        if not user_data['bp_treatment'] and user_data['systolic_bp'] > 130:
            bp_meds = st.checkbox(translate_text("Start BP Medication", language), 
                                key="sim_bp_meds_check",
                                help="Check to see how blood pressure medication would affect your risk")
            if bp_meds:
                interventions['bp_treatment'] = True
        
        # Add "Run Simulation" button but also calculate automatically
        run_button = st.button(translate_text("Run Simulation", language), key="run_sim_button")
        
        # Run simulation when button is clicked or automatically when any intervention changes
        if interventions and (run_button or True):  # Always run if interventions exist
            # Create a copy of user data with interventions applied
            modified_data = user_data.copy()
            for key, value in interventions.items():
                modified_data[key] = value
            
            # Calculate new risk using the model directly
            new_risk = model.predict_risk(modified_data)
            risk_diff = new_risk['risk_percentage'] - current_risk['risk_percentage']
            
            # Show results
            st.subheader(translate_text("Simulation Results", language))
            
            # Risk difference
            if risk_diff < 0:
                st.success(f"{translate_text('Risk reduction', language)}: {abs(risk_diff):.1f}% ")
            elif risk_diff > 0:
                st.error(f"{translate_text('Risk increase', language)}: {risk_diff:.1f}% ")
            else:
                st.info(f"{translate_text('No change in risk', language)}")
            
            # New risk
            st.markdown(f"**{translate_text('New estimated 10-year risk', language)}:** {new_risk['risk_percentage']:.1f}% ({translate_text(new_risk['risk_category'], language)})")
