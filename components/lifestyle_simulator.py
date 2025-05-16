import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.risk_model import get_model
from utils.explainer import get_explainer
from utils.translator import translate_text
import shap

def render_lifestyle_simulator(user_data, language):
    st.header(translate_text("Lifestyle Intervention Simulator", language))
    st.write(translate_text(
        "Simulate how lifestyle changes—like lowering cholesterol, losing weight, or quitting smoking—affect your heart disease risk.",
        language
    ))

    model = get_model()
    explainer = get_explainer()

    # Show current risk
    st.subheader(translate_text("Your Current Risk", language))
    current_risk = model.predict_risk(user_data)
    st.metric(
        label=translate_text("Current 10-year risk", language),
        value=f"{current_risk['risk_percentage']:.1f}%",
        delta=None
    )
    st.write(translate_text(f"Risk Category: {current_risk['risk_category']}", language))

    # Let user simulate new values
    st.subheader(translate_text("Simulate Lifestyle Changes", language))
    simulated_data = user_data.copy()

    # --- SLIDERS and CHECKBOXES for modifiable risk factors ---
    # Weight/BMI
    new_weight = st.slider(
        translate_text("Weight (kg)", language),
        min_value=max(40, int(user_data.get('weight', 70)) - 30),
        max_value=min(200, int(user_data.get('weight', 70)) + 30),
        value=int(user_data.get('weight', 70))
    )
    if new_weight != user_data.get('weight', 70):
        height_m = user_data.get('height', 170) / 100
        simulated_data['weight'] = new_weight
        simulated_data['bmi'] = new_weight / (height_m ** 2)

    # Smoking
    if user_data.get('smoker', 0):
        quit_smoking = st.checkbox(translate_text("Simulate Quitting Smoking", language), value=False)
        if quit_smoking:
            simulated_data['smoker'] = 0

    # Cholesterol
    new_total_chol = st.slider(
        translate_text("Total Cholesterol (mg/dL)", language),
        min_value=100, max_value=400,
        value=int(user_data.get('total_cholesterol', 200))
    )
    if new_total_chol != user_data.get('total_cholesterol', 200):
        simulated_data['total_cholesterol'] = new_total_chol
        simulated_data['hdl_cholesterol'] = new_total_chol * 0.25

    # HDL
    new_hdl = st.slider(
        translate_text("HDL Cholesterol (mg/dL)", language),
        min_value=20, max_value=100,
        value=int(simulated_data.get('hdl_cholesterol', user_data.get('total_cholesterol', 200) * 0.25))
    )
    if new_hdl != simulated_data.get('hdl_cholesterol', user_data.get('total_cholesterol', 200) * 0.25):
        simulated_data['hdl_cholesterol'] = new_hdl

    # Blood Pressure
    new_sbp = st.slider(
        translate_text("Systolic Blood Pressure (mmHg)", language),
        min_value=90, max_value=220,
        value=int(user_data.get('systolic_bp', 120))
    )
    if new_sbp != user_data.get('systolic_bp', 120):
        simulated_data['systolic_bp'] = new_sbp

    # BP Medication
    if not user_data.get('bp_treatment', 0):
        add_bpmed = st.checkbox(translate_text("Simulate Starting Blood Pressure Medication", language), value=False)
        if add_bpmed:
            simulated_data['bp_treatment'] = 1

    # Diabetes
    if not user_data.get('diabetes', 0):
        toggle_diabetes = st.checkbox(translate_text("Simulate Developing Diabetes", language), value=False)
        if toggle_diabetes:
            simulated_data['diabetes'] = 1

    # --- Calculate new risk ---
    st.markdown("---")
    st.subheader(translate_text("Predicted Risk After Changes", language))
    new_risk = model.predict_risk(simulated_data)
    risk_delta = new_risk['risk_percentage'] - current_risk['risk_percentage']
    st.metric(
        label=translate_text("New 10-year risk", language),
        value=f"{new_risk['risk_percentage']:.1f}%",
        delta=f"{risk_delta:+.1f}%", 
        delta_color="inverse" # green if negative
    )
    st.write(translate_text(f"Risk Category: {new_risk['risk_category']}", language))

    # --- SHAP Explanation ---
    with st.expander(translate_text("Why did my risk change?", language)):
        try:
            shap_values = explainer.get_shap_values(simulated_data)
            feature_names = model.features
            st.write(translate_text("Feature impact on NEW simulated risk:", language))
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.summary_plot(shap_values, np.array([explainer.risk_model.preprocess_user_data(simulated_data)[0]]), 
                              feature_names=feature_names, show=False)
            st.pyplot(fig)
        except Exception as e:
            st.info(translate_text(f"Could not generate SHAP explanation: {e}", language))

    st.markdown(translate_text(
        "Experiment with the sliders and checkboxes to see how your heart risk changes with different health decisions!", language
    ))