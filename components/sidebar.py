import streamlit as st
import pandas as pd
from utils.translator import translate_text

def render_sidebar():
    st.sidebar.header("User Information")
    
    # Language selector
    language = st.sidebar.selectbox(
        "Language / 言語 / Idioma:",
        ["English", "日本語", "Español", "Français", "中文"]
    )
    
    # User inputs with keys to ensure values are properly updated
    st.sidebar.subheader(translate_text("Personal Information", language))
    
    age = st.sidebar.slider(translate_text("Age", language), 20, 90, 40, key="age_slider")
    gender = st.sidebar.radio(translate_text("Gender", language), ["Male", "Female"], key="gender_radio")
    
    st.sidebar.subheader(translate_text("Medical Parameters", language))
    
    total_cholesterol = st.sidebar.slider(translate_text("Total Cholesterol (mg/dL)", language), 100, 400, 200, key="chol_slider")
    hdl_cholesterol = st.sidebar.slider(translate_text("HDL Cholesterol (mg/dL)", language), 20, 100, 50, key="hdl_slider")
    systolic_bp = st.sidebar.slider(translate_text("Systolic Blood Pressure (mmHg)", language), 90, 200, 120, key="bp_slider")
    
    st.sidebar.subheader(translate_text("Lifestyle Factors", language))
    
    smoker = st.sidebar.checkbox(translate_text("Currently Smoking", language), key="smoker_check")
    diabetes = st.sidebar.checkbox(translate_text("Diabetes", language), key="diabetes_check")
    bp_treatment = st.sidebar.checkbox(translate_text("On Blood Pressure Treatment", language), key="bp_treatment_check")
    
    weight = st.sidebar.number_input(translate_text("Weight (kg)", language), 40.0, 200.0, 70.0, key="weight_input")
    height = st.sidebar.number_input(translate_text("Height (cm)", language), 120.0, 220.0, 170.0, key="height_input")
    
    # Calculate BMI
    bmi = weight / ((height/100) ** 2)
    
    # Display calculated BMI
    st.sidebar.metric(translate_text("BMI", language), f"{bmi:.1f}")
    
    # Create user data dictionary
    user_data = {
        "age": age,
        "gender": gender,
        "total_cholesterol": total_cholesterol,
        "hdl_cholesterol": hdl_cholesterol,
        "systolic_bp": systolic_bp,
        "smoker": smoker,
        "diabetes": diabetes,
        "bp_treatment": bp_treatment,
        "bmi": bmi,
        "weight": weight,
        "height": height
    }
    
    # Add a debug section
    if st.sidebar.checkbox("Show Debug Info", False):
        st.sidebar.write("Current User Data:")
        st.sidebar.json(user_data)
    
    return language, user_data
