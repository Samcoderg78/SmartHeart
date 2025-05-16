import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.risk_model import get_model
from utils.translator import translate_text

def render_risk_calculator(user_data, language):
    st.header(translate_text("Heart Disease Risk Calculator", language))
    st.write(translate_text(
        "This calculator uses the Framingham Heart Study model to predict your 10-year risk of developing cardiovascular disease.",
        language
    ))
    
    # Get model and predict risk
    # ---- To use logistic regression: get_model("logreg")
    model = get_model()  # or get_model("ensemble") / get_model("logreg")
    
    # Debug: Show user data input table
    st.write("Predicting with these values:")
    st.write(pd.DataFrame([user_data]))
    
    # Make prediction
    risk_result = model.predict_risk(user_data)
    
    # Display results
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader(translate_text("Your 10-Year Heart Disease Risk", language))
        fig, ax = plt.subplots(figsize=(10, 6))
        def get_color(risk_pct):
            if risk_pct < 5:
                return "#2ECC71"  # Green
            elif risk_pct < 10:
                return "#F39C12"  # Orange
            elif risk_pct < 20:
                return "#E67E22"  # Dark Orange
            else:
                return "#E74C3C"  # Red
        risk_color = get_color(risk_result['risk_percentage'])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        ax.add_patch(plt.Rectangle((0, 0), 10, 5, fc='#f9f9f9'))
        theta = np.linspace(0, np.pi, 100)
        x = 5 + 4 * np.cos(theta)
        y = 1 + 4 * np.sin(theta)
        ax.plot(x, y, color='#ccc', linewidth=10)
        risk_theta = np.pi * min(risk_result['risk_percentage'], 100) / 100
        risk_x = 5 + 4 * np.cos(np.linspace(0, risk_theta, 100))
        risk_y = 1 + 4 * np.sin(np.linspace(0, risk_theta, 100))
        ax.plot(risk_x, risk_y, color=risk_color, linewidth=10)
        needle_x = 5 + 3.8 * np.cos(risk_theta)
        needle_y = 1 + 3.8 * np.sin(risk_theta)
        ax.plot([5, needle_x], [1, needle_y], color='#333', linewidth=2)
        ax.add_patch(plt.Circle((5, 1), 0.2, fc='#333'))
        for pct, label in [(0, "0%"), (25, "25%"), (50, "50%"), (75, "75%"), (100, "100%")]:
            marker_theta = np.pi * pct / 100
            marker_x = 5 + 4 * np.cos(marker_theta)
            marker_y = 1 + 4 * np.sin(marker_theta)
            ax.text(marker_x, marker_y + 0.3, label, ha='center', va='center')
        ax.text(5, 3, f"{risk_result['risk_percentage']:.1f}%", 
               ha='center', va='center', fontsize=24, fontweight='bold', color=risk_color)
        ax.text(5, 2.3, translate_text(risk_result['risk_category'], language), 
               ha='center', va='center', fontsize=16, color=risk_color)
        ax.axis('off')
        st.pyplot(fig)
    
    with col2:
        st.subheader(translate_text("Risk Category", language))
        category_styles = {
            "Low Risk": "background-color: #2ECC71; padding: 10px; border-radius: 5px; color: white;",
            "Moderate Risk": "background-color: #F39C12; padding: 10px; border-radius: 5px; color: white;",
            "High Risk": "background-color: #E67E22; padding: 10px; border-radius: 5px; color: white;",
            "Very High Risk": "background-color: #E74C3C; padding: 10px; border-radius: 5px; color: white;"
        }
        st.markdown(f"<div style='{category_styles[risk_result['risk_category']]}'><h3>{translate_text(risk_result['risk_category'], language)}</h3></div>", unsafe_allow_html=True)
        st.subheader(translate_text("What This Means", language))
        interpretations = {
            "Low Risk": "Your 10-year risk of heart disease is low. Continue your healthy habits!",
            "Moderate Risk": "You have a moderate risk of heart disease. Consider lifestyle improvements.",
            "High Risk": "Your risk is elevated. Consult a healthcare provider about risk reduction strategies.",
            "Very High Risk": "Your risk is significantly elevated. We strongly recommend consulting a healthcare provider."
        }
        st.write(translate_text(interpretations[risk_result['risk_category']], language))
        st.subheader(translate_text("Next Steps", language))
        st.write(translate_text("Use the Lifestyle Simulator to see how changes can improve your risk profile.", language))
        st.write(translate_text("Generate a Health Report Card to share with your healthcare provider.", language))