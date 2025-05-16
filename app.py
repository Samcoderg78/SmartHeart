import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image

# Set page config - MUST be the first Streamlit command
# and only called ONCE
st.set_page_config(
    page_title="SmartHeart",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import components
from components.sidebar import render_sidebar
from components.risk_calculator import render_risk_calculator
from components.lifestyle_simulator import render_lifestyle_simulator
from components.report_card import render_report_card
from components.time_series_tracker import render_time_series_tracker

# Import utilities
from utils.translator import translate_text

# Apply custom sakura theme
def apply_sakura_theme():
    try:
        with open('assets/css/sakura_theme.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load theme: {e}")

def main():
    # Apply sakura theme
    apply_sakura_theme()
    
    # Header
    col1, col2 = st.columns([1, 5])
    with col1:
        try:
            st.image('assets/images/logo.png', width=100)
        except:
            st.info("Logo image not found")
    with col2:
        st.title("SmartHeart: AI-Powered Heart Disease Risk Analyzer")
        st.markdown("Predict your 10-year cardiovascular risk and explore personalized interventions")
    
    # Sidebar for user inputs and language selection
    language, user_data = render_sidebar()
    
    # Main tabs
    tabs = st.tabs(["Risk Calculator", "Lifestyle Simulator", "Health Report Card", "Time-Series Tracker"])
    
    with tabs[0]:
        render_risk_calculator(user_data, language)
    
    with tabs[1]:
        render_lifestyle_simulator(user_data, language)
    
    with tabs[2]:
        render_report_card(user_data, language)
    
    with tabs[3]:
        render_time_series_tracker(user_data, language)
    
    # Footer
    st.markdown("---")
    st.markdown("*SmartHeart: Empowering you with personalized heart health insights.*")

if __name__ == "__main__":
    main()