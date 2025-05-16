import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from models.risk_model import get_model
from utils.translator import translate_text
from datetime import datetime

def render_time_series_tracker(user_data, language):
    st.header(translate_text("Time-Series Risk Tracker", language))
    st.write(translate_text(
        "Track how your predicted heart disease risk changes over time. "
        "Each entry represents a snapshot of your risk based on your health at that time.",
        language
    ))

    # --- 0. Session State for risk history ---
    if "risk_history" not in st.session_state:
        st.session_state.risk_history = []

    # Compute current risk
    model = get_model()
    risk_result = model.predict_risk(user_data)

    # Let user "save" this point in time
    if st.button(translate_text("Save my current risk snapshot", language)):
        st.session_state.risk_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "risk_percentage": round(risk_result["risk_percentage"], 2),
            "risk_category": risk_result["risk_category"],
            "age": user_data["age"]
            # ... add more user params if you wish, e.g., weight, bp, etc.
        })
        st.success(translate_text("Snapshot saved!", language))

    # --- 1. Display Time Series ---
    if st.session_state.risk_history:
        df = pd.DataFrame(st.session_state.risk_history)
        st.subheader(translate_text("Your Risk Trend Over Time", language))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["risk_percentage"], mode="lines+markers",
            marker=dict(color="red"),
            name=translate_text("10-year Risk (%)", language)
        ))
        fig.update_layout(
            xaxis_title=translate_text("Date / Time", language),
            yaxis_title=translate_text("Predicted Risk (%)", language),
            yaxis=dict(range=[0, max(15, df["risk_percentage"].max() + 5)]),
            title=translate_text("Heart Disease Risk Over Time", language),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df)
        # Download CSV button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            translate_text("Download My Risk History (CSV)", language),
            csv,
            "risk_history.csv",
            "text/csv"
        )
    else:
        st.info(translate_text("No risk snapshots saved yet. Click the button above to save your first one!", language))