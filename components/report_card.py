import streamlit as st
import tempfile
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

from models.risk_model import get_model

def generate_risk_gauge(risk_percentage, risk_category):
    """Create a gauge plot (matplotlib) and return it as a BytesIO image."""
    fig, ax = plt.subplots(figsize=(4,2.5))
    def get_color(risk_pct):
        if risk_pct < 5:
            return "#2ECC71"
        elif risk_pct < 10:
            return "#F39C12"
        elif risk_pct < 20:
            return "#E67E22"
        else:
            return "#E74C3C"
    color = get_color(risk_percentage)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.add_patch(plt.Rectangle((0,0),10,5,fc='#f9f9f9'))
    theta = np.linspace(0,np.pi,100)
    x = 5+4*np.cos(theta)
    y = 1+4*np.sin(theta)
    ax.plot(x,y,color='#ccc',linewidth=12)
    risk_theta = np.pi * min(risk_percentage, 100)/100
    risk_x = 5+4*np.cos(np.linspace(0, risk_theta, 100))
    risk_y = 1+4*np.sin(np.linspace(0, risk_theta, 100))
    ax.plot(risk_x, risk_y, color=color, linewidth=14)
    needle_x = 5+3.8*np.cos(risk_theta)
    needle_y = 1+3.8*np.sin(risk_theta)
    ax.plot([5, needle_x],[1, needle_y],'k-',lw=2)
    ax.add_patch(plt.Circle((5,1), 0.18, fc='#333'))
    ax.text(5, 2.8, f"{risk_percentage:.1f}%", ha='center', va='center', fontsize=16, fontweight='bold', color=color)
    ax.text(5, 1.9, risk_category, ha='center', va='center', fontsize=11, color=color)
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches='tight', dpi=110)
    plt.close(fig)
    buf.seek(0)
    return buf

def render_report_card(user_data, language):
    st.header("Downloadable Health Report Card (PDF)")
    model = get_model()
    risk_result = model.predict_risk(user_data)
    st.write("👤 **User Profile**")
    st.json(user_data)
    st.write("❤️ **Predicted 10-Year Heart Disease Risk:** ", f"{risk_result['risk_percentage']:.1f}% ({risk_result['risk_category']})")

    if st.button("Generate PDF Report Card"):
        # 1. Gauge chart as image (matplotlib)
        gauge_buf = generate_risk_gauge(risk_result['risk_percentage'], risk_result['risk_category'])

        # 2. PDF Generation
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            c = canvas.Canvas(tmpfile.name, pagesize=A4)
            width, height = A4

            c.setFont("Helvetica-Bold", 18)
            c.setFillColor(colors.HexColor("#E74C3C"))
            c.drawString(60, height - 60, "SmartHeart Health Report Card")

            c.setFont("Helvetica", 13)
            c.setFillColor(colors.black)
            y = height - 100
            c.drawString(60, y, f"Gender: {user_data['gender']}")
            c.drawString(260, y, f"Age: {user_data['age']}")
            y -= 25
            c.drawString(60, y, f"Total Cholesterol: {user_data['total_cholesterol']} mg/dL")
            c.drawString(300, y, f"HDL: {user_data['hdl_cholesterol']} mg/dL")
            y -= 25
            c.drawString(60, y, f"Systolic BP: {user_data['systolic_bp']} mmHg")
            c.drawString(300, y, f"Smoker: {'Yes' if user_data['smoker'] else 'No'}")
            y -= 25
            c.drawString(60, y, f"Diabetes: {'Yes' if user_data['diabetes'] else 'No'}")
            c.drawString(170, y, f"BP Treatment: {'Yes' if user_data['bp_treatment'] else 'No'}")
            c.drawString(340, y, f"BMI: {user_data['bmi']:.1f}")

            y -= 38
            c.setFont("Helvetica-Bold", 14)
            c.drawString(60, y, "Your 10-year Heart Disease Risk:")
            c.setFont("Helvetica-Bold", 22)
            c.setFillColor(colors.HexColor("#E74C3C" if risk_result["risk_percentage"]>10 else "#2ECC71"))
            c.drawString(60, y-30, f"{risk_result['risk_percentage']:.1f}%")
            c.setFont("Helvetica", 13)
            c.setFillColor(colors.black)
            c.drawString(160, y-24, f"({risk_result['risk_category']})")

            # Insert risk gauge chart
            c.drawImage(ImageReader(gauge_buf), 330, y-60, width=150, height=90)

            y -= 75
            c.setFont("Helvetica", 12)
            c.drawString(60, y, "Explanation: Your risk is calculated based on age, cholesterol, blood pressure, BMI, smoking, and diabetes status using the Framingham model with AI improvement.")
            c.drawString(60, y-16, "Talk with your healthcare provider for further personalized advice and medical decisions.")

            c.showPage()
            c.save()

            pdf_data = tmpfile.read()
            st.success("PDF Report Created!")
            st.download_button(
                label="📄 Download Your PDF Report Card",
                data=pdf_data,
                file_name="SmartHeart_HeartRisk_Report.pdf",
                mime="application/pdf"
            )