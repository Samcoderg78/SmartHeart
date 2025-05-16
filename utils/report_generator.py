import os
import time
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import numpy as np
import base64
from models.risk_model import get_model
from utils.explainer import get_explainer
from utils.translator import translate_text

class HealthReportGenerator:
    def __init__(self, language="English"):
        self.language = language
        self.styles = getSampleStyleSheet()
        
        # Add custom styles
        self.styles.add(ParagraphStyle(
            name='SakuraTitle',
            parent=self.styles['Title'],
            textColor=colors.HexColor("#FF69B4"),
            spaceAfter=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='SakuraHeading',
            parent=self.styles['Heading2'],
            textColor=colors.HexColor("#FF69B4"),
            spaceAfter=6
        ))
    
    def generate_risk_chart(self, risk_percentage):
        """Generate a gauge chart for the risk percentage"""
        fig, ax = plt.subplots(figsize=(5, 3))
        
        # Define the risk color scale
        def get_color(risk_pct):
            if risk_pct < 5:
                return "#2ECC71"  # Green
            elif risk_pct < 10:
                return "#F39C12"  # Orange
            elif risk_pct < 20:
                return "#E67E22"  # Dark Orange
            else:
                return "#E74C3C"  # Red
        
        risk_color = get_color(risk_percentage)
        
        # Create a semicircular gauge
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        ax.add_patch(plt.Rectangle((0, 0), 10, 5, fc='#f9f9f9'))
        
        # Draw the gauge background
        theta = np.linspace(0, np.pi, 100)
        x = 5 + 4 * np.cos(theta)
        y = 1 + 4 * np.sin(theta)
        ax.plot(x, y, color='#ccc', linewidth=10)
        
        # Draw the risk indicator
        risk_theta = np.pi * min(risk_percentage, 100) / 100
        risk_x = 5 + 4 * np.cos(np.linspace(0, risk_theta, 100))
        risk_y = 1 + 4 * np.sin(np.linspace(0, risk_theta, 100))
        ax.plot(risk_x, risk_y, color=risk_color, linewidth=10)
        
        # Add indicator needle
        needle_x = 5 + 3.8 * np.cos(risk_theta)
        needle_y = 1 + 3.8 * np.sin(risk_theta)
        ax.plot([5, needle_x], [1, needle_y], color='#333', linewidth=2)
        ax.add_patch(plt.Circle((5, 1), 0.2, fc='#333'))
        
        # Add risk percentage in the center
        ax.text(5, 2.5, f"{risk_percentage:.1f}%", 
               ha='center', va='center', fontsize=24, fontweight='bold', color=risk_color)
        
        # Hide axes
        ax.axis('off')
        
        # Save to a BytesIO object
        img_data = BytesIO()
        plt.savefig(img_data, format='png', bbox_inches='tight', dpi=150)
        img_data.seek(0)
        plt.close()
        
        return img_data
    
    def generate_report(self, user_data, time_series_data=None):
        """Generate a PDF health report
        
        Args:
            user_data (dict): User health data
            time_series_data (dict, optional): Time series data for trends
            
        Returns:
            BytesIO: PDF file as bytes
        """
        # Initialize PDF buffer and document
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, 
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=72)
        
        # Get model predictions
        model = get_model()
        explainer = get_explainer()
        risk_result = model.predict_risk(user_data)
        
        # Create document content
        content = []
        
        # Add title
        title = translate_text("SmartHeart: Personal Health Report", self.language)
        content.append(Paragraph(title, self.styles['SakuraTitle']))
        
        # Add date
        date_text = translate_text(f"Generated on: {time.strftime('%B %d, %Y')}", self.language)
        content.append(Paragraph(date_text, self.styles['Normal']))
        content.append(Spacer(1, 0.2*inch))
        
        # Add personal information section
        content.append(Paragraph(translate_text("Personal Information", self.language), self.styles['SakuraHeading']))
        
        personal_data = [
            [translate_text("Age", self.language), f"{user_data['age']} years"],
            [translate_text("Gender", self.language), translate_text(user_data['gender'], self.language)],
            [translate_text("Height", self.language), f"{user_data['height']} cm"],
            [translate_text("Weight", self.language), f"{user_data['weight']} kg"],
            [translate_text("BMI", self.language), f"{user_data['bmi']:.1f} kg/m²"]
        ]
        
        personal_table = Table(personal_data, colWidths=[2*inch, 3*inch])
        personal_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lavender),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        content.append(personal_table)
        content.append(Spacer(1, 0.2*inch))
        
        # Add health metrics section
        content.append(Paragraph(translate_text("Health Metrics", self.language), self.styles['SakuraHeading']))
        
        health_data = [
            [translate_text("Total Cholesterol", self.language), f"{user_data['total_cholesterol']} mg/dL"],
            [translate_text("HDL Cholesterol", self.language), f"{user_data['hdl_cholesterol']} mg/dL"],
            [translate_text("Systolic Blood Pressure", self.language), f"{user_data['systolic_bp']} mmHg"],
            [translate_text("Smoking Status", self.language), translate_text("Yes" if user_data['smoker'] else "No", self.language)],
            [translate_text("Diabetes", self.language), translate_text("Yes" if user_data['diabetes'] else "No", self.language)],
            [translate_text("Blood Pressure Treatment", self.language), translate_text("Yes" if user_data['bp_treatment'] else "No", self.language)]
        ]
        
        health_table = Table(health_data, colWidths=[2*inch, 3*inch])
        health_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightpink),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        content.append(health_table)
        content.append(Spacer(1, 0.2*inch))
        
        # Add risk assessment section
        content.append(Paragraph(translate_text("Heart Disease Risk Assessment", self.language), self.styles['SakuraHeading']))
        
        # Add risk gauge chart
        risk_chart = self.generate_risk_chart(risk_result['risk_percentage'])
        img = Image(risk_chart, width=4*inch, height=2.4*inch)
        content.append(img)
        
        # Add risk category text
        risk_category_text = translate_text(f"Risk Category: {risk_result['risk_category']}", self.language)
        content.append(Paragraph(risk_category_text, self.styles['Heading3']))
        
        # Risk interpretation
        interpretations = {
            "Low Risk": "Your 10-year risk of heart disease is low. Continue your healthy habits!",
            "Moderate Risk": "You have a moderate risk of heart disease. Consider lifestyle improvements.",
            "High Risk": "Your risk is elevated. Consult a healthcare provider about risk reduction strategies.",
            "Very High Risk": "Your risk is significantly elevated. We strongly recommend consulting a healthcare provider."
        }
        
        interpretation_text = translate_text(interpretations[risk_result['risk_category']], self.language)
        content.append(Paragraph(interpretation_text, self.styles['Normal']))
        content.append(Spacer(1, 0.2*inch))
        
        # Add recommendations section
        content.append(Paragraph(translate_text("Personalized Recommendations", self.language), self.styles['SakuraHeading']))
        
        # Generate recommendations based on risk factors
        recommendations = []
        
        if user_data['smoker']:
            recommendations.append(translate_text("• Quit smoking - this could significantly reduce your heart disease risk.", self.language))
        
        if user_data['bmi'] > 25:
            recommendations.append(translate_text(f"• Aim for a healthier weight - a BMI of 18.5-24.9 is considered optimal for heart health.", self.language))
        
        if user_data['systolic_bp'] > 130:
            recommendations.append(translate_text("• Work on lowering your blood pressure through diet, exercise, and medication if prescribed.", self.language))
        
        if user_data['total_cholesterol'] > 200:
            recommendations.append(translate_text("• Reduce your total cholesterol through diet changes and medication if prescribed.", self.language))
        
        if user_data['hdl_cholesterol'] < 40:
            recommendations.append(translate_text("• Increase your HDL (good) cholesterol through exercise and healthy diet choices.", self.language))
        
        if not recommendations:
            recommendations.append(translate_text("• Continue maintaining your current healthy habits!", self.language))
        
        for recommendation in recommendations:
            content.append(Paragraph(recommendation, self.styles['Normal']))
        
        # Add time series data if available
        if time_series_data and 'risk_trend' in time_series_data:
            content.append(Spacer(1, 0.2*inch))
            content.append(Paragraph(translate_text("Risk Trend Over Time", self.language), self.styles['SakuraHeading']))
            
            # Add time series chart if available
            if 'trend_chart' in time_series_data:
                trend_img = Image(time_series_data['trend_chart'], width=5*inch, height=3*inch)
                content.append(trend_img)
            
            # Add trend description
            if 'trend_description' in time_series_data:
                content.append(Paragraph(translate_text(time_series_data['trend_description'], self.language), self.styles['Normal']))
        
        # Add disclaimer
        content.append(Spacer(1, 0.5*inch))
        disclaimer = translate_text(
            "DISCLAIMER: This report is for informational purposes only and is not a substitute for professional medical advice. "
            "Always consult with qualified healthcare providers for medical advice, diagnosis, and treatment.",
            self.language
        )
        content.append(Paragraph(disclaimer, self.styles['Italic']))
        
        # Build the PDF
        doc.build(content)
        buffer.seek(0)
        
        return buffer

# Singleton instance
_generator_instance = None

def get_report_generator(language="English"):
    global _generator_instance
    _generator_instance = HealthReportGenerator(language)  # Always create a new instance for language changes
    return _generator_instance