import streamlit as st
import base64
from utils.report_generator import get_report_generator
from utils.translator import translate_text
import time

def render_report_card(user_data, language):
    st.header(translate_text("Personalized Health Report Card", language))
    st.write(translate_text(
        "Generate a comprehensive health report that you can download and share with your healthcare provider.",
        language
    ))
    
    # Create columns for options and preview
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(translate_text("Report Options", language))
        
        # Include sections
        st.write(translate_text("Sections to include:", language))
        include_personal = st.checkbox(translate_text("Personal Information", language), value=True)
        include_metrics = st.checkbox(translate_text("Health Metrics", language), value=True)
        include_risk = st.checkbox(translate_text("Risk Assessment", language), value=True)
        include_recommendations = st.checkbox(translate_text("Recommendations", language), value=True)
        
        # Additional notes
        additional_notes = st.text_area(translate_text("Additional Notes for your Doctor", language), 
                                        height=100, max_chars=500)
        
        # Generate PDF button
        if st.button(translate_text("Generate PDF Report", language), type="primary"):
            with st.spinner(translate_text("Generating your report...", language)):
                # Get report generator
                report_generator = get_report_generator(language)
                
                # Add any time series data if available (placeholder for now)
                time_series_data = None
                try:
                    # This would come from the time series tracker in a real implementation
                    time_series_data = st.session_state.get('time_series_data', None)
                except:
                    pass
                
                # Generate the report
                pdf_buffer = report_generator.generate_report(user_data, time_series_data)
                
                # Create download link
                pdf_base64 = base64.b64encode(pdf_buffer.read()).decode('utf-8')
                
                # Create a download link
                href = f'<a href="data:application/pdf;base64,{pdf_base64}" download="health_report_{int(time.time())}.pdf">Click here to download your report</a>'
                
                # Store in session state for preview
                if 'pdf_base64' not in st.session_state:
                    st.session_state.pdf_base64 = pdf_base64
                else:
                    st.session_state.pdf_base64 = pdf_base64
                
                # Success message
                st.success(translate_text("Report generated successfully!", language))
                st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        st.subheader(translate_text("Report Preview", language))
        
        # Show PDF preview if available
        if 'pdf_base64' in st.session_state:
            # Display PDF using iframe
            pdf_display = f'<iframe src="data:application/pdf;base64,{st.session_state.pdf_base64}" width="700" height="600" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
        else:
            # Show placeholder
            st.info(translate_text("Your report preview will appear here after generation.", language))
            
            # Show sample preview image
            try:
                st.image("assets/sample_reports/sample_report.pdf", caption=translate_text("Sample Report", language))
            except:
                st.markdown("""
                ```
                Sample Report Preview
                ====================
                
                Personal Information:
                - Age: 45 years
                - Gender: Male
                - BMI: 27.5
                
                Risk Assessment:
                - 10-Year Risk: 15.2%
                - Category: High Risk
                
                Recommendations:
                - Focus on lowering blood pressure
                - Consider weight management
                - Discuss medication options with your doctor
                ```
                """)