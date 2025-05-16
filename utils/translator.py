import os
import streamlit as st
from google.cloud import translate_v2 as translate

def get_translate_client():
    """Get or initialize the translation client
    
    In production, set up proper authentication using environment variables:
    GOOGLE_APPLICATION_CREDENTIALS pointing to your service account JSON file
    
    For demo/development, we'll try to load credentials or use a dummy translator
    """
    try:
        # Try to initialize the Google Translate client
        translate_client = translate.Client()
        return translate_client
    except Exception as e:
        print(f"Error initializing Google Translate client: {e}")
        print("Using fallback translator")
        
        # Create a dummy translator class for demo purposes
        class DummyTranslator:
            def translate(self, text, target_language=None):
                return {"translatedText": text}
        
        return DummyTranslator()

def get_language_code(language_name):
    """Convert language name to appropriate code for Google Translate API"""
    language_map = {
        "English": "en",
        "日本語": "ja",
        "Español": "es",
        "Français": "fr",
        "中文": "zh-CN"
    }
    return language_map.get(language_name, "en")

def translate_text(text, language="English"):
    """Translate text to the specified language using Google Translate API
    
    Args:
        text (str): Text to translate
        language (str): Target language name (e.g., "日本語", "Español")
        
    Returns:
        str: Translated text
    """
    # No translation needed for English
    if language == "English":
        return text
    
    # Convert language name to code
    target_language = get_language_code(language)
    
    # Cache translations to avoid repeated API calls
    cache_key = f"{text}_{target_language}"
    if "translation_cache" not in st.session_state:
        st.session_state.translation_cache = {}
    
    if cache_key in st.session_state.translation_cache:
        return st.session_state.translation_cache[cache_key]
        
    try:
        # Get translation client
        translate_client = get_translate_client()
        
        # Translate text
        result = translate_client.translate(text, target_language=target_language)
        translated_text = result.get("translatedText", text)
        
        # Cache the result
        st.session_state.translation_cache[cache_key] = translated_text
        
        return translated_text
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Fallback to original text