# config.py
import streamlit as st

# Custom CSS
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
    }
    .stProgress > div > div > div > div {
        background-color: #3B82F6;
    }
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
"""

# Page configuration
PAGE_CONFIG = {
    'page_title': "E-commerce Analytics Dashboard",
    'page_icon': "🛒",
    'layout': "wide",
    'initial_sidebar_state': "expanded"
}

# Initialize session state
def init_session_state():
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'features_extracted' not in st.session_state:
        st.session_state.features_extracted = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = {}
    if 'shap_values' not in st.session_state:
        st.session_state.shap_values = {}