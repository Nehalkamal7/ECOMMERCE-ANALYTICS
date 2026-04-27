# app.py
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from config import CUSTOM_CSS, PAGE_CONFIG, init_session_state
import pages.data_overview as data_overview
import pages.feature_engineering as feature_engineering
import pages.neural_networks as neural_networks
import pages.pattern_recognition as pattern_recognition
import pages.predictive_models as predictive_models
import pages.customer_segmentation as customer_segmentation
import pages.insights_patterns as insights_patterns
import pages.model_interpretability as model_interpretability
import pages.export_results as export_results







# Set page configuration
st.set_page_config(**PAGE_CONFIG)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🛒 Advanced E-commerce Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
This dashboard provides comprehensive analytics for e-commerce data including **neural networks**, 
**advanced pattern recognition**, customer segmentation, purchase prediction, and behavioral analysis.
""")

# Initialize session state
init_session_state()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["📊 Data Overview", "🔍 Advanced Feature Engineering", "🧠 Neural Networks", 
     "🔬 Advanced Pattern Recognition", "🎯 Predictive Models", "👥 Customer Segmentation",
     "📈 Insights & Patterns", "📊 Model Interpretability", "📤 Export Results"]
)

# Route to appropriate page
if page == "📊 Data Overview":
    data_overview.render()
elif page == "🔍 Advanced Feature Engineering":
    feature_engineering.render()
elif page == "🧠 Neural Networks":
    neural_networks.render()
elif page == "🔬 Advanced Pattern Recognition":
    pattern_recognition.render()
elif page == "🎯 Predictive Models":
    predictive_models.render()
elif page == "👥 Customer Segmentation":
    customer_segmentation.render()
elif page == "📈 Insights & Patterns":
    insights_patterns.render()
elif page == "📊 Model Interpretability":
    model_interpretability.render()
elif page == "📤 Export Results":
    export_results.render()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    **Advanced E-commerce Analytics Dashboard**  
    Version 2.0 with Neural Networks & Pattern Recognition
    
    Features:
    - 🧠 Neural Network Models
    - 🔬 Advanced Pattern Recognition
    - 📊 SHAP Model Interpretability
    - 🎯 Advanced Predictive Models
    - 👥 Customer Segmentation
    - 📈 Insights & Pattern Detection
    
    Upload your CSV data to get started.
    """
)