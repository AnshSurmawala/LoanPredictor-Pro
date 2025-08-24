import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
import plotly.express as px
import plotly.graph_objects as go
import time

# Configure page with enhanced styling
st.set_page_config(
    page_title="CreditRisk Pro | AI-Powered Risk Assessment",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': "https://www.example.com/bug",
        'About': "# CreditRisk Pro\nAI-powered credit risk assessment platform for financial institutions."
    }
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main styling */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom title styling */
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00D4AA 0%, #00A3FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s ease-out;
    }
    
    /* Subtitle styling */
    .subtitle {
        font-size: 1.2rem;
        color: #B8BCC8;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
        animation: fadeInUp 1s ease-out;
    }
    
    /* Card styling */
    .feature-card {
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.1) 0%, rgba(0, 163, 255, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
        animation: fadeIn 1.2s ease-out;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 212, 170, 0.2);
        border-color: rgba(0, 212, 170, 0.3);
    }
    
    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        transition: all 0.3s ease;
        animation: slideUp 0.8s ease-out;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #00D4AA 0%, #00A3FF 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        color: white !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 212, 170, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 212, 170, 0.4) !important;
    }
    
    /* Progress indicator */
    .progress-indicator {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0;
    }
    
    .step {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.1);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 10px;
        position: relative;
        transition: all 0.3s ease;
    }
    
    .step.active {
        background: linear-gradient(135deg, #00D4AA 0%, #00A3FF 100%);
        box-shadow: 0 4px 15px rgba(0, 212, 170, 0.4);
    }
    
    .step::after {
        content: '';
        position: absolute;
        left: 100%;
        top: 50%;
        width: 20px;
        height: 2px;
        background: rgba(255, 255, 255, 0.2);
        transform: translateY(-50%);
    }
    
    .step:last-child::after {
        display: none;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(14, 17, 23, 0.95) 0%, rgba(38, 39, 48, 0.95) 100%);
    }
    
    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Loading spinner */
    .loading-spinner {
        border: 3px solid rgba(255, 255, 255, 0.1);
        border-top: 3px solid #00D4AA;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

def create_progress_indicator():
    """Create an interactive progress indicator"""
    current_step = 1
    if st.session_state.data_loaded:
        current_step = 2
    if st.session_state.models_trained:
        current_step = 3
    
    steps = ["🏦", "📊", "🤖", "🎯"]
    step_names = ["Start", "Data", "Models", "Predict"]
    
    progress_html = '<div class="progress-indicator">'
    for i, (icon, name) in enumerate(zip(steps, step_names)):
        active_class = "active" if i < current_step else ""
        progress_html += f'<div class="step {active_class}" title="{name}">{icon}</div>'
    progress_html += '</div>'
    
    st.markdown(progress_html, unsafe_allow_html=True)

def create_animated_metrics():
    """Create animated metric cards"""
    if st.session_state.data_loaded and st.session_state.dataset is not None:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            with st.container():
                st.markdown("""
                <div class="metric-card">
                    <h3>📊</h3>
                    <h2>{:,}</h2>
                    <p>Total Records</p>
                </div>
                """.format(len(st.session_state.dataset)), unsafe_allow_html=True)
        
        with col2:
            if 'target' in st.session_state.dataset.columns:
                default_rate = st.session_state.dataset['target'].mean() * 100
                color = "#FF6B6B" if default_rate > 20 else "#4ECDC4"
                st.markdown(f"""
                <div class="metric-card">
                    <h3>⚠️</h3>
                    <h2 style="color: {color}">{default_rate:.1f}%</h2>
                    <p>Default Rate</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🏷️</h3>
                    <h2>{len(st.session_state.dataset.columns)}</h2>
                    <p>Features</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            missing_pct = (st.session_state.dataset.isnull().sum().sum() / 
                          (len(st.session_state.dataset) * len(st.session_state.dataset.columns))) * 100
            color = "#FF6B6B" if missing_pct > 10 else "#4ECDC4"
            st.markdown(f"""
            <div class="metric-card">
                <h3>❓</h3>
                <h2 style="color: {color}">{missing_pct:.1f}%</h2>
                <p>Missing Data</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            numeric_cols = len(st.session_state.dataset.select_dtypes(include=[np.number]).columns)
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔢</h3>
                <h2>{numeric_cols}</h2>
                <p>Numeric Features</p>
            </div>
            """, unsafe_allow_html=True)

def main():
    # Hero section with animated title
    st.markdown("""
    <h1 class="main-title">🏦 CreditRisk Pro</h1>
    <p class="subtitle">AI-Powered Credit Risk Assessment Platform</p>
    """, unsafe_allow_html=True)
    
    # Progress indicator
    create_progress_indicator()
    
    # Welcome section with feature cards
    st.markdown("### 🚀 Platform Overview")
    
    # Create feature cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Advanced ML Models</h3>
            <p>Train and compare multiple machine learning algorithms including Logistic Regression, Random Forest, and XGBoost for optimal prediction accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Smart Data Analysis</h3>
            <p>Comprehensive data exploration with interactive visualizations, correlation analysis, and automated feature engineering.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📈 Performance Insights</h3>
            <p>Detailed model evaluation with ROC curves, confusion matrices, business impact analysis, and risk assessment metrics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>⚡ Real-time Predictions</h3>
            <p>Instant loan default predictions with risk scoring, batch processing capabilities, and portfolio analysis tools.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Animated metrics
    if st.session_state.data_loaded and st.session_state.dataset is not None:
        st.markdown("### 📊 Dataset Insights")
        create_animated_metrics()
        
        # Create a mini chart if data is loaded
        if 'target' in st.session_state.dataset.columns:
            target_dist = st.session_state.dataset['target'].value_counts()
            
            fig = go.Figure(data=[
                go.Bar(
                    x=['No Default', 'Default'],
                    y=[target_dist.get(0, 0), target_dist.get(1, 0)],
                    marker_color=['#00D4AA', '#FF6B6B'],
                    text=[target_dist.get(0, 0), target_dist.get(1, 0)],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Target Distribution",
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        # Show getting started when no data loaded
        st.markdown("### 🚀 Quick Start Guide")
        
        guide_col1, guide_col2 = st.columns([1, 2])
        
        with guide_col1:
            st.markdown("""
            <div class="feature-card" style="height: 280px;">
                <h3>📋 Step-by-Step</h3>
                <div style="text-align: left;">
                    <p>1️⃣ <strong>Load Data</strong><br>Upload your credit risk dataset</p>
                    <p>2️⃣ <strong>Explore</strong><br>Analyze data patterns</p>
                    <p>3️⃣ <strong>Train</strong><br>Build ML models</p>
                    <p>4️⃣ <strong>Evaluate</strong><br>Compare performance</p>
                    <p>5️⃣ <strong>Predict</strong><br>Make real-time assessments</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with guide_col2:
            st.markdown("""
            <div class="feature-card" style="height: 280px;">
                <h3>🎯 Get Started</h3>
                <p style="margin-bottom: 20px;">Ready to analyze credit risk? Choose your starting point:</p>
            """, unsafe_allow_html=True)
            
            # Quick action buttons
            if st.button("🚀 Start with Sample Data", use_container_width=True):
                st.switch_page("pages/1_Data_Explorer.py")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("📤 Upload My Dataset", use_container_width=True):
                st.switch_page("pages/1_Data_Explorer.py")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Enhanced navigation section
    st.markdown("### 🧭 Navigation Hub")
    
    # Create interactive navigation cards
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    
    with nav_col1:
        if st.button("📊 Data Explorer", use_container_width=True, key="nav_data"):
            st.switch_page("pages/1_Data_Explorer.py")
        st.markdown("""
        <div style="text-align: center; color: #B8BCC8; font-size: 0.8rem; margin-top: 0.5rem;">
            Load & analyze datasets
        </div>
        """, unsafe_allow_html=True)
    
    with nav_col2:
        if st.button("🤖 Model Training", use_container_width=True, key="nav_train"):
            st.switch_page("pages/2_Model_Training.py")
        st.markdown("""
        <div style="text-align: center; color: #B8BCC8; font-size: 0.8rem; margin-top: 0.5rem;">
            Train ML algorithms
        </div>
        """, unsafe_allow_html=True)
    
    with nav_col3:
        if st.button("📈 Model Evaluation", use_container_width=True, key="nav_eval"):
            st.switch_page("pages/3_Model_Evaluation.py")
        st.markdown("""
        <div style="text-align: center; color: #B8BCC8; font-size: 0.8rem; margin-top: 0.5rem;">
            Compare performance
        </div>
        """, unsafe_allow_html=True)
    
    with nav_col4:
        if st.button("🎯 Risk Predictions", use_container_width=True, key="nav_predict"):
            st.switch_page("pages/4_Prediction_Tool.py")
        st.markdown("""
        <div style="text-align: center; color: #B8BCC8; font-size: 0.8rem; margin-top: 0.5rem;">
            Real-time predictions
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: rgba(255, 255, 255, 0.02); border-radius: 10px; margin-top: 2rem;'>
        <h4 style='color: #00D4AA; margin-bottom: 1rem;'>🏦 CreditRisk Pro</h4>
        <p style='color: #B8BCC8; margin-bottom: 0;'>Empowering financial institutions with AI-driven risk assessment</p>
        <small style='color: #666;'>Built with advanced machine learning • Streamlit framework</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
