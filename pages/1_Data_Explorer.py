import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from utils.data_processor import DataProcessor
from utils.visualizations import create_correlation_heatmap, create_distribution_plots
from data.sample_datasets import load_german_credit_data, load_lending_club_sample

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")

# Apply the same fancy styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .page-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00D4AA 0%, #00A3FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s ease-out;
    }
    
    .page-subtitle {
        font-size: 1.1rem;
        color: #B8BCC8;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .upload-section {
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.1) 0%, rgba(0, 163, 255, 0.1) 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed rgba(0, 212, 170, 0.3);
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: rgba(0, 212, 170, 0.6);
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.15) 0%, rgba(0, 163, 255, 0.15) 100%);
    }
    
    .data-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .data-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Enhanced header
st.markdown("""
<h1 class="page-header">📊 Data Explorer</h1>
<p class="page-subtitle">Load and explore credit risk datasets with interactive visualizations</p>
""", unsafe_allow_html=True)

# Enhanced sidebar with fancy styling
st.sidebar.markdown("""
<style>
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #00D4AA;
        margin-bottom: 1rem;
        text-align: center;
    }
</style>
<div class="sidebar-header">🎯 Data Loading Hub</div>
""", unsafe_allow_html=True)

data_source = st.sidebar.selectbox(
    "📂 Select Data Source",
    ["Upload CSV File", "German Credit Risk Dataset", "Lending Club Sample"],
    help="Choose your preferred data source for analysis"
)

# Initialize DataProcessor
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()

def load_data():
    """Load data based on selected source"""
    if data_source == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                return data
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                return None
    
    elif data_source == "German Credit Risk Dataset":
        return load_german_credit_data()
    
    elif data_source == "Lending Club Sample":
        return load_lending_club_sample()
    
    return None

# Load data
if st.sidebar.button("Load Data") or st.session_state.data_loaded:
    data = load_data()
    
    if data is not None:
        st.session_state.dataset = data
        st.session_state.data_loaded = True
        
        # Data overview
        st.subheader("📋 Dataset Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(data.head(10), use_container_width=True)
        
        with col2:
            st.write("**Dataset Info:**")
            st.write(f"- Rows: {len(data):,}")
            st.write(f"- Columns: {len(data.columns):,}")
            st.write(f"- Memory Usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Data types
            st.write("**Data Types:**")
            dtype_counts = data.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                st.write(f"- {dtype}: {count}")
        
        # Missing values analysis
        st.subheader("🔍 Missing Values Analysis")
        
        missing_data = data.isnull().sum()
        missing_pct = (missing_data / len(data)) * 100
        
        if missing_data.sum() > 0:
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing Percentage': missing_pct.values
            }).sort_values('Missing Count', ascending=False)
            
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.dataframe(missing_df, use_container_width=True)
            
            with col2:
                # Missing values visualization
                fig = px.bar(
                    missing_df.head(10), 
                    x='Missing Percentage', 
                    y='Column',
                    orientation='h',
                    title="Top 10 Columns with Missing Values"
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found in the dataset!")
        
        # Target variable analysis (if exists)
        target_columns = ['target', 'default', 'bad_loan', 'loan_status', 'class']
        target_col = None
        
        for col in target_columns:
            if col in data.columns:
                target_col = col
                break
        
        if target_col:
            st.subheader(f"🎯 Target Variable Analysis: {target_col}")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                # Target distribution
                target_counts = data[target_col].value_counts()
                fig = px.pie(
                    values=target_counts.values,
                    names=target_counts.index,
                    title="Target Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Target statistics
                st.write("**Target Statistics:**")
                if data[target_col].dtype in ['int64', 'float64']:
                    st.write(f"- Mean: {data[target_col].mean():.3f}")
                    st.write(f"- Std: {data[target_col].std():.3f}")
                
                value_counts = data[target_col].value_counts()
                for value, count in value_counts.items():
                    pct = (count / len(data)) * 100
                    st.write(f"- {value}: {count:,} ({pct:.1f}%)")
            
            with col3:
                # Class imbalance visualization
                if len(target_counts) == 2:
                    imbalance_ratio = target_counts.min() / target_counts.max()
                    st.metric("Class Imbalance Ratio", f"{imbalance_ratio:.3f}")
                    
                    if imbalance_ratio < 0.1:
                        st.warning("⚠️ Severe class imbalance detected!")
                    elif imbalance_ratio < 0.3:
                        st.warning("⚠️ Moderate class imbalance detected!")
                    else:
                        st.success("✅ Balanced classes")
        
        # Numerical features analysis
        st.subheader("📊 Numerical Features Analysis")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        if numeric_cols:
            # Statistical summary
            st.write("**Statistical Summary:**")
            st.dataframe(data[numeric_cols].describe(), use_container_width=True)
            
            # Distribution plots
            if len(numeric_cols) > 0:
                selected_features = st.multiselect(
                    "Select features to visualize distributions:",
                    numeric_cols,
                    default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
                )
                
                if selected_features:
                    n_cols = min(2, len(selected_features))
                    n_rows = (len(selected_features) + n_cols - 1) // n_cols
                    
                    fig = make_subplots(
                        rows=n_rows, 
                        cols=n_cols,
                        subplot_titles=selected_features
                    )
                    
                    for i, feature in enumerate(selected_features):
                        row = i // n_cols + 1
                        col = i % n_cols + 1
                        
                        fig.add_trace(
                            go.Histogram(x=data[feature], name=feature, showlegend=False),
                            row=row, col=col
                        )
                    
                    fig.update_layout(height=300 * n_rows, title="Feature Distributions")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Categorical features analysis
        st.subheader("📝 Categorical Features Analysis")
        
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_cols:
            selected_cat_feature = st.selectbox(
                "Select categorical feature to analyze:",
                categorical_cols
            )
            
            if selected_cat_feature:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Value counts
                    value_counts = data[selected_cat_feature].value_counts().head(10)
                    st.write(f"**Top 10 values in {selected_cat_feature}:**")
                    st.dataframe(value_counts.to_frame(), use_container_width=True)
                
                with col2:
                    # Bar chart
                    fig = px.bar(
                        x=value_counts.values,
                        y=value_counts.index,
                        orientation='h',
                        title=f"Distribution of {selected_cat_feature}"
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        if len(numeric_cols) > 1:
            st.subheader("🔗 Correlation Analysis")
            
            # Calculate correlation matrix
            corr_matrix = data[numeric_cols].corr()
            
            # Correlation heatmap
            fig = create_correlation_heatmap(corr_matrix)
            st.plotly_chart(fig, use_container_width=True)
            
            # High correlations
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:  # High correlation threshold
                        high_corr_pairs.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': corr_val
                        })
            
            if high_corr_pairs:
                st.write("**High Correlations (|r| > 0.7):**")
                high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', key=abs, ascending=False)
                st.dataframe(high_corr_df, use_container_width=True)
        
        # Outlier detection
        st.subheader("🎯 Outlier Detection")
        
        if numeric_cols:
            selected_outlier_feature = st.selectbox(
                "Select feature for outlier analysis:",
                numeric_cols,
                key="outlier_feature"
            )
            
            if selected_outlier_feature:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Box plot
                    fig = px.box(
                        y=data[selected_outlier_feature],
                        title=f"Box Plot: {selected_outlier_feature}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Outlier statistics using IQR method
                    Q1 = data[selected_outlier_feature].quantile(0.25)
                    Q3 = data[selected_outlier_feature].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = data[
                        (data[selected_outlier_feature] < lower_bound) | 
                        (data[selected_outlier_feature] > upper_bound)
                    ]
                    
                    st.write("**Outlier Statistics (IQR Method):**")
                    st.write(f"- Lower Bound: {lower_bound:.2f}")
                    st.write(f"- Upper Bound: {upper_bound:.2f}")
                    st.write(f"- Outliers Count: {len(outliers):,}")
                    st.write(f"- Outliers Percentage: {(len(outliers)/len(data)*100):.1f}%")

else:
    # Enhanced getting started section
    st.markdown("""
    <div class="upload-section">
        <h2>🚀 Ready to Get Started?</h2>
        <p>Select a data source from the sidebar and click <strong>'Load Data'</strong> to begin your credit risk analysis journey!</p>
        <div style="margin-top: 1.5rem;">
            <span style="background: rgba(0, 212, 170, 0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;">
                📊 Interactive Visualizations
            </span>
            <span style="background: rgba(0, 163, 255, 0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;">
                🔍 Smart Analytics
            </span>
            <span style="background: rgba(255, 107, 107, 0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;">
                ⚡ Real-time Processing
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced sample data information
    st.markdown("### 📚 Available Sample Datasets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="data-card">
            <h4 style="color: #00D4AA;">🇩🇪 German Credit Risk Dataset</h4>
            <div style="margin-top: 1rem;">
                <span style="background: rgba(0, 212, 170, 0.2); padding: 0.3rem 0.6rem; border-radius: 10px; font-size: 0.8rem; margin-right: 0.5rem;">
                    1,000 records
                </span>
                <span style="background: rgba(0, 163, 255, 0.2); padding: 0.3rem 0.6rem; border-radius: 10px; font-size: 0.8rem; margin-right: 0.5rem;">
                    20 features
                </span>
            </div>
            <p style="margin-top: 1rem; color: #B8BCC8;">
                Classic benchmark dataset for binary credit classification. Perfect for learning and prototyping.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="data-card">
            <h4 style="color: #00A3FF;">🏦 Lending Club Sample</h4>
            <div style="margin-top: 1rem;">
                <span style="background: rgba(255, 107, 107, 0.2); padding: 0.3rem 0.6rem; border-radius: 10px; font-size: 0.8rem; margin-right: 0.5rem;">
                    Comprehensive
                </span>
                <span style="background: rgba(76, 175, 80, 0.2); padding: 0.3rem 0.6rem; border-radius: 10px; font-size: 0.8rem; margin-right: 0.5rem;">
                    Real structure
                </span>
            </div>
            <p style="margin-top: 1rem; color: #B8BCC8;">
                Detailed loan characteristics with financial metrics and performance indicators for advanced analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
