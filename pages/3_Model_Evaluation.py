import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

st.set_page_config(page_title="Model Evaluation", page_icon="📈", layout="wide")

st.title("📈 Model Evaluation")
st.markdown("Comprehensive evaluation and comparison of trained machine learning models.")

# Check if models are trained
if not st.session_state.get('models_trained', False):
    st.error("❌ No trained models found. Please train models first.")
    if st.button("Go to Model Training"):
        st.switch_page("pages/2_Model_Training.py")
    st.stop()

# Get trained models and data
trained_models = st.session_state.trained_models
X_train = st.session_state.X_train
X_test = st.session_state.X_test
y_train = st.session_state.y_train
y_test = st.session_state.y_test

# Model selection for detailed analysis
st.sidebar.header("Evaluation Options")
selected_model = st.sidebar.selectbox(
    "Select Model for Detailed Analysis:",
    list(trained_models.keys())
)

evaluation_type = st.sidebar.selectbox(
    "Evaluation Type:",
    ["Overview", "Detailed Analysis", "Model Comparison", "Business Metrics"]
)

# Overview Section
if evaluation_type == "Overview":
    st.subheader("📊 Model Performance Overview")
    
    # Calculate metrics for all models
    overview_results = []
    predictions = {}
    probabilities = {}
    
    for model_name, model in trained_models.items():
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        predictions[model_name] = y_pred
        probabilities[model_name] = y_pred_proba
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_pred_proba)
        
        overview_results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUC-ROC': auc
        })
    
    # Display overview table
    overview_df = pd.DataFrame(overview_results)
    st.dataframe(overview_df.round(4), use_container_width=True)
    
    # Performance visualization
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    fig = go.Figure()
    
    for _, row in overview_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[metric] for metric in metrics],
            theta=metrics,
            fill='toself',
            name=row['Model']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Model Performance Radar Chart"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ROC Curves Comparison
    st.subheader("📈 ROC Curves Comparison")
    
    fig = go.Figure()
    
    for model_name in trained_models.keys():
        fpr, tpr, _ = roc_curve(y_test, probabilities[model_name])
        auc = roc_auc_score(y_test, probabilities[model_name])
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {auc:.3f})',
            line=dict(width=3)
        ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random (AUC = 0.500)',
        line=dict(dash='dash', color='red', width=2)
    ))
    
    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Detailed Analysis Section
elif evaluation_type == "Detailed Analysis":
    st.subheader(f"🔍 Detailed Analysis: {selected_model}")
    
    model = trained_models[selected_model]
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{accuracy:.4f}")
    
    with col2:
        precision = precision_score(y_test, y_pred, average='weighted')
        st.metric("Precision", f"{precision:.4f}")
    
    with col3:
        recall = recall_score(y_test, y_pred, average='weighted')
        st.metric("Recall", f"{recall:.4f}")
    
    with col4:
        auc = roc_auc_score(y_test, y_pred_proba)
        st.metric("AUC-ROC", f"{auc:.4f}")
    
    # Confusion Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=['Predicted 0', 'Predicted 1'],
            y=['Actual 0', 'Actual 1'],
            annotation_text=cm,
            colorscale='Blues'
        )
        fig.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Convert to DataFrame for better display
        report_df = pd.DataFrame(report).transpose().round(4)
        st.dataframe(report_df, use_container_width=True)
    
    # ROC and Precision-Recall Curves
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{selected_model} (AUC = {auc:.3f})',
            line=dict(width=3)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash', color='red')
        ))
        
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Precision-Recall Curve")
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall_vals, y=precision_vals,
            mode='lines',
            name=f'{selected_model}',
            line=dict(width=3)
        ))
        
        fig.update_layout(
            xaxis_title='Recall',
            yaxis_title='Precision',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Prediction Distribution
    st.subheader("Prediction Probability Distribution")
    
    fig = go.Figure()
    
    # Separate probabilities by actual class
    prob_class_0 = y_pred_proba[y_test == 0]
    prob_class_1 = y_pred_proba[y_test == 1]
    
    fig.add_trace(go.Histogram(
        x=prob_class_0,
        name='Actual Class 0',
        opacity=0.7,
        nbinsx=30
    ))
    
    fig.add_trace(go.Histogram(
        x=prob_class_1,
        name='Actual Class 1',
        opacity=0.7,
        nbinsx=30
    ))
    
    fig.update_layout(
        title='Distribution of Predicted Probabilities by Actual Class',
        xaxis_title='Predicted Probability',
        yaxis_title='Count',
        barmode='overlay'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Model Comparison Section
elif evaluation_type == "Model Comparison":
    st.subheader("⚖️ Model Comparison")
    
    # Calculate comprehensive metrics
    comparison_data = []
    
    for model_name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1-Score': f1_score(y_test, y_pred, average='weighted'),
            'AUC-ROC': roc_auc_score(y_test, y_pred_proba)
        }
        
        comparison_data.append(metrics)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Heatmap comparison
    metrics_only = comparison_df.set_index('Model')
    
    fig = px.imshow(
        metrics_only.T,
        aspect="auto",
        color_continuous_scale="RdYlGn",
        title="Model Performance Heatmap"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed comparison table
    st.subheader("📋 Detailed Metrics Comparison")
    
    # Add ranking
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']:
        comparison_df[f'{metric}_Rank'] = comparison_df[metric].rank(ascending=False, method='min').astype(int)
    
    st.dataframe(comparison_df.round(4), use_container_width=True)
    
    # Best model identification
    st.subheader("🏆 Model Rankings")
    
    ranking_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    for metric in ranking_metrics:
        best_model = comparison_df.loc[comparison_df[metric].idxmax(), 'Model']
        best_score = comparison_df[metric].max()
        
        st.write(f"**Best {metric}**: {best_model} ({best_score:.4f})")

# Business Metrics Section
elif evaluation_type == "Business Metrics":
    st.subheader("💼 Business Impact Analysis")
    
    # Business parameters
    st.subheader("Business Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_loan_amount = st.number_input("Average Loan Amount ($)", value=10000, min_value=1000)
    
    with col2:
        cost_of_default = st.slider("Cost of Default (%)", 0, 100, 80)
    
    with col3:
        profit_margin = st.slider("Profit Margin (%)", 0, 50, 10)
    
    # Calculate business metrics for each model
    business_results = []
    
    for model_name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Confusion matrix elements
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Business calculations
        total_loans = len(y_test)
        approved_loans = np.sum(y_pred == 0)  # Assuming 0 means approved
        rejected_loans = np.sum(y_pred == 1)  # Assuming 1 means rejected
        
        # Revenue from approved loans (assuming they don't default)
        revenue_from_approved = tn * avg_loan_amount * (profit_margin / 100)
        
        # Losses from defaults among approved loans
        losses_from_defaults = fp * avg_loan_amount * (cost_of_default / 100)
        
        # Opportunity cost from rejected good loans
        opportunity_cost = fn * avg_loan_amount * (profit_margin / 100)
        
        # Net profit
        net_profit = revenue_from_approved - losses_from_defaults - opportunity_cost
        
        business_results.append({
            'Model': model_name,
            'Approved Loans': approved_loans,
            'Rejected Loans': rejected_loans,
            'Approval Rate (%)': (approved_loans / total_loans) * 100,
            'Revenue ($)': revenue_from_approved,
            'Losses ($)': losses_from_defaults,
            'Opportunity Cost ($)': opportunity_cost,
            'Net Profit ($)': net_profit,
            'ROI (%)': (net_profit / (total_loans * avg_loan_amount)) * 100
        })
    
    # Display business metrics
    business_df = pd.DataFrame(business_results)
    st.dataframe(business_df.round(2), use_container_width=True)
    
    # Visualize business impact
    fig = px.bar(
        business_df,
        x='Model',
        y='Net Profit ($)',
        title='Net Profit by Model',
        color='Net Profit ($)',
        color_continuous_scale='RdYlGn'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ROI Comparison
    fig = px.bar(
        business_df,
        x='Model',
        y='ROI (%)',
        title='Return on Investment by Model',
        color='ROI (%)',
        color_continuous_scale='RdYlGn'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Best model for business
    best_business_model = business_df.loc[business_df['Net Profit ($)'].idxmax(), 'Model']
    best_profit = business_df['Net Profit ($)'].max()
    
    st.success(f"🏆 **Best Model for Business**: {best_business_model} with ${best_profit:,.2f} net profit")

# Model Selection for Deployment
st.markdown("---")
st.subheader("🚀 Model Selection for Deployment")

recommended_model = None
recommendation_reason = ""

# Simple recommendation logic
if 'overview_df' in locals():
    # Recommend based on AUC-ROC if overview is available
    best_auc_model = overview_df.loc[overview_df['AUC-ROC'].idxmax(), 'Model']
    recommended_model = best_auc_model
    recommendation_reason = f"Highest AUC-ROC score ({overview_df['AUC-ROC'].max():.4f})"
elif 'business_df' in locals():
    # Recommend based on business metrics
    best_business_model = business_df.loc[business_df['Net Profit ($)'].idxmax(), 'Model']
    recommended_model = best_business_model
    recommendation_reason = f"Highest net profit (${business_df['Net Profit ($)'].max():,.2f})"

if recommended_model:
    st.info(f"💡 **Recommended Model**: {recommended_model}")
    st.write(f"**Reason**: {recommendation_reason}")
    
    if st.button("🎯 Use This Model for Predictions"):
        st.session_state.selected_deployment_model = recommended_model
        st.switch_page("pages/4_Prediction_Tool.py")

# Navigation
col1, col2 = st.columns(2)

with col1:
    if st.button("🤖 Retrain Models"):
        st.switch_page("pages/2_Model_Training.py")

with col2:
    if st.button("🎯 Make Predictions"):
        st.switch_page("pages/4_Prediction_Tool.py")
