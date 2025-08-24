import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Prediction Tool", page_icon="🎯", layout="wide")

st.title("🎯 Loan Default Prediction Tool")
st.markdown("Make real-time predictions for new loan applications using trained models.")

# Check if models are trained
if not st.session_state.get('models_trained', False):
    st.error("❌ No trained models found. Please train models first.")
    if st.button("Go to Model Training"):
        st.switch_page("pages/2_Model_Training.py")
    st.stop()

# Get trained models and feature information
trained_models = st.session_state.trained_models
feature_names = st.session_state.feature_names
label_encoders = st.session_state.get('label_encoders', {})

# Model selection
st.sidebar.header("Prediction Settings")

# Select model for prediction
if 'selected_deployment_model' in st.session_state:
    default_model = st.session_state.selected_deployment_model
else:
    default_model = list(trained_models.keys())[0]

selected_model_name = st.sidebar.selectbox(
    "Select Model:",
    list(trained_models.keys()),
    index=list(trained_models.keys()).index(default_model) if default_model in trained_models else 0
)

selected_model = trained_models[selected_model_name]

# Prediction mode
prediction_mode = st.sidebar.selectbox(
    "Prediction Mode:",
    ["Single Application", "Batch Predictions", "Risk Assessment"]
)

# Single Application Mode
if prediction_mode == "Single Application":
    st.subheader("📝 Loan Application Form")
    
    # Create input form based on original features
    input_data = {}
    
    # Get original data to understand feature ranges
    original_data = st.session_state.dataset
    
    # Organize features into categories
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Personal Information**")
        
        # Try to identify common loan features
        for feature in feature_names:
            if feature in original_data.columns:
                if original_data[feature].dtype in ['int64', 'float64']:
                    # Numeric feature
                    min_val = float(original_data[feature].min())
                    max_val = float(original_data[feature].max())
                    mean_val = float(original_data[feature].mean())
                    
                    # Special handling for common features
                    if 'age' in feature.lower():
                        input_data[feature] = st.slider(
                            f"{feature.replace('_', ' ').title()}",
                            min_value=18, max_value=100, value=35
                        )
                    elif 'amount' in feature.lower() or 'income' in feature.lower():
                        input_data[feature] = st.number_input(
                            f"{feature.replace('_', ' ').title()}",
                            min_value=min_val, max_value=max_val, value=mean_val
                        )
                    elif 'duration' in feature.lower() or 'term' in feature.lower():
                        input_data[feature] = st.slider(
                            f"{feature.replace('_', ' ').title()}",
                            min_value=int(min_val), max_value=int(max_val), value=int(mean_val)
                        )
                    else:
                        input_data[feature] = st.number_input(
                            f"{feature.replace('_', ' ').title()}",
                            value=mean_val
                        )
                else:
                    # Categorical feature
                    unique_values = original_data[feature].unique()
                    if feature in label_encoders:
                        # Show original values but encode them
                        try:
                            original_values = label_encoders[feature].classes_
                            selected_original = st.selectbox(
                                f"{feature.replace('_', ' ').title()}",
                                original_values
                            )
                            input_data[feature] = label_encoders[feature].transform([selected_original])[0]
                        except:
                            input_data[feature] = st.selectbox(
                                f"{feature.replace('_', ' ').title()}",
                                unique_values
                            )
                    else:
                        input_data[feature] = st.selectbox(
                            f"{feature.replace('_', ' ').title()}",
                            unique_values
                        )
    
    with col2:
        st.write("**Financial Information**")
        
        # Continue with remaining features
        remaining_features = [f for f in feature_names if f not in input_data]
        
        for feature in remaining_features[:10]:  # Limit to avoid too many inputs
            if feature in original_data.columns:
                if original_data[feature].dtype in ['int64', 'float64']:
                    min_val = float(original_data[feature].min())
                    max_val = float(original_data[feature].max())
                    mean_val = float(original_data[feature].mean())
                    
                    input_data[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}",
                        value=mean_val,
                        key=f"input_{feature}"
                    )
                else:
                    unique_values = original_data[feature].unique()
                    if feature in label_encoders:
                        try:
                            original_values = label_encoders[feature].classes_
                            selected_original = st.selectbox(
                                f"{feature.replace('_', ' ').title()}",
                                original_values,
                                key=f"input_{feature}"
                            )
                            input_data[feature] = label_encoders[feature].transform([selected_original])[0]
                        except:
                            input_data[feature] = st.selectbox(
                                f"{feature.replace('_', ' ').title()}",
                                unique_values,
                                key=f"input_{feature}"
                            )
                    else:
                        input_data[feature] = st.selectbox(
                            f"{feature.replace('_', ' ').title()}",
                            unique_values,
                            key=f"input_{feature}"
                        )
    
    # Make prediction
    if st.button("🔮 Predict Default Risk", type="primary"):
        try:
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            
            # Ensure all features are present
            for feature in feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0  # Default value for missing features
            
            # Reorder columns to match training data
            input_df = input_df[feature_names]
            
            # Apply scaling if it was used
            if 'feature_scaler' in st.session_state:
                numeric_cols = original_data.select_dtypes(include=[np.number]).columns
                numeric_indices = [i for i, col in enumerate(feature_names) if col in numeric_cols]
                if numeric_indices:
                    input_scaled = input_df.copy()
                    input_scaled.iloc[:, numeric_indices] = st.session_state.feature_scaler.transform(
                        input_df.iloc[:, numeric_indices]
                    )
                    input_df = input_scaled
            
            # Make prediction
            prediction = selected_model.predict(input_df)[0]
            prediction_proba = selected_model.predict_proba(input_df)[0] if hasattr(selected_model, 'predict_proba') else [1-prediction, prediction]
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.error("❌ **HIGH RISK**")
                    st.write("Loan application likely to default")
                else:
                    st.success("✅ **LOW RISK**")
                    st.write("Loan application unlikely to default")
            
            with col2:
                default_probability = prediction_proba[1] * 100
                st.metric("Default Probability", f"{default_probability:.1f}%")
            
            with col3:
                no_default_probability = prediction_proba[0] * 100
                st.metric("No Default Probability", f"{no_default_probability:.1f}%")
            
            # Probability visualization
            fig = go.Figure(go.Bar(
                x=['No Default', 'Default'],
                y=[prediction_proba[0], prediction_proba[1]],
                marker_color=['green', 'red']
            ))
            
            fig.update_layout(
                title="Default Risk Probability",
                yaxis_title="Probability",
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk assessment
            st.subheader("📊 Risk Assessment")
            
            if default_probability < 20:
                risk_level = "Very Low"
                risk_color = "green"
                recommendation = "✅ Approve loan with standard terms"
            elif default_probability < 40:
                risk_level = "Low"
                risk_color = "lightgreen"
                recommendation = "✅ Approve loan, consider standard terms"
            elif default_probability < 60:
                risk_level = "Medium"
                risk_color = "orange"
                recommendation = "⚠️ Approve with higher interest rate or additional collateral"
            elif default_probability < 80:
                risk_level = "High"
                risk_color = "red"
                recommendation = "❌ Consider rejection or require significant collateral"
            else:
                risk_level = "Very High"
                risk_color = "darkred"
                recommendation = "❌ Reject loan application"
            
            st.markdown(f"""
            **Risk Level**: <span style="color: {risk_color}; font-weight: bold;">{risk_level}</span>
            
            **Recommendation**: {recommendation}
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Batch Predictions Mode
elif prediction_mode == "Batch Predictions":
    st.subheader("📊 Batch Loan Predictions")
    
    uploaded_file = st.file_uploader("Upload CSV file with loan applications", type="csv")
    
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            
            st.write("**Uploaded Data Preview:**")
            st.dataframe(batch_data.head(), use_container_width=True)
            
            if st.button("🔮 Generate Batch Predictions"):
                # Process batch data
                predictions = []
                probabilities = []
                
                for idx, row in batch_data.iterrows():
                    try:
                        # Prepare row data
                        row_data = {}
                        for feature in feature_names:
                            if feature in batch_data.columns:
                                row_data[feature] = row[feature]
                            else:
                                row_data[feature] = 0  # Default value
                        
                        input_df = pd.DataFrame([row_data])
                        input_df = input_df[feature_names]
                        
                        # Apply scaling if used
                        if 'feature_scaler' in st.session_state:
                            numeric_cols = original_data.select_dtypes(include=[np.number]).columns
                            numeric_indices = [i for i, col in enumerate(feature_names) if col in numeric_cols]
                            if numeric_indices:
                                input_scaled = input_df.copy()
                                input_scaled.iloc[:, numeric_indices] = st.session_state.feature_scaler.transform(
                                    input_df.iloc[:, numeric_indices]
                                )
                                input_df = input_scaled
                        
                        # Predict
                        pred = selected_model.predict(input_df)[0]
                        pred_proba = selected_model.predict_proba(input_df)[0] if hasattr(selected_model, 'predict_proba') else [1-pred, pred]
                        
                        predictions.append(pred)
                        probabilities.append(pred_proba[1])
                        
                    except Exception as e:
                        predictions.append(-1)  # Error code
                        probabilities.append(0)
                
                # Add predictions to data
                batch_data['Prediction'] = predictions
                batch_data['Default_Probability'] = probabilities
                batch_data['Risk_Level'] = pd.cut(
                    batch_data['Default_Probability'],
                    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
                )
                
                # Display results
                st.subheader("📋 Batch Prediction Results")
                st.dataframe(batch_data, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_applications = len(batch_data)
                    st.metric("Total Applications", total_applications)
                
                with col2:
                    high_risk = len(batch_data[batch_data['Prediction'] == 1])
                    st.metric("High Risk", high_risk)
                
                with col3:
                    avg_risk = batch_data['Default_Probability'].mean() * 100
                    st.metric("Average Risk", f"{avg_risk:.1f}%")
                
                with col4:
                    approval_rate = (len(batch_data[batch_data['Prediction'] == 0]) / total_applications) * 100
                    st.metric("Approval Rate", f"{approval_rate:.1f}%")
                
                # Risk distribution
                risk_counts = batch_data['Risk_Level'].value_counts()
                
                fig = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title="Risk Level Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                csv = batch_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error processing batch file: {str(e)}")

# Risk Assessment Mode
elif prediction_mode == "Risk Assessment":
    st.subheader("📈 Portfolio Risk Assessment")
    
    # Risk analysis parameters
    col1, col2 = st.columns(2)
    
    with col1:
        portfolio_size = st.number_input("Portfolio Size", min_value=100, max_value=10000, value=1000)
        avg_loan_amount = st.number_input("Average Loan Amount ($)", min_value=1000, value=10000)
    
    with col2:
        risk_threshold = st.slider("Risk Threshold (%)", 0, 100, 50)
        simulation_runs = st.number_input("Simulation Runs", min_value=100, max_value=1000, value=500)
    
    if st.button("🎯 Run Risk Assessment"):
        # Generate synthetic portfolio based on original data characteristics
        np.random.seed(42)
        
        portfolio_predictions = []
        portfolio_probabilities = []
        
        # Sample from original data to create synthetic portfolio
        for _ in range(portfolio_size):
            # Create synthetic application based on original data distribution
            synthetic_app = {}
            
            for feature in feature_names:
                if feature in original_data.columns:
                    if original_data[feature].dtype in ['int64', 'float64']:
                        # Sample from normal distribution based on original data
                        mean_val = original_data[feature].mean()
                        std_val = original_data[feature].std()
                        synthetic_app[feature] = np.random.normal(mean_val, std_val)
                    else:
                        # Random choice from categories
                        synthetic_app[feature] = np.random.choice(original_data[feature].dropna().values)
                else:
                    synthetic_app[feature] = 0
            
            # Make prediction
            try:
                input_df = pd.DataFrame([synthetic_app])
                input_df = input_df[feature_names]
                
                # Apply scaling if used
                if 'feature_scaler' in st.session_state:
                    numeric_cols = original_data.select_dtypes(include=[np.number]).columns
                    numeric_indices = [i for i, col in enumerate(feature_names) if col in numeric_cols]
                    if numeric_indices:
                        input_scaled = input_df.copy()
                        input_scaled.iloc[:, numeric_indices] = st.session_state.feature_scaler.transform(
                            input_df.iloc[:, numeric_indices]
                        )
                        input_df = input_scaled
                
                pred = selected_model.predict(input_df)[0]
                pred_proba = selected_model.predict_proba(input_df)[0] if hasattr(selected_model, 'predict_proba') else [1-pred, pred]
                
                portfolio_predictions.append(pred)
                portfolio_probabilities.append(pred_proba[1])
                
            except:
                portfolio_predictions.append(0)
                portfolio_probabilities.append(0.1)
        
        # Risk assessment results
        portfolio_df = pd.DataFrame({
            'Loan_ID': range(1, portfolio_size + 1),
            'Prediction': portfolio_predictions,
            'Default_Probability': portfolio_probabilities
        })
        
        # Calculate risk metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            expected_defaults = sum(portfolio_probabilities)
            st.metric("Expected Defaults", f"{expected_defaults:.1f}")
        
        with col2:
            default_rate = (expected_defaults / portfolio_size) * 100
            st.metric("Expected Default Rate", f"{default_rate:.1f}%")
        
        with col3:
            high_risk_loans = len(portfolio_df[portfolio_df['Default_Probability'] > risk_threshold/100])
            st.metric("High Risk Loans", high_risk_loans)
        
        with col4:
            expected_loss = expected_defaults * avg_loan_amount * 0.6  # Assuming 60% loss given default
            st.metric("Expected Loss ($)", f"${expected_loss:,.0f}")
        
        # Risk distribution
        fig = px.histogram(
            portfolio_df,
            x='Default_Probability',
            nbins=30,
            title="Portfolio Risk Distribution"
        )
        fig.add_vline(x=risk_threshold/100, line_dash="dash", line_color="red", 
                     annotation_text=f"Risk Threshold ({risk_threshold}%)")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Value at Risk (VaR) calculation
        potential_losses = np.array(portfolio_probabilities) * avg_loan_amount * 0.6
        var_95 = np.percentile(potential_losses, 95)
        var_99 = np.percentile(potential_losses, 99)
        
        st.subheader("📊 Value at Risk (VaR)")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("95% VaR", f"${var_95:,.0f}")
        
        with col2:
            st.metric("99% VaR", f"${var_99:,.0f}")

# Model information
st.markdown("---")
st.subheader("ℹ️ Model Information")

col1, col2 = st.columns(2)

with col1:
    st.write(f"**Selected Model**: {selected_model_name}")
    st.write(f"**Features Used**: {len(feature_names)}")
    
    # Model parameters (if available)
    if hasattr(selected_model, 'get_params'):
        params = selected_model.get_params()
        st.write("**Key Parameters**:")
        for key, value in list(params.items())[:5]:  # Show first 5 parameters
            st.write(f"- {key}: {value}")

with col2:
    # Feature importance (if available)
    if hasattr(selected_model, 'feature_importances_'):
        actual_model = selected_model.best_estimator_ if hasattr(selected_model, 'best_estimator_') else selected_model
        
        if hasattr(actual_model, 'feature_importances_'):
            top_features = pd.DataFrame({
                'Feature': feature_names,
                'Importance': actual_model.feature_importances_
            }).sort_values('Importance', ascending=False).head(5)
            
            st.write("**Top 5 Important Features**:")
            for _, row in top_features.iterrows():
                st.write(f"- {row['Feature']}: {row['Importance']:.3f}")

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Back to Data Explorer"):
        st.switch_page("pages/1_Data_Explorer.py")

with col2:
    if st.button("🤖 Retrain Models"):
        st.switch_page("pages/2_Model_Training.py")

with col3:
    if st.button("📈 Model Evaluation"):
        st.switch_page("pages/3_Model_Evaluation.py")
