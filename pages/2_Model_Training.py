import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.model_trainer import ModelTrainer
from utils.data_processor import DataProcessor

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")

st.title("🤖 Model Training")
st.markdown("Train multiple machine learning models to predict loan default probability.")

# Check if data is loaded
if not st.session_state.get('data_loaded', False) or st.session_state.dataset is None:
    st.error("❌ No data loaded. Please load data in the Data Explorer first.")
    if st.button("Go to Data Explorer"):
        st.switch_page("pages/1_Data_Explorer.py")
    st.stop()

# Initialize components
if 'model_trainer' not in st.session_state:
    st.session_state.model_trainer = ModelTrainer()

if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()

data = st.session_state.dataset

# Sidebar configuration
st.sidebar.header("Training Configuration")

# Target variable selection
target_columns = ['target', 'default', 'bad_loan', 'loan_status', 'class']
available_targets = [col for col in target_columns if col in data.columns]

if not available_targets:
    # Let user select any column as target
    available_targets = data.columns.tolist()

target_column = st.sidebar.selectbox(
    "Select Target Variable:",
    available_targets
)

# Feature selection
st.sidebar.subheader("Feature Selection")
all_features = [col for col in data.columns if col != target_column]

# Automatic feature type detection
numeric_features = data[all_features].select_dtypes(include=[np.number]).columns.tolist()
categorical_features = data[all_features].select_dtypes(include=['object', 'category']).columns.tolist()

selected_numeric = st.sidebar.multiselect(
    "Select Numeric Features:",
    numeric_features,
    default=numeric_features[:10] if len(numeric_features) > 10 else numeric_features
)

selected_categorical = st.sidebar.multiselect(
    "Select Categorical Features:",
    categorical_features,
    default=categorical_features[:5] if len(categorical_features) > 5 else categorical_features
)

# Model selection
st.sidebar.subheader("Model Selection")
models_to_train = st.sidebar.multiselect(
    "Select Models to Train:",
    ["Logistic Regression", "Random Forest", "XGBoost"],
    default=["Logistic Regression", "Random Forest", "XGBoost"]
)

# Training parameters
st.sidebar.subheader("Training Parameters")
test_size = st.sidebar.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random State", value=42, min_value=0)
use_hyperparameter_tuning = st.sidebar.checkbox("Enable Hyperparameter Tuning", value=False)

# Data preprocessing section
st.subheader("📊 Data Preprocessing")

selected_features = selected_numeric + selected_categorical

if len(selected_features) == 0:
    st.warning("⚠️ Please select at least one feature to proceed.")
    st.stop()

# Show preprocessing steps
col1, col2 = st.columns([2, 1])

with col1:
    st.write("**Selected Features:**")
    feature_info = []
    for feature in selected_features:
        dtype = str(data[feature].dtype)
        missing_count = data[feature].isnull().sum()
        unique_count = data[feature].nunique()
        
        feature_info.append({
            'Feature': feature,
            'Type': dtype,
            'Missing': missing_count,
            'Unique Values': unique_count
        })
    
    feature_df = pd.DataFrame(feature_info)
    st.dataframe(feature_df, use_container_width=True)

with col2:
    st.write("**Preprocessing Summary:**")
    st.write(f"- Target: {target_column}")
    st.write(f"- Numeric Features: {len(selected_numeric)}")
    st.write(f"- Categorical Features: {len(selected_categorical)}")
    st.write(f"- Total Features: {len(selected_features)}")
    st.write(f"- Training Size: {int(len(data) * (1 - test_size)):,}")
    st.write(f"- Test Size: {int(len(data) * test_size):,}")

# Feature engineering options
st.subheader("🔧 Feature Engineering")

feature_engineering_options = st.columns(3)

with feature_engineering_options[0]:
    create_ratios = st.checkbox("Create Financial Ratios", value=True)

with feature_engineering_options[1]:
    handle_outliers = st.checkbox("Handle Outliers", value=False)

with feature_engineering_options[2]:
    scale_features = st.checkbox("Scale Numeric Features", value=True)

# Start training
if st.button("🚀 Start Training", type="primary"):
    
    # Prepare data
    with st.spinner("Preparing data..."):
        # Prepare features and target
        X = data[selected_features].copy()
        y = data[target_column].copy()
        
        # Handle missing values
        for col in selected_numeric:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].median())
        
        for col in selected_categorical:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
        
        # Encode categorical variables
        label_encoders = {}
        for col in selected_categorical:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
        
        # Feature engineering
        if create_ratios and len(selected_numeric) >= 2:
            # Create some common financial ratios
            numeric_cols = [col for col in selected_numeric if col in X.columns]
            if len(numeric_cols) >= 2:
                X[f'{numeric_cols[0]}_to_{numeric_cols[1]}_ratio'] = X[numeric_cols[0]] / (X[numeric_cols[1]] + 1e-6)
        
        # Handle target variable
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            st.session_state.target_encoder = le_target
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features if requested
        if scale_features:
            scaler = StandardScaler()
            numeric_indices = [i for i, col in enumerate(X.columns) if col in selected_numeric]
            if numeric_indices:
                X_train_scaled = X_train.copy()
                X_test_scaled = X_test.copy()
                
                X_train_scaled.iloc[:, numeric_indices] = scaler.fit_transform(X_train.iloc[:, numeric_indices])
                X_test_scaled.iloc[:, numeric_indices] = scaler.transform(X_test.iloc[:, numeric_indices])
                
                X_train, X_test = X_train_scaled, X_test_scaled
                st.session_state.feature_scaler = scaler
        
        # Store processed data
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.feature_names = X.columns.tolist()
        st.session_state.label_encoders = label_encoders
        st.session_state.processed_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
    
    st.success("✅ Data preparation completed!")
    
    # Train models
    st.subheader("🎯 Model Training Results")
    
    trained_models = {}
    training_results = []
    
    for model_name in models_to_train:
        with st.spinner(f"Training {model_name}..."):
            
            if model_name == "Logistic Regression":
                if use_hyperparameter_tuning:
                    param_grid = {
                        'C': [0.01, 0.1, 1, 10, 100],
                        'solver': ['liblinear', 'lbfgs']
                    }
                    model = GridSearchCV(
                        LogisticRegression(random_state=random_state),
                        param_grid,
                        cv=5,
                        scoring='roc_auc'
                    )
                else:
                    model = LogisticRegression(random_state=random_state)
            
            elif model_name == "Random Forest":
                if use_hyperparameter_tuning:
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5, 10]
                    }
                    model = GridSearchCV(
                        RandomForestClassifier(random_state=random_state),
                        param_grid,
                        cv=5,
                        scoring='roc_auc'
                    )
                else:
                    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
            
            elif model_name == "XGBoost":
                if use_hyperparameter_tuning:
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2]
                    }
                    model = GridSearchCV(
                        xgb.XGBClassifier(random_state=random_state, eval_metric='logloss'),
                        param_grid,
                        cv=5,
                        scoring='roc_auc'
                    )
                else:
                    model = xgb.XGBClassifier(random_state=random_state, eval_metric='logloss')
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            train_pred_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else train_pred
            test_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else test_pred
            
            # Calculate metrics
            train_auc = roc_auc_score(y_train, train_pred_proba)
            test_auc = roc_auc_score(y_test, test_pred_proba)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            
            # Store results
            trained_models[model_name] = model
            training_results.append({
                'Model': model_name,
                'Train AUC': train_auc,
                'Test AUC': test_auc,
                'CV AUC (mean)': cv_scores.mean(),
                'CV AUC (std)': cv_scores.std(),
                'Overfitting': train_auc - test_auc
            })
    
    # Store trained models
    st.session_state.trained_models = trained_models
    st.session_state.models_trained = True
    
    # Display results
    results_df = pd.DataFrame(training_results)
    st.dataframe(results_df, use_container_width=True)
    
    # Visualize model performance
    fig = go.Figure()
    
    for _, row in results_df.iterrows():
        fig.add_trace(go.Scatter(
            x=['Train AUC', 'Test AUC', 'CV AUC (mean)'],
            y=[row['Train AUC'], row['Test AUC'], row['CV AUC (mean)']],
            mode='lines+markers',
            name=row['Model'],
            line=dict(width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Metric",
        yaxis_title="AUC Score",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance for tree-based models
    st.subheader("🔍 Feature Importance")
    
    for model_name, model in trained_models.items():
        if hasattr(model, 'feature_importances_'):
            # Get the actual model (in case of GridSearchCV)
            actual_model = model.best_estimator_ if hasattr(model, 'best_estimator_') else model
            
            if hasattr(actual_model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': actual_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    importance_df.head(15),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f"Top 15 Feature Importances - {model_name}"
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
    
    st.success("🎉 Model training completed successfully!")
    
    # Navigation to evaluation
    if st.button("📈 Evaluate Models"):
        st.switch_page("pages/3_Model_Evaluation.py")

# Show current status
if st.session_state.get('models_trained', False):
    st.info("✅ Models are already trained. You can re-train with different parameters or proceed to evaluation.")
    
    if st.button("📈 Go to Model Evaluation"):
        st.switch_page("pages/3_Model_Evaluation.py")
