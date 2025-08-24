# Overview

A comprehensive credit risk assessment platform built with Streamlit that enables financial institutions to analyze loan default probability using machine learning. The application provides end-to-end functionality from data exploration to model deployment, featuring multiple ML algorithms (Logistic Regression, Random Forest, XGBoost) with interactive visualizations and real-time prediction capabilities.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Streamlit-based web application with multi-page architecture
- **Page Structure**: Modular design with separate pages for different functionalities:
  - Main dashboard (`app.py`) - Landing page with overview and quick stats
  - Data Explorer (`1_Data_Explorer.py`) - Data loading and exploratory data analysis
  - Model Training (`2_Model_Training.py`) - ML model training interface
  - Model Evaluation (`3_Model_Evaluation.py`) - Performance metrics and comparisons
  - Prediction Tool (`4_Prediction_Tool.py`) - Real-time loan default predictions
- **State Management**: Session-based state management for data persistence across pages
- **UI Components**: Interactive widgets, metrics displays, and responsive layouts with sidebar controls

## Data Processing Architecture
- **Data Handler**: Centralized `DataProcessor` class for data preprocessing pipeline
- **Feature Engineering**: Automatic detection of numeric/categorical features with custom financial ratio creation
- **Data Quality**: Comprehensive missing value handling using median/mode imputation strategies
- **Encoding Strategy**: Label encoding for categorical variables with encoder persistence
- **Scaling**: StandardScaler for numeric feature normalization

## Machine Learning Architecture
- **Model Framework**: Scikit-learn based with XGBoost integration
- **Supported Algorithms**: 
  - Logistic Regression with balanced class weights
  - Random Forest with ensemble methodology
  - XGBoost with gradient boosting
- **Training Pipeline**: Automated hyperparameter tuning with GridSearchCV
- **Model Evaluation**: Comprehensive metrics including ROC-AUC, precision-recall, confusion matrices
- **Cross-Validation**: Stratified K-fold validation for robust performance estimation

## Visualization System
- **Plotting Library**: Plotly for interactive visualizations with Matplotlib fallback
- **Chart Types**: Correlation heatmaps, distribution plots, ROC curves, precision-recall curves
- **Business Intelligence**: Model comparison dashboards and performance metrics visualization

## Data Sources
- **Built-in Datasets**: German Credit Risk dataset from UCI repository
- **Custom Upload**: CSV file upload functionality for proprietary datasets
- **Sample Data**: Lending Club sample dataset support
- **Data Validation**: Automatic target variable detection and feature type inference

# External Dependencies

## Core ML Libraries
- **scikit-learn**: Primary machine learning framework for model training and evaluation
- **XGBoost**: Gradient boosting implementation for advanced ensemble methods
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing foundation

## Visualization Dependencies
- **Plotly Express/Graph Objects**: Interactive plotting and dashboard creation
- **Seaborn**: Statistical visualization library
- **Matplotlib**: Base plotting functionality

## Web Framework
- **Streamlit**: Core web application framework with built-in widgets and state management

## Data Processing
- **sklearn.preprocessing**: Feature scaling, encoding, and imputation utilities
- **sklearn.model_selection**: Train-test splitting and cross-validation tools
- **sklearn.metrics**: Comprehensive model evaluation metrics

## Dataset Sources
- **UCI Machine Learning Repository**: German Credit Risk dataset via HTTPS API
- **Local CSV Files**: User-uploaded datasets through Streamlit file uploader