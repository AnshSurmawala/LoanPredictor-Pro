import pandas as pd
import numpy as np
import streamlit as st
import os

def load_german_credit_data():
    """
    Load German Credit Risk dataset.
    This function attempts to load the German Credit dataset from UCI repository
    or provides a template structure if not available.
    """
    try:
        # Try to load from UCI repository
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
        
        # Column names for German Credit dataset
        column_names = [
            'checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
            'savings_status', 'employment', 'installment_commitment', 'personal_status',
            'other_parties', 'residence_since', 'property_magnitude', 'age',
            'other_payment_plans', 'housing', 'existing_credits', 'job',
            'num_dependents', 'own_telephone', 'foreign_worker', 'target'
        ]
        
        # Load data
        data = pd.read_csv(url, sep=' ', header=None, names=column_names)
        
        # Convert target to binary (1=bad credit, 0=good credit)
        data['target'] = data['target'] - 1  # Original is 1,2 -> convert to 0,1
        
        st.success("✅ German Credit dataset loaded successfully from UCI repository")
        return data
        
    except Exception as e:
        st.error(f"❌ Could not load German Credit dataset from UCI repository: {str(e)}")
        
        # Provide information about the expected dataset structure
        st.info("""
        **German Credit Risk Dataset Structure:**
        
        The German Credit dataset should contain the following features:
        - checking_status: Status of existing checking account
        - duration: Duration in months
        - credit_history: Credit history
        - purpose: Purpose of the loan
        - credit_amount: Credit amount
        - savings_status: Savings account/bonds status
        - employment: Present employment since
        - installment_commitment: Installment rate as percentage of disposable income
        - personal_status: Personal status and sex
        - other_parties: Other debtors/guarantors
        - residence_since: Present residence since
        - property_magnitude: Property
        - age: Age in years
        - other_payment_plans: Other installment plans
        - housing: Housing
        - existing_credits: Number of existing credits
        - job: Job classification
        - num_dependents: Number of people being liable to provide maintenance
        - own_telephone: Telephone
        - foreign_worker: Foreign worker
        - target: Credit risk (0=good, 1=bad)
        
        Please upload a CSV file with these columns or download the dataset from:
        https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
        """)
        
        return None

def load_lending_club_sample():
    """
    Load a sample Lending Club dataset structure.
    This provides the expected structure for a Lending Club dataset.
    """
    
    st.info("""
    **Lending Club Dataset Structure:**
    
    A typical Lending Club dataset should contain features such as:
    
    **Loan Information:**
    - loan_amnt: The listed amount of the loan
    - funded_amnt: The total amount committed to that loan
    - term: The number of payments on the loan (36 or 60 months)
    - int_rate: Interest Rate on the loan
    - installment: The monthly payment owed by the borrower
    - grade: LC assigned loan grade
    - sub_grade: LC assigned loan subgrade
    - purpose: A category provided by the borrower for the loan request
    
    **Borrower Information:**
    - annual_inc: The self-reported annual income provided by the borrower
    - emp_length: Employment length in years
    - home_ownership: The home ownership status provided by the borrower
    - verification_status: Indicates if income was verified
    - dti: Debt-to-income ratio
    
    **Credit Information:**
    - fico_range_low: The lower boundary range the borrower's FICO belongs to
    - fico_range_high: The upper boundary range the borrower's FICO belongs to
    - open_acc: The number of open credit lines in the borrower's credit file
    - pub_rec: Number of derogatory public records
    - revol_bal: Total credit revolving balance
    - revol_util: Revolving line utilization rate
    - total_acc: The total number of credit lines currently in the borrower's credit file
    
    **Geographic Information:**
    - addr_state: The state provided by the borrower in the loan application
    - zip_code: The first 3 numbers of the zip code provided by the borrower
    
    **Target Variable:**
    - loan_status: Current status of the loan (Fully Paid, Charged Off, etc.)
    
    **Expected Data Sources:**
    - Download from Kaggle: "Lending Club Loan Data"
    - Official Lending Club historical data
    - Financial institution loan datasets with similar structure
    
    Please upload a CSV file with similar loan features for analysis.
    """)
    
    return None

def validate_credit_dataset(df, dataset_type="general"):
    """
    Validate that the uploaded dataset has the expected structure for credit risk analysis.
    
    Parameters:
    df: pandas DataFrame - the dataset to validate
    dataset_type: str - type of dataset ("german", "lending_club", "general")
    
    Returns:
    dict: validation results with success status and messages
    """
    
    validation_results = {
        'is_valid': True,
        'messages': [],
        'warnings': [],
        'suggestions': []
    }
    
    # Basic validation
    if df is None or df.empty:
        validation_results['is_valid'] = False
        validation_results['messages'].append("Dataset is empty or None")
        return validation_results
    
    # Check minimum number of rows
    if len(df) < 100:
        validation_results['warnings'].append(f"Dataset has only {len(df)} rows. Recommend at least 1000 rows for reliable model training.")
    
    # Check for potential target variables
    target_candidates = ['target', 'default', 'bad_loan', 'loan_status', 'class', 'y']
    found_targets = [col for col in target_candidates if col in df.columns]
    
    if not found_targets:
        validation_results['warnings'].append(
            f"No obvious target variable found. Expected one of: {target_candidates}. "
            "You may need to specify the target variable manually."
        )
    
    # Check for numeric features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_features) < 3:
        validation_results['warnings'].append(
            f"Only {len(numeric_features)} numeric features found. "
            "Credit risk models typically require multiple numeric features."
        )
    
    # Check for missing values
    missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    if missing_percentage > 50:
        validation_results['warnings'].append(
            f"Dataset has {missing_percentage:.1f}% missing values. "
            "High missing data may affect model performance."
        )
    
    # Dataset-specific validations
    if dataset_type == "german":
        expected_features = ['checking_status', 'duration', 'credit_amount', 'age', 'target']
        missing_features = [f for f in expected_features if f not in df.columns]
        if missing_features:
            validation_results['warnings'].append(
                f"Missing expected German Credit features: {missing_features}"
            )
    
    elif dataset_type == "lending_club":
        expected_features = ['loan_amnt', 'annual_inc', 'dti', 'loan_status']
        missing_features = [f for f in expected_features if f not in df.columns]
        if missing_features:
            validation_results['warnings'].append(
                f"Missing expected Lending Club features: {missing_features}"
            )
    
    # Check for common credit risk features
    common_features = [
        'amount', 'income', 'age', 'duration', 'employment', 'credit', 'debt',
        'payment', 'balance', 'history', 'score', 'rate', 'term'
    ]
    
    found_features = []
    for feature_pattern in common_features:
        matching_cols = [col for col in df.columns if feature_pattern in col.lower()]
        found_features.extend(matching_cols)
    
    if len(found_features) < 5:
        validation_results['suggestions'].append(
            "Consider including more relevant credit risk features such as: "
            "loan amount, annual income, credit score, employment length, debt-to-income ratio, "
            "payment history, account balances, etc."
        )
    
    # Check data types
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    if len(object_cols) > len(df.columns) * 0.7:
        validation_results['suggestions'].append(
            f"Dataset has {len(object_cols)} categorical features out of {len(df.columns)} total. "
            "Consider if some should be converted to numeric features."
        )
    
    validation_results['messages'].append(f"Dataset validated: {len(df)} rows, {len(df.columns)} columns")
    
    return validation_results

def get_dataset_info():
    """
    Provide information about available datasets and how to obtain them.
    """
    
    dataset_info = {
        'german_credit': {
            'name': 'German Credit Risk Dataset',
            'description': 'Classic credit scoring dataset with 1000 instances and 20 attributes',
            'source': 'UCI Machine Learning Repository',
            'url': 'https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)',
            'size': '1,000 records',
            'features': 20,
            'target': 'Binary classification (good/bad credit)',
            'use_case': 'Ideal for learning credit risk modeling fundamentals'
        },
        
        'lending_club': {
            'name': 'Lending Club Loan Data',
            'description': 'Comprehensive loan data with detailed borrower and loan characteristics',
            'source': 'Kaggle / Lending Club Historical Data',
            'url': 'https://www.kaggle.com/datasets/wordsforthewise/lending-club',
            'size': '2+ million records (varies by year)',
            'features': '100+ features',
            'target': 'Loan status (Fully Paid, Charged Off, etc.)',
            'use_case': 'Real-world loan performance prediction'
        },
        
        'custom_requirements': {
            'minimum_features': [
                'Loan amount or credit amount',
                'Borrower income or annual income',
                'Age or employment length',
                'Credit history or credit score',
                'Target variable (default/non-default)'
            ],
            'recommended_features': [
                'Debt-to-income ratio',
                'Employment status',
                'Loan purpose',
                'Interest rate',
                'Loan term',
                'Property ownership',
                'Number of dependents',
                'Previous defaults'
            ],
            'data_format': 'CSV file with headers',
            'encoding': 'UTF-8 recommended',
            'missing_values': 'Clearly marked (NaN, NULL, empty)'
        }
    }
    
    return dataset_info

def create_sample_data_template():
    """
    Create a template showing the expected structure for credit risk data.
    This is for demonstration of structure only, not actual data.
    """
    
    template_structure = {
        'Loan Information': [
            'loan_amount', 'loan_term', 'interest_rate', 'monthly_payment',
            'loan_purpose', 'loan_grade'
        ],
        'Borrower Demographics': [
            'age', 'employment_length', 'annual_income', 'home_ownership',
            'marital_status', 'number_of_dependents'
        ],
        'Credit Information': [
            'credit_score', 'credit_history_length', 'number_of_open_accounts',
            'total_debt', 'debt_to_income_ratio', 'credit_utilization',
            'number_of_inquiries', 'delinquencies_last_2_years'
        ],
        'Geographic': [
            'state', 'zip_code', 'region'
        ],
        'Target Variable': [
            'default_status'  # 0 = No Default, 1 = Default
        ]
    }
    
    return template_structure

def suggest_feature_engineering():
    """
    Provide suggestions for feature engineering in credit risk datasets.
    """
    
    suggestions = {
        'financial_ratios': [
            'debt_to_income_ratio = total_debt / annual_income',
            'loan_to_income_ratio = loan_amount / annual_income',
            'payment_to_income_ratio = monthly_payment / (annual_income / 12)',
            'credit_utilization = current_balance / credit_limit'
        ],
        
        'derived_features': [
            'age_groups = categorize age into groups (young, middle, senior)',
            'income_levels = categorize income into levels (low, medium, high)',
            'employment_stability = employment_length > 2 years',
            'high_risk_purpose = loan_purpose in high-risk categories'
        ],
        
        'interaction_features': [
            'age_income_interaction = age * log(annual_income)',
            'credit_score_utilization = credit_score * (1 - credit_utilization)',
            'loan_term_amount = loan_term * log(loan_amount)'
        ],
        
        'temporal_features': [
            'years_since_first_credit = current_year - first_credit_year',
            'recent_inquiry_flag = inquiries_last_6_months > 0',
            'recent_delinquency = delinquencies_last_2_years > 0'
        ]
    }
    
    return suggestions

# Utility function to detect dataset type
def detect_dataset_type(df):
    """
    Attempt to detect the type of credit dataset based on column names.
    """
    
    column_names = [col.lower() for col in df.columns]
    
    # German Credit indicators
    german_indicators = ['checking_status', 'credit_history', 'foreign_worker', 'property_magnitude']
    german_score = sum(1 for indicator in german_indicators if any(indicator in col for col in column_names))
    
    # Lending Club indicators
    lending_club_indicators = ['loan_amnt', 'funded_amnt', 'grade', 'sub_grade', 'emp_title', 'fico_range']
    lending_club_score = sum(1 for indicator in lending_club_indicators if any(indicator in col for col in column_names))
    
    if german_score >= 2:
        return 'german'
    elif lending_club_score >= 2:
        return 'lending_club'
    else:
        return 'general'

