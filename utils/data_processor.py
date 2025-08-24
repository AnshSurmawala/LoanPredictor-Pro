import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.is_fitted = False
    
    def detect_feature_types(self, df):
        """Detect numeric and categorical features"""
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        return numeric_features, categorical_features
    
    def handle_missing_values(self, df, numeric_features, categorical_features):
        """Handle missing values in the dataset"""
        df_clean = df.copy()
        
        # Handle numeric missing values
        if numeric_features:
            df_clean[numeric_features] = self.numeric_imputer.fit_transform(df_clean[numeric_features])
        
        # Handle categorical missing values
        if categorical_features:
            df_clean[categorical_features] = self.categorical_imputer.fit_transform(df_clean[categorical_features])
        
        return df_clean
    
    def encode_categorical_features(self, df, categorical_features):
        """Encode categorical features using Label Encoding"""
        df_encoded = df.copy()
        
        for feature in categorical_features:
            if feature in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[feature] = le.fit_transform(df_encoded[feature].astype(str))
                self.label_encoders[feature] = le
        
        return df_encoded
    
    def create_financial_ratios(self, df):
        """Create financial ratio features"""
        df_ratios = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Common financial ratios if relevant columns exist
        ratio_mappings = [
            ('income', 'loan_amount', 'loan_to_income_ratio'),
            ('monthly_payment', 'income', 'payment_to_income_ratio'),
            ('loan_amount', 'property_value', 'loan_to_value_ratio'),
            ('debt', 'income', 'debt_to_income_ratio'),
            ('credit_limit', 'balance', 'credit_utilization'),
        ]
        
        for numerator_pattern, denominator_pattern, ratio_name in ratio_mappings:
            # Find columns that match patterns
            numerator_cols = [col for col in numeric_cols if numerator_pattern in col.lower()]
            denominator_cols = [col for col in numeric_cols if denominator_pattern in col.lower()]
            
            if numerator_cols and denominator_cols:
                numerator_col = numerator_cols[0]
                denominator_col = denominator_cols[0]
                
                # Create ratio with safety check for division by zero
                df_ratios[ratio_name] = df_ratios[numerator_col] / (df_ratios[denominator_col] + 1e-6)
                
                # Cap extreme ratios
                df_ratios[ratio_name] = np.clip(df_ratios[ratio_name], 0, 10)
        
        return df_ratios
    
    def detect_outliers(self, df, method='iqr', threshold=1.5):
        """Detect outliers using IQR or Z-score method"""
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_info = {}
        
        for feature in numeric_features:
            if method == 'iqr':
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs((df[feature] - df[feature].mean()) / df[feature].std())
                outliers = df[z_scores > threshold]
            
            outlier_info[feature] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100,
                'indices': outliers.index.tolist()
            }
        
        return outlier_info
    
    def handle_outliers(self, df, method='cap', threshold=1.5):
        """Handle outliers by capping, removing, or transforming"""
        df_clean = df.copy()
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for feature in numeric_features:
            if method == 'cap':
                Q1 = df_clean[feature].quantile(0.25)
                Q3 = df_clean[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                df_clean[feature] = np.clip(df_clean[feature], lower_bound, upper_bound)
            
            elif method == 'log_transform':
                # Apply log transformation for positive values
                if (df_clean[feature] > 0).all():
                    df_clean[feature] = np.log1p(df_clean[feature])
        
        return df_clean
    
    def scale_features(self, df, features_to_scale=None):
        """Scale numeric features"""
        df_scaled = df.copy()
        
        if features_to_scale is None:
            features_to_scale = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if features_to_scale:
            df_scaled[features_to_scale] = self.scaler.fit_transform(df_scaled[features_to_scale])
        
        return df_scaled
    
    def prepare_target_variable(self, target_series):
        """Prepare target variable for binary classification"""
        target_clean = target_series.copy()
        
        # Handle different target formats
        if target_clean.dtype == 'object':
            # Common mappings for credit risk datasets
            risk_mappings = {
                'good': 0, 'bad': 1,
                'low': 0, 'high': 1,
                'no': 0, 'yes': 1,
                'current': 0, 'default': 1,
                'fully paid': 0, 'charged off': 1,
                'approved': 0, 'rejected': 1
            }
            
            # Apply mappings if found
            for original, mapped in risk_mappings.items():
                mask = target_clean.str.lower().str.contains(original, na=False)
                target_clean.loc[mask] = mapped
            
            # If still object type, use label encoding
            if target_clean.dtype == 'object':
                le = LabelEncoder()
                target_clean = le.fit_transform(target_clean)
                self.label_encoders['target'] = le
        
        return target_clean
    
    def get_feature_statistics(self, df):
        """Get comprehensive feature statistics"""
        stats = {}
        
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Numeric feature statistics
        if numeric_features:
            stats['numeric'] = {
                'count': len(numeric_features),
                'features': numeric_features,
                'statistics': df[numeric_features].describe()
            }
        
        # Categorical feature statistics
        if categorical_features:
            cat_stats = {}
            for feature in categorical_features:
                cat_stats[feature] = {
                    'unique_count': df[feature].nunique(),
                    'most_frequent': df[feature].mode().iloc[0] if not df[feature].mode().empty else None,
                    'value_counts': df[feature].value_counts().head(10)
                }
            
            stats['categorical'] = {
                'count': len(categorical_features),
                'features': categorical_features,
                'statistics': cat_stats
            }
        
        # Missing value statistics
        missing_stats = df.isnull().sum()
        stats['missing_values'] = {
            'total_missing': missing_stats.sum(),
            'features_with_missing': missing_stats[missing_stats > 0].to_dict(),
            'missing_percentage': (missing_stats / len(df) * 100).round(2).to_dict()
        }
        
        return stats
    
    def full_preprocessing_pipeline(self, df, target_column=None, 
                                   create_ratios=True, handle_outliers_flag=True,
                                   scale_features_flag=True):
        """Complete preprocessing pipeline"""
        
        # Separate features and target
        if target_column and target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
        else:
            X = df.copy()
            y = None
        
        # Detect feature types
        numeric_features, categorical_features = self.detect_feature_types(X)
        
        # Handle missing values
        X_clean = self.handle_missing_values(X, numeric_features, categorical_features)
        
        # Create financial ratios
        if create_ratios:
            X_clean = self.create_financial_ratios(X_clean)
            # Update feature types after creating ratios
            numeric_features, categorical_features = self.detect_feature_types(X_clean)
        
        # Handle outliers
        if handle_outliers_flag:
            X_clean = self.handle_outliers(X_clean, method='cap')
        
        # Encode categorical features
        X_encoded = self.encode_categorical_features(X_clean, categorical_features)
        
        # Scale features
        if scale_features_flag:
            X_scaled = self.scale_features(X_encoded, numeric_features)
        else:
            X_scaled = X_encoded
        
        # Process target variable
        if y is not None:
            y_clean = self.prepare_target_variable(y)
        else:
            y_clean = None
        
        self.feature_names = X_scaled.columns.tolist()
        self.is_fitted = True
        
        return X_scaled, y_clean
    
    def transform_new_data(self, df):
        """Transform new data using fitted preprocessors"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming new data")
        
        df_transformed = df.copy()
        
        # Handle missing values
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if numeric_features:
            df_transformed[numeric_features] = self.numeric_imputer.transform(df_transformed[numeric_features])
        
        if categorical_features:
            df_transformed[categorical_features] = self.categorical_imputer.transform(df_transformed[categorical_features])
        
        # Encode categorical features
        for feature in categorical_features:
            if feature in self.label_encoders:
                # Handle unknown categories
                known_categories = set(self.label_encoders[feature].classes_)
                df_transformed[feature] = df_transformed[feature].astype(str)
                
                # Replace unknown categories with most frequent
                unknown_mask = ~df_transformed[feature].isin(known_categories)
                if unknown_mask.any():
                    most_frequent = self.label_encoders[feature].classes_[0]
                    df_transformed.loc[unknown_mask, feature] = most_frequent
                
                df_transformed[feature] = self.label_encoders[feature].transform(df_transformed[feature])
        
        # Scale features
        numeric_features_to_scale = [col for col in numeric_features if col in df_transformed.columns]
        if numeric_features_to_scale:
            df_transformed[numeric_features_to_scale] = self.scaler.transform(df_transformed[numeric_features_to_scale])
        
        # Ensure same columns as training data
        for feature in self.feature_names:
            if feature not in df_transformed.columns:
                df_transformed[feature] = 0
        
        return df_transformed[self.feature_names]
