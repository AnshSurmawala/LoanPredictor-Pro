import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.model_results = {}
        self.best_models = {}
        
    def initialize_models(self, random_state=42, handle_imbalance=True):
        """Initialize all available models"""
        
        # Determine class weight strategy
        class_weight = 'balanced' if handle_imbalance else None
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=random_state,
                class_weight=class_weight,
                max_iter=1000
            ),
            
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                class_weight=class_weight
            ),
            
            'XGBoost': xgb.XGBClassifier(
                random_state=random_state,
                eval_metric='logloss'
            )
        }
        
        # Define hyperparameter grids for tuning
        self.param_grids = {
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs'],
                'penalty': ['l1', 'l2']
            },
            
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
    
    def train_single_model(self, model_name, X_train, y_train, hyperparameter_tuning=False, cv_folds=5):
        """Train a single model with optional hyperparameter tuning"""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        model = self.models[model_name]
        
        if hyperparameter_tuning and model_name in self.param_grids:
            print(f"Performing hyperparameter tuning for {model_name}...")
            
            # Use StratifiedKFold for cross-validation
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=self.param_grids[model_name],
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            # Store best model
            best_model = grid_search.best_estimator_
            self.best_models[model_name] = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_
            }
            
            trained_model = best_model
            
        else:
            # Train model without hyperparameter tuning
            model.fit(X_train, y_train)
            trained_model = model
        
        self.trained_models[model_name] = trained_model
        return trained_model
    
    def evaluate_model(self, model_name, model, X_train, y_train, X_test, y_test):
        """Comprehensive model evaluation"""
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            train_proba = model.predict_proba(X_train)[:, 1]
            test_proba = model.predict_proba(X_test)[:, 1]
        else:
            train_proba = train_pred
            test_proba = test_pred
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            
            # Training metrics
            'train_accuracy': accuracy_score(y_train, train_pred),
            'train_precision': precision_score(y_train, train_pred, average='weighted'),
            'train_recall': recall_score(y_train, train_pred, average='weighted'),
            'train_f1': f1_score(y_train, train_pred, average='weighted'),
            'train_auc': roc_auc_score(y_train, train_proba),
            
            # Test metrics
            'test_accuracy': accuracy_score(y_test, test_pred),
            'test_precision': precision_score(y_test, test_pred, average='weighted'),
            'test_recall': recall_score(y_test, test_pred, average='weighted'),
            'test_f1': f1_score(y_test, test_pred, average='weighted'),
            'test_auc': roc_auc_score(y_test, test_proba),
            
            # Overfitting indicators
            'accuracy_diff': accuracy_score(y_train, train_pred) - accuracy_score(y_test, test_pred),
            'auc_diff': roc_auc_score(y_train, train_proba) - roc_auc_score(y_test, test_proba),
        }
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        metrics.update({
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
        })
        
        # Store detailed results
        self.model_results[model_name] = {
            'metrics': metrics,
            'train_predictions': train_pred,
            'test_predictions': test_pred,
            'train_probabilities': train_proba,
            'test_probabilities': test_proba,
            'confusion_matrix': confusion_matrix(y_test, test_pred),
            'classification_report': classification_report(y_test, test_pred, output_dict=True)
        }
        
        return metrics
    
    def train_all_models(self, X_train, y_train, X_test, y_test, 
                        model_names=None, hyperparameter_tuning=False):
        """Train and evaluate all specified models"""
        
        if model_names is None:
            model_names = list(self.models.keys())
        
        results = []
        
        for model_name in model_names:
            print(f"\nTraining {model_name}...")
            
            # Train model
            trained_model = self.train_single_model(
                model_name, X_train, y_train, hyperparameter_tuning
            )
            
            # Evaluate model
            metrics = self.evaluate_model(
                model_name, trained_model, X_train, y_train, X_test, y_test
            )
            
            results.append(metrics)
            
            print(f"Completed {model_name}")
            print(f"  - Test AUC: {metrics['test_auc']:.4f}")
            print(f"  - Test Accuracy: {metrics['test_accuracy']:.4f}")
        
        return pd.DataFrame(results)
    
    def get_feature_importance(self, model_name):
        """Get feature importance for tree-based models"""
        
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        
        # Handle GridSearchCV wrapper
        if hasattr(model, 'best_estimator_'):
            actual_model = model.best_estimator_
        else:
            actual_model = model
        
        if hasattr(actual_model, 'feature_importances_'):
            return actual_model.feature_importances_
        elif hasattr(actual_model, 'coef_'):
            # For logistic regression, use absolute coefficients
            return np.abs(actual_model.coef_[0])
        else:
            return None
    
    def get_model_coefficients(self, model_name):
        """Get model coefficients for interpretable models"""
        
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        
        # Handle GridSearchCV wrapper
        if hasattr(model, 'best_estimator_'):
            actual_model = model.best_estimator_
        else:
            actual_model = model
        
        if hasattr(actual_model, 'coef_'):
            return actual_model.coef_[0]
        else:
            return None
    
    def predict_proba_new_data(self, model_name, X_new):
        """Make probability predictions on new data"""
        
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X_new)
        else:
            # For models without predict_proba, return predictions as probabilities
            predictions = model.predict(X_new)
            return np.column_stack([1 - predictions, predictions])
    
    def predict_new_data(self, model_name, X_new):
        """Make predictions on new data"""
        
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        return model.predict(X_new)
    
    def get_roc_curve_data(self, model_name):
        """Get ROC curve data for plotting"""
        
        if model_name not in self.model_results:
            raise ValueError(f"Model {model_name} not evaluated yet")
        
        results = self.model_results[model_name]
        y_test = results.get('y_test')
        test_proba = results['test_probabilities']
        
        if y_test is not None:
            fpr, tpr, thresholds = roc_curve(y_test, test_proba)
            return fpr, tpr, thresholds
        else:
            return None, None, None
    
    def get_precision_recall_curve_data(self, model_name):
        """Get Precision-Recall curve data for plotting"""
        
        if model_name not in self.model_results:
            raise ValueError(f"Model {model_name} not evaluated yet")
        
        results = self.model_results[model_name]
        y_test = results.get('y_test')
        test_proba = results['test_probabilities']
        
        if y_test is not None:
            precision, recall, thresholds = precision_recall_curve(y_test, test_proba)
            return precision, recall, thresholds
        else:
            return None, None, None
    
    def compare_models(self):
        """Compare all trained models"""
        
        if not self.model_results:
            raise ValueError("No models have been evaluated yet")
        
        comparison_data = []
        
        for model_name, results in self.model_results.items():
            metrics = results['metrics']
            comparison_data.append({
                'Model': model_name,
                'Test AUC': metrics['test_auc'],
                'Test Accuracy': metrics['test_accuracy'],
                'Test Precision': metrics['test_precision'],
                'Test Recall': metrics['test_recall'],
                'Test F1': metrics['test_f1'],
                'CV AUC (mean)': metrics['cv_auc_mean'],
                'CV AUC (std)': metrics['cv_auc_std'],
                'Overfitting (AUC)': metrics['auc_diff']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Add rankings
        for metric in ['Test AUC', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1']:
            comparison_df[f'{metric}_Rank'] = comparison_df[metric].rank(ascending=False, method='min')
        
        return comparison_df
    
    def get_best_model(self, metric='test_auc'):
        """Get the best model based on specified metric"""
        
        if not self.model_results:
            raise ValueError("No models have been evaluated yet")
        
        best_score = -np.inf
        best_model_name = None
        
        for model_name, results in self.model_results.items():
            score = results['metrics'].get(metric)
            if score and score > best_score:
                best_score = score
                best_model_name = model_name
        
        if best_model_name:
            return {
                'model_name': best_model_name,
                'model': self.trained_models[best_model_name],
                'score': best_score,
                'metric': metric
            }
        else:
            return None
