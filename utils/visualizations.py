import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

def create_correlation_heatmap(corr_matrix, title="Feature Correlation Matrix"):
    """Create an interactive correlation heatmap using Plotly"""
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Features",
        yaxis_title="Features",
        width=800,
        height=800
    )
    
    return fig

def create_distribution_plots(data, features, target_column=None):
    """Create distribution plots for multiple features"""
    
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"Distribution of {feature}" for feature in features]
    )
    
    for i, feature in enumerate(features):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        if target_column and target_column in data.columns:
            # Create separate distributions for each target class
            for target_value in data[target_column].unique():
                subset = data[data[target_column] == target_value]
                
                fig.add_trace(
                    go.Histogram(
                        x=subset[feature],
                        name=f"{target_column}={target_value}",
                        opacity=0.7,
                        legendgroup=f"group{target_value}",
                        showlegend=(i == 0)  # Only show legend for first subplot
                    ),
                    row=row, col=col
                )
        else:
            # Single distribution
            fig.add_trace(
                go.Histogram(x=data[feature], showlegend=False),
                row=row, col=col
            )
    
    fig.update_layout(
        height=300 * n_rows,
        title="Feature Distributions",
        barmode='overlay'
    )
    
    return fig

def create_box_plots(data, features, target_column=None):
    """Create box plots for feature analysis"""
    
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"Box Plot: {feature}" for feature in features]
    )
    
    for i, feature in enumerate(features):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        if target_column and target_column in data.columns:
            # Separate box plots for each target class
            for target_value in data[target_column].unique():
                subset = data[data[target_column] == target_value]
                
                fig.add_trace(
                    go.Box(
                        y=subset[feature],
                        name=f"{target_column}={target_value}",
                        legendgroup=f"group{target_value}",
                        showlegend=(i == 0)
                    ),
                    row=row, col=col
                )
        else:
            fig.add_trace(
                go.Box(y=data[feature], showlegend=False),
                row=row, col=col
            )
    
    fig.update_layout(
        height=300 * n_rows,
        title="Feature Box Plots"
    )
    
    return fig

def create_roc_curve(y_true, y_prob_dict, title="ROC Curves Comparison"):
    """Create ROC curve plot for multiple models"""
    
    fig = go.Figure()
    
    for model_name, y_prob in y_prob_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = np.trapz(tpr, fpr)
        
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {auc_score:.3f})',
            line=dict(width=3)
        ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random (AUC = 0.500)',
        line=dict(dash='dash', color='red', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        width=600,
        height=500
    )
    
    return fig

def create_precision_recall_curve(y_true, y_prob_dict, title="Precision-Recall Curves"):
    """Create Precision-Recall curve plot for multiple models"""
    
    fig = go.Figure()
    
    for model_name, y_prob in y_prob_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=model_name,
            line=dict(width=3)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Recall',
        yaxis_title='Precision',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        width=600,
        height=500
    )
    
    return fig

def create_confusion_matrix_heatmap(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """Create confusion matrix heatmap"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    if labels is None:
        labels = ['Class 0', 'Class 1']
    
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=labels,
        y=labels,
        annotation_text=cm,
        colorscale='Blues',
        showscale=True
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        width=500,
        height=500
    )
    
    return fig

def create_feature_importance_plot(importance_dict, top_n=15, title="Feature Importance"):
    """Create feature importance plot"""
    
    # Convert to DataFrame and sort
    importance_df = pd.DataFrame(list(importance_dict.items()), 
                                columns=['Feature', 'Importance'])
    importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
    
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title=title
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=max(400, 30 * len(importance_df))
    )
    
    return fig

def create_model_comparison_radar(results_df, metrics=None):
    """Create radar chart for model comparison"""
    
    if metrics is None:
        metrics = ['Test AUC', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1']
    
    fig = go.Figure()
    
    for _, row in results_df.iterrows():
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
        title="Model Performance Comparison"
    )
    
    return fig

def create_prediction_probability_histogram(y_true, y_prob, model_name="Model"):
    """Create histogram of prediction probabilities by true class"""
    
    fig = go.Figure()
    
    # Separate probabilities by actual class
    prob_class_0 = y_prob[y_true == 0]
    prob_class_1 = y_prob[y_true == 1]
    
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
        title=f'Prediction Probability Distribution - {model_name}',
        xaxis_title='Predicted Probability',
        yaxis_title='Count',
        barmode='overlay'
    )
    
    return fig

def create_calibration_plot(y_true, y_prob_dict, n_bins=10, title="Calibration Plot"):
    """Create calibration plot to assess prediction reliability"""
    
    fig = go.Figure()
    
    for model_name, y_prob in y_prob_dict.items():
        # Calculate calibration
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        observed_frequencies = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                observed_frequencies.append(y_true[in_bin].mean())
        
        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=observed_frequencies,
            mode='lines+markers',
            name=model_name,
            line=dict(width=3)
        ))
    
    # Add perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(dash='dash', color='red', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Mean Predicted Probability',
        yaxis_title='Fraction of Positives',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def create_learning_curves(train_sizes, train_scores, val_scores, title="Learning Curves"):
    """Create learning curves plot"""
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig = go.Figure()
    
    # Training scores
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=train_mean,
        mode='lines+markers',
        name='Training Score',
        line=dict(color='blue'),
        error_y=dict(type='data', array=train_std, color='blue', thickness=1)
    ))
    
    # Validation scores
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=val_mean,
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='red'),
        error_y=dict(type='data', array=val_std, color='red', thickness=1)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Training Set Size',
        yaxis_title='Score',
        showlegend=True
    )
    
    return fig

def create_residual_plot(y_true, y_pred, title="Residual Plot"):
    """Create residual plot for regression analysis"""
    
    residuals = y_true - y_pred
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(
            size=8,
            opacity=0.6,
            color=residuals,
            colorscale='RdBu',
            showscale=True,
            colorbar=dict(title="Residual")
        )
    ))
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title=title,
        xaxis_title='Predicted Values',
        yaxis_title='Residuals'
    )
    
    return fig
