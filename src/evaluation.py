# File: pycaret-ml-app/src/evaluation.py
# Functions for model evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pycaret.classification import plot_model as clf_plot_model, pull as clf_pull
from pycaret.regression import plot_model as reg_plot_model, pull as reg_pull
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config.settings import CLASSIFICATION_METRICS, REGRESSION_METRICS, FIG_SIZE, COLOR_PALETTE

def get_metrics(model, X_test, y_test, task_type):
    """
    Get performance metrics for the model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        task_type: Task type (classification/regression)
        
    Returns:
        dict: Dictionary of performance metrics
    """
    try:
        # Get predictions
        y_pred = model.predict(X_test)
        
        metrics = {}
        
        if task_type == "classification":
            # Classification metrics
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
            
            metrics['Accuracy'] = accuracy_score(y_test, y_pred)
            
            # For multi-class, use macro average
            if len(np.unique(y_test)) > 2:
                metrics['F1'] = f1_score(y_test, y_pred, average='macro')
                metrics['Precision'] = precision_score(y_test, y_pred, average='macro')
                metrics['Recall'] = recall_score(y_test, y_pred, average='macro')
                
                # Try to calculate AUC for multi-class if possible
                try:
                    # Get probability predictions
                    y_pred_proba = model.predict_proba(X_test)
                    metrics['AUC'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                except:
                    metrics['AUC'] = None
            else:
                # Binary classification
                metrics['F1'] = f1_score(y_test, y_pred)
                metrics['Precision'] = precision_score(y_test, y_pred)
                metrics['Recall'] = recall_score(y_test, y_pred)
                
                try:
                    # Get probability predictions for class 1
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    metrics['AUC'] = roc_auc_score(y_test, y_pred_proba)
                except:
                    metrics['AUC'] = None
        
        else:  # regression
            # Regression metrics
            metrics['MAE'] = mean_absolute_error(y_test, y_pred)
            metrics['MSE'] = mean_squared_error(y_test, y_pred)
            metrics['RMSE'] = np.sqrt(metrics['MSE'])
            metrics['R2'] = r2_score(y_test, y_pred)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            try:
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                metrics['MAPE'] = mape
            except:
                metrics['MAPE'] = None
        
        return metrics
        
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return {}

def plot_confusion_matrix(model, X_test, y_test):
    """
    Plot confusion matrix for classification models
    
    Args:
        model: Trained classification model
        X_test: Test features
        y_test: Test target
        
    Returns:
        matplotlib.figure.Figure: Confusion matrix plot
    """
    try:
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        
        # Set labels
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        # Return the figure
        return fig
        
    except Exception as e:
        st.error(f"Error plotting confusion matrix: {str(e)}")
        return None

def plot_feature_importance(model, X):
    """
    Plot feature importance
    
    Args:
        model: Trained model
        X: Features dataframe
        
    Returns:
        matplotlib.figure.Figure: Feature importance plot
    """
    try:
        # Check if model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        # Check if model has coef_ attribute (linear models)
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
            if importances.ndim > 1:
                importances = importances.mean(axis=0)
        else:
            st.warning("Model doesn't support feature importance extraction")
            return None
        
        # Create dataframe for feature importances
        feature_names = X.columns
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        
        # Plot horizontal bar chart
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette=COLOR_PALETTE, ax=ax)
        
        # Set labels
        ax.set_title('Feature Importance')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        
        # Return the figure
        return fig
        
    except Exception as e:
        st.error(f"Error plotting feature importance: {str(e)}")
        return None

def plot_learning_curve(model, X_train, y_train, task_type):
    """
    Plot learning curve
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training target
        task_type: Task type (classification/regression)
        
    Returns:
        matplotlib.figure.Figure: Learning curve plot
    """
    try:
        from sklearn.model_selection import learning_curve
        
        # Define scoring metric based on task type
        scoring = 'accuracy' if task_type == 'classification' else 'r2'
        
        # Calculate learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, 
            scoring=scoring,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=5
        )
        
        # Calculate mean and standard deviation
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        
        # Plot learning curve
        ax.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
        ax.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
        
        # Plot standard deviation
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
        ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
        
        # Set labels
        ax.set_title('Learning Curve')
        ax.set_xlabel('Training Examples')
        ax.set_ylabel(f'Score ({scoring})')
        ax.legend(loc='best')
        ax.grid(True)
        
        # Return the figure
        return fig
        
    except Exception as e:
        st.error(f"Error plotting learning curve: {str(e)}")
        return None

def plot_prediction_error(y_test, y_pred, task_type):
    """
    Plot prediction error
    
    Args:
        y_test: Actual values
        y_pred: Predicted values
        task_type: Task type (classification/regression)
        
    Returns:
        matplotlib.figure.Figure: Prediction error plot
    """
    try:
        if task_type != 'regression':
            st.warning("Prediction error plot is only available for regression tasks")
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        
        # Create dataframe for plotting
        error_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Error': y_test - y_pred
        })
        
        # Plot scatter with regression line
        sns.regplot(x='Actual', y='Predicted', data=error_df, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ax=ax)
        
        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect prediction')
        
        # Set labels
        ax.set_title('Actual vs Predicted')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.legend()
        
        # Return the figure
        return fig
        
    except Exception as e:
        st.error(f"Error plotting prediction error: {str(e)}")
        return None

def plot_residuals(y_test, y_pred):
    """
    Plot residuals for regression models
    
    Args:
        y_test: Actual values
        y_pred: Predicted values
        
    Returns:
        matplotlib.figure.Figure: Residuals plot
    """
    try:
        # Calculate residuals
        residuals = y_test - y_pred
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_SIZE[0]*2, FIG_SIZE[1]))
        
        # Plot residuals vs predicted values
        ax1.scatter(y_pred, residuals, alpha=0.5)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_title('Residuals vs Predicted Values')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        
        # Plot residuals distribution
        sns.histplot(residuals, kde=True, ax=ax2)
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_title('Residuals Distribution')
        ax2.set_xlabel('Residuals')
        
        # Adjust layout
        plt.tight_layout()
        
        # Return the figure
        return fig
        
    except Exception as e:
        st.error(f"Error plotting residuals: {str(e)}")
        return None

def get_classification_report(y_test, y_pred):
    """
    Get classification report as a dataframe
    
    Args:
        y_test: Actual values
        y_pred: Predicted values
        
    Returns:
        pandas.DataFrame: Classification report as a dataframe
    """
    try:
        # Get classification report as text
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Convert to dataframe
        report_df = pd.DataFrame(report).transpose()
        
        return report_df
        
    except Exception as e:
        st.error(f"Error generating classification report: {str(e)}")
        return None