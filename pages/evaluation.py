# File: pycaret-ml-app/pages/evaluation.py
# Page for model evaluation functionality

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.evaluation import get_metrics, plot_confusion_matrix, plot_feature_importance, plot_learning_curve, plot_prediction_error, plot_residuals, get_classification_report

def show():
    """Display the evaluation page"""
    st.header("Model Evaluation")
    
    # Check if trained model is available
    if 'trained_model' not in st.session_state:
        st.warning("Please train a model first.")
        if st.button("Go to Model Training"):
            st.session_state.page = "Model Training"
            st.rerun()
        return
    
    # Get data from session state
    processed_data = st.session_state.processed_data
    X_test = processed_data['X_test']
    y_test = processed_data['y_test']
    target_column = processed_data['target_column']
    task_type = st.session_state.task_type
    model = st.session_state.trained_model
    
    # Performance metrics
    st.subheader("Performance Metrics")
    
    metrics = get_metrics(model, X_test, y_test, task_type)
    if metrics:
        # Create a dataframe for better display
        metrics_df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': [f"{v:.4f}" if isinstance(v, (int, float)) else str(v) for v in metrics.values()]
        })
        st.dataframe(metrics_df)
    
    # Confusion matrix (for classification)
    if task_type == 'classification':
        st.subheader("Confusion Matrix")
        
        cm_fig = plot_confusion_matrix(model, X_test, y_test)
        if cm_fig:
            st.pyplot(cm_fig)
            
        # Classification report
        st.subheader("Classification Report")
        y_pred = model.predict(X_test)
        report = get_classification_report(y_test, y_pred)
        if report is not None:
            st.dataframe(report.style.format(precision=3))
    
    # Prediction error (for regression)
    if task_type == 'regression':
        st.subheader("Prediction Error")
        
        y_pred = model.predict(X_test)
        error_fig = plot_prediction_error(y_test, y_pred, task_type)
        if error_fig:
            st.pyplot(error_fig)
            
        # Residuals plot
        st.subheader("Residuals Analysis")
        
        residuals_fig = plot_residuals(y_test, y_pred)
        if residuals_fig:
            st.pyplot(residuals_fig)
    
    # Feature importance
    st.subheader("Feature Importance")
    
    importance_fig = plot_feature_importance(model, X_test)
    if importance_fig:
        st.pyplot(importance_fig)
    
    # Learning curve
    st.subheader("Learning Curve")
    
    X_train = processed_data['X_train']
    y_train = processed_data['y_train']
    learning_fig = plot_learning_curve(model, X_train, y_train, task_type)
    if learning_fig:
        st.pyplot(learning_fig)
    
    # Save model
    st.subheader("Save Model")
    
    model_name = st.text_input("Enter a name for your model:", f"{task_type}_model")
    
    if st.button("Save Model"):
        from src.model_training import save_model
        
        success = save_model(model, model_name, task_type)
        if success:
            st.session_state.saved_model_name = model_name
            st.success(f"Model saved as '{model_name}' successfully!")
    
    # Next step button
    if st.button("Proceed to Prediction"):
        st.session_state.page = "Prediction"
        st.rerun()