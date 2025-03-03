# File: pycaret-ml-app/src/prediction.py
# Functions for making predictions

import os
import pandas as pd
import numpy as np
import streamlit as st
from pycaret.classification import predict_model as clf_predict_model, load_model as clf_load_model
from pycaret.regression import predict_model as reg_predict_model, load_model as reg_load_model
from config.settings import TASK_TYPES, MODEL_SAVE_DIR

def predict_model(model, data, task_type):
    """
    Make predictions using trained model
    
    Args:
        model: Trained model
        data: Data to make predictions on
        task_type: Task type (classification/regression)
        
    Returns:
        pandas.DataFrame: Dataframe with predictions
    """
    try:
        if task_type not in TASK_TYPES:
            st.error(f"Invalid task type: {task_type}. Must be one of {TASK_TYPES}")
            return None
            
        with st.spinner("Making predictions..."):
            if task_type == "classification":
                predictions = clf_predict_model(model, data=data)
            else:  # regression
                predictions = reg_predict_model(model, data=data)
                
            st.success("Predictions completed successfully")
            return predictions
            
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return None

def load_model(model_path, task_type):
    """
    Load a trained model from disk
    
    Args:
        model_path: Path to saved model
        task_type: Task type (classification/regression)
        
    Returns:
        object: Loaded model
    """
    try:
        if task_type not in TASK_TYPES:
            st.error(f"Invalid task type: {task_type}. Must be one of {TASK_TYPES}")
            return None
            
        with st.spinner(f"Loading model from {model_path}..."):
            if task_type == "classification":
                model = clf_load_model(model_path)
            else:  # regression
                model = reg_load_model(model_path)
                
            st.success(f"Model loaded successfully from {model_path}")
            return model
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_input_data(input_data, preprocessing_pipeline):
    """
    Process input data for prediction
    
    Args:
        input_data: Raw input data (DataFrame)
        preprocessing_pipeline: Preprocessing pipeline metadata
        
    Returns:
        pandas.DataFrame: Processed data ready for prediction
    """
    try:
        # Make a copy of the input data
        processed_data = input_data.copy()
        
        # Apply preprocessing steps in the correct order
        
        # 1. Drop columns if specified
        if 'dropped_columns' in preprocessing_pipeline:
            columns_to_drop = [col for col in preprocessing_pipeline['dropped_columns'] 
                              if col in processed_data.columns]
            if columns_to_drop:
                processed_data = processed_data.drop(columns=columns_to_drop)
        
        # 2. Handle missing values
        if 'missing_values' in preprocessing_pipeline:
            method = preprocessing_pipeline['missing_values']['method']
            columns = preprocessing_pipeline['missing_values'].get('columns')
            fill_value = preprocessing_pipeline['missing_values'].get('fill_value')
            
            # Apply to specified columns or all columns with missing values
            if columns is None:
                columns = processed_data.columns[processed_data.isnull().any()].tolist()
                
            for col in columns:
                if col not in processed_data.columns:
                    continue
                    
                if method == 'mean' and pd.api.types.is_numeric_dtype(processed_data[col]):
                    processed_data[col] = processed_data[col].fillna(processed_data[col].mean())
                elif method == 'median' and pd.api.types.is_numeric_dtype(processed_data[col]):
                    processed_data[col] = processed_data[col].fillna(processed_data[col].median())
                elif method == 'mode':
                    processed_data[col] = processed_data[col].fillna(processed_data[col].mode()[0])
                elif method == 'constant':
                    processed_data[col] = processed_data[col].fillna(fill_value)
        
        # 3. Encode categorical variables
        if 'encoders' in preprocessing_pipeline and 'categorical_encoding' in preprocessing_pipeline:
            method = preprocessing_pipeline['categorical_encoding']['method']
            encoders = preprocessing_pipeline['encoders']
            
            if method == 'label':
                for col, encoder in encoders.items():
                    if col in processed_data.columns:
                        processed_data[col] = encoder.transform(processed_data[col].astype(str))
            elif method == 'one-hot':
                for col, column_names in encoders.items():
                    if col in processed_data.columns:
                        # Create one-hot encoded columns
                        dummies = pd.get_dummies(processed_data[col], prefix=col, drop_first=False)
                        
                        # Ensure all expected columns exist, fill with 0 if not
                        for dummy_col in column_names:
                            if dummy_col not in dummies.columns:
                                dummies[dummy_col] = 0
                                
                        # Keep only the expected columns in the correct order
                        dummies = dummies[column_names]
                        
                        # Add dummies to the dataframe and drop original column
                        processed_data = pd.concat([processed_data, dummies], axis=1)
                        processed_data = processed_data.drop(col, axis=1)
        
        # 4. Scale features
        if 'scaler' in preprocessing_pipeline and preprocessing_pipeline['scaler'] is not None:
            scaler = preprocessing_pipeline['scaler']
            columns = preprocessing_pipeline['feature_scaling'].get('columns')
            
            if columns is None:
                columns = processed_data.select_dtypes(include=['number']).columns.tolist()
                
            # Ensure all columns exist in the dataframe
            columns = [col for col in columns if col in processed_data.columns]
            
            if columns:
                processed_data[columns] = scaler.transform(processed_data[columns])
        
        return processed_data
        
    except Exception as e:
        st.error(f"Error processing input data: {str(e)}")
        return None

def interpret_predictions(predictions, task_type):
    """
    Interpret and format predictions
    
    Args:
        predictions: Model predictions DataFrame
        task_type: Task type (classification/regression)
        
    Returns:
        pandas.DataFrame: Formatted predictions with interpretations
    """
    try:
        if predictions is None:
            return None
            
        # Make a copy of the predictions
        interpreted = predictions.copy()
        
        if task_type == "classification":
            # Rename prediction column to something more user-friendly
            # PyCaret typically uses 'Label' for classification predictions
            if 'Label' in interpreted.columns:
                interpreted = interpreted.rename(columns={'Label': 'Predicted_Class'})
                
            # If probability columns exist, create a 'Confidence' column with the highest probability
            prob_cols = [col for col in interpreted.columns if col.startswith('Score_')]
            if prob_cols:
                interpreted['Confidence'] = interpreted[prob_cols].max(axis=1).round(4)
                
        else:  # regression
            # Rename prediction column to something more user-friendly
            # PyCaret typically uses 'prediction_label' for regression predictions
            if 'prediction_label' in interpreted.columns:
                interpreted = interpreted.rename(columns={'prediction_label': 'Predicted_Value'})
                
        return interpreted
        
    except Exception as e:
        st.error(f"Error interpreting predictions: {str(e)}")
        return predictions  # Return original predictions if error occurs

def generate_input_form(df, target_column=None):
    """
    Generate a form for manual input data
    
    Args:
        df: Original DataFrame to base the form on
        target_column: Target column to exclude from the form
        
    Returns:
        dict: Dictionary of input values
    """
    try:
        st.subheader("Enter values for prediction")
        
        # Create a dictionary to store input values
        input_data = {}
        
        # Get column types
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        datetime_cols = df.select_dtypes(include=['datetime']).columns
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        # Create input fields for each feature
        for i, col in enumerate(df.columns):
            # Skip target column
            if col == target_column:
                continue
                
            # Alternate between columns for better layout
            input_col = col1 if i % 2 == 0 else col2
            
            # Create appropriate input field based on column type
            with input_col:
                if col in numeric_cols:
                    # For numeric columns, use number input
                    default_val = float(df[col].mean()) if not pd.isna(df[col].mean()) else 0.0
                    input_data[col] = st.number_input(
                        f"{col}:", 
                        value=default_val,
                        format="%.4f" if df[col].dtype == 'float64' else "%d"
                    )
                elif col in categorical_cols:
                    # For categorical columns, use selectbox
                    options = df[col].dropna().unique().tolist()
                    default_val = options[0] if options else ""
                    input_data[col] = st.selectbox(f"{col}:", options, index=0)
                elif col in datetime_cols:
                    # For datetime columns, use date input
                    default_date = pd.Timestamp.now().date()
                    input_data[col] = st.date_input(f"{col}:", default_date)
                else:
                    # For any other type, use text input
                    input_data[col] = st.text_input(f"{col}:", "")
        
        # Create a DataFrame from the input data
        if input_data:
            input_df = pd.DataFrame([input_data])
            return input_df
        else:
            return None
            
    except Exception as e:
        st.error(f"Error generating input form: {str(e)}")
        return None

def export_predictions(predictions, filename="predictions.csv"):
    """
    Export predictions to a CSV file
    
    Args:
        predictions: Predictions DataFrame
        filename: Name of the file to save
        
    Returns:
        bytes: CSV file as bytes for download
    """
    try:
        # Convert DataFrame to CSV
        csv = predictions.to_csv(index=False)
        
        # Return as bytes for Streamlit download button
        return csv.encode('utf-8')
        
    except Exception as e:
        st.error(f"Error exporting predictions: {str(e)}")
        return None