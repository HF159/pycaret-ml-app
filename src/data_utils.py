# File: pycaret-ml-app/src/data_utils.py
# Utilities for data loading and basic operations

import pandas as pd
import numpy as np
import streamlit as st
import json
import os
from sklearn.datasets import load_iris, load_diabetes
from config.settings import SUPPORTED_FORMATS, SAMPLE_DATASETS, DATA_SAVE_DIR
try:
    # For scikit-learn >= 1.2.0
    from sklearn.datasets import fetch_california_housing
except ImportError:
    # Fallback option
    pass

def load_data(uploaded_file):
    """
    Load data from uploaded file
    
    Args:
        uploaded_file: StreamlitUploadedFile object
        
    Returns:
        pandas.DataFrame: Loaded data or None if loading fails
    """
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension == 'xlsx':
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        elif file_extension == 'json':
            df = pd.read_json(uploaded_file)
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def load_sample_dataset(dataset_name):
    """
    Load a sample dataset
    
    Args:
        dataset_name: Name of the sample dataset
        
    Returns:
        pandas.DataFrame: Sample dataset
    """
    if dataset_name == "Iris":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['target_name'] = [data.target_names[i] for i in data.target]
        return df
    elif dataset_name == "Boston Housing":
        try:
            # For scikit-learn >= 1.2.0
            data = fetch_california_housing()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['MEDV'] = data.target
            return df
        except:
            st.error("California Housing dataset is not available in your scikit-learn version")
            return None
    elif dataset_name == "Diabetes":
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    else:
        st.error(f"Unknown sample dataset: {dataset_name}")
        return None

def get_data_info(df):
    """
    Get basic information about the dataset
    
    Args:
        df: pandas.DataFrame
        
    Returns:
        dict: Information about the dataset
    """
    if df is None:
        return None
        
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'total_missing': df.isnull().sum().sum(),
        'numeric_columns': list(df.select_dtypes(include=['number']).columns),
        'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
        'datetime_columns': list(df.select_dtypes(include=['datetime']).columns),
        'summary_stats': df.describe().to_dict()
    }
    
    return info

def detect_task_type(df, target_column):
    """
    Detect if the task is classification or regression
    
    Args:
        df: pandas.DataFrame
        target_column: Name of the target column
        
    Returns:
        str: "classification" or "regression"
    """
    if target_column not in df.columns:
        st.error(f"Target column '{target_column}' not found in dataframe")
        return None
        
    # Get the data type of the target column
    dtype = df[target_column].dtype
    
    # Check the number of unique values
    n_unique = df[target_column].nunique()
    
    # If categorical or few unique values, likely classification
    if dtype == 'object' or dtype == 'category' or (dtype == 'int64' and n_unique <= 10):
        return "classification"
    # If numeric with many unique values, likely regression
    else:
        return "regression"

def save_dataset(df, filename, directory=DATA_SAVE_DIR):
    """
    Save dataset to disk
    
    Args:
        df: pandas.DataFrame
        filename: Name of the file to save
        directory: Directory to save the file in
        
    Returns:
        bool: Success status
    """
    try:
        # Ensure directory exists
        os.makedirs(directory, exist_ok=True)
        
        # Add .csv extension if not present
        if not filename.endswith('.csv'):
            filename += '.csv'
            
        # Save the dataframe
        filepath = os.path.join(directory, filename)
        df.to_csv(filepath, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving dataset: {str(e)}")
        return False

def get_column_types(df):
    """
    Categorize columns by their data types
    
    Args:
        df: pandas.DataFrame
        
    Returns:
        dict: Dictionary with column categories
    """
    column_types = {
        'numeric': list(df.select_dtypes(include=['number']).columns),
        'categorical': list(df.select_dtypes(include=['object', 'category']).columns),
        'datetime': list(df.select_dtypes(include=['datetime']).columns),
        'all': list(df.columns)
    }
    return column_types

def get_suitable_target_columns(df):
    """
    Get columns that might be suitable as target variables
    
    Args:
        df: pandas.DataFrame
        
    Returns:
        list: List of potential target columns
    """
    # Numeric columns with more than one unique value
    numeric_targets = [col for col in df.select_dtypes(include=['number']).columns 
                      if df[col].nunique() > 1]
    
    # Categorical columns with more than one and fewer than 100 unique values
    categorical_targets = [col for col in df.select_dtypes(include=['object', 'category']).columns 
                          if 1 < df[col].nunique() < 100]
    
    return numeric_targets + categorical_targets