# File: pycaret-ml-app/src/model_training.py
# Functions for model training using PyCaret

import os
import pandas as pd
import numpy as np
import streamlit as st
from pycaret.classification import setup as clf_setup, compare_models as clf_compare
from pycaret.classification import create_model as clf_create_model, tune_model as clf_tune_model
from pycaret.classification import save_model as clf_save_model, load_model as clf_load_model
from pycaret.regression import setup as reg_setup, compare_models as reg_compare
from pycaret.regression import create_model as reg_create_model, tune_model as reg_tune_model
from pycaret.regression import save_model as reg_save_model, load_model as reg_load_model
from config.settings import TASK_TYPES, RANDOM_STATE, MODEL_SAVE_DIR

def initialize_setup(df, target_column, task_type, preprocess=True, normalize=True, session_id=RANDOM_STATE):
    """
    Initialize PyCaret setup
    
    Args:
        df: pandas.DataFrame
        target_column: Target column name
        task_type: Task type (classification/regression)
        preprocess: Whether to preprocess data
        normalize: Whether to normalize data
        session_id: Random seed for reproducibility
        
    Returns:
        object: PyCaret setup object
    """
    try:
        if task_type not in TASK_TYPES:
            st.error(f"Invalid task type: {task_type}. Must be one of {TASK_TYPES}")
            return None
            
        # Check if dataset is very small for classification
        if task_type == "classification":
            class_counts = df[target_column].value_counts()
            min_class_count = class_counts.min()
            
            # For very small datasets, adjust setup parameters
            if min_class_count <= 5:
                st.warning("Small dataset detected. Adjusting parameters for better compatibility.")
                
                # Common setup parameters adjusted for small datasets - using only well-supported parameters
                setup_params = {
                    'data': df,
                    'target': target_column,
                    'preprocess': preprocess,
                    'normalize': normalize,
                    'session_id': session_id,
                    'verbose': False,
                    'html': False,
                    'n_jobs': 1,                 # Use single job to avoid parallel processing issues
                    'fix_imbalance': False,      # Disable automatic imbalance fixing
                    'remove_outliers': False,    # Disable outlier removal to preserve samples
                    'fold': 2                    # Use fewer folds for cross-validation
                }
                
                # If very small dataset, also limit neighbors for KNN-based algorithms
                if min_class_count < 5:
                    # For KNN-based methods, n_neighbors must be <= n_samples in smallest class
                    neighbors = max(1, min_class_count - 1)  # At least 1, at most class size - 1
                    setup_params['n_neighbors'] = neighbors
                    st.info(f"Setting n_neighbors={neighbors} for KNN-based algorithms due to small class size.")
            else:
                # Standard setup parameters for normal-sized datasets
                setup_params = {
                    'data': df,
                    'target': target_column,
                    'preprocess': preprocess,
                    'normalize': normalize,
                    'session_id': session_id,
                    'verbose': False,
                    'html': False
                }
        else:
            # Standard setup parameters for regression
            setup_params = {
                'data': df,
                'target': target_column,
                'preprocess': preprocess,
                'normalize': normalize,
                'session_id': session_id,
                'verbose': False,
                'html': False
            }
        
        # Initialize setup based on task type
        with st.spinner(f"Setting up {task_type} experiment..."):
            if task_type == "classification":
                # Add classification-specific parameters here if needed
                setup = clf_setup(**setup_params)
            else:  # regression
                # Add regression-specific parameters here if needed
                setup = reg_setup(**setup_params)
                
            st.success(f"{task_type.capitalize()} setup completed successfully")
            return setup
            
    except Exception as e:
        st.error(f"Error in PyCaret setup: {str(e)}")
        
        # Give more specific advice based on the error
        if "n_neighbors" in str(e):
            st.info("Your dataset is too small for KNN-based algorithms. Try removing KNN from your model selection.")
        elif "unexpected keyword" in str(e):
            # Handle version compatibility issues
            st.info("You may be using a different version of PyCaret than expected. Trying minimal setup...")
            
            # Try a minimal setup with fewer parameters
            try:
                minimal_params = {
                    'data': df,
                    'target': target_column,
                    'session_id': session_id,
                    'verbose': False,
                    'html': False
                }
                
                if task_type == "classification":
                    setup = clf_setup(**minimal_params)
                else:
                    setup = reg_setup(**minimal_params)
                    
                st.success("Setup completed with minimal parameters.")
                return setup
            except Exception as e2:
                st.error(f"Minimal setup also failed: {str(e2)}")
        
        st.info("Try selecting fewer models or using a different approach for very small datasets.")
        return None
    
def train_models(task_type, n_models=5, sort_by=None):
    """
    Train and compare multiple models
    
    Args:
        task_type: Task type (classification/regression)
        n_models: Number of top models to return
        sort_by: Metric to sort models by
        
    Returns:
        object: Comparison results
    """
    try:
        if task_type not in TASK_TYPES:
            st.error(f"Invalid task type: {task_type}. Must be one of {TASK_TYPES}")
            return None
            
        with st.spinner(f"Training and comparing {n_models} models..."):
            if task_type == "classification":
                # Default sorting metric for classification
                if sort_by is None:
                    sort_by = 'Accuracy'
                models = clf_compare(n_select=n_models, sort=sort_by)
            else:  # regression
                # Default sorting metric for regression
                if sort_by is None:
                    sort_by = 'R2'
                models = reg_compare(n_select=n_models, sort=sort_by)
                
            st.success(f"Model comparison completed. Models sorted by {sort_by}")
            return models
            
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return None

def create_specific_model(model_name, task_type):
    """
    Create a specific model
    
    Args:
        model_name: Name of the model to create
        task_type: Task type (classification/regression)
        
    Returns:
        object: Created model
    """
    try:
        if task_type not in TASK_TYPES:
            st.error(f"Invalid task type: {task_type}. Must be one of {TASK_TYPES}")
            return None
            
        with st.spinner(f"Creating {model_name} model..."):
            if task_type == "classification":
                model = clf_create_model(model_name)
            else:  # regression
                model = reg_create_model(model_name)
                
            st.success(f"{model_name} model created successfully")
            return model
            
    except Exception as e:
        st.error(f"Error creating model: {str(e)}")
        return None

def tune_model(model, task_type, optimization_metric=None, n_iter=10, n_fold=5):
    """
    Tune a model's hyperparameters
    
    Args:
        model: Model to tune
        task_type: Task type (classification/regression)
        optimization_metric: Metric to optimize
        n_iter: Number of iterations for tuning
        n_fold: Number of folds for cross-validation
        
    Returns:
        object: Tuned model
    """
    try:
        if task_type not in TASK_TYPES:
            st.error(f"Invalid task type: {task_type}. Must be one of {TASK_TYPES}")
            return None
            
        # Set default optimization metrics if none provided
        if optimization_metric is None:
            optimization_metric = 'Accuracy' if task_type == 'classification' else 'R2'
            
        with st.spinner(f"Tuning model to optimize {optimization_metric}..."):
            if task_type == "classification":
                tuned_model = clf_tune_model(model, 
                                           optimize=optimization_metric,
                                           n_iter=n_iter,
                                           n_fold=n_fold)
            else:  # regression
                tuned_model = reg_tune_model(model, 
                                           optimize=optimization_metric,
                                           n_iter=n_iter,
                                           n_fold=n_fold)
                
            st.success(f"Model tuning completed. Optimized for {optimization_metric}")
            return tuned_model
            
    except Exception as e:
        st.error(f"Error tuning model: {str(e)}")
        return None

def save_model(model, model_name, task_type, directory=MODEL_SAVE_DIR):
    """
    Save trained model to disk
    
    Args:
        model: Trained model
        model_name: Name to save the model as
        task_type: Task type (classification/regression)
        directory: Directory to save the model in
        
    Returns:
        bool: Success status
    """
    try:
        if task_type not in TASK_TYPES:
            st.error(f"Invalid task type: {task_type}. Must be one of {TASK_TYPES}")
            return False
            
        # Ensure directory exists
        os.makedirs(directory, exist_ok=True)
        
        # Create full path
        model_path = os.path.join(directory, model_name)
        
        with st.spinner(f"Saving model as {model_name}..."):
            if task_type == "classification":
                clf_save_model(model, model_path)
            else:  # regression
                reg_save_model(model, model_path)
                
            st.success(f"Model saved successfully as {model_path}")
            return True
            
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")
        return False

def load_saved_model(model_path, task_type):
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

def get_available_models(task_type):
    """
    Get list of available models for the given task type
    
    Args:
        task_type: Task type (classification/regression)
        
    Returns:
        list: List of available model names
    """
    if task_type == "classification":
        # Common classification models
        return ['lr', 'knn', 'nb', 'dt', 'svm', 'rf', 'xgboost', 'lightgbm', 'catboost']
    else:  # regression
        # Common regression models
        return ['lr', 'lasso', 'ridge', 'en', 'knn', 'dt', 'svm', 'rf', 'xgboost', 'lightgbm', 'catboost']
    
    