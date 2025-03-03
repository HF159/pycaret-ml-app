# File: pycaret-ml-app/pages/model_training.py
# Page for model training functionality

import streamlit as st
import pandas as pd
import numpy as np
from src.model_training import initialize_setup, train_models, create_specific_model, tune_model, get_available_models
from config.settings import TASK_TYPES, CLASSIFICATION_METRICS, REGRESSION_METRICS

def show():
    """Display the model training page"""
    st.header("Model Training")
    
    # Check if processed data is available
    if 'processed_data' not in st.session_state:
        st.warning("Please preprocess your data first.")
        if st.button("Go to Preprocessing"):
            st.session_state.page = "Preprocessing"
            st.rerun()
        return
    
    # Get processed data
    processed_data = st.session_state.processed_data
    X_train = processed_data['X_train']
    y_train = processed_data['y_train']
    target_column = processed_data['target_column']
    
    # Task type selection/detection
    st.subheader("Task Type")
    
    # Detect task type based on target variable
    y_unique_count = len(np.unique(y_train))
    suggested_task_type = "classification" if y_unique_count <= 10 else "regression"
    
    task_type = st.selectbox(
        "Select the type of machine learning task",
        TASK_TYPES,
        index=TASK_TYPES.index(suggested_task_type)
    )
    
    # Store task type in session state
    st.session_state.task_type = task_type
    
    # -------- NEW SECTION FOR CLASS BALANCING --------
    
    # Check for imbalanced classes if classification is selected
    if task_type == "classification":
        st.subheader("Class Balance Check")
        
        # Get class distribution
        class_counts = pd.Series(y_train).value_counts()
        
        # Display class distribution in a dataframe for better visualization
        class_df = pd.DataFrame({
            'Class': class_counts.index,
            'Count': class_counts.values
        })
        st.write("Target class distribution:")
        st.dataframe(class_df)
        
        # Identify classes with fewer than 2 samples
        min_class_count = class_counts.min()
        rare_classes = class_counts[class_counts < 2].index.tolist()
        
        if min_class_count < 2:
            st.error(f"⚠️ The smallest class has only {min_class_count} instance(s)!")
            st.warning("PyCaret requires at least 2 samples per class for classification.")
            
            st.subheader("Fix Class Imbalance")
            imbalance_fix = st.radio(
                "Choose how to handle imbalanced classes:",
                ["Remove minority classes", "Duplicate minority class samples", "Switch to regression"]
            )
            
            if imbalance_fix == "Remove minority classes":
                # Find the problematic classes (those with < 2 samples)
                st.write(f"Classes to be removed: {rare_classes}")
                
                if st.button("Apply class filtering"):
                    # Create filtered copies of the data
                    X_train_df = pd.DataFrame(X_train)
                    y_train_series = pd.Series(y_train)
                    
                    # Filter out rare classes
                    mask = ~y_train_series.isin(rare_classes)
                    X_train_fixed = X_train_df[mask].reset_index(drop=True)
                    y_train_fixed = y_train_series[mask].reset_index(drop=True)
                    
                    # Update the data
                    processed_data['X_train'] = X_train_fixed
                    processed_data['y_train'] = y_train_fixed
                    st.session_state.processed_data = processed_data
                    
                    # Show new distribution
                    new_counts = pd.Series(y_train_fixed).value_counts()
                    new_class_df = pd.DataFrame({
                        'Class': new_counts.index,
                        'Count': new_counts.values
                    })
                    
                    st.success("Minority classes removed successfully!")
                    st.write("New class distribution:")
                    st.dataframe(new_class_df)
                    
                    # Refresh data variables after update
                    X_train = processed_data['X_train']
                    y_train = processed_data['y_train']
            
            elif imbalance_fix == "Duplicate minority class samples":
                # Find the problematic classes
                st.write(f"Classes to be duplicated: {rare_classes}")
                
                if st.button("Apply duplication"):
                    # Convert to DataFrame/Series if they're not already
                    X_train_df = pd.DataFrame(X_train)
                    y_train_series = pd.Series(y_train)
                    
                    # Duplicate samples from rare classes
                    for cls in rare_classes:
                        # Find samples of this class
                        cls_indices = y_train_series[y_train_series == cls].index.tolist()
                        
                        if len(cls_indices) > 0:
                            # Number of duplicates needed
                            duplicates_needed = 2 - len(cls_indices)
                            
                            # Duplicate the samples
                            for _ in range(duplicates_needed):
                                # Choose one sample to duplicate 
                                idx = cls_indices[0]
                                
                                # Add duplicated sample
                                sample_X = X_train_df.loc[[idx]].copy()
                                X_train_df = pd.concat([X_train_df, sample_X], ignore_index=True)
                                y_train_series = pd.concat([y_train_series, pd.Series([cls])], ignore_index=True)
                    
                    # Update the data
                    processed_data['X_train'] = X_train_df
                    processed_data['y_train'] = y_train_series
                    st.session_state.processed_data = processed_data
                    
                    # Show new distribution
                    new_counts = pd.Series(y_train_series).value_counts()
                    new_class_df = pd.DataFrame({
                        'Class': new_counts.index,
                        'Count': new_counts.values
                    })
                    
                    st.success("Minority classes duplicated successfully!")
                    st.write("New class distribution:")
                    st.dataframe(new_class_df)
                    
                    # Refresh data variables after update
                    X_train = processed_data['X_train']
                    y_train = processed_data['y_train']
            
            elif imbalance_fix == "Switch to regression":
                st.info("If your target variable is ordinal (e.g., ratings 1-5), regression might be more appropriate.")
                if st.button("Switch to regression"):
                    # Update task type
                    task_type = "regression"
                    st.session_state.task_type = "regression"
                    st.rerun()
    
    # -------- END OF NEW SECTION --------
    
    # Display appropriate metrics for the selected task type
    metrics = CLASSIFICATION_METRICS if task_type == "classification" else REGRESSION_METRICS
    st.write(f"Optimization metrics for {task_type}: {', '.join(metrics)}")
    
    # Check if we need to reinitialize setup due to task type change
    reinitialize_needed = False
    if 'setup_initialized_for' not in st.session_state or st.session_state.setup_initialized_for != task_type:
        reinitialize_needed = True
        if 'setup_initialized' in st.session_state:
            del st.session_state.setup_initialized
    
    # Initialize PyCaret setup
    setup_button_label = "Reinitialize Setup" if reinitialize_needed else "Initialize Setup"
    if st.button(setup_button_label):
        with st.spinner("Initializing PyCaret setup..."):
            # Check for class imbalance again before setup
            if task_type == "classification":
                class_counts = pd.Series(y_train).value_counts()
                min_class_count = class_counts.min()
                
                if min_class_count < 2:
                    st.error("Cannot initialize setup: Some classes still have fewer than 2 samples.")
                    st.warning("Please use the class balancing options above before initializing.")
                    st.stop()
            
            # Create a combined dataframe for setup
            train_df = X_train.copy()
            train_df[target_column] = y_train
            
            # Initialize setup
            setup = initialize_setup(train_df, target_column, task_type)
            
            if setup is not None:
                st.session_state.setup_initialized = True
                st.session_state.setup_initialized_for = task_type
                st.success(f"PyCaret setup initialized successfully for {task_type}!")
    
    # Only show model selection after setup is initialized
    if 'setup_initialized' in st.session_state and st.session_state.setup_initialized and st.session_state.setup_initialized_for == task_type:
        # Model selection
        st.subheader("Model Selection")
        
        # Option to compare multiple models or select specific model
        model_selection_method = st.radio(
            "Model selection method",
            ["Compare multiple models", "Select specific model"]
        )
        
        if model_selection_method == "Compare multiple models":
            # Number of models to compare
            n_models = st.slider(
                "Number of top models to compare",
                min_value=1, max_value=10, value=3, step=1
            )
            
            # Metric to sort by
            sort_metric = st.selectbox(
                "Select metric to optimize",
                metrics,
                index=0
            )
            
            # Train models button
            if st.button("Train and Compare Models"):
                with st.spinner(f"Training and comparing {n_models} models..."):
                    # Train models
                    models = train_models(task_type, n_models=n_models, sort_by=sort_metric)
                    
                    if models is not None:
                        # Store models in session state
                        st.session_state.models = models
                        
                        # FIX: Store the first (best) model as the trained model
                        # When PyCaret's compare_models returns multiple models, it returns them as a list
                        # sorted by the specified metric, so the first model is the best one
                        if isinstance(models, list) and len(models) > 0:
                            st.session_state.trained_model = models[0]
                        else:
                            st.session_state.trained_model = models
                            
                        st.success("Models trained and compared successfully!")
                        
                        # Display model comparison results
                        st.write("Model comparison results will be shown in the Evaluation page.")
        
        else:  # Select specific model
            # Get available models
            available_models = get_available_models(task_type)
            
            # Model selection
            selected_model = st.selectbox(
                "Select a model to train",
                available_models
            )
            
            # Train model button
            if st.button("Train Model"):
                with st.spinner(f"Training {selected_model} model..."):
                    # Create specific model
                    model = create_specific_model(selected_model, task_type)
                    
                    if model is not None:
                        # Store model in session state
                        st.session_state.trained_model = model
                        st.success(f"{selected_model} model trained successfully!")
                        
                        # Display model information
                        st.write("Model details will be shown in the Evaluation page.")
            
            # Model tuning (only show if a model has been trained)
            if 'trained_model' in st.session_state:
                st.subheader("Model Tuning")
                
                # Metric to optimize
                tune_metric = st.selectbox(
                    "Select metric to optimize during tuning",
                    metrics,
                    index=0
                )
                
                # Number of tuning iterations
                n_iter = st.slider(
                    "Number of tuning iterations",
                    min_value=5, max_value=50, value=10, step=5
                )
                
                # Tune model button
                if st.button("Tune Model"):
                    with st.spinner(f"Tuning model to optimize {tune_metric}..."):
                        # Tune model
                        tuned_model = tune_model(
                            st.session_state.trained_model, 
                            task_type, 
                            optimization_metric=tune_metric,
                            n_iter=n_iter
                        )
                        
                        if tuned_model is not None:
                            # Store tuned model in session state
                            st.session_state.trained_model = tuned_model
                            st.success("Model tuned successfully!")
    
    # Display warning if setup has not been initialized for the current task type
    elif 'setup_initialized' in st.session_state and st.session_state.setup_initialized and st.session_state.setup_initialized_for != task_type:
        st.warning(f"You've changed the task type from {st.session_state.setup_initialized_for} to {task_type}. Please reinitialize setup for the new task type.")
    
    # Next step button
    if 'trained_model' in st.session_state and ('setup_initialized_for' not in st.session_state or st.session_state.setup_initialized_for == task_type):
        if st.button("Proceed to Evaluation"):
            st.session_state.page = "Evaluation"
            st.rerun()