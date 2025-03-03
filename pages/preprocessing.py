# File: pycaret-ml-app/pages/preprocessing.py
# Page for data preprocessing functionality

import streamlit as st
import pandas as pd
from src.preprocessing import handle_missing_values, encode_categorical, split_data, feature_scaling, drop_columns, preprocess_pipeline
from config.settings import PREPROCESSING_METHODS, DEFAULT_SPLIT_RATIO, RANDOM_STATE

def show():
    """Display the preprocessing page"""
    st.header("Data Preprocessing")
    
    # Check if data is loaded
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please upload a dataset first.")
        if st.button("Go to Data Upload"):
            st.session_state.page = "Data Upload"
            st.rerun()
        return
    
    # Check if target column is selected
    if 'target_column' not in st.session_state:
        st.warning("Please select a target column first.")
        if st.button("Go to Data Exploration"):
            st.session_state.page = "Data Exploration"
            st.rerun()
        return
    
    df = st.session_state.data
    target_column = st.session_state.target_column
    
    # Initialize preprocessing steps
    if 'preprocessing_steps' not in st.session_state:
        st.session_state.preprocessing_steps = {}
    
    # Column selection/dropping
    st.subheader("Column Selection")
    
    all_columns = df.columns.tolist()
    columns_to_keep = st.multiselect(
        "Select columns to keep in the analysis (target column will be kept automatically)",
        all_columns,
        default=[col for col in all_columns if col != target_column]
    )
    
    # Always include target column
    if target_column not in columns_to_keep:
        columns_to_keep.append(target_column)
    
    # Update preprocessing step
    columns_to_drop = [col for col in all_columns if col not in columns_to_keep]
    st.session_state.preprocessing_steps['drop_columns'] = columns_to_drop
    
    # Missing value handling
    st.subheader("Missing Value Handling")
    
    # Get columns with missing values
    missing_cols = [col for col in columns_to_keep if df[col].isnull().sum() > 0]
    
    if missing_cols:
        st.write(f"Columns with missing values: {', '.join(missing_cols)}")
        
        missing_method = st.selectbox(
            "Select method for handling missing values",
            PREPROCESSING_METHODS['missing_values'],
            index=0
        )
        
        selected_missing_cols = st.multiselect(
            "Select columns to apply missing value handling (leave empty for all columns with missing values)",
            missing_cols,
            default=missing_cols
        )
        
        fill_value = None
        if missing_method == 'constant':
            fill_value = st.text_input("Constant value to fill missing values:", "0")
        
        # Update preprocessing step
        st.session_state.preprocessing_steps['missing_values'] = {
            'method': missing_method,
            'columns': selected_missing_cols if selected_missing_cols else None,
            'fill_value': fill_value
        }
    else:
        st.info("No missing values found in the selected columns.")
    
    # Categorical encoding
    st.subheader("Categorical Variable Encoding")
    
    # Get categorical columns
    categorical_cols = [col for col in columns_to_keep if df[col].dtype == 'object' or df[col].dtype == 'category']
    
    if categorical_cols:
        st.write(f"Categorical columns: {', '.join(categorical_cols)}")
        
        encoding_method = st.selectbox(
            "Select encoding method for categorical variables",
            PREPROCESSING_METHODS['encoding_methods'],
            index=0
        )
        
        selected_cat_cols = st.multiselect(
            "Select columns to encode (leave empty for all categorical columns)",
            categorical_cols,
            default=categorical_cols
        )
        
        # Update preprocessing step
        st.session_state.preprocessing_steps['categorical_encoding'] = {
            'method': encoding_method,
            'columns': selected_cat_cols if selected_cat_cols else None
        }
    else:
        st.info("No categorical columns found in the selected columns.")
    
    # Feature scaling
    st.subheader("Feature Scaling")
    
    # Get numeric columns
    numeric_cols = [col for col in columns_to_keep if col != target_column and 
                   (df[col].dtype == 'int64' or df[col].dtype == 'float64')]
    
    if numeric_cols:
        st.write(f"Numeric columns: {', '.join(numeric_cols)}")
        
        scaling_method = st.selectbox(
            "Select scaling method for numeric features",
            PREPROCESSING_METHODS['scaling_methods'],
            index=0
        )
        
        selected_num_cols = st.multiselect(
            "Select columns to scale (leave empty for all numeric columns)",
            numeric_cols,
            default=numeric_cols
        )
        
        # Update preprocessing step
        st.session_state.preprocessing_steps['feature_scaling'] = {
            'method': scaling_method,
            'columns': selected_num_cols if selected_num_cols else None
        }
    else:
        st.info("No numeric columns found in the selected columns.")
    
    # Train-test split
    st.subheader("Train-Test Split")
    
    test_size = st.slider(
        "Select test size (percentage of data to use for testing)",
        min_value=10, max_value=50, value=int(DEFAULT_SPLIT_RATIO * 100), step=5
    ) / 100
    
    st.session_state.preprocessing_steps['test_size'] = test_size
    
    # Process data button
    if st.button("Process Data"):
        with st.spinner("Processing data..."):
            try:
                # Apply preprocessing pipeline
                processed_df, preprocessing_metadata = preprocess_pipeline(df, st.session_state.preprocessing_steps)
                
                # Split data
                X_train, X_test, y_train, y_test = split_data(
                    processed_df, target_column, test_size=test_size, random_state=RANDOM_STATE
                )
                
                if X_train is not None:
                    # Store processed data and metadata in session state
                    st.session_state.processed_data = {
                        'X_train': X_train,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_test': y_test,
                        'preprocessing_metadata': preprocessing_metadata,
                        'target_column': target_column
                    }
                    
                    # Display success message
                    st.success("Data processed successfully!")
                    
                    # Display shapes
                    st.write(f"Training set: {X_train.shape[0]} rows, {X_train.shape[1]} columns")
                    st.write(f"Testing set: {X_test.shape[0]} rows, {X_test.shape[1]} columns")
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
    
    # Next step button
    if 'processed_data' in st.session_state:
        if st.button("Proceed to Model Training"):
            st.session_state.page = "Model Training"
            st.rerun()