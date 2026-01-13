# File: pycaret-ml-app/src/preprocessing.py
# Functions for data preprocessing

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from config.settings import PREPROCESSING_METHODS, DEFAULT_SPLIT_RATIO, RANDOM_STATE

def handle_missing_values(df, method, columns=None, fill_value=None):
    """
    Handle missing values in the dataframe
    
    Args:
        df: pandas.DataFrame
        method: Method to handle missing values (mean, median, mode, drop, constant)
        columns: List of columns to apply the method to (None for all)
        fill_value: Value to use when method is 'constant'
        
    Returns:
        pandas.DataFrame: Processed dataframe
    """
    # Make a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # If no columns specified, use all columns with missing values
    if columns is None:
        columns = [col for col in df.columns if df[col].isnull().sum() > 0]
    
    # If no columns with missing values, return the original dataframe
    if not columns:
        return processed_df
    
    # Handle different methods
    if method == 'drop':
        # Check if dropping would remove all rows
        if processed_df[columns].isnull().any(axis=1).sum() == len(processed_df):
            st.warning("Cannot drop all rows with missing values as it would result in an empty dataset. Falling back to mean imputation.")
            # Fall back to mean imputation
            for col in columns:
                # Skip columns that don't exist or have no missing values
                if col not in processed_df.columns or processed_df[col].isnull().sum() == 0:
                    continue
                    
                if pd.api.types.is_numeric_dtype(processed_df[col]):
                    processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
                else:
                    # For non-numeric columns, use mode
                    if not processed_df[col].empty and processed_df[col].dropna().size > 0:
                        processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
                    else:
                        # If column is empty or all NaN, fill with a placeholder
                        processed_df[col] = processed_df[col].fillna("Unknown")
        else:
            processed_df = processed_df.dropna(subset=columns)
    else:
        for col in columns:
            # Skip columns that don't exist or have no missing values
            if col not in processed_df.columns or processed_df[col].isnull().sum() == 0:
                continue
                
            # Apply the appropriate imputation method
            if method == 'mean' and pd.api.types.is_numeric_dtype(processed_df[col]):
                # Check if column has non-NaN values
                if processed_df[col].dropna().size > 0:
                    processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
                else:
                    # If all values are NaN, this column should have been dropped earlier
                    st.warning(f"Column '{col}' has all NaN values. Consider dropping it.")
                    processed_df[col] = processed_df[col].fillna(processed_df[col].mean())  # Will be NaN
            elif method == 'median' and pd.api.types.is_numeric_dtype(processed_df[col]):
                # Check if column has non-NaN values
                if processed_df[col].dropna().size > 0:
                    processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                else:
                    # If all values are NaN, this column should have been dropped earlier
                    st.warning(f"Column '{col}' has all NaN values. Consider dropping it.")
                    processed_df[col] = processed_df[col].fillna(processed_df[col].median())  # Will be NaN
            elif method == 'mode':
                # Check if column has non-NaN values
                if processed_df[col].dropna().size > 0:
                    processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
                else:
                    # If all values are NaN, this column should have been dropped earlier
                    st.warning(f"Column '{col}' has all NaN values. Consider dropping it.")
                    if pd.api.types.is_numeric_dtype(processed_df[col]):
                        # For numeric, leave as NaN to be caught by validation
                        pass
                    else:
                        processed_df[col] = processed_df[col].fillna("Unknown")
            elif method == 'constant':
                # Convert fill_value to numeric for numeric columns
                if pd.api.types.is_numeric_dtype(processed_df[col]) and fill_value is not None:
                    try:
                        numeric_fill = float(fill_value)
                        processed_df[col] = processed_df[col].fillna(numeric_fill)
                    except ValueError:
                        # If cannot convert to float, use median as fallback
                        if processed_df[col].dropna().size > 0:
                            processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                            st.warning(f"Could not convert '{fill_value}' to number for column '{col}'. Using median instead.")
                        else:
                            st.warning(f"Could not convert '{fill_value}' to number and column '{col}' has no valid data.")
                else:
                    # Use string as is for non-numeric columns
                    processed_df[col] = processed_df[col].fillna(str(fill_value) if fill_value is not None else "Unknown")
            else:
                st.warning(f"Method '{method}' not applicable for column '{col}', using appropriate fallback.")
                # Fall back to appropriate method
                if processed_df[col].dropna().size > 0:
                    if pd.api.types.is_numeric_dtype(processed_df[col]):
                        processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                    else:
                        processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
                else:
                    # If all values are NaN, this column should have been dropped earlier
                    st.warning(f"Column '{col}' has all NaN values and should be dropped.")
    
    return processed_df

def encode_categorical(df, method, columns=None):
    """
    Encode categorical variables
    
    Args:
        df: pandas.DataFrame
        method: Encoding method (label, one-hot, none)
        columns: List of columns to encode (None for all categorical)
        
    Returns:
        tuple: (pandas.DataFrame with encoded categories, dict of encoders)
    """
    # If method is none, return the original dataframe
    if method == 'none':
        return df, {}
        
    # Make a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # If no columns specified, use all categorical columns
    if columns is None:
        columns = list(df.select_dtypes(include=['object', 'category']).columns)
    
    # If no categorical columns to encode, return the original dataframe
    if not columns:
        return processed_df, {}
    
    # Store encoders for later use in transforming new data
    encoders = {}
    
    if method == 'label':
        for col in columns:
            # Skip columns that don't exist
            if col not in processed_df.columns:
                continue
                
            # Handle columns with only one unique value
            if processed_df[col].nunique() <= 1:
                st.warning(f"Column '{col}' has only one unique value. Skipping encoding.")
                continue
                
            # Fill NaN values before encoding (LabelEncoder can't handle NaN)
            processed_df[col] = processed_df[col].fillna("Unknown")
                
            # Create and fit a label encoder
            le = LabelEncoder()
            processed_df[col] = le.fit_transform(processed_df[col].astype(str))
            encoders[col] = le
            
    elif method == 'one-hot':
        for col in columns:
            # Skip columns that don't exist
            if col not in processed_df.columns:
                continue
                
            # Handle columns with only one unique value
            if processed_df[col].nunique() <= 1:
                st.warning(f"Column '{col}' has only one unique value. Skipping encoding.")
                continue
                
            # Fill NaN values before encoding
            processed_df[col] = processed_df[col].fillna("Unknown")
                
            # Create one-hot encoded columns
            dummies = pd.get_dummies(processed_df[col], prefix=col, drop_first=False)
            processed_df = pd.concat([processed_df, dummies], axis=1)
            processed_df = processed_df.drop(col, axis=1)
            encoders[col] = dummies.columns.tolist()  # Store column names rather than encoder
    
    return processed_df, encoders

def feature_scaling(df, method='standard', columns=None):
    """
    Scale features in the dataframe
    
    Args:
        df: pandas.DataFrame
        method: Scaling method (standard, minmax, robust, none)
        columns: List of columns to scale (None for all numeric)
        
    Returns:
        tuple: (pandas.DataFrame with scaled features, scaler object)
    """
    # If method is none, return the original dataframe
    if method == 'none':
        return df, None
    
    # Make a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # If no columns specified, use all numeric columns
    if columns is None:
        columns = list(df.select_dtypes(include=['number']).columns)
    
    # Filter out columns that don't exist
    columns = [col for col in columns if col in processed_df.columns]
    
    # If no numeric columns to scale, return the original dataframe
    if not columns:
        return processed_df, None
    
    # Ensure we have data to scale
    if processed_df.empty or processed_df[columns].dropna().empty:
        st.warning("No data available for scaling. Skipping feature scaling.")
        return processed_df, None
    
    # Select the appropriate scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        st.warning(f"Unknown scaling method: {method}, using StandardScaler")
        scaler = StandardScaler()
    
    try:
        # Scale the selected columns
        # Only select rows without NaN values for fitting
        non_nan_rows = processed_df[columns].dropna().index
        
        if len(non_nan_rows) > 0:
            # Fit on non-NaN data
            scaler.fit(processed_df.loc[non_nan_rows, columns])
            
            # Transform only rows without NaN values
            processed_df.loc[non_nan_rows, columns] = scaler.transform(processed_df.loc[non_nan_rows, columns])
        else:
            st.warning("No complete rows available for scaling. Skipping feature scaling.")
            return processed_df, None
    except Exception as e:
        st.error(f"Error during feature scaling: {str(e)}")
        return processed_df, None
    
    return processed_df, scaler

def split_data(df, target_column, test_size=DEFAULT_SPLIT_RATIO, random_state=RANDOM_STATE):
    """
    Split data into training and testing sets
    
    Args:
        df: pandas.DataFrame
        target_column: Target column name
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if target_column not in df.columns:
        st.error(f"Target column '{target_column}' not found in dataframe")
        return None, None, None, None
    
    # Check if we have enough data to split
    if len(df) < 2:
        st.error("Not enough data for train-test split. Need at least 2 rows.")
        return None, None, None, None
    
    # Split features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Split into training and testing sets
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test
    except Exception as e:
        st.error(f"Error splitting data: {str(e)}")
        return None, None, None, None

def drop_columns(df, columns_to_drop):
    """
    Drop specified columns from the dataframe
    
    Args:
        df: pandas.DataFrame
        columns_to_drop: List of columns to drop
        
    Returns:
        pandas.DataFrame: Dataframe with dropped columns
    """
    # Make a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Drop columns that exist in the dataframe
    columns_to_drop = [col for col in columns_to_drop if col in processed_df.columns]
    if columns_to_drop:
        processed_df = processed_df.drop(columns_to_drop, axis=1)
    
    return processed_df

def auto_preprocess_data(df, target_column):
    """
    Automatically preprocess data with smart defaults

    Args:
        df: pandas.DataFrame
        target_column: Target column name

    Returns:
        tuple: (processed dataframe, preprocessing metadata)
    """
    processed_df = df.copy()
    metadata = {}

    # Check if dataframe is not empty
    if processed_df.empty:
        st.error("Cannot preprocess an empty dataframe")
        return None, metadata

    try:
        # 1. Drop rows with too many missing values (>50% of columns are missing)
        missing_threshold = 0.5
        initial_rows = len(processed_df)

        # Calculate how many missing values each row has
        missing_per_row = processed_df.isnull().sum(axis=1)
        total_columns = len(processed_df.columns)

        # Identify rows where more than 50% of values are missing
        rows_to_keep = missing_per_row / total_columns <= missing_threshold
        processed_df = processed_df[rows_to_keep]

        rows_dropped = initial_rows - len(processed_df)

        if rows_dropped > 0:
            st.warning(f"🗑️ Dropped {rows_dropped} row(s) with >50% missing values ({rows_dropped/initial_rows*100:.1f}% of data)")
            metadata['rows_dropped_missing'] = rows_dropped
        else:
            metadata['rows_dropped_missing'] = 0

        # Check if we still have enough data
        if len(processed_df) < 10:
            st.error("❌ Too few rows remaining after dropping rows with missing values. Please review your data quality.")
            return None, metadata

        # 2. Handle remaining missing values intelligently
        missing_count = processed_df.isnull().sum().sum()
        if missing_count > 0:
            # Get columns with missing values (excluding target)
            missing_cols = [col for col in processed_df.columns
                          if col != target_column and processed_df[col].isnull().sum() > 0]

            # Handle numeric columns with median (more robust than mean)
            numeric_missing = [col for col in missing_cols if pd.api.types.is_numeric_dtype(processed_df[col])]
            if numeric_missing:
                processed_df = handle_missing_values(processed_df, 'median', numeric_missing)

            # Handle categorical columns with mode
            categorical_missing = [col for col in missing_cols if not pd.api.types.is_numeric_dtype(processed_df[col])]
            if categorical_missing:
                processed_df = handle_missing_values(processed_df, 'mode', categorical_missing)

            metadata['missing_values_handled'] = len(missing_cols)
        else:
            metadata['missing_values_handled'] = 0

        # 2. Encode target column if it's categorical (for classification)
        if processed_df[target_column].dtype == 'object' or processed_df[target_column].dtype == 'category':
            st.info(f"🔄 Encoding target column '{target_column}' for classification")
            from sklearn.preprocessing import LabelEncoder
            target_encoder = LabelEncoder()
            processed_df[target_column] = target_encoder.fit_transform(processed_df[target_column])
            metadata['target_encoder'] = target_encoder
            metadata['target_encoded'] = True
        else:
            metadata['target_encoded'] = False

        # 3. Encode categorical feature variables (excluding target)
        categorical_cols = [col for col in processed_df.columns if col != target_column and
                          (processed_df[col].dtype == 'object' or processed_df[col].dtype == 'category')]

        if categorical_cols:
            processed_df, encoders = encode_categorical(processed_df, 'label', categorical_cols)
            metadata['encoders'] = encoders
            metadata['categorical_encoded'] = len(categorical_cols)
        else:
            metadata['categorical_encoded'] = 0

        # 4. Drop columns with zero variance (all same value)
        columns_to_drop = []
        for col in processed_df.columns:
            if col != target_column and processed_df[col].nunique() <= 1:
                columns_to_drop.append(col)

        if columns_to_drop:
            processed_df = processed_df.drop(columns=columns_to_drop)
            metadata['columns_dropped'] = columns_to_drop
        else:
            metadata['columns_dropped'] = []

        # Note: We don't do feature scaling here because PyCaret will handle it

        return processed_df, metadata

    except Exception as e:
        st.error(f"Error in automatic preprocessing: {str(e)}")
        return None, {}

def preprocess_pipeline(df, preprocessing_steps):
    """
    Apply a series of preprocessing steps to the dataframe

    Args:
        df: pandas.DataFrame
        preprocessing_steps: Dict of preprocessing steps to apply

    Returns:
        tuple: (processed dataframe, preprocessing metadata)
    """
    processed_df = df.copy()
    metadata = {}

    # Check if dataframe is not empty
    if processed_df.empty:
        st.error("Cannot preprocess an empty dataframe")
        return processed_df, metadata

    try:
        # Drop columns
        if 'drop_columns' in preprocessing_steps:
            processed_df = drop_columns(processed_df, preprocessing_steps['drop_columns'])
            metadata['dropped_columns'] = preprocessing_steps['drop_columns']

        # Handle missing values
        if 'missing_values' in preprocessing_steps:
            processed_df = handle_missing_values(
                processed_df,
                preprocessing_steps['missing_values']['method'],
                preprocessing_steps['missing_values'].get('columns'),
                preprocessing_steps['missing_values'].get('fill_value')
            )
            metadata['missing_values'] = preprocessing_steps['missing_values']

        # Check if we still have data after handling missing values
        if processed_df.empty:
            st.error("Preprocessing resulted in an empty dataframe")
            return df.copy(), {}

        # Encode categorical variables
        if 'categorical_encoding' in preprocessing_steps:
            processed_df, encoders = encode_categorical(
                processed_df,
                preprocessing_steps['categorical_encoding']['method'],
                preprocessing_steps['categorical_encoding'].get('columns')
            )
            metadata['encoders'] = encoders
            metadata['categorical_encoding'] = preprocessing_steps['categorical_encoding']

        # Scale features
        if 'feature_scaling' in preprocessing_steps:
            processed_df, scaler = feature_scaling(
                processed_df,
                preprocessing_steps['feature_scaling']['method'],
                preprocessing_steps['feature_scaling'].get('columns')
            )
            metadata['scaler'] = scaler
            metadata['feature_scaling'] = preprocessing_steps['feature_scaling']

        return processed_df, metadata
    except Exception as e:
        st.error(f"Error in preprocessing pipeline: {str(e)}")
        return df.copy(), {}