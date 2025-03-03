# File: pycaret-ml-app/pages/data_exploration.py
# Page for data exploration functionality

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_utils import get_data_info, get_suitable_target_columns
from config.settings import FIG_SIZE, COLOR_PALETTE, SAMPLE_ROWS
def analyze_target_distribution(df, target_column):
    """
    Analyze and display the distribution of target values, 
    with warnings for imbalanced classes
    
    Args:
        df: pandas.DataFrame
        target_column: Name of the target column
    """
    if target_column not in df.columns:
        st.warning(f"Target column '{target_column}' not found in dataframe")
        return
    
    # Get value counts of target
    value_counts = df[target_column].value_counts()
    
    # Display value counts
    st.write("### Target Class Distribution")
    st.write(value_counts)
    
    # Check for imbalanced classes
    min_class_count = value_counts.min()
    if min_class_count < 2:
        st.error(f"⚠️ Warning: The smallest class has only {min_class_count} instance(s). Classification models require at least 2 samples per class.")
        st.info("Consider the following options to address this issue:")
        st.info("1. Choose a different target column with better class distribution")
        st.info("2. Filter out rare classes (those with too few samples)")
        st.info("3. Use data augmentation techniques to generate more samples for rare classes")
        st.info("4. Switch to regression if the target is continuous or ordinal")
    elif min_class_count < 5:
        st.warning(f"⚠️ Warning: The smallest class has only {min_class_count} instances. This may lead to poor model performance.")
        st.info("Consider using techniques like SMOTE for oversampling the minority class.")

def show():
    """Display the data exploration page"""
    st.header("Data Exploration")
    
    # Check if data is loaded
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("Please upload a dataset first.")
        if st.button("Go to Data Upload"):
            st.session_state.page = "Data Upload"
            st.rerun()
        return
    
    df = st.session_state.data
    
    # Display basic information
    st.subheader("Dataset Overview")
    data_info = get_data_info(df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Rows: {data_info['shape'][0]}")
        st.write(f"Columns: {data_info['shape'][1]}")
        st.write(f"Missing values: {data_info['total_missing']}")
    
    with col2:
        st.write(f"Numeric columns: {len(data_info['numeric_columns'])}")
        st.write(f"Categorical columns: {len(data_info['categorical_columns'])}")
        st.write(f"Datetime columns: {len(data_info['datetime_columns'])}")
    
    # Display data types
    st.subheader("Data Types")
    dtypes_df = pd.DataFrame({
        'Column': data_info['dtypes'].keys(),
        'Type': [str(t) for t in data_info['dtypes'].values()]
    })
    st.dataframe(dtypes_df)
    
    # Display data sample
    st.subheader("Data Sample")
    st.dataframe(df.head(SAMPLE_ROWS))
    
    # Data visualization
    st.subheader("Data Visualization")
    
    viz_type = st.selectbox(
        "Select visualization type",
        ["Distribution", "Correlation Matrix", "Box Plot"]
    )
    
    if viz_type == "Distribution":
        numeric_cols = data_info['numeric_columns']
        if numeric_cols:
            selected_col = st.selectbox("Select column for histogram", numeric_cols)
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            sns.histplot(df[selected_col].dropna(), kde=True, ax=ax)
            st.pyplot(fig)
        else:
            st.info("No numeric columns available for histogram visualization.")
    
    elif viz_type == "Correlation Matrix":
        numeric_df = df[data_info['numeric_columns']]
        if len(numeric_df.columns) > 1:
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            corr_matrix = numeric_df.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.info("Need at least two numeric columns for correlation matrix.")
    
    elif viz_type == "Box Plot":
        numeric_cols = data_info['numeric_columns']
        if numeric_cols:
            selected_col = st.selectbox("Select column for box plot", numeric_cols)
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            sns.boxplot(y=df[selected_col], ax=ax)
            st.pyplot(fig)
        else:
            st.info("No numeric columns available for box plot visualization.")
    
    # Target variable selection
    st.subheader("Target Variable Selection")
    
    suitable_targets = get_suitable_target_columns(df)
    if suitable_targets:
        target_col = st.selectbox(
            "Select target variable for prediction",
            suitable_targets
        )
        
        # Store target column in session state
        st.session_state.target_column = target_col
        
        # Show distribution of target variable
        st.write(f"Distribution of target variable: {target_col}")
        
        if target_col in data_info['numeric_columns']:
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            sns.histplot(df[target_col].dropna(), kde=True, ax=ax)
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            value_counts = df[target_col].value_counts()
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
    else:
        st.warning("No suitable target columns found.")
    
    # Next step button
    if 'target_column' in st.session_state:
        if st.button("Proceed to Preprocessing"):
            st.session_state.page = "Preprocessing"
            st.rerun()