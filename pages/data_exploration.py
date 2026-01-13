# File: pycaret-ml-app/pages/data_exploration.py
# Page for data exploration functionality

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_utils import get_data_info, get_suitable_target_columns, detect_task_type
from config.settings import FIG_SIZE, SAMPLE_ROWS

def show():
    """Display the data exploration page"""
    st.header("🔍 Data Exploration")
    st.info("Step 2: Explore your data and select the target variable for prediction")

    # Check if data is loaded
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first.")
        if st.button("⬅️ Go to Data Upload"):
            st.session_state.page = "Data Upload"
            st.rerun()
        return

    df = st.session_state.data

    # Display basic information
    st.subheader("📊 Dataset Summary")
    data_info = get_data_info(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", data_info['shape'][0])
    with col2:
        st.metric("Columns", data_info['shape'][1])
    with col3:
        st.metric("Numeric Columns", len(data_info['numeric_columns']))
    with col4:
        st.metric("Categorical Columns", len(data_info['categorical_columns']))

    # Data preview
    st.subheader("👀 Data Preview")
    st.dataframe(df.head(SAMPLE_ROWS), use_container_width=True)

    # Statistical summary
    with st.expander("📈 View Statistical Summary"):
        st.dataframe(df.describe(), use_container_width=True)

    # Target variable selection
    st.markdown("---")
    st.subheader("🎯 Target Variable Selection")
    st.write("Select the column you want to predict (target variable)")

    suitable_targets = get_suitable_target_columns(df)

    if not suitable_targets:
        st.error("❌ No suitable target columns found. Please ensure your dataset has at least one column with multiple unique values.")
        return

    target_col = st.selectbox(
        "Select target variable for prediction:",
        suitable_targets,
        help="This is the column that the model will learn to predict"
    )

    # Store target column in session state
    st.session_state.target_column = target_col

    # Auto-detect task type
    task_type = detect_task_type(df, target_col)
    if task_type:
        st.session_state.task_type = task_type

        # Show task type with icon
        if task_type == "classification":
            st.success(f"✅ Detected Task Type: **Classification** (Predicting categories)")
        else:
            st.success(f"✅ Detected Task Type: **Regression** (Predicting numeric values)")

    # Show target distribution
    st.subheader(f"📊 Target Variable Distribution: {target_col}")

    try:
        if target_col in data_info['numeric_columns']:
            # For numeric targets (regression)
            col1, col2 = st.columns(2)

            with col1:
                # Statistics
                st.write("**Statistics:**")
                stats_df = pd.DataFrame({
                    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                    'Value': [
                        f"{df[target_col].mean():.2f}",
                        f"{df[target_col].median():.2f}",
                        f"{df[target_col].std():.2f}",
                        f"{df[target_col].min():.2f}",
                        f"{df[target_col].max():.2f}"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)

            with col2:
                # Histogram
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(df[target_col].dropna(), kde=True, ax=ax, color='steelblue')
                ax.set_title(f'Distribution of {target_col}')
                ax.set_xlabel(target_col)
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
                plt.close()
        else:
            # For categorical targets (classification)
            value_counts = df[target_col].value_counts()

            col1, col2 = st.columns(2)

            with col1:
                # Value counts table
                st.write("**Class Distribution:**")
                count_df = pd.DataFrame({
                    'Class': value_counts.index,
                    'Count': value_counts.values,
                    'Percentage': [f"{(v/len(df)*100):.1f}%" for v in value_counts.values]
                })
                st.dataframe(count_df, use_container_width=True, hide_index=True)

                # Check for class imbalance
                min_class_count = value_counts.min()
                if min_class_count < 2:
                    st.error(f"⚠️ Warning: The smallest class has only {min_class_count} instance(s). This may cause errors during model training.")
                    st.info("💡 Tip: Consider collecting more data or removing rare classes.")
                elif min_class_count < 10:
                    st.warning(f"⚠️ Warning: The smallest class has only {min_class_count} instances. This may affect model performance.")

            with col2:
                # Bar chart
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax, palette='viridis')
                ax.set_title(f'Distribution of {target_col}')
                ax.set_xlabel(target_col)
                ax.set_ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    except Exception as e:
        st.error(f"Error displaying target distribution: {str(e)}")

    # Next step button
    if 'target_column' in st.session_state:
        st.markdown("---")
        if st.button("➡️ Proceed to Preprocessing", use_container_width=True, type="primary"):
            st.session_state.page = "Preprocessing"
            st.rerun()
