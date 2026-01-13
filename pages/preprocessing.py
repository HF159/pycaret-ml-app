# File: pycaret-ml-app/pages/preprocessing.py
# Page for data preprocessing functionality - Simplified with automatic preprocessing

import streamlit as st
import pandas as pd
from src.preprocessing import auto_preprocess_data, split_data
from config.settings import DEFAULT_SPLIT_RATIO, RANDOM_STATE

def show():
    """Display the preprocessing page"""
    st.header("⚙️ Data Preprocessing")
    st.info("Step 3: Prepare your data for model training")

    # Check if data is loaded
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first.")
        if st.button("⬅️ Go to Data Upload"):
            st.session_state.page = "Data Upload"
            st.rerun()
        return

    # Check if target column is selected
    if 'target_column' not in st.session_state:
        st.warning("⚠️ Please select a target column first.")
        if st.button("⬅️ Go to Data Exploration"):
            st.session_state.page = "Data Exploration"
            st.rerun()
        return

    df = st.session_state.data
    target_column = st.session_state.target_column

    # Display current dataset info
    st.subheader("📊 Current Dataset")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Target Column", target_column)

    # Show what preprocessing will do
    st.subheader("🔧 Automatic Preprocessing")
    st.write("The following operations will be performed automatically:")

    preprocessing_steps = []

    # Check for rows with too many missing values
    missing_per_row = df.isnull().sum(axis=1)
    total_columns = len(df.columns)
    rows_with_many_missing = (missing_per_row / total_columns > 0.5).sum()

    if rows_with_many_missing > 0:
        preprocessing_steps.append(f"✓ Drop {rows_with_many_missing} row(s) with >50% missing values")
    else:
        preprocessing_steps.append("✓ No rows need dropping (all have <50% missing)")

    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        preprocessing_steps.append(f"✓ Handle {missing_count} missing values (numeric: median, categorical: mode)")
    else:
        preprocessing_steps.append("✓ No missing values found")

    # Check for categorical columns
    categorical_cols = [col for col in df.columns if col != target_column and
                       (df[col].dtype == 'object' or df[col].dtype == 'category')]
    if categorical_cols:
        preprocessing_steps.append(f"✓ Encode {len(categorical_cols)} categorical column(s) using Label Encoding")
    else:
        preprocessing_steps.append("✓ No categorical encoding needed")

    # Check for numeric columns
    numeric_cols = [col for col in df.columns if col != target_column and
                   (df[col].dtype == 'int64' or df[col].dtype == 'float64')]
    if numeric_cols:
        preprocessing_steps.append(f"✓ No feature scaling (PyCaret will handle it)")

    preprocessing_steps.append("✓ Split data into training and testing sets")

    for step in preprocessing_steps:
        st.write(step)

    # Manual column dropping option
    st.markdown("---")
    st.subheader("🗑️ Drop Columns (Optional)")
    st.write("Select columns you want to remove before preprocessing:")

    # Get all columns except target
    available_columns = [col for col in df.columns if col != target_column]

    # Create multiselect for column dropping
    columns_to_drop = st.multiselect(
        "Select columns to drop:",
        available_columns,
        default=[],
        help="These columns will be removed before training. Useful for removing ID columns or irrelevant features."
    )

    if columns_to_drop:
        st.info(f"✅ Will drop {len(columns_to_drop)} column(s): {', '.join(columns_to_drop)}")
    else:
        st.info("💡 No columns selected for dropping. All columns will be used (except target).")

    # Train-test split ratio
    st.markdown("---")
    st.subheader("📊 Train-Test Split Configuration")

    test_size = st.slider(
        "Test set size (percentage of data):",
        min_value=10,
        max_value=40,
        value=int(DEFAULT_SPLIT_RATIO * 100),
        step=5,
        help="Percentage of data to reserve for testing"
    ) / 100

    # Show split details
    total_rows = len(df)
    train_rows = int(total_rows * (1 - test_size))
    test_rows = total_rows - train_rows

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Rows", train_rows, f"{(1-test_size)*100:.0f}%")
    with col2:
        st.metric("Testing Rows", test_rows, f"{test_size*100:.0f}%")

    # Warning for very small datasets
    if test_rows < 10:
        st.warning(f"⚠️ Warning: Test set will have only {test_rows} rows. Consider using a smaller test size for small datasets.")

    # Process data button
    st.markdown("---")
    if st.button("🚀 Start Preprocessing", use_container_width=True, type="primary"):
        with st.spinner("Processing data... Please wait"):
            try:
                # Drop user-selected columns first
                df_to_process = df.copy()
                if columns_to_drop:
                    df_to_process = df_to_process.drop(columns=columns_to_drop)
                    st.info(f"Dropped {len(columns_to_drop)} column(s) as requested.")

                # Apply automatic preprocessing
                processed_df, preprocessing_metadata = auto_preprocess_data(df_to_process, target_column)

                if processed_df is None or processed_df.empty:
                    st.error("❌ Preprocessing failed. Please check your data.")
                    return

                # Split data
                X_train, X_test, y_train, y_test = split_data(
                    processed_df, target_column, test_size=test_size, random_state=RANDOM_STATE
                )

                if X_train is None:
                    st.error("❌ Data splitting failed. Please check your data.")
                    return

                # Store processed data and metadata in session state
                st.session_state.processed_data = {
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'preprocessing_metadata': preprocessing_metadata,
                    'target_column': target_column
                }

                # Store target encoder if it exists
                if 'target_encoder' in preprocessing_metadata:
                    st.session_state.target_encoder = preprocessing_metadata['target_encoder']

                # Display success message
                st.success("✅ Data processed successfully!")

                # Display results
                st.subheader("✅ Preprocessing Complete")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Training Samples", X_train.shape[0])
                with col2:
                    st.metric("Testing Samples", X_test.shape[0])
                with col3:
                    st.metric("Features", X_train.shape[1])

                # Show preprocessing summary
                with st.expander("📋 View Preprocessing Details"):
                    st.write("**Applied Operations:**")
                    if 'rows_dropped_missing' in preprocessing_metadata and preprocessing_metadata['rows_dropped_missing'] > 0:
                        st.write(f"- Rows dropped (>50% missing values): {preprocessing_metadata['rows_dropped_missing']}")
                    if 'target_encoded' in preprocessing_metadata and preprocessing_metadata['target_encoded']:
                        st.write(f"- Target column '{target_column}' encoded for classification")
                    if 'missing_values_handled' in preprocessing_metadata:
                        st.write(f"- Missing values filled: {preprocessing_metadata['missing_values_handled']} column(s)")
                    if 'categorical_encoded' in preprocessing_metadata:
                        st.write(f"- Feature columns encoded: {preprocessing_metadata['categorical_encoded']}")
                    if 'columns_dropped' in preprocessing_metadata and preprocessing_metadata['columns_dropped']:
                        st.write(f"- Zero-variance columns dropped: {preprocessing_metadata['columns_dropped']}")

                # Show preview of processed data
                with st.expander("👀 View Processed Data Sample"):
                    st.write("**Training Features (X_train):**")
                    st.dataframe(X_train.head(), use_container_width=True)
                    st.write("**Training Target (y_train):**")
                    st.write(y_train.head())

            except Exception as e:
                st.error(f"❌ Error during preprocessing: {str(e)}")
                st.info("💡 Tip: Check if your data has any unusual values or if all values in a column are identical.")
                import traceback
                with st.expander("🔍 View Error Details"):
                    st.code(traceback.format_exc())

    # Next step button
    if 'processed_data' in st.session_state:
        st.markdown("---")
        st.success("✅ Data is ready for model training!")
        if st.button("➡️ Proceed to Model Training", use_container_width=True, type="primary"):
            st.session_state.page = "Model Training"
            st.rerun()
