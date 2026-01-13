# File: pycaret-ml-app/pages/data_upload.py
# Page for data upload functionality

import streamlit as st
import pandas as pd
from src.data_utils import load_data, get_data_info, clean_dataframe
from config.settings import SUPPORTED_FORMATS, SAMPLE_ROWS

def show():
    """Display the data upload page"""
    st.header("📁 Data Upload")
    st.info("Step 1: Upload your dataset to begin the machine learning workflow")

    # File upload widget
    st.subheader("Upload Your Dataset")
    st.write("Supported formats: CSV, Excel (.xlsx), JSON")
    uploaded_file = st.file_uploader("Choose a file", type=SUPPORTED_FORMATS)

    if uploaded_file is not None:
        # Load the data
        df = load_data(uploaded_file)

        if df is not None:
            # Display success message
            st.success(f"✅ File uploaded successfully: {uploaded_file.name}")

            # Clean the dataframe (remove problematic columns)
            df_cleaned, removed_columns = clean_dataframe(df)

            # Show warning if columns were removed
            if removed_columns:
                st.warning(f"⚠️ Removed {len(removed_columns)} problematic column(s) automatically")
                with st.expander("🔍 View Removed Columns"):
                    for col_name, reason in removed_columns:
                        st.write(f"• **{col_name}**: {reason}")
                    st.info("💡 These columns can't be used for machine learning (complex data types, all missing, or no variance)")

                # Use cleaned dataframe
                df = df_cleaned

            # Get detailed data information
            data_info = get_data_info(df)

            # Display comprehensive data overview
            st.subheader("📊 Data Overview")

            # Basic statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", df.shape[0])
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", data_info['total_missing'])

            # Data types summary
            st.subheader("📋 Column Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Numeric:** {len(data_info['numeric_columns'])}")
                if data_info['numeric_columns']:
                    st.caption(", ".join(data_info['numeric_columns'][:3]) + ("..." if len(data_info['numeric_columns']) > 3 else ""))
            with col2:
                st.write(f"**Categorical:** {len(data_info['categorical_columns'])}")
                if data_info['categorical_columns']:
                    st.caption(", ".join(data_info['categorical_columns'][:3]) + ("..." if len(data_info['categorical_columns']) > 3 else ""))
            with col3:
                st.write(f"**Datetime:** {len(data_info['datetime_columns'])}")
                if data_info['datetime_columns']:
                    st.caption(", ".join(data_info['datetime_columns'][:3]))

            # Show detailed column types
            with st.expander("🔍 View Detailed Column Types"):
                try:
                    dtypes_df = pd.DataFrame({
                        'Column': list(data_info['dtypes'].keys()),
                        'Type': [str(t) for t in data_info['dtypes'].values()]
                    })
                    st.dataframe(dtypes_df, use_container_width=True)
                except Exception as e:
                    st.write("Column Types:")
                    for col, dtype in data_info['dtypes'].items():
                        st.write(f"• {col}: {dtype}")

            # Show missing values details if any
            if data_info['total_missing'] > 0:
                with st.expander("⚠️ View Missing Values Details"):
                    try:
                        missing_df = pd.DataFrame({
                            'Column': list(data_info['missing_values'].keys()),
                            'Missing Count': list(data_info['missing_values'].values())
                        })
                        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
                        st.dataframe(missing_df, use_container_width=True)
                    except Exception as e:
                        st.write("Missing Values:")
                        for col, count in data_info['missing_values'].items():
                            if count > 0:
                                st.write(f"• {col}: {count} missing")

            # Show sample of the data
            st.subheader("👀 Data Preview")
            try:
                st.dataframe(df.head(SAMPLE_ROWS), use_container_width=True)
            except Exception as e:
                # Fallback: convert complex columns to strings
                st.warning("⚠️ Some columns have complex data types. Converting to strings for display.")
                df_display = df.head(SAMPLE_ROWS).copy()
                for col in df_display.columns:
                    try:
                        if df_display[col].dtype == 'object':
                            df_display[col] = df_display[col].astype(str)
                    except:
                        pass
                st.dataframe(df_display, use_container_width=True)

            # Save to session state
            st.session_state.data = df
            st.session_state.filename = uploaded_file.name

            # Clear any previous processing steps
            if 'processed_data' in st.session_state:
                del st.session_state.processed_data
            if 'trained_model' in st.session_state:
                del st.session_state.trained_model
            if 'setup_initialized' in st.session_state:
                del st.session_state.setup_initialized

    # Next step button
    if 'data' in st.session_state and st.session_state.data is not None:
        st.markdown("---")
        if st.button("➡️ Proceed to Data Exploration", use_container_width=True, type="primary"):
            st.session_state.page = "Data Exploration"
            st.rerun()
    else:
        st.warning("⚠️ Please upload a dataset to continue")