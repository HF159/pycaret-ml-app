# File: pycaret-ml-app/pages/data_upload.py
# Page for data upload functionality

import streamlit as st
import pandas as pd
from src.data_utils import load_data, load_sample_dataset, get_data_info
from config.settings import SUPPORTED_FORMATS, SAMPLE_DATASETS, SAMPLE_ROWS

def show():
    """Display the data upload page"""
    st.header("Data Upload")
    
    # File upload widget
    st.write("Upload your dataset (CSV, Excel, or JSON)")
    uploaded_file = st.file_uploader("Choose a file", type=SUPPORTED_FORMATS)
    
    if uploaded_file is not None:
        # Load the data
        df = load_data(uploaded_file)
        
        if df is not None:
            # Display basic file information
            st.success(f"File uploaded successfully: {uploaded_file.name}")
            st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Show sample of the data
            st.subheader("Data Preview")
            st.dataframe(df.head(SAMPLE_ROWS))
            
            # Save to session state
            st.session_state.data = df
            st.session_state.filename = uploaded_file.name
    
    # Sample dataset option
    st.write("Or use a sample dataset")
    sample_dataset = st.selectbox(
        "Select a sample dataset",
        ["None"] + list(SAMPLE_DATASETS.keys())
    )
    
    if sample_dataset != "None":
        # Load the sample dataset
        df = load_sample_dataset(sample_dataset)
        
        if df is not None:
            # Display information about the dataset
            st.success(f"Sample dataset loaded: {sample_dataset}")
            st.write(SAMPLE_DATASETS[sample_dataset]["description"])
            st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Show sample of the data
            st.subheader("Data Preview")
            st.dataframe(df.head(SAMPLE_ROWS))
            
            # Save to session state
            st.session_state.data = df
            st.session_state.filename = f"sample_{sample_dataset.lower().replace(' ', '_')}.csv"

    # Next step button
    if 'data' in st.session_state and st.session_state.data is not None:
        if st.button("Proceed to Data Exploration"):
            st.session_state.page = "Data Exploration"
            st.rerun()