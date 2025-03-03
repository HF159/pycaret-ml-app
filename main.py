# File: pycaret-ml-app/main.py
# Main entry point for the PyCaret ML application

import streamlit as st
from pages import data_upload, data_exploration, preprocessing, model_training, evaluation, prediction

# App configuration
st.set_page_config(
    page_title="PyCaret ML App",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "Data Upload"

# App title and description
st.title("PyCaret Machine Learning App")
st.write("A simple machine learning application for educational purposes in computational linguistics")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["Data Upload", "Data Exploration", "Preprocessing", "Model Training", "Evaluation", "Prediction"],
    index=["Data Upload", "Data Exploration", "Preprocessing", "Model Training", "Evaluation", "Prediction"].index(st.session_state.page)
)

# Update current page in session state
st.session_state.page = page

# Show workflow progress
st.sidebar.progress(
    (["Data Upload", "Data Exploration", "Preprocessing", "Model Training", "Evaluation", "Prediction"].index(page) + 1) / 6
)

# Display current workflow step
st.sidebar.write(f"Current step: {page}")

# Display data info if available
if 'data' in st.session_state and st.session_state.data is not None:
    st.sidebar.write(f"Dataset: {getattr(st.session_state, 'filename', 'Custom dataset')}")
    st.sidebar.write(f"Rows: {st.session_state.data.shape[0]}, Columns: {st.session_state.data.shape[1]}")

    if 'target_column' in st.session_state:
        st.sidebar.write(f"Target: {st.session_state.target_column}")

    if 'task_type' in st.session_state:
        st.sidebar.write(f"Task: {st.session_state.task_type}")

# Page routing
if page == "Data Upload":
    data_upload.show()
elif page == "Data Exploration":
    data_exploration.show()
elif page == "Preprocessing":
    preprocessing.show()
elif page == "Model Training":
    model_training.show()
elif page == "Evaluation":
    evaluation.show()
elif page == "Prediction":
    prediction.show()

# Footer
st.sidebar.markdown("---")
st.sidebar.write("PyCaret ML App v1.0")
st.sidebar.write("For educational purposes")