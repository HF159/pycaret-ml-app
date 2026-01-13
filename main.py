# File: main.py
# Main entry point for the PyCaret ML Application - Simplified and Clear

import streamlit as st
from pages import data_upload, data_exploration, preprocessing, model_training, evaluation, prediction

# App configuration
st.set_page_config(
    page_title="PyCaret ML App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .workflow-step {
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 5px;
        background-color: #f0f2f6;
    }
    .workflow-step.active {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .workflow-step.completed {
        background-color: #28a745;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Helper functions (defined before use)
def check_step_completed(step_name):
    """Check if a workflow step has been completed"""
    if step_name == "Data Upload":
        return 'data' in st.session_state and st.session_state.data is not None
    elif step_name == "Data Exploration":
        return 'target_column' in st.session_state
    elif step_name == "Preprocessing":
        return 'processed_data' in st.session_state
    elif step_name == "Model Training":
        return 'trained_model' in st.session_state
    elif step_name == "Evaluation":
        return 'trained_model' in st.session_state
    elif step_name == "Prediction":
        return 'trained_model' in st.session_state
    return False

def calculate_progress():
    """Calculate overall workflow progress"""
    steps = [
        'data' in st.session_state,
        'target_column' in st.session_state,
        'processed_data' in st.session_state,
        'setup_initialized' in st.session_state,
        'trained_model' in st.session_state,
        'trained_model' in st.session_state  # Prediction step uses the same check
    ]
    completed = sum(steps)
    return int((completed / len(steps)) * 100)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "Data Upload"

# Main header
st.markdown('<div class="main-header">🤖 PyCaret Machine Learning App</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar navigation
with st.sidebar:
    st.title("📋 Workflow")
    st.write("Follow these steps to build your ML model:")

    # Define workflow steps
    workflow_steps = [
        ("Data Upload", "📁"),
        ("Data Exploration", "🔍"),
        ("Preprocessing", "⚙️"),
        ("Model Training", "🤖"),
        ("Evaluation", "📊"),
        ("Prediction", "🔮")
    ]

    # Display workflow steps with status
    for step_name, icon in workflow_steps:
        # Determine step status
        if st.session_state.page == step_name:
            status = "active"
            status_icon = "▶️"
        elif check_step_completed(step_name):
            status = "completed"
            status_icon = "✅"
        else:
            status = ""
            status_icon = "⏸️"

        # Create button for each step
        col1, col2 = st.columns([1, 4])
        with col1:
            st.write(status_icon)
        with col2:
            if st.button(f"{icon} {step_name}", key=f"nav_{step_name}", use_container_width=True):
                st.session_state.page = step_name
                st.rerun()

    st.markdown("---")

    # Workflow Progress
    st.subheader("📈 Progress")
    progress = calculate_progress()
    st.progress(progress / 100)
    st.write(f"{progress}% Complete")

    # Show current dataset info if available
    if 'data' in st.session_state and st.session_state.data is not None:
        st.markdown("---")
        st.subheader("📊 Current Dataset")
        st.write(f"**File:** {getattr(st.session_state, 'filename', 'N/A')}")
        st.write(f"**Rows:** {st.session_state.data.shape[0]}")
        st.write(f"**Columns:** {st.session_state.data.shape[1]}")

        if 'target_column' in st.session_state:
            st.write(f"**Target:** {st.session_state.target_column}")

        if 'task_type' in st.session_state:
            st.write(f"**Task:** {st.session_state.task_type.capitalize()}")

    st.markdown("---")

    # App Info
    st.caption("PyCaret ML App v2.0")
    st.caption("Simplified for Education")

# Page routing
page_functions = {
    "Data Upload": data_upload.show,
    "Data Exploration": data_exploration.show,
    "Preprocessing": preprocessing.show,
    "Model Training": model_training.show,
    "Evaluation": evaluation.show,
    "Prediction": prediction.show
}

# Display current page
if st.session_state.page in page_functions:
    page_functions[st.session_state.page]()
else:
    st.error("Invalid page selected")
