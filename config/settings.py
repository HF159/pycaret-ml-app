# File: pycaret-ml-app/config/settings.py
# Configuration settings for the application

# Supported file formats
SUPPORTED_FORMATS = ["csv", "xlsx", "json"]

# Maximum file size (in MB)
MAX_FILE_SIZE = 200

# Default train-test split ratio
DEFAULT_SPLIT_RATIO = 0.2

# Default random state for reproducibility
RANDOM_STATE = 42

# Supported preprocessing methods
PREPROCESSING_METHODS = {
    "missing_values": ["mean", "median", "mode", "drop", "constant"],
    "encoding_methods": ["label", "one-hot", "none"],
    "scaling_methods": ["standard", "minmax", "robust", "none"]
}

# Supported ML task types
TASK_TYPES = ["classification", "regression"]

# Default metrics
CLASSIFICATION_METRICS = ["Accuracy", "F1", "AUC", "Precision", "Recall"]
REGRESSION_METRICS = ["MAE", "MSE", "RMSE", "R2", "MAPE"]

# Sample datasets
SAMPLE_DATASETS = {
    "Iris": {
        "task": "classification",
        "description": "Classic flower classification dataset with 3 classes"
    },
    "Boston Housing": {
        "task": "regression",
        "description": "Boston housing price prediction dataset"
    },
    "Diabetes": {
        "task": "regression",
        "description": "Diabetes progression prediction dataset"
    }
}

# File size warning threshold (in MB)
FILE_SIZE_WARNING = 50

# Number of rows to display in data samples
SAMPLE_ROWS = 10

# Maximum number of models to compare
MAX_MODELS_TO_COMPARE = 5

# Model save directory
MODEL_SAVE_DIR = "models/"

# Data save directory
DATA_SAVE_DIR = "data/"

# Allowed figure sizes for plots
FIG_SIZE = (10, 6)

# Color palette for plots
COLOR_PALETTE = "viridis"