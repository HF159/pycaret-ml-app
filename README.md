# PyCaret Machine Learning App

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.18.0%2B-red)](https://streamlit.io/)
[![PyCaret](https://img.shields.io/badge/PyCaret-3.0.0%2B-green)](https://pycaret.org/)

A comprehensive Streamlit-based machine learning application using PyCaret for educational purposes in a Master's program in computational linguistics. This application offers a simple and intuitive interface for students to experiment with machine learning concepts without requiring extensive programming knowledge.

## Features

### Data Upload and Exploration
- Support for CSV, Excel, and JSON file formats
- Basic data overview (types, missing values)
- Visual exploration with histograms, correlation matrices, and box plots
- Automatic target variable suggestions

### Preprocessing
- Drop unnecessary columns
- Handle missing values with various methods
- Encode categorical data with Label or One-Hot Encoding
- Feature scaling options
- Train/test split customization

### Model Training
- Automatic detection of classification or regression tasks
- Model selection from PyCaret's comprehensive library
- Class imbalance handling for classification tasks
- Hyperparameter tuning capabilities

### Model Comparison and Evaluation
- Compare multiple models using relevant metrics
- Visualize model performance with:
  - Confusion matrices for classification
  - Residual plots for regression
  - Feature importance graphs
  - Learning curves
- Detailed performance metrics

### Prediction
- Make predictions on test data
- Upload new data for batch predictions
- Input individual data points through a dynamic form
- Download prediction results

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python, PyCaret, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Sharing**: Ngrok (for public access when needed)

## Screenshots
![image](https://github.com/user-attachments/assets/6877c8a5-b326-40d7-946a-cd9ce45cb189)

![image](https://github.com/user-attachments/assets/b0691145-9fbb-4ec6-9776-1e73ab862e83)



## Setup Instructions

### Prerequisites

- Python 3.8 to 3.11
- pip (Python package installer)
- Virtual environment tool (recommended)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/YOUR-USERNAME/pycaret-ml-app.git
cd pycaret-ml-app
```

2. **Create and activate a virtual environment**

```bash
# Using venv
python -m venv env

# On Windows
env\Scripts\activate

# On MacOS/Linux
source env/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

> **Note**: Installing PyCaret might take some time as it installs several dependencies.

4. **Create necessary directories**

```bash
mkdir -p models data
```

## Usage

### Running the App

```bash
streamlit run main.py
```

This will start the Streamlit server, and the app should automatically open in your default web browser. If not, you can access it at `http://localhost:8501`.

### Step-by-Step Guide

1. **Data Upload**
   - Upload your dataset (CSV, Excel, or JSON)
   - Alternatively, select from available sample datasets

2. **Data Exploration**
   - View basic statistics and information about your dataset
   - Explore data distributions and correlations
   - Select your target variable for prediction

3. **Preprocessing**
   - Choose columns to include in your analysis
   - Handle missing values with methods like mean, median, or constant values
   - Encode categorical variables
   - Apply feature scaling if needed
   - Set your train/test split ratio

4. **Model Training**
   - The app will suggest whether classification or regression is appropriate
   - Choose to compare multiple models or focus on a specific one
   - For small or imbalanced datasets, use the balancing options
   - Tune hyperparameters for better performance

5. **Evaluation**
   - View detailed performance metrics
   - Examine visualizations of model performance
   - Save your trained model for future use

6. **Prediction**
   - Generate predictions on your test set
   - Upload new data for batch predictions
   - Use the form for individual predictions
   - Download prediction results

## Project Structure

```
pycaret-ml-app/
│
├── main.py                # Main application entry point
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
│
├── config/                # Configuration files
│   └── settings.py        # Application settings
│
├── src/                   # Source code
│   ├── data_utils.py      # Data loading and utilities
│   ├── evaluation.py      # Model evaluation functions
│   ├── model_training.py  # Model training functions
│   ├── prediction.py      # Prediction functions
│   └── preprocessing.py   # Data preprocessing functions
│
├── pages/                 # Streamlit UI pages
│   ├── data_upload.py     # Data upload page
│   ├── data_exploration.py # Data exploration page
│   ├── preprocessing.py   # Preprocessing page
│   ├── model_training.py  # Model training page
│   ├── evaluation.py      # Evaluation page
│   └── prediction.py      # Prediction page
│
├── models/                # Saved models (gitignored)
└── data/                  # Saved datasets (gitignored)
```

## Testing

To ensure the application works correctly:

1. Start with sample datasets to verify all functionality
2. Test with small datasets first (<1000 rows)
3. Gradually increase to medium-sized datasets (1000-10000 rows)
4. For computational linguistics specific use cases, test with text-heavy datasets

## Dependencies

The application requires the following main Python packages:

- streamlit >= 1.18.0
- pycaret >= 3.0.0
- pandas >= 1.5.0
- numpy >= 1.23.0
- matplotlib >= 3.5.0
- seaborn >= 0.12.0
- scikit-learn >= 1.0.0
- openpyxl >= 3.0.0
- pyngrok >= 5.2.0

All dependencies are listed in the `requirements.txt` file.

## Future Enhancements

- Add more advanced ML features (feature engineering, cross-validation options)
- Implement additional visualization types for linguistic data
- Support for specialized NLP models and text analysis
- Export trained models for external use
- Create a companion tutorial series

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Created for educational purposes in a Master's program in computational linguistics.
