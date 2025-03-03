# PyCaret Machine Learning App

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.18.0%2B-red)
![PyCaret](https://img.shields.io/badge/PyCaret-3.0.0%2B-green)

A comprehensive Streamlit-based machine learning application using PyCaret for educational purposes in a Master's program in computational linguistics. This application offers a simple and intuitive interface for students to experiment with machine learning concepts without requiring extensive programming knowledge.

## 📋 Table of Contents
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Screenshots](#screenshots)
- [Setup Instructions](#setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Running the App](#running-the-app)
  - [Step-by-Step Guide](#step-by-step-guide)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### 1. Data Upload and Exploration
- Support for CSV, Excel, and JSON file formats
- Basic data overview (types, missing values)
- Visual exploration with histograms, correlation matrices, and box plots
- Automatic target variable suggestions

### 2. Preprocessing
- Drop unnecessary columns
- Handle missing values with various methods
- Encode categorical data with Label or One-Hot Encoding
- Feature scaling options
- Train/test split customization

### 3. Model Training
- Automatic detection of classification or regression tasks
- Model selection from PyCaret's comprehensive library
- Class imbalance handling for classification tasks
- Hyperparameter tuning capabilities

### 4. Model Comparison and Evaluation
- Compare multiple models using relevant metrics
- Visualize model performance with:
  - Confusion matrices for classification
  - Residual plots for regression
  - Feature importance graphs
  - Learning curves
- Detailed performance metrics

### 5. Prediction
- Make predictions on test data
- Upload new data for batch predictions
- Input individual data points through a dynamic form
- Download prediction results

### 6. Educational Workflow
- Step-by-step guided process
- Clear explanations at each stage
- Progress tracking

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python, PyCaret, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Sharing**: Ngrok (for public access when needed)

## 📷 Screenshots

[To be added - screenshots of different app pages and features]

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment tool (recommended)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/pycaret-ml-app.git
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

## 📱 Usage

### Running the App

```bash
streamlit run main.py
```

This will start the Streamlit server, and the app should automatically open in your default web browser. If not, you can access it at `http://localhost:8501`.

### Step-by-Step Guide

#### 1. Data Upload
- Upload your dataset (CSV, Excel, or JSON)
- Or select from sample datasets (Iris, Boston Housing, Diabetes)

#### 2. Data Exploration
- View basic statistics and information about your dataset
- Visualize distributions and correlations
- Select your target variable for prediction

#### 3. Preprocessing
- Choose which columns to include in your analysis
- Handle missing values with methods like mean, median, or constant values
- Encode categorical variables
- Apply feature scaling if necessary
- Set your train/test split ratio

#### 4. Model Training
- The app will suggest whether classification or regression is appropriate
- Choose to compare multiple models or focus on a specific one
- For small or imbalanced datasets, use the provided balancing options
- Tune hyperparameters for better performance

#### 5. Evaluation
- View detailed performance metrics
- Examine visual representations of model performance
- Save your trained model for future use

#### 6. Prediction
- Generate predictions on your test set
- Upload new data for batch predictions
- Use the form for individual predictions
- Download prediction results

### 💡 Tips for Best Results

- Start with small, clean datasets to understand the workflow
- For computational linguistics applications, ensure text data is properly preprocessed
- Classification works best when target classes are well-balanced
- For regression tasks, check for outliers that might affect model performance
- Experiment with different preprocessing methods to see their impact

## 📁 Project Structure

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

## 🧪 Testing

To ensure the application works correctly:

1. Start with sample datasets to verify all functionality
2. Test with small datasets first (<1000 rows)
3. Gradually increase to medium-sized datasets (1000-10000 rows)
4. For computational linguistics specific use cases, test with text-heavy datasets

```bash
# Future tests will be added here
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

[MIT License](LICENSE) - feel free to use and adapt for educational purposes.

---

## 📚 Additional Resources

- [PyCaret Documentation](https://pycaret.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Computational Linguistics Resources](#) [To be added]

---

Created for educational purposes in a Master's program in computational linguistics.#   p y c a r e t - m l - a p p  
 