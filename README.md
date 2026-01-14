# 🤖 PyCaret Machine Learning App

A web application for machine learning built with **Streamlit** and **PyCaret**.

---

## 📋 Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning)

### Installation Steps

#### 1. Get the Code

```bash
# Download the project ZIP file and extract it
# Then navigate to the folder
cd pycaret-ml-app
```

#### 2. Create Virtual Environment (Recommended)

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate (choose based on your terminal):
# PowerShell or CMD:
venv\Scripts\activate

# Git Bash or WSL:
source venv/Scripts/activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🌐 Ngrok Setup (For Public Access)

### First Time Setup

#### 1. Sign Up for Ngrok

1. Go to [https://ngrok.com/](https://ngrok.com/)
2. Click "Sign up" and create a free account
3. After signing in, go to your dashboard

#### 2. Get Your Authtoken

1. In the ngrok dashboard, navigate to "Your Authtoken" section
2. Copy your authtoken (looks like: `2abc123def456ghi789jkl012mno345_6PQRSTUVWXYZabcdefghij`)

#### 3. Configure Ngrok

Run this command once (replace `YOUR_AUTHTOKEN` with your actual token):

```bash
ngrok config add-authtoken YOUR_AUTHTOKEN
```

This only needs to be done once. Ngrok will remember your authtoken.

---

## 🚀 How to Run

### Option 1: Run Locally

```bash
streamlit run main.py
```
The app will be available at `http://localhost:8501`

### Option 2: Run with Ngrok (Public Access)

**Using Shell Scripts:**

**Windows:**
```bash
# Double-click start_app.bat and choose option 2
```

**Linux/Mac:**
```bash
chmod +x start_app.sh
./start_app.sh
# Choose option 2 when prompted
```

**Or Run Directly:**
```bash
python run_with_ngrok.py
```

The script will:
1. Start the Streamlit app
2. Create a public ngrok tunnel
3. Display a public URL (e.g., `https://xxxx-xx-xx-xxx-xxx.ngrok.io`)
4. Share this URL with anyone to access your app

---

## 🌐 Ngrok Configuration

The `run_with_ngrok.py` script automatically:
- Starts Streamlit on port 8501
- Creates an ngrok tunnel to expose the app publicly
- Provides a shareable URL

**Note:** The ngrok URL changes each time you restart the app (unless you use a paid ngrok account with custom domains).
