Energy Consumption Prediction System

Problem Statement: Predictive Analytics for Energy Consumption Forecasting

This project consists of a ML model that predicts energy consumption (in MW) for a given date and time using multiple machine learning approaches including Linear Regression, XGBoost, and Random Forest. The frontend was built using HTML and Bootstrap, whereas the backend was built via Django. The project analyzes temporal patterns including hour-of-day, day-of-week, and seasonal variations to provide accurate energy consumption predictions. Users can track their prediction history, view analytics, and manage their profile through an interactive dashboard.

Colab File: (https://colab.research.google.com/drive/1WTW3W8QAdFnHVQ_Yktg1J_yxpDYiiCLN?usp=sharing)

1) Quick Start

### Prerequisites
- Python 3.8+
- MySQL Server 8.0+ (running)
- MySQL client libraries

### Setup Steps

**For Windows Users (Recommended):**
```bash
# 1. Run the Windows setup script
python setup_windows.py

# 2. Follow the prompts to enter MySQL credentials

# 3. Update main.py with the generated configuration

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Run the app
python main.py runserver
```

**For Linux/Mac Users:**
1. **Install MySQL dependencies:**
   ```bash
   # On macOS
   brew install mysql
   pip install mysqlclient
   
   # On Ubuntu/Debian
   sudo apt-get install python3-dev default-libmysqlclient-dev build-essential
   pip install mysqlclient
   ```

2. **Setup MySQL database:**
   ```bash
   python setup_mysql.py
   ```

3. **Configure and run:**
   ```bash
   pip install -r requirements.txt
   python main.py runserver
   ```

**Open browser:** `http://127.0.0.1:8000/`

2) Key Features

- Real-time predictions using trained ML models (RandomForest, XGBoost, LinearRegression)
- SHAP explainability showing feature contributions and why each prediction was made
- Model switching via dropdown selection for comparing different algorithms
- Temporal pattern analysis including hour-of-day, day-of-week, and seasonal variations
- Professional UI with modern dark theme and glassmorphism design
- Interactive dashboard for energy consumption forecasting

3) Machine Learning Models

Available Models:
- RandomForest.joblib (primary) - Ensemble method with high accuracy
- XGBoost.joblib (secondary) - Gradient boosting for complex patterns
- LinearRegression.joblib (fallback) - Simple baseline model

Feature Engineering:
- Temporal features: Hour, day of week, month
- Lag features: lag1 (previous hour), lag24 (same hour yesterday), lag168 (same hour last week)
- Rolling statistics: 24-hour and 168-hour rolling mean and standard deviation
- Seasonal patterns: Monthly and weekly consumption variations

4) Technical Architecture

- **Framework:** Django (minimal single-file approach)
- **Database:** MySQL 8.0+ for data persistence
- **Frontend:** HTML with inline CSS, Bootstrap styling
- **Backend:** Python with scikit-learn, XGBoost, SHAP
- **Explainability:** SHAP (SHapley Additive exPlanations) for model interpretability
- **Visualization:** Dynamic SHAP plots embedded as base64 images
- Visualization: Dynamic SHAP plots embedded as base64 images

5) File Structure

```
├── main.py                    # Complete Django application
├── templates/index.html       # Single HTML template with Bootstrap
├── requirements.txt           # Python dependencies
├── RandomForest.joblib        # Trained Random Forest model
├── XGBoost.joblib            # Trained XGBoost model
├── LinearRegression.joblib    # Trained Linear Regression model
├── AEP_hourly.csv            # Original AEP dataset
└── Internship (1).ipynb      # ML training notebook (Colab)
```

6) Dataset

The project uses the AEP (American Electric Power) Hourly Dataset containing historical energy consumption data with temporal patterns for training and validation.

7) Implementation Highlights

- Minimal Architecture: Single-file Django app with no separate models/migrations
- Real-time Explainability: Dynamic SHAP value computation for each prediction
- Model Flexibility: Easy switching between different ML algorithms
- Production Ready: Robust error handling and fallback mechanisms
- Responsive Design: Works seamlessly on desktop and mobile devices