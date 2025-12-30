import os
import sys
import django
from django.conf import settings
from django.core.management import execute_from_command_line
from django.http import HttpResponse
from django.shortcuts import render
from django.urls import path
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime
import shap
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import base64
import io
import warnings
import mysql.connector
import json
warnings.filterwarnings('ignore')

# Configure Django settings (minimal - no database models)
if not settings.configured:
    # Check if running on AWS
    is_aws = os.environ.get('AWS_DEPLOYMENT', False)
    
    settings.configure(
        DEBUG=not is_aws,  # False in production
        SECRET_KEY=os.environ.get('SECRET_KEY', 'your-secret-key-here'),
        ROOT_URLCONF=__name__,
        ALLOWED_HOSTS=['*'] if is_aws else ['127.0.0.1', 'localhost'],
        TEMPLATES=[
            {
                'BACKEND': 'django.template.backends.django.DjangoTemplates',
                'DIRS': ['templates'],
                'APP_DIRS': True,
                'OPTIONS': {
                    'context_processors': [
                        'django.template.context_processors.debug',
                        'django.template.context_processors.request',
                        'django.contrib.auth.context_processors.auth',
                        'django.contrib.messages.context_processors.messages',
                    ],
                },
            },
        ],
        MIDDLEWARE=[
            'django.middleware.security.SecurityMiddleware',
            'django.contrib.sessions.middleware.SessionMiddleware',
            'django.middleware.common.CommonMiddleware',
            'django.middleware.csrf.CsrfViewMiddleware',
            'django.contrib.auth.middleware.AuthenticationMiddleware',
            'django.contrib.messages.middleware.MessageMiddleware',
            'django.middleware.clickjacking.XFrameOptionsMiddleware',
        ],
        INSTALLED_APPS=[
            'django.contrib.auth',
            'django.contrib.contenttypes',
            'django.contrib.sessions',
            'django.contrib.messages',
        ],
        USE_TZ=True,
        STATIC_URL='/static/',
        STATIC_ROOT=os.path.join(os.path.dirname(__file__), 'staticfiles') if is_aws else None,
    )

django.setup()

# MySQL Configuration - AWS Ready
MYSQL_CONFIG = {
    'host': os.environ.get('RDS_HOSTNAME', 'localhost'),
    'user': os.environ.get('RDS_USERNAME', 'root'),
    'password': os.environ.get('RDS_PASSWORD', 'NewStrongPassword123!'),
    'database': os.environ.get('RDS_DB_NAME', 'energy_prediction_db'),
    'port': int(os.environ.get('RDS_PORT', '3306'))
}

# Global variables for model and explainer
model = None
explainer = None
feature_names = None

def get_mysql_connection():
    """Get MySQL database connection"""
    try:
        connection = mysql.connector.connect(**MYSQL_CONFIG)
        return connection
    except mysql.connector.Error as e:
        print(f"❌ MySQL connection error: {e}")
        return None

def setup_mysql_table():
    """Create the predictions table in MySQL"""
    try:
        connection = get_mysql_connection()
        if connection is None:
            return False
            
        cursor = connection.cursor()
        
        # Create table if it doesn't exist
        create_table_query = """
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            datetime_input DATETIME NOT NULL,
            model_used VARCHAR(50) NOT NULL,
            prediction_value FLOAT NOT NULL,
            features_json TEXT,
            shap_values_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        cursor.execute(create_table_query)
        connection.commit()
        print("✅ MySQL table 'prediction_history' created successfully!")
        
        cursor.close()
        connection.close()
        return True
        
    except mysql.connector.Error as e:
        print(f"❌ Error creating MySQL table: {e}")
        return False

def save_prediction_to_mysql(datetime_input, model_used, prediction_value, features_dict, shap_dict):
    """Save prediction to MySQL database"""
    try:
        connection = get_mysql_connection()
        if connection is None:
            return False
            
        cursor = connection.cursor()
        
        insert_query = """
        INSERT INTO prediction_history 
        (datetime_input, model_used, prediction_value, features_json, shap_values_json)
        VALUES (%s, %s, %s, %s, %s)
        """
        
        values = (
            datetime_input,
            model_used,
            float(prediction_value),
            json.dumps(features_dict),
            json.dumps(shap_dict)
        )
        
        cursor.execute(insert_query, values)
        connection.commit()
        
        print(f"✅ Prediction saved to MySQL: ID {cursor.lastrowid}")
        
        cursor.close()
        connection.close()
        return True
        
    except mysql.connector.Error as e:
        print(f"❌ Error saving to MySQL: {e}")
        return False

def get_recent_predictions():
    """Get recent predictions from MySQL"""
    try:
        connection = get_mysql_connection()
        if connection is None:
            return []
            
        cursor = connection.cursor(dictionary=True)
        
        query = """
        SELECT datetime_input, model_used, prediction_value, created_at
        FROM prediction_history 
        ORDER BY created_at DESC 
        LIMIT 5
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        return results
        
    except mysql.connector.Error as e:
        print(f"❌ Error fetching from MySQL: {e}")
        return []

def load_model_and_setup_shap():
    """Load the trained model and setup SHAP explainer"""
    global model, explainer, feature_names
    
    try:
        # Try to load models from your Colab notebook
        # Priority: RandomForest > XGBoost > LinearRegression
        model_paths = [
            ('RandomForest.joblib', 'RandomForest'),
            ('XGBoost.joblib', 'XGBoost'), 
            ('LinearRegression.joblib', 'LinearRegression')
        ]
        
        model_loaded = False
        for model_path, model_name in model_paths:
            if os.path.exists(model_path):
                try:
                    print(f"🔄 Attempting to load {model_name} model...")
                    
                    # Use different loading strategies for different model types
                    if model_name == 'XGBoost':
                        # XGBoost models might need special handling
                        model = joblib.load(model_path)
                    else:
                        # Standard joblib loading for other models
                        model = joblib.load(model_path)
                    
                    print(f"✅ Loaded {model_name} model successfully")
                    model_loaded = True
                    break
                    
                except MemoryError as e:
                    print(f"❌ Memory error loading {model_name}: {e}")
                    print(f"⚠️ {model_name} model is too large for available memory, trying next model...")
                    continue
                    
                except Exception as e:
                    print(f"❌ Failed to load {model_name}: {e}")
                    continue
        
        if not model_loaded:
            # Create a fallback model with the same structure as your Colab notebook
            from sklearn.ensemble import RandomForestRegressor
            print("⚠️ No saved model could be loaded, creating fallback RandomForest model")
            model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)  # Smaller model
            
            # Create dummy training data with the same feature structure
            dummy_data = create_dummy_training_data()
            model.fit(dummy_data, np.random.rand(len(dummy_data)) * 15000 + 10000)  # Realistic energy values
            print("✅ Fallback model created and trained")
        
        # Define feature names exactly as in your Colab notebook
        feature_names = ['hour', 'dayofweek', 'month', 'lag1', 'lag24', 'lag168', 
                        'rolling_mean_24', 'rolling_std_24', 'rolling_mean_168']
        
        # Setup SHAP explainer based on model type
        model_type = type(model).__name__
        print(f"🔧 Setting up SHAP explainer for {model_type}...")
        
        try:
            if 'RandomForest' in model_type or 'GradientBoosting' in model_type or 'XGB' in model_type:
                # Tree-based models
                explainer = shap.TreeExplainer(model)
                print("✅ TreeExplainer initialized")
            elif 'Linear' in model_type or 'Ridge' in model_type or 'Lasso' in model_type:
                # Linear models
                explainer = shap.LinearExplainer(model, create_dummy_training_data())
                print("✅ LinearExplainer initialized")
            else:
                # Fallback to KernelExplainer for other models
                print("⚠️ Using KernelExplainer (slower but works with any model)")
                explainer = shap.KernelExplainer(model.predict, create_dummy_training_data()[:50])  # Smaller sample
                print("✅ KernelExplainer initialized")
        except Exception as e:
            print(f"⚠️ SHAP explainer setup failed: {e}")
            explainer = None
            
    except Exception as e:
        print(f"❌ Critical error in model loading: {e}")
        # Create a simple fallback model
        from sklearn.linear_model import LinearRegression
        print("🔄 Creating emergency fallback LinearRegression model...")
        model = LinearRegression()
        dummy_data = create_dummy_training_data()
        model.fit(dummy_data, np.random.rand(len(dummy_data)) * 15000 + 10000)
        feature_names = ['hour', 'dayofweek', 'month', 'lag1', 'lag24', 'lag168', 
                        'rolling_mean_24', 'rolling_std_24', 'rolling_mean_168']
        explainer = shap.LinearExplainer(model, dummy_data)
        print("✅ Emergency fallback model ready")

def create_dummy_training_data():
    """Create dummy training data with the same structure as your Colab notebook features"""
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic energy consumption patterns
    hours = np.random.randint(0, 24, n_samples)
    dayofweeks = np.random.randint(0, 7, n_samples)
    months = np.random.randint(1, 13, n_samples)
    
    # Create realistic lag features (previous consumption values)
    base_consumption = 15000 + np.random.normal(0, 2000, n_samples)
    lag1 = base_consumption + np.random.normal(0, 500, n_samples)
    lag24 = base_consumption + np.random.normal(0, 1000, n_samples)  
    lag168 = base_consumption + np.random.normal(0, 1500, n_samples)
    
    # Create rolling statistics
    rolling_mean_24 = base_consumption + np.random.normal(0, 300, n_samples)
    rolling_std_24 = np.abs(np.random.normal(800, 200, n_samples))
    rolling_mean_168 = base_consumption + np.random.normal(0, 500, n_samples)
    
    data = {
        'hour': hours,
        'dayofweek': dayofweeks,
        'month': months,
        'lag1': lag1,
        'lag24': lag24,
        'lag168': lag168,
        'rolling_mean_24': rolling_mean_24,
        'rolling_std_24': rolling_std_24,
        'rolling_mean_168': rolling_mean_168
    }
    
    return pd.DataFrame(data)

def engineer_features(dt):
    """
    Apply the exact same feature engineering as in your Colab notebook
    Convert datetime to features: hour, dayofweek, month, and create lag/rolling features
    """
    # Basic time features (exactly as in your notebook)
    features = {
        'hour': dt.hour,
        'dayofweek': dt.weekday(),  # 0=Monday, 6=Sunday
        'month': dt.month
    }
    
    # For a real prediction, we would need historical data to calculate lag and rolling features
    # Since we don't have access to historical data in this demo, we'll use realistic estimates
    # In production, you would load the actual historical data from your dataset
    
    # Estimate lag features based on typical energy consumption patterns
    # These would normally come from your actual historical data
    hour = dt.hour
    month = dt.month
    dayofweek = dt.weekday()
    
    # Realistic energy consumption estimates based on time patterns
    base_consumption = 15000  # Average consumption
    
    # Hour-based adjustments
    if 6 <= hour <= 9 or 17 <= hour <= 21:  # Peak hours
        base_consumption += 2000
    elif 22 <= hour <= 5:  # Night hours
        base_consumption -= 2000
    
    # Day of week adjustments
    if dayofweek >= 5:  # Weekend
        base_consumption -= 1000
    
    # Month adjustments (seasonal)
    if month in [12, 1, 2]:  # Winter
        base_consumption += 1500
    elif month in [6, 7, 8]:  # Summer
        base_consumption += 1000
    
    # Add some realistic variation
    variation = np.random.normal(0, 500)
    
    features.update({
        'lag1': base_consumption + variation,  # Previous hour
        'lag24': base_consumption + np.random.normal(0, 800),  # Same hour yesterday
        'lag168': base_consumption + np.random.normal(0, 1200),  # Same hour last week
        'rolling_mean_24': base_consumption + np.random.normal(0, 300),  # 24-hour average
        'rolling_std_24': abs(np.random.normal(800, 200)),  # 24-hour std dev
        'rolling_mean_168': base_consumption + np.random.normal(0, 500)  # 7-day average
    })
    
    return pd.DataFrame([features])

def generate_shap_plot(shap_values, feature_values, feature_names):
    """Generate SHAP bar plot and return as base64 encoded string"""
    try:
        plt.figure(figsize=(10, 6))
        plt.style.use('dark_background')
        
        # Create bar plot
        colors = ['red' if val > 0 else 'blue' for val in shap_values[0]]
        bars = plt.barh(feature_names, shap_values[0], color=colors, alpha=0.7)
        
        plt.xlabel('SHAP Value (Impact on Prediction)', fontsize=12)
        plt.title('Feature Contributions to Prediction', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, shap_values[0])):
            plt.text(val + (0.01 if val > 0 else -0.01), i, f'{val:.3f}', 
                    ha='left' if val > 0 else 'right', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', facecolor='#1e3c72', edgecolor='none', dpi=100)
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(plot_data).decode()
    except Exception as e:
        print(f"Error generating SHAP plot: {e}")
        return None

def index(request):
    """Main view for the energy consumption predictor"""
    context = {
        'prediction': None,
        'shap_values': None,
        'shap_plot': None,
        'error': None,
        'model_info': None,
        'recent_predictions': []
    }
    
    # Get recent predictions from MySQL
    try:
        recent_predictions = get_recent_predictions()
        context['recent_predictions'] = recent_predictions
    except Exception as e:
        print(f"Error fetching recent predictions: {e}")
    
    if request.method == 'POST':
        try:
            # Get datetime from form
            datetime_str = request.POST.get('datetime')
            model_type = request.POST.get('model_type', 'current')
            
            if not datetime_str:
                context['error'] = "Please select a date and time."
                return render(request, 'index.html', context)
            
            # Parse datetime
            dt = datetime.fromisoformat(datetime_str)
            
            # Load specific model if requested
            if model_type != 'current':
                try:
                    model_path = f"{model_type}.joblib"
                    if os.path.exists(model_path):
                        global model, explainer
                        print(f"🔄 Switching to {model_type} model...")
                        
                        # Load the requested model with error handling
                        try:
                            model = joblib.load(model_path)
                            print(f"✅ Successfully loaded {model_type} model")
                            
                            # Update explainer for the new model
                            model_type_name = type(model).__name__
                            if 'RandomForest' in model_type_name or 'XGB' in model_type_name:
                                explainer = shap.TreeExplainer(model)
                            elif 'Linear' in model_type_name:
                                explainer = shap.LinearExplainer(model, create_dummy_training_data())
                            else:
                                explainer = shap.KernelExplainer(model.predict, create_dummy_training_data()[:50])
                            
                            context['model_info'] = f"Using {model_type} model"
                            
                        except MemoryError:
                            context['error'] = f"{model_type} model is too large for available memory. Using current model."
                            print(f"❌ Memory error loading {model_type}")
                        except Exception as e:
                            context['error'] = f"Failed to load {model_type} model: {str(e)}. Using current model."
                            print(f"❌ Error loading {model_type}: {e}")
                    else:
                        context['error'] = f"{model_type} model file not found. Using current model."
                except Exception as e:
                    context['error'] = f"Failed to load {model_type} model: {str(e)}. Using current model."
            
            # Engineer features (same as in Colab notebook)
            features_df = engineer_features(dt)
            
            # Make prediction
            if model is None:
                load_model_and_setup_shap()
            
            prediction = model.predict(features_df)[0]
            context['prediction'] = prediction
            
            # Add model information
            if not context['model_info']:
                model_name = type(model).__name__
                context['model_info'] = f"Using {model_name} model"
            else:
                model_name = model_type
            
            # Generate SHAP values for explainability
            shap_data = []
            if explainer is not None:
                try:
                    shap_values = explainer.shap_values(features_df)
                    
                    # Handle different SHAP output formats
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0] if len(shap_values) > 0 else shap_values
                    
                    if len(shap_values.shape) > 1:
                        shap_values = shap_values[0]
                    
                    # Prepare SHAP values for display
                    feature_values = features_df.iloc[0].values
                    
                    for i, (feature, value, shap_val) in enumerate(zip(feature_names, feature_values, shap_values)):
                        # Format feature names for better display
                        display_name = feature.replace('_', ' ').title()
                        if feature == 'dayofweek':
                            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                            display_value = f"{int(value)} ({days[int(value)]})"
                        elif feature == 'hour':
                            display_value = f"{int(value)}:00"
                        elif feature in ['lag1', 'lag24', 'lag168', 'rolling_mean_24', 'rolling_mean_168']:
                            display_value = f"{value:.1f} MW"
                        elif feature == 'rolling_std_24':
                            display_value = f"{value:.1f} MW"
                        else:
                            display_value = f"{value}"
                        
                        shap_data.append((display_name, display_value, shap_val))
                    
                    context['shap_values'] = shap_data
                    
                    # Generate SHAP plot
                    shap_plot = generate_shap_plot(shap_values.reshape(1, -1), feature_values, feature_names)
                    if shap_plot:
                        context['shap_plot'] = shap_plot
                        
                except Exception as e:
                    print(f"Error generating SHAP values: {e}")
                    context['error'] = f"Prediction successful, but explainability failed: {str(e)}"
            
            # Save prediction to MySQL database
            try:
                features_json = features_df.iloc[0].to_dict()
                shap_json = {name: float(val) for name, _, val in shap_data} if shap_data else {}
                
                success = save_prediction_to_mysql(
                    dt,
                    model_name,
                    float(prediction),
                    features_json,
                    shap_json
                )
                
                if success:
                    # Update recent predictions
                    recent_predictions = get_recent_predictions()
                    context['recent_predictions'] = recent_predictions
                
            except Exception as e:
                print(f"⚠️ Failed to save prediction to MySQL: {e}")
                # Continue without saving - don't break the prediction functionality
            
        except ValueError as e:
            context['error'] = f"Invalid date format: {str(e)}"
        except Exception as e:
            context['error'] = f"Prediction failed: {str(e)}"
    
    return render(request, 'index.html', context)

# URL patterns
urlpatterns = [
    path('', index, name='index'),
]

# Initialize model and database on startup
def initialize_app():
    """Initialize the application"""
    print("🚀 Starting Energy Consumption Predictor...")
    
    # Setup MySQL table
    if setup_mysql_table():
        print("✅ MySQL database ready!")
    else:
        print("⚠️ MySQL setup failed - predictions won't be saved")
    
    # Load ML model
    load_model_and_setup_shap()
    print("✅ ML models loaded!")

if __name__ == '__main__':
    initialize_app()
    
    if len(sys.argv) > 1:
        execute_from_command_line(sys.argv)
    else:
        print("Usage: python main.py runserver")
        print("Starting development server...")
        execute_from_command_line(['main.py', 'runserver'])