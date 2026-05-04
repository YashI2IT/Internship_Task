import os
import sys
from dotenv import load_dotenv

load_dotenv()
import django
from django.conf import settings
from django.core.management import execute_from_command_line
from django.http import HttpResponse
from django.shortcuts import render
from django.urls import path
from django.views.decorators.csrf import csrf_exempt
from django.core.wsgi import get_wsgi_application
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
    # Check if running on AWS or Render
    is_aws = os.environ.get('AWS_DEPLOYMENT', False)
    is_render = os.environ.get('RENDER', False)
    is_deployment = is_aws or is_render
    
    settings.configure(
        DEBUG=os.environ.get('DEBUG', str(not is_deployment)) == 'True',
        SECRET_KEY=os.environ.get('SECRET_KEY', 'fallback-secret-key-for-local-dev'),
        ROOT_URLCONF=__name__,
        ALLOWED_HOSTS=['*'] if is_deployment else ['127.0.0.1', 'localhost'],
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
            'whitenoise.middleware.WhiteNoiseMiddleware',
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
        SESSION_ENGINE='django.contrib.sessions.backends.signed_cookies',
        SESSION_COOKIE_NAME='energy_session',
        USE_TZ=True,
        STATIC_URL='/static/',
        STATIC_ROOT=os.path.join(os.path.dirname(__file__), 'staticfiles') if is_aws else None,
    )

django.setup()

# Expose WSGI application for Gunicorn
application = get_wsgi_application()

# MySQL Configuration - AWS Ready
MYSQL_CONFIG = {
    'host': os.environ.get('RDS_HOSTNAME', 'localhost'),
    'user': os.environ.get('RDS_USERNAME', 'root'),
    'password': os.environ.get('RDS_PASSWORD', ''),
    'database': os.environ.get('RDS_DB_NAME', 'energy_prediction_db'),
    'port': int(os.environ.get('RDS_PORT', '3306'))
}

# Initialize the app (setup tables, load models)
try:
    initialize_app()
except Exception as e:
    print(f"⚠️ App initialization warning: {e}")

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
        SELECT datetime_input, model_used, prediction_value, shap_values_json, created_at
        FROM prediction_history 
        ORDER BY created_at DESC 
        LIMIT 10
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        cursor.close()
        connection.close()
        
        return results
        
    except mysql.connector.Error as e:
        print(f"❌ Error fetching from MySQL: {e}")
        return []

def normalize_predictions(predictions):
    """Ensure datetime fields are proper datetime objects, and JSON strings are handled."""
    from datetime import datetime as dt_type
    import json
    normalized = []
    for p in predictions:
        p = dict(p)
        # Dates
        for field in ('datetime_input', 'created_at'):
            val = p.get(field)
            if isinstance(val, str):
                try: p[field] = dt_type.fromisoformat(val)
                except: p[field] = None
        
        # Ensure shap_values_json is usable as a data attribute (keep as string or parse as needed)
        # If it's a dict, convert to string for data-attribute safety
        shap = p.get('shap_values_json')
        if isinstance(shap, dict):
            p['shap_values_json'] = json.dumps(shap)
            
        normalized.append(p)
    return normalized

def load_model_and_setup_shap():
    """Load the trained model and setup SHAP explainer"""
    global model, explainer, feature_names
    
    try:
        # Check environment
        is_render = os.environ.get('RENDER', False)
        
        # Try to load models from your Colab notebook
        # Priority: RandomForest > XGBoost > LinearRegression
        model_paths = [
            ('models/RandomForest.joblib', 'RandomForest'),
            ('models/XGBoost.joblib', 'XGBoost'), 
            ('models/LinearRegression.joblib', 'LinearRegression')
        ]
        
        # On Render Free tier, skip RandomForest to avoid OOM
        if is_render:
             print("🚀 Running on Render - skipping heavy RandomForest model to save memory")
             model_paths = [p for p in model_paths if p[1] != 'RandomForest']
        
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
            model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
            
            # Create training data from actual dataset or use statistics
            dummy_data = create_dummy_training_data()
            
            # Generate target values based on actual dataset statistics
            # AEP dataset: mean=15499.5, std=2591.4
            target_values = np.random.normal(15499.5, 2591.4, len(dummy_data))
            target_values = np.clip(target_values, 9581, 25695)  # Clip to actual dataset range
            
            model.fit(dummy_data, target_values)
            print("✅ Fallback model created and trained on dataset statistics")
        
        # Define feature names exactly as in your Colab notebook
        feature_names = ['hour', 'dayofweek', 'month', 'lag1', 'lag24', 'lag168', 
                        'rolling_mean_24', 'rolling_std_24', 'rolling_mean_168']
        
        # Setup SHAP explainer based on model type with memory-efficient approach
        model_type = type(model).__name__
        print(f"🔧 Setting up SHAP explainer for {model_type}...")
        
        try:
            if 'RandomForest' in model_type or 'GradientBoosting' in model_type or 'XGB' in model_type:
                # Tree-based models - use smaller background dataset to reduce memory
                background_data = create_dummy_training_data()[:100]  # Use only 100 samples
                explainer = shap.TreeExplainer(model, background_data, feature_perturbation='interventional')
                print("✅ TreeExplainer initialized (memory-optimized)")
            elif 'Linear' in model_type or 'Ridge' in model_type or 'Lasso' in model_type:
                # Linear models
                background_data = create_dummy_training_data()[:100]
                explainer = shap.LinearExplainer(model, background_data)
                print("✅ LinearExplainer initialized")
            else:
                # Fallback to KernelExplainer for other models
                print("⚠️ Using KernelExplainer (slower but works with any model)")
                background_data = create_dummy_training_data()[:50]  # Even smaller sample
                explainer = shap.KernelExplainer(model.predict, background_data)
                print("✅ KernelExplainer initialized")
        except MemoryError as e:
            print(f"⚠️ SHAP explainer memory error: {e}")
            print("⚠️ Running without SHAP explainability (predictions will still work)")
            explainer = None
        except Exception as e:
            print(f"⚠️ SHAP explainer setup failed: {e}")
            print("⚠️ Running without SHAP explainability (predictions will still work)")
            explainer = None
            
    except Exception as e:
        print(f"❌ Critical error in model loading: {e}")
        # Create a simple fallback model
        from sklearn.linear_model import LinearRegression
        print("🔄 Creating emergency fallback LinearRegression model...")
        model = LinearRegression()
        dummy_data = create_dummy_training_data()
        
        # Generate target values based on actual dataset statistics
        target_values = np.random.normal(15499.5, 2591.4, len(dummy_data))
        target_values = np.clip(target_values, 9581, 25695)
        
        model.fit(dummy_data, target_values)
        feature_names = ['hour', 'dayofweek', 'month', 'lag1', 'lag24', 'lag168', 
                        'rolling_mean_24', 'rolling_std_24', 'rolling_mean_168']
        try:
            explainer = shap.LinearExplainer(model, dummy_data[:100])
            print("✅ Emergency fallback model ready with SHAP")
        except:
            explainer = None
            print("✅ Emergency fallback model ready (without SHAP)")

def create_dummy_training_data():
    """Load actual historical data from AEP dataset for SHAP background"""
    try:
        # Try to load actual historical data
        csv_path = 'data/AEP_hourly.csv'
        if os.path.exists(csv_path):
            data = pd.read_csv(csv_path)
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            data = data.set_index('Datetime')
            data = data.sort_index()
            
            # Feature engineering (same as training)
            data['hour'] = data.index.hour
            data['dayofweek'] = data.index.dayofweek
            data['month'] = data.index.month
            
            # Lag features
            data['lag1'] = data['AEP_MW'].shift(1)
            data['lag24'] = data['AEP_MW'].shift(24)
            data['lag168'] = data['AEP_MW'].shift(168)
            
            # Rolling statistics
            data['rolling_mean_24'] = data['AEP_MW'].rolling(window=24).mean()
            data['rolling_std_24'] = data['AEP_MW'].rolling(window=24).std()
            data['rolling_mean_168'] = data['AEP_MW'].rolling(window=168).mean()
            
            # Drop NaN and select features
            data = data.dropna()
            feature_columns = ['hour', 'dayofweek', 'month', 'lag1', 'lag24', 'lag168',
                             'rolling_mean_24', 'rolling_std_24', 'rolling_mean_168']
            
            # Return a sample of actual data
            return data[feature_columns].sample(n=min(1000, len(data)), random_state=42)
    except Exception as e:
        print(f"⚠️  Could not load historical data: {e}")
    
    # Fallback: create synthetic data based on dataset statistics
    # These values are from the actual AEP dataset statistics
    np.random.seed(42)
    n_samples = 1000
    
    # Temporal features
    hours = np.random.randint(0, 24, n_samples)
    dayofweeks = np.random.randint(0, 7, n_samples)
    months = np.random.randint(1, 13, n_samples)
    
    # Based on actual AEP dataset: mean=15499.5, std=2591.4
    base_consumption = np.random.normal(15499.5, 2591.4, n_samples)
    
    # Lag features with realistic correlations
    lag1 = base_consumption + np.random.normal(0, 500, n_samples)
    lag24 = base_consumption + np.random.normal(0, 800, n_samples)
    lag168 = base_consumption + np.random.normal(0, 1200, n_samples)
    
    # Rolling statistics
    rolling_mean_24 = base_consumption + np.random.normal(0, 300, n_samples)
    rolling_std_24 = np.abs(np.random.normal(800, 200, n_samples))
    rolling_mean_168 = base_consumption + np.random.normal(0, 400, n_samples)
    
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
    
    # Try to load actual historical data for lag and rolling features
    try:
        csv_path = 'data/AEP_hourly.csv'
        if os.path.exists(csv_path):
            data = pd.read_csv(csv_path)
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            data = data.set_index('Datetime')
            data = data.sort_index()
            
            # Find the closest historical data point before the target datetime
            historical_data = data[data.index < dt]
            
            if len(historical_data) >= 168:  # Need at least 1 week of history
                # Get actual lag values
                lag1_value = historical_data['AEP_MW'].iloc[-1]  # 1 hour ago
                lag24_value = historical_data['AEP_MW'].iloc[-24] if len(historical_data) >= 24 else lag1_value
                lag168_value = historical_data['AEP_MW'].iloc[-168] if len(historical_data) >= 168 else lag24_value
                
                # Calculate actual rolling statistics
                recent_24h = historical_data['AEP_MW'].iloc[-24:]
                recent_168h = historical_data['AEP_MW'].iloc[-168:]
                
                features.update({
                    'lag1': lag1_value,
                    'lag24': lag24_value,
                    'lag168': lag168_value,
                    'rolling_mean_24': recent_24h.mean(),
                    'rolling_std_24': recent_24h.std(),
                    'rolling_mean_168': recent_168h.mean()
                })
                
                print(f"✅ Using actual historical data for feature engineering")
                return pd.DataFrame([features])
    except Exception as e:
        print(f"⚠️  Could not load historical data: {e}")
    
    # Fallback: Use statistical patterns from the actual AEP dataset
    # Dataset statistics: mean=15499.5 MW, std=2591.4 MW, range=[9581, 25695]
    hour = dt.hour
    month = dt.month
    dayofweek = dt.weekday()
    
    # Start with dataset mean
    base_consumption = 15499.5
    
    # Hour-based adjustments (from actual dataset patterns)
    if 6 <= hour <= 9:  # Morning peak
        base_consumption += 1800
    elif 17 <= hour <= 21:  # Evening peak
        base_consumption += 2200
    elif 0 <= hour <= 5:  # Night hours
        base_consumption -= 2500
    elif 10 <= hour <= 16:  # Midday
        base_consumption += 800
    
    # Day of week adjustments (from actual dataset patterns)
    if dayofweek >= 5:  # Weekend (Saturday=5, Sunday=6)
        base_consumption -= 1200
    
    # Month adjustments (seasonal patterns from actual data)
    if month in [12, 1, 2]:  # Winter
        base_consumption += 1600
    elif month in [6, 7, 8]:  # Summer
        base_consumption += 1400
    elif month in [3, 4, 5, 9, 10, 11]:  # Spring/Fall
        base_consumption -= 400
    
    # Add realistic variation based on dataset std
    variation = np.random.normal(0, 400)
    
    features.update({
        'lag1': base_consumption + variation,
        'lag24': base_consumption + np.random.normal(0, 600),
        'lag168': base_consumption + np.random.normal(0, 900),
        'rolling_mean_24': base_consumption + np.random.normal(0, 250),
        'rolling_std_24': abs(np.random.normal(750, 180)),
        'rolling_mean_168': base_consumption + np.random.normal(0, 350)
    })
    
    print(f"ℹ️  Using statistical patterns for feature engineering (no historical data)")
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
        'recent_predictions': [],
        'active_tab': 'dashboard',
        'shap_chart_labels': None,
        'shap_chart_values': None
    }
    
    # Get recent predictions: MySQL first, session fallback
    try:
        db_preds = get_recent_predictions()
        if db_preds:
            context['recent_predictions'] = normalize_predictions(db_preds)
        else:
            context['recent_predictions'] = normalize_predictions(request.session.get('recent_predictions', []))
    except Exception as e:
        print(f"Error fetching recent predictions: {e}")
        context['recent_predictions'] = normalize_predictions(request.session.get('recent_predictions', []))

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
                    model_path = f"models/{model_type}.joblib"
                    if os.path.exists(model_path):
                        global model, explainer
                        print(f"🔄 Switching to {model_type} model...")
                        
                        # Check memory safety on Render
                        if is_render and model_type == 'RandomForest':
                             context['error'] = "Random Forest is disabled on Render Free tier to prevent crashes. Using XGBoost instead."
                             model_type = 'XGBoost'
                             model_path = "models/XGBoost.joblib"

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
                            context['error'] = f"{model_type} model is too large for available memory. Using XGBoost."
                            print(f"❌ Memory error loading {model_type}")
                        except Exception as e:
                            context['error'] = f"Failed to load {model_type} model: {str(e)}. Using current model."
                            print(f"❌ Error loading {model_type}: {e}")
                    else:
                        # Silently skip warning if it's RandomForest on Render
                        if not (is_render and model_type == 'RandomForest'):
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
                    
                    # Flatten shap_values if it's multidimensional (common with XGB/RF)
                    flat_shap = np.array(shap_values).flatten()
                    
                    for i, (feature, value, shap_val) in enumerate(zip(feature_names, feature_values, flat_shap)):
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
                        
                        shap_data.append((display_name, display_value, float(shap_val)))
                    
                    context['shap_values'] = shap_data
                    
                    # Add JSON serialized data for Chart.js
                    labels = [item[0] for item in shap_data]
                    values = [item[2] for item in shap_data]
                    context['shap_chart_labels'] = json.dumps(labels)
                    context['shap_chart_values'] = json.dumps(values)
                    print(f"✅ SHAP data generated: {len(labels)} features")
                    
                    # Generate SHAP plot (keeping as fallback)
                    shap_plot = generate_shap_plot(shap_values.reshape(1, -1), feature_values, feature_names)
                    if shap_plot:
                        context['shap_plot'] = shap_plot
                        
                except MemoryError as e:
                    print(f"⚠️ SHAP memory error during prediction: {e}")
                    print("⚠️ Prediction successful but explainability skipped due to memory constraints")
                    # Prediction still works, just no SHAP values
                except Exception as e:
                    print(f"⚠️ Error generating SHAP values: {e}")
                    # Prediction still works, just no SHAP values
            else:
                print("ℹ️  SHAP explainer not available - prediction without explainability")
            
            # Save prediction to MySQL database
            new_record = {
                'datetime_input': dt,
                'model_used': model_name,
                'prediction_value': float(prediction),
                'created_at': datetime.now()
            }
            db_saved = False
            try:
                features_json = features_df.iloc[0].to_dict()
                shap_json = {name: float(val) for name, _, val in shap_data} if shap_data else {}
                db_saved = save_prediction_to_mysql(dt, model_name, float(prediction), features_json, shap_json)
            except Exception as e:
                print(f"⚠️ Failed to save prediction to MySQL: {e}")

            # Always update recent predictions — use DB if available, fallback to session
            if db_saved:
                context['recent_predictions'] = normalize_predictions(get_recent_predictions())
                request.session['recent_predictions'] = []
            else:
                # Session fallback: keep last 10 predictions in memory
                session_preds = request.session.get('recent_predictions', [])
                session_preds.insert(0, {
                    'datetime_input': dt.strftime('%Y-%m-%dT%H:%M:%S'),
                    'model_used': model_name,
                    'prediction_value': float(prediction),
                    'created_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                })
                request.session['recent_predictions'] = session_preds[:10]
                request.session.modified = True
                context['recent_predictions'] = normalize_predictions(session_preds[:10])
                context['db_warning'] = True
            
        except ValueError as e:
            context['error'] = f"Invalid date format: {str(e)}"
        except Exception as e:
            context['error'] = f"Prediction failed: {str(e)}"
    
    return render(request, 'index.html', context)

# URL patterns
def export_csv(request):
    """Export prediction history as a CSV download."""
    import csv
    from django.http import HttpResponse

    db_preds = get_recent_predictions()
    session_preds = request.session.get('recent_predictions', [])
    all_preds = normalize_predictions(db_preds if db_preds else session_preds)

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="energy_predictions.csv"'

    writer = csv.writer(response)
    writer.writerow(['Record', 'Target Date & Time', 'Model Used', 'Forecast (MW)', 'Status', 'Created At'])

    for i, pred in enumerate(all_preds, start=1):
        dt_input = pred.get('datetime_input')
        created  = pred.get('created_at')
        writer.writerow([
            f'#{i:04d}',
            dt_input.strftime('%d %b %Y %H:%M') if dt_input else '',
            pred.get('model_used', ''),
            f"{pred.get('prediction_value', ''):.2f}" if pred.get('prediction_value') is not None else '',
            'Completed',
            created.strftime('%d %b %Y %H:%M') if created else ''
        ])

    return response

def history_view(request):
    db_preds = get_recent_predictions()
    session_preds = request.session.get('recent_predictions', [])
    all_preds = db_preds if db_preds else session_preds
    return render(request, 'index.html', {'active_tab': 'history', 'recent_predictions': normalize_predictions(all_preds)})

def models_view(request):
    return render(request, 'index.html', {'active_tab': 'models'})

def analytics_view(request):
    db_preds = get_recent_predictions()
    session_preds = request.session.get('recent_predictions', [])
    all_preds = normalize_predictions(db_preds if db_preds else session_preds)
    
    # Calculate average load
    avg_load = 0
    if all_preds:
        total = sum(float(p.get('prediction_value', 0)) for p in all_preds)
        avg_load = total / len(all_preds)
    
    return render(request, 'index.html', {
        'active_tab': 'analytics', 
        'recent_predictions': all_preds,
        'avg_load': avg_load
    })

def settings_view(request):
    return render(request, 'index.html', {'active_tab': 'settings'})

urlpatterns = [
    path('', index, name='index'),
    path('history/', history_view, name='history'),
    path('models/', models_view, name='models'),
    path('analytics/', analytics_view, name='analytics'),
    path('settings/', settings_view, name='settings'),
    path('export/', export_csv, name='export'),
]

# Initialize model and database on startup
def initialize_app():
    """Initialize the application"""
    print("🚀 Starting Energy Consumption Predictor...")
    
    # Check which models are available
    print("\n📦 Checking available models...")
    available_models = []
    for model_file in ['models/RandomForest.joblib', 'models/XGBoost.joblib', 'models/LinearRegression.joblib']:
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"   ✅ {model_file:<35} ({size_mb:.2f} MB)")
            available_models.append(model_file)
        else:
            print(f"   ❌ {model_file:<35} (Not found)")
    
    if not available_models:
        print("\n⚠️  WARNING: No model files found!")
        print("   Please run: python fix_and_train.py")
        print("   The app will use a fallback model with reduced accuracy.\n")
    else:
        print(f"\n✅ Found {len(available_models)} model(s)")
    
    # Setup MySQL table
    print("\n🗄️  Setting up database...")
    if setup_mysql_table():
        print("✅ MySQL database ready!")
    else:
        print("⚠️  MySQL setup failed - predictions will use session storage")
    
    # Load ML model
    print("\n🤖 Loading ML models...")
    load_model_and_setup_shap()
    print("✅ ML models loaded!")
    
    print("\n" + "="*60)
    print("✅ APPLICATION READY!")
    print("="*60)
    print("📍 Server will start at: http://127.0.0.1:8000/")
    print("="*60 + "\n")

if __name__ == '__main__':
    initialize_app()
    
    if len(sys.argv) > 1:
        execute_from_command_line(sys.argv)
    else:
        print("Usage: python main.py runserver")
        print("Starting development server...")
        execute_from_command_line(['main.py', 'runserver'])
