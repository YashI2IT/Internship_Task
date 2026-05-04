"""
Train all ML models for Energy Consumption Prediction
Run this script to generate model files in the models/ directory
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ENERGYLOGIC - MODEL TRAINING")
print("=" * 70)

# Step 1: Load data
print("\n[1/6] Loading dataset...")
try:
    data = pd.read_csv('data/AEP_hourly.csv')
    print(f"✓ Loaded {len(data):,} records")
except FileNotFoundError:
    print("❌ Error: data/AEP_hourly.csv not found!")
    print("   Please ensure the dataset is in the data/ directory")
    exit(1)

# Step 2: Preprocess
print("\n[2/6] Preprocessing data...")
data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.set_index('Datetime')
data = data.sort_index()

# Step 3: Feature engineering
print("\n[3/6] Engineering features...")
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

# Drop NaN
data = data.dropna()
print(f"✓ Features created: {len(data):,} samples after cleaning")

# Prepare features
feature_columns = ['hour', 'dayofweek', 'month', 'lag1', 'lag24', 'lag168',
                   'rolling_mean_24', 'rolling_std_24', 'rolling_mean_168']

X = data[feature_columns]
y = data['AEP_MW']

# Step 4: Split data
print("\n[4/6] Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)
print(f"✓ Train: {len(X_train):,} | Test: {len(X_test):,}")

# Step 5: Train models
print("\n" + "=" * 70)
print("TRAINING MODELS")
print("=" * 70)

models = {}
results = {}

# 1. Linear Regression
print("\n[MODEL 1/3] Linear Regression")
print("   Training...")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

results['LinearRegression'] = {
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
    'mae': mean_absolute_error(y_test, y_pred_lr),
    'r2': r2_score(y_test, y_pred_lr)
}
models['LinearRegression'] = lr
print("   ✓ Training complete")

# 2. XGBoost
print("\n[MODEL 2/3] XGBoost")
print("   Training (this may take a minute)...")
xgb = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

results['XGBoost'] = {
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
    'mae': mean_absolute_error(y_test, y_pred_xgb),
    'r2': r2_score(y_test, y_pred_xgb)
}
models['XGBoost'] = xgb
print("   ✓ Training complete")

# 3. Random Forest
print("\n[MODEL 3/3] Random Forest")
print("   Training (this may take a minute)...")
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

results['RandomForest'] = {
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    'mae': mean_absolute_error(y_test, y_pred_rf),
    'r2': r2_score(y_test, y_pred_rf)
}
models['RandomForest'] = rf
print("   ✓ Training complete")

# Step 6: Display results
print("\n" + "=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)
print(f"\n{'Model':<20} {'RMSE (MW)':<15} {'MAE (MW)':<15} {'R² Score':<10}")
print("-" * 70)

for model_name, metrics in results.items():
    print(f"{model_name:<20} {metrics['rmse']:<15.2f} {metrics['mae']:<15.2f} {metrics['r2']:<10.4f}")

# Find best model
best_model = min(results.items(), key=lambda x: x[1]['rmse'])
print(f"\n🏆 Best Model: {best_model[0]} (RMSE: {best_model[1]['rmse']:.2f} MW)")

# Step 7: Save models
print("\n" + "=" * 70)
print("SAVING MODELS")
print("=" * 70)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

saved_files = []
for model_name, model in models.items():
    filename = f"models/{model_name}.joblib"
    joblib.dump(model, filename)
    print(f"✓ Saved: {filename}")
    saved_files.append(filename)

# Step 8: Verify saved files
print("\n" + "=" * 70)
print("VERIFICATION")
print("=" * 70)

for filename in saved_files:
    if os.path.exists(filename):
        size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"✓ {filename:<35} ({size_mb:.2f} MB)")
        
        # Test loading
        try:
            test_model = joblib.load(filename)
            test_pred = test_model.predict(X_test[:1])
            print(f"  └─ Load test: PASSED (prediction: {test_pred[0]:.2f} MW)")
        except Exception as e:
            print(f"  └─ Load test: FAILED ({e})")
    else:
        print(f"✗ {filename} not found")

print("\n" + "=" * 70)
print("✓ ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
print("=" * 70)
print("\nYou can now run the application:")
print("  • Docker: docker-compose up")
print("  • Local: python main.py runserver")
