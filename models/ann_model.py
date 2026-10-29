"""
ann_model.py - Artificial Neural Network (MLP) regressor for rainfall amount prediction.
Handles training, evaluation metrics comparison with baseline, and model serialization.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'historical_weather.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'trained_ann.pkl')
METRICS_PATH = os.path.join(BASE_DIR, 'models', 'evaluation_metrics.json')

FEATURE_COLUMNS = [
    'temperature',
    'relative_humidity',
    'atmospheric_pressure',
    'wind_speed',
    'previous_rainfall'
]
TARGET_COLUMN = 'rainfall_amount'

def train_and_evaluate(data_path=DATA_PATH, model_save_path=MODEL_PATH, metrics_save_path=METRICS_PATH):
    if not os.path.exists(data_path):
        from data.generate_data import generate_historical_weather
        df = generate_historical_weather(num_samples=3000)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    # 80/20 Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Train Baseline Model (Historical Average Predictor)
    baseline_model = DummyRegressor(strategy='mean')
    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)

    baseline_mae = mean_absolute_error(y_test, y_pred_baseline)
    baseline_rmse = root_mean_squared_error(y_test, y_pred_baseline)
    baseline_r2 = r2_score(y_test, y_pred_baseline)

    # 2. Train Artificial Neural Network (Multi-Layer Perceptron)
    ann_model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate_init=0.005,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    ann_model.fit(X_train_scaled, y_train)

    y_pred_ann = ann_model.predict(X_test_scaled)
    # Clip negative predictions as rainfall cannot be negative
    y_pred_ann = np.maximum(0.0, y_pred_ann)

    ann_mae = mean_absolute_error(y_test, y_pred_ann)
    ann_rmse = root_mean_squared_error(y_test, y_pred_ann)
    ann_r2 = r2_score(y_test, y_pred_ann)

    # Compile Evaluation Metrics
    metrics = {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': FEATURE_COLUMNS,
        'ann_model': {
            'mae': round(float(ann_mae), 3),
            'rmse': round(float(ann_rmse), 3),
            'r2_score': round(float(ann_r2), 4),
            'iterations': int(ann_model.n_iter_),
            'hidden_layers': [64, 32, 16]
        },
        'baseline_model': {
            'name': 'Historical Average Baseline',
            'mae': round(float(baseline_mae), 3),
            'rmse': round(float(baseline_rmse), 3),
            'r2_score': round(float(baseline_r2), 4)
        },
        'improvement': {
            'mae_reduction_percent': round(float((baseline_mae - ann_mae) / baseline_mae * 100), 2),
            'rmse_reduction_percent': round(float((baseline_rmse - ann_rmse) / baseline_rmse * 100), 2)
        }
    }

    # Save model and scaler
    model_artifacts = {
        'model': ann_model,
        'scaler': scaler,
        'features': FEATURE_COLUMNS
    }
    joblib.dump(model_artifacts, model_save_path)

    with open(metrics_save_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("ANN Training & Evaluation Complete:")
    print(f"  ANN -> MAE: {ann_mae:.3f} mm | RMSE: {ann_rmse:.3f} mm | R2: {ann_r2:.4f}")
    print(f"  Baseline -> MAE: {baseline_mae:.3f} mm | RMSE: {baseline_rmse:.3f} mm | R2: {baseline_r2:.4f}")
    print(f"  Saved artifacts to {model_save_path} and {metrics_save_path}")

    return metrics

class RainfallPredictor:
    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            train_and_evaluate()
        artifacts = joblib.load(model_path)
        self.model = artifacts['model']
        self.scaler = artifacts['scaler']
        self.features = artifacts['features']

    def predict(self, temp, humidity, pressure, wind_speed, prev_rain):
        """
        Takes raw meteorological parameters and returns predicted rainfall amount in mm.
        """
        x_raw = np.array([[temp, humidity, pressure, wind_speed, prev_rain]])
        x_scaled = self.scaler.transform(x_raw)
        pred = self.model.predict(x_scaled)[0]
        return round(float(max(0.0, pred)), 2)

if __name__ == '__main__':
    train_and_evaluate()

