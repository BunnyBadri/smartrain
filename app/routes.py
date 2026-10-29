"""
routes.py - Flask routes and REST API endpoints for the SmartRain Web Application.
"""

import os
import json
import csv
import io
from flask import Blueprint, render_template, request, jsonify, Response
from models.ann_model import RainfallPredictor, METRICS_PATH
from models.fuzzy_engine import RainfallFuzzySystem
from app.database import save_prediction, get_recent_predictions, clear_history

main_bp = Blueprint('main', __name__)

predictor = None
fuzzy_system = RainfallFuzzySystem()

def get_predictor():
    global predictor
    if predictor is None:
        predictor = RainfallPredictor()
    return predictor

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        temp = float(data.get('temperature', 28.0))
        humidity = float(data.get('relative_humidity', 65.0))
        pressure = float(data.get('atmospheric_pressure', 1010.0))
        wind = float(data.get('wind_speed', 15.0))
        prev_rain = float(data.get('previous_rainfall', 0.0))

        # 1. Soft Computing - Artificial Neural Network
        pred = get_predictor()
        predicted_rainfall = pred.predict(temp, humidity, pressure, wind, prev_rain)

        # 2. Soft Computing - Fuzzy Logic Risk Assessment
        fuzzy_result = fuzzy_system.evaluate_risk(predicted_rainfall, humidity, pressure)

        # 3. Persistence to SQLite
        top_rule_text = fuzzy_result['fired_rules'][0]['text'] if fuzzy_result['fired_rules'] else 'Default Baseline'
        save_prediction(
            temp=temp,
            humidity=humidity,
            pressure=pressure,
            wind_speed=wind,
            prev_rain=prev_rain,
            pred_rain=predicted_rainfall,
            risk_score=fuzzy_result['risk_score'],
            risk_level=fuzzy_result['risk_level'],
            top_rule=top_rule_text
        )

        return jsonify({
            'status': 'success',
            'inputs': {
                'temperature': temp,
                'relative_humidity': humidity,
                'atmospheric_pressure': pressure,
                'wind_speed': wind,
                'previous_rainfall': prev_rain
            },
            'predicted_rainfall': predicted_rainfall,
            'risk_score': fuzzy_result['risk_score'],
            'risk_level': fuzzy_result['risk_level'],
            'color': fuzzy_result['color'],
            'advisory': fuzzy_result['advisory'],
            'fired_rules': fuzzy_result['fired_rules'],
            'memberships': fuzzy_result['memberships']
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@main_bp.route('/api/metrics', methods=['GET'])
def get_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        return jsonify({'status': 'success', 'metrics': metrics})
    return jsonify({'status': 'pending', 'message': 'Metrics not computed yet.'})

@main_bp.route('/api/history', methods=['GET'])
def history():
    records = get_recent_predictions(limit=25)
    return jsonify({'status': 'success', 'history': records})

@main_bp.route('/api/history/clear', methods=['POST'])
def clear_db():
    clear_history()
    return jsonify({'status': 'success', 'message': 'History cleared'})

@main_bp.route('/api/export-csv', methods=['GET'])
def export_csv():
    records = get_recent_predictions(limit=100)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        'ID', 'Timestamp', 'Temperature (C)', 'Relative Humidity (%)',
        'Pressure (hPa)', 'Wind Speed (km/h)', 'Previous Rain (mm)',
        'Predicted Rain (mm)', 'Risk Score', 'Risk Level', 'Active Rule'
    ])
    for r in records:
        writer.writerow([
            r['id'], r['timestamp'], r['temperature'], r['relative_humidity'],
            r['atmospheric_pressure'], r['wind_speed'], r['previous_rainfall'],
            r['predicted_rainfall'], r['risk_score'], r['risk_level'], r['top_rule']
        ])
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=smartrain_predictions.csv"}
    )
