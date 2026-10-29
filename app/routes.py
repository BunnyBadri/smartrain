"""
routes.py - Flask routes and REST API endpoints for the SmartRain Web Application.
Now includes automated live geospatial weather ingestion via Open-Meteo API.
"""

import os
import json
import csv
import io
import requests
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

@main_bp.route('/api/live-weather', methods=['GET'])
def live_weather():
    try:
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        city_query = request.args.get('city', type=str)

        location_label = "Detected Station"

        if (lat is None or lon is None) and city_query:
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_query}&count=1"
            geo_res = requests.get(geo_url, timeout=5).json()
            results = geo_res.get('results', [])
            if results:
                lat = results[0]['latitude']
                lon = results[0]['longitude']
                location_label = f"{results[0]['name']}, {results[0].get('country', '')}"
            else:
                return jsonify({'status': 'error', 'message': 'City not found in meteorological index'}), 404
        elif lat is not None and lon is not None:
            location_label = f"Lat {lat:.2f}, Lon {lon:.2f}"
            # Reverse geocode
            try:
                rev_url = f"https://api.bigdatacloud.net/data/reverse-geocode-client?latitude={lat}&longitude={lon}&localityLanguage=en"
                rev_res = requests.get(rev_url, timeout=3).json()
                city = rev_res.get('locality') or rev_res.get('city') or rev_res.get('principalSubdivision')
                country = rev_res.get('countryName')
                if city and country:
                    location_label = f"{city}, {country}"
                elif country:
                    location_label = f"{country} ({lat:.2f}, {lon:.2f})"
            except Exception:
                pass
        else:
            # Fallback default location (Bengaluru, India)
            lat, lon = 12.9716, 77.5946
            location_label = "Bengaluru, India (Station Default)"

        # Query Open-Meteo Real-Time Weather API
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,pressure_msl,surface_pressure,wind_speed_10m,precipitation"
            f"&hourly=precipitation&past_days=1"
        )
        w_res = requests.get(weather_url, timeout=6).json()
        curr = w_res.get('current', {})
        hourly = w_res.get('hourly', {})

        temp = float(curr.get('temperature_2m', 26.0))
        humidity = float(curr.get('relative_humidity_2m', 65.0))
        pressure = float(curr.get('pressure_msl') or curr.get('surface_pressure') or 1012.0)
        wind = float(curr.get('wind_speed_10m', 12.0))

        # Sum past 24 hourly rainfall amounts
        rain_hourly = hourly.get('precipitation', [])
        prev_rain = float(round(sum(rain_hourly[:24]), 1) if len(rain_hourly) >= 24 else 0.0)

        # Soft Computing Pipeline
        pred = get_predictor()
        predicted_rainfall = pred.predict(temp, humidity, pressure, wind, prev_rain)
        fuzzy_result = fuzzy_system.evaluate_risk(predicted_rainfall, humidity, pressure)

        # Persistence
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
            top_rule=top_rule_text,
            location=location_label
        )

        return jsonify({
            'status': 'success',
            'source': 'live_satellite_feed',
            'location': {
                'name': location_label,
                'latitude': round(lat, 4),
                'longitude': round(lon, 4),
                'elevation': w_res.get('elevation', 0),
                'timezone': w_res.get('timezone', 'UTC')
            },
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
        return jsonify({'status': 'error', 'message': str(e)}), 500

@main_bp.route('/api/search-city', methods=['GET'])
def search_city():
    query = request.args.get('q', '').strip()
    if not query or len(query) < 2:
        return jsonify({'status': 'success', 'results': []})
    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={query}&count=5"
        res = requests.get(url, timeout=5).json()
        items = [
            {
                'name': r['name'],
                'country': r.get('country', ''),
                'admin1': r.get('admin1', ''),
                'latitude': r['latitude'],
                'longitude': r['longitude']
            }
            for r in res.get('results', [])
        ]
        return jsonify({'status': 'success', 'results': items})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@main_bp.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        temp = float(data.get('temperature', 28.0))
        humidity = float(data.get('relative_humidity', 65.0))
        pressure = float(data.get('atmospheric_pressure', 1010.0))
        wind = float(data.get('wind_speed', 15.0))
        prev_rain = float(data.get('previous_rainfall', 0.0))
        location = data.get('location', 'Manual / Simulation')

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
            top_rule=top_rule_text,
            location=location
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
        'ID', 'Timestamp', 'Location', 'Temperature (C)', 'Relative Humidity (%)',
        'Pressure (hPa)', 'Wind Speed (km/h)', 'Previous Rain (mm)',
        'Predicted Rain (mm)', 'Risk Score', 'Risk Level', 'Active Rule'
    ])
    for r in records:
        writer.writerow([
            r['id'], r['timestamp'], r.get('location', 'Simulated'),
            r['temperature'], r['relative_humidity'],
            r['atmospheric_pressure'], r['wind_speed'], r['previous_rainfall'],
            r['predicted_rainfall'], r['risk_score'], r['risk_level'], r['top_rule']
        ])
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=smartrain_predictions.csv"}
    )
