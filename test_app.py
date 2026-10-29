"""
test_app.py - Integration and unit tests for SmartRain application.
"""

import unittest
import json
import os
from app import create_app
from models.ann_model import RainfallPredictor
from models.fuzzy_engine import RainfallFuzzySystem

class SmartRainTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()
        cls.predictor = RainfallPredictor()
        cls.fuzzy = RainfallFuzzySystem()

    def test_ann_predictor(self):
        # High rain scenario: High humidity (90%), Low pressure (995 hPa), High prior rain (40mm)
        rain_high = self.predictor.predict(
            temp=25.0, humidity=90.0, pressure=995.0, wind_speed=30.0, prev_rain=40.0
        )
        self.assertGreater(rain_high, 10.0, "High rain scenario should predict > 10mm")

        # Dry scenario: Low humidity (25%), High pressure (1020 hPa), 0 prior rain
        rain_low = self.predictor.predict(
            temp=38.0, humidity=25.0, pressure=1020.0, wind_speed=5.0, prev_rain=0.0
        )
        self.assertLess(rain_low, 10.0, "Dry scenario should predict < 10mm")
        print(f"[PASS] ANN Predictor: High Rain = {rain_high}mm | Low Rain = {rain_low}mm")

    def test_fuzzy_risk_engine(self):
        # Extreme rain test
        res_ext = self.fuzzy.evaluate_risk(predicted_rain=75.0, humidity=95.0, pressure=995.0)
        self.assertIn(res_ext['risk_level'], ['High', 'Very High'])
        self.assertGreaterEqual(res_ext['risk_score'], 55.0)

        # Mild rain test
        res_low = self.fuzzy.evaluate_risk(predicted_rain=1.2, humidity=30.0, pressure=1018.0)
        self.assertEqual(res_low['risk_level'], 'Low')
        self.assertLessEqual(res_low['risk_score'], 35.0)
        print(f"[PASS] Fuzzy Risk: Extreme = {res_ext['risk_level']} ({res_ext['risk_score']}) | Mild = {res_low['risk_level']} ({res_low['risk_score']})")

    def test_api_index(self):
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'SmartRain', res.data)
        print("[PASS] GET / returns 200 and loads UI")

    def test_api_predict(self):
        payload = {
            'temperature': 26.0,
            'relative_humidity': 85.0,
            'atmospheric_pressure': 1002.0,
            'wind_speed': 28.0,
            'previous_rainfall': 30.0
        }
        res = self.client.post('/api/predict', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('predicted_rainfall', data)
        self.assertIn('risk_level', data)
        self.assertIn('fired_rules', data)
        self.assertTrue(len(data['fired_rules']) > 0)
        print(f"[PASS] POST /api/predict: {data['predicted_rainfall']}mm -> {data['risk_level']}")

    def test_api_metrics(self):
        res = self.client.get('/api/metrics')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('ann_model', data['metrics'])
        self.assertIn('baseline_model', data['metrics'])
        print(f"[PASS] GET /api/metrics: ANN R2={data['metrics']['ann_model']['r2_score']}")

    def test_api_live_weather(self):
        # Test automated live weather ingestion for coordinates
        res = self.client.get('/api/live-weather?lat=12.9716&lon=77.5946')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('location', data)
        self.assertIn('inputs', data)
        self.assertIn('predicted_rainfall', data)
        self.assertIn('risk_level', data)
        print(f"[PASS] GET /api/live-weather: {data['location']['name']} -> Temp {data['inputs']['temperature']}C, Rain {data['predicted_rainfall']}mm ({data['risk_level']})")

    def test_api_search_city(self):
        res = self.client.get('/api/search-city?q=Mumbai')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertTrue(len(data['results']) > 0)
        self.assertEqual(data['results'][0]['name'], 'Mumbai')
        print(f"[PASS] GET /api/search-city: Found {len(data['results'])} stations for Mumbai")

    def test_api_history_and_export(self):
        res = self.client.get('/api/history')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'success')

        res_csv = self.client.get('/api/export-csv')
        self.assertEqual(res_csv.status_code, 200)
        self.assertEqual(res_csv.mimetype, 'text/csv')
        print("[PASS] History & CSV Export APIs working")

if __name__ == '__main__':
    unittest.main()

