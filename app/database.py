"""
database.py - SQLite storage for SmartRain simulation history and logs.
"""

import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'smartrain_history.db')

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                temperature REAL NOT NULL,
                relative_humidity REAL NOT NULL,
                atmospheric_pressure REAL NOT NULL,
                wind_speed REAL NOT NULL,
                previous_rainfall REAL NOT NULL,
                predicted_rainfall REAL NOT NULL,
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                top_rule TEXT
            )
        ''')
        conn.commit()
    finally:
        conn.close()

def save_prediction(temp, humidity, pressure, wind_speed, prev_rain, pred_rain, risk_score, risk_level, top_rule=""):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO prediction_history (
                timestamp, temperature, relative_humidity, atmospheric_pressure,
                wind_speed, previous_rainfall, predicted_rainfall,
                risk_score, risk_level, top_rule
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            temp, humidity, pressure, wind_speed, prev_rain,
            pred_rain, risk_score, risk_level, top_rule
        ))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()

def get_recent_predictions(limit=20):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM prediction_history
            ORDER BY id DESC
            LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

def clear_history():
    conn = get_connection()
    try:
        conn.execute('DELETE FROM prediction_history')
        conn.commit()
    finally:
        conn.close()
