"""
generate_data.py - Generates realistic meteorological historical weather records
for training the SmartRain Artificial Neural Network.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_historical_weather(num_samples=2500, random_seed=42):
    np.random.seed(random_seed)
    
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=i % 365) for i in range(num_samples)]
    
    # Generate seasonal cycles
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    
    # Monsoon season simulation (peak around days 160 to 270 - June to September)
    monsoon_factor = np.exp(-((day_of_year - 215) ** 2) / (2 * 45 ** 2))
    
    # Temperature (°C): 18°C to 42°C, cooler during peak rain, hotter before monsoon
    temp_base = 28 + 7 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    temperature = temp_base - 5 * monsoon_factor + np.random.normal(0, 2.5, num_samples)
    temperature = np.clip(temperature, 14.0, 45.0)
    
    # Relative Humidity (%): 25% to 98%, heavily increased by monsoon
    humidity_base = 45 + 38 * monsoon_factor
    humidity = humidity_base + np.random.normal(0, 8.0, num_samples)
    humidity = np.clip(humidity, 20.0, 99.0)
    
    # Atmospheric Pressure (hPa): Normal ~1012 hPa. Drops during storm depressions (down to 992-1002 hPa)
    pressure_base = 1013 - 10 * monsoon_factor
    pressure = pressure_base + np.random.normal(0, 3.5, num_samples)
    pressure = np.clip(pressure, 990.0, 1026.0)
    
    # Wind speed (km/h): 4 to 55 km/h
    wind_speed = 10 + 12 * monsoon_factor + np.random.exponential(5.0, num_samples)
    wind_speed = np.clip(wind_speed, 2.0, 60.0)
    
    # Previous day's rainfall (mm)
    prev_rain_prob = 0.15 + 0.55 * monsoon_factor
    prev_rain_mask = np.random.rand(num_samples) < prev_rain_prob
    previous_rainfall = np.where(
        prev_rain_mask,
        np.random.gamma(shape=1.8, scale=12.0, size=num_samples),
        0.0
    )
    previous_rainfall = np.clip(previous_rainfall, 0.0, 150.0)
    
    # Target rainfall calculation (Physical meteorological nonlinear relationship):
    # - High humidity + Low pressure + High prior rain + Moderate/High wind = High rain
    pressure_deficit = np.maximum(0, 1013 - pressure)
    humidity_excess = np.maximum(0, humidity - 60)
    
    latent_rain_score = (
        0.035 * (humidity_excess ** 1.8) +
        1.8 * pressure_deficit +
        0.18 * previous_rainfall +
        0.25 * wind_speed +
        -0.3 * (temperature - 26) ** 2 / 10
    )
    
    # Rain occurrence probability
    rain_prob = 1.0 / (1.0 + np.exp(-(latent_rain_score - 12) / 4))
    rain_mask = np.random.rand(num_samples) < rain_prob
    
    rainfall_amount = np.where(
        rain_mask,
        latent_rain_score * 0.8 + np.random.exponential(scale=6.0, size=num_samples),
        0.0
    )
    rainfall_amount = np.clip(rainfall_amount, 0.0, 180.0)
    
    df = pd.DataFrame({
        'date': [d.strftime('%Y-%m-%d') for d in dates],
        'temperature': np.round(temperature, 1),
        'relative_humidity': np.round(humidity, 1),
        'atmospheric_pressure': np.round(pressure, 1),
        'wind_speed': np.round(wind_speed, 1),
        'previous_rainfall': np.round(previous_rainfall, 1),
        'rainfall_amount': np.round(rainfall_amount, 1)
    })
    
    return df

if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(data_dir, 'historical_weather.csv')
    df = generate_historical_weather(num_samples=3000)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} historical weather records saved to {output_path}")
    print(df.describe())

