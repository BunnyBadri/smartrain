"""
fetch_real_data.py - Ingests authentic real-world meteorological historical observations
from the ECMWF ERA5 / Open-Meteo Global Historical Reanalysis Archive (2018-2024).
"""

import os
import requests
import pandas as pd
import numpy as np

STATIONS = [
    {
        "name": "Mumbai Coastal Station",
        "latitude": 19.0760,
        "longitude": 72.8777,
        "climate": "Coastal Southwest Monsoon (Heavy Inundation & Squalls)"
    },
    {
        "name": "Bengaluru Plateau Station",
        "latitude": 12.9716,
        "longitude": 77.5946,
        "climate": "Southern Deccan Plateau (Convective Thunderstorms & Moderate Rain)"
    },
    {
        "name": "Delhi Subtropical Station",
        "latitude": 28.6139,
        "longitude": 77.2090,
        "climate": "Northern Subtropical Monsoon (High Thermal Gradient)"
    }
]

START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

def fetch_station_data(station):
    name = station["name"]
    lat = station["latitude"]
    lon = station["longitude"]
    
    print(f"Fetching real historical records for {name} ({lat}, {lon}) from {START_DATE} to {END_DATE}...")
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={START_DATE}&end_date={END_DATE}"
        f"&daily=temperature_2m_mean,relative_humidity_2m_mean,surface_pressure_mean,wind_speed_10m_max,precipitation_sum"
        f"&timezone=Asia%2FKolkata"
    )
    
    res = requests.get(url, timeout=45)
    res.raise_for_status()
    payload = res.json()
    daily = payload.get("daily", {})
    
    df = pd.DataFrame({
        "date": daily["time"],
        "station": name,
        "temperature": daily["temperature_2m_mean"],
        "relative_humidity": daily["relative_humidity_2m_mean"],
        "atmospheric_pressure": daily["surface_pressure_mean"],
        "wind_speed": daily["wind_speed_10m_max"],
        "rainfall_amount": daily["precipitation_sum"]
    })
    
    # Calculate prior 24h rainfall (shifted by 1 day per station)
    df["previous_rainfall"] = df["rainfall_amount"].shift(1).fillna(0.0)
    
    # Round to 1 decimal place
    df["temperature"] = df["temperature"].round(1)
    df["relative_humidity"] = df["relative_humidity"].round(1)
    df["atmospheric_pressure"] = df["atmospheric_pressure"].round(1)
    df["wind_speed"] = df["wind_speed"].round(1)
    df["previous_rainfall"] = df["previous_rainfall"].round(1)
    df["rainfall_amount"] = df["rainfall_amount"].round(1)
    
    print(f"  -> Retrieved {len(df)} days of real records for {name}. Max Rain: {df['rainfall_amount'].max()} mm")
    return df

def build_real_historical_dataset():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(data_dir, "historical_weather.csv")
    
    station_dfs = []
    for s in STATIONS:
        df_s = fetch_station_data(s)
        station_dfs.append(df_s)
        
    combined_df = pd.concat(station_dfs, ignore_index=True)
    
    # Shuffle to ensure diverse meteorological distribution across folds
    combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Save clean dataset
    combined_df.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Saved {len(combined_df)} real-life meteorological records to {output_path}")
    print("\nDataset Summary Statistics:")
    print(combined_df[["temperature", "relative_humidity", "atmospheric_pressure", "wind_speed", "previous_rainfall", "rainfall_amount"]].describe())
    return combined_df

if __name__ == "__main__":
    build_real_historical_dataset()
