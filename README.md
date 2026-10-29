# SmartRain: Autonomous Meteorological Intelligence & Decision-Support Workstation

A production-grade, software-based decision-support system integrating an **Artificial Neural Network (ANN)** for rainfall regression and an interpretable **Fuzzy Logic (Mamdani Inference)** engine for operational risk categorization.

Trained on **7,671 authentic real-world meteorological observations** and powered by **automated geospatial live satellite ingestion**.

---

## 🌟 Key Capabilities

1. **Autonomous Geospatial Ingestion (No Manual Inputs Required)**:
   * Uses device GPS (HTML5 Geolocation API) to pinpoint the user's location.
   * Ingests real-time atmospheric readings from the **Open-Meteo Global Satellite & Radar API** (backed by DWD, ECMWF, and NOAA).
   * Automatically extracts Dry-Bulb Temperature, Relative Humidity, Barometric Pressure, 10m Wind Velocity, and Past 24h Precipitation.
   * Includes a worldwide city search bar to monitor any weather station globally.

2. **Authentic Real-World Training Dataset (ECMWF ERA5 Archive)**:
   * **7,671 real-world daily observation records** collected across 7 full years (2018–2024).
   * Spans three diverse meteorological zones:
     * **Mumbai Coastal Station**: Coastal Southwest Monsoon (heavy inundation & squalls, max rain: 199.9 mm).
     * **Bengaluru Plateau Station**: Southern Deccan Plateau (convective thunderstorms & moderate showers, max rain: 69.6 mm).
     * **Delhi Subtropical Station**: Northern Subtropical Monsoon (high thermal gradient, max rain: 122.7 mm).

3. **Artificial Neural Network (ANN) Regressor**:
   * Multi-Layer Perceptron (MLP) architecture with 3 hidden layers `[64 -> 32 -> 16]` and ReLU activations.
   * Achieves **$MAE = 2.626\text{ mm}$** (vs. Baseline Dummy $5.652\text{ mm}$, a **53.5% error reduction**).
   * Achieves **$R^2 = 0.5502$** on real-world chaotic meteorological data.

4. **Explainable Mamdani Fuzzy Inference System (FIS)**:
   * Converts numerical rainfall forecasts into human-centered risk tiers: **Low**, **Moderate**, **High**, **Critical / Very High**.
   * Interactive **HTML5 Canvas continuous curves** plotting piecewise triangular & trapezoidal membership functions with dynamic crosshair intersections ($\alpha$-cuts).
   * **2D Meteorological Risk Matrix** plotting precipitation rate against soil/humidity saturation.

---

## 📊 Experimental Evaluation vs Baseline

Evaluated on 7,671 authentic observations with an 80/20 train/test split (6,136 training samples / 1,535 test samples):

| Metric | ANN (Multi-Layer Perceptron) | Historical Baseline | Improvement |
| :--- | :---: | :---: | :---: |
| **MAE (Mean Absolute Error)** | **2.626 mm** | 5.652 mm | **-53.54% Error** |
| **RMSE (Root Mean Sq. Error)** | **8.577 mm** | 12.793 mm | **-32.96% Error** |
| **$R^2$ Score (Goodness of Fit)** | **0.5502** | -0.0007 | **+0.5509** |

---

## 🛠️ Tech Stack & Citations

* **Backend & ML**: Python 3.14+, Flask 3.1, Scikit-learn, NumPy, Pandas, Joblib
* **Data Ingestion**: ECMWF ERA5 Reanalysis Archive & Open-Meteo Satellite Feed
* **Soft Computing Engine**: Custom Mamdani Fuzzy Inference System & Multi-Layer Perceptron
* **Frontend**: Vanilla JavaScript (ES6+), HTML5 Canvas, SVG Arc Meter, Modern Industrial Dark CSS
* **Database**: SQLite3 persistence with simulation audit log & CSV export

### References & Citations for Academic Defense
1. **Abu Saleh, M. et al. (2024)**: *A comprehensive review towards resilient rainfall forecasting models using artificial intelligence techniques.* Green Technologies and Sustainability.
2. **Ruiz-Hurtado, A.F. et al. (2024)**: *Neural Networks and Fuzzy Logic-Based Approaches for Precipitation Estimation: A Systematic Review.* Ingeniería e Investigación.
3. **ECMWF (European Centre for Medium-Range Weather Forecasts)**: *ERA5 Global Atmospheric Reanalysis Data (2018–2024).*
4. **India Meteorological Department (IMD)**: *Monsoon Rainfall & Regional Meteorological Standards.*

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Fetch Real Historical Data & Retrain Model (Optional - Already Pre-trained)
```bash
python data/fetch_real_data.py
python models/ann_model.py
```

### 3. Run Web Workstation
```bash
python run.py
```
Open your web browser at:
```
http://localhost:5000
```
