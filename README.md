# SmartRain: Intelligent Rainfall Prediction & Risk Assessment

A software-based decision-support system integrating **Artificial Neural Networks (ANN)** and **Fuzzy Logic (Mamdani Inference)** for meteorological forecasting and risk categorization.

Developed in accordance with the Soft Computing Course / Review II & Final Review academic guidelines.

---

## 🌟 Key Features

1. **Artificial Neural Network (ANN) Regressor**:
   * Multi-Layer Perceptron (MLP) trained on 3,000 meteorological observation records.
   * Inputs: *Temperature*, *Relative Humidity*, *Atmospheric Pressure*, *Wind Speed*, and *Prior 24h Rainfall*.
   * Output: Numerical Rainfall Prediction in millimeters ($mm$).
   * Performance: Achieves an **$R^2$ score of 0.7651** and reduces MAE by **61.3%** compared to historical-average baseline.

2. **Interpretable Fuzzy Logic Risk Layer**:
   * Mamdani Fuzzy Inference System evaluating rainfall volume and atmospheric factors.
   * Converts numerical forecasts into human-understandable risk levels: **Low**, **Moderate**, **High**, **Very High**.
   * Transparent rule trace displaying active linguistic rules and degree of membership ($\alpha$).

3. **Interactive Web Application (Dashboard)**:
   * Modern, responsive interface with real-time synchronized range sliders and presets (*Monsoon Cloudburst*, *Dry Sunny Day*, *Pre-Monsoon Squall*, *Overcast Showers*).
   * Visual gauge meter and dynamic color-coded alert badges.
   * Actionable operational advisories for agricultural and municipal planning.
   * Persistent SQLite simulation audit log with CSV export capability.

---

## 🛠️ Tech Stack

* **Backend & ML**: Python 3.14+, Flask 3.1, Scikit-learn, NumPy, Pandas, Joblib
* **Soft Computing Engine**: Custom Mamdani Fuzzy Inference System & Multi-Layer Perceptron
* **Storage**: SQLite3
* **Frontend**: HTML5, CSS3 (Modern Glassmorphic Dark UI), Vanilla JavaScript (ES6+)

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Dependencies
Ensure Python 3.10+ is installed. Install required packages:
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset & Train ANN Model (Optional - Already pre-trained)
```bash
python data/generate_data.py
python models/ann_model.py
```

### 3. Run the Web Server
```bash
python run.py
```
Open your web browser and navigate to:
```
http://localhost:5000
```

### 4. Run Automated Test Suite
```bash
python test_app.py
```

---

## 📊 Experimental Evaluation vs Baseline

| Metric | ANN (Multi-Layer Perceptron) | Historical Baseline | Improvement |
| :--- | :---: | :---: | :---: |
| **MAE (Mean Absolute Error)** | **4.682 mm** | 12.107 mm | **-61.32% Error** |
| **RMSE (Root Mean Sq. Error)** | **6.787 mm** | 14.018 mm | **-51.58% Error** |
| **$R^2$ Score (Goodness of Fit)** | **0.7651** | -0.0022 | **+0.7673** |

---

## 💡 Soft Computing Justification (For Viva Defense)

* **Why ANN?**
  * Meteorological parameters interact in complex, non-linear ways (e.g. pressure drops + humidity spikes). ANNs excel at universal non-linear function approximation from historical records.
* **Why Fuzzy Logic?**
  * Machine learning models are numerical "black boxes". Fuzzy logic uses linguistic rules (e.g., *“IF Rain is HEAVY AND Humidity is HIGH THEN Risk is VERY HIGH”*) to provide explainable, human-centered decision support for non-technical stakeholders (farmers, disaster managers).
