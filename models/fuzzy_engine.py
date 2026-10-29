"""
fuzzy_engine.py - Interpretable Fuzzy Logic Decision Layer for Rainfall Risk Assessment.
Implements a Mamdani Fuzzy Inference System with centroid defuzzification and
rule-firing explainability.
"""

import numpy as np

def trimf(x, abc):
    """Triangular membership function"""
    a, b, c = abc
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    else:
        return (c - x) / (c - b) if c != b else 1.0

def trapmf(x, abcd):
    """Trapezoidal membership function"""
    a, b, c, d = abcd
    if x <= a or x >= d:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a) if b != a else 1.0
    elif b <= x <= c:
        return 1.0
    else:
        return (d - x) / (d - c) if d != c else 1.0

class RainfallFuzzySystem:
    def __init__(self):
        # Membership parameter definitions
        self.rain_mf = {
            'low': ('trap', [0.0, 0.0, 5.0, 15.0]),
            'moderate': ('tri', [10.0, 25.0, 45.0]),
            'heavy': ('tri', [35.0, 55.0, 80.0]),
            'extreme': ('trap', [65.0, 95.0, 200.0, 200.0])
        }
        
        self.humidity_mf = {
            'low': ('trap', [0.0, 0.0, 35.0, 55.0]),
            'moderate': ('tri', [45.0, 65.0, 80.0]),
            'high': ('trap', [70.0, 85.0, 100.0, 100.0])
        }
        
        self.pressure_mf = {
            'low': ('trap', [970.0, 970.0, 1000.0, 1008.0]),
            'normal': ('tri', [1004.0, 1013.0, 1020.0]),
            'high': ('trap', [1016.0, 1024.0, 1040.0, 1040.0])
        }
        
        # Output risk membership functions (domain 0 to 100)
        self.risk_mf = {
            'low': ('trap', [0.0, 0.0, 15.0, 32.0]),
            'moderate': ('tri', [25.0, 45.0, 65.0]),
            'high': ('tri', [55.0, 72.0, 88.0]),
            'very_high': ('trap', [78.0, 90.0, 100.0, 100.0])
        }

    def _calc_mf(self, val, mf_def):
        mf_type, params = mf_def
        if mf_type == 'tri':
            return trimf(val, params)
        elif mf_type == 'trap':
            return trapmf(val, params)
        return 0.0

    def evaluate_risk(self, predicted_rain, humidity, pressure=1013.0):
        """
        Evaluates risk score and category given predicted rainfall (mm),
        relative humidity (%), and optional atmospheric pressure (hPa).
        
        Returns:
            dict containing:
                - risk_score: crisp numeric score (0 to 100)
                - risk_level: 'Low', 'Moderate', 'High', or 'Very High'
                - color: hex code for UI badge
                - advisory: practical recommendation
                - fired_rules: list of rules with membership degrees
                - memberships: breakdown of input memberships
        """
        # 1. Fuzzification
        r_low = self._calc_mf(predicted_rain, self.rain_mf['low'])
        r_mod = self._calc_mf(predicted_rain, self.rain_mf['moderate'])
        r_heavy = self._calc_mf(predicted_rain, self.rain_mf['heavy'])
        r_ext = self._calc_mf(predicted_rain, self.rain_mf['extreme'])

        h_low = self._calc_mf(humidity, self.humidity_mf['low'])
        h_mod = self._calc_mf(humidity, self.humidity_mf['moderate'])
        h_high = self._calc_mf(humidity, self.humidity_mf['high'])

        p_low = self._calc_mf(pressure, self.pressure_mf['low'])
        p_norm = self._calc_mf(pressure, self.pressure_mf['normal'])

        # 2. Rule Base Evaluation (Mamdani AND = MIN)
        rules = [
            # Extreme / Very High risk conditions
            {
                'name': 'Rule 1',
                'rule_text': 'IF Predicted Rain is EXTREME THEN Risk is VERY HIGH',
                'strength': r_ext,
                'consequent': 'very_high'
            },
            {
                'name': 'Rule 2',
                'rule_text': 'IF Rain is HEAVY AND Humidity is HIGH THEN Risk is VERY HIGH',
                'strength': min(r_heavy, h_high),
                'consequent': 'very_high'
            },
            {
                'name': 'Rule 3',
                'rule_text': 'IF Rain is HEAVY AND Pressure is LOW THEN Risk is VERY HIGH',
                'strength': min(r_heavy, p_low),
                'consequent': 'very_high'
            },
            # High risk conditions
            {
                'name': 'Rule 4',
                'rule_text': 'IF Rain is HEAVY AND Humidity is MODERATE THEN Risk is HIGH',
                'strength': min(r_heavy, h_mod),
                'consequent': 'high'
            },
            {
                'name': 'Rule 5',
                'rule_text': 'IF Rain is MODERATE AND Humidity is HIGH THEN Risk is HIGH',
                'strength': min(r_mod, h_high),
                'consequent': 'high'
            },
            {
                'name': 'Rule 6',
                'rule_text': 'IF Rain is MODERATE AND Pressure is LOW THEN Risk is HIGH',
                'strength': min(r_mod, p_low),
                'consequent': 'high'
            },
            # Moderate risk conditions
            {
                'name': 'Rule 7',
                'rule_text': 'IF Rain is MODERATE AND Humidity is MODERATE THEN Risk is MODERATE',
                'strength': min(r_mod, h_mod),
                'consequent': 'moderate'
            },
            {
                'name': 'Rule 8',
                'rule_text': 'IF Rain is LOW AND Humidity is HIGH THEN Risk is MODERATE',
                'strength': min(r_low, h_high),
                'consequent': 'moderate'
            },
            # Low risk conditions
            {
                'name': 'Rule 9',
                'rule_text': 'IF Rain is LOW AND Humidity is LOW THEN Risk is LOW',
                'strength': min(r_low, h_low),
                'consequent': 'low'
            },
            {
                'name': 'Rule 10',
                'rule_text': 'IF Rain is LOW AND Humidity is MODERATE THEN Risk is LOW',
                'strength': min(r_low, h_mod),
                'consequent': 'low'
            }
        ]

        # Aggregate rule activations per output category
        category_strengths = {
            'low': max([r['strength'] for r in rules if r['consequent'] == 'low'] or [0.0]),
            'moderate': max([r['strength'] for r in rules if r['consequent'] == 'moderate'] or [0.0]),
            'high': max([r['strength'] for r in rules if r['consequent'] == 'high'] or [0.0]),
            'very_high': max([r['strength'] for r in rules if r['consequent'] == 'very_high'] or [0.0])
        }

        # 3. Defuzzification via Centroid Discretization
        x_domain = np.linspace(0, 100, 201)
        aggregated = np.zeros_like(x_domain)

        for cat, strength in category_strengths.items():
            if strength > 0.0:
                mf_vals = np.array([self._calc_mf(x, self.risk_mf[cat]) for x in x_domain])
                # Mamdani implication (MIN clipping)
                clipped = np.minimum(strength, mf_vals)
                # MAX aggregation
                aggregated = np.maximum(aggregated, clipped)

        sum_agg = np.sum(aggregated)
        if sum_agg == 0:
            risk_score = 10.0 # Default safe fallback
        else:
            risk_score = float(np.sum(x_domain * aggregated) / sum_agg)

        # Categorize crisp score
        if risk_score < 28.0:
            level = 'Low'
            color = '#10B981' # Emerald Green
            advisory = 'Favorable weather conditions. Normal agricultural and urban operations can continue without disruption.'
        elif risk_score < 56.0:
            level = 'Moderate'
            color = '#F59E0B' # Amber Yellow
            advisory = 'Moderate precipitation likely. Farmers should monitor irrigation levels; municipal drainage teams should maintain readiness.'
        elif risk_score < 78.0:
            level = 'High'
            color = '#F97316' # Orange
            advisory = 'High rainfall risk. Potential waterlogging in low-lying areas. Suspend outdoor crop harvesting and prepare flood gates.'
        else:
            level = 'Very High'
            color = '#EF4444' # Crimson Red
            advisory = 'Severe weather alert. Imminent danger of heavy inundation, flash flooding, and transport disruption. Emergency protocols recommended.'

        # Extract active/fired rules (strength > 0.05)
        active_rules = [
            {
                'name': r['name'],
                'text': r['rule_text'],
                'strength': round(float(r['strength']), 3),
                'consequent': r['consequent'].replace('_', ' ').title()
            }
            for r in rules if r['strength'] >= 0.05
        ]
        active_rules.sort(key=lambda x: x['strength'], reverse=True)

        return {
            'risk_score': round(risk_score, 1),
            'risk_level': level,
            'color': color,
            'advisory': advisory,
            'fired_rules': active_rules,
            'memberships': {
                'rain': {
                    'low': round(float(r_low), 2),
                    'moderate': round(float(r_mod), 2),
                    'heavy': round(float(r_heavy), 2),
                    'extreme': round(float(r_ext), 2)
                },
                'humidity': {
                    'low': round(float(h_low), 2),
                    'moderate': round(float(h_mod), 2),
                    'high': round(float(h_high), 2)
                }
            }
        }

if __name__ == '__main__':
    fuzzy = RainfallFuzzySystem()
    print("Testing Fuzzy Engine:")
    for rain, hum, pres in [(2.5, 40, 1016), (28.0, 75, 1008), (68.0, 92, 998)]:
        res = fuzzy.evaluate_risk(rain, hum, pres)
        print(f"Rain: {rain}mm, Hum: {hum}%, Pres: {pres}hPa -> Risk: {res['risk_score']} ({res['risk_level']})")
        print(f"Top Rule: {res['fired_rules'][0] if res['fired_rules'] else 'None'}")
