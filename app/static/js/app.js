// SmartRain Frontend Controller

const PRESETS = {
  monsoon: {
    temp: 24.5,
    humidity: 92,
    pressure: 994,
    wind: 36,
    prevRain: 58
  },
  squall: {
    temp: 33.5,
    humidity: 78,
    pressure: 1001,
    wind: 44,
    prevRain: 8
  },
  moderate: {
    temp: 27.0,
    humidity: 68,
    pressure: 1010,
    wind: 18,
    prevRain: 14
  },
  sunny: {
    temp: 37.5,
    humidity: 28,
    pressure: 1018,
    wind: 10,
    prevRain: 0
  }
};

function syncVal(inputId, displayId, suffix = '') {
  const val = parseFloat(document.getElementById(inputId).value).toFixed(1);
  document.getElementById(displayId).textContent = val;
}

function applyPreset(presetKey) {
  const data = PRESETS[presetKey];
  if (!data) return;

  document.getElementById('tempInput').value = data.temp;
  document.getElementById('tempVal').textContent = data.temp.toFixed(1);

  document.getElementById('humidityInput').value = data.humidity;
  document.getElementById('humidityVal').textContent = data.humidity.toFixed(1);

  document.getElementById('pressureInput').value = data.pressure;
  document.getElementById('pressureVal').textContent = data.pressure.toFixed(1);

  document.getElementById('windInput').value = data.wind;
  document.getElementById('windVal').textContent = data.wind.toFixed(1);

  document.getElementById('prevRainInput').value = data.prevRain;
  document.getElementById('prevRainVal').textContent = data.prevRain.toFixed(1);

  // Automatically execute prediction for seamless viva demo
  runPrediction();
}

async function runPrediction() {
  const submitBtn = document.getElementById('submitBtn');
  const spinner = document.getElementById('spinner');
  const btnText = submitBtn.querySelector('.btn-text');

  const payload = {
    temperature: parseFloat(document.getElementById('tempInput').value),
    relative_humidity: parseFloat(document.getElementById('humidityInput').value),
    atmospheric_pressure: parseFloat(document.getElementById('pressureInput').value),
    wind_speed: parseFloat(document.getElementById('windInput').value),
    previous_rainfall: parseFloat(document.getElementById('prevRainInput').value)
  };

  try {
    submitBtn.disabled = true;
    spinner.style.display = 'block';
    btnText.textContent = 'Computing Soft Computing Inference...';

    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    if (data.status === 'success') {
      updateUI(data);
      loadHistory();
    } else {
      alert('Error during prediction: ' + (data.message || 'Unknown error'));
    }
  } catch (err) {
    console.error('Request failed:', err);
    alert('Server communication error. Check console for details.');
  } finally {
    submitBtn.disabled = false;
    spinner.style.display = 'none';
    btnText.textContent = 'Execute Soft Computing Pipeline';
  }
}

function updateUI(data) {
  const rain = data.predicted_rainfall;
  const score = data.risk_score;
  const level = data.risk_level;
  const color = data.color;

  // 1. ANN Predicted Rainfall Display
  const rainEl = document.getElementById('predRainValue');
  rainEl.textContent = rain.toFixed(1);

  // Rainfall descriptor
  const descEl = document.getElementById('predRainDesc');
  if (rain < 2.5) {
    descEl.textContent = 'No significant precipitation expected (< 2.5 mm)';
  } else if (rain < 15.0) {
    descEl.textContent = 'Light showers / drizzle (2.5 - 15 mm)';
  } else if (rain < 40.0) {
    descEl.textContent = 'Moderate steady rainfall (15 - 40 mm)';
  } else if (rain < 70.0) {
    descEl.textContent = 'Heavy downpour (40 - 70 mm)';
  } else {
    descEl.textContent = 'Torrential cloudburst / extreme rain (> 70 mm)';
  }

  // Gauge bar fill (0 to 80mm normalized)
  const pct = Math.min(100, Math.max(0, (rain / 80) * 100));
  document.getElementById('gaugeFill').style.width = pct + '%';

  // 2. Fuzzy Risk Assessment Badge
  const badge = document.getElementById('riskBadge');
  badge.textContent = level.toUpperCase() + ' RISK';
  badge.style.color = color;
  badge.style.borderColor = color;
  badge.style.background = `${color}20`; // 12% opacity
  badge.style.boxShadow = `0 0 20px ${color}40`;

  document.getElementById('riskScore').textContent = score.toFixed(1);

  // Advisory Alert Box
  const advisoryBox = document.getElementById('advisoryBox');
  advisoryBox.style.borderColor = `${color}40`;
  document.getElementById('advisoryText').textContent = data.advisory;

  // 3. Fuzzy Explainability: Active Rules
  const rulesList = document.getElementById('rulesList');
  if (data.fired_rules && data.fired_rules.length > 0) {
    rulesList.innerHTML = data.fired_rules.map(r => `
      <div class="rule-item">
        <span class="rule-text"><strong>${r.name}:</strong> ${r.text}</span>
        <span class="rule-strength">α = ${r.strength}</span>
      </div>
    `).join('');
  } else {
    rulesList.innerHTML = '<div class="placeholder-text">Baseline safe conditions active.</div>';
  }

  // 4. Input Membership Breakdown
  renderMembershipBars('rainMemberships', data.memberships.rain, '#60a5fa');
  renderMembershipBars('humidityMemberships', data.memberships.humidity, '#06b6d4');
}

function renderMembershipBars(containerId, mfs, color) {
  const container = document.getElementById(containerId);
  if (!mfs) return;

  container.innerHTML = Object.entries(mfs).map(([name, val]) => {
    const pct = (val * 100).toFixed(0);
    return `
      <div class="mf-row">
        <span class="mf-name">${name}</span>
        <div class="mf-bar-track">
          <div class="mf-bar-fill" style="width: ${pct}%; background: ${color};"></div>
        </div>
        <span class="mf-val">${val}</span>
      </div>
    `;
  }).join('');
}

async function loadMetrics() {
  try {
    const res = await fetch('/api/metrics');
    const json = await res.json();
    if (json.status === 'success') {
      const m = json.metrics;
      document.getElementById('annMae').textContent = `${m.ann_model.mae} mm`;
      document.getElementById('maeImprovement').textContent = `Baseline MAE: ${m.baseline_model.mae} mm`;

      document.getElementById('annRmse').textContent = `${m.ann_model.rmse} mm`;
      document.getElementById('rmseImprovement').textContent = `Baseline RMSE: ${m.baseline_model.rmse} mm`;

      document.getElementById('annR2').textContent = m.ann_model.r2_score;
      document.getElementById('overallImprovement').textContent = `-${m.improvement.mae_reduction_percent}% Error`;
    }
  } catch (e) {
    console.error('Could not load metrics:', e);
  }
}

async function loadHistory() {
  try {
    const res = await fetch('/api/history');
    const json = await res.json();
    if (json.status === 'success') {
      const tbody = document.getElementById('historyTableBody');
      if (!json.history || json.history.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" class="placeholder-text">No simulations run yet.</td></tr>';
        return;
      }

      tbody.innerHTML = json.history.map(row => {
        let badgeColor = '#10b981';
        if (row.risk_level === 'Moderate') badgeColor = '#f59e0b';
        else if (row.risk_level === 'High') badgeColor = '#f97316';
        else if (row.risk_level === 'Very High') badgeColor = '#ef4444';

        return `
          <tr>
            <td>${row.timestamp.split(' ')[1] || row.timestamp}</td>
            <td>${row.temperature}°C</td>
            <td>${row.relative_humidity}%</td>
            <td>${row.atmospheric_pressure} hPa</td>
            <td>${row.wind_speed} km/h</td>
            <td>${row.previous_rainfall} mm</td>
            <td><strong>${row.predicted_rainfall} mm</strong></td>
            <td>${row.risk_score}</td>
            <td>
              <span class="table-badge" style="background: ${badgeColor}20; color: ${badgeColor}; border: 1px solid ${badgeColor}60;">
                ${row.risk_level}
              </span>
            </td>
          </tr>
        `;
      }).join('');
    }
  } catch (e) {
    console.error('Could not load history:', e);
  }
}

async function clearHistoryLog() {
  if (!confirm('Clear all simulation records?')) return;
  try {
    await fetch('/api/history/clear', { method: 'POST' });
    loadHistory();
  } catch (e) {
    console.error(e);
  }
}

// Initial Boot
document.addEventListener('DOMContentLoaded', () => {
  loadMetrics();
  loadHistory();
  // Run an initial prediction with default values
  runPrediction();
});
