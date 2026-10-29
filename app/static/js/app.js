// SmartRain Meteorological Intelligence Workstation Engine

const PRESETS = {
  monsoon: { temp: 24.5, humidity: 92.0, pressure: 994.0, wind: 36.0, prevRain: 58.0 },
  squall: { temp: 33.5, humidity: 78.0, pressure: 1001.0, wind: 44.0, prevRain: 8.0 },
  moderate: { temp: 27.0, humidity: 68.0, pressure: 1010.0, wind: 18.0, prevRain: 14.0 },
  sunny: { temp: 37.5, humidity: 28.0, pressure: 1018.0, wind: 10.0, prevRain: 0.0 }
};

let currentPrediction = null;
let debounceTimer = null;

// Mathematical Membership Formulas
function trimf(x, [a, b, c]) {
  if (x <= a || x >= c) return 0.0;
  if (a < x && x <= b) return b !== a ? (x - a) / (b - a) : 1.0;
  return c !== b ? (c - x) / (c - b) : 1.0;
}

function trapmf(x, [a, b, c, d]) {
  if (x <= a || x >= d) return 0.0;
  if (a < x && x < b) return b !== a ? (x - a) / (b - a) : 1.0;
  if (b <= x && x <= c) return 1.0;
  return d !== c ? (d - x) / (d - c) : 1.0;
}

// Tab Switching
function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-tab') === tabId);
  });
  document.querySelectorAll('.tab-content').forEach(content => {
    content.classList.toggle('active', content.id === tabId);
  });

  if (tabId === 'tab-fuzzy' && currentPrediction) {
    setTimeout(renderAllFuzzyCurves, 50);
  }
}

// Input Synchronization & Micro-Steppers
function syncFromSlider(sliderId, numId, decimals = 1) {
  const val = parseFloat(document.getElementById(sliderId).value);
  document.getElementById(numId).value = val.toFixed(decimals);
  triggerStream();
}

function syncFromNum(sliderId, numId, decimals = 1) {
  let val = parseFloat(document.getElementById(numId).value);
  const slider = document.getElementById(sliderId);
  if (isNaN(val)) val = parseFloat(slider.value);
  val = Math.min(parseFloat(slider.max), Math.max(parseFloat(slider.min), val));
  slider.value = val;
  document.getElementById(numId).value = val.toFixed(decimals);
  triggerStream();
}

function stepVal(sliderId, numId, delta, min, max, decimals = 1) {
  let val = parseFloat(document.getElementById(numId).value) + delta;
  val = Math.min(max, Math.max(min, val));
  document.getElementById(sliderId).value = val;
  document.getElementById(numId).value = val.toFixed(decimals);
  triggerStream();
}

function triggerStream() {
  if (document.getElementById('realtimeToggle').checked) {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(runInference, 180);
  }
}

function applyPreset(presetKey) {
  const p = PRESETS[presetKey];
  if (!p) return;

  setInputValue('tempInput', 'tempNum', p.temp);
  setInputValue('humidityInput', 'humidityNum', p.humidity);
  setInputValue('pressureInput', 'pressureNum', p.pressure);
  setInputValue('windInput', 'windNum', p.wind);
  setInputValue('prevRainInput', 'prevRainNum', p.prevRain);

  runInference();
}

function setInputValue(sliderId, numId, val) {
  document.getElementById(sliderId).value = val;
  document.getElementById(numId).value = val.toFixed(1);
}

// Main Soft Computing Inference Routine
async function runInference() {
  const t0 = performance.now();
  const payload = {
    temperature: parseFloat(document.getElementById('tempNum').value),
    relative_humidity: parseFloat(document.getElementById('humidityNum').value),
    atmospheric_pressure: parseFloat(document.getElementById('pressureNum').value),
    wind_speed: parseFloat(document.getElementById('windNum').value),
    previous_rainfall: parseFloat(document.getElementById('prevRainNum').value)
  };

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    const t1 = performance.now();

    document.getElementById('latencyDisplay').textContent = `${Math.round(t1 - t0)} ms`;

    if (data.status === 'success') {
      currentPrediction = data;
      updateConsoleDisplays(data);
      updateRulesTable(data.fired_rules);
      renderAllFuzzyCurves();
      loadAuditLog();
    }
  } catch (err) {
    console.error('Inference pipeline error:', err);
  }
}

// Update Console Displays
function updateConsoleDisplays(data) {
  const rain = data.predicted_rainfall;
  const score = data.risk_score;
  const level = data.risk_level;
  const color = data.color;

  // 1. Arc Gauge
  document.getElementById('predRainVal').textContent = rain.toFixed(1);
  const arcLength = 235.6; // Circumference for r=75
  const clampedRain = Math.min(100, Math.max(0, rain));
  const dashOffset = arcLength - (clampedRain / 100) * arcLength;
  const meter = document.getElementById('gaugeMeter');
  meter.style.strokeDashoffset = dashOffset;
  meter.style.stroke = color;

  // Badge Class
  const classBadge = document.getElementById('rainClassBadge');
  if (rain < 2.5) {
    classBadge.textContent = 'DRY / TRACE PRECIP';
    classBadge.style.color = '#94a3b8';
  } else if (rain < 15.0) {
    classBadge.textContent = 'LIGHT PRECIPITATION';
    classBadge.style.color = '#38bdf8';
  } else if (rain < 40.0) {
    classBadge.textContent = 'STEADY SHOWERS';
    classBadge.style.color = '#f59e0b';
  } else if (rain < 70.0) {
    classBadge.textContent = 'HEAVY DOWNPOUR';
    classBadge.style.color = '#f97316';
  } else {
    classBadge.textContent = 'SEVERE CLOUDBURST';
    classBadge.style.color = '#ef4444';
  }

  // 2. 2D Risk Matrix Active Cell
  document.querySelectorAll('.matrix-cell').forEach(cell => {
    cell.classList.remove('active-cell', 'cell-emerald', 'cell-amber', 'cell-orange', 'cell-red');
  });

  let rainTier = 'low';
  if (rain >= 65.0) rainTier = 'ext';
  else if (rain >= 35.0) rainTier = 'heavy';
  else if (rain >= 12.0) rainTier = 'mod';

  const hum = data.inputs.relative_humidity;
  let humTier = 'low';
  if (hum >= 75.0) humTier = 'high';
  else if (hum >= 50.0) humTier = 'mod';

  const activeCellId = `cell-${rainTier}-${humTier}`;
  const cell = document.getElementById(activeCellId);
  if (cell) {
    cell.classList.add('active-cell');
    if (level === 'Low') cell.classList.add('cell-emerald');
    else if (level === 'Moderate') cell.classList.add('cell-amber');
    else if (level === 'High') cell.classList.add('cell-orange');
    else cell.classList.add('cell-red');
  }

  // 3. Risk Classification Banner & Crisp Score
  const badge = document.getElementById('riskBadgeLarge');
  badge.textContent = `${level.toUpperCase()} RISK`;
  badge.style.color = color;
  badge.style.borderColor = color;
  badge.style.background = `${color}18`;

  document.getElementById('crispScoreNum').textContent = score.toFixed(1);
  document.getElementById('crispScoreNum').style.color = color;

  // 4. Protocol Advisory
  const pBox = document.getElementById('protocolBox');
  pBox.style.borderLeftColor = color;
  document.getElementById('protocolText').textContent = data.advisory;
}

// Continuous HTML5 Canvas Mathematical Curve Drawing
function renderAllFuzzyCurves() {
  if (!currentPrediction) return;
  drawRainfallCurves(currentPrediction.predicted_rainfall);
  drawHumidityCurves(currentPrediction.inputs.relative_humidity);
}

function drawRainfallCurves(currentVal) {
  const canvas = document.getElementById('canvasRain');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  document.getElementById('canvasRainValue').textContent = `${currentVal.toFixed(1)} mm`;

  const paddingLeft = 35;
  const paddingRight = 15;
  const paddingTop = 15;
  const paddingBottom = 25;
  const plotW = w - paddingLeft - paddingRight;
  const plotH = h - paddingTop - paddingBottom;

  const xToCanvas = (val) => paddingLeft + (Math.min(100, Math.max(0, val)) / 100) * plotW;
  const yToCanvas = (mu) => paddingTop + (1.0 - mu) * plotH;

  // Grid lines
  ctx.strokeStyle = '#182436';
  ctx.lineWidth = 1;
  ctx.beginPath();
  [0, 0.25, 0.5, 0.75, 1.0].forEach(mu => {
    const y = yToCanvas(mu);
    ctx.moveTo(paddingLeft, y);
    ctx.lineTo(w - paddingRight, y);
  });
  ctx.stroke();

  // Y-axis labels
  ctx.fillStyle = '#64748b';
  ctx.font = '10px JetBrains Mono';
  ctx.textAlign = 'right';
  ctx.fillText('1.0', paddingLeft - 6, yToCanvas(1.0) + 4);
  ctx.fillText('0.5', paddingLeft - 6, yToCanvas(0.5) + 4);
  ctx.fillText('0.0', paddingLeft - 6, yToCanvas(0.0) + 4);

  // X-axis labels
  ctx.textAlign = 'center';
  [0, 20, 40, 60, 80, 100].forEach(val => {
    ctx.fillText(`${val}`, xToCanvas(val), h - 8);
  });

  const curves = [
    { name: 'low', fn: (x) => trapmf(x, [0, 0, 5, 15]), color: '#60a5fa' },
    { name: 'moderate', fn: (x) => trimf(x, [10, 25, 45]), color: '#38bdf8' },
    { name: 'heavy', fn: (x) => trimf(x, [35, 55, 80]), color: '#f59e0b' },
    { name: 'extreme', fn: (x) => trapmf(x, [65, 95, 100, 100]), color: '#ef4444' }
  ];

  curves.forEach(c => {
    ctx.strokeStyle = c.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let px = 0; px <= plotW; px += 2) {
      const xVal = (px / plotW) * 100;
      const mu = c.fn(xVal);
      const cx = paddingLeft + px;
      const cy = yToCanvas(mu);
      if (px === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }
    ctx.stroke();
  });

  // Vertical Crosshair Indicator at currentVal
  const cx = xToCanvas(currentVal);
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(cx, paddingTop);
  ctx.lineTo(cx, paddingTop + plotH);
  ctx.stroke();
  ctx.setLineDash([]);

  // Intersection Dots
  curves.forEach(c => {
    const mu = c.fn(currentVal);
    if (mu > 0.02) {
      const cy = yToCanvas(mu);
      ctx.fillStyle = c.color;
      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  });
}

function drawHumidityCurves(currentVal) {
  const canvas = document.getElementById('canvasHumidity');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  document.getElementById('canvasHumidityValue').textContent = `${currentVal.toFixed(1)} %`;

  const paddingLeft = 35;
  const paddingRight = 15;
  const paddingTop = 15;
  const paddingBottom = 25;
  const plotW = w - paddingLeft - paddingRight;
  const plotH = h - paddingTop - paddingBottom;

  const xToCanvas = (val) => paddingLeft + (Math.min(100, Math.max(0, val)) / 100) * plotW;
  const yToCanvas = (mu) => paddingTop + (1.0 - mu) * plotH;

  // Grid lines
  ctx.strokeStyle = '#182436';
  ctx.lineWidth = 1;
  ctx.beginPath();
  [0, 0.25, 0.5, 0.75, 1.0].forEach(mu => {
    const y = yToCanvas(mu);
    ctx.moveTo(paddingLeft, y);
    ctx.lineTo(w - paddingRight, y);
  });
  ctx.stroke();

  // Y-axis labels
  ctx.fillStyle = '#64748b';
  ctx.font = '10px JetBrains Mono';
  ctx.textAlign = 'right';
  ctx.fillText('1.0', paddingLeft - 6, yToCanvas(1.0) + 4);
  ctx.fillText('0.5', paddingLeft - 6, yToCanvas(0.5) + 4);
  ctx.fillText('0.0', paddingLeft - 6, yToCanvas(0.0) + 4);

  // X-axis labels
  ctx.textAlign = 'center';
  [0, 20, 40, 60, 80, 100].forEach(val => {
    ctx.fillText(`${val}%`, xToCanvas(val), h - 8);
  });

  const curves = [
    { name: 'low', fn: (x) => trapmf(x, [0, 0, 35, 55]), color: '#34d399' },
    { name: 'moderate', fn: (x) => trimf(x, [45, 65, 80]), color: '#fbbf24' },
    { name: 'high', fn: (x) => trapmf(x, [70, 85, 100, 100]), color: '#f87171' }
  ];

  curves.forEach(c => {
    ctx.strokeStyle = c.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let px = 0; px <= plotW; px += 2) {
      const xVal = (px / plotW) * 100;
      const mu = c.fn(xVal);
      const cx = paddingLeft + px;
      const cy = yToCanvas(mu);
      if (px === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    }
    ctx.stroke();
  });

  // Vertical Crosshair
  const cx = xToCanvas(currentVal);
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(cx, paddingTop);
  ctx.lineTo(cx, paddingTop + plotH);
  ctx.stroke();
  ctx.setLineDash([]);

  // Intersection Dots
  curves.forEach(c => {
    const mu = c.fn(currentVal);
    if (mu > 0.02) {
      const cy = yToCanvas(mu);
      ctx.fillStyle = c.color;
      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  });
}

// Update Active Rules Table
function updateRulesTable(rules) {
  const tbody = document.getElementById('rulesTableBody');
  if (!rules || rules.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="table-placeholder">No rules activated at current parameters.</td></tr>';
    return;
  }

  tbody.innerHTML = rules.map(r => {
    const pct = Math.round(r.strength * 100);
    return `
      <tr>
        <td class="font-mono"><strong>${r.name}</strong></td>
        <td>${r.text}</td>
        <td><span class="font-mono">${r.consequent}</span></td>
        <td>
          <div class="alpha-bar-wrap">
            <div class="alpha-track">
              <div class="alpha-fill" style="width: ${pct}%;"></div>
            </div>
            <span class="font-mono text-cyan">${r.strength}</span>
          </div>
        </td>
        <td>
          <span class="status-badge active">FIRED</span>
        </td>
      </tr>
    `;
  }).join('');
}

// Load Benchmark Metrics
async function loadBenchmarkMetrics() {
  try {
    const res = await fetch('/api/metrics');
    const json = await res.json();
    if (json.status === 'success') {
      const m = json.metrics;
      document.getElementById('bmAnnMae').textContent = `${m.ann_model.mae} mm`;
      document.getElementById('bmBaseMae').textContent = `${m.baseline_model.mae} mm`;

      document.getElementById('bmAnnRmse').textContent = `${m.ann_model.rmse} mm`;
      document.getElementById('bmBaseRmse').textContent = `${m.baseline_model.rmse} mm`;

      document.getElementById('bmAnnR2').textContent = m.ann_model.r2_score;
      document.getElementById('bmImprovement').textContent = `-${m.improvement.mae_reduction_percent} %`;
    }
  } catch (e) {
    console.error('Failed to load benchmarks:', e);
  }
}

// Load Audit History Table with Recall Functionality
async function loadAuditLog() {
  try {
    const res = await fetch('/api/history');
    const json = await res.json();
    if (json.status === 'success') {
      const tbody = document.getElementById('auditTableBody');
      if (!json.history || json.history.length === 0) {
        tbody.innerHTML = '<tr><td colspan="10" class="table-placeholder">No audit logs recorded yet.</td></tr>';
        return;
      }

      tbody.innerHTML = json.history.map(row => {
        let badgeColor = '#10b981';
        if (row.risk_level === 'Moderate') badgeColor = '#f59e0b';
        else if (row.risk_level === 'High') badgeColor = '#f97316';
        else if (row.risk_level === 'Very High') badgeColor = '#ef4444';

        const rowJson = JSON.stringify({
          temp: row.temperature,
          humidity: row.relative_humidity,
          pressure: row.atmospheric_pressure,
          wind: row.wind_speed,
          prevRain: row.previous_rainfall
        }).replace(/"/g, '&quot;');

        return `
          <tr>
            <td class="font-mono">${row.timestamp.split(' ')[1] || row.timestamp}</td>
            <td class="font-mono">${row.temperature}°C</td>
            <td class="font-mono">${row.relative_humidity}%</td>
            <td class="font-mono">${row.atmospheric_pressure} hPa</td>
            <td class="font-mono">${row.wind_speed} km/h</td>
            <td class="font-mono">${row.previous_rainfall} mm</td>
            <td class="font-mono"><strong>${row.predicted_rainfall} mm</strong></td>
            <td class="font-mono">${row.risk_score}</td>
            <td>
              <span class="status-badge" style="color: ${badgeColor}; border-color: ${badgeColor}60; background: ${badgeColor}15;">
                ${row.risk_level}
              </span>
            </td>
            <td>
              <button type="button" class="recall-btn" onclick='recallScenario(${rowJson})'>Recall</button>
            </td>
          </tr>
        `;
      }).join('');
    }
  } catch (e) {
    console.error('Failed to load audit history:', e);
  }
}

function recallScenario(params) {
  setInputValue('tempInput', 'tempNum', params.temp);
  setInputValue('humidityInput', 'humidityNum', params.humidity);
  setInputValue('pressureInput', 'pressureNum', params.pressure);
  setInputValue('windInput', 'windNum', params.wind);
  setInputValue('prevRainInput', 'prevRainNum', params.prevRain);
  switchTab('tab-console');
  runInference();
}

async function clearAuditHistory() {
  if (!confirm('Clear all telemetry audit logs?')) return;
  try {
    await fetch('/api/history/clear', { method: 'POST' });
    loadAuditLog();
  } catch (e) {
    console.error(e);
  }
}

// Window resize redraw for responsive canvas
window.addEventListener('resize', () => {
  if (currentPrediction) renderAllFuzzyCurves();
});

// Boot Initialization
document.addEventListener('DOMContentLoaded', () => {
  loadBenchmarkMetrics();
  loadAuditLog();
  runInference();
});
