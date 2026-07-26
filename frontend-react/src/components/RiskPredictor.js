import React, { useState } from 'react';
import { predictRisk, explainRisk, analyzeStructure, generateReportUrl } from '../services/api';

const FIELDS = [
  { key: 'building_age', label: 'Building Age', min: 1, max: 80, step: 1, def: 25, unit: 'yrs' },
  { key: 'corrosion_level', label: 'Corrosion Level', min: 0, max: 1, step: 0.01, def: 0.3, unit: '' },
  { key: 'crack_width', label: 'Crack Width', min: 0, max: 10, step: 0.1, def: 2.0, unit: 'mm' },
  { key: 'crack_density', label: 'Crack Density', min: 0, max: 20, step: 0.1, def: 5.0, unit: '/m²' },
  { key: 'moisture_content', label: 'Moisture', min: 1, max: 15, step: 0.1, def: 5.0, unit: '%' },
  { key: 'compressive_strength', label: 'Compressive Str.', min: 12, max: 55, step: 0.5, def: 35.0, unit: 'MPa' },
  { key: 'temperature', label: 'Temperature', min: 10, max: 45, step: 0.5, def: 25.0, unit: '°C' },
  { key: 'humidity', label: 'Humidity', min: 30, max: 90, step: 1, def: 60.0, unit: '%' },
  { key: 'load_stress', label: 'Load Stress', min: 5, max: 35, step: 0.5, def: 15.0, unit: 'MPa' },
];

function RiskPredictor({ onNavigate, onPrediction }) {
  const [form, setForm] = useState(Object.fromEntries(FIELDS.map(f => [f.key, f.def])));
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [shap, setShap] = useState(null);
  const [rec, setRec] = useState(null);
  const [err, setErr] = useState(null);
  const [dlPdf, setDlPdf] = useState(false);

  const predict = async (e) => {
    e.preventDefault();
    setLoading(true); setErr(null);
    try {
      const pred = await predictRisk(form);
      setResults(pred);
      const s = await explainRisk(form);
      setShap(s.waterfall_b64);
      const a = await analyzeStructure(form);
      setRec(a.recommendation);
      if (onPrediction) onPrediction(form, pred, a.recommendation);
    } catch { setErr('Prediction failed. Check backend.'); }
    setLoading(false);
  };

  const downloadPdf = async () => {
    setDlPdf(true);
    try {
      const r = await fetch(generateReportUrl(), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(form) });
      const b = await r.blob();
      const a = document.createElement('a');
      a.href = URL.createObjectURL(b);
      a.download = 'building_report.pdf';
      document.body.appendChild(a); a.click(); a.remove();
    } catch { alert('Download failed'); }
    setDlPdf(false);
  };

  const riskPill = (level) => {
    if (level === 'Critical') return 'critical';
    if (level === 'Moderate') return 'moderate';
    return 'safe';
  };

  return (
    <div className="anim-fade-up">
      <div className="pg-header">
        <h1>Risk Predictor</h1>
        <p>Predict structural health index, risk level, and remaining useful life</p>
      </div>

      <div className="grid-2">
        {/* Input panel */}
        <div className="g-card">
          <div className="section-title">Parameters</div>
          <form onSubmit={predict}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2px 20px' }}>
              {FIELDS.map(f => (
                <div key={f.key} className="slider-group">
                  <div className="slider-header">
                    <span className="slider-name">{f.label}</span>
                    <span className="slider-val">{form[f.key]}{f.unit}</span>
                  </div>
                  <input
                    type="range" className="slider-input"
                    min={f.min} max={f.max} step={f.step}
                    value={form[f.key]}
                    onChange={e => setForm(p => ({ ...p, [f.key]: parseFloat(e.target.value) }))}
                  />
                </div>
              ))}
            </div>
            <button type="submit" className="btn btn-primary btn-lg btn-full" disabled={loading} style={{ marginTop: 14 }}>
              {loading ? 'Predicting...' : 'Predict Structural Health'}
            </button>
          </form>
          {err && <div className="banner error" style={{ marginTop: 14 }}>{err}</div>}
        </div>

        {/* Results panel */}
        <div className="g-card">
          <div className="section-title">Results</div>
          {!results ? (
            <div className="empty">
              <h4>No predictions yet</h4>
              <p>Adjust parameters and click Predict</p>
            </div>
          ) : (
            <div className="anim-scale">
              <div className="metrics-row" style={{ marginBottom: 14 }}>
                <div className="m-card purple">
                  <div className="m-label">SHI Score</div>
                  <div className="m-value">{results.shi}</div>
                  <div className="m-sub">of 100</div>
                </div>
                <div className="m-card" style={{ textAlign: 'center' }}>
                  <div className="m-label">Risk Level</div>
                  <div style={{ marginTop: 6 }}>
                    <span className={`m-pill ${riskPill(results.risk_level)}`}>
                      {results.risk_level}
                    </span>
                  </div>
                </div>
                <div className="m-card emerald">
                  <div className="m-label">RUL</div>
                  <div className="m-value">{results.rul}</div>
                  <div className="m-sub">years</div>
                </div>
              </div>

              <div className="m-card amber" style={{ textAlign: 'center', marginBottom: 18 }}>
                <div className="m-label">Est. Repair Cost</div>
                <div className="m-value" style={{ fontSize: 22 }}>{results.repair_cost}</div>
              </div>

              {rec && (
                <div className="section" style={{ marginBottom: 18 }}>
                  <div className="section-title">AI Recommendation</div>
                  <div className="report-text">{rec}</div>
                </div>
              )}

              {shap && (
                <div className="section" style={{ marginBottom: 18 }}>
                  <div className="section-title">SHAP Feature Importance</div>
                  <img src={`data:image/png;base64,${shap}`} alt="SHAP" className="result-img" />
                </div>
              )}

              <div style={{ display: 'flex', gap: 10 }}>
                <button className="btn btn-emerald" style={{ flex: 1 }} onClick={downloadPdf} disabled={dlPdf}>
                  {dlPdf ? 'Generating...' : 'Download PDF'}
                </button>
                <button className="btn btn-secondary" style={{ flex: 1 }} onClick={() => onNavigate('chat')}>
                  Chat about results
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default RiskPredictor;
