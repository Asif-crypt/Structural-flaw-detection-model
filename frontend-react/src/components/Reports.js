import React, { useState } from 'react';
import { analyzeStructure, generateReportUrl } from '../services/api';

function Reports({ onNavigate, lastInputs, lastPredictions, lastRecommendation }) {
  const [rec, setRec] = useState(lastRecommendation || null);
  const [preds, setPreds] = useState(lastPredictions || null);
  const [loading, setLoading] = useState(false);
  const [dlPdf, setDlPdf] = useState(false);
  const [err, setErr] = useState(null);

  const inputs = lastInputs || {
    building_age: 30, corrosion_level: 0.5, crack_width: 3.0, crack_density: 8.0,
    moisture_content: 7.0, compressive_strength: 30.0, temperature: 28.0, humidity: 60.0, load_stress: 18.0,
  };

  const generate = async () => {
    setLoading(true); setErr(null);
    try { const r = await analyzeStructure(inputs); setPreds(r.predictions); setRec(r.recommendation); }
    catch { setErr('Report generation failed.'); }
    setLoading(false);
  };

  const downloadPdf = async () => {
    setDlPdf(true);
    try {
      const r = await fetch(generateReportUrl(), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(inputs) });
      const b = await r.blob(); const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = 'structural_health_report.pdf';
      document.body.appendChild(a); a.click(); a.remove();
    } catch { alert('Download failed'); }
    setDlPdf(false);
  };

  const riskPill = (l) => l === 'Critical' ? 'critical' : l === 'Moderate' ? 'moderate' : 'safe';

  return (
    <div className="anim-fade-up">
      <div className="pg-header">
        <h1>AI Reports</h1>
        <p>Generate comprehensive structural inspection reports</p>
      </div>

      {!lastInputs && (
        <div className="banner warn">
          Run the <span style={{ fontWeight: 700, cursor: 'pointer', textDecoration: 'underline' }} onClick={() => onNavigate('risk')}>Risk Predictor</span> first for accurate results. Using defaults.
        </div>
      )}

      <div style={{ display: 'flex', gap: 10, marginBottom: 24 }}>
        <button className="btn btn-primary btn-lg" onClick={generate} disabled={loading}>
          {loading ? 'Generating...' : 'Generate AI Report'}
        </button>
        {rec && (
          <button className="btn btn-emerald btn-lg" onClick={downloadPdf} disabled={dlPdf}>
            {dlPdf ? 'Downloading...' : 'Download PDF'}
          </button>
        )}
      </div>

      {err && <div className="banner error">{err}</div>}

      {rec && (
        <div className="anim-scale">
          {preds && (
            <div className="metrics-row" style={{ marginBottom: 24 }}>
              <div className="m-card purple"><div className="m-label">SHI</div><div className="m-value">{preds.shi}</div></div>
              <div className="m-card" style={{ textAlign: 'center' }}><div className="m-label">Risk</div><div style={{ marginTop: 6 }}><span className={`m-pill ${riskPill(preds.risk_level)}`}>{preds.risk_level}</span></div></div>
              <div className="m-card emerald"><div className="m-label">RUL</div><div className="m-value">{preds.rul}y</div></div>
            </div>
          )}
          <div className="g-card">
            <div className="section-title">Inspection Report</div>
            <div className="report-text">{rec}</div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Reports;
