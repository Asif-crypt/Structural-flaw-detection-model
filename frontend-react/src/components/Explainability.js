import React, { useState } from 'react';
import { explainRisk } from '../services/api';

function Explainability({ onNavigate, lastInputs }) {
  const [shap, setShap] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

  const PLOTS = [
    { title: 'SHI Feature Importance (Global Bar)', file: 'shi_shap_bar.png' },
    { title: 'SHI Feature Impact (Beeswarm)', file: 'shi_shap_beeswarm.png' },
    { title: 'Risk "Critical" Class Importance', file: 'risk_shap_bar.png' },
  ];

  const generate = async () => {
    if (!lastInputs) return;
    setLoading(true); setErr(null);
    try { setShap((await explainRisk(lastInputs)).waterfall_b64); }
    catch { setErr('Failed to generate SHAP waterfall.'); }
    setLoading(false);
  };

  return (
    <div className="anim-fade-up">
      <div className="pg-header">
        <h1>Explainability</h1>
        <p>SHAP feature importance and model interpretability</p>
      </div>

      <div className="g-card" style={{ marginBottom: 24 }}>
        <div className="section-title">Local Explanation (Waterfall)</div>
        {lastInputs ? (
          <>
            <div className="banner info" style={{ marginBottom: 14 }}>
              Generate a SHAP waterfall showing how each feature contributed to your specific prediction.
            </div>
            <button className="btn btn-primary" onClick={generate} disabled={loading} style={{ marginBottom: 16 }}>
              {loading ? 'Generating...' : 'Generate SHAP Waterfall'}
            </button>
            {err && <div className="banner error">{err}</div>}
            {shap && (
              <div className="anim-scale">
                <img src={`data:image/png;base64,${shap}`} alt="SHAP Waterfall" className="result-img" />
                <div className="img-caption">SHAP waterfall for your input parameters</div>
              </div>
            )}
          </>
        ) : (
          <div className="banner warn">
            Run a prediction in the{' '}
            <span style={{ fontWeight: 700, cursor: 'pointer', textDecoration: 'underline' }} onClick={() => onNavigate('risk')}>
              Risk Predictor
            </span>{' '}
            first.
          </div>
        )}
      </div>

      <div className="g-card">
        <div className="section-title">Global Explanations</div>
        <div className="banner info" style={{ marginBottom: 14 }}>
          Model-level explanations computed across the full training dataset.
        </div>
        {PLOTS.map(p => <Expander key={p.file} title={p.title} file={p.file} />)}
      </div>
    </div>
  );
}

function Expander({ title, file }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="expander">
      <button className="expander-head" onClick={() => setOpen(!open)}>
        {title}
        <span className={`expander-arrow ${open ? 'open' : ''}`}>▼</span>
      </button>
      {open && (
        <div className="expander-body">
          <img
            src={`${process.env.PUBLIC_URL || ''}/shap/${file}`}
            alt={title}
            className="result-img"
            onError={e => { e.target.style.display = 'none'; e.target.parentElement.innerHTML = '<div class="banner warn" style="margin:0">Plot not found. Run scripts/explain.py to generate.</div>'; }}
          />
        </div>
      )}
    </div>
  );
}

export default Explainability;
