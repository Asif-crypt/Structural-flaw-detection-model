import React, { useState, useRef } from 'react';
import { analyzeImage, analyzeThermal } from '../services/api';

function CrackDetection({ onNavigate }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [mode, setMode] = useState('rgb');
  const [dragging, setDragging] = useState(false);
  const ref = useRef(null);

  const pick = (f) => {
    if (f && f.type.startsWith('image/')) {
      setFile(f); setPreview(URL.createObjectURL(f)); setResults(null); setErr(null);
    }
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true); setErr(null);
    try {
      setResults(mode === 'thermal' ? await analyzeThermal(file) : await analyzeImage(file));
    } catch { setErr('Analysis failed. Check backend.'); }
    setLoading(false);
  };

  return (
    <div className="anim-fade-up">
      <div className="pg-header">
        <h1>Crack Detection</h1>
        <p>Upload concrete or masonry images for AI-powered analysis</p>
      </div>

      <div className="toggle-group">
        <button className={`toggle-opt ${mode === 'rgb' ? 'active' : ''}`} onClick={() => { setMode('rgb'); setResults(null); }}>
          Standard (RGB)
        </button>
        <button className={`toggle-opt ${mode === 'thermal' ? 'active' : ''}`} onClick={() => { setMode('thermal'); setResults(null); }}>
          Thermal / IR
        </button>
      </div>

      <div className="grid-2">
        {/* Upload */}
        <div className="g-card">
          <div className="section-title">Upload Image</div>
          <input ref={ref} type="file" accept="image/*" onChange={e => pick(e.target.files[0])} style={{ display: 'none' }} />
          <div
            className={`drop-zone ${dragging ? 'dragging' : ''} ${preview ? 'has-image' : ''}`}
            onClick={() => ref.current?.click()}
            onDrop={e => { e.preventDefault(); setDragging(false); pick(e.dataTransfer.files[0]); }}
            onDragOver={e => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            style={preview ? { backgroundImage: `url(${preview})`, backgroundSize: 'cover', backgroundPosition: 'center', height: 300 } : { minHeight: 260, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}
          >
            {!preview && (
              <>
                <h4>{dragging ? 'Drop image here' : 'Click or drag an image'}</h4>
                <span>JPG, PNG up to 10MB</span>
              </>
            )}
          </div>

          {preview && (
            <button className="btn btn-ghost btn-full" style={{ marginTop: 8 }} onClick={() => { setFile(null); setPreview(null); setResults(null); }}>
              Remove image
            </button>
          )}

          <button className="btn btn-primary btn-lg btn-full" style={{ marginTop: 14 }} onClick={analyze} disabled={!file || loading}>
            {loading ? 'Analyzing...' : 'Run Detection'}
          </button>

          {err && <div className="banner error" style={{ marginTop: 14 }}>{err}</div>}
        </div>

        {/* Results */}
        <div className="g-card">
          <div className="section-title">Results</div>

          {loading ? (
            <div className="loader-wrap"><div className="loader" /><div className="loader-text">Running detection pipeline...</div></div>
          ) : !results ? (
            <div className="empty"><h4>No results</h4><p>Upload an image and run detection</p></div>
          ) : (
            <div className="anim-scale">
              {mode === 'thermal' ? (
                <>
                  <div className="metrics-row" style={{ marginBottom: 16 }}>
                    <div className="m-card rose"><div className="m-label">Anomaly Density</div><div className="m-value">{results.density}%</div></div>
                    <div className="m-card amber"><div className="m-label">Anomaly Pixels</div><div className="m-value">{results.anomaly_area?.toLocaleString()}</div></div>
                  </div>
                  <div className="section-title">Thermal Anomaly Overlay</div>
                  <img src={`data:image/png;base64,${results.mask_image_b64}`} alt="Thermal" className="result-img" />
                  <div className="img-caption">Moisture and heat bridge anomalies</div>
                </>
              ) : (
                <>
                  <div className="metrics-row" style={{ marginBottom: 16 }}>
                    <div className="m-card purple"><div className="m-label">Detections</div><div className="m-value">{results.count}</div></div>
                    <div className="m-card rose"><div className="m-label">Coverage</div><div className="m-value">{results.density}%</div></div>
                    <div className="m-card amber"><div className="m-label">Est. Depth</div><div className="m-value">{results.max_crack_depth}mm</div></div>
                  </div>

                  <div className="section-title">YOLOv8 Detection</div>
                  <img src={`data:image/png;base64,${results.annotated_image_b64}`} alt="Detection" className="result-img" />
                  <div className="img-caption">{results.count} crack region(s) · conf ≥ 0.25</div>

                  <div className="hr" />

                  <div className="section-title">U-Net Segmentation</div>
                  <img src={`data:image/png;base64,${results.mask_image_b64}`} alt="Segmentation" className="result-img" />
                  <div className="img-caption">{results.density}% coverage · {results.crack_area?.toLocaleString()} px</div>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default CrackDetection;
