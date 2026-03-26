import ScoreGauge from "./ScoreGauge";

function scoreColor(v) {
  if (v < 0.35) return "var(--low)";
  if (v < 0.65) return "var(--med)";
  return "var(--hi)";
}

export default function ResultPanel({ result }) {
  if (!result) return null;
  const { final_score, risk_level, explanation, breakdown, flags, ocr_text, modules_used } = result;

  return (
    <div className="result">
      <ScoreGauge score={final_score} level={risk_level} />

      {/* Explanation */}
      <div className="sec">
        <div className="sec-title"><span className="sec-icon">📋</span>Analysis Summary</div>
        <p className="expl">{explanation}</p>
      </div>

      {/* Flags */}
      {flags?.length > 0 && (
        <div className="sec">
          <div className="sec-title">
            <span className="sec-icon">🚩</span>Red Flags ({flags.length})
          </div>
          <ul className="flags">
            {flags.map((f, i) => (
              <li key={i} className="flag" style={{ animationDelay: `${i * 0.06}s` }}>
                <span className="flag-dot" />
                {f}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Breakdown */}
      {breakdown && Object.keys(breakdown).filter(k => k !== "final").length > 0 && (
        <div className="sec">
          <div className="sec-title"><span className="sec-icon">📊</span>Module Breakdown</div>
          <div className="bgrid">
            {Object.entries(breakdown)
              .filter(([k]) => k !== "final")
              .map(([k, v]) => (
                <div key={k} className="bcard">
                  <div className="bmod">{k}</div>
                  <div className="bscore" style={{ color: scoreColor(v) }}>
                    {Math.round(v * 100)}%
                  </div>
                  <div className="bbar">
                    <div className="bfill" style={{ width: `${Math.round(v * 100)}%` }} />
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* OCR */}
      {ocr_text && (
        <div className="sec">
          <details className="ocr">
            <summary>
              <span className="ocr-arrow">▶</span>
              Extracted text from image ({ocr_text.length} chars)
            </summary>
            <pre className="ocr-pre">{ocr_text}</pre>
          </details>
        </div>
      )}

      {/* Modules */}
      {modules_used?.length > 0 && (
        <div className="mpills">
          <span className="mlabel">Modules:</span>
          {modules_used.map(m => <span key={m} className="mpill">{m}</span>)}
        </div>
      )}
    </div>
  );
}
