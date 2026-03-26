import { useEffect, useState } from "react";

const META = {
  LOW:      { icon: "✅", label: "Safe",     desc: "Looks like a legitimate posting" },
  MEDIUM:   { icon: "⚠️",  label: "Caution",  desc: "Some suspicious signals detected" },
  HIGH:     { icon: "🚨", label: "Danger",   desc: "Multiple red flags found" },
  CRITICAL: { icon: "☠️", label: "Scam",     desc: "Highly likely to be fraudulent" },
  UNKNOWN:  { icon: "❓", label: "Unknown",  desc: "Insufficient data to analyze" },
};

export default function ScoreGauge({ score, level }) {
  const [displayed, setDisplayed] = useState(0);
  const pct = Math.round((score || 0) * 100);
  const cls = level || "UNKNOWN";
  const m   = META[cls] || META.UNKNOWN;

  // Animate counter
  useEffect(() => {
    let start = 0;
    const step = pct / 40;
    const timer = setInterval(() => {
      start += step;
      if (start >= pct) { setDisplayed(pct); clearInterval(timer); }
      else setDisplayed(Math.round(start));
    }, 20);
    return () => clearInterval(timer);
  }, [pct]);

  return (
    <div className={`gauge ${cls}`}>
      <div className="gauge-top">
        <div>
          <div className="gauge-tag">Scam Probability</div>
          <div className="gauge-level">
            <span className="gauge-icon">{m.icon}</span>
            {m.label}
          </div>
          <div className="gauge-desc">{m.desc}</div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div className="gauge-pct">{displayed}%</div>
          <div className="gauge-sub">risk score</div>
        </div>
      </div>
      <div className="track">
        <div className="fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
