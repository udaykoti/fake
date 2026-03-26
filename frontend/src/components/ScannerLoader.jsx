export default function ScannerLoader() {
  return (
    <div style={{
      display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "center", padding: "40px 0", gap: 20,
    }}>
      {/* Radar ring */}
      <div style={{ position: "relative", width: 90, height: 90 }}>
        {/* Outer ring */}
        <div style={{
          position: "absolute", inset: 0, borderRadius: "50%",
          border: "1px solid rgba(99,102,241,.25)",
          animation: "radar-ping 1.8s ease-out infinite",
        }} />
        {/* Middle ring */}
        <div style={{
          position: "absolute", inset: 12, borderRadius: "50%",
          border: "1px solid rgba(99,102,241,.35)",
          animation: "radar-ping 1.8s .3s ease-out infinite",
        }} />
        {/* Inner circle */}
        <div style={{
          position: "absolute", inset: 28, borderRadius: "50%",
          background: "rgba(99,102,241,.15)",
          border: "1px solid rgba(99,102,241,.5)",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 16,
        }}>🔍</div>
        {/* Sweep line */}
        <div style={{
          position: "absolute", inset: 0, borderRadius: "50%",
          background: "conic-gradient(from 0deg, transparent 70%, rgba(99,102,241,.4) 100%)",
          animation: "sweep 1.5s linear infinite",
        }} />
      </div>

      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: 14, fontWeight: 600, color: "#a5b4fc", marginBottom: 6 }}>
          Analyzing job posting…
        </div>
        <div style={{ fontSize: 12, color: "#3d4f6e" }}>
          Checking NLP patterns · domain signals · behavioral flags
        </div>
      </div>

      <style>{`
        @keyframes radar-ping {
          0%   { transform: scale(1);   opacity: .7; }
          100% { transform: scale(1.4); opacity: 0;  }
        }
        @keyframes sweep {
          from { transform: rotate(0deg); }
          to   { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
