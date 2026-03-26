import { useState, useRef } from "react";
import { analyzeJob } from "./api/analyze";
import ParticleBackground from "./components/ParticleBackground";
import ScannerLoader from "./components/ScannerLoader";
import ResultPanel from "./components/ResultPanel";
import "./index.css";

export default function App() {
  const [text, setText]           = useState("");
  const [url, setUrl]             = useState("");
  const [company, setCompany]     = useState("");
  const [image, setImage]         = useState(null);
  const [imageName, setImageName] = useState("");
  const [result, setResult]       = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState("");
  const fileRef = useRef();

  const handleFile = (e) => {
    const f = e.target.files[0];
    if (f) { setImage(f); setImageName(f.name); }
  };

  const clearFile = (e) => {
    e.stopPropagation();
    setImage(null); setImageName("");
    fileRef.current.value = "";
  };

  const submit = async (e) => {
    e.preventDefault();
    if (!text.trim() && !url.trim() && !image) {
      setError("Provide a job description, URL, or screenshot to analyze.");
      return;
    }
    setError(""); setLoading(true); setResult(null);
    try {
      const data = await analyzeJob({ text, url, companyName: company, image });
      setResult(data);
      setTimeout(() => document.getElementById("anchor")?.scrollIntoView({ behavior: "smooth" }), 80);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <ParticleBackground />
      <div className="page">
        <div className="container">

          {/* ── Header ── */}
          <header className="header">
            <div className="badge">
              <span className="badge-dot" />
              AI-Powered Scam Detection
            </div>
            <h1>Fake Job Detector</h1>
            <p>
              Paste a description, enter a URL, or upload a screenshot.
              Our AI scans for scam patterns, domain signals, and behavioral red flags in seconds.
            </p>
            <div className="stats">
              <div><div className="stat-num">8+</div><div className="stat-lbl">Scam Patterns</div></div>
              <div><div className="stat-num">4</div><div className="stat-lbl">AI Modules</div></div>
              <div><div className="stat-num">&lt;2s</div><div className="stat-lbl">Analysis</div></div>
            </div>
          </header>

          {/* ── Form card ── */}
          <div className="card">
            <form onSubmit={submit}>

              <div className="field">
                <div className="flabel"><span className="flabel-icon">📝</span>Job Description</div>
                <textarea
                  value={text}
                  onChange={e => setText(e.target.value)}
                  placeholder="Paste the full job description — title, requirements, salary, contact info…"
                />
              </div>

              <div className="grid2">
                <div className="field">
                  <div className="flabel"><span className="flabel-icon">🔗</span>Job URL</div>
                  <input type="url" value={url} onChange={e => setUrl(e.target.value)}
                    placeholder="https://example.com/jobs/123" />
                </div>
                <div className="field">
                  <div className="flabel"><span className="flabel-icon">🏢</span>Company Name</div>
                  <input type="text" value={company} onChange={e => setCompany(e.target.value)}
                    placeholder="e.g. Google, Acme Corp" />
                </div>
              </div>

              <div className="divider">or upload screenshot</div>

              <div className="field">
                <div
                  className={`upload${imageName ? " filled" : ""}`}
                  onClick={() => fileRef.current.click()}
                  role="button" tabIndex={0}
                  onKeyDown={e => e.key === "Enter" && fileRef.current.click()}
                >
                  <div className="upload-box">{imageName ? "✅" : "🖼️"}</div>
                  <div className="upload-title">
                    {imageName ? imageName : "Click to upload job screenshot"}
                  </div>
                  <div className="upload-sub">
                    {imageName
                      ? <span onClick={clearFile} style={{ color: "var(--hi)", cursor: "pointer" }}>✕ Remove file</span>
                      : "JPG · PNG · WEBP · BMP — OCR extracts text automatically"}
                  </div>
                </div>
                <input ref={fileRef} type="file" accept="image/*" onChange={handleFile} style={{ display: "none" }} />
              </div>

              {error && <div className="err"><span>⚠️</span>{error}</div>}

              <button type="submit" className="btn" disabled={loading}>
                <div className="btn-inner">
                  {loading ? <span className="spin" /> : <span>🔍</span>}
                  {loading ? "Analyzing…" : "Analyze Job Posting"}
                </div>
              </button>
            </form>

            {/* Scanner overlay while loading */}
            {loading && <ScannerLoader />}
          </div>

          <div id="anchor" />
          <ResultPanel result={result} />
        </div>
      </div>
    </>
  );
}
