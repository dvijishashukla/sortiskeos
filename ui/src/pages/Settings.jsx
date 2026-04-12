import { useState, useEffect } from "react";
import { triggerPipeline, checkHealth } from "../api";
import PageHeader from "../components/PageHeader.jsx";

export default function Settings() {
  const [esConnected, setEsConnected] = useState(false);
  const [runMode, setRunMode] = useState("local");
  const [loading, setLoading] = useState(false);
  const [pollInterval, setPollInterval] = useState(localStorage.getItem("uiPollInterval") || "30");
  const [animations, setAnimations] = useState(localStorage.getItem("uiAnimations") !== "false");

  useEffect(() => {
    checkHealth().then((status) => {
      setEsConnected(Boolean(status?.elasticsearch));
      setRunMode(status?.mode || "local");
    });
  }, []);

  const handleManualTrigger = async () => {
    setLoading(true);
    await triggerPipeline();
    setTimeout(() => {
      setLoading(false);
      alert("Manual ML Pipeline triggered dynamically!");
    }, 1500);
  };

  const handleSaveConfig = () => {
    localStorage.setItem("uiPollInterval", pollInterval);
    localStorage.setItem("uiAnimations", animations.toString());
    alert("Configuration saved locally!");
  };

  return (
    <>
      <PageHeader 
        title="System Configuration" 
        subtitle="Local UI preferences and diagnostic tools" 
        usingFallback={!esConnected}
      />

      <div style={{ padding: "0 32px 32px", maxWidth: 1200, margin: "0 auto", fontFamily: "'Roboto', sans-serif" }}>
        <div style={{ height: 32 }} />

      <section style={{ marginBottom: 40, animation: "fadeSlideUp 0.5s ease 0.05s both" }}>
        <h2 style={{ fontSize: 13, fontWeight: 600, color: "#64748b", marginBottom: 16, letterSpacing: "1.2px", textTransform: "uppercase" }}>
          Connectivity Status
        </h2>
        
        <div style={{ background: "rgba(19, 19, 31, 0.4)", backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)", border: "1px solid #1e1e2e", borderRadius: 12, padding: 24, display: "flex", justifyContent: "space-between", alignItems: "center", boxShadow: "0 0 12px #7c3aed18" }}>
          <div>
            <div style={{ fontSize: 15, fontWeight: 600, color: "#e2e8f0" }}>Elasticsearch Node</div>
            <div style={{ fontSize: 12, color: "#64748b", marginTop: 6, lineHeight: 1.4 }}>{runMode === "local" ? "Running in Local Mode with JSON-backed logs and anomaly results." : "Currently pinging http://localhost:9200 natively via Logstash mapping"}</div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, background: "rgba(13, 13, 20, 0.3)", padding: "8px 16px", borderRadius: 20, border: "1px solid #1e1e2e" }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: esConnected ? "#22c55e" : "#ef4444", boxShadow: `0 0 8px ${esConnected ? "rgba(34,197,94,0.4)" : "rgba(239,68,68,0.4)"}` }} />
            <span style={{ fontSize: 12, fontWeight: 600, color: esConnected ? "#22c55e" : "#ef4444" }}>
              {esConnected ? "CONNECTED" : "OFFLINE"}
            </span>
          </div>
        </div>
      </section>

      <section style={{ marginBottom: 40, animation: "fadeSlideUp 0.5s ease 0.1s both" }}>
        <h2 style={{ fontSize: 13, fontWeight: 600, color: "#64748b", marginBottom: 16, letterSpacing: "1.2px", textTransform: "uppercase" }}>
          Manual Agent Overrides
        </h2>
        
        <div style={{ background: "rgba(19, 19, 31, 0.4)", backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)", border: "1px solid #1e1e2e", borderRadius: 12, padding: 24, boxShadow: "0 0 12px #7c3aed18" }}>
          <div style={{ fontSize: 15, fontWeight: 600, color: "#e2e8f0", marginBottom: 6 }}>Force Machine Learning Process</div>
          <div style={{ fontSize: 12, color: "#64748b", marginBottom: 20, lineHeight: 1.4 }}>Bypass the smart-trigger and force the Edge Agent to retrain its TF-IDF model immediately.</div>
          
          <button 
            className="action-btn"
            onClick={handleManualTrigger}
            disabled={loading}
            style={{
              background: loading ? "rgba(255,255,255,0.05)" : "#7c3aed",
              color: "#fff",
              border: "1px solid #1e1e2e", borderRadius: 6, padding: "10px 24px",
              fontSize: 14, fontWeight: 600, cursor: loading ? "not-allowed" : "pointer",
            }}
          >
            {loading ? "Processing..." : "Trigger ML Pipeline"}
          </button>
        </div>
      </section>

      <section style={{ animation: "fadeSlideUp 0.5s ease 0.15s both" }}>
        <h2 style={{ fontSize: 13, fontWeight: 600, color: "#64748b", marginBottom: 16, letterSpacing: "1.2px", textTransform: "uppercase" }}>
          Dashboard Preferences
        </h2>
        
        <div style={{ background: "rgba(19, 19, 31, 0.4)", backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)", border: "1px solid #1e1e2e", borderRadius: 12, padding: 24, boxShadow: "0 0 12px #7c3aed18" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
            <div>
              <label style={{ display: "block", fontSize: 14, fontWeight: 600, color: "#e2e8f0", marginBottom: 8 }}>Auto-Refresh Interval</label>
              <div style={{ fontSize: 12, color: "#64748b", marginBottom: 12 }}>How often the frontend polls the FastAPI endpoints (seconds).</div>
              <input 
                type="number" 
                value={pollInterval} 
                onChange={e => setPollInterval(e.target.value)}
                style={{
                  background: "rgba(13, 13, 20, 0.4)", border: "1px solid #1e1e2e", borderRadius: 6,
                  color: "#e2e8f0", padding: "10px 14px", width: 140, outline: "none", fontSize: 14, transition: "border 0.2s"
                }}
              />
            </div>
            
            <label style={{ display: "flex", alignItems: "center", gap: 16, cursor: "pointer" }}>
              <input 
                type="checkbox" 
                checked={animations} 
                onChange={e => setAnimations(e.target.checked)}
                style={{ width: 18, height: 18, accentColor: "#7c3aed", cursor: "pointer" }}
              />
              <div>
                <div style={{ fontSize: 14, fontWeight: 600, color: "#e2e8f0" }}>Enable UI Animations</div>
                <div style={{ fontSize: 12, color: "#64748b", marginTop: 4 }}>Toggle glowing hover logic and transition sweeps.</div>
              </div>
            </label>
          </div>

          <div style={{ marginTop: 32, borderTop: "1px solid #1e1e2e", paddingTop: 24 }}>
            <button 
              className="action-btn"
              onClick={handleSaveConfig}
              style={{
                background: "rgba(255,255,255,0.04)", color: "#e2e8f0", border: "1px solid #1e1e2e",
                borderRadius: 6, padding: "10px 24px", fontSize: 14, fontWeight: 600, cursor: "pointer",
              }}
            >
              Save Configuration
            </button>
          </div>
        </div>
      </section>
    </div>
    </>
  );
}
