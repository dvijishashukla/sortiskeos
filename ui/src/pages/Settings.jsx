import { useState, useEffect } from "react";
import { triggerPipeline, checkHealth } from "../api";

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
    <div style={{ padding: "32px", maxWidth: 800, margin: "0 auto", fontFamily: "'Roboto', sans-serif" }}>
      <header style={{ marginBottom: 40, animation: "fadeSlideUp 0.5s ease both" }}>
        <h1 style={{ fontSize: 24, fontWeight: 600, color: "#e6edf3", margin: 0, letterSpacing: "-0.5px" }}>
          System Configuration
        </h1>
        <p style={{ color: "#6e7681", fontSize: 13, marginTop: 6, letterSpacing: "0.2px" }}>
          Local UI preferences and manual backend diagnostic tools
        </p>
      </header>

      <section style={{ marginBottom: 40, animation: "fadeSlideUp 0.5s ease 0.05s both" }}>
        <h2 style={{ fontSize: 13, fontWeight: 600, color: "#8b949e", marginBottom: 16, letterSpacing: "1.2px", textTransform: "uppercase" }}>
          Connectivity Status
        </h2>
        
        <div style={{ background: "rgba(22, 27, 34, 0.4)", backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.03)", borderRadius: 12, padding: 24, display: "flex", justifyContent: "space-between", alignItems: "center", boxShadow: "0 4px 20px rgba(0,0,0,0.1)" }}>
          <div>
            <div style={{ fontSize: 15, fontWeight: 600, color: "#e6edf3" }}>Elasticsearch Node</div>
            <div style={{ fontSize: 12, color: "#6e7681", marginTop: 6, lineHeight: 1.4 }}>{runMode === "local" ? "Running in Local Mode with JSON-backed logs and anomaly results." : "Currently pinging http://localhost:9200 natively via Logstash mapping"}</div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, background: "rgba(13, 17, 23, 0.3)", padding: "8px 16px", borderRadius: 20, border: "1px solid rgba(255,255,255,0.05)" }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: esConnected ? "#3fb950" : "#ff5f5f", boxShadow: `0 0 8px ${esConnected ? "rgba(63,185,80,0.4)" : "rgba(255,95,95,0.4)"}` }} />
            <span style={{ fontSize: 12, fontWeight: 600, color: esConnected ? "#3fb950" : "#ff5f5f" }}>
              {esConnected ? "CONNECTED" : "OFFLINE"}
            </span>
          </div>
        </div>
      </section>

      <section style={{ marginBottom: 40, animation: "fadeSlideUp 0.5s ease 0.1s both" }}>
        <h2 style={{ fontSize: 13, fontWeight: 600, color: "#8b949e", marginBottom: 16, letterSpacing: "1.2px", textTransform: "uppercase" }}>
          Manual Agent Overrides
        </h2>
        
        <div style={{ background: "rgba(22, 27, 34, 0.4)", backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.03)", borderRadius: 12, padding: 24, boxShadow: "0 4px 20px rgba(0,0,0,0.1)" }}>
          <div style={{ fontSize: 15, fontWeight: 600, color: "#e6edf3", marginBottom: 6 }}>Force Machine Learning Process</div>
          <div style={{ fontSize: 12, color: "#6e7681", marginBottom: 20, lineHeight: 1.4 }}>Bypass the smart-trigger and force the Edge Agent to retrain its TF-IDF model immediately.</div>
          
          <button 
            className="action-btn"
            onClick={handleManualTrigger}
            disabled={loading}
            style={{
              background: loading ? "rgba(255,255,255,0.05)" : "#238636",
              color: "#fff",
              border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, padding: "10px 24px",
              fontSize: 14, fontWeight: 600, cursor: loading ? "not-allowed" : "pointer",
            }}
          >
            {loading ? "Processing..." : "Trigger ML Pipeline"}
          </button>
        </div>
      </section>

      <section style={{ animation: "fadeSlideUp 0.5s ease 0.15s both" }}>
        <h2 style={{ fontSize: 13, fontWeight: 600, color: "#8b949e", marginBottom: 16, letterSpacing: "1.2px", textTransform: "uppercase" }}>
          Dashboard Preferences
        </h2>
        
        <div style={{ background: "rgba(22, 27, 34, 0.4)", backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.03)", borderRadius: 12, padding: 24, boxShadow: "0 4px 20px rgba(0,0,0,0.1)" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
            <div>
              <label style={{ display: "block", fontSize: 14, fontWeight: 600, color: "#e6edf3", marginBottom: 8 }}>Auto-Refresh Interval</label>
              <div style={{ fontSize: 12, color: "#6e7681", marginBottom: 12 }}>How often the frontend polls the FastAPI endpoints (seconds).</div>
              <input 
                type="number" 
                value={pollInterval} 
                onChange={e => setPollInterval(e.target.value)}
                style={{
                  background: "rgba(13, 17, 23, 0.4)", border: "1px solid rgba(255,255,255,0.05)", borderRadius: 6,
                  color: "#e6edf3", padding: "10px 14px", width: 140, outline: "none", fontSize: 14, transition: "border 0.2s"
                }}
              />
            </div>
            
            <label style={{ display: "flex", alignItems: "center", gap: 16, cursor: "pointer" }}>
              <input 
                type="checkbox" 
                checked={animations} 
                onChange={e => setAnimations(e.target.checked)}
                style={{ width: 18, height: 18, accentColor: "#e6734b", cursor: "pointer" }}
              />
              <div>
                <div style={{ fontSize: 14, fontWeight: 600, color: "#e6edf3" }}>Enable UI Animations</div>
                <div style={{ fontSize: 12, color: "#6e7681", marginTop: 4 }}>Toggle glowing hover logic and transition sweeps.</div>
              </div>
            </label>
          </div>

          <div style={{ marginTop: 32, borderTop: "1px solid rgba(255,255,255,0.05)", paddingTop: 24 }}>
            <button 
              className="action-btn"
              onClick={handleSaveConfig}
              style={{
                background: "rgba(255,255,255,0.04)", color: "#e6edf3", border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: 6, padding: "10px 24px", fontSize: 14, fontWeight: 600, cursor: "pointer",
              }}
            >
              Save Configuration
            </button>
          </div>
        </div>
      </section>

    </div>
  );
}
