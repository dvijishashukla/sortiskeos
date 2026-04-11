import { useEffect, useState } from "react";
import { fetchCrashes } from "../api";

export default function CrashHistory() {
  const [crashes, setCrashes] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function init() {
      const data = await fetchCrashes();
      if (data) setCrashes(data);
      setLoading(false);
    }
    init();
  }, []);

  return (
    <div style={{ padding: "32px", maxWidth: 1000, margin: "0 auto", fontFamily: "'Roboto', sans-serif" }}>
      <header style={{ marginBottom: 40, animation: "fadeSlideUp 0.5s ease both" }}>
        <h1 style={{ fontSize: 24, fontWeight: 600, color: "#e2e8f0", margin: 0, letterSpacing: "-0.5px" }}>
          Crash & Anomaly History
        </h1>
        <p style={{ color: "#64748b", fontSize: 13, marginTop: 6, letterSpacing: "0.2px" }}>
          Chronological record of clustered critical events and system crashes
        </p>
      </header>

      {loading ? (
        <div style={{ color: "#64748b", fontSize: 14 }}>Loading timeline...</div>
      ) : crashes.length === 0 ? (
        <div style={{ color: "#64748b", fontSize: 14 }}>No significant crash events recorded in Elasticsearch yet.</div>
      ) : (
        <div style={{ position: "relative", paddingLeft: 30, borderLeft: "2px solid #1e1e2e" }}>
          {crashes.map((c, i) => (
            <div key={i} style={{ position: "relative", marginBottom: 40 }}>
              <div style={{
                position: "absolute", left: -37, top: 4, width: 12, height: 12,
                borderRadius: "50%", background: "#7c3aed", border: "2px solid #0d0d14"
              }} />
              
              <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 8 }}>
                <span style={{ fontSize: 16, fontWeight: 600, color: "#e2e8f0" }}>{c.date}</span>
                <span style={{ fontSize: 13, color: "#64748b" }}>{c.time}</span>
              </div>
              
              <div style={{
                background: "rgba(19, 19, 31, 0.4)",
                backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)",
                border: "1px solid #1e1e2e",
                borderRadius: 12, padding: 24,
                boxShadow: "0 0 12px #7c3aed18",
                animation: "fadeSlideUp 0.5s ease both",
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 20, marginBottom: 16 }}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <h3 style={{ fontSize: 16, fontWeight: 600, color: "#ef4444", margin: "0 0 6px 0", letterSpacing: "0.3px" }}>
                      Automated Root Cause
                    </h3>
                    <div style={{
                      color: "#64748b",
                      fontSize: 13,
                      lineHeight: 1.5,
                      overflowWrap: "anywhere",
                      wordBreak: "break-word",
                      maxWidth: "100%",
                    }}>
                      {c.rootCause || "Unknown Application Failure"}
                    </div>
                  </div>
                  <div style={{ textAlign: "right", flexShrink: 0, minWidth: 72 }}>
                    <div style={{ fontSize: 24, fontWeight: 600, color: "#e2e8f0", whiteSpace: "nowrap" }}>
                      {Number.isFinite(c.score) ? c.score.toFixed(1) : "N/A"}
                    </div>
                    <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", letterSpacing: 1, whiteSpace: "nowrap" }}>Anomaly Score</div>
                  </div>
                </div>

                <div style={{ background: "rgba(13, 13, 20, 0.3)", border: "1px solid #1e1e2e", borderRadius: 8, padding: "16px" }}>
                  <div style={{ fontSize: 11, color: "#64748b", marginBottom: 12, display: "flex", justifyContent: "space-between", textTransform: "uppercase", letterSpacing: "1px", fontWeight: 600 }}>
                    <span>Associated Anomalies</span>
                    <span>{c.anomalies || 0} Events</span>
                  </div>
                  {c.events && c.events.length > 0 ? (
                    <ul style={{ listStyle: "none", padding: 0, margin: 0, fontSize: 12, color: "#e2e8f0", fontFamily: "'Roboto Mono', monospace" }}>
                      {c.events.slice(0, 3).map((evt, idx) => (
                        <li key={idx} style={{ padding: "4px 0", borderTop: idx > 0 ? "1px dashed #1e1e2e" : "none", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                          <span style={{ color: "#ef4444", marginRight: 8, fontWeight: "bold" }}>{evt.level}</span>
                          {evt.message}
                        </li>
                      ))}
                      {c.events.length > 3 && (
                        <li style={{ padding: "4px 0", color: "#64748b", fontStyle: "italic" }}>+ {c.events.length - 3} more...</li>
                      )}
                    </ul>
                  ) : (
                    <div style={{ fontSize: 12, color: "#64748b" }}>No expanded details available</div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
