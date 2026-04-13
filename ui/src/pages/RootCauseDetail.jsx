import { useEffect, useState } from "react";
import { fetchRootCause } from "../api.js";
import PageHeader from "../components/PageHeader.jsx";
 
const DEFAULT_CLUSTER = {
  id: 2, label: "Primary - Kernel / Network", isRoot: true,
  anomalies: 15, score: -0.15, confidence: 94,
  events: [
    { time: "22:54:37", id: "1", source: "Root cause", msg: "System did not shut down cleanly", score: -0.15 },
    { time: "22:54:35", id: "2", source: "Backend log", msg: "RSC offload failed on network adapter", score: -0.12 },
  ],
  description: "Fallback cluster data shown when the API is unavailable.",
  fix: "Verify the FastAPI and Elasticsearch services are running and then refresh this page.",
  suggestion: {},
};
 
const MODEL_STATS = [
  { label: "Algorithm",     value: "Isolation Forest" },
  { label: "Contamination", value: "0.05 (5%)" },
  { label: "Clustering",    value: "Elasticsearch + FastAPI" },
  { label: "Run mode",      value: "Live API" },
];
 
import { formatTimeShort as formatTime } from "../utils/timeFormat.js";
 
function ScoreBar({ score, max = 0.2 }) {
  const pct = Math.min(100, (Math.abs(score) / max) * 100);
  const r = Math.round(255 * (pct / 100));
  const g = Math.round(255 * (1 - pct / 100));
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <div style={{ flex: 1, height: 5, background: "#1e1e2e", borderRadius: 3, overflow: "hidden" }}>
        <div style={{
          width: `${pct}%`, height: "100%",
          background: `linear-gradient(90deg, rgb(${r},${g},80), rgb(${r},${g},40))`,
          borderRadius: 3,
          transition: "width 1s ease",
        }} />
      </div>
      <span style={{ fontFamily: "monospace", fontSize: 11, color: `rgb(${r},${g},80)`, minWidth: 44 }}>
        {score.toFixed(3)}
      </span>
    </div>
  );
}
 
function ConfidenceRing({ pct, isRoot }) {
  const r = 28;
  const circ = 2 * Math.PI * r;
  const dash = (pct / 100) * circ;
  return (
    <svg width={70} height={70} style={{ transform: "rotate(-90deg)" }}>
      <circle cx={35} cy={35} r={r} fill="none" stroke="#1e1e2e" strokeWidth={5} />
      <circle cx={35} cy={35} r={r} fill="none"
        stroke={isRoot ? "#7c3aed" : "#1e1e2e"}
        strokeWidth={5}
        strokeDasharray={`${dash} ${circ}`}
        strokeLinecap="round"
        style={{ transition: "stroke-dasharray 1.2s ease" }}
      />
      <text x={35} y={35} textAnchor="middle" dominantBaseline="middle"
        fill={isRoot ? "#7c3aed" : "#64748b"}
        fontSize={13} fontWeight={700} fontFamily="monospace"
        style={{ transform: "rotate(90deg)", transformOrigin: "35px 35px" }}>
        {pct}%
      </text>
    </svg>
  );
}
 
function CopyCommandButton({ cmd }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    if (navigator.clipboard) {
      navigator.clipboard.writeText(cmd);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <button
      onClick={handleCopy}
      style={{
        background: "#7c3aed",
        color: "white",
        border: "none",
        borderRadius: 4,
        padding: "2px 8px",
        fontSize: 11,
        cursor: "pointer",
      }}
    >
      {copied ? "Copied!" : "Copy"}
    </button>
  );
}

export default function RootCauseDetail({ onBack }) {
  const [cluster, setCluster]         = useState(DEFAULT_CLUSTER);
  const [usingFallback, setUsingFallback] = useState(false);
  const [isLoading, setIsLoading]     = useState(true);
 
  useEffect(() => {
    let cancelled = false;
 
    async function loadRootCause() {
      setIsLoading(true);
      try {
        const data = await fetchRootCause();
        if (cancelled) return;
 
        if (data && data.clusterId != null) {
          setCluster({
            id:          data.clusterId,
            label:       data.label,
            isRoot:      data.clusterId !== 0,
            anomalies:   data.anomalyCount,
            score:       data.topScore,
            confidence:  data.confidence,
            events: (data.events || []).map((e) => ({
              time:   formatTime(e.time),
              id:     e.eventId || e.id || "",
              source: e.source  || "Backend log",
              msg:    e.message || "",
              score:  e.score   || 0,
            })),
            description: data.description,
            fix:         data.fix,
            suggestion:  data.suggestion || {},
          });
          setUsingFallback(false);
        } else {
          setCluster(DEFAULT_CLUSTER);
          setUsingFallback(true);
        }
      } catch {
        if (cancelled) return;
        setCluster(DEFAULT_CLUSTER);
        setUsingFallback(true);
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }
 
    loadRootCause();
    return () => { cancelled = true; };
  }, []);
 
  return (
    <>
      <style>{`
        @keyframes fadeUp { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }
      `}</style>
 
      <PageHeader 
        title="Root Cause Analysis" 
        subtitle={usingFallback ? "Local Buffer" : "FastAPI Anomaly Clusters"} 
        usingFallback={usingFallback}
      />

      <div style={{ padding: "0 32px 32px", maxWidth: 1200, margin: "0 auto", fontFamily: "'Roboto', sans-serif" }}>
        <div style={{ height: 32 }} /> {/* Spacing spacer replacing old header mb */}
 
        {/* Model stats strip */}
        <div style={{
          background: "rgba(19, 19, 31, 0.4)", backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)",
          border: "1px solid #1e1e2e",
          borderRadius: 12, padding: "16px 20px", marginBottom: 24,
          display: "flex", gap: 0, overflowX: "auto",
          animation: "fadeUp 0.4s ease 0.05s both",
          boxShadow: "0 0 12px #7c3aed18",
        }}>
          {MODEL_STATS.map((s, i) => (
            <div key={s.label} style={{
              padding: "0 20px",
              borderLeft: i === 0 ? "none" : "1px solid #1e1e2e",
              minWidth: 90,
            }}>
              <div style={{ fontSize: 9, color: "#64748b", letterSpacing: 2, textTransform: "uppercase", marginBottom: 4 }}>{s.label}</div>
              <div style={{ fontSize: 12, color: "#e2e8f0", fontFamily: "monospace" }}>{s.value}</div>
            </div>
          ))}
        </div>
 
        {/* Cluster card */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={{
            background: cluster.isRoot ? "rgba(124, 58, 237, 0.02)" : "rgba(19, 19, 31, 0.3)",
            backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)",
            border: `1px solid #1e1e2e`,
            borderLeft: `3px solid ${cluster.isRoot ? "#7c3aed" : "rgba(255,255,255,0.1)"}`,
            borderRadius: 12, overflow: "hidden",
            animation: "fadeUp 0.4s ease 0.1s both",
            boxShadow: "0 0 12px #7c3aed18",
            opacity: isLoading ? 0.4 : 1,
            transition: "opacity 0.3s",
            pointerEvents: isLoading ? "none" : "auto",
          }}>
 
            {/* Cluster header */}
            <div style={{
              padding: "16px 20px", borderBottom: "1px solid #1e1e2e",
              display: "flex", alignItems: "center", gap: 20,
              background: "rgba(13, 13, 20, 0.3)",
            }}>
              <ConfidenceRing pct={cluster.confidence} isRoot={cluster.isRoot} />
              <div style={{ flex: 1 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
                  <span style={{ fontFamily: "monospace", fontSize: 11, color: "#64748b" }}>CLUSTER #{cluster.id ?? "N/A"}</span>
                  {cluster.isRoot && (
                    <span style={{
                      background: "rgba(124, 58, 237, 0.12)", color: "#7c3aed",
                      border: "1px solid #7c3aed33",
                      borderRadius: 4, padding: "1px 8px", fontSize: 10, letterSpacing: 1,
                    }}>ROOT CAUSE</span>
                  )}
                </div>
                <div style={{ fontSize: 14, fontWeight: 600, color: "#e2e8f0", fontFamily: "'Roboto', sans-serif" }}>{cluster.label}</div>
                <div style={{ fontSize: 12, color: "#64748b", marginTop: 4, lineHeight: 1.5 }}>{cluster.description}</div>
              </div>
              <div style={{ textAlign: "right", minWidth: 80 }}>
                <div style={{ fontSize: 22, fontWeight: 600, color: cluster.isRoot ? "#7c3aed" : "#64748b", fontFamily: "'Roboto', sans-serif" }}>{cluster.anomalies}</div>
                <div style={{ fontSize: 10, color: "#64748b", letterSpacing: 1 }}>ANOMALIES</div>
              </div>
            </div>
 
            <div style={{ padding: "12px 20px" }}>
              <div style={{ fontSize: 10, color: "#64748b", letterSpacing: 2, textTransform: "uppercase", marginBottom: 10 }}>Log Events</div>
              {cluster.events.map((ev, ei) => (
                <div key={ei} className="table-row" style={{
                  display: "grid", gridTemplateColumns: "70px 50px 160px 1fr 120px",
                  gap: 12, alignItems: "center",
                  padding: "10px 14px",
                  borderBottom: ei < cluster.events.length - 1 ? "1px solid #1e1e2e" : "none",
                }}>
                  <span style={{ fontFamily: "monospace", fontSize: 11, color: "#64748b" }}>{ev.time}</span>
                  <span style={{
                    background: "rgba(239,68,68,0.1)", color: "#ef4444",
                    border: "1px solid #ef444433",
                    borderRadius: 4, padding: "1px 6px", fontSize: 10,
                    fontFamily: "monospace", textAlign: "center",
                  }}>ID {ev.id}</span>
                  <span style={{ fontSize: 11, color: "#64748b", fontFamily: "monospace", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{ev.source}</span>
                  <span style={{ fontSize: 12, color: "#e2e8f0", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{ev.msg}</span>
                  <ScoreBar score={ev.score} />
                </div>
              ))}
            </div>
 
            {/* Fix suggestion */}
            {cluster.suggestion && (
              <div style={{
                margin: "0 20px 16px",
                background: "rgba(19, 19, 31, 0.4)", border: "1px solid #1e1e2e",
                borderRadius: 12, padding: "16px 20px", display: "flex", flexDirection: "column", gap: 12
              }}>
                {!cluster.suggestion.category ? (
                  <div style={{ fontSize: 13, color: "#64748b" }}>
                    No suggestion available
                  </div>
                ) : (
                  <>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{
                        background: "rgba(124, 58, 237, 0.12)", color: "#7c3aed",
                        border: "1px solid #7c3aed33", borderRadius: 4, padding: "2px 8px", fontSize: 11, fontWeight: 500, letterSpacing: 1
                      }}>
                        {cluster.suggestion.category}
                      </span>
                      {cluster.suggestion.confidence && (
                        <span style={{
                          background: cluster.suggestion.confidence === "High" ? "rgba(239, 68, 68, 0.1)" : cluster.suggestion.confidence === "Medium" ? "rgba(245, 158, 11, 0.1)" : "rgba(34, 197, 94, 0.1)",
                          color: cluster.suggestion.confidence === "High" ? "#ef4444" : cluster.suggestion.confidence === "Medium" ? "#f59e0b" : "#22c55e",
                          border: `1px solid ${cluster.suggestion.confidence === "High" ? "#ef444433" : cluster.suggestion.confidence === "Medium" ? "#f59e0b33" : "#22c55e33"}`,
                          borderRadius: 4, padding: "2px 8px", fontSize: 11, fontWeight: 500
                        }}>
                          {cluster.suggestion.confidence}
                        </span>
                      )}
                    </div>
                    
                    <div style={{ fontSize: 13, color: "#e2e8f0", lineHeight: 1.5 }}>
                      {cluster.suggestion.likely_cause}
                    </div>

                    {cluster.suggestion.investigate && cluster.suggestion.investigate.length > 0 && (
                      <ul style={{ margin: 0, paddingLeft: 16, color: "#e2e8f0", fontSize: 13, lineHeight: 1.6 }}>
                        {cluster.suggestion.investigate.map((item, idx) => (
                          <li key={idx} style={{ paddingLeft: 4, marginBottom: 4 }}>{item}</li>
                        ))}
                      </ul>
                    )}

                    {cluster.suggestion.commands && cluster.suggestion.commands.length > 0 && (
                      <div style={{
                        background: "#0d0d14", border: "1px solid #1e1e2e", borderRadius: 6, padding: "10px 14px",
                        fontFamily: "monospace", fontSize: 12, color: "#e2e8f0", overflowX: "auto"
                      }}>
                        {cluster.suggestion.commands.map((cmd, idx) => (
                          <div key={idx} style={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                            gap: 16,
                            marginBottom: idx < cluster.suggestion.commands.length - 1 ? 8 : 0,
                            whiteSpace: "nowrap"
                          }}>
                            <div style={{ overflowX: "auto" }}>
                              <span style={{ color: "#7c3aed", marginRight: 8, opacity: 0.8 }}>$</span>{cmd}
                            </div>
                            <CopyCommandButton cmd={cmd} />
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
 
          </div>
        </div>
 
      </div>
    </>
  );
}