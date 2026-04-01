import { useState, useEffect } from "react";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";
import { fetchStats, fetchTimeline, fetchCrashes, fetchAnomalies, triggerPipeline } from "./api.js";

const API_BASE = (process.env.REACT_APP_API_URL || "http://localhost:8000").replace(/\/$/, "");

// Mock data configuration for UI robustness when API is offline
const MOCK_ANOMALIES = [
  { id: 1, time: "22:54:37", level: "ERROR", message: "Kernel-Power Event 41 unexpected shutdown detected", score: -0.15, isRootCause: true, cluster: 2 },
  { id: 2, time: "22:54:35", level: "INFO",  message: "RSC offload failed on Hyper-V network adapter", score: -0.12, isRootCause: true, cluster: 2 },
  { id: 3, time: "22:54:30", level: "WARN",  message: "Driver unload requested: vmbushid.sys", score: -0.08, isRootCause: false, cluster: 1 },
];

const CRASH_HISTORY = [
  { id: 1, date: "2026-03-08", time: "22:54:37", rootCause: "Kernel-Power / WSL Adapter", anomalies: 15, score: -0.15 },
];

const TIMELINE_DATA = Array.from({ length: 24 }, (_, i) => ({
  hour: `${String(i).padStart(2, "0")}:00`,
  score: i === 22 ? 0.62 : i === 23 ? 0.85 : Math.random() * 0.12,
}));

// Setup logic for display conversions
function formatDisplayDate(dateString) {
  if (!dateString) return "N/A";
  const parsed = new Date(`${dateString}T00:00:00`);
  if (Number.isNaN(parsed.getTime())) return dateString;
  return parsed.toLocaleDateString(undefined, { month: "short", day: "2-digit" });
}

function formatDisplayTime(timeString) {
  if (!timeString) return "Unknown time";
  const normalized = typeof timeString === "string" && timeString.length <= 8 ? `1970-01-01T${timeString}` : timeString;
  const parsed = new Date(normalized);
  if (Number.isNaN(parsed.getTime())) return String(timeString);
  return parsed.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
}

function formatTimelineData(items) {
  if (!Array.isArray(items) || items.length === 0) return TIMELINE_DATA;
  return items.map((item, index) => {
    const rawHour = typeof item?.hour === "string" ? item.hour : "";
    const parsed = rawHour ? new Date(rawHour) : null;
    const label = parsed && !Number.isNaN(parsed.getTime())
      ? parsed.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false })
      : `${String(index).padStart(2, "0")}:00`;
    return { hour: label, score: Math.abs(Number(item?.score) || 0) };
  });
}

function formatCrashRows(items) {
  if (!Array.isArray(items) || items.length === 0) return CRASH_HISTORY;
  return items.map((item, index) => ({
    id: index + 1, date: item?.date || "", time: item?.time || "",
    rootCause: item?.rootCause || "Unknown root cause", anomalies: Number(item?.anomalies) || 0, score: Number(item?.score) || 0,
  }));
}

function formatAnomalyRows(items) {
  if (!Array.isArray(items) || items.length === 0) return MOCK_ANOMALIES;
  return items.map((item, index) => ({
    id: index + 1, time: formatDisplayTime(item?.time), level: item?.level || "INFO",
    message: item?.message || "No message available", score: Number(item?.score) || 0,
    isRootCause: Boolean(item?.isRootCause), cluster: item?.cluster ?? "",
  }));
}

function buildStats(stats) {
  return [
    { label: "Hardware Pulse", value: "ACTIVE", sub: "tracking psutil host telemetry" },
    { label: "Total Crashes", value: String(stats?.totalCrashes ?? 0), sub: "historical backend data" },
    { label: "Last Crash", value: formatDisplayDate(stats?.lastCrash?.date), sub: formatDisplayTime(stats?.lastCrash?.time) },
    { label: "DBSCAN Focus", value: stats?.rootCause ? String(stats.rootCause).slice(0, 16) : "None", sub: stats?.rootCause ? "latest event" : "no recent data" },
    { label: "Anomalies", value: String(stats?.anomalyCount ?? 0), sub: "scored extreme (< -0.05)" },
  ];
}

// Design Badges
const LEVEL_STYLES = {
  ERROR: { bg: "rgba(255,59,59,0.12)", text: "#ff5f5f", dot: "#ff3b3b" },
  WARN: { bg: "rgba(255,180,0,0.12)", text: "#ffbb33", dot: "#ffaa00" },
  INFO: { bg: "rgba(0,210,255,0.10)", text: "#33ddff", dot: "#00c8f0" },
};

function LevelBadge({ level }) {
  const s = LEVEL_STYLES[level] || LEVEL_STYLES.INFO;
  return (
    <span style={{
      background: s.bg, color: s.text, border: `1px solid ${s.dot}33`, borderRadius: 4, padding: "2px 8px",
      fontSize: 11, fontFamily: "monospace", letterSpacing: 1, display: "inline-flex", alignItems: "center", gap: 5,
    }}>
      <span style={{ width: 5, height: 5, borderRadius: "50%", background: s.dot, display: "inline-block" }} />
      {level}
    </span>
  );
}

function ScoreBadge({ score }) {
  const intensity = Math.min(1, Math.abs(score) / 0.2);
  const r = Math.round(255 * intensity);
  const g = Math.round(255 * (1 - intensity));
  return (
    <span style={{
      fontFamily: "monospace", fontSize: 12, color: `rgb(${r},${g},80)`, background: `rgba(${r},${g},80,0.1)`,
      border: `1px solid rgba(${r},${g},80,0.25)`, borderRadius: 4, padding: "2px 8px",
    }}>
      {score.toFixed(3)}
    </span>
  );
}

function StatCard({ label, value, sub, index }) {
  return (
    <div className="stat-card" style={{
      background: "rgba(22, 27, 34, 0.4)", 
      border: "1px solid rgba(255,255,255,0.03)",
      boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
      backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)",
      borderRadius: 12, padding: "24px", display: "flex", flexDirection: "column", gap: 8,
      animation: "fadeSlideUp 0.5s ease both", animationDelay: `${index * 0.08}s`,
    }}>
      <span style={{ fontSize: 11, color: "#8b949e", letterSpacing: "1.5px", textTransform: "uppercase", fontWeight: 500 }}>{label}</span>
      <span style={{ fontSize: 28, fontFamily: "'Roboto', sans-serif", fontWeight: 600, color: "#e6edf3", lineHeight: 1.1, letterSpacing: "-0.5px" }}>{value}</span>
      <span style={{ fontSize: 12, color: "#6e7681", letterSpacing: "0.2px" }}>{sub}</span>
    </div>
  );
}

function LiveDot() {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, color: "#3fb950" }}>
      <span style={{
        width: 8, height: 8, borderRadius: "50%", background: "#3fb950", boxShadow: "0 0 0 0 rgba(63,185,80,0.6)",
        animation: "livePulse 1.6s ease-out infinite", display: "inline-block",
      }} />
      LIVE AGENT
    </span>
  );
}

export default function Dashboard({ onNavigate = () => {} }) {
  const [tab, setTab] = useState("anomalies");
  const [search, setSearch] = useState("");
  const [levelFilter, setLevelFilter] = useState("ALL");
  const [time, setTime] = useState(new Date());
  
  const [stats, setStats] = useState(buildStats(null));
  const [timelineData, setTimelineData] = useState(TIMELINE_DATA);
  const [crashHistory, setCrashHistory] = useState(CRASH_HISTORY);
  const [anomalies, setAnomalies] = useState(MOCK_ANOMALIES);
  
  const [usingFallback, setUsingFallback] = useState(false);
  const [pipelineStatus, setPipelineStatus] = useState(null);
  const [isLoadingDashboard, setIsLoadingDashboard] = useState(true);

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const loadDashboard = async () => {
    setIsLoadingDashboard(true);
    try {
      const [statsData, timelineResponse, crashesResponse, anomaliesResponse] = await Promise.all([
        fetchStats(), fetchTimeline(), fetchCrashes(), fetchAnomalies({ size: 20 }),
      ]);

      const hasLivePayload = [statsData, timelineResponse, crashesResponse, anomaliesResponse].some((item) => item != null);
      setStats(buildStats(statsData));
      setTimelineData(formatTimelineData(timelineResponse));
      setCrashHistory(formatCrashRows(crashesResponse));
      setAnomalies(formatAnomalyRows(anomaliesResponse));
      setUsingFallback(!hasLivePayload);
    } catch (error) {
      setStats(buildStats(null));
      setTimelineData(TIMELINE_DATA);
      setCrashHistory(CRASH_HISTORY);
      setAnomalies(MOCK_ANOMALIES);
      setUsingFallback(true);
    } finally {
      setIsLoadingDashboard(false);
    }
  };

  useEffect(() => { loadDashboard(); }, []);

  const latestRootCause = anomalies.find((item) => item.isRootCause) || anomalies[0] || null;
  const filtered = anomalies.filter((a) => (levelFilter === "ALL" || a.level === levelFilter) && a.message.toLowerCase().includes(search.toLowerCase()));

  const handleTriggerPipeline = async () => {
    setPipelineStatus("running");
    await triggerPipeline();
    setTimeout(() => {
      setPipelineStatus("done");
      setTimeout(() => setPipelineStatus(null), 3000);
    }, 500);
  };

  const handlePrintReport = () => {
    window.print();
  };

  return (
    <>
      <style>{`
        @media print {
          body { background: white !important; color: black !important; }
          * { text-shadow: none !important; box-shadow: none !important; }
          .no-print { display: none !important; }
        }
      `}</style>

      <div style={{ minHeight: "100vh", padding: "0 0 40px", fontFamily: "'Roboto', sans-serif" }}>
        
        <header className="no-print" style={{
          background: "rgba(13, 17, 23, 0.75)",
          WebkitBackdropFilter: "blur(12px)",
          backdropFilter: "blur(12px)",
          borderBottom: "1px solid rgba(255, 255, 255, 0.05)", padding: "0 32px",
          position: "sticky", top: 0, zIndex: 100,
          boxShadow: "0 4px 20px rgba(0,0,0,0.15)",
        }}>
          <div style={{ maxWidth: 1200, margin: "0 auto", display: "flex", alignItems: "center", justifyContent: "space-between", height: 60 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
              <div style={{
                width: 32, height: 32, borderRadius: 8, background: "linear-gradient(135deg,#e6734b,#c0392b)",
                display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, boxShadow: "0 0 16px rgba(230,115,75,0.3)",
              }}>!</div>
              <div>
                <div style={{ fontSize: 16, fontWeight: 700, letterSpacing: 0.5 }}>SortiskeOS Center</div>
                <div style={{ fontSize: 10, color: "#6e7681", letterSpacing: 2, textTransform: "uppercase" }}>Intelligent Log Analysis</div>
              </div>
            </div>
            
            <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
              <span style={{ fontFamily: "monospace", fontSize: 12, color: "#6e7681" }}>
                {time.toLocaleTimeString()} · backend: {usingFallback ? "offline" : "online"}
              </span>
              
              <button className="action-btn" onClick={handlePrintReport} style={{
                background: "#238636", border: "1px solid rgba(240,246,252,0.1)", borderRadius: 6, color: "#fff",
                cursor: "pointer", padding: "6px 14px", fontSize: 13, fontWeight: 500,
              }}>
                ⇩ Export Report
              </button>

              <button className="action-btn" onClick={loadDashboard} disabled={isLoadingDashboard} style={{
                background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 6, color: isLoadingDashboard ? "#484f58" : "#8b949e",
                cursor: isLoadingDashboard ? "default" : "pointer", padding: "6px 10px", fontSize: 13, display: "flex", alignItems: "center", gap: 6,
              }}>
                ↻ Refresh
              </button>

              <LiveDot />
            </div>
          </div>
        </header>

        <main style={{ maxWidth: 1200, margin: "0 auto", padding: "28px 32px", display: "flex", flexDirection: "column", gap: 32 }}>
          
          {/* Priority Layer 1: The Root Cause Alert */}
          <div style={{
            background: "rgba(230,115,75,0.02)",
            border: "1px solid rgba(230,115,75,0.12)", borderLeft: "3px solid #e6734b",
            borderRadius: 12, padding: "24px 32px", display: "flex", alignItems: "flex-start", gap: 24,
            boxShadow: "0 4px 20px rgba(0,0,0,0.1)",
            animation: "fadeSlideUp 0.6s ease both",
          }}>
            <div style={{ fontSize: 24, marginTop: 4, fontFamily: "'Roboto Mono', monospace", color: "#e6734b", opacity: 0.8 }}>⚠</div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#e6734b", marginBottom: 8, letterSpacing: "1px", textTransform: "uppercase" }}>
                Root Cause Identified
              </div>
              <div style={{ fontSize: 16, color: "#e6edf3", lineHeight: 1.5 }}>
                {latestRootCause ? latestRootCause.message : "No root cause anomaly detected matching critical thresholds."}
              </div>
              <div style={{ marginTop: 14, display: "flex", gap: 10, flexWrap: "wrap" }}>
                {latestRootCause && [
                  `Level: ${latestRootCause.level}`,
                  `Cluster Node: C${latestRootCause.cluster || "-"}`,
                  `Severity Vector: ${latestRootCause.score.toFixed(3)}`,
                ].map((tag) => (
                  <span key={tag} style={{
                    background: "rgba(230,115,75,0.1)", color: "#e6734b", border: "1px solid rgba(230,115,75,0.2)",
                    borderRadius: 4, padding: "4px 12px", fontSize: 12, fontFamily: "'Roboto Mono', monospace", letterSpacing: 0.5,
                  }}>{tag}</span>
                ))}
              </div>
            </div>
          </div>

          {/* Priority Layer 2: KPIs */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 16 }}>
             {stats.map((s, i) => <StatCard key={s.label} {...s} index={i} />)}
          </div>

          {/* Priority Layer 3: Algorithm Metrics Timeline */}
          <div style={{ 
            background: "rgba(22, 27, 34, 0.4)", 
            border: "1px solid rgba(255,255,255,0.03)", 
            borderRadius: 12, overflow: "hidden",
            boxShadow: "0 4px 24px rgba(0,0,0,0.1)",
            backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)",
          }}>
            <div style={{ padding: "16px 20px", borderBottom: "1px solid #21262d", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <span style={{ fontSize: 14, fontWeight: 600, color: "#c9d1d9", letterSpacing: 0.5 }}>Isolation Forest Telemetry (24h)</span>
              <span style={{ fontSize: 11, color: "#6e7681", fontStyle: "italic" }}>Powered by TF-IDF vectorization & DBSCAN clustering geometry</span>
            </div>
            <div style={{ padding: "20px 10px 10px" }}>
              <ResponsiveContainer width="100%" height={240}>
                <AreaChart data={timelineData} margin={{ top: 4, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="scoreGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#e6734b" stopOpacity={0.4} />
                      <stop offset="95%" stopColor="#e6734b" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#21262d" vertical={false} />
                  <XAxis dataKey="hour" tick={{ fontSize: 11, fill: "#8b949e", fontFamily: "'Roboto Mono', monospace" }} tickLine={false} axisLine={false} interval="preserveStartEnd" />
                  <YAxis tick={{ fontSize: 11, fill: "#8b949e", fontFamily: "'Roboto Mono', monospace" }} tickLine={false} axisLine={false} />
                  <Tooltip content={<div style={{background: "#0d1117", border: "1px solid #30363d", padding: "8px", borderRadius: "6px", color: "#e6734b", fontFamily: "monospace"}}>Score Evaluated</div>} />
                  <Area type="monotone" dataKey="score" stroke="#e6734b" strokeWidth={2} fill="url(#scoreGrad)" dot={false} activeDot={{ r: 4, fill: "#e6734b" }} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Priority Layer 4: Raw Clustering Feeds */}
          <div style={{ 
            background: "rgba(22, 27, 34, 0.3)", 
            border: "1px solid rgba(255,255,255,0.03)", 
            borderRadius: 12, overflow: "hidden",
            boxShadow: "0 4px 24px rgba(0,0,0,0.1)",
            backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)",
          }}>
             <div style={{ padding: "16px 20px", borderBottom: "1px solid #21262d", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <span style={{ fontSize: 14, fontWeight: 600, color: "#c9d1d9", letterSpacing: 0.5 }}>Recent Evaluated Anomalies ({filtered.length})</span>
               <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                  <select value={levelFilter} onChange={(e) => setLevelFilter(e.target.value)} style={{ background: "#0d1117", border: "1px solid #30363d", color: "#c9d1d9", borderRadius: 4, padding: "6px 10px", fontSize: 12 }}>
                    {["ALL", "ERROR", "WARN", "INFO"].map((l) => <option key={l}>{l}</option>)}
                  </select>
                </div>
            </div>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead style={{ background: "rgba(13, 17, 23, 0.3)", borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                <tr>
                   <th style={{ padding: "14px 20px", color: "#8b949e", textAlign: "left", fontSize: 11, fontWeight: 500, letterSpacing: "1.2px", textTransform: "uppercase" }}>Time</th>
                   <th style={{ padding: "14px 20px", color: "#8b949e", textAlign: "left", fontSize: 11, fontWeight: 500, letterSpacing: "1.2px", textTransform: "uppercase" }}>Level</th>
                   <th style={{ padding: "14px 20px", color: "#8b949e", textAlign: "left", fontSize: 11, fontWeight: 500, letterSpacing: "1.2px", textTransform: "uppercase" }}>Message</th>
                   <th style={{ padding: "14px 20px", color: "#8b949e", textAlign: "left", fontSize: 11, fontWeight: 500, letterSpacing: "1.2px", textTransform: "uppercase" }}>Score</th>
                   <th style={{ padding: "14px 20px", color: "#8b949e", textAlign: "center", fontSize: 11, fontWeight: 500, letterSpacing: "1.2px", textTransform: "uppercase" }}>Root Cause</th>
                   <th style={{ padding: "14px 20px", color: "#8b949e", textAlign: "left", fontSize: 11, fontWeight: 500, letterSpacing: "1.2px", textTransform: "uppercase" }}>Cluster</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((a, i) => (
                  <tr key={a.id} className="table-row" style={{ borderBottom: "1px solid rgba(255,255,255,0.03)" }}>
                    <td style={{ padding: "14px 20px", fontFamily: "'Roboto Mono', monospace", fontSize: 12, color: "#8b949e" }}>{a.time}</td>
                    <td style={{ padding: "14px 20px" }}><LevelBadge level={a.level} /></td>
                    <td style={{ padding: "14px 20px", fontSize: 13, color: "#c9d1d9", maxWidth: 400 }}><span style={{ display: "block", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{a.message}</span></td>
                    <td style={{ padding: "14px 20px" }}><ScoreBadge score={a.score} /></td>
                    <td style={{ padding: "14px 20px", textAlign: "center" }}>{a.isRootCause ? <span style={{ color: "#3fb950", fontSize: 16 }}>OK</span> : <span style={{ color: "#484f58" }}>-</span>}</td>
                     <td style={{ padding: "14px 20px", fontFamily: "'Roboto Mono', monospace", fontSize: 12, color: "#6e7681" }}>C{a.cluster}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
        </main>
      </div>
    </>
  );
}
