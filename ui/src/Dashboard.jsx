import { useState, useEffect } from "react";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, ReferenceDot,
  BarChart, Bar, PieChart, Pie, Cell
} from "recharts";
import { fetchStats, fetchTimeline, fetchCrashes, fetchAnomalies, triggerPipeline } from "./api.js";
import { Toast } from "./App.jsx";
import ReportExport from "./components/ReportExport.jsx";
import PageHeader from "./components/PageHeader.jsx";
import { formatTimeShort as formatDisplayTime, formatTimeLabel as formatChartTimeLabel } from "./utils/timeFormat.js";
import { toDisplayText, toDisplayNumber, summarizeRootCause } from "./utils/displayValue.js";

function useCountUp(target, duration = 1000) {
  const [count, setCount] = useState(0);
  useEffect(() => {
    if (!target) return;
    let start = 0;
    const increment = target / (duration / 16);
    const timer = setInterval(() => {
      start += increment;
      if (start >= target) {
        setCount(target);
        clearInterval(timer);
      } else {
        setCount(Math.floor(start));
      }
    }, 16);
    return () => clearInterval(timer);
  }, [target]);
  return count;
}

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

function formatTimelineData(items) {
  if (!Array.isArray(items)) return TIMELINE_DATA;
  if (items.length === 0) return [];
  return items.map((item, index) => {
    const rawHour = typeof item?.hour === "string" ? item.hour : "";
    const label = rawHour ? formatChartTimeLabel(rawHour) : `${String(index).padStart(2, "0")}:00`;
    return { hour: label, score: Math.abs(Number(item?.score) || 0) };
  });
}

function getTimelineDomain(items) {
  if (!Array.isArray(items) || items.length === 0) return [0, 1];
  const scores = items.map((item) => Number(item?.score) || 0);
  const maxScore = Math.max(...scores, 0);
  if (maxScore <= 0) return [0, 1];
  const paddedMax = maxScore < 0.1 ? Number((maxScore * 1.6).toFixed(3)) : Number((maxScore * 1.2).toFixed(3));
  return [0, paddedMax || 1];
}

function formatCrashRows(items) {
  if (!Array.isArray(items)) return CRASH_HISTORY;
  return items.map((item, index) => ({
    id: index + 1,
    date: toDisplayText(item?.date, ""),
    time: toDisplayText(item?.time, ""),
    rootCause: summarizeRootCause(toDisplayText(item?.rootCause, "Unknown root cause")),
    anomalies: toDisplayNumber(item?.anomalies, 0),
    score: toDisplayNumber(item?.score, 0),
  }));
}

function formatAnomalyRows(items) {
  if (!Array.isArray(items)) return MOCK_ANOMALIES;
  return items.map((item, index) => ({
    id: index + 1,
    time: formatDisplayTime(toDisplayText(item?.time, "")),
    level: toDisplayText(item?.level, "INFO"),
    message: summarizeRootCause(toDisplayText(item?.message, "No message available")),
    score: toDisplayNumber(item?.score, 0),
    isRootCause: Boolean(item?.isRootCause),
    cluster: toDisplayText(item?.cluster, ""),
    count: Math.max(1, toDisplayNumber(item?.count, 1)),
    suggestion: item?.suggestion || {},
    rootCause: summarizeRootCause(toDisplayText(item?.rootCause, "")),
  }));
}

function buildStats(stats) {
  return [
    { label: "Hardware Pulse", value: "ACTIVE", sub: "tracking psutil host telemetry" },
    { label: "Crash Events", value: String(stats?.totalCrashes ?? 0), sub: "actual system failures" },
    { label: "Issues in Crash Window", value: String(stats?.totalIssues ?? 0), sub: "associated anomaly occurrences" },
    { label: "Last Crash", value: formatDisplayDate(stats?.lastCrash?.date), sub: formatDisplayTime(stats?.lastCrash?.timestamp || stats?.lastCrash?.time) },
    { label: "DBSCAN Focus", value: summarizeRootCause(stats?.rootCause), sub: stats?.rootCause ? "latest event" : "no recent data" },
    { label: "Anomalies in Crash Window", value: String(stats?.anomalyCount ?? 0), sub: "unique anomaly groups" },
  ];
}

// Design Badges
const LEVEL_STYLES = {
  ERROR: { bg: "rgba(239, 68, 68, 0.12)", text: "#ef4444", dot: "#ef4444", shadow: "0 0 8px #ef444466" },
  WARN: { bg: "rgba(245, 158, 11, 0.12)", text: "#f59e0b", dot: "#f59e0b", shadow: "0 0 8px #f59e0b66" },
  INFO: { bg: "rgba(59, 130, 246, 0.10)", text: "#3b82f6", dot: "#3b82f6", shadow: "none" },
};

function LevelBadge({ level }) {
  const s = LEVEL_STYLES[level] || LEVEL_STYLES.INFO;
  return (
    <span style={{
      background: s.bg, color: s.text, boxShadow: s.shadow,
      borderRadius: 999, padding: "2px 10px",
      fontSize: 11, fontFamily: "monospace", fontWeight: 600, letterSpacing: 1, 
      display: "inline-flex", alignItems: "center", gap: 5,
    }}>
      <span style={{ width: 5, height: 5, borderRadius: "50%", background: s.dot, display: "inline-block", boxShadow: s.shadow }} />
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
  const isNumberFormat = typeof value === 'string' ? /^\d+$/.test(value) : Number.isInteger(value);
  const targetNumber = isNumberFormat ? parseInt(value, 10) : 0;
  const animatedCount = useCountUp(targetNumber);

  return (
    <div className="stat-card" style={{
      padding: "16px 20px", display: "flex", flexDirection: "column", gap: 4,
      background: "rgba(255,255,255,0.01)", border: "1px solid rgba(255,255,255,0.03)", borderRadius: 8,
      animation: "fadeSlideUp 0.5s ease both", animationDelay: `${index * 0.08}s`,
    }}>
      <span style={{ fontSize: 11, color: "#64748b", letterSpacing: "1.5px", textTransform: "uppercase", fontWeight: 500 }}>{label}</span>
      
      <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
        <span style={{
          fontSize: 26,
          fontFamily: "'Roboto', sans-serif",
          fontWeight: 600,
          color: "#e2e8f0",
          lineHeight: 1.1,
          letterSpacing: "-0.5px",
          ...(label.toUpperCase() === "DBSCAN FOCUS" 
            ? { overflow: 'hidden', 
                textOverflow: 'ellipsis', 
                whiteSpace: 'nowrap', 
                maxWidth: '100%' } 
            : {})
        }}>
          {isNumberFormat ? animatedCount : value}
        </span>
      </div>
      <span style={{ fontSize: 12, color: "#64748b", letterSpacing: "0.2px" }}>{sub}</span>
    </div>
  );
}

// LiveDot is now handled in PageHeader.jsx components

export default function Dashboard({ onNavigate = () => {} }) {
  const [tab, setTab] = useState("anomalies");
  const [search, setSearch] = useState("");
  const [levelFilter, setLevelFilter] = useState("ALL");
  
  const [stats, setStats] = useState(buildStats(null));
  const [timelineData, setTimelineData] = useState(TIMELINE_DATA);
  const [crashHistory, setCrashHistory] = useState(CRASH_HISTORY);
  const [anomalies, setAnomalies] = useState(MOCK_ANOMALIES);
  const [rawStats, setRawStats] = useState(null);
  const [crashTime, setCrashTime] = useState(null);
  
  const [usingFallback, setUsingFallback] = useState(false);
  const [pipelineStatus, setPipelineStatus] = useState(null);
  const [isLoadingDashboard, setIsLoadingDashboard] = useState(true);
  const [toast, setToast] = useState(null);
  const [tamperDetected, setTamperDetected] = useState(false);
  const [antiforensics, setAntiforensics] = useState({ detected: false, count: 0, events: [] });

  // Dashboard initialization
  useEffect(() => {
    fetchAllData();
  }, []);

  const fetchAllData = async () => {
    setIsLoadingDashboard(true);
    try {
      const [statsData, timelineResponse, crashesResponse, anomaliesResponse] = await Promise.all([
        fetchStats(), fetchTimeline(), fetchCrashes(), fetchAnomalies({ size: 20 }),
      ]);

      const hasLivePayload = [statsData, timelineResponse, crashesResponse, anomaliesResponse].some((item) => item != null);
      setStats(buildStats(statsData));
      
      const cTime = statsData?.crash_time || null;
      setCrashTime(cTime);
      setTimelineData(formatTimelineData(timelineResponse));
      setCrashHistory(formatCrashRows(crashesResponse));
      setAnomalies(formatAnomalyRows(anomaliesResponse));
      setRawStats(statsData);
      setUsingFallback(!hasLivePayload);
      setTamperDetected(statsData?.tamper_detected || false);
      setAntiforensics(statsData?.antiforensics || { detected: false, count: 0, events: [] });
    } catch (error) {
      setStats(buildStats(null));
      setTimelineData(TIMELINE_DATA);
      setCrashHistory(CRASH_HISTORY);
      setAnomalies(MOCK_ANOMALIES);
      setUsingFallback(true);
      setTamperDetected(false);
      setAntiforensics({ detected: false, count: 0, events: [] });
    } finally {
      setIsLoadingDashboard(false);
    }
  };

  // Dashboard initialization hook (removed duplicate)

  const latestRootCause = anomalies.find((item) => item.isRootCause) || anomalies[0] || null;
  const hasScoredAnomalies = Array.isArray(anomalies) && anomalies.length > 0;
  const crashMarkerLabel = formatChartTimeLabel(crashHistory[0]?.time);
  const timelineDomain = getTimelineDomain(timelineData);
  const filtered = anomalies.filter((a) => (levelFilter === "ALL" || a.level === levelFilter) && a.message.toLowerCase().includes(search.toLowerCase()));

  const errorCount = anomalies.filter((a) => a.level === "ERROR").length;
  const warnCount = anomalies.filter((a) => a.level === "WARN").length;
  const infoCount = anomalies.filter((a) => a.level === "INFO").length;
  const totalCount = errorCount + warnCount + infoCount || 1;

  const clusterCounts = anomalies.reduce((acc, curr) => {
    const c = curr.cluster;
    if (c !== undefined && c !== null && c !== "") {
      acc[c] = (acc[c] || 0) + 1;
    }
    return acc;
  }, {});
  const clusterKeys = Object.keys(clusterCounts).sort((a,b)=>a-b);
  const clusterData = clusterKeys.map(k => ({ name: `Cluster ${k}`, value: clusterCounts[k] }));
  const donutColors = ["#7c3aed", "#ec4899", "#3b82f6", "#22c55e", "#f59e0b", "#06b6d4"];

  const severityData = [{
    name: "Severity",
    ERROR: errorCount,
    WARN: warnCount,
    INFO: infoCount,
  }];

  const renderBarLabel = (props) => {
    const { x, y, width, height, value } = props;
    if (!value) return null;
    const percent = (value / totalCount) * 100;
    if (percent <= 8) return null;
    return (
      <text x={x + width / 2} y={y + height / 2 + 1} fill="#ffffff" textAnchor="middle" dominantBaseline="central" fontSize={11} fontWeight={600} style={{ pointerEvents: 'none' }}>
        {percent.toFixed(0)}%
      </text>
    );
  };


  const handleReanalyze = async () => {
    setIsLoadingDashboard(true);
    try {
      await fetch(`${API_BASE}/pipeline/run`, {
        method: 'POST'
      });
    } catch(e) {}
    setTimeout(() => {
      fetchAllData();
      setIsLoadingDashboard(false);
    }, 10000);
  };

  const handleTriggerPipeline = async () => {
    setPipelineStatus("running");
    try {
      await triggerPipeline();
      setToast({ message: 'Pipeline completed successfully', type: 'success' });
    } catch {
      setToast({ message: 'Pipeline failed to run', type: 'error' });
    }
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
          body * { visibility: hidden; }
          #report-export,
          #report-export * { visibility: visible; }
          #report-export {
            display: block !important;
            position: absolute;
            top: 0; left: 0;
            width: 100%;
            font-family: 'Inter', sans-serif;
            color: #000;
            background: #fff;
            padding: 32px 40px;
          }
          .report-header { margin-bottom: 16px; }
          .report-code-block {
            background: #f4f4f4;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 8px 12px;
            font-family: monospace;
            font-size: 12px;
            margin: 4px 0;
          }
          .report-table { 
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
          }
          .report-table th {
            background: #f4f4f4;
            border-bottom: 2px solid #000;
            padding: 6px 8px;
            text-align: left;
          }
          .report-table td {
            padding: 5px 8px;
            border-bottom: 1px solid #eee;
          }
          .report-table tr:nth-child(even) td {
            background: #fafafa;
          }
          .error-row { border-left: 3px solid #cc0000; }
          .summary-box {
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 16px;
            margin: 16px 0;
            background: #fafafa;
          }
          .confidence-high { color: #cc0000; font-weight: 600; }
          .confidence-medium { color: #cc6600; font-weight: 600; }
          .confidence-low { color: #006600; font-weight: 600; }
          .report-footer { margin-top: 24px; }
          .page-number::after { content: counter(page); }
        }
        @keyframes pulse {
          0% { box-shadow: 0 0 0 0 rgba(124, 58, 237, 0.7); }
          70% { box-shadow: 0 0 0 8px rgba(124, 58, 237, 0); }
          100% { box-shadow: 0 0 0 0 rgba(124, 58, 237, 0); }
        }
        .table-row {
          border-bottom: 1px solid transparent;
          transition: all 0.15s ease;
        }
        .table-row:hover {
          background: #1e1e2e;
          transform: translateY(-1px);
        }
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.05); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.15); }
      `}</style>

      <div style={{ minHeight: "100vh", padding: "0 0 40px", fontFamily: "'Roboto', sans-serif" }}>
        
        <PageHeader 
          title="System Dashboard" 
          subtitle="Active Session Monitor" 
          usingFallback={usingFallback}
          pipelineStatus={pipelineStatus}
          actions={
            <>
              <button className="action-btn" onClick={handlePrintReport} style={{
                background: "#238636", border: "1px solid rgba(240,246,252,0.1)", borderRadius: 6, color: "#fff",
                cursor: "pointer", padding: "6px 14px", fontSize: 13, fontWeight: 500,
              }}>
                ⇩ Export Report
              </button>

              <button className="action-btn" onClick={handleReanalyze} disabled={isLoadingDashboard} style={{
                background: "rgba(124, 58, 237, 0.1)", border: "1px solid #7c3aed33", borderRadius: 6, color: isLoadingDashboard ? "#64748b" : "#7c3aed",
                cursor: isLoadingDashboard ? "default" : "pointer", padding: "6px 14px", fontSize: 13, fontWeight: 500, display: "flex", alignItems: "center", gap: 6,
              }}>
                ↻ Re-analyze
              </button>
            </>
          }
        />
        {crashTime && (
          <div style={{
            fontSize: '13px',
            color: '#64748b',
            marginTop: '-24px',
            marginBottom: '24px',
            paddingLeft: '32px'
          }}>
            Crash Report ? {new Date(crashTime).toLocaleString()}
          </div>
        )}

        <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 32px 32px", display: "flex", flexDirection: "column", gap: 32 }}>
          <div style={{ height: 24 }} />

        {tamperDetected && (
          <div style={{
            background: "#ef444420", border: "1px solid #ef4444", color: "#ef4444",
            borderRadius: 8, padding: "10px 16px", fontSize: 13, fontWeight: 500,
          }}>
            ⚠ TAMPER ALERT: Log file was modified before analysis. Results may be unreliable.
          </div>
        )}

        {antiforensics.detected && (
          <div style={{
            background: "#f59e0b20", border: "1px solid #f59e0b", color: "#f59e0b",
            borderRadius: 8, padding: "10px 16px", fontSize: 13, fontWeight: 500,
          }}>
            <div>⚠ ANTI-FORENSICS ALERT: Event logs were cleared {antiforensics.count} time(s) before this crash. Investigation integrity compromised.</div>
            {antiforensics.events?.length > 0 && (
              <details style={{ marginTop: 8, cursor: "pointer", borderTop: "1px solid rgba(245, 158, 11, 0.3)", paddingTop: 8 }}>
                <summary style={{ outline: "none" }}>View removed log sequences</summary>
                <ul style={{ marginTop: 8, marginBottom: 0, paddingLeft: 20, fontFamily: "monospace", fontSize: 12, opacity: 0.9 }}>
                  {antiforensics.events.map((e, idx) => (
                    <li key={idx} style={{ marginBottom: 4, opacity: 0.8 }}>
                      [{e.timestamp}] Windows Event ID {e.event_id} (Channel: {e.channel})
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        )}


          {/* Priority Layer 1: The Root Cause Alert */}
          <div style={{
            background: "rgba(124, 58, 237, 0.02)",
            border: "1px solid #1e1e2e", borderLeft: "3px solid #7c3aed",
            borderRadius: 12, padding: "24px 32px", display: "flex", alignItems: "flex-start", gap: 24,
            boxShadow: "0 0 12px #7c3aed18",
            animation: "fadeSlideUp 0.6s ease both",
          }}>
            <div style={{ fontSize: 24, marginTop: 4, fontFamily: "'Roboto Mono', monospace", color: "#7c3aed", opacity: 0.8 }}>⚠</div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#7c3aed", marginBottom: 8, letterSpacing: "1px", textTransform: "uppercase" }}>
                Root Cause Identified
              </div>
              
              {(() => {
                const s = rawStats?.suggestion?.category ? rawStats.suggestion : (latestRootCause?.suggestion || {});
                const isAnalyzing = !s.category || s.category === "Unknown/Generic Error" || !hasScoredAnomalies;
                const likelyCauseSource = toDisplayText(s.likely_cause, rawStats?.rootCause || latestRootCause?.message || "");
                const summarizedCause = hasScoredAnomalies
                  ? summarizeRootCause(likelyCauseSource)
                  : "Latest crash window has no negative anomaly scores yet. Run Re-analyze after fresh logs are ingested.";

                return (
                  <>
                    <div style={{ fontSize: 24, fontWeight: 800, color: "#e2e8f0", marginBottom: 4 }}>
                      {hasScoredAnomalies
                        ? (isAnalyzing ? "Analyzing..." : toDisplayText(s.category, "Analyzing..."))
                        : "No Scored Anomalies"}
                    </div>
                    
                    {isAnalyzing && (
                      <div style={{ fontSize: 12, color: "#64748b", marginBottom: 12, fontStyle: "italic" }}>
                        Run Re-analyze for fresh results
                      </div>
                    )}

                    <div style={{ fontSize: 15, color: "#94a3b8", marginBottom: 12, whiteSpace: "pre-wrap", overflowWrap: "break-word" }} title={likelyCauseSource}>
                      {summarizedCause}
                    </div>
                    <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center", marginBottom: 12 }}>
                      <span style={{
                        background: "rgba(124, 58, 237, 0.2)", color: "#a78bfa",
                        borderRadius: 4, padding: "2px 8px", fontSize: 11, fontWeight: 700
                      }}>
                        {hasScoredAnomalies ? `${toDisplayText(s.confidence, "Medium")} CONFIDENCE` : "WAITING FOR DATA"}
                      </span>
                      {hasScoredAnomalies && (
                        <>
                          <span style={{ color: "#475569", fontSize: 12 }}>�</span>
                          <span style={{ color: "#94a3b8", fontSize: 12 }}>Cluster C{latestRootCause?.cluster ?? "-"}</span>
                          <span style={{ color: "#475569", fontSize: 12 }}>�</span>
                          <span style={{ color: "#94a3b8", fontSize: 12 }}>Severity {latestRootCause?.score?.toFixed(3) || "0.000"}</span>
                        </>
                      )}
                    </div>

                    {hasScoredAnomalies && s.investigate?.[0] && (
                      <div style={{ 
                        fontSize: 13, color: "#7c3aed", background: "rgba(124, 58, 237, 0.05)", 
                        padding: "8px 12px", borderRadius: 6, border: "1px solid rgba(124, 58, 237, 0.1)"
                      }}>
                        <span style={{ fontWeight: 700 }}>HINT:</span> {toDisplayText(s.investigate[0], "No hint available")}
                      </div>
                    )}
                  </>
                );
              })()}
            </div>
          </div>

          {/* Priority Layer 2: Grid Row 1 (KPI Matrix + Donut) */}
          <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 24 }}>
            {/* KPI Matrix */}
            <div style={{
              background: "#13131f",
              border: "1px solid #1e1e2e",
              borderRadius: 12,
              padding: "20px",
              boxShadow: "0 0 12px #7c3aed18",
              display: "flex", flexDirection: "column", gap: 16
            }}>
              <div style={{ fontSize: 13, color: "#64748b", fontWeight: 500 }}>System Health Matrix</div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
                {stats.slice(0,3).map((s, i) => <StatCard key={s.label} {...s} index={i} />)}
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
                {stats.slice(3,6).map((s, i) => <StatCard key={s.label} {...s} index={i + 3} />)}
              </div>
            </div>

            {/* Cluster Donut */}
            <div style={{
              background: "#13131f",
              border: "1px solid #1e1e2e",
              borderRadius: 12,
              padding: "20px",
              boxShadow: "0 0 12px #7c3aed18",
              display: "flex", flexDirection: "column"
            }}>
               <div style={{ fontSize: 13, color: "#64748b", marginBottom: 8, fontWeight: 500 }}>Open Anomalies by Classification</div>
               <div style={{ flex: 1, display: "flex", position: "relative", alignItems: "center", justifyContent: "center" }}>
                  <ResponsiveContainer width="100%" height={200}>
                    <PieChart>
                      {clusterData.length > 0 ? (
                        <Pie data={clusterData} dataKey="value" nameKey="name" innerRadius={70} outerRadius={100} paddingAngle={2} stroke="none">
                          {clusterData.map((entry, index) => <Cell key={`cell-${index}`} fill={donutColors[index % donutColors.length]} />)}
                        </Pie>
                      ) : (
                        <Pie data={[{ value: 1 }]} dataKey="value" innerRadius={70} outerRadius={100} stroke="none" fill="rgba(255,255,255,0.05)" />
                      )}
                      <Tooltip contentStyle={{ background: "#0d0d14", border: "1px solid #1e1e2e", borderRadius: 6, color: "#e2e8f0" }} itemStyle={{ color: "#e2e8f0" }} />
                    </PieChart>
                  </ResponsiveContainer>
                  <div style={{ position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)", textAlign: "center", pointerEvents: "none" }}>
                    <div style={{ fontSize: 24, fontWeight: 700, color: "#e2e8f0" }}>{rawStats?.anomalyCount ?? 0}</div>
                    <div style={{ fontSize: 10, color: "#64748b", letterSpacing: 1 }}>TOTAL</div>
                  </div>
               </div>
               <div style={{ display: "flex", justifyContent: "center", gap: "12px 16px", flexWrap: "wrap", marginTop: 12 }}>
                 {clusterData.map((entry, index) => (
                   <div key={entry.name} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "#e2e8f0" }}>
                     <span style={{ width: 8, height: 8, borderRadius: "50%", background: donutColors[index % donutColors.length] }} />
                     {entry.name} <span style={{ color: "#64748b" }}>{entry.value}</span>
                   </div>
                 ))}
                 {clusterData.length === 0 && (
                   <div style={{ fontSize: 12, color: "#64748b" }}>No Data</div>
                 )}
               </div>
            </div>
          </div>

          {/* Priority Layer 2.5: Severity Distribution */}
          <div style={{
            background: "#13131f",
            border: "1px solid #1e1e2e",
            borderRadius: 12,
            padding: "16px",
            boxShadow: "0 0 12px #7c3aed18"
          }}>
            <div style={{ fontSize: 13, color: "#64748b", marginBottom: 8, fontWeight: 500 }}>
              Log Severity Distribution
            </div>
            <div style={{ height: 24, width: "100%", borderRadius: 8, overflow: "hidden" }}>
              <ResponsiveContainer width="100%" height={24}>
                <BarChart layout="vertical" data={severityData} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
                  <XAxis type="number" hide domain={[0, totalCount]} />
                  <YAxis type="category" dataKey="name" hide />
                  <Tooltip 
                    cursor={{ fill: 'transparent' }}
                    formatter={(value, name) => [`${value} (${((value / totalCount) * 100).toFixed(1)}%)`, name]}
                    contentStyle={{ background: "#0d0d14", border: "1px solid #1e1e2e", borderRadius: 6, fontSize: 12, color: "#e2e8f0" }}
                    itemStyle={{ padding: 0 }}
                  />
                  <Bar dataKey="ERROR" stackId="a" fill="#ef4444" label={renderBarLabel} isAnimationActive={false} />
                  <Bar dataKey="WARN" stackId="a" fill="#f59e0b" label={renderBarLabel} isAnimationActive={false} />
                  <Bar dataKey="INFO" stackId="a" fill="#3b82f6" label={renderBarLabel} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Priority Layer 3: Algorithm Metrics Timeline */}
          <div style={{ 
            background: "rgba(19, 19, 31, 0.4)", 
            border: "1px solid #1e1e2e", 
            borderRadius: 12, overflow: "hidden",
            boxShadow: "0 0 12px #7c3aed18",
            backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)",
          }}>
            <div style={{ padding: "16px 20px", borderBottom: "1px solid #1e1e2e", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <span style={{ fontSize: 14, fontWeight: 600, color: "#e2e8f0", letterSpacing: 0.5 }}>Isolation Forest Around Crash Time</span>
              <span style={{ fontSize: 11, color: "#64748b", fontStyle: "italic" }}>5-minute anomaly buckets centered on the latest detected crash window</span>
            </div>
            <div style={{ padding: "20px 10px 10px" }}>
              <ResponsiveContainer width="100%" height={240}>
                <AreaChart data={timelineData} margin={{ top: 4, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="scoreGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#7c3aed" stopOpacity={0.8} />
                      <stop offset="100%" stopColor="#7c3aed" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="hour" tick={{ fontSize: 11, fill: "#64748b", fontFamily: "'Roboto Mono', monospace" }} tickLine={false} axisLine={false} interval="preserveStartEnd" />
                  <YAxis
                    domain={timelineDomain}
                    tickFormatter={(value) => Number(value).toFixed(value < 1 ? 2 : 1)}
                    tick={{ fontSize: 11, fill: "#64748b", fontFamily: "'Roboto Mono', monospace" }}
                    tickLine={false}
                    axisLine={false}
                  />
                  {crashMarkerLabel ? <ReferenceDot x={crashMarkerLabel} y={timelineData.find(d => d.hour === crashMarkerLabel)?.score || 0} r={6} fill="#ef4444" stroke="#000" strokeWidth={2} /> : null}
                  <Tooltip content={<div style={{background: "#0d0d14", border: "1px solid #1e1e2e", padding: "8px", borderRadius: "6px", color: "#7c3aed", fontFamily: "monospace"}}>Score Evaluated</div>} />
                  <Area type="monotone" dataKey="score" stroke="#7c3aed" strokeWidth={2} fill="url(#scoreGrad)" dot={false} activeDot={{ r: 4, fill: "#7c3aed" }} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Priority Layer 4: Raw Clustering Feeds */}
          <div style={{ 
            background: "rgba(19, 19, 31, 0.3)", 
            border: "1px solid #1e1e2e", 
            borderRadius: 12, overflow: "hidden",
            boxShadow: "0 0 12px #7c3aed18",
            backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)",
          }}>
             <div style={{ padding: "16px 20px", borderBottom: "1px solid #1e1e2e", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
               <span style={{ fontSize: 14, fontWeight: 600, color: "#e2e8f0", letterSpacing: 0.5 }}>Top Open Alerts ({filtered.length})</span>
               <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
                  <div style={{ position: "relative", display: "flex", alignItems: "center" }}>
                    <span style={{ position: "absolute", left: 12, color: "#64748b", fontSize: 12 }}>🔍</span>
                    <input 
                      type="text" 
                      value={search} 
                      onChange={(e) => setSearch(e.target.value)} 
                      placeholder="Search alerts" 
                      style={{ 
                        background: "#0d0d14", border: "1px solid #1e1e2e", color: "#e2e8f0", 
                        borderRadius: 999, padding: "6px 16px 6px 36px", fontSize: 12, width: 220, outline: "none"
                      }} 
                    />
                  </div>
                  <div style={{ position: "relative" }}>
                    <select value={levelFilter} onChange={(e) => setLevelFilter(e.target.value)} 
                      style={{ 
                        appearance: "none", WebkitAppearance: "none", cursor: "pointer",
                        background: "#0d0d14", border: "1px solid #1e1e2e", color: "#e2e8f0", 
                        borderRadius: 999, padding: "6px 28px 6px 16px", fontSize: 12, outline: "none"
                      }}
                    >
                      <option value="ALL">All Severities</option>
                      <option value="ERROR">Error</option>
                      <option value="WARN">Warning</option>
                      <option value="INFO">Info</option>
                    </select>
                    <span style={{ position: "absolute", right: 12, top: "50%", transform: "translateY(-50%)", color: "#64748b", fontSize: 10, pointerEvents: "none" }}>▼</span>
                  </div>
               </div>
            </div>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead style={{ background: "rgba(13, 13, 20, 0.3)", borderBottom: "1px solid #1e1e2e" }}>
                <tr>
                   <th style={{ padding: "14px 24px", color: "#64748b", textAlign: "left", fontSize: 11, fontWeight: 500, letterSpacing: "1px", textTransform: "uppercase" }}>Reported Time</th>
                   <th style={{ padding: "14px 20px", color: "#64748b", textAlign: "left", fontSize: 11, fontWeight: 500, letterSpacing: "1px", textTransform: "uppercase" }}>Severity</th>
                   <th style={{ padding: "14px 20px", color: "#64748b", textAlign: "left", fontSize: 11, fontWeight: 500, letterSpacing: "1px", textTransform: "uppercase" }}>Alert Name</th>
                   <th style={{ padding: "14px 20px", color: "#64748b", textAlign: "left", fontSize: 11, fontWeight: 500, letterSpacing: "1px", textTransform: "uppercase" }}>Risk Score</th>
                   <th style={{ padding: "14px 20px", color: "#64748b", textAlign: "center", fontSize: 11, fontWeight: 500, letterSpacing: "1px", textTransform: "uppercase" }}>Source Node</th>
                   <th style={{ padding: "14px 24px 14px 10px", color: "#64748b", textAlign: "right", fontSize: 11, fontWeight: 500, letterSpacing: "1px", textTransform: "uppercase" }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((a, i) => (
                  <tr key={a.id} className="table-row">
                    <td style={{ padding: "14px 24px", fontFamily: "'Roboto Mono', monospace", fontSize: 12, color: "#64748b" }}>{a.time}</td>
                    <td style={{ padding: "14px 20px" }}><LevelBadge level={a.level} /></td>
                    <td style={{ padding: "14px 20px", fontSize: 12, color: "#e2e8f0", maxWidth: 400 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
                        <span style={{ display: "block", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{a.message}</span>
                        {a.count > 1 && (
                          <span style={{
                            flexShrink: 0,
                            background: "rgba(124, 58, 237, 0.12)",
                            color: "#a78bfa",
                            border: "1px solid rgba(124, 58, 237, 0.25)",
                            borderRadius: 999,
                            padding: "2px 8px",
                            fontSize: 10,
                            fontFamily: "monospace",
                          }}>
                            x{a.count}
                          </span>
                        )}
                      </div>
                    </td>
                    <td style={{ padding: "14px 20px" }}><ScoreBadge score={a.score} /></td>
                    <td style={{ padding: "14px 20px", textAlign: "center" }}>
                      <span style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 11, color: "#64748b", background: "rgba(255,255,255,0.02)", padding: "4px 8px", borderRadius: 4, border: "1px solid rgba(255,255,255,0.03)" }}>
                        C{a.cluster} {a.isRootCause ? "🔥" : ""}
                      </span>
                    </td>
                    <td style={{ padding: "14px 24px 14px 10px", textAlign: "right", color: "#64748b", fontSize: 16, cursor: "pointer", userSelect: "none" }}>⋮</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
        </div>
      </div>
      {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}
      <ReportExport
        anomalies={anomalies}
        rawStats={rawStats}
        tamperDetected={tamperDetected}
        antiforensics={antiforensics}
        clusterCounts={clusterCounts}
      />
    </>
  );
}


