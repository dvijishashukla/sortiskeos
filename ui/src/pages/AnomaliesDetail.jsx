import { useEffect, useState } from "react";
import {
  ScatterChart, Scatter, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, ReferenceLine,
} from "recharts";
import { fetchAnomalies } from "../api.js";
import PageHeader from "../components/PageHeader.jsx";

const API_BASE = (process.env.REACT_APP_API_URL || "http://localhost:8000").replace(/\/$/, "");

const ALL_ANOMALIES = [
  { id: 1, time: "22:54:37", level: "ERROR", source: "Backend log", message: "System did not shut down cleanly - Event 41", score: -0.150, cluster: 2, method: "Fallback" },
  { id: 2, time: "22:54:35", level: "INFO", source: "Backend log", message: "RSC offload failed on Hyper-V network adapter", score: -0.122, cluster: 2, method: "Fallback" },
  { id: 3, time: "22:54:33", level: "WARN", source: "Backend log", message: "vmbushid service terminated unexpectedly", score: -0.111, cluster: 2, method: "Fallback" },
  { id: 4, time: "22:54:30", level: "WARN", source: "Backend log", message: "Driver failed to load - vmbushid.sys", score: -0.103, cluster: 2, method: "Fallback" },
  { id: 5, time: "22:54:28", level: "ERROR", source: "Backend log", message: "BugCheck 0x000000EF: Critical process died", score: -0.098, cluster: 2, method: "Fallback" },
];

const CLUSTER_COLORS = { 1: "#3b82f6", 2: "#e6734b", 3: "#a855f7" };

import { formatTimeShort as formatTime } from "../utils/timeFormat.js";

function formatAnomalies(items) {
  if (!Array.isArray(items) || items.length === 0) return ALL_ANOMALIES;

  return items.map((item, index) => ({
    id: index + 1,
    time: formatTime(item?.time),
    level: item?.level || "INFO",
    source: item?.isRootCause ? "Root cause" : "Backend log",
    message: item?.message || "No message available",
    score: Number(item?.score) || 0,
    cluster: Number(item?.cluster ?? 0),
    method: item?.isRootCause ? "FastAPI" : "FastAPI",
  }));
}

const LEVEL_STYLE = {
  ERROR: { bg: "rgba(239, 68, 68, 0.12)", text: "#ef4444", dot: "#ef4444" },
  WARN: { bg: "rgba(245, 158, 11, 0.12)", text: "#f59e0b", dot: "#f59e0b" },
  INFO: { bg: "rgba(59, 130, 246, 0.10)", text: "#3b82f6", dot: "#3b82f6" },
};

function LevelBadge({ level }) {
  const s = LEVEL_STYLE[level] || LEVEL_STYLE.INFO;
  return (
    <span style={{
      background: s.bg, color: s.text, border: `1px solid ${s.dot}33`,
      borderRadius: 4, padding: "2px 7px", fontSize: 10,
      fontFamily: "monospace", letterSpacing: 1,
      display: "inline-flex", alignItems: "center", gap: 4,
    }}>
      <span style={{ width: 4, height: 4, borderRadius: "50%", background: s.dot, display: "inline-block" }} />
      {level}
    </span>
  );
}

function MethodTag({ method }) {
  return (
    <span style={{
      background: "rgba(59,130,246,0.1)", color: "#60a5fa",
      border: "1px solid rgba(59,130,246,0.2)",
      borderRadius: 4, padding: "1px 7px", fontSize: 10,
      fontFamily: "monospace",
    }}>{method}</span>
  );
}

function CustomDot(props) {
  const { cx, cy, payload } = props;
  const color = CLUSTER_COLORS[payload.cluster] || "#6e7681";
  return <circle cx={cx} cy={cy} r={5} fill={color} fillOpacity={0.8} stroke={color} strokeWidth={1} />;
}

function CustomScatterTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={{
      background: "#0d0d14", border: "1px solid #1e1e2e",
      borderRadius: 6, padding: "8px 14px", fontSize: 11,
    }}>
      <div style={{ color: "#64748b" }}>Log #{d.x} - Cluster {d.cluster || "N/A"}</div>
      <div style={{ color: CLUSTER_COLORS[d.cluster] || "#e2e8f0", fontFamily: "monospace", marginTop: 2 }}>score: -{d.y.toFixed(3)}</div>
      <div style={{ color: "#e2e8f0", marginTop: 2 }}>{d.label}</div>
    </div>
  );
}

export default function AnomaliesDetail({ onBack }) {
  const [filter, setFilter] = useState("ALL");
  const [clusterFilter, setClusterFilter] = useState("ALL");
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState("score");
  const [anomalies, setAnomalies] = useState(ALL_ANOMALIES);
  const [usingFallback, setUsingFallback] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function loadAnomalies() {
      setIsLoading(true);
      try {
        const params = { size: 100 };
        if (filter !== "ALL") params.level = filter;
        if (clusterFilter !== "ALL") params.cluster = clusterFilter;
        if (search) params.search = search;

        const response = await fetchAnomalies(params);
        if (cancelled) return;
        setAnomalies(formatAnomalies(response));
        setUsingFallback(!Array.isArray(response));
      } catch (error) {
        if (cancelled) return;
        setAnomalies(ALL_ANOMALIES);
        setUsingFallback(true);
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }

    loadAnomalies();

    return () => {
      cancelled = true;
    };
  }, [filter, clusterFilter, search]);

  const filtered = anomalies
    .filter((a) => filter === "ALL" || a.level === filter)
    .filter((a) => clusterFilter === "ALL" || String(a.cluster) === clusterFilter)
    .filter((a) => a.message.toLowerCase().includes(search.toLowerCase()) || a.source.toLowerCase().includes(search.toLowerCase()))
    .sort((a, b) => sort === "score" ? a.score - b.score : a.time.localeCompare(b.time));

  const byCluster = [1, 2, 3].map((c) => ({ id: c, count: anomalies.filter((a) => a.cluster === c).length }));
  const scatterData = anomalies.map((a, i) => ({ x: i + 1, y: Math.abs(a.score), cluster: a.cluster, label: a.source }));

  return (
    <>
      <style>{`
        @keyframes fadeUp { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
        tr:hover td { background: rgba(48,54,61,0.3) !important; }
      `}</style>

      <PageHeader 
        title="Anomaly Deep Dive" 
        subtitle={`${anomalies.length} anomalies detected`} 
        usingFallback={usingFallback}
      />

      <div style={{ padding: "0 32px 32px", maxWidth: 1200, margin: "0 auto" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: 12, marginBottom: 20, marginTop: 24, animation: "fadeUp 0.4s ease both" }}>
          <div style={{ display: "flex", gap: 12 }}>
            {byCluster.map((c) => (
              <div key={c.id} style={{
                background: "rgba(15, 23, 42, 0.6)", backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)",
                border: `1px solid ${(CLUSTER_COLORS[c.id] || "#64748b")}33`,
                borderRadius: 8, padding: "8px 16px", textAlign: "center", boxShadow: "0 0 12px #7c3aed18",
              }}>
                <div style={{ fontSize: 18, fontWeight: 600, color: CLUSTER_COLORS[c.id], fontFamily: "'Roboto', sans-serif" }}>{c.count}</div>
                <div style={{ fontSize: 9, color: "#64748b", letterSpacing: "1px", fontWeight: 600, textTransform: "uppercase" }}>Cluster {c.id}</div>
              </div>
            ))}
          </div>
        </div>

        <div style={{
          background: "rgba(19, 19, 31, 0.4)", backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)",
          border: "1px solid #1e1e2e",
          borderRadius: 12, padding: "20px 16px 12px", marginBottom: 20,
          animation: "fadeUp 0.4s ease 0.06s both",
          boxShadow: "0 8px 24px rgba(0,0,0,0.15)",
        }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14, padding: "0 4px" }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: "#e2e8f0", letterSpacing: 0.3 }}>Anomaly Score Distribution</span>
            <div style={{ display: "flex", gap: 16 }}>
              {Object.entries(CLUSTER_COLORS).map(([c, col]) => (
                <span key={c} style={{ fontSize: 11, color: col, display: "flex", alignItems: "center", gap: 5 }}>
                  <span style={{ width: 8, height: 8, borderRadius: "50%", background: col, display: "inline-block" }} />
                  Cluster {c}
                </span>
              ))}
            </div>
          </div>
          <ResponsiveContainer width="100%" height={160}>
            <ScatterChart margin={{ top: 4, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" />
              <XAxis dataKey="x" type="number" tick={{ fontSize: 10, fill: "#64748b", fontFamily: "monospace" }} tickLine={false} axisLine={false} label={{ value: "log index", position: "insideBottom", offset: -2, fontSize: 10, fill: "#64748b" }} />
              <YAxis dataKey="y" type="number" tick={{ fontSize: 10, fill: "#64748b", fontFamily: "monospace" }} tickLine={false} axisLine={false} />
              <ReferenceLine y={0.05} stroke="#7c3aed" strokeDasharray="4 4" strokeOpacity={0.5} label={{ value: "threshold", position: "right", fontSize: 10, fill: "#7c3aed" }} />
              <Tooltip content={<CustomScatterTooltip />} />
              <Scatter data={scatterData} shape={<CustomDot />} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        <div style={{
          display: "flex", gap: 10, marginBottom: 16, flexWrap: "wrap",
          animation: "fadeUp 0.4s ease 0.1s both",
        }}>
          {["ALL","ERROR","WARN","INFO"].map((l) => (
            <button key={l} onClick={() => setFilter(l)} style={{
              background: filter === l ? "#1e1e2e" : "transparent",
              border: `1px solid ${filter === l ? "#7c3aed44" : "#1e1e2e"}`,
              color: filter === l ? "#e2e8f0" : "#64748b",
              borderRadius: 6, padding: "5px 14px", cursor: "pointer",
              fontSize: 11, fontFamily: "monospace", letterSpacing: 1,
            }}>{l}</button>
          ))}
          <div style={{ width: 1, background: "#1e1e2e", margin: "0 4px" }} />
          {["ALL","1","2","3"].map((c) => (
            <button key={c} className="action-btn" onClick={() => setClusterFilter(c)} style={{
              background: clusterFilter === c ? "rgba(19, 19, 31, 0.6)" : "transparent",
              border: `1px solid ${clusterFilter === c ? (CLUSTER_COLORS[c] || "#7c3aed") + "44" : "#1e1e2e"}`,
              color: clusterFilter === c ? (CLUSTER_COLORS[c] || "#e2e8f0") : "#64748b",
              borderRadius: 6, padding: "5px 14px", cursor: "pointer",
              fontSize: 11, fontFamily: "monospace", letterSpacing: 1,
            }}>{c === "ALL" ? "ALL CLUSTERS" : `C${c}`}</button>
          ))}
          <input
            placeholder="Search source or message..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{
              marginLeft: "auto", background: "rgba(13, 13, 20, 0.5)", border: "1px solid #1e1e2e",
              color: "#e2e8f0", borderRadius: 6, padding: "5px 14px",
              fontSize: 12, width: 220, outline: "none",
            }}
          />
          <select value={sort} onChange={(e) => setSort(e.target.value)} style={{
            background: "rgba(13, 13, 20, 0.5)", border: "1px solid #1e1e2e",
            color: "#e2e8f0", borderRadius: 6, padding: "5px 10px",
            fontSize: 12, cursor: "pointer", outline: "none",
          }}>
            <option value="score">Sort: Score</option>
            <option value="time">Sort: Time</option>
          </select>
        </div>

        <div style={{
          background: "rgba(19, 19, 31, 0.3)", backdropFilter: "blur(12px)", WebkitBackdropFilter: "blur(12px)",
          border: "1px solid #1e1e2e",
          borderRadius: 12, overflow: "hidden",
          animation: "fadeUp 0.4s ease 0.14s both",
          boxShadow: "0 0 12px #7c3aed18",
          opacity: isLoading ? 0.4 : 1,
          transition: "opacity 0.3s",
          pointerEvents: isLoading ? "none" : "auto",
        }}>
          <table style={{ borderCollapse: "collapse", width: "100%" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid #1e1e2e", background: "rgba(13, 13, 20, 0.3)" }}>
                {["#","TIME","LEVEL","SOURCE","MESSAGE","SCORE","CLUSTER","METHOD"].map((h) => (
                  <th key={h} style={{ padding: "14px 16px", fontSize: 10, color: "#64748b", letterSpacing: "1.2px", textTransform: "uppercase", textAlign: "left", fontWeight: 500 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((a, i) => (
                <tr key={a.id} className="table-row" style={{
                  borderBottom: "1px solid #1e1e2e",
                  animation: `fadeUp 0.35s ease ${i * 0.03}s both`,
                }}>
                  <td style={{ padding: "12px 16px", fontFamily: "monospace", fontSize: 11, color: "#64748b" }}>{a.id}</td>
                  <td style={{ padding: "12px 16px", fontFamily: "monospace", fontSize: 11, color: "#64748b", whiteSpace: "nowrap" }}>{a.time}</td>
                  <td style={{ padding: "12px 16px" }}><LevelBadge level={a.level} /></td>
                  <td style={{ padding: "12px 16px", fontSize: 11, color: "#64748b", fontFamily: "monospace", maxWidth: 130, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{a.source}</td>
                  <td style={{ padding: "12px 16px", fontSize: 12, color: "#e2e8f0", maxWidth: 280, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{a.message}</td>
                  <td style={{ padding: "12px 16px" }}>
                    <span style={{
                      fontFamily: "monospace", fontSize: 11,
                      color: `rgb(${Math.round(255 * Math.min(1, Math.abs(a.score)/0.2))},${Math.round(255 * (1 - Math.min(1, Math.abs(a.score)/0.2)))},80)`,
                    }}>{a.score.toFixed(3)}</span>
                  </td>
                  <td style={{ padding: "12px 16px" }}>
                    <span style={{
                      width: 8, height: 8, borderRadius: "50%",
                      background: CLUSTER_COLORS[a.cluster] || "#64748b",
                      display: "inline-block", marginRight: 8,
                      boxShadow: `0 0 8px ${(CLUSTER_COLORS[a.cluster] || "#64748b")}66`,
                    }} />
                    <span style={{ fontFamily: "monospace", fontSize: 11, color: CLUSTER_COLORS[a.cluster] || "#64748b" }}>C{a.cluster || "-"}</span>
                  </td>
                  <td style={{ padding: "12px 16px" }}><MethodTag method={a.method} /></td>
                </tr>
              ))}
              {filtered.length === 0 && (
                <tr><td colSpan={8} style={{ padding: 28, textAlign: "center", color: "#64748b", fontSize: 13 }}>No anomalies match filters</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
