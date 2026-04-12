import { useEffect, useState } from "react";
import { fetchLogs } from "../api";
import PageHeader from "../components/PageHeader.jsx";

function formatTime(value) {
  if (!value) return "Unknown";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value);
  return parsed.toLocaleString([], {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function normalizeLogs(items) {
  if (!Array.isArray(items)) return [];
  return items.map((item, index) => ({
    id: `${item?.time || "log"}-${index}`,
    time: item?.time || "",
    level: item?.level || "INFO",
    source: item?.source || "unknown",
    message: item?.message || "No message available",
  }));
}

function levelTone(level) {
  if (level === "ERROR" || level === "CRITICAL") {
    return {
      text: "#ef4444",
      bg: "rgba(239, 68, 68, 0.12)",
      border: "rgba(239, 68, 68, 0.24)",
    };
  }
  if (level === "WARN") {
    return {
      text: "#f59e0b",
      bg: "rgba(245, 158, 11, 0.12)",
      border: "rgba(245, 158, 11, 0.24)",
    };
  }
  return {
    text: "#3b82f6",
    bg: "rgba(59, 130, 246, 0.12)",
    border: "rgba(59, 130, 246, 0.22)",
  };
}

export default function RawLogs() {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("ALL");
  const [search, setSearch] = useState("");
  const [error, setError] = useState("");

  const loadData = async () => {
    setLoading(true);
    setError("");

    const data = await fetchLogs({
      size: 100,
      level: filter === "ALL" ? "" : filter,
      search,
    });

    if (Array.isArray(data)) {
      setLogs(normalizeLogs(data));
      if (data.length === 0) {
        setError(
          search || filter !== "ALL"
            ? "No logs match the current filters."
            : "No local logs are available yet. Run log collection or wait for new events."
        );
      }
    } else {
      setLogs([]);
      setError("The log stream is unavailable right now. Check that the FastAPI backend is running.");
    }

    setLoading(false);
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 10000);
    return () => clearInterval(interval);
  }, [filter, search]);

  return (
    <>
      <PageHeader 
        title="Raw Log Stream" 
        subtitle="Clean event table for collected logs" 
        usingFallback={logs.length === 0 && !loading && search === "" && filter === "ALL"}
        actions={
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            <input
              type="text"
              placeholder="Search logs"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              style={{
                padding: "8px 14px",
                borderRadius: 8,
                border: "1px solid #1e1e2e",
                background: "rgba(13, 13, 20, 0.55)",
                color: "#fff",
                outline: "none",
                width: 180,
                fontSize: 12,
              }}
            />
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              style={{
                padding: "8px 14px",
                borderRadius: 8,
                border: "1px solid #1e1e2e",
                background: "rgba(13, 13, 20, 0.55)",
                color: "#e2e8f0",
                outline: "none",
                cursor: "pointer",
                fontSize: 12,
              }}
            >
              <option value="ALL">All Levels</option>
              <option value="ERROR">Errors</option>
              <option value="WARN">Warnings</option>
              <option value="INFO">Info</option>
            </select>
            <button
              onClick={loadData}
              disabled={loading}
              style={{
                padding: "8px 14px",
                borderRadius: 8,
                border: "1px solid #1e1e2e",
                background: "rgba(255,255,255,0.03)",
                color: loading ? "#64748b" : "#e2e8f0",
                cursor: loading ? "default" : "pointer",
                fontSize: 12,
              }}
            >
              ↻ Refresh
            </button>
          </div>
        }
      />

      <div style={{ padding: "0 32px 32px", maxWidth: 1200, margin: "0 auto", fontFamily: "'Roboto', sans-serif" }}>
        <div style={{ height: 32 }} />

      <div style={{
        background: "rgba(19, 19, 31, 0.42)",
        backdropFilter: "blur(12px)",
        WebkitBackdropFilter: "blur(12px)",
        border: "1px solid #1e1e2e",
        borderRadius: 14,
        overflow: "hidden",
        boxShadow: "0 0 12px #7c3aed18",
        animation: "fadeSlideUp 0.5s ease 0.1s both",
      }}>
        <div style={{
          padding: "14px 18px",
          borderBottom: "1px solid #1e1e2e",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 12,
          flexWrap: "wrap",
        }}>
          <div style={{ fontSize: 12, color: "#64748b", letterSpacing: "0.4px" }}>
            {loading ? "Refreshing log table..." : `${logs.length} rows loaded`}
          </div>
          <div style={{ fontSize: 12, color: "#64748b" }}>
            Showing time, level, source, and message only
          </div>
        </div>

        {loading && logs.length === 0 ? (
          <div style={{ padding: 44, textAlign: "center", color: "#64748b" }}>Loading logs...</div>
        ) : logs.length === 0 ? (
          <div style={{ padding: 44, textAlign: "center" }}>
            <div style={{ color: "#e2e8f0", fontSize: 15, fontWeight: 600, marginBottom: 8 }}>No logs to display</div>
            <div style={{ color: "#64748b", fontSize: 13, lineHeight: 1.5 }}>{error}</div>
          </div>
        ) : (
          <div style={{ maxHeight: "calc(100vh - 220px)", overflowY: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left", tableLayout: "fixed" }}>
              <thead style={{ background: "rgba(13, 13, 20, 0.34)", position: "sticky", top: 0, zIndex: 1 }}>
                <tr>
                  <th style={{ width: "22%", padding: "14px 18px", color: "#64748b", fontWeight: 500, fontSize: 11, letterSpacing: "1px", textTransform: "uppercase" }}>Time</th>
                  <th style={{ width: "14%", padding: "14px 18px", color: "#64748b", fontWeight: 500, fontSize: 11, letterSpacing: "1px", textTransform: "uppercase" }}>Level</th>
                  <th style={{ width: "22%", padding: "14px 18px", color: "#64748b", fontWeight: 500, fontSize: 11, letterSpacing: "1px", textTransform: "uppercase" }}>Source</th>
                  <th style={{ width: "42%", padding: "14px 18px", color: "#64748b", fontWeight: 500, fontSize: 11, letterSpacing: "1px", textTransform: "uppercase" }}>Message</th>
                </tr>
              </thead>
              <tbody>
                {logs.map((log) => {
                  const tone = levelTone(log.level);
                  return (
                    <tr key={log.id} className="table-row" style={{ borderTop: "1px solid #1e1e2e" }}>
                      <td style={{ padding: "16px 18px", color: "#64748b", fontSize: 12, verticalAlign: "top", fontFamily: "'Roboto Mono', monospace" }}>
                        {formatTime(log.time)}
                      </td>
                      <td style={{ padding: "16px 18px", verticalAlign: "top" }}>
                        <span style={{
                          display: "inline-flex",
                          alignItems: "center",
                          padding: "4px 10px",
                          borderRadius: 999,
                          color: tone.text,
                          background: tone.bg,
                          border: `1px solid ${tone.border}`,
                          fontSize: 11,
                          fontWeight: 700,
                          letterSpacing: "0.8px",
                        }}>
                          {log.level}
                        </span>
                      </td>
                      <td style={{ padding: "16px 18px", color: "#e2e8f0", fontSize: 12, verticalAlign: "top", fontFamily: "'Roboto Mono', monospace", wordBreak: "break-word" }}>
                        {log.source}
                      </td>
                      <td style={{ padding: "16px 18px", color: "#e2e8f0", fontSize: 13, lineHeight: 1.55, verticalAlign: "top", wordBreak: "break-word" }}>
                        {log.message}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
    </>
  );
}
