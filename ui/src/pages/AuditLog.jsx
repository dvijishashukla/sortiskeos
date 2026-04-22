import { useEffect, useState, useCallback } from "react";
import { fetchAuditLog, verifyAuditLog } from "../api";
import PageHeader from "../components/PageHeader.jsx";

const ACTION_COLORS = {
  pipeline_start: { bg: "rgba(124,58,237,0.15)", color: "#a78bfa", border: "#7c3aed" },
  pipeline_complete: { bg: "rgba(34,197,94,0.12)", color: "#4ade80", border: "#22c55e" },
  pipeline_triggered: { bg: "rgba(124,58,237,0.15)", color: "#a78bfa", border: "#7c3aed" },
  tamper_detected: { bg: "rgba(239,68,68,0.15)", color: "#f87171", border: "#ef4444" },
  antiforensics_detected: { bg: "rgba(239,68,68,0.15)", color: "#f87171", border: "#ef4444" },
  settings_changed: { bg: "rgba(245,158,11,0.15)", color: "#fbbf24", border: "#f59e0b" },
};

function ActionPill({ action }) {
  const style = ACTION_COLORS[action] || {
    bg: "rgba(100,116,139,0.15)",
    color: "#94a3b8",
    border: "#64748b",
  };

  return (
    <span
      style={{
        background: style.bg,
        color: style.color,
        border: `1px solid ${style.border}40`,
        borderRadius: 5,
        padding: "3px 10px",
        fontSize: 11,
        fontFamily: "'Roboto Mono', monospace",
        letterSpacing: 0.5,
        whiteSpace: "nowrap",
      }}
    >
      {action}
    </span>
  );
}

function IntegrityBadge({ result, loading }) {
  if (loading) {
    return (
      <span style={{ color: "#64748b", fontSize: 12, fontFamily: "'Roboto Mono', monospace" }}>
        verifying...
      </span>
    );
  }

  if (!result) {
    return (
      <span style={{ color: "#64748b", fontSize: 12, fontFamily: "'Roboto Mono', monospace" }}>
        No data
      </span>
    );
  }

  if (result.valid) {
    return (
      <span
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 6,
          background: "rgba(34,197,94,0.12)",
          border: "1px solid rgba(34,197,94,0.3)",
          color: "#4ade80",
          borderRadius: 6,
          padding: "5px 14px",
          fontSize: 12,
          fontFamily: "'Roboto Mono', monospace",
          fontWeight: 600,
          boxShadow: "0 0 12px rgba(34,197,94,0.15)",
        }}
      >
        <span style={{ fontSize: 10 }}>●</span> Chain Intact
      </span>
    );
  }

  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        background: "rgba(239,68,68,0.12)",
        border: "1px solid rgba(239,68,68,0.3)",
        color: "#f87171",
        borderRadius: 6,
        padding: "5px 14px",
        fontSize: 12,
        fontFamily: "'Roboto Mono', monospace",
        fontWeight: 600,
        boxShadow: "0 0 12px rgba(239,68,68,0.15)",
      }}
    >
      <span style={{ fontSize: 10 }}>▲</span> Chain Broken at Entry {result.broken_at ?? "?"}
    </span>
  );
}

function parseAuditTimestamp(value) {
  if (!value) return null;
  const raw = String(value).trim();
  if (!raw) return null;
  const normalized = /(?:Z|[+-]\d{2}:\d{2})$/i.test(raw) ? raw : `${raw}Z`;
  const parsed = new Date(normalized);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function formatAuditTimestamp(value) {
  const parsed = parseAuditTimestamp(value);
  if (!parsed) return "—";
  return parsed.toLocaleString([], {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

export default function AuditLog() {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [verifyResult, setVerify] = useState(null);
  const [verifying, setVerifying] = useState(false);
  const [lastRefresh, setLastRefresh] = useState(null);

  const load = useCallback(async () => {
    const data = await fetchAuditLog();
    setEntries(data || []);
    setLastRefresh(new Date());
    setLoading(false);
  }, []);

  const runVerify = useCallback(async () => {
    setVerifying(true);
    const result = await verifyAuditLog();
    setVerify(result);
    setVerifying(false);
  }, []);

  useEffect(() => {
    load();
    const interval = setInterval(load, 60000);
    return () => clearInterval(interval);
  }, [load]);

  return (
    <>
      <PageHeader
        title="Audit Log"
        subtitle="Cryptographic SHA-256 event chain"
        usingFallback={false}
        actions={
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <IntegrityBadge result={verifyResult} loading={verifying} />
            <button
              id="verify-integrity-btn"
              onClick={runVerify}
              disabled={verifying}
              style={{
                background: "rgba(124,58,237,0.15)",
                border: "1px solid rgba(124,58,237,0.4)",
                color: "#a78bfa",
                padding: "7px 16px",
                borderRadius: 7,
                fontSize: 12,
                fontFamily: "'Roboto Mono', monospace",
                fontWeight: 600,
                cursor: verifying ? "not-allowed" : "pointer",
                letterSpacing: 0.5,
                transition: "all 0.2s ease",
              }}
              onMouseOver={(e) => {
                if (!verifying) e.target.style.background = "rgba(124,58,237,0.28)";
              }}
              onMouseOut={(e) => {
                e.target.style.background = "rgba(124,58,237,0.15)";
              }}
            >
              {verifying ? "Verifying..." : "Verify Integrity"}
            </button>
            <button
              id="refresh-audit-btn"
              onClick={() => {
                setLoading(true);
                load();
              }}
              style={{
                background: "rgba(19,19,31,0.6)",
                border: "1px solid #1e1e2e",
                color: "#64748b",
                padding: "7px 14px",
                borderRadius: 7,
                fontSize: 12,
                fontFamily: "'Roboto Mono', monospace",
                cursor: "pointer",
                transition: "all 0.2s ease",
              }}
              onMouseOver={(e) => {
                e.target.style.color = "#e2e8f0";
              }}
              onMouseOut={(e) => {
                e.target.style.color = "#64748b";
              }}
            >
              ↻ Refresh
            </button>
          </div>
        }
      />

      <div
        style={{
          padding: "0 32px 32px",
          maxWidth: 1200,
          margin: "0 auto",
          fontFamily: "'Roboto', sans-serif",
        }}
      >
        <div style={{ height: 32 }} />
        {lastRefresh && (
          <div
            style={{
              marginBottom: 20,
              fontSize: 11,
              color: "#64748b",
              fontFamily: "'Roboto Mono', monospace",
            }}
          >
            Last refreshed: {lastRefresh.toLocaleTimeString()} · Auto-refresh every 60s · {entries.length} entries
          </div>
        )}

        <div
          style={{
            background: "rgba(19,19,31,0.4)",
            backdropFilter: "blur(12px)",
            WebkitBackdropFilter: "blur(12px)",
            border: "1px solid #1e1e2e",
            borderRadius: 12,
            overflow: "hidden",
            boxShadow: "0 0 20px rgba(124,58,237,0.08)",
          }}
        >
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "180px 180px 1fr 90px",
              gap: 0,
              padding: "12px 20px",
              borderBottom: "1px solid #1e1e2e",
              background: "rgba(13,13,20,0.5)",
            }}
          >
            {["Timestamp", "Action", "Detail", "Hash"].map((col) => (
              <div
                key={col}
                style={{
                  fontSize: 10,
                  color: "#64748b",
                  fontWeight: 700,
                  textTransform: "uppercase",
                  letterSpacing: 1.2,
                  fontFamily: "'Roboto Mono', monospace",
                }}
              >
                {col}
              </div>
            ))}
          </div>

          {loading ? (
            <div style={{ padding: "40px 20px", textAlign: "center", color: "#64748b", fontSize: 13 }}>
              Loading audit entries...
            </div>
          ) : entries.length === 0 ? (
            <div style={{ padding: "40px 20px", textAlign: "center" }}>
              <div style={{ fontSize: 32, marginBottom: 12 }}>⊛</div>
              <div style={{ color: "#64748b", fontSize: 14 }}>No audit entries yet.</div>
              <div style={{ color: "#64748b", fontSize: 12, marginTop: 4 }}>
                Trigger the pipeline or change a setting to generate the first entry.
              </div>
            </div>
          ) : (
            entries.map((entry, i) => {
              const isCritical =
                entry.action === "tamper_detected" || entry.action === "antiforensics_detected";

              return (
                <div
                  key={i}
                  className="table-row"
                  style={{
                    display: "grid",
                    gridTemplateColumns: "180px 180px 1fr 90px",
                    gap: 0,
                    padding: "13px 20px",
                    borderBottom: i < entries.length - 1 ? "1px solid rgba(30,30,46,0.6)" : "none",
                    background: isCritical ? "rgba(239,68,68,0.03)" : "transparent",
                    alignItems: "center",
                  }}
                >
                  <div
                    style={{
                      fontSize: 12,
                      color: "#94a3b8",
                      fontFamily: "'Roboto Mono', monospace",
                      letterSpacing: 0.3,
                    }}
                  >
                    {formatAuditTimestamp(entry.timestamp)}
                  </div>

                  <div>
                    <ActionPill action={entry.action} />
                  </div>

                  <div
                    style={{
                      fontSize: 12,
                      color: "#64748b",
                      fontFamily: "'Roboto Mono', monospace",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {entry.detail && Object.keys(entry.detail).length > 0 ? (
                      Object.entries(entry.detail)
                        .map(([key, itemValue]) => `${key}: ${itemValue}`)
                        .join(" · ")
                    ) : (
                      <span style={{ color: "#2d2d44" }}>—</span>
                    )}
                  </div>

                  <div
                    style={{
                      fontSize: 11,
                      color: "#4c1d95",
                      fontFamily: "'Roboto Mono', monospace",
                      letterSpacing: 0.5,
                      userSelect: "all",
                    }}
                    title={entry.hash}
                  >
                    {entry.hash ? entry.hash.slice(0, 8) : "—"}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </>
  );
}
