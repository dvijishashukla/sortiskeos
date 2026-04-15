/**
 * ReportExport.jsx
 * ─────────────────────────────────────────────────────────────────────────────
 * Hidden during normal use (display:none).
 * Becomes the ONLY visible element during window.print() via @media print CSS.
 * All data is passed as props — zero API calls.
 */
import { summarizeRootCause } from "../utils/displayValue.js";

export default function ReportExport({ anomalies, rawStats, tamperDetected, antiforensics, clusterCounts }) {
  const generated = new Date().toLocaleString([], {
    year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
  });

  // Pull structured suggestion data from rawStats if available
  const suggestion   = rawStats?.suggestion || {};
  const rootCause    = rawStats?.rootCause  || "Unknown";
  const label        = summarizeRootCause(suggestion?.label || rootCause || "Unknown");
  const likelyCause  = suggestion?.likely_cause || "No additional context available.";
  const confidence   = suggestion?.confidence   || "Low";
  const investigate  = Array.isArray(suggestion?.investigate)     ? suggestion.investigate     : [];
  const commands     = Array.isArray(suggestion?.commands_to_run) ? suggestion.commands_to_run : [];
  const anomalyCount = rawStats?.anomalyCount ?? anomalies.length;
  const lastCrash    = rawStats?.lastCrash
    ? `${rawStats.lastCrash.date || ""} ${rawStats.lastCrash.time || ""}`.trim()
    : "N/A";

  const top20 = anomalies.slice(0, 20);

  // Cluster summary rows
  const clusterRows = Object.entries(clusterCounts || {}).map(([id, count]) => ({
    id,
    count,
    isRootCause: top20.some((a) => String(a.cluster) === String(id) && a.isRootCause),
  }));

  const rootClusterId = top20.find((a) => a.isRootCause)?.cluster;

  const confClass =
    confidence === "High"   ? "confidence-high"   :
    confidence === "Medium" ? "confidence-medium"  :
                              "confidence-low";

  // Severity from anomalies
  const topLevel = top20.find((a) => a.isRootCause)?.level || "UNKNOWN";

  return (
    <div id="report-export" style={{ display: "none" }}>

      {/* ── 1. HEADER ─────────────────────────────────────────────────────── */}
      <div className="report-header">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <div style={{ fontSize: 22, fontWeight: 700, letterSpacing: 1 }}>SORTISKEOS</div>
            <div style={{ fontSize: 14, fontWeight: 400, color: "#444", marginTop: 2 }}>
              Incident Analysis Report
            </div>
          </div>
          <div style={{ textAlign: "right", fontSize: 12, color: "#555" }}>
            <div><strong>Generated:</strong> {generated}</div>
            <div><strong>Mode:</strong> AI-Driven Log Analysis</div>
          </div>
        </div>
        <div style={{ borderTop: "2px solid #000", marginTop: 12, paddingTop: 0 }} />
      </div>

      {/* ── 2. EXECUTIVE SUMMARY ──────────────────────────────────────────── */}
      <div className="summary-box">
        <div style={{ fontSize: 13, fontWeight: 700, textTransform: "uppercase", letterSpacing: 1, marginBottom: 10, borderBottom: "1px solid #ddd", paddingBottom: 6 }}>
          Executive Summary
        </div>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <tbody>
            <tr>
              <td style={{ padding: "3px 0", color: "#555", width: "35%" }}>Root Cause Category</td>
              <td style={{ padding: "3px 0", fontWeight: 600 }}>{label}</td>
              <td style={{ padding: "3px 0", color: "#555", width: "22%" }}>Confidence</td>
              <td style={{ padding: "3px 0" }}><span className={confClass}>{confidence}</span></td>
            </tr>
            <tr>
              <td style={{ padding: "3px 0", color: "#555" }}>Total Anomalies</td>
              <td style={{ padding: "3px 0", fontWeight: 600 }}>{anomalyCount}</td>
              <td style={{ padding: "3px 0", color: "#555" }}>Severity</td>
              <td style={{ padding: "3px 0", fontWeight: 600 }}>{topLevel}</td>
            </tr>
            <tr>
              <td style={{ padding: "3px 0", color: "#555" }}>Last Crash</td>
              <td style={{ padding: "3px 0" }}>{lastCrash}</td>
              <td style={{ padding: "3px 0", color: "#555" }}>Root Cluster</td>
              <td style={{ padding: "3px 0" }}>{rootClusterId !== undefined ? `Cluster ${rootClusterId}` : "N/A"}</td>
            </tr>
            <tr>
              <td style={{ padding: "6px 0 3px", color: "#555", verticalAlign: "top" }}>Likely Cause</td>
              <td colSpan={3} style={{ padding: "6px 0 3px" }}>{likelyCause}</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* ── 3. SUGGESTED ACTIONS ──────────────────────────────────────────── */}
      {(investigate.length > 0 || commands.length > 0) && (
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 13, fontWeight: 700, textTransform: "uppercase", letterSpacing: 1, marginBottom: 8, borderBottom: "1px solid #ddd", paddingBottom: 4 }}>
            Suggested Actions
          </div>

          {investigate.length > 0 && (
            <div style={{ marginBottom: 10 }}>
              <div style={{ fontWeight: 700, fontSize: 12, marginBottom: 4 }}>Investigate:</div>
              <ul style={{ margin: "0 0 0 20px", padding: 0, fontSize: 12, lineHeight: 1.8 }}>
                {investigate.map((item, i) => (
                  <li key={i}>{item}</li>
                ))}
              </ul>
            </div>
          )}

          {commands.length > 0 && (
            <div>
              <div style={{ fontWeight: 700, fontSize: 12, marginBottom: 4 }}>Commands to run:</div>
              {commands.map((cmd, i) => (
                <div key={i} className="report-code-block">{cmd}</div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── 4. ANOMALY BREAKDOWN TABLE ────────────────────────────────────── */}
      <div style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 13, fontWeight: 700, textTransform: "uppercase", letterSpacing: 1, marginBottom: 8, borderBottom: "1px solid #ddd", paddingBottom: 4 }}>
          Anomaly Breakdown (Top {top20.length})
        </div>
        <table className="report-table">
          <thead>
            <tr>
              <th style={{ width: "12%" }}>Timestamp</th>
              <th>Message</th>
              <th style={{ width: "8%" }}>Level</th>
              <th style={{ width: "10%" }}>Cluster</th>
              <th style={{ width: "8%" }}>Score</th>
            </tr>
          </thead>
          <tbody>
            {top20.map((row, i) => (
              <tr key={i} className={row.level === "ERROR" ? "error-row" : ""}>
                <td style={{ fontFamily: "monospace", fontSize: 11 }}>{row.time}</td>
                <td style={{ fontSize: 11, maxWidth: 300, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                    title={row.message}>{row.message}</td>
                <td style={{ fontWeight: 600, fontSize: 11, color: row.level === "ERROR" ? "#cc0000" : row.level === "WARN" ? "#cc6600" : "#333" }}>
                  {row.level}
                </td>
                <td style={{ fontFamily: "monospace", fontSize: 11, textAlign: "center" }}>
                  {row.cluster !== undefined && row.cluster !== "" ? `C${row.cluster}` : "—"}
                </td>
                <td style={{ fontFamily: "monospace", fontSize: 11, textAlign: "right" }}>
                  {typeof row.score === "number" ? row.score.toFixed(3) : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* ── 5. CLUSTER SUMMARY ────────────────────────────────────────────── */}
      {clusterRows.length > 0 && (
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 13, fontWeight: 700, textTransform: "uppercase", letterSpacing: 1, marginBottom: 8, borderBottom: "1px solid #ddd", paddingBottom: 4 }}>
            Cluster Summary
          </div>
          <table className="report-table">
            <thead>
              <tr>
                <th style={{ width: "20%" }}>Cluster ID</th>
                <th style={{ width: "20%" }}>Anomaly Count</th>
                <th>Root Cause Cluster</th>
              </tr>
            </thead>
            <tbody>
              {clusterRows.map((r, i) => (
                <tr key={i}>
                  <td style={{ fontFamily: "monospace" }}>Cluster {r.id}</td>
                  <td>{r.count}</td>
                  <td style={{ fontWeight: r.isRootCause ? 700 : 400, color: r.isRootCause ? "#cc0000" : "#333" }}>
                    {r.isRootCause ? "✓ Yes — Root Cause" : "No"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── 6. SECURITY FLAGS ─────────────────────────────────────────────── */}
      {(tamperDetected || antiforensics?.detected) && (
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 13, fontWeight: 700, textTransform: "uppercase", letterSpacing: 1, marginBottom: 8, borderBottom: "2px solid #cc0000", paddingBottom: 4, color: "#cc0000" }}>
            ⚠ Security Flags
          </div>
          <table className="report-table">
            <tbody>
              <tr>
                <td style={{ fontWeight: 600, width: "40%" }}>Log Tamper Detected</td>
                <td style={{ fontWeight: tamperDetected ? 700 : 400, color: tamperDetected ? "#cc0000" : "#333" }}>
                  {tamperDetected ? "YES — system_logs.json was modified before analysis" : "No"}
                </td>
              </tr>
              <tr>
                <td style={{ fontWeight: 600 }}>Anti-Forensics Events</td>
                <td style={{ fontWeight: antiforensics?.detected ? 700 : 400, color: antiforensics?.detected ? "#cc0000" : "#333" }}>
                  {antiforensics?.detected
                    ? `YES — ${antiforensics.count} event log clearing event(s) detected`
                    : "None"}
                </td>
              </tr>
            </tbody>
          </table>
          {antiforensics?.events?.length > 0 && (
            <div style={{ marginTop: 8 }}>
              <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 4 }}>Cleared Log Events:</div>
              {antiforensics.events.map((e, i) => (
                <div key={i} className="report-code-block">
                  [{e.timestamp}] Windows Event ID {e.event_id} — Channel: {e.channel}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── 7. FOOTER ─────────────────────────────────────────────────────── */}
      <div className="report-footer">
        <div style={{ borderTop: "1px solid #999", paddingTop: 8, display: "flex", justifyContent: "space-between", fontSize: 11, color: "#777" }}>
          <span>Generated by Sortiskeos — AI-driven log analysis</span>
          <span className="page-number">Page 1</span>
        </div>
      </div>

    </div>
  );
}
