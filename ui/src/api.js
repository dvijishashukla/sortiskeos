const BASE = "http://localhost:8000";

/**
 * Fetch dashboard stats
 * Returns: { totalCrashes, lastCrash: { date, time }, rootCause, anomalyCount }
 */
export async function fetchStats() {
  try {
    const response = await fetch(`${BASE}/dashboard/stats`);
    if (!response.ok) return null;
    return response.json();
  } catch (error) {
    return null;
  }
}

/**
 * Fetch 24-hour timeline data
 * Returns: [ { hour, score }, ... ]
 */
export async function fetchTimeline() {
  try {
    const response = await fetch(`${BASE}/dashboard/timeline`);
    if (!response.ok) return null;
    return response.json();
  } catch (error) {
    return null;
  }
}

/**
 * Fetch crash history
 * Returns: [ { date, time, rootCause, anomalies, score }, ... ]
 */
export async function fetchCrashes() {
  try {
    const response = await fetch(`${BASE}/dashboard/crashes`);
    if (!response.ok) return null;
    return response.json();
  } catch (error) {
    return null;
  }
}

/**
 * Fetch anomalies with optional filters
 * Params: { size, level, search, cluster }
 * Returns: [ { time, level, message, score, isRootCause, cluster }, ... ]
 */
export async function fetchAnomalies(params = {}) {
  try {
    const query = new URLSearchParams();
    if (params.size) query.append("size", params.size);
    if (params.level) query.append("level", params.level);
    if (params.search) query.append("search", params.search);
    if (params.cluster) query.append("cluster", params.cluster);

    const url = query.toString() ? `/anomalies?${query.toString()}` : "/anomalies";
    const response = await fetch(`${BASE}${url}`);
    if (!response.ok) return null;
    return response.json();
  } catch (error) {
    return null;
  }
}

/**
 * Fetch raw logs with optional filters
 * Params: { size, level, search }
 * Returns: [ { time, level, message, score, isRootCause, cluster }, ... ]
 */
export async function fetchLogs(params = {}) {
  try {
    const query = new URLSearchParams();
    if (params.size) query.append("size", params.size);
    if (params.level) query.append("level", params.level);
    if (params.search) query.append("search", params.search);

    const url = query.toString() ? `/logs?${query.toString()}` : "/logs";
    const response = await fetch(`${BASE}${url}`);
    if (!response.ok) return null;
    return response.json();
  } catch (error) {
    return null;
  }
}

/**
 * Fetch root cause analysis data (highest-scoring anomaly cluster)
 * Returns: { clusterId, label, confidence, anomalyCount, topScore, events, description, fix }
 */
export async function fetchRootCause() {
  try {
    const response = await fetch(`${BASE}/dashboard/rootcause`);
    if (!response.ok) return null;
    return response.json();
  } catch (error) {
    return null;
  }
}

/**
 * Trigger the ML pipeline (log collection + anomaly detection)
 * Returns: { status, message }
 */
export async function triggerPipeline() {
  try {
    const response = await fetch(`${BASE}/pipeline/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    if (!response.ok) return null;
    return response.json();
  } catch (error) {
    return null;
  }
}

/**
 * Health check for backend and Elasticsearch
 * Returns: { status, elasticsearch, mode }
 */
export async function checkHealth() {
  try {
    const response = await fetch(`${BASE}/health`);
    if (!response.ok) return null;
    return response.json();
  } catch (error) {
    return null;
  }
}

/**
 * Fetch last 100 audit log entries (newest first)
 * Returns: [ { timestamp, action, detail, hash }, ... ]
 */
export async function fetchAuditLog() {
  try {
    const response = await fetch(`${BASE}/audit/log`);
    if (!response.ok) return [];
    return response.json();
  } catch (error) {
    return [];
  }
}

/**
 * Verify the integrity of the audit log chain
 * Returns: { valid: bool, broken_at: number | null }
 */
export async function verifyAuditLog() {
  try {
    const response = await fetch(`${BASE}/audit/verify`);
    if (!response.ok) return null;
    return response.json();
  } catch (error) {
    return null;
  }
}
