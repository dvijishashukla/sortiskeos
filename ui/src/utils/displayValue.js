export function toDisplayText(value, fallback = "") {
  if (value == null) return fallback;
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);

  if (Array.isArray(value)) {
    const parts = value.map((item) => toDisplayText(item, "")).filter(Boolean);
    return parts.length ? parts.join(", ") : fallback;
  }

  if (typeof value === "object") {
    const preferredKeys = [
      "name",
      "label",
      "message",
      "title",
      "text",
      "value",
      "category",
      "likely_cause",
      "id",
    ];
    for (const key of preferredKeys) {
      const nested = toDisplayText(value[key], "");
      if (nested) return nested;
    }
    return fallback;
  }

  return fallback;
}

export function toDisplayNumber(value, fallback = 0) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

export function summarizeRootCause(message) {
  if (!message) return "Unknown Crash";

  const normalized = String(message).replace(/\s+/g, " ").trim();
  const lower = normalized.toLowerCase();
  const eventName = normalized.match(/Event Name:\s*([^:]+?)(?=\s+[A-Z][A-Za-z ]+:\s|$)/i)?.[1]?.trim();
  const removedUrl = normalized.match(/Removed URL\s*\((https?:\/\/[^)]+)\)/i)?.[1];

  const compactLabel = (value) => String(value)
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/[_\-\/]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .split(" ")
    .slice(0, 3)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
    .join(" ");

  if (eventName) {
    const simplifiedEventName = eventName.toLowerCase();
    if (simplifiedEventName.includes("startuprepaironline")) return "Startup Repair";
    if (simplifiedEventName.includes("kernel-power")) return "Kernel Power";
    if (simplifiedEventName.includes("bluescreen")) return "Blue Screen";
    if (simplifiedEventName.includes("appcrash")) return "App Crash";
    if (simplifiedEventName.includes("stoppedworking")) return "Service Failure";
    return compactLabel(eventName);
  }

  if (removedUrl) {
    try {
      const parsedUrl = new URL(removedUrl);
      const hostLabel = parsedUrl.hostname.replace(/^www\./i, "").split(".")[0];
      const pathParts = parsedUrl.pathname.split("/").filter(Boolean);
      const pathLabel = pathParts.slice(-2).join(" ");
      const label = compactLabel(pathLabel || hostLabel || "URL Event");
      if (label === "Upnp Eventing") return "UPnP Eventing";
      return label;
    } catch {
      return "URL Event";
    }
  }

  if (lower.includes("kernel-power") || lower.includes("event 41")) return "Kernel Power";
  if (lower.includes("startup repair")) return "Startup Repair";
  if (lower.includes("blue screen") || lower.includes("bugcheck")) return "Blue Screen";
  if (lower.includes("appcrash")) return "App Crash";
  if (lower.includes("driver")) return "Driver Failure";
  if (lower.includes("disk")) return "Disk Failure";
  if (lower.includes("memory")) return "Memory Error";
  if (lower.includes("network")) return "Network Fault";
  if (lower.includes("upnp")) return "UPnP Eventing";
  if (lower.includes("dns")) return "DNS Error";

  const faultBucket = normalized.match(/fault bucket\s*,?\s*type\s*\d+\s*event name:\s*([A-Za-z0-9_]+)/i)?.[1];
  if (faultBucket) return compactLabel(faultBucket);

  const leadingPhrase = normalized.match(/^[A-Za-z0-9._-]+(?:\s+[A-Za-z0-9._-]+){0,2}/)?.[0];
  if (leadingPhrase) return compactLabel(leadingPhrase);

  return "Unknown Crash";
}

