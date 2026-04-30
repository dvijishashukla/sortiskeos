export function parseAsLocal(value) {
  if (!value) return new Date("");
  let timeString = String(value);

  // If it's just a time like "14:41:09", prepending a default date allows it to parse
  if (timeString.length <= 8) {
    timeString = `1970-01-01T${timeString}`;
  }

  return new Date(timeString);
}

export function formatTimeFull(value) {
  const parsed = parseAsLocal(value);
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

export function formatTimeShort(value) {
  const parsed = parseAsLocal(value);
  if (Number.isNaN(parsed.getTime())) {
    if (typeof value === "string" && value.length >= 8) return value.slice(0, 8);
    return String(value);
  }
  return parsed.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

export function formatTimeLabel(value) {
  const parsed = parseAsLocal(value);
  if (Number.isNaN(parsed.getTime())) {
    if (typeof value === "string" && value.length > 5) return value.slice(0, 5);
    return String(value);
  }
  return parsed.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}
