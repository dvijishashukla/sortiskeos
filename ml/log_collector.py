"""
log_collector.py — HARDENED
----------------------------
Changes from original:
  - Writes timestamps in IST (UTC+5:30) not naive local time
  - Staging file written atomically (tmp → rename) — no partial-read window
  - HMAC-SHA256 manifest file written alongside staging file
    so ml_pipeline.py can verify the file wasn't tampered with
  - HMAC key loaded from SORTISKEOS_HMAC_KEY env var (set in .env)
  - Output file has 0o600 permissions (owner read/write only)
"""

import os
import sys
import json
import hmac
import hashlib
import logging
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR     = Path(__file__).parent / "collected_logs"
STAGING_FILE   = OUTPUT_DIR / "system_logs.json"
MANIFEST_FILE  = OUTPUT_DIR / "system_logs.json.hmac"
WINDOW_MINUTES = 30
MAX_EVENTS     = 5000

IST = timezone(timedelta(hours=5, minutes=30))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── HMAC key — must be set in environment ──────────────────────────────────────
HMAC_KEY = os.environ.get("SORTISKEOS_HMAC_KEY", "").encode()
if not HMAC_KEY:
    log.warning(
        "SORTISKEOS_HMAC_KEY not set. File integrity verification disabled. "
        "Set this env var to a random 32-byte hex string."
    )

# ── Event ID Classifications ───────────────────────────────────────────────────
CRITICAL_EVENT_IDS = {41, 6008, 1001, 7034, 7031, 55, 29}
WARN_EVENT_IDS     = {6006, 1014, 10010, 10016}
SHUTDOWN_EVENT_IDS = {41: "Kernel-Power: unexpected shutdown", 6008: "Dirty shutdown",
                      1074: "User shutdown/restart", 6006: "Clean shutdown"}
EVENT_ID_DESCRIPTIONS = {
    41:    "Kernel-Power: System rebooted without clean shutdown (crash/power loss)",
    6008:  "EventLog: Previous shutdown was unexpected",
    6006:  "EventLog: Clean system shutdown",
    1074:  "User or application initiated shutdown or restart",
    6005:  "EventLog service started - system boot",
    7034:  "Service crashed unexpectedly",
    7031:  "Service terminated unexpectedly",
    1001:  "BugCheck: Windows stop error (BSOD)",
    55:    "NTFS: File system corruption detected",
    29:    "Driver error detected",
}


def _to_ist_str(dt: datetime) -> str:
    """Convert any datetime to an IST-formatted string."""
    if dt.tzinfo is None:
        # assume local Windows time — convert as UTC first to be safe
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST).strftime("%Y-%m-%dT%H:%M:%S+05:30")


def collect_windows_logs(window_minutes: int = WINDOW_MINUTES) -> list:
    try:
        import win32evtlog
        import win32evtlogutil
    except ImportError:
        log.error("pywin32 not installed. Run: pip install pywin32")
        sys.exit(1)

    records  = []
    channels = ["System", "Application"]

    for channel in channels:
        log.info(f"Reading Windows Event Log channel: {channel}")
        try:
            handle = win32evtlog.OpenEventLog(None, channel)
            flags  = (win32evtlog.EVENTLOG_BACKWARDS_READ |
                      win32evtlog.EVENTLOG_SEQUENTIAL_READ)

            shutdown_time = None

            while True:
                events = win32evtlog.ReadEventLog(handle, flags, 0)
                if not events:
                    break
                for event in events:
                    if event.EventID in SHUTDOWN_EVENT_IDS:
                        shutdown_time = event.TimeGenerated
                        log.info(f"Found shutdown event {event.EventID} at {shutdown_time}")
                        break
                if shutdown_time:
                    break

            if not shutdown_time:
                shutdown_time = datetime.now()
                log.warning("No shutdown event found, using current time as window end.")

            window_start = shutdown_time - timedelta(minutes=window_minutes)
            win32evtlog.CloseEventLog(handle)
            handle = win32evtlog.OpenEventLog(None, channel)

            while True:
                events = win32evtlog.ReadEventLog(handle, flags, 0)
                if not events:
                    break
                for event in events:
                    event_time = event.TimeGenerated.replace(tzinfo=None)
                    if event_time < window_start:
                        break
                    if event_time <= shutdown_time:
                        message = _extract_message(event, channel, win32evtlogutil)
                        records.append({
                            "@timestamp": _to_ist_str(event_time),  # IST
                            "timestamp_ist": _to_ist_str(event_time),
                            "level":   _map_windows_level(event.EventType, event.EventID),
                            "source":  f"windows/{channel}",
                            "event_id": event.EventID,
                            "message": message,
                            "host":    os.environ.get("COMPUTERNAME", "unknown"),
                        })

            win32evtlog.CloseEventLog(handle)
        except Exception as e:
            log.error(f"Error reading {channel} log: {e}")

    log.info(f"Collected {len(records)} Windows events.")
    return records[:MAX_EVENTS]


def _extract_message(event, channel: str, win32evtlogutil) -> str:
    try:
        msg = win32evtlogutil.SafeFormatMessage(event, channel)
        if msg and msg.strip():
            return msg.strip().replace("\n", " ")
    except Exception:
        pass
    if event.StringInserts:
        return " | ".join(str(s) for s in event.StringInserts)
    return EVENT_ID_DESCRIPTIONS.get(event.EventID, f"Windows Event ID {event.EventID}")


def _map_windows_level(event_type: int, event_id: int = 0) -> str:
    if event_id in CRITICAL_EVENT_IDS:
        return "ERROR"
    if event_id in WARN_EVENT_IDS:
        return "WARN"
    return {1: "ERROR", 2: "WARN", 4: "INFO", 8: "DEBUG", 16: "CRITICAL"}.get(event_type, "INFO")


def _compute_hmac(data: bytes) -> str:
    if not HMAC_KEY:
        return ""
    return hmac.new(HMAC_KEY, data, hashlib.sha256).hexdigest()


def write_to_staging(records: list):
    """
    Atomically write records to the staging file.
    1. Write to a temp file in the same directory.
    2. Compute HMAC of the temp file content.
    3. Rename temp → final (atomic on same filesystem).
    4. Write HMAC manifest alongside.
    5. Set file permissions to 0o600.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    content = "\n".join(json.dumps(r) for r in records) + "\n"
    content_bytes = content.encode("utf-8")

    # Atomic write via tmp → rename
    tmp_fd, tmp_path = tempfile.mkstemp(dir=OUTPUT_DIR, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(content_bytes)
        os.replace(tmp_path, STAGING_FILE)   # atomic rename
    except Exception:
        os.unlink(tmp_path)
        raise

    # Permissions: owner read/write only
    STAGING_FILE.chmod(0o600)

    # Write HMAC manifest
    signature = _compute_hmac(content_bytes)
    if signature:
        MANIFEST_FILE.write_text(signature + "\n", encoding="utf-8")
        MANIFEST_FILE.chmod(0o600)
        log.info(f"HMAC manifest written to {MANIFEST_FILE}")
    else:
        log.warning("Skipping HMAC manifest (SORTISKEOS_HMAC_KEY not set).")

    log.info(f"Wrote {len(records)} records to {STAGING_FILE}")


def main():
    log.info("Detected OS: Windows")
    log.info("Starting log collection...")

    records = collect_windows_logs()

    if not records:
        log.warning("No log records collected. Check permissions or log sources.")
        sys.exit(0)

    write_to_staging(records)
    log.info("Log collection complete. Logstash will pick up the staging file.")


if __name__ == "__main__":
    main()
