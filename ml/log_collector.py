"""
log_collector.py
----------------
Runs automatically when the system restarts.
Detects the last shutdown/crash event and collects logs around that time window.
Windows only — reads from Windows Event Log via pywin32.

Output: writes collected logs to a staging file → picked up by Logstash.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR     = Path(__file__).parent / "collected_logs"
STAGING_FILE   = OUTPUT_DIR / "system_logs.json"  # Logstash watches this
WINDOW_MINUTES = 30    # collect logs from N minutes before the shutdown event
MAX_EVENTS     = 5000  # cap to avoid overwhelming the pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ── Event ID Classifications ───────────────────────────────────────────────────
CRITICAL_EVENT_IDS = {
    41,    # Kernel-Power: unexpected shutdown / crash
    6008,  # Dirty shutdown
    1001,  # BugCheck (BSOD)
    7034,  # Service crashed unexpectedly
    7031,  # Service terminated unexpectedly
    55,    # NTFS corruption
    29,    # Driver error
}

WARN_EVENT_IDS = {
    6006,  # Clean shutdown
    1014,  # DNS timeout
    10010, # DCOM error
    10016, # DCOM permission error
}

EVENT_ID_DESCRIPTIONS = {
    41:   "Kernel-Power: System rebooted without clean shutdown (crash/power loss)",
    6008: "EventLog: Previous shutdown was unexpected",
    6006: "EventLog: Clean system shutdown",
    1074: "User or application initiated shutdown or restart",
    6005: "EventLog service started - system boot",
    7034: "Service crashed unexpectedly",
    7031: "Service terminated unexpectedly",
    1001: "BugCheck: Windows stop error (BSOD)",
    55:   "NTFS: File system corruption detected",
    29:   "Driver error detected",
    98:   "Windows Update session started",
    12:   "Operating system started",
    13:   "Operating system shutdown",
}

SHUTDOWN_EVENT_IDS = {
    41:   "Kernel-Power: unexpected shutdown",
    6008: "EventLog: previous shutdown was unexpected",
    1074: "User initiated shutdown/restart",
    6006: "Clean shutdown",
}


# ══════════════════════════════════════════════════════════════════════════════
#  WINDOWS — reads Windows Event Log via pywin32
# ══════════════════════════════════════════════════════════════════════════════
def collect_windows_logs(window_minutes: int = WINDOW_MINUTES) -> list:
    """
    Read Windows Event Logs (System + Application channels).
    Finds the last unexpected shutdown event (Event ID 41 or 6008)
    and collects all events within window_minutes before it.
    """
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

            # First pass: find the last shutdown/crash event
            while True:
                events = win32evtlog.ReadEventLog(handle, flags, 0)
                if not events:
                    break
                for event in events:
                    if event.EventID in SHUTDOWN_EVENT_IDS:
                        shutdown_time = event.TimeGenerated
                        log.info(
                            f"Found shutdown event {event.EventID} "
                            f"at {shutdown_time}"
                        )
                        break
                if shutdown_time:
                    break

            # Default: use 1 hour ago if no shutdown event found
            if not shutdown_time:
                shutdown_time = datetime.now() - timedelta(hours=1)
                log.warning("No shutdown event found, using last 1 hour as window.")

            window_start = shutdown_time - timedelta(minutes=window_minutes)

            # Second pass: collect events inside the time window
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
                            "@timestamp": event_time.isoformat(),
                            "level":      _map_windows_level(
                                              event.EventType, event.EventID),
                            "source":     f"windows/{channel}",
                            "event_id":   event.EventID,
                            "message":    message,
                            "host":       os.environ.get("COMPUTERNAME", "unknown")
                        })

            win32evtlog.CloseEventLog(handle)

        except Exception as e:
            log.error(f"Error reading {channel} log: {e}")

    log.info(f"Collected {len(records)} Windows events.")
    return records[:MAX_EVENTS]


def _extract_message(event, channel: str, win32evtlogutil) -> str:
    """
    Extract a human-readable message from a Windows event.
    Falls back gracefully when SafeFormatMessage returns empty string.

    Priority:
      1. SafeFormatMessage  (full formatted message)
      2. StringInserts      (raw parameter strings)
      3. EVENT_ID_DESCRIPTIONS (known event ID description)
      4. Generic fallback
    """
    try:
        message = win32evtlogutil.SafeFormatMessage(event, channel)
        if message and message.strip():
            return message.strip().replace("\n", " ")
    except Exception:
        pass

    if event.StringInserts:
        return " | ".join(str(s) for s in event.StringInserts)

    if event.EventID in EVENT_ID_DESCRIPTIONS:
        return EVENT_ID_DESCRIPTIONS[event.EventID]

    return f"Windows Event ID {event.EventID}"


def _map_windows_level(event_type: int, event_id: int = 0) -> str:
    """
    Map a Windows event to a log level string.
    Checks event_id first (more specific), then falls back to event_type.
    """
    if event_id in CRITICAL_EVENT_IDS:
        return "ERROR"
    if event_id in WARN_EVENT_IDS:
        return "WARN"
    return {
        1:  "ERROR",
        2:  "WARN",
        4:  "INFO",
        8:  "DEBUG",
        16: "CRITICAL"
    }.get(event_type, "INFO")


# ══════════════════════════════════════════════════════════════════════════════
#  WRITE TO STAGING FILE — Logstash reads from here
# ══════════════════════════════════════════════════════════════════════════════
def write_to_staging(records: list):
    """
    Write collected log records as JSON Lines to the staging file.
    Logstash is configured to watch this file and ingest new entries.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(STAGING_FILE, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    log.info(f"Wrote {len(records)} records to {STAGING_FILE}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
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