"""
Hardened Windows log collector.

Key behavior:
- Writes real event time to `@timestamp` in UTC.
- Also stores `timestamp_ist` for display/debugging.
- Uses atomic writes for the local staging file.
- Optionally writes an HMAC manifest when SORTISKEOS_HMAC_KEY is set.
"""

import hashlib
import hmac
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "collected_logs"
STAGING_FILE = OUTPUT_DIR / "system_logs.json"
MANIFEST_FILE = OUTPUT_DIR / "system_logs.json.hmac"
WINDOW_MINUTES = 30
MAX_EVENTS = 5000

LOCAL_TZ = datetime.now().astimezone().tzinfo or timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HMAC_KEY = os.environ.get("SORTISKEOS_HMAC_KEY", "").encode()
if not HMAC_KEY:
    log.warning(
        "SORTISKEOS_HMAC_KEY not set. File integrity verification disabled. "
        "Set this env var to a random 32-byte hex string."
    )

CRITICAL_EVENT_IDS = {41, 6008, 1001, 7034, 7031, 55, 29}
WARN_EVENT_IDS = {6006, 1014, 10010, 10016}
SHUTDOWN_EVENT_IDS = {
    41: "Kernel-Power: unexpected shutdown",
    6008: "Dirty shutdown",
    1074: "User shutdown/restart",
    6006: "Clean shutdown",
}
EVENT_ID_DESCRIPTIONS = {
    41: "Kernel-Power: System rebooted without clean shutdown (crash/power loss)",
    6008: "EventLog: Previous shutdown was unexpected",
    6006: "EventLog: Clean system shutdown",
    1074: "User or application initiated shutdown or restart",
    6005: "EventLog service started - system boot",
    7034: "Service crashed unexpectedly",
    7031: "Service terminated unexpectedly",
    1001: "BugCheck: Windows stop error (BSOD)",
    55: "NTFS: File system corruption detected",
    29: "Driver error detected",
}


def _coerce_local_dt(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=LOCAL_TZ)
    return dt.astimezone(LOCAL_TZ)


def _to_utc_iso(dt: datetime) -> str:
    localized = _coerce_local_dt(dt).astimezone(timezone.utc)
    return localized.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _to_ist_str(dt: datetime) -> str:
    return _coerce_local_dt(dt).astimezone(IST).isoformat(timespec="seconds")


def _extract_message(event, channel: str, win32evtlogutil) -> str:
    try:
        msg = win32evtlogutil.SafeFormatMessage(event, channel)
        if msg and msg.strip():
            return msg.strip().replace("\n", " ")
    except Exception:
        pass

    if event.StringInserts:
        return " | ".join(str(item) for item in event.StringInserts)
    return EVENT_ID_DESCRIPTIONS.get(event.EventID, f"Windows Event ID {event.EventID}")


def _map_windows_level(event_type: int, event_id: int = 0) -> str:
    if event_id in CRITICAL_EVENT_IDS:
        return "ERROR"
    if event_id in WARN_EVENT_IDS:
        return "WARN"
    return {1: "ERROR", 2: "WARN", 4: "INFO", 8: "DEBUG", 16: "CRITICAL"}.get(event_type, "INFO")


def _read_recent_channel_events(channel: str, limit: int = MAX_EVENTS) -> list[dict]:
    try:
        import win32evtlog
        import win32evtlogutil
    except ImportError:
        log.error("pywin32 not installed. Run: pip install pywin32")
        sys.exit(1)

    log.info("Reading Windows Event Log channel: %s", channel)
    records: list[dict] = []
    handle = None
    try:
        handle = win32evtlog.OpenEventLog(None, channel)
        flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ

        while len(records) < limit:
            events = win32evtlog.ReadEventLog(handle, flags, 0)
            if not events:
                break

            for event in events:
                event_time = _coerce_local_dt(event.TimeGenerated)
                records.append(
                    {
                        "@timestamp": _to_utc_iso(event_time),
                        "timestamp_ist": _to_ist_str(event_time),
                        "level": _map_windows_level(event.EventType, event.EventID),
                        "source": f"windows/{channel}",
                        "event_id": event.EventID,
                        "message": _extract_message(event, channel, win32evtlogutil),
                        "host": os.environ.get("COMPUTERNAME", "unknown"),
                    }
                )
                if len(records) >= limit:
                    break
    except Exception as exc:
        log.error("Error reading %s log: %s", channel, exc)
    finally:
        if handle is not None:
            try:
                win32evtlog.CloseEventLog(handle)
            except Exception:
                pass

    return records


def collect_windows_logs(window_minutes: int = WINDOW_MINUTES) -> list[dict]:
    system_records = _read_recent_channel_events("System")
    application_records = _read_recent_channel_events("Application")

    shutdown_record = next(
        (record for record in system_records if record.get("event_id") in SHUTDOWN_EVENT_IDS),
        None,
    )
    if shutdown_record:
        shutdown_time = _parse_collected_timestamp(shutdown_record["@timestamp"])
        log.info("Found shutdown event %s at %s", shutdown_record.get("event_id"), shutdown_record.get("@timestamp"))
    else:
        shutdown_time = datetime.now(timezone.utc)
        log.warning("No shutdown event found, using current time as window end.")

    window_start = shutdown_time - timedelta(minutes=window_minutes)
    window_end = shutdown_time + timedelta(minutes=15)

    def _in_window(record: dict) -> bool:
        parsed = _parse_collected_timestamp(record.get("@timestamp"))
        return parsed is not None and window_start <= parsed <= window_end

    records = [record for record in [*system_records, *application_records] if _in_window(record)]
    records.sort(key=lambda item: item.get("@timestamp", ""), reverse=True)
    log.info("Collected %s Windows events in crash window.", len(records))
    return records[:MAX_EVENTS]


def _parse_collected_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _compute_hmac(data: bytes) -> str:
    if not HMAC_KEY:
        return ""
    return hmac.new(HMAC_KEY, data, hashlib.sha256).hexdigest()


def write_to_staging(records: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    content = "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    content_bytes = content.encode("utf-8")

    tmp_fd, tmp_path = tempfile.mkstemp(dir=OUTPUT_DIR, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "wb") as handle:
            handle.write(content_bytes)
        os.replace(tmp_path, STAGING_FILE)
    except Exception:
        os.unlink(tmp_path)
        raise

    STAGING_FILE.chmod(0o600)

    signature = _compute_hmac(content_bytes)
    if signature:
        MANIFEST_FILE.write_text(signature + "\n", encoding="utf-8")
        MANIFEST_FILE.chmod(0o600)
        log.info("HMAC manifest written to %s", MANIFEST_FILE)
    else:
        log.warning("Skipping HMAC manifest (SORTISKEOS_HMAC_KEY not set).")

    log.info("Wrote %s records to %s", len(records), STAGING_FILE)


def main() -> None:
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
