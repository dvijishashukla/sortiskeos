import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

CRASH_EVENT_IDS = {41, 6008, 1001, 1000, 1002, 7034, 7031, 55, 29}
PRIORITY_IDS = [41, 6008, 1001, 1000, 1002]
CRASH_KEYWORDS = [
    "unexpected shutdown",
    "previous shutdown was unexpected",
    "kernel power",
    "bugcheck",
    "did not shut down cleanly",
]

def parse_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        if isinstance(value, datetime):
            parsed = value
        else:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)

def is_crash_event(event: Dict[str, Any]) -> bool:
    """Checks if an event is a definitive crash marker."""
    raw_eid = event.get("event_id") or event.get("eventId")
    try:
        eid = int(raw_eid) if raw_eid is not None else None
    except (TypeError, ValueError):
        eid = None

    message = str(event.get("message", "")).lower()
    return (
        eid in CRASH_EVENT_IDS
        or any(kw in message for kw in CRASH_KEYWORDS)
        or "fault bucket" in message
        or "startuprepair" in message
        or "livekernelevent" in message
    )

def find_latest_crash_anchor(logs: List[Dict[str, Any]]) -> Tuple[Optional[datetime], Optional[Dict[str, Any]]]:
    """Finds the latest crash event in a list of logs using priority levels."""
    # Group hits by event ID for priority selection
    by_id: Dict[int, List[Dict[str, Any]]] = {eid: [] for eid in PRIORITY_IDS}
    others: List[Dict[str, Any]] = []

    for log in logs:
        raw_eid = log.get("event_id") or log.get("eventId")
        try:
            eid = int(raw_eid) if raw_eid is not None else None
        except (TypeError, ValueError):
            eid = None
        
        if eid in by_id:
            by_id[eid].append(log)
        elif is_crash_event(log):
            others.append(log)

    # Pick the latest log from the highest priority ID that exists
    for eid in PRIORITY_IDS:
        group = by_id[eid]
        if group:
            latest_log = max(group, key=lambda l: parse_timestamp(l.get("@timestamp") or l.get("time")) or datetime.min.replace(tzinfo=timezone.utc))
            return parse_timestamp(latest_log.get("@timestamp") or latest_log.get("time")), latest_log

    # Fallback to general crash events (keywords, etc)
    if others:
        latest_log = max(others, key=lambda l: parse_timestamp(l.get("@timestamp") or l.get("time")) or datetime.min.replace(tzinfo=timezone.utc))
        return parse_timestamp(latest_log.get("@timestamp") or latest_log.get("time")), latest_log
    
    return None, None

def get_forensic_window(anchor_time: Optional[datetime], current_time: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    """
    Returns the start and end of the forensic window.
    - If anchor exists: 6hr before anchor to 15min after.
    - If no anchor: Rolling 30min window from current_time.
    """
    if current_time is None:
        current_time = datetime.now(timezone.utc)
    
    if anchor_time:
        start = anchor_time - timedelta(hours=6)
        end = anchor_time + timedelta(minutes=15)
    else:
        start = current_time - timedelta(hours=6)
        end = current_time + timedelta(minutes=5) # buffer
        
    return start, end
