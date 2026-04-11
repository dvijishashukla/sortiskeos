def check_log_clearing(logs: list[dict]) -> dict:
    clearing_events = []
    for log in logs:
        event_id = log.get("event_id") or log.get("EventID")
        if str(event_id) in ("1102", "104"):
            clearing_events.append({
                "timestamp": log.get("@timestamp") or log.get("time") or log.get("timestamp"),
                "event_id": str(event_id),
                "message": log.get("message", ""),
                "channel": log.get("source", "unknown")
            })
    return {
        "detected": len(clearing_events) > 0,
        "count": len(clearing_events),
        "events": clearing_events
    }
