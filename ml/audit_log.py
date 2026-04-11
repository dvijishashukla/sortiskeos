import json
import hashlib
from datetime import datetime
from pathlib import Path

AUDIT_FILE = Path(__file__).parent / "collected_logs" / "audit.log"

def _chain_hash(prev_hash: str, entry: dict) -> str:
    raw = prev_hash + json.dumps(entry, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()

def write_audit(action: str, detail: dict = None):
    if detail is None:
        detail = {}
    
    entries = []
    if AUDIT_FILE.exists():
        try:
            entries = [json.loads(l) for l in AUDIT_FILE.read_text().splitlines() if l.strip()]
        except Exception:
            pass

    prev_hash = entries[-1]["hash"] if entries else "0"
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        "detail": detail
    }
    entry["hash"] = _chain_hash(prev_hash, entry)
    
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
