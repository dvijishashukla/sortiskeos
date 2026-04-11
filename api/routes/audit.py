import json
import hashlib
from pathlib import Path
from fastapi import APIRouter

router = APIRouter(prefix='/audit', tags=['audit'])
AUDIT_FILE = Path(__file__).resolve().parent.parent.parent / "ml" / "collected_logs" / "audit.log"

def _chain_hash(prev_hash: str, entry: dict) -> str:
    raw = prev_hash + json.dumps(entry, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()

@router.get('/log')
async def get_audit_log() -> list:
    if not AUDIT_FILE.exists():
        return []
    try:
        lines = [line.strip() for line in AUDIT_FILE.read_text().splitlines() if line.strip()]
        entries = [json.loads(line) for line in lines]
        return entries[-100:][::-1]
    except Exception:
        return []

@router.get('/verify')
async def verify_audit_log() -> dict:
    if not AUDIT_FILE.exists():
        return {"valid": True, "broken_at": None}
    try:
        lines = [line.strip() for line in AUDIT_FILE.read_text().splitlines() if line.strip()]
        entries = [json.loads(line) for line in lines]
    except Exception:
        return {"valid": False, "broken_at": -1}
    
    prev_hash = "0"
    for i, entry in enumerate(entries):
        actual_hash = entry.pop("hash", None)
        expected_hash = _chain_hash(prev_hash, entry)
        if actual_hash != expected_hash:
            return {"valid": False, "broken_at": i}
        entry["hash"] = actual_hash
        prev_hash = actual_hash
        
    return {"valid": True, "broken_at": None}
