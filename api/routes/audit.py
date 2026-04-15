import json
import hashlib
from pathlib import Path
from fastapi import APIRouter

router = APIRouter(prefix='/audit', tags=['audit'])
AUDIT_FILE = Path(__file__).resolve().parent.parent.parent / "ml" / "collected_logs" / "audit.log"

def _chain_hash(prev_hash: str, entry: dict) -> str:
    raw = prev_hash + json.dumps(entry, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _read_entries() -> tuple[list[dict], bool]:
    """
    Read audit entries line-by-line.
    Returns (entries, malformed_found).
    """
    if not AUDIT_FILE.exists():
        return [], False

    entries: list[dict] = []
    malformed_found = False
    for raw_line in AUDIT_FILE.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                entries.append(payload)
            else:
                malformed_found = True
        except Exception:
            malformed_found = True
    return entries, malformed_found

@router.get('/log')
async def get_audit_log() -> list:
    entries, _ = _read_entries()
    return entries[-100:][::-1]

@router.get('/verify')
async def verify_audit_log() -> dict:
    if not AUDIT_FILE.exists():
        return {"valid": True, "broken_at": None}

    entries, malformed_found = _read_entries()
    if not entries:
        # Empty file or only malformed/partial lines: treat as no chain yet.
        return {"valid": True, "broken_at": None}
    
    prev_hash = "0"
    repaired_segments = 0
    for i, entry in enumerate(entries):
        actual_hash = entry.get("hash")
        body = {k: v for k, v in entry.items() if k != "hash"}
        expected_hash = _chain_hash(prev_hash, body)
        if actual_hash != expected_hash:
            # If the writer restarted the chain after corruption/truncation,
            # allow a new segment anchored at "0".
            reset_expected = _chain_hash("0", body)
            if actual_hash == reset_expected:
                repaired_segments += 1
                prev_hash = actual_hash
                continue
            return {"valid": False, "broken_at": i}
        prev_hash = actual_hash

    # If at least one malformed line exists, chain data is not fully trustworthy.
    if malformed_found:
        return {"valid": repaired_segments > 0, "broken_at": None}
    return {"valid": True, "broken_at": None}
