import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from elasticsearch import AsyncElasticsearch

BASE_DIR = Path(__file__).resolve().parent
ML_DIR = BASE_DIR.parent / "ml"
COLLECTED_LOGS_DIR = ML_DIR / "collected_logs"
SYSTEM_LOGS_FILE = COLLECTED_LOGS_DIR / "system_logs.json"
ML_RESULTS_FILE = COLLECTED_LOGS_DIR / "ml_results.json"
SYSTEM_LOGS_INDEX = "system-logs-*"
ANOMALIES_INDEX = "log-anomalies"


async def is_es_available(es: Optional[AsyncElasticsearch]) -> bool:
    if es is None:
        return False
    try:
        return bool(await es.ping())
    except Exception:
        return False


def read_json_lines(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    records.append(payload)
            except json.JSONDecodeError:
                continue
    return records


def read_ml_results() -> Dict[str, Any]:
    if not ML_RESULTS_FILE.exists():
        return {}

    try:
        with ML_RESULTS_FILE.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def get_local_logs() -> List[Dict[str, Any]]:
    records = read_json_lines(SYSTEM_LOGS_FILE)
    records.sort(key=lambda item: item.get("@timestamp") or item.get("time") or "", reverse=True)
    return records


def get_local_anomalies() -> List[Dict[str, Any]]:
    payload = read_ml_results()
    anomalies = payload.get("anomalies", [])
    if not isinstance(anomalies, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in anomalies:
        if not isinstance(item, dict):
            continue
        normalized.append(item)

    normalized.sort(key=lambda item: item.get("@timestamp") or item.get("time") or "", reverse=True)
    return normalized


def get_local_summary() -> Dict[str, Any]:
    payload = read_ml_results()
    summary = payload.get("summary", {})
    return summary if isinstance(summary, dict) else {}
