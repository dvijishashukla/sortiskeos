"""
ml_pipeline.py
--------------
Resilient ML pipeline for Intelligent Log Analysis.

Pipeline:
  1. Fetch logs from Elasticsearch or local file fallback
  2. Normalize log messages
  3. Build TF-IDF feature matrix
  4. Detect anomalies (Isolation Forest)
  5. Cluster anomalies (DBSCAN)
  6. Suggest root cause
  7. Persist results to Elasticsearch or local JSON
"""

import json
import logging
import os
import re
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import solution_engine
import tamper_detection
import antiforensics_detector
import forensics_utils
from audit_log import write_audit
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

ES_HOST = "http://localhost:9200"
SOURCE_INDEX = "system-logs-*"
ANOMALY_INDEX = "log-anomalies"
BATCH_SIZE = 5000
POLL_INTERVAL = 60
ML_N_JOBS = int(os.getenv("ML_N_JOBS", "1"))

BASE_DIR = Path(__file__).resolve().parent
ROOT_ENV_FILE = BASE_DIR.parent / ".env"
ML_ENV_FILE = BASE_DIR / ".env"
load_dotenv(ROOT_ENV_FILE)
load_dotenv(ML_ENV_FILE, override=True)

COLLECTED_LOGS_DIR = BASE_DIR / "collected_logs"
LOCAL_LOG_FILE = COLLECTED_LOGS_DIR / "system_logs.json"
LOCAL_RESULTS_FILE = COLLECTED_LOGS_DIR / "ml_results.json"

logging.getLogger("elastic_transport").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Noise suppression: load known-benign Event IDs from external config so they
# can be tuned without touching this file.
# 4624/4634 (logon/logoff) are intentionally excluded from suppression —
# they are needed for lateral movement / pass-the-hash detection.
# ---------------------------------------------------------------------------
_NOISE_CONFIG_PATH = BASE_DIR / "noise_config.json"

def _load_event_id_sets() -> tuple[set[int], set[int]]:
    """Returns (noise_ids, critical_ids) loaded from noise_config.json."""
    try:
        with _NOISE_CONFIG_PATH.open("r", encoding="utf-8") as _f:
            _cfg = json.load(_f)
        noise = {int(x) for x in _cfg.get("suppressed_event_ids", [])}
        critical = {int(x) for x in _cfg.get("critical_event_ids", [])}
        log.info(
            "Event routing loaded — suppressed: %s IDs, critical bypass: %s IDs.",
            len(noise), len(critical),
        )
        return noise, critical
    except FileNotFoundError:
        log.warning("noise_config.json not found — event routing disabled.")
        return set(), set()
    except Exception as exc:
        log.warning("Failed to load noise_config.json: %s — event routing disabled.", exc)
        return set(), set()

NOISE_EVENT_IDS: set[int]
CRITICAL_EVENT_IDS: set[int]
NOISE_EVENT_IDS, CRITICAL_EVENT_IDS = _load_event_id_sets()


# Sync with config-loaded sets
CRASH_EVENT_IDS = CRITICAL_EVENT_IDS
CRASH_KEYWORDS = [
    "kernel-power",
    "rebooted without clean shutdown",
    "previous shutdown was unexpected",
    "bugcheck",
    "blue screen of death",
    "did not shut down cleanly",
]
LEVEL_WEIGHTS = {
    "CRITICAL": 2.5,
    "ERROR": 2.0,
    "WARN": 1.0,
    "WARNING": 1.0,
    "INFO": 0.3,
}
EVENT_SIGNAL_WEIGHTS = {
    "41": 4.0,
    "6008": 4.0,
    "1001": 4.0,
    "1000": 4.5,
    "1002": 4.2,
    "7034": 3.2,
    "7031": 3.2,
    "55": 4.2,
    "98": 3.8,
    "29": 4.0,
    "1014": 2.4,
    "10010": 2.4,
}
KEYWORD_SIGNAL_WEIGHTS = {
    "disk error": 4.2,
    "bad block": 4.2,
    "ntfs": 4.0,
    "controller error": 4.0,
    "io error": 4.0,
    "bugcheck": 4.0,
    "critical process died": 4.0,
    "did not shut down cleanly": 3.8,
    "unexpected shutdown": 3.8,
    "kernel power": 3.8,
    "memory": 3.0,
    "out of memory": 3.2,
    "oom": 3.2,
    "thermal": 3.0,
    "overheat": 3.2,
    "driver failed": 3.0,
    "service terminated": 2.8,
    "network adapter": 2.2,
    "dns": 2.0,
}
BENIGN_KEYWORD_PENALTIES = {
    "intelmeprov": 5.5,
    "has been registered in the windows management instrumentation namespace": 4.5,
    "may cause a security violation if it does not correctly impersonate user requests": 3.0,
    "service started successfully": 2.0,
    "service has started": 1.5,
    "successfully loaded and registered with filter manager": 2.0,
}
LEVEL_PRIORITY = {
    "CRITICAL": 4,
    "ERROR": 3,
    "WARN": 2,
    "WARNING": 2,
    "INFO": 1,
}


def _resolve_es_host() -> str:
    return os.getenv("ES_HOST", ES_HOST)


def _create_es_client(es_host: str) -> Elasticsearch:
    es_user = os.getenv("ES_USER", "elastic")
    es_password = os.getenv("ES_PASSWORD", "") or os.getenv("ELASTIC_PASSWORD", "")

    kwargs = {
        "hosts": es_host,
        "request_timeout": 3,
        "max_retries": 0,
        "retry_on_timeout": False,
    }
    if es_password:
        kwargs["basic_auth"] = (es_user, es_password)

    return Elasticsearch(
        **kwargs
    )


def _check_es_connection(es: Elasticsearch, es_host: str) -> bool:
    try:
        if es.ping():
            return True
    except Exception:
        pass

    log.warning(
        "Elasticsearch is unavailable at %s. Switching to Local Mode using %s and %s.",
        es_host,
        LOCAL_LOG_FILE.name,
        LOCAL_RESULTS_FILE.name,
    )
    return False


def _read_json_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []

    records: list[dict] = []
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


def _is_definitive_crash_log(row) -> bool:
    raw_event_id = row.get("event_id")
    try:
        event_id = int(raw_event_id) if raw_event_id is not None else None
    except (TypeError, ValueError):
        event_id = None

    message = str(row.get("message", "")).lower()
    return (
        event_id in CRASH_EVENT_IDS
        or any(keyword in message for keyword in CRASH_KEYWORDS)
        or "fault bucket" in message
        or "startuprepair" in message
        or "livekernelevent" in message
    )


def fetch_logs_from_es(es: Elasticsearch) -> pd.DataFrame:
    response = es.search(
        index=SOURCE_INDEX,
        body={
            "size": BATCH_SIZE,
            "sort": [{"@timestamp": {"order": "desc"}}],
            "query": {"match_all": {}}
        }
    )

    hits = response["hits"]["hits"]
    if not hits:
        log.warning("No logs found in Elasticsearch.")
        return pd.DataFrame()

    records = []
    for hit in hits:
        src = hit["_source"]
        records.append({
            "es_id": hit.get("_id", ""),
            "timestamp": src.get("@timestamp") or src.get("time") or datetime.now().isoformat(),
            "level": str(src.get("level", "INFO")).upper(),
            "message": src.get("message", ""),
            "source": src.get("source", "unknown"),
            "host": src.get("host", "unknown"),
            "event_id": src.get("event_id") or src.get("EventID")
        })

    df = pd.DataFrame(records)
    return df


def fetch_logs_from_file(path: Path = LOCAL_LOG_FILE) -> pd.DataFrame:
    records = _read_json_lines(path)
    if not records:
        log.warning("No local logs found at %s.", path)
        return pd.DataFrame()

    normalized = []
    for record in records[:BATCH_SIZE]:
        normalized.append({
            "timestamp": record.get("@timestamp") or record.get("time") or datetime.now().isoformat(),
            "level": str(record.get("level", "INFO")).upper(),
            "message": record.get("message") or record.get("log") or "",
            "source": record.get("source", "unknown"),
            "host": record.get("host", "unknown"),
            "event_id": record.get("event_id") or record.get("EventID")
        })

    df = pd.DataFrame(normalized)
    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    log.info("Fetched %s logs from local file %s.", len(df), path.name)
    return df


IP_PATTERN = re.compile(r"\b\d{1,3}(\.\d{1,3}){3}\b")
NUMBER_PATTERN = re.compile(r"\b\d+\b")
HEX_ID_PATTERN = re.compile(r"\b[a-f0-9]{8,}\b")
PATH_PATTERN = re.compile(r"/[\w/\-\.]+")
GUID_PATTERN = re.compile(
    r"\{?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\}?",
    re.IGNORECASE
)


def normalize_message(message: str) -> str:
    if not message or not isinstance(message, str):
        return "empty message"
    msg = GUID_PATTERN.sub(" ", message)
    msg = IP_PATTERN.sub(" ", msg)
    msg = PATH_PATTERN.sub(" ", msg)
    msg = HEX_ID_PATTERN.sub(" ", msg)
    msg = NUMBER_PATTERN.sub(" ", msg)
    msg = msg.lower()
    msg = re.sub(r"\s+", " ", msg).strip()
    return msg if msg else "empty message"
    

EXCLUDE_PATTERNS = [
    "credential manager credentials were read",
    "a logon was attempted using explicit credentials",
    "a new process has been created",
    "special privileges assigned",
    "an account was successfully logged on",
    "a user's local group membership was enumerated",
    "key migration operation",
    "key file operation", 
    "cryptographic operation",
    "microsoft software key storage",
    "microsoft connected devices platform",
    "google chromekey",
    "ecdsa_p256"
]


def parse_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_message"] = df["message"].apply(normalize_message)
    log.info("Log messages normalized.")
    return df


def build_feature_matrix(clean_messages: list, max_features: int = 500):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1
    )
    X = vectorizer.fit_transform(clean_messages).toarray()
    log.info(
        "Feature matrix shape: %s (%s logs x %s TF-IDF features).",
        X.shape,
        X.shape[0],
        X.shape[1],
    )
    return X, vectorizer


def detect_anomalies(X: np.ndarray, contamination: float = 0.05):
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=ML_N_JOBS
    )
    raw_predictions = model.fit_predict(X)
    labels = np.where(raw_predictions == -1, 1, 0)
    raw_scores = model.decision_function(X)
    max_score = float(np.max(raw_scores)) if len(raw_scores) else 0.0
    min_score = float(np.min(raw_scores)) if len(raw_scores) else 0.0
    span = max(max_score - min_score, 1e-9)
    normalized_scores = -((max_score - raw_scores) / span)
    normalized_scores = np.where(labels == 1, normalized_scores, 0.0)

    n_anomalies = int(labels.sum())
    log.info(
        "Anomaly detection complete: %s anomalies found out of %s logs (%.1f%%).",
        n_anomalies,
        len(labels),
        100 * n_anomalies / len(labels),
    )
    return labels, normalized_scores, raw_scores, model


def cluster_anomalies(X: np.ndarray, anomaly_indices: np.ndarray, eps: float = 0.8, min_samples: int = 3):
    if len(anomaly_indices) == 0:
        log.warning("No anomalies to cluster.")
        return np.array([])

    X_anomalies = X[anomaly_indices]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_anomalies)

    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean",
        n_jobs=ML_N_JOBS
    )
    cluster_labels = db.fit_predict(X_scaled)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = int((cluster_labels == -1).sum())
    log.info("DBSCAN found %s cluster(s) among anomalies (%s noise points).", n_clusters, n_noise)
    return cluster_labels


def _level_priority(level: str) -> int:
    return LEVEL_PRIORITY.get(str(level or "INFO").upper(), 1)


def _root_cause_min_priority(anomaly_df: pd.DataFrame) -> int:
    if anomaly_df.empty:
        return 1

    priorities = anomaly_df["level"].apply(_level_priority)
    if (priorities >= LEVEL_PRIORITY["ERROR"]).any():
        return LEVEL_PRIORITY["ERROR"]
    if (priorities >= LEVEL_PRIORITY["WARN"]).any():
        return LEVEL_PRIORITY["WARN"]
    return LEVEL_PRIORITY["INFO"]


def _deduplicate_anomaly_frame(anomaly_df: pd.DataFrame) -> pd.DataFrame:
    if anomaly_df.empty:
        return anomaly_df.copy()

    working = anomaly_df.copy()
    if "clean_message" not in working.columns:
        working["clean_message"] = working["message"].apply(normalize_message)

    working["level_priority"] = working["level"].apply(_level_priority)
    working["severity_score"] = working["anomaly_score"].apply(lambda value: abs(float(value or 0)))
    if "root_cause_score" not in working.columns:
        working["root_cause_score"] = 0.0
    else:
        working["root_cause_score"] = working["root_cause_score"].fillna(0.0).astype(float)
    working["count"] = 1

    deduped_rows: list[pd.Series] = []
    for _, group in working.groupby(["timestamp", "clean_message"], dropna=False, sort=False):
        ordered = group.sort_values(
            ["level_priority", "severity_score", "root_cause_score", "timestamp"],
            ascending=[False, False, False, False],
        )
        representative = ordered.iloc[0].copy()
        representative["count"] = int(len(group))
        deduped_rows.append(representative)

    deduped = pd.DataFrame(deduped_rows).reset_index(drop=True)
    deduped["count"] = deduped["count"].astype(int)
    deduped["level_priority"] = deduped["level_priority"].astype(int)
    deduped["severity_score"] = deduped["severity_score"].astype(float)
    deduped["root_cause_score"] = deduped["root_cause_score"].astype(float)
    return deduped


def suggest_root_cause(df: pd.DataFrame, anomaly_indices: np.ndarray, cluster_labels: np.ndarray, n_samples: int = 5) -> dict:
    if len(anomaly_indices) == 0:
        log.info("No anomalies detected - system appears healthy.")
        return {
            "root_cause_cluster": None,
            "anomaly_count": 0,
            "cluster_size": 0,
            "sample_messages": [],
            "log_levels": {},
            "root_cause_message": "",
            "suggestion": solution_engine.analyze_cluster([]),
        }

    anomaly_df = df.iloc[anomaly_indices].copy()
    anomaly_df["cluster"] = cluster_labels
    message_counts = anomaly_df["clean_message"].value_counts().to_dict()
    crash_anchor_time, crash_anchor_log = forensics_utils.find_latest_crash_anchor(df.to_dict("records"))
    crash_anchor_message = crash_anchor_log.get("message", "") if crash_anchor_log else ""
    anomaly_df["root_cause_score"] = anomaly_df.apply(
        lambda row: _score_root_cause_candidate(
            row,
            crash_anchor_time=crash_anchor_time,
            message_counts=message_counts,
        ),
        axis=1,
    )
    anomaly_df = _deduplicate_anomaly_frame(anomaly_df)
    anomaly_df["eligible_root_cause"] = anomaly_df["level_priority"] >= _root_cause_min_priority(anomaly_df)
    anomaly_df = anomaly_df.sort_values(
        ["eligible_root_cause", "level_priority", "severity_score", "root_cause_score", "timestamp"],
        ascending=[False, False, False, False, False],
    ).reset_index()
    anomaly_df["root_cause_rank"] = np.arange(1, len(anomaly_df) + 1)
    if "index" in anomaly_df.columns:
        df.loc[anomaly_df["index"], "root_cause_score"] = anomaly_df["root_cause_score"].astype(float).values
        df.loc[anomaly_df["index"], "root_cause_rank"] = anomaly_df["root_cause_rank"].astype(int).values
    valid_clusters = cluster_labels[cluster_labels != -1]

    print("\n" + "=" * 60)
    print("       ROOT CAUSE ANALYSIS REPORT")
    print("=" * 60)
    print(f"  Total anomalies detected  : {len(anomaly_indices)}")
    print(f"  Noise points (unclustered): {(cluster_labels == -1).sum()}")
    print(f"  Distinct failure clusters : {len(set(valid_clusters))}")
    print("=" * 60)

    if len(valid_clusters) == 0:
        print("[INFO] All anomalies are noise - no dominant cluster found.")
        _print_cluster_samples(anomaly_df[anomaly_df["cluster"] == -1], label="NOISE / UNCLUSTERED", n_samples=n_samples)
        sample_messages = anomaly_df["message"].head(n_samples).tolist()
        eligible_df = anomaly_df[anomaly_df["eligible_root_cause"]]
        top_row = eligible_df.iloc[0] if not eligible_df.empty else (anomaly_df.iloc[0] if not anomaly_df.empty else {})
        return {
            "root_cause_cluster": -1,
            "anomaly_count": int(anomaly_df["count"].sum()) if "count" in anomaly_df else len(anomaly_indices),
            "cluster_size": int(top_row.get("count", 1) or 1),
            "sample_messages": sample_messages,
            "log_levels": anomaly_df["level"].value_counts().to_dict(),
            "root_cause_message": sample_messages[0] if sample_messages else "",
            "description": (
                "No dense cluster formed, so the highest-ranked anomaly nearest the crash window "
                "was selected as the approximate root-cause candidate."
            ),
            "fix": "Inspect the top-ranked anomaly and nearby precursor events before the crash marker.",
            "top_root_cause_score": float(top_row.get("root_cause_score", 0) or 0),
            "root_cause_level": str(top_row.get("level", "") or ""),
            "root_cause_event_id": top_row.get("event_id"),
            "root_cause_timestamp": str(top_row.get("timestamp", "") or ""),
            "root_cause_count": int(top_row.get("count", 1) or 1),
            "suggestion": solution_engine.analyze_cluster(sample_messages),
        }

    cluster_summaries = []
    noise_logs = anomaly_df[anomaly_df["cluster"] == -1].copy()
    min_priority = _root_cause_min_priority(anomaly_df)
    for cluster_id in sorted(set(valid_clusters)):
        cluster_logs = anomaly_df[anomaly_df["cluster"] == cluster_id].copy()
        if cluster_logs.empty:
            continue
        cluster_size = len(cluster_logs)
        max_candidate = float(cluster_logs["root_cause_score"].max() or 0)
        mean_candidate = float(cluster_logs["root_cause_score"].mean() or 0)
        top_severity = abs(float(cluster_logs["anomaly_score"].min() or 0))
        top_level_priority = int(cluster_logs["level_priority"].max() or 1)
        repetition_bonus = min(2.5, max(0.0, cluster_size - 1) * 0.35)
        cluster_score = (max_candidate * 0.55) + (mean_candidate * 0.30) + (top_severity * 8.0) + repetition_bonus
        cluster_summaries.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_size": cluster_size,
                "cluster_score": float(cluster_score),
                "top_level_priority": top_level_priority,
                "top_severity": float(top_severity),
            }
        )

    eligible_cluster_summaries = [item for item in cluster_summaries if item["top_level_priority"] >= min_priority]
    if not eligible_cluster_summaries:
        eligible_cluster_summaries = cluster_summaries

    eligible_cluster_summaries.sort(
        key=lambda item: (
            item["top_level_priority"],
            item["top_severity"],
            item["cluster_score"],
            item["cluster_size"],
        ),
        reverse=True,
    )
    best_cluster = eligible_cluster_summaries[0]
    root_cluster_id = best_cluster["cluster_id"]
    root_cluster_size = best_cluster["cluster_size"]

    print(f"\n  [ROOT CAUSE CLUSTER] Cluster #{root_cluster_id}")
    print(f"     Log count in cluster  : {root_cluster_size}")
    print(f"     Ranking score        : {best_cluster['cluster_score']:.2f}")
    if crash_anchor_message:
        print(f"     Crash anchor         : {crash_anchor_message[:100]}")

    for cluster_info in cluster_summaries:
        cluster_id = cluster_info["cluster_id"]
        count = cluster_info["cluster_size"]
        cluster_logs = anomaly_df[anomaly_df["cluster"] == cluster_id]
        levels = cluster_logs["level"].value_counts().to_dict()
        marker = "  * ROOT CAUSE" if cluster_id == root_cluster_id else ""
        print(f"\n  --- Cluster #{cluster_id} ({count} logs){marker} ---")
        print(f"      Root-cause score: {cluster_info['cluster_score']:.2f}")
        print(f"      Log levels: {levels}")
        _print_cluster_samples(cluster_logs, label=f"Cluster #{cluster_id}", n_samples=n_samples)

    if len(noise_logs) > 0:
        print(f"\n  --- Noise / Unclustered ({len(noise_logs)} logs) ---")
        _print_cluster_samples(noise_logs, label="Noise", n_samples=n_samples)

    print("\n" + "=" * 60)

    root_logs = anomaly_df[anomaly_df["cluster"] == root_cluster_id]
    root_logs = root_logs[root_logs["eligible_root_cause"]] if (root_logs["eligible_root_cause"]).any() else root_logs
    root_logs = root_logs.sort_values(
        ["eligible_root_cause", "level_priority", "severity_score", "root_cause_score", "timestamp"],
        ascending=[False, False, False, False, False],
    )
    top_root_row = root_logs.iloc[0] if not root_logs.empty else {}
    sample_messages = root_logs["message"].head(n_samples).tolist()
    return {
        "root_cause_cluster": int(root_cluster_id),
        "anomaly_count": int(anomaly_df["count"].sum()) if "count" in anomaly_df else len(anomaly_indices),
        "cluster_size": int(root_cluster_size),
        "sample_messages": sample_messages,
        "log_levels": root_logs["level"].value_counts().to_dict(),
        "root_cause_message": str(top_root_row.get("message", sample_messages[0] if sample_messages else "") or ""),
        "description": (
            f"Selected cluster {root_cluster_id} because its anomalies were the strongest and closest "
            "to the latest crash marker, not just the most numerous."
        ),
        "fix": "Inspect the highest-ranked events in this cluster first, then correlate the surrounding precursor logs.",
        "top_root_cause_score": float(root_logs["root_cause_score"].max() or 0),
        "root_cause_level": str(top_root_row.get("level", "") or ""),
        "root_cause_event_id": top_root_row.get("event_id"),
        "root_cause_timestamp": str(top_root_row.get("timestamp", "") or ""),
        "root_cause_count": int(top_root_row.get("count", 1) or 1),
        "suggestion": solution_engine.analyze_cluster(sample_messages),
    }


def _print_cluster_samples(cluster_df: pd.DataFrame, label: str, n_samples: int):
    samples = cluster_df[["timestamp", "level", "message"]].head(n_samples)
    print(f"\n      Sample logs from {label}:")
    for _, row in samples.iterrows():
        print(f"        [{row['timestamp']}] {row['level']:7s}  {row['message']}")


def _find_latest_crash_anchor(df: pd.DataFrame) -> tuple[Optional[datetime], str]:
    best_time: Optional[datetime] = None
    best_message = ""

    for _, row in df.iterrows():
        raw_event_id = row.get("event_id")
        try:
            event_id = int(raw_event_id) if raw_event_id is not None else None
        except (TypeError, ValueError):
            event_id = None

        message = str(row.get("message", ""))
        normalized = message.lower()
        is_crash = (
            event_id in CRASH_EVENT_IDS
            or any(keyword in normalized for keyword in CRASH_KEYWORDS)
        )
        if not is_crash:
            continue

        parsed = forensics_utils.parse_timestamp(row.get("timestamp"))
        if parsed is None:
            continue
        if best_time is None or parsed > best_time:
            best_time = parsed
            best_message = message

    return best_time, best_message


def _score_root_cause_candidate(
    row: pd.Series,
    *,
    crash_anchor_time: Optional[datetime],
    message_counts: dict[str, int],
) -> float:
    message = str(row.get("message", "") or "")
    clean_message = str(row.get("clean_message", "") or "")
    level = str(row.get("level", "INFO") or "INFO").upper()
    try:
        event_id = str(int(row.get("event_id"))) if row.get("event_id") is not None else ""
    except (TypeError, ValueError):
        event_id = str(row.get("event_id") or "")

    score = min(5.0, abs(float(row.get("anomaly_score", 0) or 0)) * 25.0)
    score += LEVEL_WEIGHTS.get(level, 0.4)
    score += EVENT_SIGNAL_WEIGHTS.get(event_id, 0.0)

    lowered = message.lower()
    for keyword, weight in KEYWORD_SIGNAL_WEIGHTS.items():
        if keyword in lowered:
            score += weight
    for keyword, penalty in BENIGN_KEYWORD_PENALTIES.items():
        if keyword in lowered:
            score -= penalty

    repeat_count = message_counts.get(clean_message, 1)
    score += min(2.5, max(0, repeat_count - 1) * 0.4)

    event_time = forensics_utils.parse_timestamp(row.get("timestamp"))
    if crash_anchor_time is not None and event_time is not None:
        delta_seconds = (crash_anchor_time - event_time).total_seconds()
        if delta_seconds < -30:
            score -= 2.5
        elif delta_seconds < 0:
            score += 0.5
        elif delta_seconds <= 30:
            score += 4.5
        elif delta_seconds <= 120:
            score += 3.8
        elif delta_seconds <= 600:
            score += 3.0
        elif delta_seconds <= 1800:
            score += 2.0
        elif delta_seconds <= 7200:
            score += 0.8

    return round(float(score), 4)


def _infer_label(messages: list[str]) -> str:
    lowered = [message.lower() for message in messages if isinstance(message, str)]
    if any("disk" in message or "i/o" in message for message in lowered):
        return "Disk I/O"
    if any("memory" in message or "oom" in message or "commit" in message for message in lowered):
        return "Memory Pressure"
    if any("network" in message or "adapter" in message or "rsc" in message for message in lowered):
        return "Kernel / Network"
    return "Kernel / Network"


def save_results_locally(df: pd.DataFrame, anomaly_indices: np.ndarray, cluster_labels: np.ndarray, root_cause: dict, mode: str, tamper_detected: bool = False, antiforensics: dict = None) -> None:
    COLLECTED_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    anomalies = []
    root_cluster = root_cause.get("root_cause_cluster")
    root_timestamp = str(root_cause.get("root_cause_timestamp", "") or "")
    root_message = str(root_cause.get("root_cause_message", "") or "")
    if len(anomaly_indices) > 0:
        anomaly_df = df.iloc[anomaly_indices].copy()
        anomaly_df["cluster"] = cluster_labels
        anomaly_df = _deduplicate_anomaly_frame(anomaly_df)
        for _, row in anomaly_df.iterrows():
            row_rank = row.get("root_cause_rank", 0)
            safe_rank = int(row_rank) if pd.notna(row_rank) else 0
            is_root_cause = (
                (root_timestamp and str(row.get("timestamp", "") or "") == root_timestamp and str(row.get("message", "") or "") == root_message)
                or safe_rank == 1
            )
            count = int(row.get("count", 1) or 1)
            anomalies.append({
                "@timestamp": row["timestamp"],
                "time": row["timestamp"],
                "timestamp": row["timestamp"],
                "level": row["level"],
                "message": row["message"],
                "source": row.get("source", "unknown"),
                "host": row.get("host", "unknown"),
                "score": float(row.get("anomaly_score", 0)),
                "anomaly_score": float(row.get("anomaly_score", 0)),
                "raw_model_score": float(row.get("raw_model_score", 0) or 0),
                "risk_score": float(-row.get("anomaly_score", 0)) if row.get("anomaly_score", 0) < 0 else 0.0,
                "cluster": int(row["cluster"]),
                "cluster_id": int(row["cluster"]),
                "root_cause_score": float(row.get("root_cause_score", 0) or 0),
                "root_cause_rank": safe_rank,
                "isRootCause": is_root_cause,
                "is_root_cause": is_root_cause,
                "count": count,
                "rootCause": root_message if (root_cluster is not None and int(root_cluster) >= 0 and int(row["cluster"]) == int(root_cluster)) or is_root_cause else "",
                "suggestion": root_cause.get("suggestion", {}) if ((root_cluster is not None and int(root_cluster) >= 0 and int(row["cluster"]) == int(root_cluster)) or is_root_cause) else {},
                # scoring_method distinguishes ML-scored vs rule-bypassed critical events
                "scoring_method": str(row.get("scoring_method") or "IsolationForest"),
                "method": "IsolationForest+DBSCAN",
                "source_log_id": row.get("es_id"),
                "event_id": row.get("event_id"),
                # False for all docs reaching output — suppressed events are dropped earlier
                "noise_suppressed": False,
            })

    top_score = min((item["score"] for item in anomalies), default=0)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "mode": mode,
        "source_file": str(LOCAL_LOG_FILE),
        "summary": {
            "root_cause_cluster": root_cluster,
            "root_cause_message": root_cause.get("root_cause_message", ""),
            "anomaly_count": int(root_cause.get("anomaly_count", len(anomalies)) or 0),
            "cluster_size": int(root_cause.get("cluster_size", len(anomalies)) or 0),
            "sample_messages": root_cause.get("sample_messages", []),
            "log_levels": root_cause.get("log_levels", {}),
            "top_score": float(top_score),
            "top_root_cause_score": float(root_cause.get("top_root_cause_score", 0) or 0),
            "label": _infer_label(root_cause.get("sample_messages", [])),
            "description": (
                root_cause.get("description")
                or (
                    f"Local Mode identified {int(root_cause.get('anomaly_count', len(anomalies)) or 0)} anomalies "
                    f"from {LOCAL_LOG_FILE.name}."
                )
            ),
            "fix": root_cause.get("fix") or "Review the top anomalies in this cluster and correlate them with the collected local system logs.",
            "suggestion": root_cause.get("suggestion") or solution_engine.analyze_cluster([]),
            "tamper_detected": tamper_detected,
            "antiforensics": antiforensics or {"detected": False, "count": 0, "events": []},
        },
        "root_cause": {
            "cluster": root_cluster,
            "message": root_cause.get("root_cause_message", ""),
            "level": root_cause.get("root_cause_level", ""),
            "event_id": root_cause.get("root_cause_event_id"),
            "timestamp": root_cause.get("root_cause_timestamp", ""),
            "count": int(root_cause.get("root_cause_count", 1) or 1),
            "score": float(root_cause.get("top_root_cause_score", 0) or 0),
        },
        "anomalies": anomalies,
    }

    with LOCAL_RESULTS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    log.info("Saved %s anomalies to local results file %s.", len(anomalies), LOCAL_RESULTS_FILE.name)


def push_anomalies_to_es(es: Elasticsearch, df: pd.DataFrame, anomaly_indices: np.ndarray, cluster_labels: np.ndarray, root_cause: dict):
    if len(anomaly_indices) == 0:
        log.info("No anomalies to push.")
        return

    anomaly_df = df.iloc[anomaly_indices].copy()
    anomaly_df["cluster"] = cluster_labels
    anomaly_df = _deduplicate_anomaly_frame(anomaly_df)
    root_cluster = root_cause.get("root_cause_cluster")
    root_message = root_cause.get("root_cause_message", "")
    root_timestamp = str(root_cause.get("root_cause_timestamp", "") or "")

    actions = []
    for _, row in anomaly_df.iterrows():
        cluster_id = int(row["cluster"])
        row_rank = row.get("root_cause_rank", 0)
        safe_rank = int(row_rank) if pd.notna(row_rank) else 0
        is_root_cause = (
            (root_timestamp and str(row.get("timestamp", "") or "") == root_timestamp and str(row.get("message", "") or "") == root_message)
            or safe_rank == 1
        )
        source_log_id = row.get("es_id")
        source_timestamp = row.get("timestamp")
        source_message = row.get("message", "")
        stable_id = hashlib.sha1(
            f"{source_timestamp}|{row.get('clean_message', source_message)}".encode("utf-8")
        ).hexdigest()
        actions.append({
            "_index": ANOMALY_INDEX,
            "_id": stable_id,
            "_source": {
                "@timestamp": row["timestamp"],
                "time": row["timestamp"],
                "timestamp": row["timestamp"],
                "level": row["level"],
                "message": row["message"],
                "source": row.get("source", "unknown"),
                "host": row.get("host", "unknown"),
                "score": float(row.get("anomaly_score", 0)),
                "anomaly_score": float(row.get("anomaly_score", 0)),
                "raw_model_score": float(row.get("raw_model_score", 0) or 0),
                "risk_score": float(-row.get("anomaly_score", 0)) if row.get("anomaly_score", 0) < 0 else 0.0,
                "cluster": cluster_id,
                "cluster_id": cluster_id,
                "root_cause_score": float(row.get("root_cause_score", 0) or 0),
                "root_cause_rank": safe_rank,
                "isRootCause": is_root_cause,
                "is_root_cause": is_root_cause,
                "count": int(row.get("count", 1) or 1),
                "rootCause": root_message if ((root_cluster is not None and int(root_cluster) >= 0 and cluster_id == int(root_cluster)) or is_root_cause) else "",
                "suggestion": root_cause.get("suggestion", {}) if ((root_cluster is not None and int(root_cluster) >= 0 and cluster_id == int(root_cluster)) or is_root_cause) else {},
                # scoring_method distinguishes ML-scored vs rule-bypassed critical events
                "scoring_method": str(row.get("scoring_method") or "IsolationForest"),
                "method": "IsolationForest+DBSCAN",
                "source_log_id": source_log_id,
                "event_id": row.get("event_id"),
                # False for all docs reaching output — suppressed events are dropped earlier
                "noise_suppressed": False,
                "source_timestamp": source_timestamp,
                "analysed_at": datetime.now().isoformat(),
            }
        })

    helpers.bulk(es, actions)
    log.info("Pushed %s anomalies to index '%s'.", len(actions), ANOMALY_INDEX)


def run_analysis(es: Optional[Elasticsearch] = None, mode: str = "local"):
    log.info("%s", "-" * 50)
    log.info("Starting ML analysis cycle in %s mode...", mode)
    write_audit("pipeline_start")

    tamper_detected = False
    if es is None:
        hash_file_path = COLLECTED_LOGS_DIR / "system_logs.hash"
        if LOCAL_LOG_FILE.exists() and not tamper_detection.verify_hash(LOCAL_LOG_FILE, hash_file_path):
            log.critical("TAMPER ALERT: system_logs.json has been modified since last collection")
            tamper_detected = True
            write_audit("tamper_detected", {"file": "system_logs.json"})

    df = fetch_logs_from_es(es) if es is not None else fetch_logs_from_file()
    
    antiforensics_result = None
    if not df.empty:
        antiforensics_result = antiforensics_detector.check_log_clearing(df.to_dict("records"))
        if antiforensics_result.get("detected"):
            log.critical("ANTI-FORENSICS ALERT: Log clearing detected %s time(s) before crash analysis", antiforensics_result.get("count"))
            write_audit("antiforensics_detected", {"count": antiforensics_result.get("count")})

    if df.empty:
        log.warning("No logs to analyze.")
        save_results_locally(df, np.array([]), np.array([]), {}, mode, tamper_detected, antiforensics_result)
        if es is None and LOCAL_LOG_FILE.exists():
            tamper_detection.save_hash(LOCAL_LOG_FILE, COLLECTED_LOGS_DIR / "system_logs.hash")
        return

    # --- Step 0: Shared Anchor Logic & Forensic Window Calculation ---
    logs_list = df.to_dict("records")
    anchor_time, anchor_log = forensics_utils.find_latest_crash_anchor(logs_list)
    window_start, window_end = forensics_utils.get_forensic_window(anchor_time)
    
    if anchor_time:
        log.info("Crash anchor detected at %s. Applying 6-hour forensic window.", anchor_time)
    else:
        log.info("No crash anchor detected. Applying rolling 30-minute window.")

    # --- Step 1: DirectCritical Bypass (BEFORE time filtering) ---
    # Bug Fix: Extract critical events from the FULL batch so they aren't lost to windowing.
    df["_eid"] = pd.to_numeric(df["event_id"], errors="coerce")
    critical_mask = df["_eid"].isin(CRITICAL_EVENT_IDS) if CRITICAL_EVENT_IDS else pd.Series(False, index=df.index)
    critical_df = df[critical_mask].drop(columns=["_eid"]).copy()
    
    if not critical_df.empty:
        log.info("Critical bypass: Captured %s events from full batch.", len(critical_df))
        critical_df["anomaly"] = 1
        critical_df["anomaly_score"] = -0.95
        critical_df["raw_model_score"] = -0.95
        critical_df["scoring_method"] = "DirectCritical"

    # --- Step 2: Time Window Filtering (for non-critical logs) ---
    parsed_times = df["timestamp"].apply(forensics_utils.parse_timestamp)
    df = df[parsed_times.apply(lambda ts: ts is not None and window_start <= ts <= window_end)].copy()
    log.info("Forensic window filtered: %s logs remain for ML analysis.", len(df))

    df = parse_and_normalize(df)

    # --- Step 3: Suppress known-noisy Event IDs ---
    if NOISE_EVENT_IDS:
        pre_noise_count = len(df)
        df["_event_id_int"] = pd.to_numeric(df["event_id"], errors="coerce")
        df = df[~df["_event_id_int"].isin(NOISE_EVENT_IDS)].drop(columns=["_event_id_int"]).reset_index(drop=True)
        suppressed = pre_noise_count - len(df)
        if suppressed > 0:
            log.info("Noise suppression: dropped %s events.", suppressed)

    # --- Step 4: Filter out excluded text patterns ---
    initial_count = len(df)
    df = df[~df["clean_message"].str.contains("|".join(EXCLUDE_PATTERNS), case=False, na=False)].reset_index(drop=True)
    if len(df) < initial_count:
        log.info("Filtered out %s normal logs.", initial_count - len(df))

    # --- Step 5: ML scoring on filtered non-critical logs ---
    # Ensure we don't re-score critical events that were already captured in Step 1
    # although Step 1 captured them from full batch, they might be in the current windowed df too.
    df["_eid"] = pd.to_numeric(df["event_id"], errors="coerce")
    current_critical_mask = df["_eid"].isin(CRITICAL_EVENT_IDS) if CRITICAL_EVENT_IDS else pd.Series(False, index=df.index)
    ml_df = df[~current_critical_mask].drop(columns=["_eid"]).reset_index(drop=True)
    df = df.drop(columns=["_eid"])

    ml_df["anomaly"] = 0
    ml_df["anomaly_score"] = 0.0
    ml_df["raw_model_score"] = 0.0
    ml_df["scoring_method"] = "IsolationForest"

    if len(ml_df) >= 5:
        X, _ = build_feature_matrix(ml_df["clean_message"].tolist(), max_features=500)
        log_count = len(ml_df)
        contamination = 0.1 if log_count < 50 else (0.05 if log_count < 200 else 0.02)
        log.info("ML scoring %s non-critical logs (contamination=%.2f).", log_count, contamination)
        anomaly_labels, anomaly_scores, raw_model_scores, _ = detect_anomalies(X, contamination=contamination)
        ml_df["anomaly"] = anomaly_labels
        ml_df["anomaly_score"] = anomaly_scores
        ml_df["raw_model_score"] = raw_model_scores
    else:
        log.warning("Too few non-critical logs for ML scoring.")

    # --- Step 6: Merge critical + ML results ---
    # Merge Step 1 (unfiltered criticals) with Step 5 (ML results from window)
    df = pd.concat([critical_df, ml_df], ignore_index=True).sort_values(
        "timestamp", ascending=False
    ).drop_duplicates(subset=["timestamp", "message", "event_id"]).reset_index(drop=True)
    if "clean_message" not in df.columns:
        df["clean_message"] = df["message"].apply(normalize_message)
    else:
        missing_clean_message = df["clean_message"].isna()
        if missing_clean_message.any():
            df.loc[missing_clean_message, "clean_message"] = df.loc[missing_clean_message, "message"].apply(normalize_message)
    anomaly_indices = np.where(df["anomaly"] == 1)[0]

    # Build solution suggestion
    crash_messages = [
        f"EventID {row.get('event_id', '')}: {row.get('message', '')}"
        for row in critical_df.to_dict("records")
    ] if not critical_df.empty else []
    direct_suggestion = solution_engine.analyze_cluster(crash_messages) if crash_messages else None

    cluster_labels = cluster_anomalies(
        # Re-build feature matrix on full df for clustering context
        build_feature_matrix(df["clean_message"].tolist(), max_features=500)[0],
        anomaly_indices,
        eps=0.8,
        min_samples=3,
    )
    root_cause = suggest_root_cause(df, anomaly_indices, cluster_labels)

    if direct_suggestion:
        root_cause["suggestion"] = direct_suggestion
        if not root_cause.get("root_cause_message") and crash_messages:
            root_cause["root_cause_message"] = crash_messages[0]

    if es is not None:
        push_anomalies_to_es(es, df, anomaly_indices, cluster_labels, root_cause)
    
    save_results_locally(df, anomaly_indices, cluster_labels, root_cause, mode, tamper_detected, antiforensics_result)
    
    if es is None and LOCAL_LOG_FILE.exists():
        tamper_detection.save_hash(LOCAL_LOG_FILE, COLLECTED_LOGS_DIR / "system_logs.hash")

    write_audit("pipeline_complete", {
        "anomaly_count": len(anomaly_indices),
        "root_cause": (root_cause.get("root_cause_message") or "")[:80] or "none"
    })
    log.info("Analysis cycle complete.")


def run_once():
    es_host = _resolve_es_host()
    es = _create_es_client(es_host)
    if _check_es_connection(es, es_host):
        log.info("Connected to Elasticsearch at %s", es_host)
        run_analysis(es, mode="elasticsearch")
        return
    run_analysis(mode="local")


def run_realtime(interval: int = POLL_INTERVAL):
    es_host = _resolve_es_host()
    es = _create_es_client(es_host)
    log.info("Real-time mode: analysing every %ss. Ctrl+C to stop.", interval)

    while True:
        try:
            if _check_es_connection(es, es_host):
                run_analysis(es, mode="elasticsearch")
            else:
                run_analysis(mode="local")
            log.info("Sleeping %ss until next analysis...", interval)
            time.sleep(interval)
        except KeyboardInterrupt:
            log.info("Stopped by user.")
            break
        except Exception as exc:
            log.error("Pipeline error: %s. Retrying in %ss...", exc, interval)
            time.sleep(interval)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ML Pipeline for Intelligent Log Analysis")
    parser.add_argument(
        "--mode",
        choices=["once", "realtime"],
        default="once",
        help="'once' = single run, 'realtime' = continuous polling"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=POLL_INTERVAL,
        help=f"Poll interval in seconds for realtime mode [default: {POLL_INTERVAL}]"
    )
    args = parser.parse_args()

    if args.mode == "realtime":
        run_realtime(args.interval)
    else:
        run_once()
