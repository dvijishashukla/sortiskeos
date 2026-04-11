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
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import solution_engine
import tamper_detection
import antiforensics_detector
from audit_log import write_audit
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch, helpers
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
BATCH_SIZE = 1000
POLL_INTERVAL = 60
ML_N_JOBS = int(os.getenv("ML_N_JOBS", "1"))

BASE_DIR = Path(__file__).resolve().parent
COLLECTED_LOGS_DIR = BASE_DIR / "collected_logs"
LOCAL_LOG_FILE = COLLECTED_LOGS_DIR / "system_logs.json"
LOCAL_RESULTS_FILE = COLLECTED_LOGS_DIR / "ml_results.json"

logging.getLogger("elastic_transport").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def _resolve_es_host() -> str:
    return os.getenv("ES_HOST", ES_HOST)


def _create_es_client(es_host: str) -> Elasticsearch:
    return Elasticsearch(
        es_host,
        request_timeout=3,
        max_retries=0,
        retry_on_timeout=False,
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
            "timestamp": src.get("@timestamp", datetime.now().isoformat()),
            "level": str(src.get("level", "INFO")).upper(),
            "message": src.get("message", ""),
            "source": src.get("source", "unknown"),
            "host": src.get("host", "unknown"),
            "event_id": src.get("event_id") or src.get("EventID")
        })

    df = pd.DataFrame(records)
    log.info("Fetched %s logs from Elasticsearch.", len(df))
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


def parse_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_message"] = df["message"].apply(normalize_message)
    log.info("Log messages normalized.")
    return df


def build_feature_matrix(clean_messages: list, max_features: int = 100):
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


def detect_anomalies(X: np.ndarray, contamination: float = 0.2):
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=ML_N_JOBS
    )
    raw_predictions = model.fit_predict(X)
    labels = np.where(raw_predictions == -1, 1, 0)
    scores = model.decision_function(X)

    n_anomalies = int(labels.sum())
    log.info(
        "Anomaly detection complete: %s anomalies found out of %s logs (%.1f%%).",
        n_anomalies,
        len(labels),
        100 * n_anomalies / len(labels),
    )
    return labels, scores, model


def cluster_anomalies(X: np.ndarray, anomaly_indices: np.ndarray, eps: float = 0.8, min_samples: int = 2):
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
        return {
            "root_cause_cluster": None,
            "anomaly_count": len(anomaly_indices),
            "cluster_size": len(anomaly_indices),
            "sample_messages": sample_messages,
            "log_levels": anomaly_df["level"].value_counts().to_dict(),
            "root_cause_message": sample_messages[0] if sample_messages else "",
            "suggestion": solution_engine.analyze_cluster(sample_messages),
        }

    cluster_counts = Counter(valid_clusters)
    root_cluster_id, root_cluster_size = cluster_counts.most_common(1)[0]

    print(f"\n  [ROOT CAUSE CLUSTER] Cluster #{root_cluster_id}")
    print(f"     Log count in cluster  : {root_cluster_size}")

    for cluster_id, count in cluster_counts.most_common():
        cluster_logs = anomaly_df[anomaly_df["cluster"] == cluster_id]
        levels = cluster_logs["level"].value_counts().to_dict()
        marker = "  * ROOT CAUSE" if cluster_id == root_cluster_id else ""
        print(f"\n  --- Cluster #{cluster_id} ({count} logs){marker} ---")
        print(f"      Log levels: {levels}")
        _print_cluster_samples(cluster_logs, label=f"Cluster #{cluster_id}", n_samples=n_samples)

    noise_logs = anomaly_df[anomaly_df["cluster"] == -1]
    if len(noise_logs) > 0:
        print(f"\n  --- Noise / Unclustered ({len(noise_logs)} logs) ---")
        _print_cluster_samples(noise_logs, label="Noise", n_samples=n_samples)

    print("\n" + "=" * 60)

    root_logs = anomaly_df[anomaly_df["cluster"] == root_cluster_id]
    sample_messages = root_logs["message"].head(n_samples).tolist()
    return {
        "root_cause_cluster": int(root_cluster_id),
        "anomaly_count": len(anomaly_indices),
        "cluster_size": int(root_cluster_size),
        "sample_messages": sample_messages,
        "log_levels": root_logs["level"].value_counts().to_dict(),
        "root_cause_message": sample_messages[0] if sample_messages else "",
        "suggestion": solution_engine.analyze_cluster(sample_messages),
    }


def _print_cluster_samples(cluster_df: pd.DataFrame, label: str, n_samples: int):
    samples = cluster_df[["timestamp", "level", "message"]].head(n_samples)
    print(f"\n      Sample logs from {label}:")
    for _, row in samples.iterrows():
        print(f"        [{row['timestamp']}] {row['level']:7s}  {row['message']}")


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
    if len(anomaly_indices) > 0:
        anomaly_df = df.iloc[anomaly_indices].copy()
        anomaly_df["cluster"] = cluster_labels
        for _, row in anomaly_df.iterrows():
            anomalies.append({
                "@timestamp": row["timestamp"],
                "time": row["timestamp"],
                "level": row["level"],
                "message": row["message"],
                "source": row.get("source", "unknown"),
                "host": row.get("host", "unknown"),
                "score": float(row.get("anomaly_score", 0)),
                "anomaly_score": float(row.get("anomaly_score", 0)),
                "cluster": int(row["cluster"]),
                "cluster_id": int(row["cluster"]),
                "isRootCause": bool(root_cluster is not None and row["cluster"] == root_cluster),
                "is_root_cause": bool(root_cluster is not None and row["cluster"] == root_cluster),
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
            "label": _infer_label(root_cause.get("sample_messages", [])),
            "description": (
                f"Local Mode identified {int(root_cause.get('anomaly_count', len(anomalies)) or 0)} anomalies "
                f"from {LOCAL_LOG_FILE.name}."
            ),
            "fix": "Review the top anomalies in this cluster and correlate them with the collected local system logs.",
            "suggestion": root_cause.get("suggestion", {}),
            "tamper_detected": tamper_detected,
            "antiforensics": antiforensics or {"detected": False, "count": 0, "events": []},
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
    root_cluster = root_cause.get("root_cause_cluster")
    root_message = root_cause.get("root_cause_message", "")

    actions = []
    for _, row in anomaly_df.iterrows():
        cluster_id = int(row["cluster"])
        is_root_cause = bool(root_cluster is not None and cluster_id == root_cluster)
        actions.append({
            "_index": ANOMALY_INDEX,
            "_source": {
                "@timestamp": row["timestamp"],
                "time": row["timestamp"],
                "level": row["level"],
                "message": row["message"],
                "source": row.get("source", "unknown"),
                "host": row.get("host", "unknown"),
                "score": float(row.get("anomaly_score", 0)),
                "anomaly_score": float(row.get("anomaly_score", 0)),
                "cluster": cluster_id,
                "cluster_id": cluster_id,
                "isRootCause": is_root_cause,
                "is_root_cause": is_root_cause,
                "rootCause": root_message if is_root_cause else "",
                "suggestion": root_cause.get("suggestion", {}) if is_root_cause else {},
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
        if es is None:
            save_results_locally(df, np.array([]), np.array([]), {}, mode, tamper_detected, antiforensics_result)
            if LOCAL_LOG_FILE.exists():
                tamper_detection.save_hash(LOCAL_LOG_FILE, COLLECTED_LOGS_DIR / "system_logs.hash")
        return

    df = parse_and_normalize(df)
    X, _ = build_feature_matrix(df["clean_message"].tolist(), max_features=100)

    anomaly_labels, anomaly_scores, _ = detect_anomalies(X, contamination=0.2)
    df["anomaly"] = anomaly_labels
    df["anomaly_score"] = anomaly_scores
    anomaly_indices = np.where(anomaly_labels == 1)[0]

    cluster_labels = cluster_anomalies(X, anomaly_indices, eps=0.8, min_samples=2)
    root_cause = suggest_root_cause(df, anomaly_indices, cluster_labels)

    if es is not None:
        push_anomalies_to_es(es, df, anomaly_indices, cluster_labels, root_cause)
    else:
        save_results_locally(df, anomaly_indices, cluster_labels, root_cause, mode, tamper_detected, antiforensics_result)
        if LOCAL_LOG_FILE.exists():
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
