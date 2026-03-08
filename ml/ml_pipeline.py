"""
ml_pipeline.py
--------------
Self-contained ML pipeline for Intelligent Log Analysis.
No external module imports needed — everything is in this single file.

Pipeline:
  1. Fetch logs from Elasticsearch
  2. Normalize log messages
  3. Build TF-IDF feature matrix
  4. Detect anomalies (Isolation Forest)
  5. Cluster anomalies (DBSCAN)
  6. Suggest root cause
  7. Push results back to Elasticsearch

Usage:
  python ml_pipeline.py --mode once        # run once on startup
  python ml_pipeline.py --mode realtime    # poll every 60 seconds
"""

import re
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
from elasticsearch import Elasticsearch, helpers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Elasticsearch config ───────────────────────────────────────────────────────
ES_HOST      = "http://localhost:9200"
SOURCE_INDEX = "system-logs-*"   # where Logstash writes raw logs
ANOMALY_INDEX = "log-anomalies"  # where we write ML results
BATCH_SIZE   = 1000
POLL_INTERVAL = 60               # seconds between runs in realtime mode


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 1 — Fetch logs from Elasticsearch
# ══════════════════════════════════════════════════════════════════════════════
def fetch_logs_from_es(es: Elasticsearch) -> pd.DataFrame:
    """
    Query Elasticsearch for the most recent logs.
    Returns a DataFrame with: timestamp, level, message, source, host
    """
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
            "es_id":     hit["_id"],
            "timestamp": src.get("@timestamp", datetime.now().isoformat()),
            "level":     str(src.get("level", "INFO")).upper(),
            "message":   src.get("message", ""),
            "source":    src.get("source", "unknown"),
            "host":      src.get("host", "unknown")
        })

    df = pd.DataFrame(records)
    log.info(f"Fetched {len(df)} logs from Elasticsearch.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 2 — Normalize log messages
# ══════════════════════════════════════════════════════════════════════════════

# Regex patterns for cleaning log messages
IP_PATTERN     = re.compile(r"\b\d{1,3}(\.\d{1,3}){3}\b")   # IPv4 addresses
NUMBER_PATTERN = re.compile(r"\b\d+\b")                       # standalone numbers
HEX_ID_PATTERN = re.compile(r"\b[a-f0-9]{8,}\b")             # hex IDs
PATH_PATTERN   = re.compile(r"/[\w/\-\.]+")                   # file/URL paths
GUID_PATTERN   = re.compile(                                   # GUIDs like {abc-123}
    r"\{?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\}?",
    re.IGNORECASE
)


def normalize_message(message: str) -> str:
    """
    Clean a log message by removing IPs, GUIDs, numbers, hex IDs.
    Returns a lowercase clean string for TF-IDF processing.
    """
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
    """Add a clean_message column to the DataFrame."""
    df = df.copy()
    df["clean_message"] = df["message"].apply(normalize_message)
    log.info("Log messages normalized.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 3 — Feature Engineering (TF-IDF)
# ══════════════════════════════════════════════════════════════════════════════
def build_feature_matrix(clean_messages: list, max_features: int = 100):
    """
    Convert cleaned log messages into a TF-IDF numerical feature matrix.
    Returns the dense matrix and the fitted vectorizer.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1
    )
    X = vectorizer.fit_transform(clean_messages).toarray()
    log.info(f"Feature matrix shape: {X.shape} "
             f"({X.shape[0]} logs x {X.shape[1]} TF-IDF features).")
    return X, vectorizer


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 4 — Anomaly Detection (Isolation Forest)
# ══════════════════════════════════════════════════════════════════════════════
def detect_anomalies(X: np.ndarray, contamination: float = 0.2):
    """
    Run Isolation Forest to detect anomalous log entries.
    Returns anomaly labels (1=anomaly, 0=normal) and anomaly scores.
    """
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    raw_predictions = model.fit_predict(X)
    labels = np.where(raw_predictions == -1, 1, 0)
    scores = model.decision_function(X)

    n_anomalies = labels.sum()
    log.info(f"Anomaly detection complete: {n_anomalies} anomalies found "
             f"out of {len(labels)} logs ({100*n_anomalies/len(labels):.1f}%).")
    return labels, scores, model


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 5 — Clustering (DBSCAN)
# ══════════════════════════════════════════════════════════════════════════════
def cluster_anomalies(X: np.ndarray, anomaly_indices: np.ndarray,
                      eps: float = 0.8, min_samples: int = 2):
    """
    Cluster anomalous logs using DBSCAN.
    Returns cluster labels for each anomalous log (-1 = noise).
    """
    if len(anomaly_indices) == 0:
        log.warning("No anomalies to cluster.")
        return np.array([])

    X_anomalies = X[anomaly_indices]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_anomalies)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
    cluster_labels = db.fit_predict(X_scaled)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise    = (cluster_labels == -1).sum()
    log.info(f"DBSCAN found {n_clusters} cluster(s) among anomalies "
             f"({n_noise} noise points).")
    return cluster_labels


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 6 — Root Cause Suggestion
# ══════════════════════════════════════════════════════════════════════════════
def suggest_root_cause(df: pd.DataFrame, anomaly_indices: np.ndarray,
                       cluster_labels: np.ndarray, n_samples: int = 5) -> dict:
    """
    Identify the most likely root cause cluster and print a report.
    The cluster with the most log entries = likely root cause.
    """
    if len(anomaly_indices) == 0:
        log.info("No anomalies detected — system appears healthy.")
        return {}

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
        print("[INFO] All anomalies are noise — no dominant cluster found.")
        _print_cluster_samples(
            anomaly_df[anomaly_df["cluster"] == -1],
            label="NOISE / UNCLUSTERED",
            n_samples=n_samples
        )
        return {"root_cause_cluster": None, "anomaly_count": len(anomaly_indices)}

    cluster_counts = Counter(valid_clusters)
    root_cluster_id, root_cluster_size = cluster_counts.most_common(1)[0]

    print(f"\n  ⚠  ROOT CAUSE CLUSTER  →  Cluster #{root_cluster_id}")
    print(f"     Log count in cluster  : {root_cluster_size}")

    for cluster_id, count in cluster_counts.most_common():
        cluster_logs = anomaly_df[anomaly_df["cluster"] == cluster_id]
        levels  = cluster_logs["level"].value_counts().to_dict()
        marker  = "  ★ ROOT CAUSE" if cluster_id == root_cluster_id else ""
        print(f"\n  --- Cluster #{cluster_id} ({count} logs){marker} ---")
        print(f"      Log levels: {levels}")
        _print_cluster_samples(cluster_logs, label=f"Cluster #{cluster_id}",
                               n_samples=n_samples)

    noise_logs = anomaly_df[anomaly_df["cluster"] == -1]
    if len(noise_logs) > 0:
        print(f"\n  --- Noise / Unclustered ({len(noise_logs)} logs) ---")
        _print_cluster_samples(noise_logs, label="Noise", n_samples=n_samples)

    print("\n" + "=" * 60)

    root_logs = anomaly_df[anomaly_df["cluster"] == root_cluster_id]
    return {
        "root_cause_cluster": int(root_cluster_id),
        "anomaly_count":      len(anomaly_indices),
        "cluster_size":       int(root_cluster_size),
        "sample_messages":    root_logs["message"].head(n_samples).tolist(),
        "log_levels":         root_logs["level"].value_counts().to_dict()
    }


def _print_cluster_samples(cluster_df: pd.DataFrame, label: str, n_samples: int):
    """Print sample log messages from a cluster."""
    samples = cluster_df[["timestamp", "level", "message"]].head(n_samples)
    print(f"\n      Sample logs from {label}:")
    for _, row in samples.iterrows():
        print(f"        [{row['timestamp']}] {row['level']:7s}  {row['message']}")


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 7 — Push anomaly results back to Elasticsearch
# ══════════════════════════════════════════════════════════════════════════════
def push_anomalies_to_es(es: Elasticsearch, df: pd.DataFrame,
                          anomaly_indices: np.ndarray,
                          cluster_labels: np.ndarray):
    """
    Push anomalous log entries with scores and cluster IDs back to
    Elasticsearch so Kibana can visualize them.
    """
    if len(anomaly_indices) == 0:
        log.info("No anomalies to push.")
        return

    anomaly_df = df.iloc[anomaly_indices].copy()
    anomaly_df["cluster"] = cluster_labels

    actions = []
    for _, row in anomaly_df.iterrows():
        actions.append({
            "_index": ANOMALY_INDEX,
            "_source": {
                "@timestamp":    row["timestamp"],
                "level":         row["level"],
                "message":       row["message"],
                "source":        row.get("source", "unknown"),
                "host":          row.get("host", "unknown"),
                "anomaly_score": float(row.get("anomaly_score", 0)),
                "cluster_id":    int(row["cluster"]),
                "is_root_cause": bool(row["cluster"] == 0),
                "analysed_at":   datetime.now().isoformat()
            }
        })

    helpers.bulk(es, actions)
    log.info(f"Pushed {len(actions)} anomalies to index '{ANOMALY_INDEX}'.")


# ══════════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE — fetch → analyze → push results
# ══════════════════════════════════════════════════════════════════════════════
def run_analysis(es: Elasticsearch):
    """Execute one full cycle of the ML analysis pipeline."""
    log.info("─" * 50)
    log.info("Starting ML analysis cycle...")

    # 1. Fetch logs from Elasticsearch
    df = fetch_logs_from_es(es)
    if df.empty:
        log.warning("No logs to analyze.")
        return

    # 2. Normalize messages
    df = parse_and_normalize(df)

    # 3. Build TF-IDF feature matrix
    X, _ = build_feature_matrix(df["clean_message"].tolist(), max_features=100)

    # 4. Anomaly detection
    anomaly_labels, anomaly_scores, _ = detect_anomalies(X, contamination=0.2)
    df["anomaly"]       = anomaly_labels
    df["anomaly_score"] = anomaly_scores
    anomaly_indices     = np.where(anomaly_labels == 1)[0]

    # 5. Cluster anomalies
    cluster_labels = cluster_anomalies(X, anomaly_indices, eps=0.8, min_samples=2)

    # 6. Print root cause report
    suggest_root_cause(df, anomaly_indices, cluster_labels)

    # 7. Push anomalies back to Elasticsearch
    push_anomalies_to_es(es, df, anomaly_indices, cluster_labels)

    log.info("Analysis cycle complete.")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINTS
# ══════════════════════════════════════════════════════════════════════════════
def run_once():
    """Run the pipeline a single time — good for startup trigger."""
    es = Elasticsearch(ES_HOST)
    if not es.ping():
        log.error(f"Cannot connect to Elasticsearch at {ES_HOST}. Is it running?")
        return
    log.info(f"Connected to Elasticsearch at {ES_HOST}")
    run_analysis(es)


def run_realtime(interval: int = POLL_INTERVAL):
    """Run the pipeline in a loop — polls ES every interval seconds."""
    es = Elasticsearch(ES_HOST)
    if not es.ping():
        log.error(f"Cannot connect to Elasticsearch at {ES_HOST}.")
        return
    log.info(f"Real-time mode: analysing every {interval}s. Ctrl+C to stop.")

    while True:
        try:
            run_analysis(es)
            log.info(f"Sleeping {interval}s until next analysis...")
            time.sleep(interval)
        except KeyboardInterrupt:
            log.info("Stopped by user.")
            break
        except Exception as e:
            log.error(f"Pipeline error: {e}. Retrying in {interval}s...")
            time.sleep(interval)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="ML Pipeline for Intelligent Log Analysis"
    )
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