"""
sortiskeos_core.py
------------------
Advanced continuous Background Agent specifically designed to match exactly with
the Elasticsearch -> Logstash presentation architecture.

1. Thread 1 (Collector): Indiscriminately tails Windows logs + System Health (CPU/RAM),
   and dumps them efficiently to `collected_logs/system_logs.json`.
   Logstash is strictly responsible for ingesting this into ES.
2. Thread 2 (ML Engine): Polls Elasticsearch every 5 minutes for new logs natively
   ingested by Logstash. It builds a localized TF-IDF matrix, isolates anomalies,
   and assigns scores to `log-anomalies` ONLY if sufficient new logs are present.
"""

import os
import sys
import json
import time
import logging
import threading
import re
import hashlib
from datetime import datetime, timedelta, timezone
import collections
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

try:
    import win32evtlog
    import win32evtlogutil
except ImportError:
    print("FATAL: pywin32 must be installed (pip install pywin32)")
    sys.exit(1)

try:
    import psutil
except ImportError:
    print("FATAL: psutil must be installed for hardware tracking")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(threadName)s] %(message)s")
log = logging.getLogger("Core")
logging.getLogger("elastic_transport").setLevel(logging.ERROR)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "collected_logs")
STAGING_FILE = os.path.join(OUTPUT_DIR, "system_logs.json")
BOOKMARK_FILE = os.path.join(BASE_DIR, "bookmark_collector.json")
ROOT_ENV_PATH = os.path.join(os.path.dirname(BASE_DIR), ".env")
ML_ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ROOT_ENV_PATH)
load_dotenv(ML_ENV_PATH, override=True)

ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", "") or os.getenv("ELASTIC_PASSWORD", "")

CHANNELS = ["System", "Application", "Security"]
BATCH_SIZE = 5000
POLL_ES_INTERVAL = 300 # 5 minutes (CPU optimization)
POLL_WIN_INTERVAL = 120 # 2 minutes (CPU optimization)
MIN_NEW_LOGS_FOR_ML = 50 # Smart-trigger threshold

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Regex Patterns for Log Normalization
IP_PATTERN = re.compile(r"\b\d{1,3}(\.\d{1,3}){3}\b")
HEX_ID_PATTERN = re.compile(r"\b[a-f0-9]{8,}\b")
NUMBER_PATTERN = re.compile(r"\b\d+\b")
PATH_PATTERN = re.compile(r"/[\w/\-\.]+")
GUID_PATTERN = re.compile(r"\{?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\}?", re.IGNORECASE)

# Load Noise Config
def load_noise_config():
    config_path = os.path.join(BASE_DIR, "noise_config.json")
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        return set(cfg.get("suppressed_event_ids", [])), set(cfg.get("critical_event_ids", []))
    except Exception:
        return set(), {41, 6008, 1001, 1000, 1002, 7034, 7031, 55, 29}

SUPPRESSED_IDS, CRITICAL_IDS = load_noise_config()
WARN_IDS = {6006, 1014, 10010, 10016}
LEVEL_PRIORITY = {
    "CRITICAL": 4,
    "ERROR": 3,
    "WARN": 2,
    "WARNING": 2,
    "INFO": 1,
}

# ---------------------------------------------------------
# THREAD 1: LOG COLLECTION TO JSON
# ---------------------------------------------------------
def map_win_level(event_type: int, event_id: int) -> str:
    if event_id in CRITICAL_IDS: return "ERROR"
    if event_id in WARN_IDS: return "WARN"
    return {1: "ERROR", 2: "WARN", 4: "INFO", 8: "DEBUG", 16: "CRITICAL"}.get(event_type, "INFO")

def extract_message(event, channel: str) -> str:
    try:
        msg = win32evtlogutil.SafeFormatMessage(event, channel)
        if msg and msg.strip(): return msg.strip().replace("\n", " ").replace("\r", "")
    except Exception: pass
    if event.StringInserts: return " | ".join(str(s) for s in event.StringInserts)
    return f"Windows Event ID {event.EventID}"

def collector_loop():
    log.info(f"Collector started. Polling Windows Event Logs every {POLL_WIN_INTERVAL}s...")
    
    # Load bookmarks
    bookmarks = {}
    if os.path.exists(BOOKMARK_FILE):
        try:
            with open(BOOKMARK_FILE, "r") as f:
                bookmarks = json.load(f)
        except Exception: pass
        
    for ch in CHANNELS:
        if ch not in bookmarks:
            # Bug Fix: Use UTC for initial lookback to match win32evtlog.TimeGenerated
            bookmarks[ch] = (datetime.utcnow() - timedelta(hours=6)).isoformat()
        else:
            # Sanity check: If bookmark is in the future (common after IST -> UTC migration), reset it
            try:
                bm_time = datetime.fromisoformat(bookmarks[ch]).replace(tzinfo=None)
                if bm_time > datetime.utcnow() + timedelta(minutes=5):
                    log.warning(f"Future bookmark detected for {ch} ({bookmarks[ch]}). Resetting to 6hr lookback.")
                    bookmarks[ch] = (datetime.utcnow() - timedelta(hours=6)).isoformat()
            except Exception:
                bookmarks[ch] = (datetime.utcnow() - timedelta(hours=6)).isoformat()

    flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
    computer = os.environ.get("COMPUTERNAME", "unknown")

    while True:
        records = []
        for channel in CHANNELS:
            last_time = datetime.fromisoformat(bookmarks[channel]).replace(tzinfo=None)
            highest_time = last_time
            handle = None
            try:
                handle = win32evtlog.OpenEventLog(None, channel)
                stop = False
                log.info(f"Sweeping channel {channel} (lookback start: {last_time})...")
                while not stop:
                    events = win32evtlog.ReadEventLog(handle, flags, 0)
                    if not events: break
                    
                    for event in events:
                        # event.TimeGenerated is aware IST
                        evt_utc = event.TimeGenerated.astimezone(timezone.utc)
                        evt_time_naive = evt_utc.replace(tzinfo=None)
                        
                        if evt_time_naive <= last_time:
                            stop = True
                            break
                        if evt_time_naive > highest_time:
                            highest_time = evt_time_naive
                            
                        event_id = event.EventID & 0xFFFF
                        if event_id in SUPPRESSED_IDS:
                            continue
                        
                        # event.TimeGenerated is a datetime-like object from pywin32
                        # Convert to proper UTC ISO format
                        utc_ts = event.TimeGenerated.astimezone(timezone.utc)
                        
                        records.append({
                            "@timestamp": utc_ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                            "timestamp_ist": utc_ts.astimezone(None).isoformat(),
                            "level": map_win_level(event.EventType, event_id),
                            "source": f"windows/{channel}",
                            "event_id": event_id,
                            "host": computer,
                            "message": extract_message(event, channel)
                        })
                if highest_time > last_time:
                    bookmarks[channel] = highest_time.isoformat()
            except Exception as e:
                log.error(f"Collector failed on channel {channel}: {e}")
            finally:
                if handle:
                    try: win32evtlog.CloseEventLog(handle)
                    except: pass

        # Feature Addition: Hardware Health Tracking
        try:
            cpu = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory().percent
            health_level = "WARN" if (cpu > 85 or ram > 90) else "INFO"
            records.append({
                "@timestamp": datetime.utcnow().isoformat() + "Z",
                "level": health_level,
                "source": "sortiskeos/SystemHealth",
                "event_id": 9999,
                "host": computer,
                "message": f"[SYSTEM HEALTH] CPU Usage: {cpu}%, RAM Usage: {ram}%"
            })
        except Exception as e:
            log.error(f"Hardware profiling failed: {e}")

        if records:
            records.sort(key=lambda x: x["@timestamp"])
            with open(STAGING_FILE, "a", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            with open(BOOKMARK_FILE, "w") as f:
                json.dump(bookmarks, f)
            log.info(f"Wrote {len(records)} collected events (including CPU/RAM) to stash.")

        time.sleep(POLL_WIN_INTERVAL)


# ---------------------------------------------------------
# THREAD 2: MACHINE LEARNING & ES UPLOAD
# ---------------------------------------------------------

def normalize_msg(msg: str) -> str:
    msg = GUID_PATTERN.sub(" ", msg)
    msg = IP_PATTERN.sub(" ", msg)
    msg = PATH_PATTERN.sub(" ", msg)
    msg = HEX_ID_PATTERN.sub(" ", msg)
    msg = NUMBER_PATTERN.sub(" ", msg)
    msg = msg.lower()
    return re.sub(r"\s+", " ", msg).strip() or "empty"


def level_priority(level: str) -> int:
    return LEVEL_PRIORITY.get(str(level or "INFO").upper(), 1)


def root_cause_min_priority(df: pd.DataFrame) -> int:
    if df.empty:
        return 1
    priorities = df["level"].apply(level_priority)
    if (priorities >= LEVEL_PRIORITY["ERROR"]).any():
        return LEVEL_PRIORITY["ERROR"]
    if (priorities >= LEVEL_PRIORITY["WARN"]).any():
        return LEVEL_PRIORITY["WARN"]
    return LEVEL_PRIORITY["INFO"]


def deduplicate_anomalies(anomaly_df: pd.DataFrame) -> pd.DataFrame:
    if anomaly_df.empty:
        return anomaly_df.copy()

    working = anomaly_df.copy()
    working["level_priority"] = working["level"].apply(level_priority)
    working["severity_score"] = working["score"].apply(lambda value: abs(float(value or 0)))
    working["count"] = 1

    rows = []
    for _, group in working.groupby(["timestamp", "clean_message"], dropna=False, sort=False):
        ordered = group.sort_values(
            ["level_priority", "severity_score", "timestamp"],
            ascending=[False, False, False],
        )
        representative = ordered.iloc[0].copy()
        representative["count"] = int(len(group))
        rows.append(representative)

    return pd.DataFrame(rows).reset_index(drop=True)

def ml_pipeline_loop(es: Elasticsearch):
    log.info(f"ML Engine started. Smart-trigger threshold: {MIN_NEW_LOGS_FOR_ML} logs...")
    last_ml_timestamp = "1970-01-01T00:00:00Z"
    while True:
        try:
            if es.ping(): break
        except Exception: pass
        log.warning("ML Engine waiting for Elasticsearch...")
        time.sleep(10)

    while True:
        try:
            res = es.search(
                index="system-logs-*",
                body={
                    "size": BATCH_SIZE,
                    "sort": [{"@timestamp": {"order": "desc"}}],
                    "query": {"match_all": {}}
                }
            )
            hits = res.get("hits", {}).get("hits", [])
            if not hits:
                time.sleep(POLL_ES_INTERVAL)
                continue
                
            records = []
            new_log_count = 0
            latest_time_in_batch = last_ml_timestamp

            for hit in hits:
                s = hit["_source"]
                evt_time = s.get("@timestamp", "1970-01-01T00:00:00Z")
                
                if evt_time > last_ml_timestamp:
                    new_log_count += 1
                if evt_time > latest_time_in_batch:
                    latest_time_in_batch = evt_time
                    
                records.append({
                    "es_id": hit["_id"],
                    "timestamp": evt_time,
                    "level": str(s.get("level", "INFO")).upper(),
                    "message": s.get("message", "empty"),
                    "source": s.get("source", "unknown"),
                    "host": s.get("host", "unknown"),
                    "event_id": s.get("event_id") or s.get("EventID"),
                    "clean_message": normalize_msg(s.get("message", ""))
                })

            # SMART-TRIGGER RESOURCE OPTIMIZATION
            if new_log_count < MIN_NEW_LOGS_FOR_ML and last_ml_timestamp != "1970-01-01T00:00:00Z":
                log.info(f"Quiet network. Only {new_log_count} < {MIN_NEW_LOGS_FOR_ML} new logs. ML thread sleeping (CPU 0%)...")
                time.sleep(POLL_ES_INTERVAL)
                continue
                
            # Update watermark and proceed
            last_ml_timestamp = latest_time_in_batch
            df = pd.DataFrame(records)
            
            # Feature Extraction (n_jobs=1 by default avoids spiking all cores)
            vectorizer = TfidfVectorizer(max_features=100, stop_words="english", ngram_range=(1,2))
            X = vectorizer.fit_transform(df["clean_message"].tolist()).toarray()
            
            # Anomaly Detection
            model = IsolationForest(n_estimators=100, contamination=0.2, random_state=42, n_jobs=1)
            df["anomaly"] = np.where(model.fit_predict(X) == -1, 1, 0)
            df["score"] = model.decision_function(X)
            anomaly_indices = np.where(df["anomaly"] == 1)[0]
            
            if len(anomaly_indices) > 0:
                # Clustering
                X_scaled = StandardScaler().fit_transform(X[anomaly_indices])
                db = DBSCAN(eps=0.8, min_samples=2, n_jobs=1)
                cluster_labels = db.fit_predict(X_scaled)
                df["cluster"] = -1
                for idx, c_id in zip(anomaly_indices, cluster_labels):
                    df.at[idx, "cluster"] = c_id

                anomaly_df = deduplicate_anomalies(df.iloc[anomaly_indices].copy())
                min_priority = root_cause_min_priority(anomaly_df)
                anomaly_df["eligible_root_cause"] = anomaly_df["level"].apply(level_priority) >= min_priority
                anomaly_df = anomaly_df.sort_values(
                    ["eligible_root_cause", "level_priority", "severity_score", "timestamp"],
                    ascending=[False, False, False, False],
                ).reset_index(drop=True)
                anomaly_df["root_cause_rank"] = np.arange(1, len(anomaly_df) + 1)

                # Determine Root Cause cluster from deduplicated anomalies.
                root_cluster_id = -1
                valid_cluster_df = anomaly_df[anomaly_df["cluster"] >= 0].copy()
                if not valid_cluster_df.empty:
                    cluster_summaries = []
                    for cluster_id, group in valid_cluster_df.groupby("cluster"):
                        cluster_summaries.append({
                            "cluster_id": int(cluster_id),
                            "top_level_priority": int(group["level"].apply(level_priority).max() or 1),
                            "top_severity": float(group["severity_score"].max() or 0),
                            "cluster_size": int(len(group)),
                        })
                    eligible_clusters = [item for item in cluster_summaries if item["top_level_priority"] >= min_priority] or cluster_summaries
                    eligible_clusters.sort(
                        key=lambda item: (
                            item["top_level_priority"],
                            item["top_severity"],
                            item["cluster_size"],
                        ),
                        reverse=True,
                    )
                    root_cluster_id = eligible_clusters[0]["cluster_id"]

                # Push anomalies back to ES exactly how the UI expects
                actions = []
                for _, row in anomaly_df.iterrows():
                    c_id = int(row["cluster"])
                    is_root_cause = int(row.get("root_cause_rank", 0) or 0) == 1
                    stable_id = hashlib.sha1(
                        f"{row['timestamp']}|{row['clean_message']}".encode("utf-8")
                    ).hexdigest()
                    actions.append({
                        "_index": "log-anomalies",
                        "_id": stable_id,
                        "_source": {
                            "@timestamp": row["timestamp"],
                            "time": row["timestamp"],
                            "timestamp": row["timestamp"],
                            "level": row["level"],
                            "message": row["message"],
                            "source": row["source"],
                            "host": row["host"],
                            "score": float(row["score"]),
                            "anomaly_score": float(row["score"]),
                            "cluster_id": c_id,
                            "cluster": c_id,
                            "count": int(row.get("count", 1) or 1),
                            "root_cause_rank": int(row.get("root_cause_rank", 0) or 0),
                            "root_cause_score": float(row.get("severity_score", abs(float(row["score"])))),
                            "isRootCause": is_root_cause,
                            "is_root_cause": is_root_cause,
                            "rootCause": row["message"] if (c_id == root_cluster_id or is_root_cause) else "",
                            "method": "IsolationForest+DBSCAN",
                            "scoring_method": "IsolationForest",
                            "event_id": row.get("event_id"),
                            "noise_suppressed": False,
                            "analysed_at": datetime.now().isoformat()
                        }
                    })
                
                helpers.bulk(es, actions)
                log.info(f"ML Pipeline complete. Identified {len(actions)} anomalies via TF-IDF + DBSCAN.")
            else:
                log.info(f"ML Pipeline complete. No extreme anomalies detected out of {len(df)} records.")

        except Exception as e:
            log.error(f"ML Pipeline error: {e}")

        time.sleep(POLL_ES_INTERVAL)

# ---------------------------------------------------------
# MAIN ORCHESTRATOR
# ---------------------------------------------------------
def main():
    log.info("Starting Sortiskeos Advanced Agent...")
    es_kwargs = {
        "hosts": ES_HOST,
        "max_retries": 2,
        "request_timeout": 5,
    }
    if ES_PASSWORD:
        es_kwargs["basic_auth"] = (ES_USER, ES_PASSWORD)
    es = Elasticsearch(**es_kwargs)
    
    t1 = threading.Thread(target=collector_loop, name="Collector", daemon=True)
    t2 = threading.Thread(target=ml_pipeline_loop, args=(es,), name="MLPipeline", daemon=True)
    
    t1.start()
    t2.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Shutting down Sortiskeos Agent...")


if __name__ == "__main__":
    main()
