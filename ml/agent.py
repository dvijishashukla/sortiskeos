"""
agent.py
--------
Sortiskeos Unified Edge Agent.
Continuously fetches Windows Event Logs, maintains a rolling buffer, 
performs real-time TF-IDF & Isolation Forest anomaly detection, 
and pushes directly to Elasticsearch.
"""

import os
import sys
import time
import json
import logging
import re
from datetime import datetime, timedelta, timezone
import collections
import numpy as np
import pandas as pd
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("Agent")

# --- Configuration ---
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOOKMARK_FILE = os.path.join(BASE_DIR, "bookmark.json")
POLL_INTERVAL = 30  # seconds
MAX_BUFFER = 5000   # keep last N logs for training the ML model
CHANNELS = ["System", "Application", "Security"]

# Reduce library logging noise
logging.getLogger("elastic_transport").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Real-time ML buffer
event_buffer = collections.deque(maxlen=MAX_BUFFER)

IP_PATTERN = re.compile(r"\b\d{1,3}(\.\d{1,3}){3}\b")
HEX_ID_PATTERN = re.compile(r"\b[a-f0-9]{8,}\b")
GUID_PATTERN = re.compile(r"\{?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\}?", re.IGNORECASE)

# --- Mapping Windows Events to Severity ---
CRITICAL_IDS = {41, 6008, 1001, 7034, 7031, 55, 29}
WARN_IDS = {6006, 1014, 10010, 10016}

def map_level(event_type: int, event_id: int) -> str:
    if event_id in CRITICAL_IDS: return "ERROR"
    if event_id in WARN_IDS: return "WARN"
    return {1: "ERROR", 2: "WARN", 4: "INFO", 8: "DEBUG", 16: "CRITICAL"}.get(event_type, "INFO")


def extract_message(event, channel: str) -> str:
    try:
        msg = win32evtlogutil.SafeFormatMessage(event, channel)
        if msg and msg.strip():
            return msg.strip().replace("\n", " ").replace("\r", "")
    except Exception:
        pass
    if event.StringInserts:
        return " | ".join(str(s) for s in event.StringInserts)
    return f"Windows Event ID {event.EventID}"


def normalize_message(msg: str) -> str:
    msg = GUID_PATTERN.sub(" ", msg)
    msg = IP_PATTERN.sub(" ", msg)
    msg = HEX_ID_PATTERN.sub(" ", msg)
    msg = msg.lower()
    return re.sub(r"\s+", " ", msg).strip() or "empty"


def load_bookmarks():
    if os.path.exists(BOOKMARK_FILE):
        try:
            with open(BOOKMARK_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    # default to 1 hour ago if no bookmark exists
    hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
    return {ch: hour_ago for ch in CHANNELS}


def save_bookmarks(bm):
    with open(BOOKMARK_FILE, "w") as f:
        json.dump(bm, f)


def fetch_new_logs(bm):
    records = []
    handle = None
    flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
    computer = os.environ.get("COMPUTERNAME", "unknown")

    for channel in CHANNELS:
        last_time_str = bm.get(channel, (datetime.now() - timedelta(hours=1)).isoformat())
        last_time = datetime.fromisoformat(last_time_str).replace(tzinfo=None)
        highest_time = last_time
        
        try:
            handle = win32evtlog.OpenEventLog(None, channel)
            stop = False
            while not stop:
                events = win32evtlog.ReadEventLog(handle, flags, 0)
                if not events: break
                
                for event in events:
                    evt_time = event.TimeGenerated.replace(tzinfo=None)
                    if evt_time <= last_time:
                        stop = True
                        break
                        
                    if evt_time > highest_time:
                        highest_time = evt_time

                    event_id = event.EventID & 0xFFFF
                    msg = extract_message(event, channel)
                    
                    records.append({
                        "@timestamp": evt_time.isoformat() + "Z", # naive to UTC roughly
                        "level": map_level(event.EventType, event_id),
                        "source": f"windows/{channel}",
                        "event_id": event_id,
                        "host": computer,
                        "message": msg,
                        "clean_message": normalize_message(msg)
                    })
            if highest_time > last_time:
                bm[channel] = highest_time.isoformat()
                
        except Exception as e:
            log.error(f"Error reading {channel}: {e}")
        finally:
            if handle:
                try: win32evtlog.CloseEventLog(handle)
                except: pass

    # sort chronological
    records.sort(key=lambda x: x["@timestamp"])
    return records


def analyze_records(new_records):
    """
    Append new records to buffer, run TF-IDF and Isolation Forest, 
    and assign scores and clusters dynamically.
    """
    for r in new_records:
        event_buffer.append(r)

    # We need a decent minimum size to extract TF-IDF and run IF safely
    if len(event_buffer) < 20:
        for r in new_records:
            r["score"] = 0.0
            r["cluster"] = 0
            r["isRootCause"] = False
        return new_records

    # Re-build the dataframe from the buffer
    df = pd.DataFrame(list(event_buffer))
    
    # Feature extraction
    vectorizer = TfidfVectorizer(max_features=100, stop_words="english", ngram_range=(1,2))
    X = vectorizer.fit_transform(df["clean_message"].tolist()).toarray()
    
    # Anomaly Detection
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    labels = np.where(model.fit_predict(X) == -1, 1, 0)
    scores = model.decision_function(X)
    
    df["score"] = scores
    df["cluster"] = 0
    df["isRootCause"] = False

    # DBSCAN clustering on anomalies only
    anomaly_indices = np.where(labels == 1)[0]
    if len(anomaly_indices) > 0:
        X_anomalies = X[anomaly_indices]
        X_scaled = StandardScaler().fit_transform(X_anomalies)
        db = DBSCAN(eps=0.8, min_samples=2)
        cluster_labels = db.fit_predict(X_scaled)
        
        # Populate cluster ids
        for idx, c_id in zip(anomaly_indices, cluster_labels):
            df.at[idx, "cluster"] = c_id
            
        # Determine the most dense cluster = Root Cause
        valid_clusters = cluster_labels[cluster_labels != -1]
        if len(valid_clusters) > 0:
            counts = collections.Counter(valid_clusters)
            root_cluster = counts.most_common(1)[0][0]
            for idx, c_id in zip(anomaly_indices, cluster_labels):
                if c_id == root_cluster:
                    df.at[idx, "isRootCause"] = True

    # We only care about returning the updated dicts for the *new* records
    # since we are pushing them to ES.
    # The new_records correspond to the tail of the buffer.
    N = len(new_records)
    tail_df = df.tail(N)
    
    updated_new_records = tail_df.to_dict("records")
    return updated_new_records


def push_to_es(es: Elasticsearch, records: list):
    actions = []
    # e.g., system-logs-2026.03.29
    idx = f"system-logs-{datetime.now().strftime('%Y.%m.%d')}"
    for r in records:
        doc = r.copy()
        doc.pop("clean_message", None) # discard intermediate field
        actions.append({
            "_index": idx,
            "_source": doc
        })

    if actions:
        try:
            helpers.bulk(es, actions)
            log.info(f"Pushed {len(actions)} analyzed records to {idx}.")
        except Exception as e:
            log.error(f"Elasticsearch bulk index error: {e}")


def main():
    log.info("Starting Sortiskeos Edge Agent...")
    es = Elasticsearch(ES_HOST, max_retries=2, request_timeout=5)
    
    try:
        if not es.ping():
            log.warning(f"Elasticsearch not reachable at {ES_HOST}. Ensure ELK is running.")
    except Exception:
        pass

    bm = load_bookmarks()
    
    while True:
        try:
            new_logs = fetch_new_logs(bm)
            if new_logs:
                log.info(f"Fetched {len(new_logs)} new records. Processing AI pipeline...")
                analyzed_logs = analyze_records(new_logs)
                save_bookmarks(bm) # save progress
                push_to_es(es, analyzed_logs)
            else:
                log.debug(f"No new logs. Sleeping for {POLL_INTERVAL}s.")
                
        except KeyboardInterrupt:
            log.info("Agent stopped by user.")
            break
        except Exception as e:
            log.error(f"Agent loop error: {e}")
            
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
