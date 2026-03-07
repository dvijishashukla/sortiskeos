

import re
import pandas as pd
import numpy as np
from elasticsearch import Elasticsearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_score, recall_score, f1_score


# --------------------------------------------------
# 1. Connect to Elasticsearch
# --------------------------------------------------

def fetch_logs(index_pattern="hdfs-logs", size=20000):
    es = Elasticsearch("http://localhost:9200")

    es.info()  # test connection

    response = es.search(
        index=index_pattern,
        query={"match_all": {}},
        size=size
    )

    logs = []
    for hit in response["hits"]["hits"]:
        logs.append(hit["_source"].get("log_message", hit["_source"].get("message", "")))

    return pd.DataFrame(logs, columns=["raw_log"])

# --------------------------------------------------
# 2. Normalize Log Messages
# --------------------------------------------------

def normalize_log(log):
    log = re.sub(r'blk_\d+', '<*>', log)
    log = re.sub(r'\d+\.\d+\.\d+\.\d+', '<*>', log)
    log = re.sub(r'\b\d+\b', '<*>', log)
    return log


def preprocess_logs(df):
    df["template"] = df["raw_log"].apply(normalize_log)
    return df


# --------------------------------------------------
# 3. Feature Engineering (TF-IDF)
# --------------------------------------------------

def vectorize_logs(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["template"])
    return X


# --------------------------------------------------
# 4. Anomaly Detection
# --------------------------------------------------

def detect_anomalies(X):
    model = IsolationForest(contamination=0.05, random_state=42)
    predictions = model.fit_predict(X)
    return predictions


# --------------------------------------------------
# 5. Clustering
# --------------------------------------------------

def cluster_logs(X):
    clustering = DBSCAN(eps=0.5, min_samples=5)
    clusters = clustering.fit_predict(X)
    return clusters


# --------------------------------------------------
# 6. Root Cause Ranking
# --------------------------------------------------

def rank_root_causes(df):
    cluster_counts = df.groupby("cluster")["anomaly"].apply(lambda x: (x == -1).sum())
    ranked = cluster_counts.sort_values(ascending=False)
    return ranked


# --------------------------------------------------
# 7. Evaluation
# --------------------------------------------------

def evaluate_model(df):
    try:
        labels = pd.read_csv("anomaly_label.csv")
        if "Label" in labels.columns:
            y_true = labels["Label"].map({"Anomaly": -1, "Normal": 1})
            y_pred = df["anomaly"]

            print("\nEvaluation Metrics:")
            print("Precision:", precision_score(y_true, y_pred, pos_label=-1))
            print("Recall:", recall_score(y_true, y_pred, pos_label=-1))
            print("F1 Score:", f1_score(y_true, y_pred, pos_label=-1))
    except Exception as e:
        print("Evaluation skipped:", e)


# --------------------------------------------------
# Main Execution
# --------------------------------------------------

def main():
    print("Fetching logs from Elasticsearch...")
    df = fetch_logs()

    print("Preprocessing logs...")
    df = preprocess_logs(df)

    print("Vectorizing logs...")
    X = vectorize_logs(df)

    print("Detecting anomalies...")
    df["anomaly"] = detect_anomalies(X)

    print("Clustering logs...")
    df["cluster"] = cluster_logs(X)

    print("Ranking root causes...")
    ranked = rank_root_causes(df)
    print("\nTop Root Cause Clusters:")
    print(ranked.head(5))

    evaluate_model(df)


if __name__ == "__main__":
    main()