# sortiskeos – Intelligent Log Analysis & Root Cause Detection System

## Overview

sortiskeos is an AI-driven log analysis system that automatically detects anomalies and identifies root causes from large-scale system logs.

It integrates:

- ELK Stack (Elasticsearch, Logstash, Kibana)
- Machine Learning (Isolation Forest, DBSCAN)
- Log Template Normalization
- Root Cause Ranking

The system uses the HDFS log dataset from LogHub.

---

## Architecture

Raw Logs → Logstash → Elasticsearch → Python ML → Anomaly Detection → Clustering → Root Cause Ranking → Kibana Dashboard

---

## Project Structure
"# sortiskeos" 
