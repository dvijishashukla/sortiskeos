#!/usr/bin/env bash
# setup_es_security.sh
# Run ONCE after first `docker compose up` to create least-privilege roles and users.
# Usage: bash setup_es_security.sh
# Requires: .env file with ELASTIC_PASSWORD, LOGSTASH_PASSWORD, ML_PASSWORD set.

set -euo pipefail
source .env 2>/dev/null || { echo "ERROR: .env not found. Copy .env.example to .env first."; exit 1; }

ES="http://127.0.0.1:9200"
ADMIN="-u elastic:${ELASTIC_PASSWORD}"

echo "Waiting for Elasticsearch to be ready..."
until curl -sf $ADMIN "$ES/_cluster/health" | grep -qE '"status":"(green|yellow)"'; do
  echo "  still waiting..."; sleep 5
done
echo "Elasticsearch is up."

# ── 1. kibana_system password (built-in user) ────────────────────────────────
echo "Setting kibana_system password..."
curl -sf -X POST $ADMIN "$ES/_security/user/kibana_system/_password" \
  -H "Content-Type: application/json" \
  -d "{\"password\":\"${KIBANA_PASSWORD}\"}"
echo ""

# ── 2. logstash_writer role — can only write to system-logs-* ────────────────
echo "Creating logstash_writer role..."
curl -sf -X PUT $ADMIN "$ES/_security/role/logstash_writer" \
  -H "Content-Type: application/json" \
  -d '{
    "cluster": ["monitor"],
    "indices": [{
      "names": ["system-logs-*"],
      "privileges": ["create_index", "create", "index", "write"]
    }]
  }'
echo ""

echo "Creating logstash_writer user..."
curl -sf -X PUT $ADMIN "$ES/_security/user/logstash_writer" \
  -H "Content-Type: application/json" \
  -d "{
    \"password\": \"${LOGSTASH_PASSWORD}\",
    \"roles\": [\"logstash_writer\"],
    \"full_name\": \"Logstash Writer\"
  }"
echo ""

# ── 3. ml_reader_writer role — reads system-logs-*, writes log-anomalies ─────
echo "Creating ml_pipeline role..."
curl -sf -X PUT $ADMIN "$ES/_security/role/ml_pipeline" \
  -H "Content-Type: application/json" \
  -d '{
    "cluster": ["monitor"],
    "indices": [
      {
        "names": ["system-logs-*"],
        "privileges": ["read"]
      },
      {
        "names": ["log-anomalies"],
        "privileges": ["create_index", "create", "index", "write", "read"]
      }
    ]
  }'
echo ""

echo "Creating ml_pipeline user..."
curl -sf -X PUT $ADMIN "$ES/_security/user/ml_pipeline" \
  -H "Content-Type: application/json" \
  -d "{
    \"password\": \"${ML_PASSWORD}\",
    \"roles\": [\"ml_pipeline\"],
    \"full_name\": \"ML Pipeline Service\"
  }"
echo ""

echo "Security setup complete."
echo "  logstash_writer → writes system-logs-* only"
echo "  ml_pipeline     → reads system-logs-*, writes log-anomalies only"
echo "  elastic          → keep this password secret (admin only)"
