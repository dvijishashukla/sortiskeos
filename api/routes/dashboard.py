from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Request

from data_access import get_local_anomalies, get_local_logs, get_local_summary, is_es_available

router = APIRouter(prefix='/dashboard', tags=['dashboard'])


def empty_stats() -> Dict[str, Any]:
    return {
        'totalCrashes': 0,
        'lastCrash': {'date': '', 'time': ''},
        'rootCause': '',
        'anomalyCount': 0,
    }


def split_timestamp(value: str | None) -> Dict[str, str]:
    if not value:
        return {'date': '', 'time': ''}

    normalized = value.replace('Z', '+00:00')
    try:
        dt = datetime.fromisoformat(normalized)
        return {
            'date': dt.date().isoformat(),
            'time': dt.time().replace(microsecond=0).isoformat(),
        }
    except ValueError:
        return {'date': '', 'time': ''}


def rootcause_empty_state(description: str, fix: str) -> Dict[str, Any]:
    return {
        'clusterId': 0,
        'label': 'No data',
        'confidence': 0,
        'anomalyCount': 0,
        'topScore': 0,
        'events': [],
        'description': description,
        'fix': fix,
    }


def infer_label(messages: List[str]) -> str:
    if any('disk' in message or 'i/o' in message for message in messages):
        return 'Disk I/O'
    if any('memory' in message or 'oom' in message or 'commit' in message for message in messages):
        return 'Memory Pressure'
    if any('network' in message or 'adapter' in message or 'rsc' in message for message in messages):
        return 'Kernel / Network'
    return 'Kernel / Network'


@router.get('/stats')
async def get_dashboard_stats(request: Request) -> Dict[str, Any]:
    es = getattr(request.app.state, 'es', None)
    if not await is_es_available(es):
        logs = get_local_logs()
        anomalies = get_local_anomalies()
        summary = get_local_summary()
        error_logs = [item for item in logs if str(item.get('level', '')).upper() == 'ERROR']
        last_crash = split_timestamp(error_logs[0].get('@timestamp') if error_logs else None)
        root_cause = summary.get('root_cause_message') or ''
        if not root_cause:
            root_event = next((item for item in anomalies if item.get('isRootCause') or item.get('is_root_cause')), None)
            root_cause = root_event.get('message', '') if root_event else ''
        return {
            'totalCrashes': len(error_logs),
            'lastCrash': last_crash,
            'rootCause': root_cause,
            'anomalyCount': len(anomalies),
        }

    try:
        crash_count_response = await es.count(
            index='logs-*',
            body={'query': {'term': {'level.keyword': 'ERROR'}}},
        )
        anomaly_count_response = await es.count(
            index='logs-*',
            body={'query': {'range': {'score': {'lt': -0.05}}}},
        )
        last_crash_response = await es.search(
            index='logs-*',
            body={
                'size': 1,
                'sort': [{'@timestamp': {'order': 'desc', 'unmapped_type': 'date'}}],
                'query': {'term': {'level.keyword': 'ERROR'}},
            },
        )
        root_cause_response = await es.search(
            index='logs-*',
            body={
                'size': 1,
                'sort': [{'@timestamp': {'order': 'desc', 'unmapped_type': 'date'}}],
                'query': {'term': {'isRootCause': True}},
            },
        )

        last_crash_hit = last_crash_response.get('hits', {}).get('hits', [])
        root_cause_hit = root_cause_response.get('hits', {}).get('hits', [])

        last_crash_source = last_crash_hit[0].get('_source', {}) if last_crash_hit else {}
        root_cause_source = root_cause_hit[0].get('_source', {}) if root_cause_hit else {}

        last_crash = split_timestamp(last_crash_source.get('@timestamp') or last_crash_source.get('time'))
        root_cause = root_cause_source.get('rootCause') or root_cause_source.get('message') or ''

        return {
            'totalCrashes': crash_count_response.get('count', 0),
            'lastCrash': last_crash,
            'rootCause': root_cause,
            'anomalyCount': anomaly_count_response.get('count', 0),
        }
    except Exception:
        return empty_stats()


@router.get('/timeline')
async def get_dashboard_timeline(request: Request) -> List[Dict[str, Any]]:
    es = getattr(request.app.state, 'es', None)
    if not await is_es_available(es):
        anomalies = get_local_anomalies()
        now = datetime.now(timezone.utc)
        timeline_map = {
            (now - timedelta(hours=offset)).replace(minute=0, second=0, microsecond=0): []
            for offset in range(24)
        }
        for item in anomalies:
            raw_time = item.get('@timestamp') or item.get('time')
            if not raw_time:
                continue
            try:
                event_time = datetime.fromisoformat(str(raw_time).replace('Z', '+00:00'))
            except ValueError:
                continue
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=timezone.utc)
            bucket = event_time.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
            if bucket in timeline_map:
                timeline_map[bucket].append(float(item.get('score', item.get('anomaly_score', 0)) or 0))

        timeline: List[Dict[str, Any]] = []
        for bucket in sorted(timeline_map.keys()):
            values = timeline_map[bucket]
            score = sum(values) / len(values) if values else 0
            timeline.append({'hour': bucket.isoformat(), 'score': score})
        return timeline

    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=24)
    body = {
        'size': 0,
        'query': {
            'bool': {
                'filter': [
                    {'range': {'@timestamp': {'gte': start.isoformat(), 'lte': now.isoformat()}}},
                    {'range': {'score': {'lt': -0.05}}},
                ]
            }
        },
        'aggs': {
            'scores_by_hour': {
                'date_histogram': {
                    'field': '@timestamp',
                    'calendar_interval': 'hour',
                    'min_doc_count': 0,
                    'extended_bounds': {
                        'min': start.isoformat(),
                        'max': now.isoformat(),
                    },
                },
                'aggs': {
                    'avg_score': {'avg': {'field': 'score'}},
                },
            }
        },
    }

    try:
        response = await es.search(index='logs-*', body=body)
        buckets = response.get('aggregations', {}).get('scores_by_hour', {}).get('buckets', [])
        timeline: List[Dict[str, Any]] = []
        for bucket in buckets:
            value = bucket.get('avg_score', {}).get('value')
            hour = bucket.get('key_as_string', '')
            timeline.append({'hour': hour, 'score': 0 if value is None else value})
        return timeline
    except Exception:
        return []


@router.get('/crashes')
async def get_dashboard_crashes(request: Request) -> List[Dict[str, Any]]:
    es = getattr(request.app.state, 'es', None)
    if not await is_es_available(es):
        logs = get_local_logs()
        anomalies = get_local_anomalies()
        summary = get_local_summary()
        crash_logs = [item for item in logs if str(item.get('level', '')).upper() == 'ERROR'][:50]
        anomaly_count = len(anomalies)
        root_message = summary.get('root_cause_message') or ''
        score = float(summary.get('top_score', 0) or 0)
        crashes: List[Dict[str, Any]] = []
        for item in crash_logs:
            when = split_timestamp(item.get('@timestamp') or item.get('time'))
            crashes.append(
                {
                    'date': when['date'],
                    'time': when['time'],
                    'rootCause': root_message or item.get('message') or '',
                    'anomalies': anomaly_count,
                    'score': score,
                }
            )
        return crashes

    body = {
        'size': 50,
        'sort': [{'@timestamp': {'order': 'desc', 'unmapped_type': 'date'}}],
        'query': {'term': {'level.keyword': 'ERROR'}},
    }

    try:
        response = await es.search(index='logs-*', body=body)
        hits = response.get('hits', {}).get('hits', [])
        crashes: List[Dict[str, Any]] = []
        for hit in hits:
            source = hit.get('_source', {})
            when = split_timestamp(source.get('@timestamp') or source.get('time'))
            crashes.append(
                {
                    'date': when['date'],
                    'time': when['time'],
                    'rootCause': source.get('rootCause') or source.get('message') or '',
                    'anomalies': source.get('anomalies', 0),
                    'score': source.get('score', 0),
                }
            )
        return crashes
    except Exception:
        return []


@router.get('/rootcause')
async def get_dashboard_rootcause(request: Request) -> Dict[str, Any]:
    """
    Get the root cause analysis: the highest-scoring anomaly cluster.
    Returns the cluster with the most critical anomalies.
    If ES is unreachable, returns a graceful empty structure.
    """
    es = getattr(request.app.state, 'es', None)
    if not await is_es_available(es):
        anomalies = get_local_anomalies()
        summary = get_local_summary()
        if not anomalies:
            return rootcause_empty_state(
                'Local Mode is active, but no anomaly results have been generated yet.',
                'Run the ML pipeline once to create ml_results.json from the collected logs.',
            )

        cluster_id = int(summary.get('root_cause_cluster', 0) or 0)
        cluster_events = [
            item for item in anomalies
            if int(item.get('cluster', item.get('cluster_id', 0)) or 0) == cluster_id
        ]
        if not cluster_events:
            cluster_events = anomalies[:10]

        top_score = min(float(item.get('score', item.get('anomaly_score', 0)) or 0) for item in cluster_events)
        events = []
        for index, event in enumerate(cluster_events[:10]):
            events.append({
                'time': event.get('@timestamp') or event.get('time') or '',
                'eventId': str(index + 1),
                'source': 'Root cause' if event.get('isRootCause') or event.get('is_root_cause') else 'Backend log',
                'message': event.get('message') or event.get('log') or '',
                'score': float(event.get('score', event.get('anomaly_score', 0)) or 0),
            })

        anomaly_count = int(summary.get('anomaly_count', len(cluster_events)) or len(cluster_events))
        confidence = min(99, int(30 + len(cluster_events) * 5 + abs(top_score) * 100))
        messages = [str(item.get('message', '')).lower() for item in cluster_events]
        description = summary.get('description') or (
            f'Local Mode identified {len(cluster_events)} anomalies in cluster {cluster_id}.'
        )
        fix = summary.get('fix') or 'Review the top anomalies in this cluster and correlate them with the local system logs.'
        return {
            'clusterId': cluster_id,
            'label': summary.get('label') or infer_label(messages),
            'confidence': confidence,
            'anomalyCount': anomaly_count,
            'topScore': top_score,
            'events': events,
            'description': description,
            'fix': fix,
        }

    try:
        # Fetch anomalies with scores < -0.05 (anomalous events)
        body = {
            'size': 100,
            'sort': [{'score': {'order': 'asc'}}],
            'query': {'range': {'score': {'lt': -0.05}}},
        }

        response = await es.search(index='logs-*', body=body)
        hits = response.get('hits', {}).get('hits', [])

        if not hits:
            return rootcause_empty_state(
                'No anomalies found in the current data',
                'The system is operating normally',
            )

        # Group events by cluster
        clusters: Dict[int, List[Dict[str, Any]]] = {}
        for hit in hits:
            source = hit.get('_source', {})
            cluster_id = int(source.get('cluster', 0)) or 0
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(source)

        # Find cluster with lowest average score (most anomalous)
        best_cluster_id = None
        best_cluster_data = None
        best_avg_score = 0

        for cluster_id, events in clusters.items():
            avg_score = sum(e.get('score', 0) for e in events) / len(events)
            if best_cluster_id is None or avg_score < best_avg_score:
                best_cluster_id = cluster_id
                best_cluster_data = events
                best_avg_score = avg_score

        if best_cluster_id is None or not best_cluster_data:
            return rootcause_empty_state(
                'Could not identify root cause cluster',
                'More anomaly data is needed for analysis',
            )

        # Build event list
        top_score = min(e.get('score', 0) for e in best_cluster_data)
        events = []
        for i, event in enumerate(best_cluster_data[:10]):  # Top 10 events from cluster
            events.append({
                'time': event.get('@timestamp') or event.get('time') or '',
                'eventId': str(i + 1),
                'source': 'Root cause' if event.get('isRootCause') else 'Backend log',
                'message': event.get('message') or event.get('log') or '',
                'score': float(event.get('score', 0)),
            })

        # Calculate confidence based on anomaly count and score
        anomaly_count = len(best_cluster_data)
        confidence = min(99, int(30 + anomaly_count * 5 + abs(top_score) * 100))

        # Determine label based on common anomaly patterns
        messages = [e.get('message', '').lower() for e in best_cluster_data]
        label = infer_label(messages)

        description = f'{anomaly_count} anomalies detected in cluster {best_cluster_id} with average score {best_avg_score:.3f}'
        fix = f'Review the top anomalies in this cluster and correlate them with system logs to determine the underlying cause.'

        return {
            'clusterId': best_cluster_id,
            'label': label,
            'confidence': confidence,
            'anomalyCount': anomaly_count,
            'topScore': top_score,
            'events': events,
            'description': description,
            'fix': fix,
        }

    except Exception:
        return rootcause_empty_state(
            'An error occurred while querying the current data source',
            'Please check the backend logs and try again',
        )
