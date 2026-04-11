from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Request

from data_access import (
    ANOMALIES_INDEX,
    SYSTEM_LOGS_INDEX,
    get_local_anomalies,
    get_local_logs,
    get_local_summary,
    is_es_available,
)

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


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None

    normalized = str(value).replace('Z', '+00:00')
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def floor_to_bucket(value: datetime, minutes: int) -> datetime:
    bucket_minute = (value.minute // minutes) * minutes
    return value.replace(minute=bucket_minute, second=0, microsecond=0)


def pick_matching_anomaly(
    crash_source: Dict[str, Any],
    anomalies: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    crash_time = parse_timestamp(crash_source.get('@timestamp') or crash_source.get('time'))
    if crash_time is None:
        return None

    crash_message = str(crash_source.get('message') or '')
    best_match: Dict[str, Any] | None = None
    best_distance: float | None = None

    for anomaly in anomalies:
        anomaly_time = parse_timestamp(anomaly.get('@timestamp') or anomaly.get('time'))
        if anomaly_time is None:
            continue

        distance = abs((anomaly_time - crash_time).total_seconds())
        if distance > 120:
            continue

        anomaly_message = str(anomaly.get('message') or '')
        message_matches = (
            crash_message and anomaly_message and (
                crash_message == anomaly_message
                or crash_message in anomaly_message
                or anomaly_message in crash_message
            )
        )

        if not message_matches and distance > 5:
            continue

        if best_match is None or best_distance is None or distance < best_distance:
            best_match = anomaly
            best_distance = distance

    return best_match


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
            'tamper_detected': summary.get('tamper_detected', False),
            'antiforensics': summary.get('antiforensics', {'detected': False, 'count': 0, 'events': []}),
        }

    try:
        crash_count_response = await es.count(
            index=SYSTEM_LOGS_INDEX,
            body={'query': {'term': {'level.keyword': 'ERROR'}}},
        )
        anomaly_count_response = await es.count(
            index=ANOMALIES_INDEX,
            body={'query': {'match_all': {}}},
        )
        last_crash_response = await es.search(
            index=SYSTEM_LOGS_INDEX,
            body={
                'size': 1,
                'sort': [{'@timestamp': {'order': 'desc', 'unmapped_type': 'date'}}],
                'query': {'term': {'level.keyword': 'ERROR'}},
            },
        )
        root_cause_response = await es.search(
            index=ANOMALIES_INDEX,
            body={
                'size': 1,
                'sort': [{'@timestamp': {'order': 'desc', 'unmapped_type': 'date'}}],
                'query': {'term': {'is_root_cause': True}},
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
            'tamper_detected': False,
            'antiforensics': {'detected': False, 'count': 0, 'events': []},
        }
    except Exception:
        return empty_stats()


@router.get('/timeline')
async def get_dashboard_timeline(request: Request) -> List[Dict[str, Any]]:
    es = getattr(request.app.state, 'es', None)
    window_minutes = 60
    bucket_minutes = 5

    if not await is_es_available(es):
        logs = get_local_logs()
        anomalies = get_local_anomalies()
        crash_logs = [
            item for item in logs
            if str(item.get('level', '')).upper() == 'ERROR'
        ]
        crash_time = parse_timestamp(
            (crash_logs[0] if crash_logs else {}).get('@timestamp')
            or (crash_logs[0] if crash_logs else {}).get('time')
        )
        if crash_time is None:
            crash_time = datetime.now(timezone.utc)

        start = crash_time - timedelta(minutes=window_minutes)
        end = crash_time + timedelta(minutes=window_minutes)
        timeline_map = {
            start + timedelta(minutes=offset * bucket_minutes): []
            for offset in range(int(((end - start).total_seconds() // 60) / bucket_minutes) + 1)
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
            event_time = event_time.astimezone(timezone.utc)
            if event_time < start or event_time > end:
                continue
            bucket = floor_to_bucket(event_time, bucket_minutes)
            if bucket in timeline_map:
                timeline_map[bucket].append(float(item.get('score', item.get('anomaly_score', 0)) or 0))

        timeline: List[Dict[str, Any]] = []
        for bucket in sorted(timeline_map.keys()):
            values = timeline_map[bucket]
            score = min(values) if values else 0
            timeline.append({'hour': bucket.isoformat(), 'score': score})
        return timeline

    crash_response = await es.search(
        index=SYSTEM_LOGS_INDEX,
        body={
            'size': 1,
            'sort': [{'@timestamp': {'order': 'desc', 'unmapped_type': 'date'}}],
            'query': {'term': {'level.keyword': 'ERROR'}},
        },
    )
    crash_hits = crash_response.get('hits', {}).get('hits', [])
    crash_source = (crash_hits[0] if crash_hits else {}).get('_source', {})
    crash_time = parse_timestamp(crash_source.get('@timestamp') or crash_source.get('time'))
    if crash_time is None:
        crash_time = datetime.now(timezone.utc)

    start = crash_time - timedelta(minutes=window_minutes)
    end = crash_time + timedelta(minutes=window_minutes)
    body = {
        'size': 0,
        'query': {
            'bool': {
                'filter': [
                    {'range': {'@timestamp': {'gte': start.isoformat(), 'lte': end.isoformat()}}},
                    {'match_all': {}},
                ]
            }
        },
        'aggs': {
            'scores_by_hour': {
                'date_histogram': {
                    'field': '@timestamp',
                    'fixed_interval': f'{bucket_minutes}m',
                    'min_doc_count': 0,
                    'extended_bounds': {
                        'min': start.isoformat(),
                        'max': end.isoformat(),
                    },
                },
                'aggs': {
                    'worst_score': {'min': {'field': 'anomaly_score'}},
                },
            }
        },
    }

    try:
        response = await es.search(index=ANOMALIES_INDEX, body=body)
        buckets = response.get('aggregations', {}).get('scores_by_hour', {}).get('buckets', [])
        timeline: List[Dict[str, Any]] = []
        for bucket in buckets:
            value = bucket.get('worst_score', {}).get('value')
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
        response = await es.search(index=SYSTEM_LOGS_INDEX, body=body)
        hits = response.get('hits', {}).get('hits', [])
        anomaly_response = await es.search(
            index=ANOMALIES_INDEX,
            body={
                'size': 500,
                'sort': [{'@timestamp': {'order': 'desc', 'unmapped_type': 'date'}}],
                'query': {'match_all': {}},
            },
        )
        anomaly_hits = [item.get('_source', {}) for item in anomaly_response.get('hits', {}).get('hits', [])]
        crashes: List[Dict[str, Any]] = []
        for hit in hits:
            source = hit.get('_source', {})
            matching_anomaly = pick_matching_anomaly(source, anomaly_hits)
            when = split_timestamp(source.get('@timestamp') or source.get('time'))
            crashes.append(
                {
                    'date': when['date'],
                    'time': when['time'],
                    'rootCause': (
                        (matching_anomaly or {}).get('rootCause')
                        or (matching_anomaly or {}).get('message')
                        or source.get('rootCause')
                        or source.get('message')
                        or ''
                    ),
                    'anomalies': 1 if matching_anomaly else source.get('anomalies', 0),
                    'score': (
                        (matching_anomaly or {}).get('anomaly_score')
                        if matching_anomaly is not None
                        else source.get('anomaly_score', source.get('score', 0))
                    ),
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
            'suggestion': summary.get('suggestion', {}),
        }

    try:
        # Fetch anomalies with scores < -0.05 (anomalous events)
        body = {
            'size': 100,
            'sort': [{'anomaly_score': {'order': 'asc'}}],
            'query': {'match_all': {}},
        }

        response = await es.search(index=ANOMALIES_INDEX, body=body)
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
            cluster_id = int(source.get('cluster_id', source.get('cluster', 0)) or 0)
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(source)

        # Find cluster with lowest average score (most anomalous)
        best_cluster_id = None
        best_cluster_data = None
        best_avg_score = 0

        for cluster_id, events in clusters.items():
            avg_score = sum(float(e.get('anomaly_score', e.get('score', 0)) or 0) for e in events) / len(events)
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
        top_score = min(float(e.get('anomaly_score', e.get('score', 0)) or 0) for e in best_cluster_data)
        events = []
        for i, event in enumerate(best_cluster_data[:10]):  # Top 10 events from cluster
            events.append({
                'time': event.get('@timestamp') or event.get('time') or '',
                'eventId': str(i + 1),
                'source': 'Root cause' if event.get('is_root_cause', event.get('isRootCause')) else 'Backend log',
                'message': event.get('message') or event.get('log') or '',
                'score': float(event.get('anomaly_score', event.get('score', 0)) or 0),
            })

        # Calculate confidence based on anomaly count and score
        anomaly_count = len(best_cluster_data)
        confidence = min(99, int(30 + anomaly_count * 5 + abs(top_score) * 100))

        # Determine label based on common anomaly patterns
        messages = [e.get('message', '').lower() for e in best_cluster_data]
        label = infer_label(messages)

        description = f'{anomaly_count} anomalies detected in cluster {best_cluster_id} with average score {best_avg_score:.3f}'
        fix = f'Review the top anomalies in this cluster and correlate them with system logs to determine the underlying cause.'

        suggestion = {}
        for e in best_cluster_data:
            if e.get('is_root_cause', e.get('isRootCause')):
                suggestion = e.get('suggestion', {})
                break

        return {
            'clusterId': best_cluster_id,
            'label': label,
            'confidence': confidence,
            'anomalyCount': anomaly_count,
            'topScore': top_score,
            'events': events,
            'description': description,
            'fix': fix,
            'suggestion': suggestion,
        }

    except Exception:
        return rootcause_empty_state(
            'An error occurred while querying the current data source',
            'Please check the backend logs and try again',
        )
