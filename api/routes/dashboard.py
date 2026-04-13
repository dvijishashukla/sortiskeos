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


def get_crash_window():
    # Try to find crash anchor from raw logs
    logs = get_local_logs()
    if not logs:
        return None, None, None
    
    crash_ids = {41, 6008, 1001, 6006, "41", "6008", "1001", "6006"}
    crash_keywords = [
        "unexpected shutdown",
        "previous shutdown was unexpected", 
        "kernel power",
        "bugcheck",
        "blue screen"
    ]
    
    # Find most recent crash marker log
    crash_log = None
    for log in logs:
        eid = str(log.get("event_id") or log.get("EventID") or "")
        msg = str(log.get("message") or "").lower()
        is_crash = (
            eid in crash_ids or
            any(kw in msg for kw in crash_keywords)
        )
        if is_crash:
            crash_log = log
            break
    
    # Fallback to most recent log if no crash found
    if not crash_log:
        crash_log = logs[0] if logs else None
    if not crash_log:
        return None, None, None
    
    crash_time = (crash_log.get("@timestamp") or crash_log.get("time"))
    if not crash_time:
        return None, None, None
    
    ct = datetime.fromisoformat(crash_time.replace("Z", "+00:00"))
    if ct.tzinfo is None:
        ct = ct.replace(tzinfo=timezone.utc)
    
    window_start = ct - timedelta(hours=3)
    window_end = ct + timedelta(minutes=30)
    return crash_time, window_start, window_end

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


def determine_event_type(source: Dict[str, Any]) -> str:
    message = str(source.get('message', '')).lower()
    event_id = str(source.get('event_id') or source.get('eventId', ''))
    
    if event_id in ('41', '1001'):
        return 'CRASH'
    if any(phrase in message for phrase in ('power loss', 'bugcheck', 'unexpected shutdown', 'did not shut down cleanly')):
        return 'CRASH'
    return 'ISSUE'


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
    crash_time, window_start, window_end = get_crash_window()
    
    # Priority: Always read ML results from local JSON first for summary fields
    summary = get_local_summary()
    anomaly_count = summary.get('anomaly_count', 0)
    root_cause = summary.get('root_cause_message', '')
    suggestion = summary.get('suggestion', {})
    tamper_detected = summary.get('tamper_detected', False)
    antiforensics = summary.get('antiforensics', {'detected': False, 'count': 0, 'events': []})

    if not await is_es_available(es):
        logs = get_local_logs()
        error_logs = [item for item in logs if str(item.get('level', '')).upper() == 'ERROR']
        crashes_list = [item for item in error_logs if determine_event_type(item) == 'CRASH']
        issues_list = [item for item in error_logs if determine_event_type(item) == 'ISSUE']
        last_crash = split_timestamp(crashes_list[0].get('@timestamp') if crashes_list else (error_logs[0].get('@timestamp') if error_logs else None))
        
        resp = {
            'totalCrashes': len(crashes_list),
            'totalIssues': len(issues_list),
            'lastCrash': last_crash,
            'rootCause': root_cause,
            'suggestion': suggestion,
            'anomalyCount': anomaly_count or len(get_local_anomalies()),
            'tamper_detected': tamper_detected,
            'antiforensics': antiforensics,
        }
        if crash_time:
            resp['crash_time'] = crash_time
            resp['window_start'] = window_start.isoformat()
            resp['window_end'] = window_end.isoformat()
        return resp

    try:
        # Properly compute via ES count queries rather than limiting to Top 1000
        crash_query = {
            "bool": {
                "filter": [
                    {"term": {"level.keyword": "ERROR"}},
                    {"bool": {
                        "should": [
                            {"terms": {"event_id": [41, 1001, "41", "1001"]}},
                            {"terms": {"eventId": [41, 1001, "41", "1001"]}},
                            {"match_phrase": {"message": "power loss"}},
                            {"match_phrase": {"message": "bugcheck"}},
                            {"match_phrase": {"message": "unexpected shutdown"}},
                            {"match_phrase": {"message": "did not shut down cleanly"}}
                        ],
                        "minimum_should_match": 1
                    }}
                ]
            }
        }
        
        crash_count_resp = await es.count(index=SYSTEM_LOGS_INDEX, body={"query": crash_query})
        crash_count = crash_count_resp.get('count', 0)
        
        error_count_resp = await es.count(index=SYSTEM_LOGS_INDEX, body={'query': {'terms': {'level.keyword': ['ERROR', 'CRITICAL']}}})
        error_count = error_count_resp.get('count', 0)
        
        issues_count = error_count - crash_count

        anomaly_query = {'query': {'bool': {'filter': [{'range': {'@timestamp': {'gte': window_start.isoformat(), 'lte': window_end.isoformat()}}}]}}} if window_start else {'query': {'match_all': {}}}
        anomaly_count_response = await es.count(
            index=ANOMALIES_INDEX,
            body=anomaly_query,
        )
        last_crash_response = await es.search(
            index=SYSTEM_LOGS_INDEX,
            body={
                'size': 50,
                'sort': [{'@timestamp': {'order': 'desc', 'unmapped_type': 'date'}}],
                'query': {'terms': {'level.keyword': ['ERROR', 'CRITICAL']}},
            },
        )
        
        last_crash_hit = last_crash_response.get('hits', {}).get('hits', [])
        # Find the most recent actual CRASH, fallback to first error if none
        actual_crash_hit = next((h for h in last_crash_hit if determine_event_type(h.get('_source', {})) == 'CRASH'), last_crash_hit[0] if last_crash_hit else None)
        
        root_cause_response = await es.search(
            index=ANOMALIES_INDEX,
            body={
                'size': 1,
                'sort': [{'@timestamp': {'order': 'desc', 'unmapped_type': 'date'}}],
                'query': {'term': {'is_root_cause': True}},
            },
        )

        root_cause_hit = root_cause_response.get('hits', {}).get('hits', [])

        last_crash_source = actual_crash_hit.get('_source', {}) if actual_crash_hit else {}
        root_cause_source = root_cause_hit[0].get('_source', {}) if root_cause_hit else {}

        last_crash = split_timestamp(last_crash_source.get('@timestamp') or last_crash_source.get('time'))
        root_cause = root_cause_source.get('rootCause') or root_cause_source.get('message') or ''

        resp = {
            'totalCrashes': crash_count,
            'totalIssues': issues_count,
            'lastCrash': last_crash,
            'rootCause': root_cause,
            'suggestion': suggestion,
            'anomalyCount': anomaly_count or anomaly_count_response.get('count', 0),
            'tamper_detected': tamper_detected,
            'antiforensics': antiforensics,
        }
        if crash_time:
            resp['crash_time'] = crash_time
            resp['window_start'] = window_start.isoformat()
            resp['window_end'] = window_end.isoformat()
        return resp
    except Exception:
        return {
            'totalCrashes': 0,
            'totalIssues': 0,
            'lastCrash': {'date': '', 'time': ''},
            'rootCause': root_cause,
            'suggestion': suggestion,
            'anomalyCount': anomaly_count,
            'tamper_detected': tamper_detected,
            'antiforensics': antiforensics,
        }


@router.get('/timeline')
async def get_dashboard_timeline(request: Request) -> List[Dict[str, Any]]:
    es = getattr(request.app.state, 'es', None)
    crash_time, window_start, window_end = get_crash_window()
    window_minutes = 60
    bucket_minutes = 5

    if not await is_es_available(es):
        logs = get_local_logs()
        anomalies = get_local_anomalies()
        if window_start:
            anomalies = [a for a in anomalies if parse_timestamp(a.get('@timestamp') or a.get('time')) and window_start <= parse_timestamp(a.get('@timestamp') or a.get('time')) <= window_end]
        crash_logs = [
            item for item in logs
            if str(item.get('level', '')).upper() == 'ERROR'
        ]
        if window_start:
            start = window_start
            end = window_end
        else:
            local_crash_time = parse_timestamp(
                (crash_logs[0] if crash_logs else {}).get('@timestamp')
                or (crash_logs[0] if crash_logs else {}).get('time')
            )
            if local_crash_time is None:
                local_crash_time = datetime.now(timezone.utc)
            start = local_crash_time - timedelta(minutes=window_minutes)
            end = local_crash_time + timedelta(minutes=window_minutes)
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
            'query': {'terms': {'level.keyword': ['ERROR', 'CRITICAL']}},
        },
    )
    crash_hits = crash_response.get('hits', {}).get('hits', [])
    crash_source = (crash_hits[0] if crash_hits else {}).get('_source', {})
    if window_start:
        start = window_start
        end = window_end
    else:
        local_crash_time = parse_timestamp(crash_source.get('@timestamp') or crash_source.get('time'))
        if local_crash_time is None:
            local_crash_time = datetime.now(timezone.utc)
        start = local_crash_time - timedelta(minutes=window_minutes)
        end = local_crash_time + timedelta(minutes=window_minutes)
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
        if window_start:
            anomalies = [a for a in anomalies if parse_timestamp(a.get('@timestamp') or a.get('time')) and window_start <= parse_timestamp(a.get('@timestamp') or a.get('time')) <= window_end]
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
                    'type': determine_event_type(item),
                }
            )
        return crashes

    body = {
        'size': 50,
        'sort': [{'@timestamp': {'order': 'desc', 'unmapped_type': 'date'}}],
        'query': {'terms': {'level.keyword': ['ERROR', 'CRITICAL']}},
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
                    'type': determine_event_type(source),
                }
            )
        return crashes
    except Exception:
        return []


@router.get('/rootcause')
async def get_dashboard_rootcause(request: Request) -> Dict[str, Any]:
    es = getattr(request.app.state, 'es', None)
    crash_time, window_start, window_end = get_crash_window()
    
    # Priority: ALWAYS read ml_results.json first for summary/suggestion data
    summary = get_local_summary()
    local_anomalies = get_local_anomalies()
    
    # Pre-populate return structure from JSON if available
    cluster_id = int(summary.get('root_cause_cluster', 0) or 0)
    label = summary.get('label', 'No data')
    anomaly_count = int(summary.get('anomaly_count', 0) or 0)
    top_score = float(summary.get('top_score', 0) or 0)
    description = summary.get('description', '')
    fix = summary.get('fix', '')
    suggestion = summary.get('suggestion', {})
    
    if not await is_es_available(es):
        if not local_anomalies:
            return rootcause_empty_state(
                'Local Mode is active, but no anomaly results have been generated yet.',
                'Run the ML pipeline once to create ml_results.json from the collected logs.',
            )

        cluster_events = [
            item for item in local_anomalies
            if int(item.get('cluster', item.get('cluster_id', 0)) or 0) == cluster_id
        ]
        if not cluster_events:
            cluster_events = local_anomalies[:10]

        events = []
        for index, event in enumerate(cluster_events[:10]):
            events.append({
                'time': event.get('@timestamp') or event.get('time') or '',
                'eventId': str(index + 1),
                'source': 'Root cause' if event.get('isRootCause') or event.get('is_root_cause') else 'Backend log',
                'message': event.get('message') or event.get('log') or '',
                'score': float(event.get('score', event.get('anomaly_score', 0)) or 0),
            })

        return {
            'clusterId': cluster_id,
            'label': label if label != 'No data' else infer_label([e['message'].lower() for e in events]),
            'confidence': min(99, int(30 + len(cluster_events) * 5 + abs(top_score) * 100)),
            'anomalyCount': anomaly_count or len(cluster_events),
            'topScore': top_score or (min(e['score'] for e in events) if events else 0),
            'events': events,
            'description': description or f'Local Mode identified {len(cluster_events)} anomalies.',
            'fix': fix or 'Review the top anomalies in this cluster.',
            'suggestion': suggestion,
        }

    try:
        # Fetch actual matching anomalies from ES to populate the EVENT LIST ONLY
        body = {
            'size': 100,
            'sort': [{'anomaly_score': {'order': 'asc'}}],
            'query': {'bool': {'filter': [{'range': {'@timestamp': {'gte': window_start.isoformat(), 'lte': window_end.isoformat()}}}]}} if window_start else {'match_all': {}},
        }

        response = await es.search(index=ANOMALIES_INDEX, body=body)
        hits = response.get('hits', {}).get('hits', [])

        events = []
        for i, hit in enumerate(hits[:10]):
            source = hit.get('_source', {})
            events.append({
                'time': source.get('@timestamp') or source.get('time') or '',
                'eventId': str(i + 1),
                'source': 'Root cause' if source.get('is_root_cause', source.get('isRootCause')) else 'Backend log',
                'message': source.get('message') or source.get('log') or '',
                'score': float(source.get('anomaly_score', source.get('score', 0)) or 0),
            })

        # Return combined state: Summary data from JSON, event list from ES (if available)
        return {
            'clusterId': cluster_id,
            'label': label,
            'confidence': min(99, int(30 + len(events) * 5 + abs(top_score) * 100)),
            'anomalyCount': anomaly_count or len(hits),
            'topScore': top_score,
            'events': events,
            'description': description or f'{anomaly_count} anomalies detected in cluster {cluster_id}.',
            'fix': fix,
            'suggestion': suggestion,
        }

    except Exception:
        return rootcause_empty_state(
            description or 'Search failed',
            fix or 'Check background logs'
        )
