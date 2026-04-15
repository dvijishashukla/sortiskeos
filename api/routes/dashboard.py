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


async def get_latest_crash_window_from_es(es, *, window_before_min: int = 30, window_after_min: int = 10):
    if es is None:
        return None, None, None

    try:
        response = await es.search(
            index=SYSTEM_LOGS_INDEX,
            body={
                "size": 500,
                "sort": [{"@timestamp": {"order": "desc", "unmapped_type": "date"}}],
                "_source": ["@timestamp", "time", "message", "event_id", "eventId", "level"],
                "query": {"match_all": {}},
            },
        )
        hits = response.get("hits", {}).get("hits", [])
        if not hits:
            return None, None, None

        crash_ids = {"41", "6008", "1001"}
        crash_keywords = [
            "unexpected shutdown",
            "previous shutdown was unexpected",
            "kernel power",
            "bugcheck",
            "did not shut down cleanly",
        ]

        def _is_crash(source: Dict[str, Any]) -> bool:
            event_id = str(source.get("event_id") or source.get("eventId") or "")
            message = str(source.get("message") or "").lower()
            return event_id in crash_ids or any(keyword in message for keyword in crash_keywords)

        crash_hit = next((hit for hit in hits if _is_crash(hit.get("_source", {}))), hits[0])
        source = crash_hit.get("_source", {})
        crash_time_raw = source.get("@timestamp") or source.get("time")
        crash_dt = parse_timestamp(crash_time_raw)
        if crash_dt is None:
            return None, None, None

        return crash_time_raw, crash_dt - timedelta(minutes=window_before_min), crash_dt + timedelta(minutes=window_after_min)
    except Exception:
        return None, None, None

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


def to_display_cluster(raw_cluster: Any) -> int:
    try:
        cluster_id = int(raw_cluster)
    except (TypeError, ValueError):
        return 0
    if cluster_id < 0:
        return 0
    return cluster_id + 1


@router.get('/stats')
async def get_dashboard_stats(request: Request) -> Dict[str, Any]:
    es = getattr(request.app.state, 'es', None)
    crash_time, window_start, window_end = get_crash_window()

    es_ok = await is_es_available(es)

    # Local summary is authoritative only in local mode; otherwise it can
    # carry stale flags/messages from previous offline runs.
    summary = get_local_summary() if not es_ok else {}
    anomaly_count = summary.get('anomaly_count', 0)
    root_cause = summary.get('root_cause_message', '')
    suggestion = summary.get('suggestion', {})
    tamper_detected = summary.get('tamper_detected', False)
    antiforensics = summary.get('antiforensics', {'detected': False, 'count': 0, 'events': []})

    if not es_ok:
        logs = get_local_logs()
        error_logs = [item for item in logs if str(item.get('level', '')).upper() == 'ERROR']
        crashes_list = [item for item in error_logs if determine_event_type(item) == 'CRASH']
        issues_list = [item for item in error_logs if determine_event_type(item) == 'ISSUE']
        local_negative_anomalies = [
            item for item in get_local_anomalies()
            if float(item.get('anomaly_score', item.get('score', 0)) or 0) < 0
        ]
        last_crash = split_timestamp(crashes_list[0].get('@timestamp') if crashes_list else (error_logs[0].get('@timestamp') if error_logs else None))
        
        resp = {
            'totalCrashes': len(crashes_list),
            'totalIssues': len(issues_list),
            'lastCrash': last_crash,
            'rootCause': root_cause,
            'suggestion': suggestion,
            'anomalyCount': len(local_negative_anomalies),
            'tamper_detected': tamper_detected,
            'antiforensics': antiforensics,
        }
        if crash_time:
            resp['crash_time'] = crash_time
            resp['window_start'] = window_start.isoformat()
            resp['window_end'] = window_end.isoformat()
        return resp

    try:
        es_crash_time, es_window_start, es_window_end = await get_latest_crash_window_from_es(es)
        if es_crash_time and es_window_start and es_window_end:
            crash_time, window_start, window_end = es_crash_time, es_window_start, es_window_end

        window_filters: List[Dict[str, Any]] = []
        if window_start and window_end:
            window_filters.append({'range': {'@timestamp': {'gte': window_start.isoformat(), 'lte': window_end.isoformat()}}})

        # Properly compute via ES count queries rather than limiting to Top 1000
        crash_query = {
            "bool": {
                "filter": [
                    *window_filters,
                    {
                        "bool": {
                            "should": [
                                {"term": {"level.keyword": "ERROR"}},
                                {"term": {"level": "ERROR"}},
                            ],
                            "minimum_should_match": 1,
                        }
                    },
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
        
        error_count_resp = await es.count(
            index=SYSTEM_LOGS_INDEX,
            body={
                'query': {
                    'bool': {
                        'filter': [
                            *window_filters,
                            {
                                'bool': {
                                    'should': [
                                        {'terms': {'level.keyword': ['ERROR', 'CRITICAL']}},
                                        {'terms': {'level': ['ERROR', 'CRITICAL']}},
                                    ],
                                    'minimum_should_match': 1,
                                }
                            },
                        ]
                    }
                }
            },
        )
        error_count = error_count_resp.get('count', 0)
        
        issues_count = error_count - crash_count

        anomaly_filters: List[Dict[str, Any]] = []
        if window_start and window_end:
            anomaly_filters.append({'range': {'@timestamp': {'gte': window_start.isoformat(), 'lte': window_end.isoformat()}}})
        anomaly_filters.append({'range': {'anomaly_score': {'lt': 0}}})
        anomaly_query = {'query': {'bool': {'filter': anomaly_filters}}}
        anomaly_count_response = await es.count(
            index=ANOMALIES_INDEX,
            body=anomaly_query,
        )
        last_crash_response = await es.search(
            index=SYSTEM_LOGS_INDEX,
            body={
                'size': 50,
                'sort': [{'@timestamp': {'order': 'desc', 'unmapped_type': 'date'}}],
                'query': {
                    'bool': {
                        'filter': [
                            *window_filters,
                            {
                                'bool': {
                                    'should': [
                                        {'terms': {'level.keyword': ['ERROR', 'CRITICAL']}},
                                        {'terms': {'level': ['ERROR', 'CRITICAL']}},
                                    ],
                                    'minimum_should_match': 1,
                                }
                            },
                        ]
                    }
                },
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
                'query': {
                    'bool': {
                        'filter': [
                            *(
                                [{'range': {'@timestamp': {'gte': window_start.isoformat(), 'lte': window_end.isoformat()}}}]
                                if window_start and window_end else []
                            ),
                            {'term': {'is_root_cause': True}},
                            {'range': {'anomaly_score': {'lt': 0}}},
                        ]
                    }
                },
            },
        )

        root_cause_hit = root_cause_response.get('hits', {}).get('hits', [])

        last_crash_source = actual_crash_hit.get('_source', {}) if actual_crash_hit else {}
        root_cause_source = root_cause_hit[0].get('_source', {}) if root_cause_hit else {}

        last_crash = split_timestamp(last_crash_source.get('@timestamp') or last_crash_source.get('time'))
        root_cause = (
            root_cause_source.get('rootCause')
            or root_cause_source.get('message')
            or root_cause_source.get('label')
            or last_crash_source.get('message')
            or ''
        )
        effective_suggestion = suggestion or root_cause_source.get('suggestion', {})
        if not effective_suggestion and root_cause:
            effective_suggestion = {
                'category': infer_label([str(root_cause).lower()]),
                'confidence': 'Medium',
                'likely_cause': str(root_cause),
            }

        resp = {
            'totalCrashes': crash_count,
            'totalIssues': issues_count,
            'lastCrash': last_crash,
            'rootCause': root_cause,
            'suggestion': effective_suggestion,
            'anomalyCount': anomaly_count_response.get('count', 0),
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
        anomalies = [
            item for item in get_local_anomalies()
            if float(item.get('anomaly_score', item.get('score', 0)) or 0) < 0
        ]
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

    es_crash_time, es_window_start, es_window_end = await get_latest_crash_window_from_es(es)
    if es_crash_time and es_window_start and es_window_end:
        crash_time, window_start, window_end = es_crash_time, es_window_start, es_window_end

    crash_response = await es.search(
        index=SYSTEM_LOGS_INDEX,
        body={
            'size': 1,
            'sort': [{'@timestamp': {'order': 'desc', 'unmapped_type': 'date'}}],
            'query': {
                'bool': {
                    'should': [
                        {'terms': {'level.keyword': ['ERROR', 'CRITICAL']}},
                        {'terms': {'level': ['ERROR', 'CRITICAL']}},
                    ],
                    'minimum_should_match': 1,
                }
            },
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
                    {'range': {'anomaly_score': {'lt': 0}}},
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
        has_non_zero = False
        for bucket in buckets:
            value = bucket.get('worst_score', {}).get('value')
            hour = bucket.get('key_as_string', '')
            score = 0 if value is None else float(value)
            if score < 0:
                has_non_zero = True
            timeline.append({'hour': hour, 'score': score})

        if has_non_zero:
            return timeline

        # Fallback: derive timeline from raw docs when aggregation returns nulls.
        raw_response = await es.search(
            index=ANOMALIES_INDEX,
            body={
                'size': 2000,
                'sort': [{'@timestamp': {'order': 'asc', 'unmapped_type': 'date'}}],
                'query': {
                    'bool': {
                        'filter': [
                            {'range': {'@timestamp': {'gte': start.isoformat(), 'lte': end.isoformat()}}},
                            {'range': {'anomaly_score': {'lt': 0}}},
                        ]
                    }
                },
            },
        )
        raw_hits = raw_response.get('hits', {}).get('hits', [])
        if not raw_hits:
            return timeline

        timeline_map = {
            start + timedelta(minutes=offset * bucket_minutes): []
            for offset in range(int(((end - start).total_seconds() // 60) / bucket_minutes) + 1)
        }
        for hit in raw_hits:
            source = hit.get('_source', {})
            ts = parse_timestamp(source.get('@timestamp') or source.get('time'))
            if ts is None or ts < start or ts > end:
                continue
            bucket = floor_to_bucket(ts, bucket_minutes)
            score = float(source.get('anomaly_score', source.get('score', 0)) or 0)
            if bucket in timeline_map and score < 0:
                timeline_map[bucket].append(score)

        rebuilt: List[Dict[str, Any]] = []
        for bucket in sorted(timeline_map.keys()):
            values = timeline_map[bucket]
            rebuilt.append({'hour': bucket.isoformat(), 'score': min(values) if values else 0})
        return rebuilt
    except Exception:
        return []


@router.get('/crashes')
async def get_dashboard_crashes(request: Request) -> List[Dict[str, Any]]:
    es = getattr(request.app.state, 'es', None)
    if not await is_es_available(es):
        logs = get_local_logs()
        anomalies = [
            item for item in get_local_anomalies()
            if float(item.get('anomaly_score', item.get('score', 0)) or 0) < 0
        ]
        summary = get_local_summary()
        crash_logs = [item for item in logs if str(item.get('level', '')).upper() == 'ERROR'][:50]
        anomaly_count = len(anomalies)
        root_message = summary.get('root_cause_message') or ''
        score = min((float(item.get('anomaly_score', item.get('score', 0)) or 0) for item in anomalies), default=0.0)
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

    es_crash_time, es_window_start, es_window_end = await get_latest_crash_window_from_es(es)
    crash_time, window_start, window_end = get_crash_window()
    if es_crash_time and es_window_start and es_window_end:
        crash_time, window_start, window_end = es_crash_time, es_window_start, es_window_end

    window_filters: List[Dict[str, Any]] = []
    if window_start and window_end:
        window_filters.append({'range': {'@timestamp': {'gte': window_start.isoformat(), 'lte': window_end.isoformat()}}})

    body = {
        'size': 50,
        'sort': [{'@timestamp': {'order': 'desc', 'unmapped_type': 'date'}}],
        'query': {
            'bool': {
                'filter': [
                    *window_filters,
                    {
                        'bool': {
                            'should': [
                                {'terms': {'level.keyword': ['ERROR', 'CRITICAL']}},
                                {'terms': {'level': ['ERROR', 'CRITICAL']}},
                            ],
                            'minimum_should_match': 1,
                        }
                    },
                ]
            }
        },
    }

    try:
        response = await es.search(index=SYSTEM_LOGS_INDEX, body=body)
        hits = response.get('hits', {}).get('hits', [])
        crash_times = [
            parse_timestamp(
                (hit.get('_source', {}) or {}).get('@timestamp')
                or (hit.get('_source', {}) or {}).get('time')
            )
            for hit in hits
        ]
        crash_times = [dt for dt in crash_times if dt is not None]

        anomaly_query: Dict[str, Any] = {'match_all': {}}
        if crash_times:
            start = min(crash_times) - timedelta(minutes=5)
            end = max(crash_times) + timedelta(minutes=5)
            anomaly_query = {
                'range': {'@timestamp': {'gte': start.isoformat(), 'lte': end.isoformat()}}
            }

        anomaly_response = await es.search(
            index=ANOMALIES_INDEX,
            body={
                'size': 2000,
                'sort': [{'@timestamp': {'order': 'desc', 'unmapped_type': 'date'}}],
                'query': {
                    'bool': {
                        'filter': [anomaly_query, {'range': {'anomaly_score': {'lt': 0}}}],
                    }
                },
            },
        )
        anomaly_hits = [item.get('_source', {}) for item in anomaly_response.get('hits', {}).get('hits', [])]
        crashes: List[Dict[str, Any]] = []
        for hit in hits:
            source = hit.get('_source', {})
            crash_dt = parse_timestamp(source.get('@timestamp') or source.get('time'))
            if crash_dt:
                # Count anomalies within 2 minutes of crash time
                window_anomalies = [
                    a for a in anomaly_hits
                    if parse_timestamp(a.get('@timestamp') or a.get('time')) and
                    abs((parse_timestamp(a.get('@timestamp') or a.get('time')) - crash_dt).total_seconds()) <= 120
                ]
                anomaly_count = len(window_anomalies)
                # Use the most anomalous score (most negative)
                anomaly_scores = [float(a.get('anomaly_score', 0)) for a in window_anomalies if a.get('anomaly_score') is not None]
                min_score = min(anomaly_scores) if anomaly_scores else 0.0
            else:
                anomaly_count = 0
                min_score = 0.0
            
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
                    'anomalies': anomaly_count,
                    'score': min_score,
                    'cluster': (
                        to_display_cluster((matching_anomaly or {}).get('cluster_id'))
                        if matching_anomaly is not None and (matching_anomaly or {}).get('cluster_id') is not None
                        else ''
                    ),
                    'method': 'IsolationForest+DBSCAN' if matching_anomaly is not None else 'Unmatched',
                    'type': determine_event_type(source),
                }
            )
        return crashes
    except Exception:
        return []


@router.get('/rootcause')
async def get_dashboard_rootcause(request: Request) -> Dict[str, Any]:
    es = getattr(request.app.state, 'es', None)
    _, window_start, window_end = get_crash_window()

    es_ok = await is_es_available(es)

    # Local summary remains a fallback only.
    summary = get_local_summary() if not es_ok else {}
    local_anomalies = [
        item for item in get_local_anomalies()
        if float(item.get('anomaly_score', item.get('score', 0)) or 0) < 0
    ]

    cluster_id = int(summary.get('root_cause_cluster', 0) or 0)
    label = summary.get('label', 'No data')
    anomaly_count = int(summary.get('anomaly_count', 0) or 0)
    top_score = float(summary.get('top_score', 0) or 0)
    description = summary.get('description', '')
    fix = summary.get('fix', '')
    suggestion = summary.get('suggestion', {})
    
    if not es_ok:
        if not local_anomalies:
            return rootcause_empty_state(
                'Local Mode is active, but no scored anomalies (< 0) were produced for the latest run.',
                'Run the ML pipeline again on fresh crash logs and confirm anomaly_score values are negative for outliers.',
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
            'clusterId': to_display_cluster(cluster_id),
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
        es_crash_time, es_window_start, es_window_end = await get_latest_crash_window_from_es(es)
        if es_crash_time and es_window_start and es_window_end:
            window_start, window_end = es_window_start, es_window_end

        filters: List[Dict[str, Any]] = []
        if window_start:
            filters.append({'range': {'@timestamp': {'gte': window_start.isoformat(), 'lte': window_end.isoformat()}}})
        filters.append({'range': {'anomaly_score': {'lt': 0}}})

        body = {
            'size': 1000,
            'sort': [{'anomaly_score': {'order': 'asc'}}],
            'query': {'bool': {'filter': filters}} if filters else {'match_all': {}},
        }

        response = await es.search(index=ANOMALIES_INDEX, body=body)
        hits = response.get('hits', {}).get('hits', [])

        if not hits:
            return rootcause_empty_state(
                description or 'No scored anomalies were found in the latest crash window.',
                fix or 'Run the pipeline again and ensure anomalies are indexed for the active crash window.',
            )

        docs = [hit.get('_source', {}) for hit in hits]
        root_docs = [doc for doc in docs if bool(doc.get('is_root_cause', doc.get('isRootCause')))]

        if root_docs:
            chosen_cluster_raw = root_docs[0].get('cluster_id', root_docs[0].get('cluster', 0))
            cluster_docs = [doc for doc in docs if doc.get('cluster_id', doc.get('cluster')) == chosen_cluster_raw]
        else:
            counts: Dict[int, int] = {}
            for doc in docs:
                raw = doc.get('cluster_id', doc.get('cluster'))
                try:
                    cid = int(raw)
                except (TypeError, ValueError):
                    continue
                if cid < 0:
                    continue
                counts[cid] = counts.get(cid, 0) + 1

            if not counts:
                # Fallback when DBSCAN did not form any valid cluster
                top = sorted(docs, key=lambda d: float(d.get('anomaly_score', d.get('score', 0)) or 0))[:1]
                top_doc = top[0] if top else {}
                candidate_score = float(top_doc.get('anomaly_score', top_doc.get('score', 0)) or 0)
                candidate_label = (
                    top_doc.get('rootCause')
                    or top_doc.get('message')
                    or label
                    or 'Noise candidate'
                )
                return {
                    'clusterId': 0,
                    'label': candidate_label,
                    'confidence': min(99, int(40 + abs(candidate_score) * 100)),
                    'anomalyCount': len(docs),
                    'topScore': candidate_score,
                    'events': [
                        {
                            'time': doc.get('@timestamp') or doc.get('time') or '',
                            'eventId': str(i + 1),
                            'source': 'Root cause' if doc.get('is_root_cause', doc.get('isRootCause')) else 'Backend log',
                            'message': doc.get('message') or doc.get('log') or '',
                            'score': float(doc.get('anomaly_score', doc.get('score', 0)) or 0),
                        }
                        for i, doc in enumerate(sorted(docs, key=lambda d: float(d.get('anomaly_score', d.get('score', 0)) or 0))[:10])
                    ],
                    'description': description or 'No cluster could be formed. Showing the strongest negative anomaly candidate instead.',
                    'fix': fix or 'Review the top anomaly candidate and rerun pipeline after adjusting clustering parameters.',
                    'suggestion': top_doc.get('suggestion', {}) or suggestion,
                }

            chosen_cluster_raw = max(counts.items(), key=lambda x: x[1])[0]
            cluster_docs = [doc for doc in docs if int(doc.get('cluster_id', doc.get('cluster', -1)) or -1) == chosen_cluster_raw]

        cluster_docs_sorted = sorted(
            cluster_docs,
            key=lambda d: float(d.get('anomaly_score', d.get('score', 0)) or 0),
        )

        events = []
        for i, source in enumerate(cluster_docs_sorted[:10]):
            events.append({
                'time': source.get('@timestamp') or source.get('time') or '',
                'eventId': str(i + 1),
                'source': 'Root cause' if source.get('is_root_cause', source.get('isRootCause')) else 'Backend log',
                'message': source.get('message') or source.get('log') or '',
                'score': float(source.get('anomaly_score', source.get('score', 0)) or 0),
            })

        top = cluster_docs_sorted[0] if cluster_docs_sorted else {}
        effective_top_score = float(top.get('anomaly_score', top.get('score', top_score)) or 0)
        effective_label = (
            top.get('label')
            or top.get('rootCause')
            or top.get('message')
            or label
            or infer_label([str(e.get('message', '')).lower() for e in events])
        )
        effective_suggestion = suggestion or top.get('suggestion', {})

        return {
            'clusterId': to_display_cluster(chosen_cluster_raw),
            'label': effective_label,
            'confidence': min(99, int(35 + len(cluster_docs_sorted) * 4 + abs(effective_top_score) * 100)),
            'anomalyCount': len(cluster_docs_sorted),
            'topScore': effective_top_score,
            'events': events,
            'description': description or f'{len(cluster_docs_sorted)} anomalies detected in cluster C{to_display_cluster(chosen_cluster_raw)}.',
            'fix': fix or top.get('fix') or 'Investigate the earliest/highest-severity events in this cluster.',
            'suggestion': effective_suggestion,
        }

    except Exception:
        return rootcause_empty_state(
            description or 'Search failed',
            fix or 'Check background logs'
        )
