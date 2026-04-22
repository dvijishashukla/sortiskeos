from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Query, Request

from data_access import ANOMALIES_INDEX, get_local_anomalies, is_es_available
from routes.logs import format_log
from routes.dashboard import get_crash_window, get_latest_crash_window_from_es, parse_timestamp

router = APIRouter(tags=['anomalies'])


def anomaly_sort_key(item: Dict[str, Any]) -> tuple[float, float, str]:
    try:
        rank = float(item.get('rootCauseRank', 0) or 0)
    except (TypeError, ValueError):
        rank = 0.0
    try:
        score = float(item.get('rootCauseScore', item.get('score', 0)) or 0)
    except (TypeError, ValueError):
        score = 0.0
    event_time = str(item.get('time') or '')
    effective_rank = rank if rank > 0 else 999999.0
    return (effective_rank, -score, event_time)


@router.get('/anomalies')
async def get_anomalies(
    request: Request,
    size: int = Query(default=20, ge=1, le=200),
    cluster: str | None = Query(default=None),
) -> List[Dict[str, Any]]:
    es = getattr(request.app.state, 'es', None)
    _, window_start, window_end = get_crash_window()
    if not await is_es_available(es):
        raw_records = get_local_anomalies()
        if window_start:
            raw_records = [
                item for item in raw_records
                if parse_timestamp(item.get('@timestamp') or item.get('time'))
                and window_start <= parse_timestamp(item.get('@timestamp') or item.get('time')) <= window_end
            ]
        records = [format_log(item, source_is_hit=False) for item in raw_records]
        records = [item for item in records if float(item.get('score', 0) or 0) < 0]
        if cluster:
            records = [item for item in records if str(item['cluster']) == cluster]
        records.sort(key=anomaly_sort_key)
        return records[:size]

    filters: List[Dict[str, Any]] = []
    es_crash_time, es_window_start, es_window_end = await get_latest_crash_window_from_es(es)
    if es_crash_time and es_window_start and es_window_end:
        window_start, window_end = es_window_start, es_window_end
    elif window_start and window_end:
        # Keep local crash window as fallback only if ES crash marker is unavailable.
        pass
    else:
        # Last-resort safety window: keep only very recent anomalies to avoid
        # showing stale historical docs.
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(hours=2)
        window_end = now

    if window_start:
        filters.append({'range': {'@timestamp': {'gte': window_start.isoformat(), 'lte': window_end.isoformat()}}})
    filters.append({'exists': {'field': 'anomaly_score'}})
    filters.append({'range': {'anomaly_score': {'lt': 0}}})
        
    if cluster:
        try:
            requested_cluster = int(cluster)
        except ValueError:
            requested_cluster = None

        if requested_cluster is not None:
            raw_candidate = requested_cluster - 1
            filters.append(
                {
                    'bool': {
                        'should': [
                            {'term': {'cluster_id': raw_candidate}},
                            {'term': {'cluster': requested_cluster}},
                        ],
                        'minimum_should_match': 1,
                    }
                }
            )

    body = {
        'size': size,
        'sort': [
            {'root_cause_rank': {'order': 'asc', 'missing': '_last'}},
            {'root_cause_score': {'order': 'desc', 'missing': '_last'}},
            {'anomaly_score': {'order': 'asc', 'missing': '_last'}},
            {'@timestamp': {'order': 'desc', 'unmapped_type': 'date'}},
        ],
        'query': {
            'bool': {
                'filter': filters if filters else [{'match_all': {}}]
            }
        },
    }

    try:
        response = await es.search(index=ANOMALIES_INDEX, body=body)
        hits = response.get('hits', {}).get('hits', [])
        records = [format_log(hit) for hit in hits]
        records.sort(key=anomaly_sort_key)
        # Keep strict filtering in ES mode; do not return stale anomalies from
        # historical windows.
        if records:
            return records
        return []
    except Exception:
        return []
