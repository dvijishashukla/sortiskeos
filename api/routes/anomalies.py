from typing import Any, Dict, List

from fastapi import APIRouter, Query, Request

from data_access import ANOMALIES_INDEX, get_local_anomalies, is_es_available
from routes.logs import format_log
from routes.dashboard import get_crash_window, parse_timestamp

router = APIRouter(tags=['anomalies'])


@router.get('/anomalies')
async def get_anomalies(
    request: Request,
    size: int = Query(default=20, ge=1, le=200),
    cluster: str | None = Query(default=None),
) -> List[Dict[str, Any]]:
    es = getattr(request.app.state, 'es', None)
    crash_time, window_start, window_end = get_crash_window()
    if not await is_es_available(es):
        records = [format_log(item, source_is_hit=False) for item in get_local_anomalies()]
        if window_start:
            records = [a for a in records if parse_timestamp(a.get('@timestamp') or a.get('time')) and window_start <= parse_timestamp(a.get('@timestamp') or a.get('time')) <= window_end]
        if cluster:
            records = [item for item in records if str(item['cluster']) == cluster]
        return records[:size]

    filters: List[Dict[str, Any]] = []
    if window_start:
        filters.append({'range': {'@timestamp': {'gte': window_start.isoformat(), 'lte': window_end.isoformat()}}})
    else:
        filters.append({'match_all': {}})
        
    if cluster:
        filters.append({'term': {'cluster_id': cluster}})

    body = {
        'size': size,
        'sort': [{'@timestamp': {'order': 'desc', 'unmapped_type': 'date'}}],
        'query': {
            'bool': {
                'filter': filters
            }
        },
    }

    try:
        response = await es.search(index=ANOMALIES_INDEX, body=body)
        hits = response.get('hits', {}).get('hits', [])
        return [format_log(hit) for hit in hits]
    except Exception:
        return []
