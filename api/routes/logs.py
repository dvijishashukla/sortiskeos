from typing import Any, Dict, List

from fastapi import APIRouter, Query, Request

from data_access import SYSTEM_LOGS_INDEX, get_local_logs, is_es_available

router = APIRouter(tags=['logs'])


def format_log(hit: Dict[str, Any], *, source_is_hit: bool = True) -> Dict[str, Any]:
    source = hit.get('_source', {}) if source_is_hit else hit
    return {
        'time': source.get('@timestamp') or source.get('time') or '',
        'level': source.get('level') or '',
        'source': source.get('source') or source.get('host') or 'unknown',
        'message': source.get('message') or source.get('log') or '',
        'score': source.get('score', source.get('anomaly_score', 0)),
        'isRootCause': source.get('isRootCause', source.get('is_root_cause', False)),
        'cluster': source.get('cluster', source.get('cluster_id', '')),
    }


def filter_local_logs(
    records: List[Dict[str, Any]],
    *,
    size: int,
    level: str | None,
    search: str | None,
) -> List[Dict[str, Any]]:
    if level:
        records = [item for item in records if item['level'] == level.upper()]
    if search:
        query = search.lower()
        records = [
            item for item in records
            if query in item['message'].lower() or query in str(item.get('source', '')).lower()
        ]
    return records[:size]


@router.get('/logs')
async def get_logs(
    request: Request,
    size: int = Query(default=50, ge=1, le=500),
    level: str | None = Query(default=None),
    search: str | None = Query(default=None),
) -> List[Dict[str, Any]]:
    es = getattr(request.app.state, 'es', None)
    local_records = [format_log(item, source_is_hit=False) for item in get_local_logs()]

    if not await is_es_available(es):
        return filter_local_logs(local_records, size=size, level=level, search=search)

    filters: List[Dict[str, Any]] = []
    must: List[Dict[str, Any]] = []

    if level:
        filters.append({'term': {'level.keyword': level.upper()}})

    if search:
        must.append(
            {
                'multi_match': {
                    'query': search,
                    'fields': ['message^2', 'message.keyword', 'log'],
                    'type': 'best_fields',
                }
            }
        )

    query: Dict[str, Any] = {'match_all': {}}
    if filters or must:
        query = {'bool': {'filter': filters, 'must': must}}

    body = {
        'size': size,
        'sort': [{'@timestamp': {'order': 'desc', 'unmapped_type': 'date'}}],
        'query': query,
    }

    try:
        response = await es.search(index=SYSTEM_LOGS_INDEX, body=body)
        hits = response.get('hits', {}).get('hits', [])
        if hits:
            return [format_log(hit) for hit in hits]
        return filter_local_logs(local_records, size=size, level=level, search=search)
    except Exception:
        return filter_local_logs(local_records, size=size, level=level, search=search)
