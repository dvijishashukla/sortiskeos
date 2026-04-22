from typing import Any, Dict, List

from fastapi import APIRouter, Query, Request

from data_access import SYSTEM_LOGS_INDEX, get_local_logs, is_es_available

router = APIRouter(tags=['logs'])


def normalize_cluster_value(source: Dict[str, Any]) -> int | str:
    """
    Return a UI-friendly cluster id.
    - If ES document stores DBSCAN raw `cluster_id` (0-based), convert to 1-based.
    - If only `cluster` exists, preserve it as-is.
    - Noise (-1, '-1', or '-') is rendered as 'noise'.
    """
    if 'cluster_id' in source:
        try:
            raw = int(source.get('cluster_id'))
        except (TypeError, ValueError):
            return 'noise'
        if raw < 0:
            return 'noise'
        return raw + 1

    cluster_value = source.get('cluster', '')
    if cluster_value in ('-1', '-'):
        return 'noise'
    try:
        parsed = int(cluster_value)
    except (TypeError, ValueError):
        return cluster_value if cluster_value is not None else 'noise'
    if parsed < 0:
        return 'noise'
    return parsed


def format_log(hit: Dict[str, Any], *, source_is_hit: bool = True) -> Dict[str, Any]:
    source = hit.get('_source', {}) if source_is_hit else hit
    # Prefer the model-native anomaly_score when present; legacy docs may carry
    # score=0 while anomaly_score has the actual value.
    raw_score = source.get('anomaly_score', source.get('score', 0))
    return {
        'time': source.get('@timestamp') or source.get('time') or '',
        'level': source.get('level') or '',
        'source': source.get('source') or source.get('host') or 'unknown',
        'message': source.get('message') or source.get('log') or '',
        'score': raw_score,
        'isRootCause': source.get('isRootCause', source.get('is_root_cause', False)),
        'cluster': normalize_cluster_value(source),
        'rootCause': source.get('rootCause', ''),
        'rootCauseScore': source.get('root_cause_score', 0),
        'rootCauseRank': source.get('root_cause_rank', 0),
        'suggestion': source.get('suggestion', {}),
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
