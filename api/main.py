from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from elasticsearch import AsyncElasticsearch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from auth import APIKeyMiddleware, PipelineRateLimitMiddleware
from data_access import is_es_available
from routes.anomalies import router as anomalies_router
from routes.dashboard import router as dashboard_router
from routes.logs import router as logs_router
from routes.pipeline import router as pipeline_router

BASE_DIR = Path(__file__).resolve().parent
API_ENV_PATH = BASE_DIR / '.env'
load_dotenv(dotenv_path=API_ENV_PATH)

ES_HOST = os.getenv('ES_HOST', 'http://localhost:9200')
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000').split(',')
    if origin.strip()
]

app = FastAPI(title='Sortiskeos API')
app.add_middleware(PipelineRateLimitMiddleware)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.state.es_host = ES_HOST
app.state.es = AsyncElasticsearch(ES_HOST)

app.include_router(logs_router)
app.include_router(anomalies_router)
app.include_router(dashboard_router)
app.include_router(pipeline_router)


@app.get('/health')
async def health() -> dict:
    es: Optional[AsyncElasticsearch] = getattr(app.state, 'es', None)
    elasticsearch_ok = await is_es_available(es)
    return {
        'status': 'ok',
        'elasticsearch': elasticsearch_ok,
        'mode': 'elasticsearch' if elasticsearch_ok else 'local',
    }


@app.on_event('shutdown')
async def shutdown_event() -> None:
    es: Optional[AsyncElasticsearch] = getattr(app.state, 'es', None)
    if es is not None:
        try:
            await es.close()
        except Exception:
            pass
