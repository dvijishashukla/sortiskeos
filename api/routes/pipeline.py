import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter

router = APIRouter(prefix='/pipeline', tags=['pipeline'])

BASE_DIR = Path(__file__).resolve().parent.parent
ML_DIR = BASE_DIR.parent / 'ml'
LOG_COLLECTOR = ML_DIR / 'log_collector.py'
ML_PIPELINE = ML_DIR / 'ml_pipeline.py'


@router.post('/run')
async def trigger_pipeline() -> Dict[str, Any]:
    """
    Trigger the ML pipeline: log collection + anomaly detection.
    Runs both scripts as non-blocking subprocesses using subprocess.Popen.
    Returns immediately without waiting for completion.
    """
    try:
        # Validate that both script files exist
        if not LOG_COLLECTOR.exists():
            return {
                'status': 'error',
                'message': f'Log collector script not found at {LOG_COLLECTOR}',
                'code': 404,
            }
        if not ML_PIPELINE.exists():
            return {
                'status': 'error',
                'message': f'ML pipeline script not found at {ML_PIPELINE}',
                'code': 404,
            }

        # Start log collector in background
        log_collector_process = subprocess.Popen(
            [sys.executable, str(LOG_COLLECTOR)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Start ML pipeline in background
        ml_pipeline_process = subprocess.Popen(
            [sys.executable, str(ML_PIPELINE)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return {
            'status': 'started',
            'message': 'Pipeline triggered successfully',
        }

    except Exception as e:
        return {
            'status': 'error',
            'message': f'Failed to trigger pipeline: {str(e)}',
            'code': 500,
        }
