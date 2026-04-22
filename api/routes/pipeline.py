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
    Runs log collection first and only starts ML after fresh logs are written.
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

        log_collector_result = subprocess.run(
            [sys.executable, str(LOG_COLLECTOR)],
            capture_output=True,
            text=True,
        )
        if log_collector_result.returncode != 0:
            return {
                'status': 'error',
                'message': 'Log collection failed',
                'code': log_collector_result.returncode,
                'stderr': log_collector_result.stderr[-1000:],
            }

        ml_pipeline_result = subprocess.run(
            [sys.executable, str(ML_PIPELINE)],
            capture_output=True,
            text=True,
        )
        if ml_pipeline_result.returncode != 0:
            return {
                'status': 'error',
                'message': 'ML pipeline failed',
                'code': ml_pipeline_result.returncode,
                'stderr': ml_pipeline_result.stderr[-1000:],
            }

        return {
            'status': 'completed',
            'message': 'Pipeline completed successfully',
        }

    except Exception as e:
        return {
            'status': 'error',
            'message': f'Failed to trigger pipeline: {str(e)}',
            'code': 500,
        }
