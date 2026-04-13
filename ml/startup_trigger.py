import logging
import os
import subprocess
import sys
import time
import webbrowser
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LAST_RUN_FILE = BASE_DIR / 'last_run.txt'
LOG_FILE = BASE_DIR / 'trigger.log'
LOG_COLLECTOR = BASE_DIR / 'log_collector.py'
ML_PIPELINE = BASE_DIR / 'ml_pipeline.py'
ES_URL = 'http://localhost:9200'
DASHBOARD_URL = 'http://localhost:3000'
ES_WAIT_SECONDS = 60
ES_POLL_INTERVAL = 5
COOLDOWN_SECONDS = int(os.getenv('COOLDOWN_SECONDS', '300'))

logger = logging.getLogger('startup_trigger')
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def get_requests_module():
    try:
        import requests
    except ModuleNotFoundError:
        logger.error(
            'The requests package is required for the Elasticsearch health check. '
            'Install it in your active Python environment before running startup_trigger.py.'
        )
        return None
    return requests


def read_last_run() -> datetime | None:
    if not LAST_RUN_FILE.exists():
        return None

    try:
        raw_value = LAST_RUN_FILE.read_text(encoding='utf-8').strip()
        if not raw_value:
            return None
        return datetime.fromisoformat(raw_value)
    except Exception as exc:
        logger.warning('Could not read last run timestamp from %s: %s', LAST_RUN_FILE, exc)
        return None


def write_last_run(now: datetime) -> None:
    LAST_RUN_FILE.write_text(now.isoformat(), encoding='utf-8')
    logger.info('Recorded successful run timestamp in %s', LAST_RUN_FILE)


def is_in_cooldown(last_run: datetime | None, now: datetime) -> bool:
    if last_run is None:
        return False

    if last_run.tzinfo is None:
        last_run = last_run.replace(tzinfo=timezone.utc)

    elapsed_seconds = (now - last_run).total_seconds()
    if elapsed_seconds < COOLDOWN_SECONDS:
        remaining = int(COOLDOWN_SECONDS - elapsed_seconds)
        logger.info(
            'Cooldown active. Last run was %s. Skipping pipeline for another %s seconds.',
            last_run.isoformat(),
            remaining,
        )
        return True

    return False


def wait_for_elasticsearch() -> bool:
    requests = get_requests_module()
    if requests is None:
        return False

    deadline = time.monotonic() + ES_WAIT_SECONDS
    attempt = 1

    while time.monotonic() < deadline:
        try:
            response = requests.get(ES_URL, timeout=5)
            if response.ok:
                logger.info('Elasticsearch is reachable at %s', ES_URL)
                return True
            logger.warning(
                'Elasticsearch responded with status %s on attempt %s.',
                response.status_code,
                attempt,
            )
        except requests.RequestException as exc:
            logger.warning('Elasticsearch not reachable on attempt %s: %s', attempt, exc)

        attempt += 1
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        sleep_seconds = ES_POLL_INTERVAL if remaining >= ES_POLL_INTERVAL else remaining
        logger.info('Waiting %.0f seconds before checking Elasticsearch again.', sleep_seconds)
        time.sleep(sleep_seconds)

    logger.error('Elasticsearch did not become reachable within %s seconds. Exiting cleanly.', ES_WAIT_SECONDS)
    return False


def run_script(script_path: Path) -> bool:
    command = [sys.executable, str(script_path)]
    logger.info('Running %s using %s', script_path.name, sys.executable)

    try:
        result = subprocess.run(command, cwd=str(BASE_DIR), check=False)
    except Exception as exc:
        logger.error('Failed to start %s: %s', script_path.name, exc)
        return False

    if result.returncode != 0:
        logger.error('%s failed with exit code %s', script_path.name, result.returncode)
        return False

    logger.info('%s completed successfully.', script_path.name)
    return True


def open_dashboard() -> None:
    try:
        if hasattr(os, 'startfile'):
            os.startfile(DASHBOARD_URL)
            logger.info('Opened dashboard in the default browser using os.startfile: %s', DASHBOARD_URL)
            return

        opened = webbrowser.open(DASHBOARD_URL)
        if opened:
            logger.info('Opened dashboard in the default browser: %s', DASHBOARD_URL)
        else:
            logger.warning('Browser open request was not acknowledged for %s', DASHBOARD_URL)
    except Exception as exc:
        logger.warning('Failed to open dashboard URL %s: %s', DASHBOARD_URL, exc)


def main() -> int:
    logger.info('Startup trigger started.')
    
    # 1. Wait for Elasticsearch (urllib check, max 60 seconds)
    logger.info('Waiting for Elasticsearch to be ready (max 60s)...')
    max_wait = 60
    waited = 0
    es_up = False
    while waited < max_wait:
        try:
            with urllib.request.urlopen('http://localhost:9200', timeout=3) as response:
                if response.status == 200:
                    logger.info('Elasticsearch is up.')
                    es_up = True
                    break
        except Exception:
            pass
        
        logger.info('Waiting for ES... %ss', waited)
        time.sleep(5)
        waited += 5

    if not es_up:
        logger.error('Elasticsearch did not become reachable within 60 seconds.')
        # Proceeding anyway as fallback might use local logs
    
    # 2. Run log_collector.py
    if not run_script(LOG_COLLECTOR):
        logger.error('Stopping because log_collector.py did not complete successfully.')
        return 1

    # 3. Run ml_pipeline.py
    if not run_script(ML_PIPELINE):
        logger.error('Stopping because ml_pipeline.py did not complete successfully.')
        return 1

    # 4. Open browser
    open_dashboard()
    
    write_last_run(datetime.now(timezone.utc))
    logger.info('Startup trigger completed successfully.')
    
    print("\n" + "="*60)
    print("REGISTRATION REQUIRED: Run these commands as Administrator:")
    print("="*60)
    print("Task 1: schtasks /create /tn \"Sortiskeos-Services\" /tr \"c:\\sortiskeos\\run_all.bat\" /sc onlogon /rl highest /f")
    print("Task 2: schtasks /create /tn \"Sortiskeos-Analysis\" /tr \"python c:\\sortiskeos\\ml\\startup_trigger.py\" /sc onlogon /rl highest /delay 00:03:00 /f")
    print("="*60 + "\n")
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
