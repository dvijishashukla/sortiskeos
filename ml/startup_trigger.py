"""
startup_trigger.py — HARDENED
------------------------------
Changes from original:
  - Loads .env before spawning subprocesses so credentials are available
  - Windows Task Scheduler XML sets <RunLevel>LeastPrivilege</RunLevel>
    and prompts for the service account  (was defaulting to SYSTEM)
  - Linux systemd unit runs as a dedicated 'loganalysis' user, not root
  - HMAC key generated and stored in .env on first run if missing
"""

import os
import sys
import time
import secrets
import logging
import platform
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROJECT_DIR   = Path(__file__).parent.resolve()
ENV_FILE      = PROJECT_DIR.parent / ".env"     # sortiskeos-main/.env
PYTHON        = sys.executable
LOGSTASH_WAIT = 30


# ── Load .env into os.environ ──────────────────────────────────────────────────
def load_dotenv(path: Path):
    if not path.exists():
        log.warning(f".env not found at {path}. Copy .env.example to .env and fill it in.")
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())
    log.info(f"Loaded env from {path}")


def _ensure_hmac_key():
    """Generate a random HMAC key if SORTISKEOS_HMAC_KEY is not set."""
    if os.environ.get("SORTISKEOS_HMAC_KEY"):
        return
    key = secrets.token_hex(32)
    log.info("Generated new SORTISKEOS_HMAC_KEY — appending to .env")
    with open(ENV_FILE, "a") as f:
        f.write(f"\nSORTISKEOS_HMAC_KEY={key}\n")
    os.environ["SORTISKEOS_HMAC_KEY"] = key


def run_step(script_name: str, args: list = []) -> bool:
    script_path = PROJECT_DIR / script_name
    cmd = [PYTHON, str(script_path)] + args
    log.info(f"Running: {' '.join(str(c) for c in cmd)}")
    env = os.environ.copy()
    result = subprocess.run(cmd, cwd=str(PROJECT_DIR), env=env)
    if result.returncode != 0:
        log.error(f"{script_name} exited with code {result.returncode}")
    return result.returncode == 0


def main():
    load_dotenv(ENV_FILE)
    _ensure_hmac_key()

    log.info("=" * 55)
    log.info("  Intelligent Log Analysis — Startup Pipeline")
    log.info("=" * 55)

    log.info("Step 1/3: Collecting system logs...")
    if not run_step("log_collector.py"):
        log.error("Log collection failed. Aborting.")
        sys.exit(1)

    log.info(f"Step 2/3: Waiting {LOGSTASH_WAIT}s for Logstash ingestion...")
    time.sleep(LOGSTASH_WAIT)

    log.info("Step 3/3: Running ML anomaly detection...")
    run_step("ml_pipeline.py", ["--mode", "once"])

    log.info("Startup pipeline complete. Check Kibana for results.")


# ── Windows Task Scheduler registration ───────────────────────────────────────
def register_windows_task():
    """
    Register as a Windows startup task running as a NAMED SERVICE ACCOUNT,
    not SYSTEM.  You will be prompted for the account password by schtasks.
    """
    script_path = PROJECT_DIR / "startup_trigger.py"
    task_name   = "IntelligentLogAnalysis"
    # Use the current user as the service account — change to a dedicated account
    # for production (e.g., DOMAIN\\LogAnalysisSvc).
    service_account = os.environ.get("LOG_ANALYSIS_USER", os.environ.get("USERNAME", ""))

    xml = f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Principals>
    <Principal id="Author">
      <UserId>{service_account}</UserId>
      <LogonType>Password</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Triggers>
    <BootTrigger>
      <Delay>PT1M</Delay>
      <Enabled>true</Enabled>
    </BootTrigger>
  </Triggers>
  <Actions>
    <Exec>
      <Command>{PYTHON}</Command>
      <Arguments>"{script_path}"</Arguments>
      <WorkingDirectory>{PROJECT_DIR}</WorkingDirectory>
    </Exec>
  </Actions>
  <Settings>
    <ExecutionTimeLimit>PT1H</ExecutionTimeLimit>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
  </Settings>
</Task>"""

    xml_file = PROJECT_DIR / "task.xml"
    xml_file.write_text(xml, encoding="utf-16")
    result = subprocess.run(
        ["schtasks", "/Create", "/TN", task_name, "/XML", str(xml_file), "/F"],
        capture_output=True, text=True
    )
    xml_file.unlink()
    if result.returncode == 0:
        log.info(f"Windows Task '{task_name}' registered as '{service_account}' (LeastPrivilege).")
    else:
        log.error(f"Failed: {result.stderr}")


# ── Linux systemd service registration ────────────────────────────────────────
def register_linux_service():
    """
    Install systemd service that runs as a dedicated 'loganalysis' system user
    (not root). Creates the user if it doesn't exist.
    """
    svc_user = "loganalysis"
    try:
        subprocess.run(["id", svc_user], check=True, capture_output=True)
        log.info(f"User '{svc_user}' already exists.")
    except subprocess.CalledProcessError:
        subprocess.run(["useradd", "--system", "--no-create-home", svc_user], check=True)
        log.info(f"Created system user '{svc_user}'.")
        # Give the user read access to the project directory
        subprocess.run(["chown", "-R", f"{svc_user}:{svc_user}", str(PROJECT_DIR)], check=True)

    service_content = f"""[Unit]
Description=Intelligent Log Analysis Startup Pipeline
After=network.target docker.service
Wants=docker.service

[Service]
Type=oneshot
User={svc_user}
Group={svc_user}
ExecStartPre=/bin/sleep 60
ExecStart={PYTHON} {PROJECT_DIR}/startup_trigger.py
WorkingDirectory={PROJECT_DIR}
EnvironmentFile={ENV_FILE}
StandardOutput=journal
StandardError=journal
RemainAfterExit=yes
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths={PROJECT_DIR}/collected_logs

[Install]
WantedBy=multi-user.target
"""
    service_path = Path("/etc/systemd/system/log-analysis.service")
    try:
        service_path.write_text(service_content)
        subprocess.run(["systemctl", "daemon-reload"], check=True)
        log.info(f"Service written to {service_path}")
        log.info("Run: sudo systemctl enable log-analysis.service")
    except PermissionError:
        log.error("Permission denied. Run with sudo.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--register-windows", action="store_true")
    parser.add_argument("--register-linux",   action="store_true")
    args = parser.parse_args()

    if args.register_windows:
        register_windows_task()
    elif args.register_linux:
        register_linux_service()
    else:
        main()
