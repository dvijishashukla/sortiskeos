"""
startup_trigger.py
------------------
Master startup script — runs everything in the correct order when the
machine restarts after a crash or shutdown.

Execution order:
  1. log_collector.py  → reads OS logs, writes staging file
  2. (wait for Logstash to ingest the staging file into Elasticsearch)
  3. ml_pipeline.py    → reads from ES, detects anomalies, writes results back

HOW TO REGISTER THIS AS A STARTUP TASK:
─────────────────────────────────────────
WINDOWS (Task Scheduler):
  1. Open Task Scheduler → Create Task
  2. Triggers tab → New → "At startup"  OR  "At log on"
  3. Actions tab → New → Program: python
     Arguments: C:\\path\\to\\startup_trigger.py
  4. Conditions → uncheck "Start only if on AC power"
  5. Settings → check "Run task as soon as possible after scheduled start is missed"

  Or run this script once to register automatically (requires admin):
      python startup_trigger.py --register-windows

LINUX (systemd service):
  Run this script once to install the systemd service:
      sudo python startup_trigger.py --register-linux

  Then enable it:
      sudo systemctl enable log-analysis.service
      sudo systemctl start log-analysis.service
"""

import os
import sys
import time
import logging
import platform
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

PROJECT_DIR   = Path(__file__).parent.resolve()
PYTHON        = sys.executable
LOGSTASH_WAIT = 30   # seconds to wait for Logstash to ingest the staging file


def run_step(script_name: str, args: list = []):
    """Run a Python script as a subprocess and wait for it to finish."""
    script_path = PROJECT_DIR / script_name
    cmd = [PYTHON, str(script_path)] + args
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_DIR))
    if result.returncode != 0:
        log.error(f"{script_name} exited with code {result.returncode}")
    return result.returncode == 0


def main():
    log.info("=" * 55)
    log.info("  Intelligent Log Analysis — Startup Pipeline")
    log.info("=" * 55)

    # Step 1: Collect OS logs around the shutdown event
    log.info("Step 1/3: Collecting system logs...")
    success = run_step("log_collector.py")
    if not success:
        log.error("Log collection failed. Aborting.")
        sys.exit(1)

    # Step 2: Wait for Logstash to pick up and ingest the staging file
    log.info(f"Step 2/3: Waiting {LOGSTASH_WAIT}s for Logstash to ingest logs...")
    time.sleep(LOGSTASH_WAIT)

    # Step 3: Run ML analysis on the ingested logs
    log.info("Step 3/3: Running ML anomaly detection...")
    run_step("ml_pipeline.py", ["--mode", "once"])

    log.info("Startup pipeline complete. Check Kibana for results.")


# ══════════════════════════════════════════════════════════════════════════════
#  REGISTRATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def register_windows_task():
    """Register this script as a Windows Task Scheduler startup task."""
    import subprocess
    script_path = PROJECT_DIR / "startup_trigger.py"
    task_name   = "IntelligentLogAnalysis"

    xml = f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Triggers>
    <BootTrigger>
      <Delay>PT1M</Delay>  <!-- wait 1 minute after boot so network/ES is ready -->
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
  </Settings>
</Task>"""

    xml_file = PROJECT_DIR / "task.xml"
    xml_file.write_text(xml, encoding="utf-16")

    result = subprocess.run(
        ["schtasks", "/Create", "/TN", task_name, "/XML", str(xml_file), "/F"],
        capture_output=True, text=True
    )
    xml_file.unlink()  # clean up temp file

    if result.returncode == 0:
        log.info(f"Windows Task '{task_name}' registered successfully.")
    else:
        log.error(f"Failed to register task: {result.stderr}")


def register_linux_service():
    """Install a systemd service that runs this script on every boot."""
    service_content = f"""[Unit]
Description=Intelligent Log Analysis Startup Pipeline
After=network.target docker.service
Wants=docker.service

[Service]
Type=oneshot
ExecStartPre=/bin/sleep 60
ExecStart={PYTHON} {PROJECT_DIR}/startup_trigger.py
WorkingDirectory={PROJECT_DIR}
StandardOutput=journal
StandardError=journal
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
"""
    service_path = Path("/etc/systemd/system/log-analysis.service")
    try:
        service_path.write_text(service_content)
        subprocess.run(["systemctl", "daemon-reload"], check=True)
        log.info(f"Service file written to {service_path}")
        log.info("Run: sudo systemctl enable log-analysis.service")
    except PermissionError:
        log.error("Permission denied. Run with sudo.")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--register-windows", action="store_true",
                        help="Register as Windows Task Scheduler startup task")
    parser.add_argument("--register-linux", action="store_true",
                        help="Install systemd service for Linux auto-start")
    args = parser.parse_args()

    if args.register_windows:
        register_windows_task()
    elif args.register_linux:
        register_linux_service()
    else:
        main()
