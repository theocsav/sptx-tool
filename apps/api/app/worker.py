import json
import logging
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .db import fetch_run, list_runs, mark_run_running, next_queued_run, remove_from_queue, update_run
from .runner import prepare_run
from .settings import (
    CLEANUP_INTERVAL_SECONDS,
    RUN_RETENTION_DAYS,
    RUNS_DIR,
    WORKER_POLL_SECONDS,
    validate_settings,
)
from .slurm import get_job_state
from .storage import enforce_allowed_path

logger = logging.getLogger("sptx.worker")
TERMINAL_STATUSES = {"succeeded", "failed", "canceled", "error"}


def map_slurm_state(state: str) -> str:
    state = state.upper()
    if state in {"PENDING", "CONFIGURING"}:
        return "queued"
    if state in {"RUNNING", "COMPLETING"}:
        return "running"
    if state in {"COMPLETED"}:
        return "succeeded"
    if state in {"CANCELLED", "CANCELLED+"}:
        return "canceled"
    if state in {"FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}:
        return "failed"
    return "unknown"


def loop() -> None:
    last_cleanup = 0.0
    while True:
        try:
            queued = next_queued_run()
            if queued:
                run_id = queued["run_id"]
                submit = bool(queued["submit"])
                mark_run_running(run_id)
                run = fetch_run(run_id)
                if run:
                    try:
                        config_path = run.get("config_path")
                        config = {}
                        if config_path:
                            config = json.loads(Path(config_path).read_text(encoding="utf-8"))
                        run_dir, output_dir, config_path_str, job_id = prepare_run(run["run_name"], config, submit)
                        status = "submitted" if job_id else "prepared"
                        update_run(
                            run_id,
                            status=status,
                            run_dir=run_dir,
                            output_dir=output_dir,
                            config_path=config_path_str,
                            job_id=job_id,
                        )
                    except Exception as exc:
                        update_run(run_id, status="error", message=str(exc))
                remove_from_queue(run_id)

            for run in list_runs():
                job_id = run.get("job_id")
                if not job_id:
                    continue
                if run.get("status") not in {"submitted", "running", "queued"}:
                    continue
                state = get_job_state(job_id)
                if state:
                    mapped = map_slurm_state(state)
                    update_run(run["id"], status=mapped)
            now = time.time()
            if RUN_RETENTION_DAYS > 0 and now - last_cleanup >= CLEANUP_INTERVAL_SECONDS:
                cleanup_runs()
                last_cleanup = now
        except Exception as exc:
            logger.exception("Worker loop error: %s", exc)
        time.sleep(WORKER_POLL_SECONDS)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    validate_settings()
    loop()


def cleanup_runs() -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=RUN_RETENTION_DAYS)
    for run in list_runs():
        status = run.get("status")
        if status not in TERMINAL_STATUSES:
            continue
        timestamp = _parse_timestamp(run.get("updated_at") or run.get("created_at"))
        if not timestamp or timestamp > cutoff:
            continue
        run_dir = run.get("run_dir")
        if not run_dir:
            continue
        run_path = Path(run_dir)
        try:
            enforce_allowed_path(run_path, [RUNS_DIR])
        except ValueError:
            logger.warning("Skipping cleanup for run %s: run_dir outside RUNS_DIR", run.get("id"))
            continue
        output_dir = run.get("output_dir")
        if output_dir:
            try:
                output_path = Path(output_dir).resolve()
                run_resolved = run_path.resolve()
                if run_resolved == output_path or run_resolved in output_path.parents:
                    logger.info(
                        "Skipping cleanup for run %s: output_dir inside run_dir; move outputs elsewhere to enable cleanup",
                        run.get("id"),
                    )
                    continue
            except Exception as exc:
                logger.warning("Skipping cleanup for run %s: could not resolve output_dir: %s", run.get("id"), exc)
                continue
        if run_path.exists():
            shutil.rmtree(run_path, ignore_errors=True)
            logger.info("Cleaned staging for run %s at %s", run.get("id"), run_path)


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


if __name__ == "__main__":
    main()
