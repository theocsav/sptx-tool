import subprocess
from typing import Optional


def _run_command(cmd: list[str]) -> Optional[str]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def get_job_state(job_id: str) -> Optional[str]:
    sacct = _run_command(["sacct", "-j", job_id, "--format=State", "-n", "-P"])
    if sacct:
        return sacct.split("|", 1)[0].strip()
    squeue = _run_command(["squeue", "-j", job_id, "-h", "-o", "%T"])
    if squeue:
        return squeue.splitlines()[0].strip()
    return None


def cancel_job(job_id: str) -> bool:
    try:
        result = subprocess.run(["scancel", job_id], capture_output=True, text=True)
    except FileNotFoundError:
        return False
    return result.returncode == 0
