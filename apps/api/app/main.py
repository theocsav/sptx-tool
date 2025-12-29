import json
import shutil
import subprocess
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .auth import (
    authenticate,
    clear_session_cookie,
    create_session,
    ensure_csrf_cookie,
    require_csrf,
    require_session,
    set_session_cookie,
)
from .db import create_run, enqueue_run, fetch_run, init_db, list_runs, update_run
from .preflight_cache import build_cache_key, get_cached_join_result, set_cached_join_result
from .registry import dataset_manifest_hash, get_dataset, list_datasets, list_presets
from .runner import apply_dataset_registry, load_preset, prepare_run, resolve_run_paths
from .schemas import (
    LoginRequest,
    LoginResponse,
    PreflightRequest,
    PreflightResponse,
    RunCreate,
    RunRerun,
    RunResponse,
)
from .preflight_runner import run_slurm_preflight
from .settings import (
    ALLOWED_ORIGINS,
    ARTIFACT_ROOTS,
    PREFLIGHT_CHECK_PATHS,
    QUEUE_ENABLED,
    RUNS_DIR,
    DISK_WARN_FREE_GB,
    DISK_WARN_PERCENT,
    RUN_RETENTION_DAYS,
    PREFLIGHT_SLURM_FALLBACK,
    PREFLIGHT_CACHE_TTL_SECONDS,
    WORKER_ENABLED,
    validate_settings,
)
from .slurm import cancel_job
from .logging import configure_logging
from .worker import loop as worker_loop
from .storage import enforce_allowed_path, list_artifacts, safe_join
from .validation import validate_config


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    validate_settings()
    init_db()
    if WORKER_ENABLED:
        thread = threading.Thread(target=worker_loop, daemon=True)
        thread.start()
    yield


app = FastAPI(title="NicheRunner API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "disk": _disk_usage_report()}


@app.get("/health/hpg", dependencies=[Depends(require_session)])
def hpg_health() -> dict:
    checks = {
        "sbatch": _check_command("sbatch"),
        "squeue": _check_command("squeue"),
        "sacct": _check_command("sacct"),
        "runs_dir": _check_path(RUNS_DIR),
        "artifact_roots": _check_artifact_roots(),
    }
    ok = all(value is True for value in checks.values())
    return {"ok": ok, "checks": checks}


@app.post("/auth/login", response_model=LoginResponse)
def login(payload: LoginRequest, request: Request, response: Response) -> dict:
    if not authenticate(payload.username, payload.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_session(payload.username)
    set_session_cookie(response, token)
    csrf_token = ensure_csrf_cookie(request, response, rotate=True)
    return {"username": payload.username, "csrf_token": csrf_token}


@app.post("/auth/logout", dependencies=[Depends(require_session), Depends(require_csrf)])
def logout(response: Response) -> dict:
    clear_session_cookie(response)
    return {"status": "ok"}


@app.get("/auth/me")
def whoami(request: Request, response: Response, username: str = Depends(require_session)) -> dict:
    csrf_token = ensure_csrf_cookie(request, response)
    return {"username": username, "csrf_token": csrf_token}


@app.get("/auth/csrf", dependencies=[Depends(require_session)])
def csrf_token(request: Request, response: Response) -> dict:
    csrf_token_value = ensure_csrf_cookie(request, response)
    return {"csrf_token": csrf_token_value}


@app.get("/datasets", dependencies=[Depends(require_session)])
def get_datasets(
    organ: Optional[str] = Query(default=None),
    platform: Optional[str] = Query(default=None),
    preset_id: Optional[str] = Query(default=None),
) -> list[dict]:
    datasets = list_datasets()
    if preset_id:
        presets = list_presets()
        preset = next((item for item in presets if item.get("id") == preset_id), None)
        if preset:
            organ = organ or preset.get("organ")
            platform = platform or preset.get("platform")
    if organ:
        datasets = [item for item in datasets if item.get("organ") == organ]
    if platform:
        datasets = [item for item in datasets if item.get("platform") == platform]
    if preset_id:
        datasets = [
            item
            for item in datasets
            if item.get("recommended_preset") == preset_id
            or (not item.get("recommended_preset"))
        ]
    return datasets


@app.get("/presets", dependencies=[Depends(require_session)])
def get_presets(
    organ: Optional[str] = Query(default=None),
    platform: Optional[str] = Query(default=None),
) -> list[dict]:
    presets = list_presets()
    if organ:
        presets = [item for item in presets if item.get("organ") == organ]
    if platform:
        presets = [item for item in presets if item.get("platform") == platform]
    return presets


@app.get("/runs", response_model=list[RunResponse], dependencies=[Depends(require_session)])
def get_runs() -> list[dict]:
    return list_runs()


@app.get("/runs/{run_id}", response_model=RunResponse, dependencies=[Depends(require_session)])
def get_run(run_id: int) -> dict:
    run = fetch_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.post(
    "/runs/preflight",
    response_model=PreflightResponse,
    dependencies=[Depends(require_session), Depends(require_csrf)],
)
def preflight(payload: PreflightRequest) -> dict:
    if not payload.config and not payload.preset_path:
        raise HTTPException(status_code=400, detail="Provide preset_path or config")

    config = {}
    if payload.preset_path:
        config = load_preset(payload.preset_path)
    if payload.config:
        config.update(payload.config)

    try:
        config = apply_dataset_registry(config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    dataset_id = config.get("dataset_id")
    preset_id = config.get("preset_id") or config.get("id")
    if payload.preset_path:
        preset_id = Path(payload.preset_path).stem

    cached_join = None
    cache_key = None
    if dataset_id:
        dataset = get_dataset(dataset_id)
        if dataset:
            manifest_hash = dataset_manifest_hash(dataset)
            cache_key = build_cache_key(dataset_id, manifest_hash, preset_id)
            cached_join = get_cached_join_result(cache_key)

    errors, warnings, checks = validate_config(
        config,
        check_paths=payload.check_paths,
        allow_join_fallback=PREFLIGHT_SLURM_FALLBACK,
        join_key_result=cached_join,
    )
    join_status = checks.get("join_keys", {}).get("status")
    if join_status == "missing_deps" and PREFLIGHT_SLURM_FALLBACK and payload.check_paths:
        slurm_result = run_slurm_preflight(config)
        if slurm_result.get("ok"):
            errors, warnings, checks = validate_config(
                config,
                check_paths=payload.check_paths,
                allow_join_fallback=False,
                join_key_result=slurm_result["result"],
            )
        else:
            errors.append(slurm_result.get("error", "Preflight SLURM fallback failed."))

    join_result = checks.get("join_keys", {})
    if cache_key and join_result and "matched" in join_result:
        set_cached_join_result(cache_key, join_result, PREFLIGHT_CACHE_TTL_SECONDS)

    return {"ok": not errors, "errors": errors, "warnings": warnings, "checks": checks}


def _create_run_from_config(run_name: str, config: dict, submit: bool, queue: bool) -> dict:
    run_id = create_run(run_name, status="queued" if (queue or (submit and QUEUE_ENABLED)) else "created")
    try:
        run_dir, output_dir, resolved_config = resolve_run_paths(run_name, config)
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.json"
        config_path.write_text(json.dumps(resolved_config, indent=2), encoding="utf-8")
        update_run(
            run_id,
            run_dir=str(run_dir),
            output_dir=str(output_dir),
            config_path=str(config_path),
        )

        errors, warnings, _checks = validate_config(
            resolved_config,
            check_paths=PREFLIGHT_CHECK_PATHS,
            allow_join_fallback=False,
        )
        if errors:
            raise HTTPException(status_code=400, detail={"errors": errors, "warnings": warnings})

        if queue or (submit and QUEUE_ENABLED):
            enqueue_run(run_id, submit=submit)
            update_run(run_id, status="queued")
        else:
            run_dir_str, output_dir_str, config_path_str, job_id = prepare_run(run_name, config, submit)
            status = "submitted" if submit else "prepared"
            update_run(
                run_id,
                status=status,
                run_dir=run_dir_str,
                output_dir=output_dir_str,
                config_path=config_path_str,
                job_id=job_id,
            )
    except HTTPException as exc:
        message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        update_run(run_id, status="error", message=message)
        raise
    except Exception as exc:
        update_run(run_id, status="error", message=str(exc))
        raise HTTPException(status_code=500, detail="Run creation failed") from exc

    run = fetch_run(run_id)
    if not run:
        raise HTTPException(status_code=500, detail="Run creation failed")
    return run


@app.post("/runs", response_model=RunResponse, dependencies=[Depends(require_session), Depends(require_csrf)])
def create_run_endpoint(payload: RunCreate) -> dict:
    if not payload.config and not payload.preset_path:
        raise HTTPException(status_code=400, detail="Provide preset_path or config")

    config = {}
    if payload.preset_path:
        config = load_preset(payload.preset_path)
    if payload.config:
        config.update(payload.config)

    try:
        config = apply_dataset_registry(config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    run_name = payload.run_name or config.get("run_name")
    if not run_name:
        raise HTTPException(status_code=400, detail="run_name is required")

    return _create_run_from_config(run_name, config, payload.submit, payload.queue)


def _strip_runtime_keys(config: dict) -> dict:
    remove_keys = {
        "run_dir",
        "config_path",
        "patched_script",
        "run_script",
        "report_script",
        "job_id",
    }
    return {key: value for key, value in config.items() if key not in remove_keys}


@app.post("/runs/{run_id}/rerun", response_model=RunResponse, dependencies=[Depends(require_session), Depends(require_csrf)])
def rerun_run(run_id: int, payload: RunRerun) -> dict:
    run = fetch_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    config_path = run.get("config_path")
    if not config_path:
        raise HTTPException(status_code=400, detail="Run has no config_path")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    config = _strip_runtime_keys(config)
    run_dir = run.get("run_dir")
    output_dir = run.get("output_dir")
    if run_dir and output_dir:
        try:
            run_dir_path = Path(run_dir).resolve()
            output_dir_path = Path(output_dir).resolve()
            if run_dir_path == output_dir_path or run_dir_path in output_dir_path.parents:
                config.pop("output_dir", None)
        except OSError:
            pass
    config["run_name"] = payload.run_name
    report_title = config.get("report_title")
    if isinstance(report_title, str) and report_title.startswith("NicheRunner "):
        config["report_title"] = f"NicheRunner {payload.run_name}"
    try:
        config = apply_dataset_registry(config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _create_run_from_config(payload.run_name, config, payload.submit, payload.queue)


def read_tail(path: Path, max_bytes: int = 65536) -> str:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        seek = max(size - max_bytes, 0)
        handle.seek(seek)
        data = handle.read()
    return data.decode("utf-8", errors="replace")


@app.get("/runs/{run_id}/logs", dependencies=[Depends(require_session)])
def get_logs(run_id: int, path: Optional[str] = Query(default=None)) -> dict:
    run = fetch_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    output_dir = run.get("output_dir")
    if not output_dir:
        raise HTTPException(status_code=400, detail="Run has no output_dir")

    output_path = Path(output_dir)
    try:
        enforce_allowed_path(output_path, ARTIFACT_ROOTS)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if path:
        try:
            log_path = safe_join(output_path, path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid log path") from exc
    else:
        candidates = list(output_path.glob("*.out")) + list(output_path.glob("*.err"))
        if not candidates:
            raise HTTPException(status_code=404, detail="No log files found")
        log_path = max(candidates, key=lambda p: p.stat().st_mtime)

    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log file not found")

    return {"path": str(log_path), "content": read_tail(log_path)}


@app.get("/runs/{run_id}/artifacts", dependencies=[Depends(require_session)])
def get_artifacts(run_id: int, path: Optional[str] = Query(default="")) -> dict:
    run = fetch_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    output_dir = run.get("output_dir")
    if not output_dir:
        raise HTTPException(status_code=400, detail="Run has no output_dir")

    base = Path(output_dir)
    try:
        enforce_allowed_path(base, ARTIFACT_ROOTS)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {"items": list_artifacts(base, path)}


@app.get("/runs/{run_id}/artifact", dependencies=[Depends(require_session)])
def get_artifact(run_id: int, path: str = Query(...)) -> FileResponse:
    run = fetch_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    output_dir = run.get("output_dir")
    if not output_dir:
        raise HTTPException(status_code=400, detail="Run has no output_dir")

    base = Path(output_dir)
    try:
        enforce_allowed_path(base, ARTIFACT_ROOTS)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    try:
        file_path = safe_join(base, path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid artifact path") from exc
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")

    return FileResponse(path=str(file_path), filename=file_path.name)


@app.post("/runs/{run_id}/cancel", dependencies=[Depends(require_session), Depends(require_csrf)])
def cancel_run(run_id: int) -> dict:
    run = fetch_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    job_id = run.get("job_id")
    if job_id and cancel_job(job_id):
        update_run(run_id, status="canceled", message="Canceled via scancel")
        return {"status": "canceled"}
    update_run(run_id, status="canceled", message="Canceled without job_id")
    return {"status": "canceled"}


def _check_command(name: str) -> bool:
    try:
        result = subprocess.run([name, "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def _check_path(path: Path) -> bool:
    return path.exists()


def _check_artifact_roots() -> bool:
    return all(root.exists() for root in ARTIFACT_ROOTS)


def _disk_usage_report() -> dict:
    roots = [RUNS_DIR] + ARTIFACT_ROOTS
    seen = set()
    items = []
    warnings = []
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        if not root.exists():
            items.append({"path": str(root), "exists": False})
            continue
        usage = shutil.disk_usage(root)
        total_gb = round(usage.total / (1024**3), 1)
        free_gb = round(usage.free / (1024**3), 1)
        percent_free = round((usage.free / usage.total) * 100, 1) if usage.total else 0.0
        warning = free_gb < DISK_WARN_FREE_GB or percent_free < DISK_WARN_PERCENT
        if warning:
            warnings.append(str(root))
        items.append(
            {
                "path": str(root),
                "exists": True,
                "total_gb": total_gb,
                "free_gb": free_gb,
                "percent_free": percent_free,
                "warning": warning,
            }
        )
    return {
        "roots": items,
        "warnings": warnings,
        "thresholds": {"free_gb": DISK_WARN_FREE_GB, "percent_free": DISK_WARN_PERCENT},
        "retention_days": RUN_RETENTION_DAYS,
    }
