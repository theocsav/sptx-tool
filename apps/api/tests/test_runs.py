import os
import importlib
from pathlib import Path

from fastapi.testclient import TestClient


def create_client(tmp_path: Path) -> TestClient:
    os.environ["DB_PATH"] = str(tmp_path / "runs.db")
    os.environ["RUNS_DIR"] = str(tmp_path / "runs")
    os.environ["PRESETS_DIR"] = str(tmp_path / "presets")
    os.environ["ARTIFACT_ROOTS"] = str(tmp_path)
    os.environ["WORKER_ENABLED"] = "false"
    os.environ["QUEUE_ENABLED"] = "true"
    os.environ["SESSION_SECRET"] = "test-secret"
    os.environ["PREFLIGHT_CHECK_PATHS"] = "false"
    os.environ["BASIC_AUTH_USER"] = "test-user"
    os.environ["BASIC_AUTH_PASS"] = "test-pass"

    from app import settings, db, main, runner, worker, registry, validation
    importlib.reload(settings)
    importlib.reload(db)
    importlib.reload(runner)
    importlib.reload(worker)
    importlib.reload(registry)
    importlib.reload(validation)
    importlib.reload(main)

    return TestClient(main.app)


def test_create_run_queued(tmp_path: Path) -> None:
    client = create_client(tmp_path)
    with client:
        login = client.post("/auth/login", json={"username": "test-user", "password": "test-pass"})
        assert login.status_code == 200
        csrf_token = login.json().get("csrf_token")
        assert csrf_token

        output_dir = tmp_path / "outputs"
        data_path = tmp_path / "data.h5ad"
        ref_path = tmp_path / "ref.h5ad"
        meta_path = tmp_path / "meta.csv"

        payload = {
            "run_name": "test-run",
            "queue": True,
            "config": {
                "output_dir": str(output_dir),
                "cosmx_h5ad_path": str(data_path),
                "reference_h5ad_path": str(ref_path),
                "cell_metadata_path": str(meta_path),
                "n_components": 10,
            },
        }
        resp = client.post("/runs", json=payload, headers={"X-CSRF-Token": csrf_token or ""})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "queued"
        
