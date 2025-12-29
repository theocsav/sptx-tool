import os
from pathlib import Path

from fastapi.testclient import TestClient


def create_client(tmp_path: Path) -> TestClient:
    os.environ["DB_PATH"] = str(tmp_path / "runs.db")
    os.environ["RUNS_DIR"] = str(tmp_path / "runs")
    os.environ["PRESETS_DIR"] = str(tmp_path / "presets")
    os.environ["ARTIFACT_ROOTS"] = str(tmp_path)
    os.environ["WORKER_ENABLED"] = "false"
    os.environ["QUEUE_ENABLED"] = "false"
    os.environ["SESSION_SECRET"] = "test-secret"
    os.environ["PREFLIGHT_CHECK_PATHS"] = "false"
    os.environ["BASIC_AUTH_USER"] = "test-user"
    os.environ["BASIC_AUTH_PASS"] = "test-pass"

    from app.main import app

    return TestClient(app)


def test_login_and_me(tmp_path: Path) -> None:
    client = create_client(tmp_path)
    resp = client.post("/auth/login", json={"username": "test-user", "password": "test-pass"})
    assert resp.status_code == 200
    assert resp.json()["username"] == "test-user"
    assert resp.json().get("csrf_token")
    assert "set-cookie" in resp.headers

    me = client.get("/auth/me")
    assert me.status_code == 200
    assert me.json()["username"] == "test-user"
