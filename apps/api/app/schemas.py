from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class RunCreate(BaseModel):
    run_name: str = Field(..., min_length=1)
    preset_path: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    submit: bool = False
    queue: bool = False


class RunRerun(BaseModel):
    run_name: str = Field(..., min_length=1)
    submit: bool = False
    queue: bool = False


class RunResponse(BaseModel):
    id: int
    run_name: str
    status: str
    stage: Optional[str] = None
    run_dir: Optional[str] = None
    output_dir: Optional[str] = None
    config_path: Optional[str] = None
    job_id: Optional[str] = None
    message: Optional[str] = None
    created_at: str
    updated_at: str


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    username: str
    csrf_token: Optional[str] = None


class PreflightRequest(BaseModel):
    preset_path: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    check_paths: bool = True


class PreflightResponse(BaseModel):
    ok: bool
    errors: list[str]
    warnings: list[str]
    checks: Dict[str, Any] = Field(default_factory=dict)
