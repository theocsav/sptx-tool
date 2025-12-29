import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .registry import get_dataset
from .settings import ARTIFACT_ROOTS, PRESETS_DIR, REPO_ROOT, RUNS_DIR
from .storage import enforce_allowed_path

PIPELINE_RUNNER = REPO_ROOT / "run_pipeline.py"


def load_preset(preset_path: str) -> Dict[str, Any]:
    preset_file = Path(preset_path)
    if not preset_file.is_absolute():
        preset_file = PRESETS_DIR / preset_file
    preset_file = preset_file.resolve()
    presets_root = PRESETS_DIR.resolve()
    if presets_root not in preset_file.parents and preset_file != presets_root:
        raise ValueError("Preset path must be inside the presets directory.")
    if not preset_file.exists():
        raise FileNotFoundError(f"Preset not found: {preset_file}")
    return json.loads(preset_file.read_text(encoding="utf-8"))


def apply_dataset_registry(config: Dict[str, Any]) -> Dict[str, Any]:
    dataset_id = config.get("dataset_id")
    if not dataset_id:
        return config
    dataset = get_dataset(dataset_id)
    if not dataset:
        raise ValueError(f"Dataset not found: {dataset_id}")

    mapping = {
        "staged_path": "cosmx_h5ad_path",
        "cosmx_h5ad_path": "cosmx_h5ad_path",
        "cell_metadata_path": "cell_metadata_path",
        "reference_h5ad_path": "reference_h5ad_path",
        "ref_model_dir": "ref_model_dir",
    }
    merged = dict(config)
    for source, dest in mapping.items():
        if dest not in merged and dataset.get(source):
            merged[dest] = dataset[source]
    if "dataset_label" not in merged and dataset.get("label"):
        merged["dataset_label"] = dataset["label"]
    return merged


def resolve_run_paths(run_name: str, config: Dict[str, Any]) -> tuple[Path, Path, Dict[str, Any]]:
    config = dict(config)
    config["run_name"] = run_name
    requested_run_dir = config.get("run_dir")
    run_dir = Path(requested_run_dir) if requested_run_dir else RUNS_DIR / run_name
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    if RUNS_DIR.resolve() not in run_dir.resolve().parents and run_dir.resolve() != RUNS_DIR.resolve():
        raise ValueError("run_dir must be inside RUNS_DIR")

    output_dir = config.get("output_dir")
    if not output_dir:
        output_dir = str(run_dir / "outputs")
        config["output_dir"] = output_dir
    enforce_allowed_path(Path(output_dir), ARTIFACT_ROOTS)
    return run_dir, Path(output_dir), config


def prepare_run(run_name: str, config: Dict[str, Any], submit: bool) -> Tuple[str, str, str, Optional[str]]:
    run_dir, output_dir, config = resolve_run_paths(run_name, config)
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    emit_sbatch = submit or bool(config.get("slurm", {}).get("enabled"))
    cmd = [sys.executable, str(PIPELINE_RUNNER), "--config", str(config_path)]
    if emit_sbatch:
        cmd.append("--emit-sbatch")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "Pipeline preparation failed")

    job_id = None
    if submit:
        submit_path = run_dir / "submit.sh"
        if not submit_path.exists():
            raise FileNotFoundError("submit.sh not found; use --emit-sbatch or enable slurm in config")
        submit_result = subprocess.run(["sbatch", str(submit_path)], capture_output=True, text=True)
        if submit_result.returncode != 0:
            raise RuntimeError(submit_result.stderr or submit_result.stdout or "sbatch failed")
        match = re.search(r"Submitted batch job (\d+)", submit_result.stdout)
        if match:
            job_id = match.group(1)

    return str(run_dir), str(output_dir), str(config_path), job_id
