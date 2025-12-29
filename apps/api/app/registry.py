import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .settings import DATASETS_REGISTRY_PATH, PRESETS_DIR


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def list_datasets() -> List[Dict[str, Any]]:
    if not DATASETS_REGISTRY_PATH.exists():
        return []
    data = _load_json(DATASETS_REGISTRY_PATH)
    if isinstance(data, dict):
        data = data.get("datasets", [])
    if not isinstance(data, list):
        return []
    return data


def get_dataset(dataset_id: str) -> Optional[Dict[str, Any]]:
    for dataset in list_datasets():
        if dataset.get("id") == dataset_id:
            return dataset
    return None


def list_presets() -> List[Dict[str, Any]]:
    presets: List[Dict[str, Any]] = []
    if not PRESETS_DIR.exists():
        return presets
    for path in sorted(PRESETS_DIR.glob("*.json")):
        try:
            preset = _load_json(path)
        except json.JSONDecodeError:
            continue
        if not isinstance(preset, dict):
            continue
        preset.setdefault("id", path.stem)
        preset["path"] = path.as_posix()
        presets.append(preset)
    return presets


def dataset_manifest_hash(dataset: Dict[str, Any]) -> str:
    payload = {
        "schema_manifest": dataset.get("schema_manifest", {}),
        "metadata_columns": dataset.get("metadata_columns", []),
        "staged_path": dataset.get("staged_path"),
        "cell_metadata_path": dataset.get("cell_metadata_path"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
