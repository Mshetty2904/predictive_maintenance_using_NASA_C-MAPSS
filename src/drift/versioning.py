"""Non-destructive local model versioning and validation comparison."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


class ModelVersionStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_candidate(self, dataset, model_name, model_path, metrics, is_better, source="local"):
        dataset_dir = self.root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        version_number = len(list(dataset_dir.glob("version_*"))) + 1
        version_dir = dataset_dir / f"version_{version_number:03d}"
        version_dir.mkdir()
        destination = version_dir / Path(model_path).name
        shutil.copy2(model_path, destination)
        record = {
            "dataset": dataset,
            "model": model_name,
            "version": version_number,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "source": source,
            "decision": "promote" if is_better else "discard",
        }
        (version_dir / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        if is_better:
            (dataset_dir / "current.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        return record

