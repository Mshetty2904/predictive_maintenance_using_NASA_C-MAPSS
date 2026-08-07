"""Decision-only retraining trigger and RMSE promotion rule."""

import json
from pathlib import Path

from config import (
    MIN_CONSECUTIVE_DRIFT_BATCHES,
    MIN_CONSECUTIVE_STABLE_BATCHES,
    MIN_DRIFTED_SENSORS_PER_BATCH,
)


def retraining_decision(drift_rows):
    """Return a trigger decision without starting training."""
    drifted = any(row.get("status") == "drift" for row in drift_rows)
    return {
        "drift_detected": drifted,
        "action": "trigger_retraining" if drifted else "continue_prediction",
        "drifted_sensors": sorted({row["sensor"] for row in drift_rows if row.get("status") == "drift"}),
    }


class ConsecutiveDriftGate:
    """Require consecutive drifted batches before requesting retraining."""

    def __init__(
        self,
        required_batches=MIN_CONSECUTIVE_DRIFT_BATCHES,
        required_sensors=MIN_DRIFTED_SENSORS_PER_BATCH,
        stable_reset_batches=MIN_CONSECUTIVE_STABLE_BATCHES,
    ):
        self.required_batches = int(required_batches)
        self.required_sensors = int(required_sensors)
        self.stable_reset_batches = int(stable_reset_batches)
        self.count = 0
        self.stable_count = 0
        self.triggered = False

    def update(self, drift_rows):
        drifted_sensors = sorted({row["sensor"] for row in drift_rows if row.get("status") == "drift"})
        batch_drifted = len(drifted_sensors) >= self.required_sensors
        self.count = self.count + 1 if batch_drifted else 0
        self.stable_count = 0 if batch_drifted else self.stable_count + 1
        if self.triggered and self.stable_count >= self.stable_reset_batches:
            self.triggered = False
            self.count = 0
        triggered = self.count >= self.required_batches and not self.triggered
        if triggered:
            self.triggered = True
        return {
            "drift_detected": batch_drifted,
            "drifted_sensor_count": len(drifted_sensors),
            "required_drifted_sensors": self.required_sensors,
            "consecutive_drift_batches": self.count,
            "required_consecutive_batches": self.required_batches,
            "stable_batches_to_reset": self.stable_reset_batches,
            "retraining_episode_active": self.triggered,
            "action": (
                "trigger_retraining" if triggered
                else "retraining_pending" if self.triggered and batch_drifted
                else "continue_prediction"
            ),
            "drifted_sensors": drifted_sensors,
        }


def compare_rmse(current_rmse, candidate_rmse):
    """Promote only a candidate with strictly lower RMSE."""
    return float(candidate_rmse) < float(current_rmse)


def save_trigger(record, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
