"""Reference statistics and per-batch PSI/KS decisions."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import (
    DRIFT_CALIBRATION_QUANTILE,
    DRIFT_MIN_BATCH_ROWS,
    DRIFT_REQUIRE_BOTH_TESTS,
    KS_PVALUE,
    PSI_THRESHOLD,
)
from .ks_test import ks_test
from .psi import population_stability_index


class DriftDetector:
    def __init__(
        self,
        reference_path,
        psi_threshold=PSI_THRESHOLD,
        ks_pvalue=KS_PVALUE,
        calibration_quantile=DRIFT_CALIBRATION_QUANTILE,
        require_both=DRIFT_REQUIRE_BOTH_TESTS,
    ):
        self.reference_path = Path(reference_path)
        self.psi_threshold = float(psi_threshold)
        self.ks_pvalue = float(ks_pvalue)
        self.calibration_quantile = float(calibration_quantile)
        self.require_both = bool(require_both)
        self.reference = None
        if self.reference_path.exists():
            self.reference = json.loads(self.reference_path.read_text(encoding="utf-8"))

    def fit_reference(self, frame, sensor_columns, engine_column="Engine_ID", calibrate=True):
        self.reference = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sensors": {
                column: {
                    "values": frame[column].dropna().astype(float).tolist(),
                    "mean": float(frame[column].mean()),
                    "std": float(frame[column].std(ddof=0)),
                }
                for column in sensor_columns
                if column in frame
            },
            "thresholds": {},
        }
        if calibrate and engine_column in frame:
            calibration = {sensor: {"psi": [], "ks_pvalue": []} for sensor in sensor_columns}
            for _, batch in frame.groupby(engine_column, sort=True):
                for sensor in sensor_columns:
                    if sensor not in batch or sensor not in self.reference["sensors"]:
                        continue
                    current = batch[sensor].dropna().astype(float).to_numpy()
                    if len(current) < DRIFT_MIN_BATCH_ROWS:
                        continue
                    reference = np.asarray(self.reference["sensors"][sensor]["values"], dtype=float)
                    calibration[sensor]["psi"].append(population_stability_index(reference, current))
                    calibration[sensor]["ks_pvalue"].append(ks_test(reference, current)[1])
            for sensor, values in calibration.items():
                if not values["psi"]:
                    self.reference["thresholds"][sensor] = {
                        "psi": self.psi_threshold,
                        "ks_pvalue": self.ks_pvalue,
                    }
                    continue
                # The upper PSI and lower KS-p-value tails define the normal
                # operating envelope. Defaults prevent thresholds becoming
                # more permissive than the configured research thresholds.
                self.reference["thresholds"][sensor] = {
                    "psi": max(self.psi_threshold, float(np.quantile(values["psi"], self.calibration_quantile))),
                    "ks_pvalue": min(self.ks_pvalue, float(np.quantile(values["ks_pvalue"], 1.0 - self.calibration_quantile))),
                }
        self.reference_path.parent.mkdir(parents=True, exist_ok=True)
        self.reference_path.write_text(json.dumps(self.reference), encoding="utf-8")
        return self.reference

    def detect(self, batch, sensor_columns, engine_id=None, severity="normal", batch_id=None):
        if not self.reference:
            raise RuntimeError("Reference statistics are not fitted or loaded")
        timestamp = datetime.now(timezone.utc).isoformat()
        rows = []
        for sensor in sensor_columns:
            if sensor not in batch or sensor not in self.reference["sensors"]:
                continue
            reference = np.asarray(self.reference["sensors"][sensor]["values"], dtype=float)
            current = batch[sensor].dropna().astype(float).to_numpy()
            if len(current) < DRIFT_MIN_BATCH_ROWS:
                continue
            psi = population_stability_index(reference, current)
            ks_stat, pvalue = ks_test(reference, current)
            threshold = self.reference.get("thresholds", {}).get(sensor, {})
            psi_threshold = float(threshold.get("psi", self.psi_threshold))
            ks_threshold = float(threshold.get("ks_pvalue", self.ks_pvalue))
            psi_flag = psi > psi_threshold
            ks_flag = pvalue < ks_threshold
            drift = bool(psi_flag and ks_flag) if self.require_both else bool(psi_flag or ks_flag)
            rows.append({
                "timestamp": timestamp,
                "batch_id": batch_id,
                "engine_id": engine_id,
                "severity": severity,
                "sensor": sensor,
                "psi": psi,
                "ks_statistic": ks_stat,
                "ks_pvalue": pvalue,
                "status": "drift" if drift else "stable",
                "psi_threshold": psi_threshold,
                "ks_pvalue_threshold": ks_threshold,
                "psi_flag": psi_flag,
                "ks_flag": ks_flag,
                "decision_rule": "AND" if self.require_both else "OR",
                "sample_size": len(current),
                "small_batch": len(current) < 20,
            })
        return rows
