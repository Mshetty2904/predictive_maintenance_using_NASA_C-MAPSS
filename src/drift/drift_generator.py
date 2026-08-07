"""Controlled, reproducible drift injection for dissertation experiments."""

import numpy as np


class DriftGenerator:
    SEVERITY_MULTIPLIERS = {
        "normal": 1.00,
        "small": 1.05,
        "medium": 1.15,
        "severe": 1.30,
    }

    def __init__(self, severity="normal", noise_std=0.0, random_state=42):
        severity = severity.lower()
        if severity not in self.SEVERITY_MULTIPLIERS:
            raise ValueError(f"Unknown severity: {severity}")
        self.severity = severity
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(random_state)

    def transform(self, batch, sensor_columns):
        result = batch.copy()
        multiplier = self.SEVERITY_MULTIPLIERS[self.severity]
        for column in sensor_columns:
            if column not in result:
                continue
            values = result[column].to_numpy(dtype=float)
            # Multiplicative drift is applied to the deviation from the
            # reference centre so zero-centred normalized sensors remain
            # interpretable. Gaussian noise is additive in sensor units.
            shifted = values * multiplier
            if self.noise_std:
                shifted = shifted + self.rng.normal(0.0, self.noise_std, len(values))
            result[column] = shifted
        return result
