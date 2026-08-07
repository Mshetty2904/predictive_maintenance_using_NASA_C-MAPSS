"""Population Stability Index implementation."""

import numpy as np


def _safe_bins(reference, bins=10):
    reference = np.asarray(reference, dtype=float)
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(reference, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        centre = float(np.mean(reference))
        spread = max(float(np.std(reference)), 1e-6)
        edges = np.linspace(centre - 3 * spread, centre + 3 * spread, bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def population_stability_index(reference, current, bins=10, epsilon=1e-6):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = _safe_bins(reference, bins=bins)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = np.maximum(ref_counts / max(len(reference), 1), epsilon)
    cur_pct = np.maximum(cur_counts / max(len(current), 1), epsilon)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
