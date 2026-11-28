"""Explainable statistical anomaly detection for one-dimensional samples."""

from __future__ import annotations

import numpy as np
import pandas as pd

from field_sensor_radar.models import AnomalyFinding


def detect_anomalies(
    values: pd.Series | np.ndarray | list[float],
    *,
    z_threshold: float = 3.0,
    window_size: int = 9,
    window_threshold: float = 3.0,
) -> tuple[AnomalyFinding, ...]:
    """Flag global Z-score and moving-window outliers with plain explanations."""

    # The default window is odd to center the local comparison on each sample.
    if z_threshold <= 0 or window_threshold <= 0:
        raise ValueError("Anomaly thresholds must be positive.")
    if window_size < 3:
        raise ValueError("window_size must be at least 3.")
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(numeric)
    if finite.sum() < 3:
        return ()

    clean = numeric[finite]
    mean = float(np.mean(clean))
    std = float(np.std(clean, ddof=0))
    z_scores = np.zeros_like(numeric)
    if std > 0:
        z_scores[finite] = np.abs((clean - mean) / std)
    z_flag = finite & (z_scores > z_threshold)

    series = pd.Series(numeric)
    minimum = max(3, window_size // 2)
    local_mean = series.rolling(window_size, center=True, min_periods=minimum).mean().to_numpy()
    local_std = series.rolling(window_size, center=True, min_periods=minimum).std(ddof=0).to_numpy()
    local_z = np.zeros_like(numeric)
    usable = finite & np.isfinite(local_mean) & np.isfinite(local_std) & (local_std > 0)
    local_z[usable] = np.abs((numeric[usable] - local_mean[usable]) / local_std[usable])
    window_flag = usable & (local_z > window_threshold)

    findings: list[AnomalyFinding] = []
    for index in np.flatnonzero(z_flag | window_flag):
        methods: list[str] = []
        phrases: list[str] = []
        if z_flag[index]:
            methods.append("z_score")
            phrases.append(f"global Z-score {z_scores[index]:.2f} exceeds {z_threshold:g}")
        if window_flag[index]:
            methods.append("moving_window")
            phrases.append(f"local deviation {local_z[index]:.2f} exceeds {window_threshold:g}")
        findings.append(
            AnomalyFinding(
                index=int(index),
                value=float(numeric[index]),
                methods=tuple(methods),
                explanation="; ".join(phrases),
            )
        )
    return tuple(findings)
