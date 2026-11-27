"""Typed data structures shared by the application layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd


class QualityStatus(str, Enum):
    """Overall data quality classification."""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass(frozen=True)
class SensorDataset:
    """Normalized tabular sensor data and non-fatal ingestion observations."""

    frame: pd.DataFrame
    source_name: str
    source_format: str
    metadata: dict[str, Any] = field(default_factory=dict)
    invalid_cells: dict[str, tuple[int, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationIssue:
    """One explainable data-quality finding; row indices are zero-based."""

    code: str
    severity: str
    message: str
    count: int
    row_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class QualityReport:
    """Dataset-level quality classification and issue summary."""

    source_name: str
    records_checked: int
    score: float
    status: QualityStatus
    metrics: dict[str, int | float]
    issues: tuple[ValidationIssue, ...]


@dataclass(frozen=True)
class AnomalyFinding:
    """An anomalous sample and the detectors that flagged it."""

    index: int
    value: float
    methods: tuple[str, ...]
    explanation: str


@dataclass(frozen=True)
class SignalAnalysis:
    """Scalar signal metrics plus a one-sided FFT spectrum."""

    sample_count: int
    sample_rate_hz: float
    mean: float
    rms: float
    peak_to_peak: float
    dominant_frequency_hz: float | None
    dominant_phase_rad: float | None
    noise_floor_amplitude: float | None
    peak_to_noise_floor_db: float | None
    frequencies_hz: tuple[float, ...]
    amplitudes: tuple[float, ...]

    def summary(self) -> dict[str, int | float | None]:
        """Return JSON-friendly metrics without the full spectrum arrays."""

        return {
            "sample_count": self.sample_count,
            "sample_rate_hz": self.sample_rate_hz,
            "mean": self.mean,
            "rms": self.rms,
            "peak_to_peak": self.peak_to_peak,
            "dominant_frequency_hz": self.dominant_frequency_hz,
            "dominant_phase_rad": self.dominant_phase_rad,
            "noise_floor_amplitude": self.noise_floor_amplitude,
            "peak_to_noise_floor_db": self.peak_to_noise_floor_db,
        }
