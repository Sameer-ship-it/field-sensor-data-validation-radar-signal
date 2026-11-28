"""Sensor record quality checks and deterministic quality classification."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd

from field_sensor_radar.models import QualityReport, QualityStatus, SensorDataset, ValidationIssue

LOGGER = logging.getLogger(__name__)


def _issue(code: str, severity: str, message: str, indices: Iterable[int]) -> ValidationIssue | None:
    rows = tuple(sorted(set(int(index) for index in indices)))
    if not rows:
        return None
    return ValidationIssue(code=code, severity=severity, message=message, count=len(rows), row_indices=rows)


def validate_dataset(
    dataset: SensorDataset,
    *,
    sampling_tolerance: float = 0.20,
    robust_z_threshold: float = 3.5,
) -> QualityReport:
    """Validate record completeness, timing, GPS range, and sensor outliers.

    Score deductions are 40 points per fraction of rows with a failing issue and
    10 points per fraction of rows with a warning issue. Status is FAIL when any
    failing issue exists or the score is below 60; WARNING when any warning
    exists or the score is below 90; otherwise PASS.
    """

    if sampling_tolerance < 0 or robust_z_threshold <= 0:
        raise ValueError("Validation thresholds must be non-negative; Z threshold must be positive.")
    frame = dataset.frame
    size = len(frame)
    issues: list[ValidationIssue] = []
    invalid_cells = dataset.invalid_cells

    for column in frame.columns:
        missing_rows = np.flatnonzero(frame[column].isna().to_numpy())
        if column in invalid_cells:
            invalid_set = set(invalid_cells[column])
            missing_rows = np.asarray([row for row in missing_rows if int(row) not in invalid_set])
            invalid_issue = _issue(
                f"invalid_{column}",
                "fail" if column in {"timestamp", "reading"} else "warning",
                f"{len(invalid_cells[column])} non-empty {column} value(s) could not be parsed.",
                invalid_cells[column],
            )
            if invalid_issue:
                issues.append(invalid_issue)
        missing_issue = _issue(
            f"missing_{column}",
            "fail" if column in {"timestamp", "reading", "sensor_id"} else "warning",
            f"{len(missing_rows)} record(s) have a missing {column} value.",
            missing_rows,
        )
        if missing_issue:
            issues.append(missing_issue)

    duplicate_rows: set[int] = set()
    valid_time = frame["timestamp"].notna() & frame["sensor_id"].notna()
    duplicate_mask = frame.loc[valid_time].duplicated(subset=["sensor_id", "timestamp"], keep=False)
    duplicate_rows.update(int(row) for row in frame.loc[valid_time].index[duplicate_mask])
    duplicate_issue = _issue(
        "duplicate_timestamp", "warning",
        f"{len(duplicate_rows)} record(s) share a timestamp within the same sensor.",
        duplicate_rows,
    )
    if duplicate_issue:
        issues.append(duplicate_issue)

    irregular_rows: set[int] = set()
    valid_timed = frame.loc[valid_time].sort_values(["sensor_id", "timestamp"])
    for _, group in valid_timed.groupby("sensor_id", dropna=True):
        if len(group) < 4:
            continue
        deltas = group["timestamp"].diff().dt.total_seconds().to_numpy()
        positive = deltas[np.isfinite(deltas) & (deltas > 0)]
        if len(positive) < 3:
            continue
        baseline = float(np.median(positive))
        if baseline <= 0:
            continue
        for local_position, delta in enumerate(deltas):
            if local_position == 0 or not np.isfinite(delta) or delta <= 0:
                continue
            if abs(float(delta) - baseline) / baseline > sampling_tolerance:
                irregular_rows.add(int(group.index[local_position]))
    irregular_issue = _issue(
        "irregular_sampling", "warning",
        f"{len(irregular_rows)} record(s) follow a sampling interval outside the configured tolerance.",
        irregular_rows,
    )
    if irregular_issue:
        issues.append(irregular_issue)

    latitude = pd.to_numeric(frame["latitude"], errors="coerce")
    longitude = pd.to_numeric(frame["longitude"], errors="coerce")
    gps_invalid = np.flatnonzero(
        (latitude.notna() & ~latitude.between(-90, 90) | longitude.notna() & ~longitude.between(-180, 180)).to_numpy()
    )
    gps_issue = _issue(
        "invalid_gps", "fail",
        f"{len(gps_invalid)} record(s) contain latitude outside [-90, 90] or longitude outside [-180, 180].",
        gps_invalid,
    )
    if gps_issue:
        issues.append(gps_issue)

    abnormal_rows: set[int] = set()
    readings = pd.to_numeric(frame["reading"], errors="coerce")
    for _, group in frame.assign(_reading=readings).groupby("sensor_id", dropna=True):
        values = group["_reading"].dropna()
        if len(values) < 5:
            continue
        median = float(values.median())
        mad = float(np.median(np.abs(values.to_numpy(dtype=float) - median)))
        if mad <= 0:
            continue
        robust_scores = 0.6745 * (values.to_numpy(dtype=float) - median) / mad
        abnormal_rows.update(int(row) for row in values.index[np.abs(robust_scores) > robust_z_threshold])
    abnormal_issue = _issue(
        "abnormal_reading", "warning",
        f"{len(abnormal_rows)} reading(s) exceed the robust Z-score threshold {robust_z_threshold:g}.",
        abnormal_rows,
    )
    if abnormal_issue:
        issues.append(abnormal_issue)

    fail_rows = {row for item in issues if item.severity == "fail" for row in item.row_indices}
    warning_rows = {row for item in issues if item.severity == "warning" for row in item.row_indices}
    score = 100.0
    if size:
        score -= 40.0 * len(fail_rows) / size
        score -= 10.0 * len(warning_rows - fail_rows) / size
    score = round(max(0.0, min(100.0, score)), 1)
    if any(item.severity == "fail" for item in issues) or score < 60:
        status = QualityStatus.FAIL
    elif issues or score < 90:
        status = QualityStatus.WARNING
    else:
        status = QualityStatus.PASS

    metrics = {
        "missing_values": int(sum(item.count for item in issues if item.code.startswith("missing_"))),
        "duplicate_timestamps": len(duplicate_rows),
        "irregular_sampling_rows": len(irregular_rows),
        "invalid_gps_rows": len(gps_invalid),
        "abnormal_readings": len(abnormal_rows),
        "unparseable_values": int(sum(item.count for item in issues if item.code.startswith("invalid_"))),
    }
    report = QualityReport(
        source_name=dataset.source_name,
        records_checked=size,
        score=score,
        status=status,
        metrics=metrics,
        issues=tuple(issues),
    )
    LOGGER.info("Validated %s: %s, quality score %.1f", dataset.source_name, status.value, score)
    return report
