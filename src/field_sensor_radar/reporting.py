"""Portable JSON and CSV report construction and serialization."""

from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from field_sensor_radar.models import AnomalyFinding, QualityReport, SignalAnalysis, SensorDataset

LOGGER = logging.getLogger(__name__)


def build_technical_report(
    dataset: SensorDataset,
    quality: QualityReport,
    *,
    signal_analysis: SignalAnalysis | None = None,
    anomalies: tuple[AnomalyFinding, ...] = (),
) -> dict[str, Any]:
    """Build a JSON-ready technical summary with explicit simulation context."""

    return {
        "report_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_provenance": {
            "classification": dataset.metadata.get("classification", "unspecified; verify source before operational use"),
            "source_name": dataset.source_name,
            "source_format": dataset.source_format,
            "metadata": dataset.metadata,
        },
        "dataset": {
            "records": len(dataset.frame),
            "sensors": sorted(str(value) for value in dataset.frame["sensor_id"].dropna().unique()),
            "columns": list(dataset.frame.columns),
        },
        "data_quality": {
            "score": quality.score,
            "status": quality.status.value,
            "records_checked": quality.records_checked,
            "metrics": quality.metrics,
            "issues": [
                {
                    "code": issue.code,
                    "severity": issue.severity,
                    "message": issue.message,
                    "count": issue.count,
                    "row_indices_zero_based": list(issue.row_indices),
                }
                for issue in quality.issues
            ],
        },
        "signal_analysis": signal_analysis.summary() if signal_analysis else None,
        "anomalies": [
            {
                "index": finding.index,
                "value": finding.value,
                "methods": list(finding.methods),
                "explanation": finding.explanation,
            }
            for finding in anomalies
        ],
        "limitations": [
            "A report describes only the supplied dataset and configured software checks.",
            "Synthetic sample files are simulated and do not represent real sensor hardware measurements.",
            "The spectral noise-floor estimate is descriptive and is not a calibrated signal-to-noise ratio.",
            "No hardware, field deployment, or radar performance validation is implied.",
        ],
    }


def json_bytes(report: dict[str, Any]) -> bytes:
    """Serialize a report as readable UTF-8 JSON."""

    return json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")


def issues_csv_bytes(quality: QualityReport) -> bytes:
    """Serialize findings to a portable CSV, including a header when empty."""

    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=["code", "severity", "message", "count", "row_indices_zero_based"],
    )
    writer.writeheader()
    for issue in quality.issues:
        writer.writerow(
            {
                "code": issue.code,
                "severity": issue.severity,
                "message": issue.message,
                "count": issue.count,
                "row_indices_zero_based": ";".join(str(index) for index in issue.row_indices),
            }
        )
    return output.getvalue().encode("utf-8")


def write_reports(
    report: dict[str, Any], quality: QualityReport, output_dir: str | Path
) -> tuple[Path, Path]:
    """Write the JSON summary and issue CSV to an output directory."""

    destination = Path(output_dir)
    try:
        destination.mkdir(parents=True, exist_ok=True)
        json_path = destination / "technical_summary.json"
        csv_path = destination / "validation_issues.csv"
        json_path.write_bytes(json_bytes(report))
        csv_path.write_bytes(issues_csv_bytes(quality))
    except OSError as exc:
        LOGGER.exception("Unable to write reports to %s", destination)
        raise OSError(f"Unable to write reports to {destination}: {exc}") from exc
    LOGGER.info("Wrote reports to %s", destination)
    return json_path, csv_path
