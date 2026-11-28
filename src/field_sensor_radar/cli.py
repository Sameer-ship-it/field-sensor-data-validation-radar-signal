"""Command-line validation and report generation."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

from field_sensor_radar.anomaly import detect_anomalies
from field_sensor_radar.exceptions import SensorDataError
from field_sensor_radar.ingestion import load_dataset
from field_sensor_radar.logging_config import configure_logging
from field_sensor_radar.plotting import save_signal_analysis_figure
from field_sensor_radar.reporting import build_technical_report, write_reports
from field_sensor_radar.signals import analyze_signal
from field_sensor_radar.validation import validate_dataset

LOGGER = logging.getLogger(__name__)


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sample rate must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("sample rate must be finite and greater than zero")
    return parsed


def _estimate_sample_rate(frame) -> float | None:
    timed = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    deltas = timed.groupby("sensor_id")["timestamp"].diff().dt.total_seconds().dropna()
    positive = deltas[deltas > 0]
    if positive.empty:
        return None
    interval = float(positive.median())
    return 1.0 / interval if interval > 0 else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate field sensor data and analyze its sampled readings.")
    parser.add_argument("input", type=Path, help="Input .csv or .json sensor dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"), help="Directory for JSON and CSV reports")
    parser.add_argument("--sensor-id", help="Analyze one sensor; defaults to the first available sensor")
    parser.add_argument("--sample-rate-hz", type=_positive_float, help="Override the sample rate inferred from timestamps")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        dataset = load_dataset(args.input)
        quality = validate_dataset(dataset)
        frame = dataset.frame
        sensors = frame["sensor_id"].dropna().unique()
        chosen_sensor = args.sensor_id or (str(sensors[0]) if len(sensors) else None)
        signal_analysis = None
        anomalies = ()
        if chosen_sensor is not None:
            readings = frame.loc[frame["sensor_id"] == chosen_sensor, "reading"]
            valid = readings.dropna().to_numpy(dtype=float)
            sample_rate = (
                args.sample_rate_hz
                if args.sample_rate_hz is not None
                else _estimate_sample_rate(frame.loc[frame["sensor_id"] == chosen_sensor])
            )
            if sample_rate and len(valid) >= 2:
                signal_analysis = analyze_signal(valid, sample_rate)
                anomalies = detect_anomalies(valid)
            else:
                LOGGER.warning("Signal analysis skipped: insufficient samples or unknown sample rate.")
        report = build_technical_report(dataset, quality, signal_analysis=signal_analysis, anomalies=anomalies)
        json_path, csv_path = write_reports(report, quality, args.output_dir)
        figure_path = None
        if signal_analysis is not None:
            figure_path = save_signal_analysis_figure(valid, signal_analysis, args.output_dir / "signal_analysis.png")
        print(f"Quality status: {quality.status.value} ({quality.score:.1f}/100)")
        print(f"JSON report: {json_path}")
        print(f"CSV issues: {csv_path}")
        if figure_path is not None:
            print(f"Signal figure: {figure_path}")
        return 0
    except (SensorDataError, OSError, ValueError) as exc:
        LOGGER.error("Analysis failed: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
