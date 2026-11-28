"""CSV and JSON ingestion with schema checks and non-fatal cell coercion."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import IO, Any

import pandas as pd

from field_sensor_radar.exceptions import DatasetReadError, DatasetSchemaError
from field_sensor_radar.models import SensorDataset

LOGGER = logging.getLogger(__name__)
REQUIRED_COLUMNS = ("timestamp", "sensor_id", "latitude", "longitude", "reading")
NUMERIC_COLUMNS = ("latitude", "longitude", "reading", "sample_rate_hz")


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = [str(column).strip().lower() for column in frame.columns]
    if len(set(normalized)) != len(normalized):
        raise DatasetSchemaError("Column names are duplicated after trimming and case normalization.")
    copied = frame.copy()
    copied.columns = normalized
    return copied


def _read_json(source: str | Path | IO[str] | IO[bytes]) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        if hasattr(source, "read"):
            payload = json.load(source)
        else:
            with Path(source).open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise DatasetReadError(f"Could not read JSON dataset: {exc}") from exc

    metadata: dict[str, Any] = {}
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("records")
        metadata = payload.get("metadata", {})
    else:
        raise DatasetSchemaError("JSON must contain a list of records or an object with a 'records' list.")
    if not isinstance(records, list):
        raise DatasetSchemaError("JSON field 'records' must be a list.")
    if not isinstance(metadata, dict):
        raise DatasetSchemaError("JSON field 'metadata' must be an object when provided.")
    try:
        return pd.DataFrame.from_records(records), metadata
    except (ValueError, TypeError) as exc:
        raise DatasetSchemaError(f"JSON records do not form a tabular dataset: {exc}") from exc


def load_dataset(source: str | Path | IO[str] | IO[bytes]) -> SensorDataset:
    """Load a CSV or JSON dataset and normalize supported field types.

    Required columns are ``timestamp``, ``sensor_id``, ``latitude``,
    ``longitude``, and ``reading``. Unparseable values are retained as missing
    values and recorded in ``invalid_cells`` for the validation report.
    """

    name = getattr(source, "name", None) or (Path(source).name if not hasattr(source, "read") else "uploaded_dataset")
    suffix = Path(str(name)).suffix.lower()
    try:
        if suffix == ".csv":
            frame = pd.read_csv(source)
            metadata: dict[str, Any] = {}
            source_format = "csv"
        elif suffix == ".json":
            frame, metadata = _read_json(source)
            source_format = "json"
        else:
            raise DatasetSchemaError("Unsupported file type. Choose a .csv or .json file.")
    except DatasetSchemaError:
        raise
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError, TypeError) as exc:
        raise DatasetReadError(f"Could not read {name}: {exc}") from exc

    if frame.empty:
        raise DatasetSchemaError("Dataset contains no records.")
    frame = _normalize_columns(frame)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetSchemaError("Missing required columns: " + ", ".join(missing))

    invalid_cells: dict[str, tuple[int, ...]] = {}
    raw_timestamps = frame["timestamp"].copy()
    # ``format='mixed'`` accepts ISO timestamps that omit fractional seconds on
    # whole-second rows while retaining milliseconds on other rows.
    parsed_timestamps = pd.to_datetime(raw_timestamps, format="mixed", errors="coerce", utc=True)
    bad_timestamps = tuple(
        int(index)
        for index, (raw, parsed) in enumerate(zip(raw_timestamps, parsed_timestamps, strict=False))
        if pd.notna(raw) and pd.isna(parsed)
    )
    if bad_timestamps:
        invalid_cells["timestamp"] = bad_timestamps
    frame["timestamp"] = parsed_timestamps

    for column in NUMERIC_COLUMNS:
        if column not in frame.columns:
            continue
        raw_values = frame[column].copy()
        parsed_values = pd.to_numeric(raw_values, errors="coerce")
        bad_indices = tuple(
            int(index)
            for index, (raw, parsed) in enumerate(zip(raw_values, parsed_values, strict=False))
            if pd.notna(raw) and pd.isna(parsed)
        )
        if bad_indices:
            invalid_cells[column] = bad_indices
        frame[column] = parsed_values

    frame["sensor_id"] = frame["sensor_id"].astype("string")
    LOGGER.info("Loaded %d records from %s (%s)", len(frame), name, source_format)
    if invalid_cells:
        LOGGER.warning("Found unparseable values while loading %s: %s", name, invalid_cells)
    return SensorDataset(
        frame=frame,
        source_name=str(name),
        source_format=source_format,
        metadata=metadata,
        invalid_cells=invalid_cells,
    )
