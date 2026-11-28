import pandas as pd

from field_sensor_radar.models import QualityStatus, SensorDataset
from field_sensor_radar.validation import validate_dataset


def test_clean_records_pass_quality_checks():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=8, freq="100ms", tz="UTC"),
            "sensor_id": ["S1"] * 8,
            "latitude": [12.0] * 8,
            "longitude": [77.0] * 8,
            "reading": [1.0, 1.1, 0.9, 1.05, 0.98, 1.02, 0.95, 1.03],
        }
    )
    result = validate_dataset(SensorDataset(frame, "clean.csv", "csv"))
    assert result.status == QualityStatus.PASS
    assert result.score == 100.0


def test_invalid_gps_and_duplicate_timestamp_are_reported():
    timestamps = pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z", "2025-01-01T00:00:01Z"])
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "sensor_id": ["S1"] * 3,
            "latitude": [0.0, 91.0, 0.0],
            "longitude": [0.0, 0.0, 0.0],
            "reading": [1.0, 1.0, 1.0],
        }
    )
    result = validate_dataset(SensorDataset(frame, "issues.csv", "csv"))
    codes = {issue.code for issue in result.issues}
    assert result.status == QualityStatus.FAIL
    assert {"invalid_gps", "duplicate_timestamp"}.issubset(codes)
