import io

import pytest

from field_sensor_radar.exceptions import DatasetSchemaError
from field_sensor_radar.ingestion import load_dataset


def test_loads_json_records_and_metadata():
    source = io.StringIO(
        '{"metadata":{"classification":"synthetic"},"records":['
        '{"timestamp":"2025-01-01T00:00:00Z","sensor_id":"S1","latitude":1,"longitude":2,"reading":3}]}'
    )
    source.name = "capture.json"
    dataset = load_dataset(source)
    assert dataset.source_format == "json"
    assert dataset.metadata["classification"] == "synthetic"
    assert dataset.frame.loc[0, "reading"] == 3


def test_missing_required_column_fails_fast():
    source = io.StringIO("timestamp,sensor_id,latitude,longitude\n2025-01-01,S1,1,2\n")
    source.name = "broken.csv"
    with pytest.raises(DatasetSchemaError, match="reading"):
        load_dataset(source)


def test_unparseable_values_are_preserved_as_validation_metadata():
    source = io.StringIO(
        "timestamp,sensor_id,latitude,longitude,reading\n"
        "not-a-time,S1,1,2,not-a-number\n"
    )
    source.name = "bad-values.csv"
    dataset = load_dataset(source)
    assert dataset.invalid_cells["timestamp"] == (0,)
    assert dataset.invalid_cells["reading"] == (0,)
