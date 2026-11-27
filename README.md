# Field Sensor Data Validation & Radar Signal Analysis System

A modular Python toolkit for ingesting sensor captures, reporting data-quality issues, examining sampled signals, and producing technical summaries. The included examples are **synthetic** and make no claim of hardware, field, RF, or radar performance validation.

## What it does

- Reads CSV and JSON records with required timestamps, sensor IDs, GPS coordinates, and numeric readings.
- Reports missing and unparseable values, duplicate sensor timestamps, irregular sampling, invalid GPS ranges, and robust sensor-reading outliers.
- Calculates a configurable data-quality score and `PASS`, `WARNING`, or `FAIL` classification.
- Computes signal mean, RMS, peak-to-peak amplitude, dominant FFT frequency, dominant-bin phase estimate, and a median spectral noise-floor estimate.
- Applies Butterworth low-pass and high-pass filters and runs explainable global Z-score and moving-window anomaly checks.
- Exports a JSON technical summary and CSV validation findings; the CLI also saves a static PNG signal/FFT figure when signal analysis is available.
- Includes a Streamlit dashboard, CLI, synthetic example files, and pytest unit tests.

## Quick start

Python 3.10 or later is supported.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run the dashboard:

```bash
streamlit run streamlit_app.py
```

Run the command-line report generator:

```bash
python -m field_sensor_radar.cli data/synthetic_field_log.csv --output-dir reports
```

Or use the installed entry point:

```bash
field-sensor-report data/synthetic_capture.json --output-dir reports
```

Run unit tests:

```bash
python -m pytest
```

## Project layout

```text
.
├── data/                 # Clearly labeled synthetic CSV and JSON examples
├── docs/                 # Architecture and input schema
├── scripts/              # Reproducible synthetic-data generator
├── src/field_sensor_radar # Reusable ingestion, validation, signal, and report modules
├── tests/                # Unit tests for critical behavior
├── streamlit_app.py      # Interactive dashboard
├── pyproject.toml
└── requirements.txt
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module boundaries, score formula, and signal-processing assumptions, and [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md) for accepted fields.

## Quality score interpretation

The starting score is 100. Unique rows with failing findings contribute a 40-point weighted deduction and unique warning-only rows contribute a 10-point weighted deduction, each scaled by the fraction of input rows affected. Scores are clamped to 0–100. Any failing issue or score below 60 is `FAIL`; any remaining issue or score below 90 is `WARNING`; otherwise the dataset is `PASS`. Tune these software defaults against the operating context before using the score for decisions.

## Input notes

Required fields are `timestamp`, `sensor_id`, `latitude`, `longitude`, and `reading`. JSON can be a list of record objects or an object containing `records` and optional `metadata`. Timestamps are parsed as UTC. Optional columns such as `sample_rate_hz`, `gain_db`, and `capture_id` are retained.

The CLI estimates sample rate from each selected sensor's median positive timestamp interval. The dashboard uses the same estimate for plotting and filtering. Digital filters and FFT metrics assume a sufficiently uniform sequence; review the sampling validation findings before interpreting them. A CLI sample-rate override is available when acquisition metadata provides a better value.

## Scope and limitations

This is a software analysis and simulation project. It does not acquire data from a physical sensor, calibrate instruments, model a complete radar chain, validate RF emissions, or establish field readiness. FFT phase is a numerical estimate relative to the sample window and is not absolute phase calibration. Review source provenance, timestamps, sampling assumptions, and configuration before operational use.

## License

Released under the MIT License. See [LICENSE](LICENSE).
