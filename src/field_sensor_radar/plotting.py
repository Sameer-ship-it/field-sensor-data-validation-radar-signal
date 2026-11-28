"""Static Matplotlib figures for reproducible offline technical reports."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from field_sensor_radar.models import SignalAnalysis

LOGGER = logging.getLogger(__name__)


def save_signal_analysis_figure(
    values: np.ndarray | list[float], analysis: SignalAnalysis, output_path: str | Path
) -> Path:
    """Save time-domain samples and the associated one-sided FFT spectrum."""

    data = np.asarray(values, dtype=float).reshape(-1)
    if len(data) != analysis.sample_count:
        raise ValueError("Signal sample count does not match the supplied analysis.")
    destination = Path(output_path)
    figure, (time_axis, frequency_axis) = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    try:
        time_axis.plot(np.arange(len(data)) / analysis.sample_rate_hz, data, color="#1565c0", linewidth=1.0)
        time_axis.set_title("Sensor signal")
        time_axis.set_xlabel("Time (s)")
        time_axis.set_ylabel("Reading")
        time_axis.grid(True, alpha=0.25)

        frequency_axis.plot(analysis.frequencies_hz, analysis.amplitudes, color="#ef6c00", linewidth=1.0)
        frequency_axis.set_title("One-sided FFT magnitude")
        frequency_axis.set_xlabel("Frequency (Hz)")
        frequency_axis.set_ylabel("Amplitude")
        frequency_axis.grid(True, alpha=0.25)

        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=140, format="png")
    except OSError:
        LOGGER.exception("Unable to save signal analysis figure to %s", destination)
        raise
    finally:
        plt.close(figure)
    return destination
