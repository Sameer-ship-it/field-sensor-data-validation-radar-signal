"""Signal generation, FFT characterization, and digital filter utilities."""

from __future__ import annotations

import numpy as np
from scipy import signal

from field_sensor_radar.exceptions import SignalProcessingError
from field_sensor_radar.models import SignalAnalysis


def generate_sample_signal(
    *,
    sample_rate_hz: float = 200.0,
    duration_s: float = 2.0,
    frequency_hz: float = 12.0,
    amplitude: float = 1.0,
    phase_rad: float = 0.35,
    noise_std: float = 0.04,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a reproducible synthetic sinusoid plus Gaussian noise."""

    if sample_rate_hz <= 0 or duration_s <= 0 or noise_std < 0:
        raise ValueError("Sample rate and duration must be positive; noise standard deviation cannot be negative.")
    if frequency_hz < 0 or frequency_hz >= sample_rate_hz / 2:
        raise ValueError("Signal frequency must be in [0, Nyquist).")
    count = max(2, int(round(sample_rate_hz * duration_s)))
    time_s = np.arange(count, dtype=float) / sample_rate_hz
    rng = np.random.default_rng(seed)
    values = amplitude * np.sin(2 * np.pi * frequency_hz * time_s + phase_rad)
    values += rng.normal(0.0, noise_std, size=count)
    return time_s, values


def analyze_signal(values: np.ndarray | list[float], sample_rate_hz: float) -> SignalAnalysis:
    """Compute basic amplitude metrics and a window-corrected one-sided FFT.

    FFT phase is the phase of the dominant positive-frequency complex bin under
    the NumPy DFT convention; it is most interpretable for a near-stationary tone.
    """

    data = np.asarray(values, dtype=float).reshape(-1)
    if sample_rate_hz <= 0 or not np.isfinite(sample_rate_hz):
        raise SignalProcessingError("sample_rate_hz must be a finite positive number.")
    if len(data) < 2:
        raise SignalProcessingError("At least two finite samples are required for FFT analysis.")
    if not np.isfinite(data).all():
        raise SignalProcessingError("Signal contains non-finite values; clean or select finite samples first.")

    centered = data - float(np.mean(data))
    window = np.hanning(len(data))
    coherent_gain = float(np.mean(window))
    if coherent_gain <= 0:
        window = np.ones(len(data), dtype=float)
        coherent_gain = 1.0
    spectrum = np.fft.rfft(centered * window)
    frequencies = np.fft.rfftfreq(len(data), d=1.0 / sample_rate_hz)
    amplitudes = np.abs(spectrum) / (len(data) * coherent_gain)
    if len(amplitudes) > 1:
        amplitudes[1:-1 if len(data) % 2 == 0 else None] *= 2.0
    if len(amplitudes):
        amplitudes[0] = 0.0
    dominant_index: int | None = None
    if len(amplitudes) > 1 and float(np.max(amplitudes[1:])) > 0:
        dominant_index = int(np.argmax(amplitudes[1:]) + 1)
    dominant_frequency = float(frequencies[dominant_index]) if dominant_index is not None else None
    dominant_phase = float(np.angle(spectrum[dominant_index])) if dominant_index is not None else None
    noise_floor: float | None = None
    peak_to_floor_db: float | None = None
    if dominant_index is not None:
        candidate_indices = np.arange(1, len(amplitudes))
        adjacent_peak_bins = np.abs(candidate_indices - dominant_index) <= 1
        noise_bins = amplitudes[candidate_indices[~adjacent_peak_bins]]
        if len(noise_bins):
            noise_floor = float(np.median(noise_bins))
            if noise_floor > 0:
                peak_to_floor_db = float(20.0 * np.log10(amplitudes[dominant_index] / noise_floor))

    return SignalAnalysis(
        sample_count=len(data),
        sample_rate_hz=float(sample_rate_hz),
        mean=float(np.mean(data)),
        rms=float(np.sqrt(np.mean(np.square(data)))),
        peak_to_peak=float(np.ptp(data)),
        dominant_frequency_hz=dominant_frequency,
        dominant_phase_rad=dominant_phase,
        noise_floor_amplitude=noise_floor,
        peak_to_noise_floor_db=peak_to_floor_db,
        frequencies_hz=tuple(float(value) for value in frequencies),
        amplitudes=tuple(float(value) for value in amplitudes),
    )


def apply_filter(
    values: np.ndarray | list[float],
    *,
    sample_rate_hz: float,
    cutoff_hz: float,
    kind: str,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth low-pass or high-pass filter to finite samples."""

    data = np.asarray(values, dtype=float).reshape(-1)
    if kind not in {"lowpass", "highpass"}:
        raise SignalProcessingError("kind must be 'lowpass' or 'highpass'.")
    if order < 1:
        raise SignalProcessingError("Filter order must be a positive integer.")
    nyquist = sample_rate_hz / 2.0
    if sample_rate_hz <= 0 or not 0 < cutoff_hz < nyquist:
        raise SignalProcessingError("cutoff_hz must be strictly between 0 and the Nyquist frequency.")
    if len(data) < 2 or not np.isfinite(data).all():
        raise SignalProcessingError("Filtering requires at least two finite samples.")

    sos = signal.butter(order, cutoff_hz, btype=kind, fs=sample_rate_hz, output="sos")
    try:
        return signal.sosfiltfilt(sos, data)
    except ValueError:
        # Zero-phase padding cannot be applied to very short captures. A causal
        # second-order-section pass still provides a safe result for inspection.
        return signal.sosfilt(sos, data)
