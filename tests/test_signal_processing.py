import numpy as np
import pytest

from field_sensor_radar.anomaly import detect_anomalies
from field_sensor_radar.exceptions import SignalProcessingError
from field_sensor_radar.signals import analyze_signal, apply_filter, generate_sample_signal


def test_fft_finds_synthetic_tone_near_expected_frequency():
    _, values = generate_sample_signal(sample_rate_hz=200, duration_s=4, frequency_hz=12, noise_std=0.0)
    result = analyze_signal(values, 200)
    assert result.dominant_frequency_hz == pytest.approx(12, abs=0.3)
    assert result.rms == pytest.approx(1 / np.sqrt(2), rel=0.03)


def test_fft_reports_a_descriptive_noise_floor_for_noisy_signal():
    _, values = generate_sample_signal(sample_rate_hz=200, duration_s=4, frequency_hz=12, noise_std=0.04)
    result = analyze_signal(values, 200)
    assert result.noise_floor_amplitude is not None
    assert result.noise_floor_amplitude > 0
    assert result.peak_to_noise_floor_db is not None


def test_filter_reduces_high_frequency_component():
    rate = 200.0
    time = np.arange(1000) / rate
    values = np.sin(2 * np.pi * 5 * time) + 0.5 * np.sin(2 * np.pi * 50 * time)
    filtered = apply_filter(values, sample_rate_hz=rate, cutoff_hz=15, kind="lowpass")
    assert np.std(filtered - np.sin(2 * np.pi * 5 * time)) < 0.08


def test_bad_cutoff_is_rejected():
    with pytest.raises(SignalProcessingError):
        apply_filter([0, 1, 0], sample_rate_hz=10, cutoff_hz=5, kind="lowpass")


def test_anomaly_result_explains_extreme_sample():
    findings = detect_anomalies(
        [1, 1.1, 0.9, 1.05, 0.95, 20, 1.0, 0.98, 1.02],
        z_threshold=2.0,
        window_threshold=2.0,
    )
    assert any(item.index == 5 and item.explanation for item in findings)
