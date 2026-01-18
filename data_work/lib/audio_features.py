"""Minimal audio feature extraction helpers for analysis scripts."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

FEATURE_KEYS = (
    "rms",
    "spectral_centroid",
    "spectral_bandwidth",
    "zero_crossing_rate",
    "onset_strength",
)


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return np.mean(audio, axis=0)
    raise ValueError("Audio must be 1D or 2D array.")


def compute_audio_features(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Compute simple audio features without external dependencies."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return {key: 0.0 for key in FEATURE_KEYS}

    mono = _to_mono(audio)
    if mono.size == 0:
        return {key: 0.0 for key in FEATURE_KEYS}

    rms = float(np.sqrt(np.mean(mono**2)))

    spectrum = np.fft.rfft(mono)
    magnitudes = np.abs(spectrum)
    mag_sum = float(np.sum(magnitudes))
    if mag_sum > 0:
        freqs = np.fft.rfftfreq(mono.size, d=1.0 / sample_rate)
        centroid = float(np.sum(freqs * magnitudes) / mag_sum)
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitudes) / mag_sum))
    else:
        centroid = 0.0
        bandwidth = 0.0

    if mono.size < 2:
        zero_crossing_rate = 0.0
    else:
        sign_changes = np.sum((mono[:-1] * mono[1:]) < 0)
        zero_crossing_rate = float(sign_changes / (mono.size - 1))

    frame = min(1024, mono.size)
    hop = max(1, frame // 2)
    energies = [
        float(np.mean(mono[i : i + frame] ** 2)) for i in range(0, mono.size - frame + 1, hop)
    ]
    if len(energies) < 2:
        onset_strength = 0.0
    else:
        diffs = np.diff(energies)
        onset_strength = float(np.mean(np.clip(diffs, 0.0, None)))

    return {
        "rms": rms,
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "zero_crossing_rate": zero_crossing_rate,
        "onset_strength": onset_strength,
    }


def feature_distance(base: Mapping[str, float], other: Mapping[str, float]) -> float:
    """Compute average relative distance across known features."""
    total = 0.0
    for key in FEATURE_KEYS:
        base_val = float(base.get(key, 0.0))
        other_val = float(other.get(key, 0.0))
        denom = abs(base_val) + 1e-6
        total += abs(other_val - base_val) / denom
    return total / len(FEATURE_KEYS)
