import numpy as np

from data_work.lib.audio_features import FEATURE_KEYS, compute_audio_features, feature_distance


def test_compute_audio_features_zero_signal() -> None:
    audio = np.zeros(1024, dtype=np.float32)
    features = compute_audio_features(audio, sample_rate=8000)
    assert set(features) == set(FEATURE_KEYS)
    assert all(features[key] == 0.0 for key in FEATURE_KEYS)


def test_compute_audio_features_sine_signal() -> None:
    sample_rate = 8000
    t = np.linspace(0.0, 1.0, sample_rate, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    features = compute_audio_features(audio.astype(np.float32), sample_rate=sample_rate)
    assert features["rms"] > 0.0
    assert 200.0 <= features["spectral_centroid"] <= 1000.0
    assert features["zero_crossing_rate"] > 0.0


def test_feature_distance_relative_scale() -> None:
    base = {key: 1.0 for key in FEATURE_KEYS}
    other = dict(base)
    other["rms"] = 2.0
    distance = feature_distance(base, other)
    assert distance > 0.0
    assert distance < 1.0
