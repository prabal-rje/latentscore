from pathlib import Path

import numpy as np

from latentscore.audio import ensure_audio_contract, write_wav


def test_write_wav_accepts_sequence(tmp_path: Path) -> None:
    target = tmp_path / "seq.wav"
    samples = [0.0, 0.1, -0.1, 0.0]

    write_wav(target, samples, sample_rate=22_050)

    assert target.exists()
    assert target.stat().st_size > 0


def test_ensure_audio_contract_skip_peak() -> None:
    audio = np.array([2.0, -2.0], dtype=np.float32)
    out = ensure_audio_contract(audio, check_peak=False)
    assert np.allclose(out, audio)
