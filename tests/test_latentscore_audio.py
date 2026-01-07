from pathlib import Path

from latentscore.audio import write_wav


def test_write_wav_accepts_sequence(tmp_path: Path) -> None:
    target = tmp_path / "seq.wav"
    samples = [0.0, 0.1, -0.1, 0.0]

    write_wav(target, samples, sample_rate=22_050)

    assert target.exists()
    assert target.stat().st_size > 0
