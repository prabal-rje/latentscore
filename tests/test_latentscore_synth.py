import numpy as np

from latentscore.config import MusicConfig
from latentscore.synth import _butter_cached, assemble


def test_butter_cache_key_stability() -> None:
    b1, a1 = _butter_cached("low", 0.5)
    b2, a2 = _butter_cached("low", 0.5)
    assert b1 is b2
    assert a1 is a2


def test_assemble_accepts_internal_config() -> None:
    config = MusicConfig().to_internal()
    np.random.seed(0)
    audio = assemble(config, duration=1.0, rng=np.random.default_rng(0))
    assert audio.size > 0


def test_stereo_affects_output() -> None:
    config = MusicConfig().to_internal()
    narrow = config.model_copy(update={"stereo": 0.0})
    wide = config.model_copy(update={"stereo": 1.0})

    np.random.seed(0)
    audio_narrow = assemble(narrow, duration=1.0, rng=np.random.default_rng(0))
    np.random.seed(0)
    audio_wide = assemble(wide, duration=1.0, rng=np.random.default_rng(0))

    assert not np.allclose(audio_narrow, audio_wide)
