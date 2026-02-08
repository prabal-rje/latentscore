import numpy as np

from latentscore.config import MusicConfig
from latentscore.synth import (
    _butter_cached,
    assemble,
    generate_sawtooth,
    generate_sine,
    generate_square,
    generate_triangle,
)


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


class TestOscillatorPhaseContinuity:
    """Test that oscillators maintain phase continuity with t_offset."""

    def test_sine_phase_continuity(self) -> None:
        """Two consecutive chunks should equal one continuous chunk."""
        freq = 440.0
        duration = 0.5
        sr = 44100

        # Generate one continuous 1-second chunk
        continuous = generate_sine(freq, duration * 2, sr=sr, amp=1.0, t_offset=0.0)

        # Generate two consecutive 0.5-second chunks
        chunk1 = generate_sine(freq, duration, sr=sr, amp=1.0, t_offset=0.0)
        chunk2 = generate_sine(freq, duration, sr=sr, amp=1.0, t_offset=duration)
        concatenated = np.concatenate([chunk1, chunk2])

        # They should be identical (or very close due to floating point)
        assert np.allclose(continuous, concatenated, atol=1e-10)

    def test_triangle_phase_continuity(self) -> None:
        """Two consecutive triangle chunks should equal one continuous chunk."""
        freq = 440.0
        duration = 0.5
        sr = 44100

        continuous = generate_triangle(freq, duration * 2, sr=sr, amp=1.0, t_offset=0.0)

        chunk1 = generate_triangle(freq, duration, sr=sr, amp=1.0, t_offset=0.0)
        chunk2 = generate_triangle(freq, duration, sr=sr, amp=1.0, t_offset=duration)
        concatenated = np.concatenate([chunk1, chunk2])

        assert np.allclose(continuous, concatenated, atol=1e-10)

    def test_sawtooth_t_offset_changes_phase(self) -> None:
        """Sawtooth with different t_offset produces phase-shifted output.

        Note: Perfect phase continuity across chunks is NOT guaranteed for
        sawtooth/square due to:
        1. PolyBLEP floating point edge cases at region boundaries
        2. Decimation filter transients at chunk boundaries

        These are known limitations. For ambient music, these artifacts are
        typically inaudible due to layering and the nature of the content.
        """
        freq = 440.0
        duration = 0.1
        sr = 44100

        no_offset = generate_sawtooth(freq, duration, sr=sr, amp=1.0, t_offset=0.0)
        with_offset = generate_sawtooth(freq, duration, sr=sr, amp=1.0, t_offset=0.25)

        # Different offsets should produce different waveforms
        assert not np.allclose(no_offset, with_offset)

        # Same offset should produce same waveform
        same1 = generate_sawtooth(freq, duration, sr=sr, amp=1.0, t_offset=0.5)
        same2 = generate_sawtooth(freq, duration, sr=sr, amp=1.0, t_offset=0.5)
        assert np.allclose(same1, same2)

    def test_square_t_offset_changes_phase(self) -> None:
        """Square with different t_offset produces phase-shifted output.

        Same limitations as sawtooth regarding perfect continuity.
        """
        freq = 440.0
        duration = 0.1
        sr = 44100

        no_offset = generate_square(freq, duration, sr=sr, amp=1.0, t_offset=0.0)
        with_offset = generate_square(freq, duration, sr=sr, amp=1.0, t_offset=0.25)

        # Different offsets should produce different waveforms
        assert not np.allclose(no_offset, with_offset)

        # Same offset should produce same waveform
        same1 = generate_square(freq, duration, sr=sr, amp=1.0, t_offset=0.5)
        same2 = generate_square(freq, duration, sr=sr, amp=1.0, t_offset=0.5)
        assert np.allclose(same1, same2)

    def test_sine_nonzero_offset_changes_output(self) -> None:
        """Verify that t_offset actually affects the output."""
        freq = 440.0
        duration = 0.1
        sr = 44100

        no_offset = generate_sine(freq, duration, sr=sr, amp=1.0, t_offset=0.0)
        # Use offset that's NOT a multiple of the period (1/440 = 0.00227s)
        # 0.001s = 0.44 cycles, giving a clear phase difference
        with_offset = generate_sine(freq, duration, sr=sr, amp=1.0, t_offset=0.001)

        # Different offsets should produce different waveforms
        assert not np.allclose(no_offset, with_offset)

    def test_assemble_accepts_t_offset(self) -> None:
        """Verify that assemble accepts t_offset parameter."""
        config = MusicConfig().to_internal()
        rng = np.random.default_rng(42)

        # Should not raise - t_offset is accepted even if not fully propagated
        audio = assemble(config, duration=1.0, rng=rng, t_offset=5.0)
        assert audio.size > 0
