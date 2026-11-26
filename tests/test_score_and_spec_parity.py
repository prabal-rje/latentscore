"""DSP quality, reactive Score, and shared spec parity tests."""

from __future__ import annotations

import json
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import latentscore as ls
from latentscore import score as score_mod
from latentscore.config import MusicConfig, Step
from latentscore.errors import PlaybackError
from latentscore.synth import (
    SAMPLE_RATE,
    _apply_master_hall_reverb,
    apply_adsr,
    apply_humanize,
    assemble,
    build_harmony_plan,
    fade_into_nothingness,
    generate_sine,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = REPO_ROOT / "spec"


def test_bundled_mood_lexicon_matches_root_spec() -> None:
    """Prevent drift between root spec/mood_lexicon.json and packaged copy."""
    root_bytes = (SPEC_DIR / "mood_lexicon.json").read_bytes()
    pkg_bytes = (REPO_ROOT / "latentscore" / "_mood_lexicon.json").read_bytes()
    assert pkg_bytes == root_bytes


def test_mood_lexicon_golden_texts() -> None:
    """Verify Python mood_from_text matches every golden text including boundary guards."""
    spec = json.loads((SPEC_DIR / "mood_lexicon.json").read_text(encoding="utf-8"))
    for case in spec["golden_texts"]:
        cfg = ls.mood_from_text(case["text"])
        dumped = cfg.model_dump()
        for key, expected in case["expect"].items():
            assert dumped[key] == expected, f"{case['text']!r} field {key}: got {dumped[key]!r}"


@given(
    samples=st.integers(min_value=64, max_value=44100),
    attack=st.floats(min_value=0.0, max_value=2.0),
    decay=st.floats(min_value=0.0, max_value=2.0),
    sustain=st.floats(min_value=0.0, max_value=1.0),
    release=st.floats(min_value=0.0, max_value=2.0),
)
@settings(max_examples=40, deadline=None)
def test_adsr_always_click_free_at_boundaries(
    samples: int, attack: float, decay: float, sustain: float, release: float
) -> None:
    """Property: ADSR envelope always starts and ends at exact zero without clicks."""
    ones = np.ones(samples, dtype=np.float64)
    env = apply_adsr(ones, attack, decay, sustain, release, sr=SAMPLE_RATE)
    assert len(env) == samples
    assert float(env[0]) == 0.0
    assert float(env[-1]) == 0.0
    assert float(np.min(env)) >= -1e-12
    assert float(np.max(env)) <= 1.0 + 1e-12


def test_humanize_introduces_zero_hf_hiss() -> None:
    """Sub-2 Hz humanization must not inject broadband hiss above 4 kHz."""
    tol = json.loads((SPEC_DIR / "golden_cases.json").read_text(encoding="utf-8"))["tolerances"]
    tone = generate_sine(220.0, 2.0, sr=SAMPLE_RATE, amp=0.5)
    out = apply_humanize(tone, 0.8, sr=SAMPLE_RATE, rng=np.random.default_rng(7))
    spec = np.abs(np.fft.rfft(out)) ** 2
    freqs = np.fft.rfftfreq(len(out), d=1.0 / SAMPLE_RATE)
    hf_ratio = float(np.sum(spec[freqs > 4000.0]) / (np.sum(spec) + 1e-12))
    assert hf_ratio < tol["humanize_hf_ratio_above_4khz"]


def test_master_hall_reverb_calibrated_for_pure_tone() -> None:
    """Peak-calibrated hall reverb keeps a 0.35-amp tone well below clipping."""
    tol = json.loads((SPEC_DIR / "golden_cases.json").read_text(encoding="utf-8"))["tolerances"]
    tone = generate_sine(330.0, 3.0, sr=SAMPLE_RATE, amp=0.35)
    wet = _apply_master_hall_reverb(tone, 0.95, sr=SAMPLE_RATE)
    assert float(np.max(np.abs(wet))) <= tol["hall_reverb_max_tone_peak"]


def test_golden_cases_audio_invariants() -> None:
    """Verify every golden render case satisfies peak, DC, click, and audibility bounds."""
    spec = json.loads((SPEC_DIR / "golden_cases.json").read_text(encoding="utf-8"))
    tol = spec["tolerances"]

    for case in spec["render_cases"]:
        cfg = MusicConfig.model_validate(case["config"]).to_internal()
        audio = assemble(
            cfg,
            duration=float(case["duration"]),
            normalize=True,
            rng=np.random.default_rng(int(case["seed"])),
            t_offset=0.0,
        )
        assert np.all(np.isfinite(audio))
        peak = float(np.max(np.abs(audio)))
        assert tol["min_audible_peak"] <= peak <= tol["max_peak"]
        assert abs(float(np.mean(audio))) < tol["max_abs_dc"]
        assert float(np.max(np.abs(np.diff(audio)))) < tol["max_sample_jump"]

        if "t_offset_second" in case:
            t2 = float(case["t_offset_second"])
            audio2 = assemble(
                cfg,
                duration=float(case["duration"]),
                normalize=True,
                rng=np.random.default_rng(int(case["seed"])),
                t_offset=t2,
            )
            assert np.all(np.isfinite(audio2))
            h1 = build_harmony_plan(cfg, t2, np.random.default_rng(101), 0.0)
            h2 = build_harmony_plan(cfg, float(case["duration"]), np.random.default_rng(101), t2)
            assert len(h1.chords) >= 1 and len(h2.chords) >= 1


def test_score_glide_convergence_and_continuity(tmp_path: Path) -> None:
    """Verify Score converges discrete fields over small blocks and keeps smooth boundaries."""
    score = ls.Score("quiet morning desk", seed=11, glide_seconds=2.0, block_seconds=0.5)
    assert score.current_config.mode == "major"

    # Idempotent steering must converge all discrete fields within glide_seconds.
    t1 = score.steer("rainy midnight coffee shop")
    t2 = score.steer("rainy midnight coffee shop")
    assert t1 == t2
    assert t1.texture == "vinyl_crackle"

    # Pull 4 x 0.5 s = 2.0 s across block boundaries and verify zero seam clicks.
    blocks = [score.next_block(0.5) for _ in range(4)]
    for prev, nxt in zip(blocks, blocks[1:]):
        assert abs(float(nxt[0]) - float(prev[-1])) < 0.08
    # Discrete fields MUST have converged to rainy midnight coffee shop!
    assert score.current_config.root == "d"
    assert score.current_config.mode == "dorian"
    assert score.current_config.texture == "vinyl_crackle"

    # Identical re-steer with a longer glide_seconds after convergence must NOT rewind state!
    settled_brightness = score.internal_config.brightness
    score.steer("rainy midnight coffee shop", glide_seconds=10.0)
    score.next_block(0.5)
    assert score.internal_config.brightness == settled_brightness

    # Mixing fill() and next_block() must stay strictly sequential with no clicks.
    a = score.fill(1024)
    b = score.next_block(0.25)
    c = score.fill(2048)
    assert len(a) == 1024 and len(b) == int(round(0.25 * SAMPLE_RATE)) and len(c) == 2048
    assert abs(float(b[0]) - float(a[-1])) < 0.08
    assert abs(float(c[0]) - float(b[-1])) < 0.08

    # Nudging space must preserve untouched continuous tempo!
    custom = MusicConfig().to_internal().model_copy(update={"tempo": 0.42})
    s2 = ls.Score(custom, glide_seconds=0.0, block_seconds=0.5)
    custom.tempo = 0.99  # Caller mutation must NOT mutate Score!
    assert abs(s2.internal_config.tempo - 0.42) < 1e-6
    s2.nudge(space=Step(+1))
    s2.next_block(0.5)
    assert abs(s2.internal_config.tempo - 0.42) < 1e-6

    faded = fade_into_nothingness(np.ones(4410, dtype=np.float64), fade_sec=0.05)
    assert float(faded[-1]) == 0.0

    out_wav = score.save(tmp_path / "reactive.wav", duration=1.2, fade_out_seconds=0.2)
    assert out_wav.exists() and out_wav.stat().st_size > 44


def test_score_play_timeout_closes_delayed_stream_without_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_start = threading.Event()
    wrote = threading.Event()
    closed = threading.Event()

    class _DelayedStream:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def start(self) -> None:
            release_start.wait(timeout=2.0)

        def write(self, _data: object) -> None:
            wrote.set()

        def stop(self) -> None:
            pass

        def close(self) -> None:
            closed.set()

    monkeypatch.setattr(score_mod, "_PLAY_STARTUP_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(OutputStream=_DelayedStream))

    s = ls.Score("quiet morning desk", block_seconds=0.5)
    with pytest.raises(PlaybackError, match="Timed out"):
        s.play()
    assert not s.is_playing
    release_start.set()
    assert closed.wait(timeout=1.5)
    assert not wrote.is_set()
