"""Reactive local ambient music engine for interactive Python applications.

Provides zero-download deterministic text-to-mood mapping (``mood_from_text``)
and the stateful ``Score`` engine for live steering, gentle drift, pull-based
audio generation, WAV export, and non-blocking local playback.

File map:
- Types and constants: immutable lexicon rules and drift configuration.
- Lexicon mood steering: word-boundary deterministic keyword mapping matching
  ``spec/mood_lexicon.json``.
- Reactive Score engine: explicit glide convergence, canonical FIFO timeline,
  lock-free steering during rendering, and prebuffered ``sounddevice`` playback.
"""

from __future__ import annotations

import json
import math
import queue
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, TypeGuard

import numpy as np
from numpy.typing import NDArray

from .audio import SAMPLE_RATE, FloatArray, ensure_audio_contract, write_wav
from .config import (
    ConfigInput,
    MusicConfig,
    MusicConfigUpdate,
    Step,
    SynthConfig,
    coerce_internal_config,
    merge_internal_config,
)
from .errors import InvalidConfigError, PlaybackError
from .synth import (
    assemble,
    compute_bar_aligned_duration,
    fade_into_nothingness,
    interpolate_configs,
    soft_limit,
)

# === Types and constants ===

_OVERLAP_SECONDS = 0.45
_FIXED_SESSION_GAIN = 0.62
_PLAY_STARTUP_TIMEOUT_SECONDS = 5.0
_WORD_RE = re.compile(r"[a-z]+")
_DEFAULT_DRIFT_MOODS: tuple[str, ...] = (
    "quiet morning desk",
    "warm cedar zen meditation",
    "rainy midnight coffee shop",
    "vast deep space nebula with stars",
    "soft ocean mist and slow drift",
    "gentle golden sunrise",
)

_ROOT_CIRCLE: tuple[str, ...] = (
    "c",
    "g",
    "d",
    "a",
    "e",
    "b",
    "f#",
    "c#",
    "g#",
    "d#",
    "a#",
    "f",
)

_LABEL_SCALES: Mapping[str, tuple[tuple[str, float], ...]] = {
    "tempo": (("very_slow", 0.15), ("slow", 0.3), ("medium", 0.5), ("fast", 0.7), ("very_fast", 0.9)),
    "brightness": (("very_dark", 0.1), ("dark", 0.3), ("medium", 0.5), ("bright", 0.7), ("very_bright", 0.9)),
    "space": (("dry", 0.1), ("small", 0.3), ("medium", 0.5), ("large", 0.7), ("vast", 0.95)),
    "motion": (("static", 0.1), ("slow", 0.3), ("medium", 0.5), ("fast", 0.7), ("chaotic", 0.9)),
    "stereo": (("mono", 0.0), ("narrow", 0.25), ("medium", 0.5), ("wide", 0.75), ("ultra_wide", 1.0)),
    "echo": (("none", 0.0), ("subtle", 0.25), ("medium", 0.5), ("heavy", 0.75), ("infinite", 0.95)),
    "human": (("robotic", 0.0), ("tight", 0.15), ("natural", 0.3), ("loose", 0.5), ("drunk", 0.8)),
}


@dataclass(frozen=True, slots=True)
class _LexiconEntry:
    keywords: tuple[str, ...]
    updates: tuple[tuple[str, str | int | float | bool], ...]


@dataclass(frozen=True, slots=True)
class _LexiconGroup:
    group: str
    first_match: bool
    entries: tuple[_LexiconEntry, ...]


# === Lexicon mood steering ===


def _is_mapping(val: object) -> TypeGuard[Mapping[str, object]]:
    return isinstance(val, dict)


def _is_list(val: object) -> TypeGuard[list[object]]:
    return isinstance(val, list)


def _parse_entry(raw: object) -> _LexiconEntry:
    assert _is_mapping(raw), "Lexicon entry must be an object"
    raw_kw = raw.get("keywords")
    raw_set = raw.get("set")
    assert _is_list(raw_kw) and _is_mapping(raw_set)
    keywords = tuple(str(word).lower() for word in raw_kw)
    updates: list[tuple[str, str | int | float | bool]] = []
    for key, value in raw_set.items():
        assert isinstance(value, str | int | float | bool)
        updates.append((str(key), value))
    return _LexiconEntry(keywords=keywords, updates=tuple(updates))


def _load_lexicon_groups() -> tuple[_LexiconGroup, ...]:
    raw_text = (
        resources.files(__package__ or "latentscore")
        .joinpath("_mood_lexicon.json")
        .read_text(encoding="utf-8")
    )
    payload: object = json.loads(raw_text)
    assert _is_mapping(payload), "Lexicon payload must be a JSON object"
    raw_rules = payload.get("rules")
    assert _is_list(raw_rules), "Lexicon rules must be a list"

    groups: list[_LexiconGroup] = []
    for raw_group in raw_rules:
        assert _is_mapping(raw_group)
        raw_entries = raw_group.get("entries")
        assert _is_list(raw_entries)
        groups.append(
            _LexiconGroup(
                group=str(raw_group.get("group", "")),
                first_match=bool(raw_group.get("first_match", True)),
                entries=tuple(_parse_entry(item) for item in raw_entries),
            )
        )
    return tuple(groups)


_LEXICON_GROUPS: tuple[_LexiconGroup, ...] = _load_lexicon_groups()


_INFLECTION_SUFFIXES: frozenset[str] = frozenset(
    {"s", "es", "y", "ed", "ing", "ly", "er", "ness"}
)


def _matches_any(tokens: tuple[str, ...], keywords: Sequence[str]) -> bool:
    """Match keywords on word boundaries (plus common suffixes for >= 4-letter stems)."""
    for kw in keywords:
        allow_inflection = len(kw) >= 4
        for tok in tokens:
            if tok == kw:
                return True
            if allow_inflection and tok.startswith(kw) and tok[len(kw) :] in _INFLECTION_SUFFIXES:
                return True
    return False


def mood_from_text(text: str, *, base: MusicConfig | None = None) -> MusicConfig:
    """Map free-form text deterministically to a ``MusicConfig`` with zero downloads.

    Uses word-boundary matching over the shared ``spec/mood_lexicon.json`` rules
    so ordinary words like ``"office"`` or ``"breakfast"`` never trigger stray moods.

    Args:
        text: Natural-language mood or scene description.
        base: Optional starting configuration (defaults to ``MusicConfig()``).

    Returns:
        Validated ``MusicConfig`` reflecting matched mood cues.

    Examples:
        >>> cfg = mood_from_text("quiet morning desk")
        >>> (cfg.mode, cfg.tempo, cfg.space, cfg.density)
        ('major', 'slow', 'small', 3)
        >>> cfg2 = mood_from_text("nice office breakfast by the window")
        >>> (cfg2.tempo, cfg2.texture, cfg2.space)
        ('medium', 'shimmer', 'medium')
    """
    current = base if base is not None else MusicConfig()
    data: dict[str, Any] = current.model_dump()
    tokens = tuple(_WORD_RE.findall(text.lower()))

    for group in _LEXICON_GROUPS:
        for entry in group.entries:
            if _matches_any(tokens, entry.keywords):
                for field_name, field_value in entry.updates:
                    data[field_name] = field_value
                if group.first_match:
                    break

    return MusicConfig.model_validate(data)


def _approx_label(value: float, table: Sequence[tuple[str, float]]) -> str:
    return min(table, key=lambda pair: abs(pair[1] - value))[0]


def _snap_phrase_len(bars: int) -> int:
    if bars <= 3:
        return 2
    if bars <= 6:
        return 4
    return 8


def _internal_to_public(cfg: SynthConfig) -> MusicConfig:
    """Convert an internal numeric SynthConfig to its closest public MusicConfig."""
    data: dict[str, Any] = cfg.model_dump()
    for field_name, table in _LABEL_SCALES.items():
        data[field_name] = _approx_label(float(getattr(cfg, field_name)), table)
    data["phrase_len_bars"] = _snap_phrase_len(int(cfg.phrase_len_bars))
    return MusicConfig.model_validate(data)


def _coerce_owned_target(value: str | ConfigInput) -> SynthConfig:
    if isinstance(value, str):
        return mood_from_text(value).to_internal()
    return coerce_internal_config(value).model_copy(deep=True)


def _raised_cosine_fade(length: int) -> NDArray[np.float64]:
    if length <= 1:
        return np.ones(max(0, length), dtype=np.float64)
    u = np.linspace(0.0, math.pi, length, dtype=np.float64)
    return np.asarray(0.5 * (1.0 - np.cos(u)), dtype=np.float64)


def _neighbor_root(root: str, step: int) -> str:
    idx = _ROOT_CIRCLE.index(root) if root in _ROOT_CIRCLE else 0
    return _ROOT_CIRCLE[(idx + step) % len(_ROOT_CIRCLE)]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _apply_nudge_preserving_continuous(
    current_target: SynthConfig,
    update: MusicConfigUpdate,
) -> SynthConfig:
    """Apply only the touched fields in ``update`` without snapping untouched floats."""
    step_updates: dict[str, Any] = {}
    abs_fields: dict[str, Any] = {}
    for field_name in MusicConfigUpdate.model_fields:
        val = getattr(update, field_name)
        if val is None:
            continue
        if isinstance(val, Step):
            if field_name == "density":
                step_updates["density"] = max(2, min(6, int(current_target.density) + val.delta))
            elif field_name in _LABEL_SCALES:
                scale = _LABEL_SCALES[field_name]
                cur_val = float(getattr(current_target, field_name))
                cur_idx = min(range(len(scale)), key=lambda i: abs(scale[i][1] - cur_val))
                next_idx = max(0, min(len(scale) - 1, cur_idx + val.delta))
                step_updates[field_name] = scale[next_idx][1]
            else:
                base_pub = _internal_to_public(current_target)
                stepped_pub = MusicConfigUpdate.model_validate({field_name: val}).apply_to(base_pub)
                step_updates[field_name] = getattr(stepped_pub.to_internal(), field_name)
        else:
            abs_fields[field_name] = val

    merged = current_target.model_copy(update=step_updates) if step_updates else current_target
    if abs_fields:
        abs_update = MusicConfigUpdate.model_validate(abs_fields).to_internal()
        merged = merge_internal_config(merged, abs_update)
    return merged


# === Reactive Score engine ===


class Score:
    """Stateful local ambient music engine with smooth steering and live playback.

    Calling ``steer(text)`` updates the target mood idempotently (calling it once
    or ten times with the same text aims for the exact same destination without
    restarting an active glide) while audio smoothly converges over ``glide_seconds``.

    Examples:
        >>> score = Score("quiet morning desk", seed=7, block_seconds=1.0)
        >>> block = score.next_block()
        >>> len(block), block.dtype == np.float32
        (44100, True)
        >>> target = score.steer("rainy midnight coffee shop")
        >>> target.texture
        'vinyl_crackle'
    """

    def __init__(
        self,
        initial: str | ConfigInput | None = None,
        *,
        seed: int = 42,
        glide_seconds: float = 3.0,
        block_seconds: float = 1.5,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        if not math.isfinite(block_seconds) or block_seconds <= 0.1:
            raise InvalidConfigError("block_seconds must be finite and > 0.1")
        if not math.isfinite(glide_seconds) or glide_seconds < 0.0:
            raise InvalidConfigError("glide_seconds must be finite and >= 0.0")
        if sample_rate != SAMPLE_RATE:
            raise InvalidConfigError(f"Only sample_rate={SAMPLE_RATE} is supported")

        start_cfg = (
            _coerce_owned_target(initial) if initial is not None else MusicConfig().to_internal()
        )
        self._state_lock = threading.Lock()
        self._render_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()

        self._current: SynthConfig = start_cfg
        self._glide_from: SynthConfig = start_cfg
        self._target: SynthConfig = start_cfg
        self._glide_elapsed: float = 0.0
        self._glide_seconds: float = float(glide_seconds)
        self._block_seconds: float = float(block_seconds)
        self._sample_rate: int = int(sample_rate)
        self._session_seed: int = int(seed) & 0x7FFFFFFF
        self._drift_rng = np.random.default_rng(self._session_seed ^ 0x517CC1B7)
        self._drift_step: int = 0
        self._t_offset: float = 0.0
        self._prev_tail: NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._fifo: NDArray[np.float32] = np.empty(0, dtype=np.float32)

        self._session_stop: threading.Event | None = None
        self._fade_out_seconds: float = 0.35
        self._play_thread: threading.Thread | None = None

    @property
    def sample_rate(self) -> int:
        """Output sample rate in Hz (44100)."""
        return self._sample_rate

    @property
    def current_config(self) -> MusicConfig:
        """Return the closest public ``MusicConfig`` for the audible state."""
        with self._state_lock:
            return _internal_to_public(self._current)

    @property
    def target_config(self) -> MusicConfig:
        """Return the closest public ``MusicConfig`` for the target state."""
        with self._state_lock:
            return _internal_to_public(self._target)

    @property
    def internal_config(self) -> SynthConfig:
        """Return a defensive copy of the current numeric ``SynthConfig``."""
        with self._state_lock:
            return self._current.model_copy(deep=True)

    @property
    def is_playing(self) -> bool:
        """Return True while background playback is active."""
        with self._lifecycle_lock:
            thread = self._play_thread
            return thread is not None and thread.is_alive()

    def _set_target_locked(self, resolved: SynthConfig, glide_override: float | None = None) -> None:
        if glide_override is not None:
            self._glide_seconds = glide_override
        if resolved != self._target:
            self._glide_from = self._current
            self._target = resolved
            self._glide_elapsed = 0.0
        elif self._current == self._target:
            self._glide_from = self._target
            self._glide_elapsed = self._glide_seconds

    def steer(
        self,
        mood: str | ConfigInput,
        *,
        glide_seconds: float | None = None,
    ) -> MusicConfig:
        """Set a new target mood idempotently and glide smoothly toward it.

        Never blocks on audio rendering.

        Args:
            mood: Text prompt, ``MusicConfig``, or numeric ``SynthConfig``.
            glide_seconds: Optional override for transition glide duration.

        Returns:
            Public ``MusicConfig`` representing the resolved target.
        """
        if glide_seconds is not None and (not math.isfinite(glide_seconds) or glide_seconds < 0.0):
            raise InvalidConfigError("glide_seconds must be finite and >= 0.0")
        resolved = _coerce_owned_target(mood)
        with self._state_lock:
            self._set_target_locked(resolved, glide_seconds)
            return _internal_to_public(resolved)

    def nudge(
        self,
        update: MusicConfigUpdate | None = None,
        **fields: object,
    ) -> MusicConfig:
        """Adjust target fields with absolute values or ``Step`` deltas.

        Preserves all untouched continuous parameters cleanly.
        """
        if update is not None and fields:
            raise InvalidConfigError("Pass either a MusicConfigUpdate or keyword fields, not both")
        resolved_update = (
            update if update is not None else MusicConfigUpdate.model_validate(fields)
        )
        with self._state_lock:
            next_internal = _apply_nudge_preserving_continuous(self._target, resolved_update)
            self._set_target_locked(next_internal)
            return _internal_to_public(next_internal)

    def drift(self, *, rng: np.random.Generator | None = None) -> MusicConfig:
        """Glide one step to a harmonically neighboring mood along a smooth gradient."""
        gen = rng if rng is not None else self._drift_rng
        with self._state_lock:
            cur = self._target
            step_dir = 1 if float(gen.random()) >= 0.5 else -1
            next_root = (
                _neighbor_root(cur.root, step_dir) if float(gen.random()) < 0.6 else cur.root
            )
            modes = ("minor", "dorian", "major", "mixolydian")
            next_mode = (
                str(modes[int(gen.integers(0, len(modes)))])
                if float(gen.random()) < 0.35
                else cur.mode
            )
            next_tempo = _clamp(cur.tempo + float(gen.uniform(-0.08, 0.08)), 0.15, 0.85)
            next_bright = _clamp(cur.brightness + float(gen.uniform(-0.14, 0.14)), 0.12, 0.88)
            next_space = _clamp(cur.space + float(gen.uniform(-0.12, 0.12)), 0.20, 0.95)
            next_motion = _clamp(cur.motion + float(gen.uniform(-0.12, 0.12)), 0.12, 0.85)
            next_echo = _clamp(cur.echo + float(gen.uniform(-0.12, 0.12)), 0.15, 0.88)

            self._drift_step += 1
            if self._drift_step % 4 == 0:
                anchor_text = _DEFAULT_DRIFT_MOODS[
                    (self._drift_step // 4) % len(_DEFAULT_DRIFT_MOODS)
                ]
                anchor = mood_from_text(anchor_text).to_internal()
                drifted = interpolate_configs(cur, anchor, 0.45)
            else:
                drifted = cur.model_copy(
                    update={
                        "root": next_root,
                        "mode": next_mode,
                        "tempo": next_tempo,
                        "brightness": next_bright,
                        "space": next_space,
                        "motion": next_motion,
                        "echo": next_echo,
                    }
                )
            self._set_target_locked(drifted)
            return _internal_to_public(drifted)

    def _advance_config_snapshot(self, step_seconds: float) -> tuple[SynthConfig, float, int]:
        with self._state_lock:
            if self._glide_seconds <= 1e-6 or self._current == self._target:
                self._current = self._target
                self._glide_from = self._target
                self._glide_elapsed = self._glide_seconds
            else:
                self._glide_elapsed = min(self._glide_seconds, self._glide_elapsed + step_seconds)
                progress = self._glide_elapsed / self._glide_seconds
                if progress >= 1.0:
                    self._current = self._target
                    self._glide_from = self._target
                else:
                    self._current = interpolate_configs(self._glide_from, self._target, progress)
            return self._current, self._t_offset, self._session_seed

    def _render_block_locked(self, dur: float) -> FloatArray:
        cfg, t_offset, session_seed = self._advance_config_snapshot(dur)
        overlap_sec = min(_OVERLAP_SECONDS, dur * 0.35)
        overlap_n = max(16, int(round(overlap_sec * self._sample_rate)))
        block_n = max(1, int(round(dur * self._sample_rate)))
        render_dur = (block_n + overlap_n) / float(self._sample_rate)

        # Render outside _state_lock so live steer()/nudge() calls never stall!
        raw = assemble(
            cfg,
            duration=render_dur,
            normalize=False,
            rng=np.random.default_rng(session_seed),
            t_offset=t_offset,
        )
        calibrated = soft_limit(raw * _FIXED_SESSION_GAIN, ceiling=0.92)
        head: NDArray[np.float64] = np.array(calibrated[:block_n], dtype=np.float64, copy=True)
        tail: NDArray[np.float64] = np.array(
            calibrated[block_n : block_n + overlap_n], dtype=np.float64, copy=True
        )

        if self._prev_tail.size > 0:
            blend_n = min(int(head.size), int(self._prev_tail.size))
            fade_in = _raised_cosine_fade(blend_n)
            head[:blend_n] = (
                self._prev_tail[:blend_n] * (1.0 - fade_in) + head[:blend_n] * fade_in
            )

        self._prev_tail = tail
        with self._state_lock:
            self._t_offset += block_n / float(self._sample_rate)
        return ensure_audio_contract(head, sample_rate=self._sample_rate, check_peak=True)

    def fill(self, num_frames: int) -> FloatArray:
        """Pull an exact number of mono float32 frames sequentially across blocks."""
        if num_frames < 0:
            raise InvalidConfigError("num_frames must be >= 0")
        if num_frames == 0:
            return np.empty(0, dtype=np.float32)

        with self._render_lock:
            out: FloatArray = np.empty(num_frames, dtype=np.float32)
            written = 0
            while written < num_frames:
                if self._fifo.size == 0:
                    self._fifo = self._render_block_locked(self._block_seconds)
                take = min(num_frames - written, int(self._fifo.size))
                out[written : written + take] = self._fifo[:take]
                self._fifo = self._fifo[take:]
                written += take
            return out

    def next_block(self, seconds: float | None = None) -> FloatArray:
        """Pull ``seconds`` of seamless mono float32 audio through the canonical timeline."""
        dur = float(seconds) if seconds is not None else self._block_seconds
        if not math.isfinite(dur) or dur <= 0.05:
            raise InvalidConfigError("Block duration must be finite and > 0.05 seconds")
        frames = max(1, int(round(dur * self._sample_rate)))
        return self.fill(frames)

    def save(
        self,
        path: str | Path,
        *,
        duration: float = 16.0,
        fade_out_seconds: float = 0.6,
    ) -> Path:
        """Render ``duration`` seconds with a smooth fade-out and save as WAV."""
        if not math.isfinite(duration) or duration <= 0.0:
            raise InvalidConfigError("duration must be finite and > 0")
        frames = max(1, int(round(duration * self._sample_rate)))
        audio = self.fill(frames).astype(np.float64)
        faded = fade_into_nothingness(audio, fade_sec=fade_out_seconds, sr=self._sample_rate)
        return write_wav(path, faded, sample_rate=self._sample_rate)

    def _enqueue_until_abort(
        self,
        pcm_q: queue.Queue[FloatArray | None],
        chunk: FloatArray | None,
        abort_ev: threading.Event,
    ) -> bool:
        while not abort_ev.is_set():
            try:
                pcm_q.put(chunk, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def _run_playback_producer(
        self,
        pcm_q: queue.Queue[FloatArray | None],
        stop_ev: threading.Event,
        consumer_done_ev: threading.Event,
        block_size: int,
    ) -> None:
        while not stop_ev.is_set() and not consumer_done_ev.is_set():
            chunk = self.fill(block_size)
            if not self._enqueue_until_abort(pcm_q, chunk, stop_ev):
                break
        if consumer_done_ev.is_set():
            return
        fade_sec = self._fade_out_seconds
        fade_frames = max(block_size, int(round(fade_sec * self._sample_rate)))
        tail = self.fill(fade_frames).astype(np.float64)
        faded = fade_into_nothingness(
            tail, fade_sec=fade_sec, sr=self._sample_rate
        ).astype(np.float32)
        if not self._enqueue_until_abort(pcm_q, faded, consumer_done_ev):
            return
        self._enqueue_until_abort(pcm_q, None, consumer_done_ev)

    def _run_playback_consumer(
        self,
        sd: Any,
        pcm_q: queue.Queue[FloatArray | None],
        stop_ev: threading.Event,
        ready_ev: threading.Event,
        startup_err: list[BaseException],
        block_size: int,
    ) -> None:
        consumer_done_ev = threading.Event()
        prod = threading.Thread(
            target=self._run_playback_producer,
            args=(pcm_q, stop_ev, consumer_done_ev, block_size),
            name="latentscore-synth",
            daemon=True,
        )
        try:
            stream = sd.OutputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                blocksize=block_size,
            )
            stream.start()
        except BaseException as exc:
            startup_err.append(exc)
            ready_ev.set()
            return

        if stop_ev.is_set():
            stream.stop()
            stream.close()
            ready_ev.set()
            return

        prod.start()
        ready_ev.set()
        try:
            self._drain_playback_queue(stream, pcm_q, prod)
        finally:
            stop_ev.set()
            consumer_done_ev.set()
            prod.join(timeout=2.0)
            stream.stop()
            stream.close()

    def _drain_playback_queue(
        self,
        stream: Any,
        pcm_q: queue.Queue[FloatArray | None],
        prod: threading.Thread,
    ) -> None:
        while True:
            try:
                item = pcm_q.get(timeout=0.1)
            except queue.Empty:
                if not prod.is_alive() and pcm_q.empty():
                    break
                continue
            if item is None:
                break
            stream.write(item[:, np.newaxis])

    def _clear_session_if_current(self, thread: threading.Thread) -> None:
        with self._lifecycle_lock:
            if self._play_thread is thread:
                self._play_thread = None
                self._session_stop = None

    def play(self, *, block_size: int = 2048) -> Score:
        """Start non-blocking prebuffered playback in background threads.

        Propagates audio-device startup errors and timeouts synchronously.
        """
        with self._lifecycle_lock:
            if self._play_thread is not None and self._play_thread.is_alive():
                return self

            try:
                import sounddevice as sd_module  # type: ignore[import]
            except ImportError as exc:
                raise PlaybackError(
                    "Live Score playback requires sounddevice. Install sounddevice or use .save()."
                ) from exc

            safe_block = max(256, int(block_size))
            stop_ev = threading.Event()
            ready_ev = threading.Event()
            startup_err: list[BaseException] = []
            pcm_q: queue.Queue[FloatArray | None] = queue.Queue(maxsize=4)
            for _ in range(2):
                pcm_q.put(self.fill(safe_block))

            thread = threading.Thread(
                target=self._run_playback_consumer,
                args=(sd_module, pcm_q, stop_ev, ready_ev, startup_err, safe_block),
                name="latentscore-play",
                daemon=True,
            )
            self._session_stop = stop_ev
            self._play_thread = thread
            thread.start()

        started = ready_ev.wait(timeout=_PLAY_STARTUP_TIMEOUT_SECONDS)
        if not started:
            stop_ev.set()
            self._clear_session_if_current(thread)
            raise PlaybackError("Timed out waiting for audio output stream to start")
        if startup_err:
            self._clear_session_if_current(thread)
            raise PlaybackError(
                f"Failed to open audio output: {startup_err[0]}"
            ) from startup_err[0]
        return self

    def stop(self, *, fade_out_seconds: float = 0.35) -> None:
        """Fade background playback smoothly into silence and wait for clean exit."""
        with self._lifecycle_lock:
            stop_ev = self._session_stop
            thread = self._play_thread
            if thread is None or stop_ev is None:
                return
            self._fade_out_seconds = max(0.02, float(fade_out_seconds))
            stop_ev.set()

        thread.join(timeout=max(2.5, self._fade_out_seconds + 2.0))
        with self._lifecycle_lock:
            if self._play_thread is thread and not thread.is_alive():
                self._play_thread = None
                self._session_stop = None

    def __enter__(self) -> Score:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()


__all__ = [
    "Score",
    "compute_bar_aligned_duration",
    "fade_into_nothingness",
    "mood_from_text",
]
