"""Build static browser assets for zero-backend LatentScore hosting.

Bundles ``latentscore/synth.py`` directly from this repository with the pure-NumPy
SciPy shim (``demo/scripts/scipy_shim.py``) so Pyodide browser workers execute
the exact same repaired DSP as native Python with zero external patch drift.

File map:
- Types and constants: repo paths and Pyodide header/shim wrappers.
- Pyodide synth bundler: combines ``scipy_shim.py`` + ``latentscore/synth.py``.
- CLI entry point: writes ``demo/frontend/public/py/latentscore_synth.py``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

# === Types and constants ===

DEMO_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = DEMO_DIR.parent
DEFAULT_SYNTH_SOURCE = REPO_ROOT / "latentscore" / "synth.py"
DEFAULT_SHIM_PATH = DEMO_DIR / "scripts" / "scipy_shim.py"
DEFAULT_OUT_PATH = DEMO_DIR / "frontend" / "public" / "py" / "latentscore_synth.py"

_SYNTH_HEADER = '''"""Self-contained Pyodide synth module for LatentScore browser mode.

Generated directly from latentscore/synth.py with a pure-NumPy/JS-biquad SciPy
shim so browser workers run exact LatentScore DSP with Pyodide core + NumPy.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

'''

_JS_BIQUAD_HOOK = '''
# === Optional JS host fast-path for 2nd-order biquad lfilter ===

_JS_BIQUAD_FN: Any = None


def _resolve_js_biquad() -> Any:
    """Look up globalThis._latentscore_biquad when running inside Pyodide."""
    global _JS_BIQUAD_FN
    if _JS_BIQUAD_FN is not None:
        return _JS_BIQUAD_FN
    try:
        import js  # type: ignore[import-not-found]

        fn = getattr(js, "_latentscore_biquad", None)
        if fn is not None:
            _JS_BIQUAD_FN = fn
            return fn
    except ImportError:
        return None
    return None


def _pyodide_biquad_kernel(
    arr: NDArray[np.float64],
    b0: float,
    b1: float,
    b2: float,
    a1: float,
    a2: float,
) -> NDArray[np.float64]:
    """Run biquad recurrence in JS Float64Array if available, else Python."""
    js_fn = _resolve_js_biquad()
    if js_fn is None:
        return _biquad_python(arr, b0, b1, b2, a1, a2)
    from pyodide.ffi import to_js  # type: ignore[import-not-found]

    result = js_fn(to_js(arr), b0, b1, b2, a1, a2)
    return np.asarray(result.to_py(), dtype=np.float64)


def lfilter(
    b: NDArray[np.float64],
    a: NDArray[np.float64],
    signal: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Drop-in scipy.signal.lfilter using JS host biquad inside Pyodide."""
    return _shim_lfilter(b, a, signal, kernel=_pyodide_biquad_kernel)
'''

_SYNTH_CONFIG_STUB = '''
# === Lightweight SynthConfig (zero Pydantic runtime dependency) ===

AccentStyle: TypeAlias = str
AttackStyle: TypeAlias = str
BassStyle: TypeAlias = str
ChordExtensions: TypeAlias = str
DensityLevel: TypeAlias = int
GrainStyle: TypeAlias = str
ModeName: TypeAlias = str
PadStyle: TypeAlias = str
RhythmStyle: TypeAlias = str
RootNote: TypeAlias = str
TextureStyle: TypeAlias = str


@dataclass(frozen=True, slots=True)
class SynthConfig:
    """Numeric synthesis configuration matching latentscore._MusicConfigInternal."""

    tempo: float = 0.35
    root: str = "c"
    mode: str = "minor"
    brightness: float = 0.5
    space: float = 0.6
    density: int = 5
    bass: str = "drone"
    pad: str = "warm_slow"
    melody: str = "contemplative"
    rhythm: str = "minimal"
    texture: str = "shimmer"
    accent: str = "bells"
    motion: float = 0.5
    attack: str = "medium"
    stereo: float = 0.5
    depth: bool = False
    echo: float = 0.5
    human: float = 0.0
    grain: str = "clean"
    melody_engine: str = "pattern"
    phrase_len_bars: int = 4
    melody_density: float = 0.45
    syncopation: float = 0.20
    swing: float = 0.0
    motif_repeat_prob: float = 0.50
    step_bias: float = 0.75
    chromatic_prob: float = 0.05
    cadence_strength: float = 0.65
    register_min_oct: int = 4
    register_max_oct: int = 6
    tension_curve: str = "arc"
    harmony_style: str = "auto"
    chord_change_bars: int = 1
    chord_extensions: str = "triads"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SynthConfig:
        """Build a SynthConfig from a JSON dictionary, ignoring unknown keys."""
        valid = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


class _SoundFileStub:
    """Placeholder so unused file-export helpers keep module structure intact."""

    @staticmethod
    def write(*_args: object, **_kwargs: object) -> None:
        raise NotImplementedError("File export happens in browser Web Audio")


sf = _SoundFileStub()
'''

_RENDER_ENTRYPOINT = '''
# === Browser worker entry points ===


def render_pcm_f32(
    config_json: str,
    duration: float = 60.0,
    t_offset: float = 0.0,
    normalize: bool = True,
    seed: int | None = None,
) -> NDArray[np.float32]:
    """Render mono float32 PCM samples from an internal SynthConfig JSON string."""
    raw = json.loads(config_json)
    cfg = SynthConfig.from_dict(raw)
    if seed is not None:
        rng = np.random.default_rng(seed)
        np.random.seed(int(seed) & 0xFFFFFFFF)
    else:
        rng = np.random.default_rng()
    samples = assemble(
        cfg,
        duration=float(duration),
        normalize=bool(normalize),
        rng=rng,
        t_offset=float(t_offset),
    )
    arr = np.asarray(samples, dtype=np.float32)
    return np.clip(arr, -1.0, 1.0)
'''


# === Pyodide synth bundler ===


def _extract_shim_body(shim_source: str) -> str:
    marker = "# === Types and constants ==="
    idx = shim_source.find(marker)
    assert idx != -1, "SciPy shim marker missing"
    body = shim_source[idx:]
    return body.replace("def lfilter(", "def _shim_lfilter(", 1)


def _strip_synth_imports(synth_source: str) -> str:
    marker = "MusicConfig: TypeAlias = SynthConfig"
    idx = synth_source.find(marker)
    assert idx != -1, f"Expected marker {marker!r} in synth.py"
    return synth_source[idx:]


def build_pyodide_synth_module(
    synth_path: Path = DEFAULT_SYNTH_SOURCE,
    shim_path: Path = DEFAULT_SHIM_PATH,
) -> str:
    """Assemble standalone Pyodide synth Python source from latentscore/synth.py."""
    synth_source = synth_path.read_text(encoding="utf-8")
    shim_source = shim_path.read_text(encoding="utf-8")
    parts = (
        _SYNTH_HEADER,
        _extract_shim_body(shim_source),
        _JS_BIQUAD_HOOK,
        _SYNTH_CONFIG_STUB,
        _strip_synth_imports(synth_source),
        _RENDER_ENTRYPOINT,
    )
    return "\n".join(parts)


# === CLI entry point ===


def main(argv: Sequence[str] | None = None) -> None:
    """Regenerate demo/frontend/public/py/latentscore_synth.py from repo synth.py."""
    parser = argparse.ArgumentParser(description="Bundle latentscore/synth.py for Pyodide.")
    parser.add_argument("--synth", type=Path, default=DEFAULT_SYNTH_SOURCE)
    parser.add_argument("--shim", type=Path, default=DEFAULT_SHIM_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args(argv)

    source = build_pyodide_synth_module(synth_path=args.synth, shim_path=args.shim)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(source, encoding="utf-8")
    print(f"Wrote {args.out} ({len(source.encode('utf-8')) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
