"""Pure-NumPy SciPy signal shim for running LatentScore synthesis in Pyodide.

Replaces the three scipy.signal functions used by latentscore/synth.py
(2nd-order butter, biquad lfilter, and FIR zero-phase decimate) so browser
WASM builds load only Pyodide core + NumPy instead of a ~30 MB SciPy wheel.

File map:
- Types and constants: immutable filter types and numerical guards.
- Filter design: exact bilinear-transform 2nd-order Butterworth coefficients.
- Filtering and decimation: biquad IIR filter with optional JS host fast path
  and Hamming-windowed sinc FIR decimation.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

# === Types and constants ===

Float64Array: TypeAlias = NDArray[np.float64]
FilterKind: TypeAlias = Literal["low", "high"]
BiquadKernel: TypeAlias = Callable[
    [Float64Array, float, float, float, float, float], Float64Array
]

_SQRT2 = math.sqrt(2.0)
_MIN_CUTOFF = 1e-6
_MAX_CUTOFF = 1.0 - 1e-6

# === Filter design ===


def butter(
    order: int,
    wn: float,
    btype: FilterKind = "low",
    output: Literal["ba"] = "ba",
) -> tuple[Float64Array, Float64Array]:
    """Compute exact 2nd-order digital Butterworth biquad coefficients.

    Args:
        order: Filter order (must be 2 for latentscore).
        wn: Cutoff frequency normalized to Nyquist (0 < wn < 1).
        btype: Either ``"low"`` or ``"high"``.
        output: Must be ``"ba"``.

    Returns:
        Tuple ``(b, a)`` of length-3 float64 coefficient arrays with ``a[0] == 1``.

    Raises:
        ValueError: If order, output, btype, or cutoff is outside supported bounds.

    Examples:
        >>> b, a = butter(2, 0.25, btype="low", output="ba")
        >>> round(float(a[0]), 6)
        1.0
        >>> round(float(b.sum() / a.sum()), 6)
        1.0
    """
    if order != 2:
        raise ValueError(f"Only 2nd-order Butterworth is supported, got {order}")
    if output != "ba":
        raise ValueError(f"Only 'ba' output is supported, got {output!r}")
    if not (_MIN_CUTOFF <= wn <= _MAX_CUTOFF):
        raise ValueError(f"Cutoff wn must lie in (0, 1), got {wn}")

    k = math.tan(math.pi * wn * 0.5)
    k2 = k * k
    norm = 1.0 / (1.0 + _SQRT2 * k + k2)

    a1 = 2.0 * (k2 - 1.0) * norm
    a2 = (1.0 - _SQRT2 * k + k2) * norm
    a = np.array([1.0, a1, a2], dtype=np.float64)

    match btype:
        case "low":
            b0 = k2 * norm
            b = np.array([b0, 2.0 * b0, b0], dtype=np.float64)
        case "high":
            b0 = norm
            b = np.array([b0, -2.0 * b0, b0], dtype=np.float64)
        case _:
            raise ValueError(f"Unsupported btype: {btype!r}")

    return b, a


# === Filtering and decimation ===


def _biquad_python(
    signal: Float64Array,
    b0: float,
    b1: float,
    b2: float,
    a1: float,
    a2: float,
) -> Float64Array:
    """Direct-Form-II Transposed biquad reference loop in pure Python."""
    out = np.empty(len(signal), dtype=np.float64)
    z1 = 0.0
    z2 = 0.0
    for idx, xi in enumerate(signal.tolist()):
        yi = b0 * xi + z1
        z1 = b1 * xi - a1 * yi + z2
        z2 = b2 * xi - a2 * yi
        out[idx] = yi
    return out


def lfilter(
    b: Float64Array,
    a: Float64Array,
    signal: Float64Array,
    *,
    kernel: BiquadKernel | None = None,
) -> Float64Array:
    """Apply a causal 2nd-order IIR biquad filter to a 1-D signal.

    Args:
        b: Numerator coefficients of length 3.
        a: Denominator coefficients of length 3 with non-zero ``a[0]``.
        signal: Input 1-D array.
        kernel: Optional fast-path biquad implementation (e.g. JS host hook).

    Returns:
        Filtered 1-D float64 array with the same length as ``signal``.

    Raises:
        ValueError: If coefficients are not length 3 or ``a[0] == 0``.

    Examples:
        >>> b, a = butter(2, 0.5, btype="low")
        >>> out = lfilter(b, a, np.ones(16, dtype=np.float64))
        >>> len(out)
        16
    """
    if len(b) != 3 or len(a) != 3:
        raise ValueError("lfilter shim expects 2nd-order biquad coefficients (len 3)")
    a0 = float(a[0])
    if a0 == 0.0:
        raise ValueError("Denominator a[0] must be non-zero")

    arr = np.asarray(signal, dtype=np.float64)
    if arr.size == 0:
        return np.empty(0, dtype=np.float64)

    inv_a0 = 1.0 / a0
    b0 = float(b[0]) * inv_a0
    b1 = float(b[1]) * inv_a0
    b2 = float(b[2]) * inv_a0
    a1 = float(a[1]) * inv_a0
    a2 = float(a[2]) * inv_a0

    runner = kernel if kernel is not None else _biquad_python
    return runner(arr, b0, b1, b2, a1, a2)


def _fir_lowpass_taps(num_taps: int, cutoff: float) -> Float64Array:
    """Build Hamming-windowed sinc FIR lowpass taps normalized to unit DC gain."""
    half = (num_taps - 1) * 0.5
    n = np.arange(num_taps, dtype=np.float64) - half
    sinc_arg = cutoff * n
    h = np.sinc(sinc_arg) * np.hamming(num_taps)
    total = float(np.sum(h))
    assert total != 0.0, "FIR filter tap sum must be non-zero"
    return np.asarray(h / total, dtype=np.float64)


def _convolve_same_reflect(signal: Float64Array, taps: Float64Array) -> Float64Array:
    """Convolve signal with symmetric FIR taps using odd reflection padding."""
    pad = len(taps) // 2
    if len(signal) <= 1 or pad == 0:
        return signal.copy()
    edge = min(pad, len(signal) - 1)
    left = 2.0 * signal[0] - signal[edge:0:-1]
    right = 2.0 * signal[-1] - signal[-2 : -(edge + 2) : -1]
    padded = np.concatenate((left, signal, right))
    if edge < pad:
        padded = np.pad(padded, (pad - edge, pad - edge), mode="edge")
    return np.asarray(np.convolve(padded, taps, mode="valid"), dtype=np.float64)


def decimate(
    signal: Float64Array,
    factor: int,
    *,
    ftype: Literal["fir"] = "fir",
    zero_phase: bool = True,
) -> Float64Array:
    """Downsample a 1-D signal by an integer factor with FIR anti-aliasing.

    Args:
        signal: Input 1-D array.
        factor: Positive integer downsampling factor (e.g. 2).
        ftype: Must be ``"fir"``.
        zero_phase: If True, filters forward and backward for zero phase shift.

    Returns:
        Downsampled 1-D float64 array.

    Raises:
        ValueError: If factor < 1 or ftype != ``"fir"``.

    Examples:
        >>> x = np.ones(20, dtype=np.float64)
        >>> y = decimate(x, 2, ftype="fir", zero_phase=True)
        >>> len(y), round(float(y.mean()), 6)
        (10, 1.0)
    """
    if factor < 1:
        raise ValueError(f"Decimation factor must be >= 1, got {factor}")
    if ftype != "fir":
        raise ValueError(f"Only ftype='fir' is supported, got {ftype!r}")

    arr = np.asarray(signal, dtype=np.float64)
    if factor == 1 or arr.size == 0:
        return arr.copy()

    num_taps = 20 * factor + 1
    taps = _fir_lowpass_taps(num_taps, 1.0 / factor)
    if zero_phase:
        # Symmetric odd-length FIR convolved with centered reflect padding is
        # already exact zero-phase (matching scipy.signal.decimate ftype="fir").
        filtered = _convolve_same_reflect(arr, taps)
    else:
        raw_conv = np.convolve(arr, taps, mode="full")[: len(arr)]
        filtered = np.asarray(raw_conv, dtype=np.float64)
    return np.asarray(filtered[::factor], dtype=np.float64)
