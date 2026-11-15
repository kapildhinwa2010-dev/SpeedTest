# SpeedTest - Timing tests for compression methods as a library
# Part of the Kapildhinwa2010-dev Toolkit.

"""Wall-clock timing suite for the four toolkit compression strategies.

Lineage: all toolkit repos. Measures single-call latency (median of
repeats) for SVD rank, quantization, Fourier basis and hybrid compression
on a given matrix, alongside the usual accuracy benchmark.

HONEST LIMITATIONS: timings are machine-dependent single-process Python
timings - they exclude serialization and decompression, and say nothing
about GPU or batched inference performance. Compare methods only within
one run on one machine.
"""

from __future__ import annotations

import time

import numpy as np

__all__ = [
    "svd_compress",
    "quantize",
    "fourier_compress",
    "hybrid_compress",
    "benchmark",
    "time_methods",
    "speed_report",
]


def _check(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("matrix must be non-empty and 2-D")
    return matrix


def svd_compress(matrix: np.ndarray, rank: int = 4) -> np.ndarray:
    matrix = _check(matrix)
    if rank < 1:
        raise ValueError("rank must be >= 1")
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    rank = min(rank, s.size)
    return (u[:, :rank] * s[:rank]) @ vt[:rank, :]


def quantize(matrix: np.ndarray, bits: int = 8) -> np.ndarray:
    matrix = _check(matrix)
    if bits < 1 or bits > 16:
        raise ValueError("bits must be in [1, 16]")
    lo, hi = float(matrix.min()), float(matrix.max())
    levels = 2 ** bits - 1
    scale = (hi - lo) / levels if hi > lo else 1.0
    codes = np.clip(np.round((matrix - lo) / scale), 0, levels)
    return codes * scale + lo


def fourier_compress(matrix: np.ndarray, num_coeffs: int = 4) -> np.ndarray:
    matrix = _check(matrix)
    if num_coeffs < 1:
        raise ValueError("num_coeffs must be >= 1")
    t = np.linspace(0.0, 2.0 * np.pi, matrix.shape[1], endpoint=False)
    cols = [np.ones_like(t)]
    for k in range(1, num_coeffs + 1):
        cols.append(np.cos(k * t))
        cols.append(np.sin(k * t))
    basis = np.stack(cols, axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(basis, matrix.T, rcond=None)
    return coeffs.T @ basis.T


def hybrid_compress(matrix: np.ndarray, rank: int = 4, bits: int = 8) -> np.ndarray:
    """SVD low-rank followed by quantization."""
    return quantize(svd_compress(matrix, rank=rank), bits=bits)


def _methods():
    return {
        "svd_rank4": lambda m: svd_compress(m, rank=4),
        "quantize8": lambda m: quantize(m, bits=8),
        "fourier4": lambda m: fourier_compress(m, num_coeffs=4),
        "hybrid": lambda m: hybrid_compress(m, rank=4, bits=8),
    }


def benchmark(matrix: np.ndarray) -> list:
    """Run all methods on the CLEAN matrix; error vs the matrix itself."""
    matrix = _check(matrix)
    results = []
    norm = float(np.linalg.norm(matrix))
    for name, fn in _methods().items():
        approx = fn(matrix)
        rel_error = float(np.linalg.norm(matrix - approx)) / norm if norm else 0.0
        results.append({
            "method": name,
            "relative_error": rel_error,
            "passed": rel_error < 0.5,
        })
    return results


def time_methods(matrix: np.ndarray, repeats: int = 5) -> dict:
    """Wall-clock time of each compression method (median of ``repeats``).

    Returns {method: {"median_seconds": s, "min_seconds": s}}. Times are
    single-call latencies on the given matrix size; they do NOT include
    serialization or decompression, and absolute numbers depend on the
    machine. Compare methods only within one run.
    """
    matrix = _check(matrix)
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    timing = {}
    for name, fn in _methods().items():
        samples = []
        for _ in range(int(repeats)):
            start = time.perf_counter()
            fn(matrix)
            samples.append(time.perf_counter() - start)
        samples_sorted = sorted(samples)
        mid = len(samples_sorted) // 2
        median = samples_sorted[mid] if len(samples_sorted) % 2 else (samples_sorted[mid - 1] + samples_sorted[mid]) / 2.0
        timing[name] = {"median_seconds": median, "min_seconds": samples_sorted[0]}
    return timing


def speed_report(matrix: np.ndarray, repeats: int = 5) -> dict:
    """Combine benchmark errors with timing; flags the fastest method.

    "fastest" is by median wall-clock time only - it says nothing about
    reconstruction quality; check "benchmark" for accuracy trade-offs.
    """
    timing = time_methods(matrix, repeats=repeats)
    fastest = min(timing, key=lambda n: timing[n]["median_seconds"])
    return {"timing": timing, "fastest": fastest, "errors": benchmark(matrix)}