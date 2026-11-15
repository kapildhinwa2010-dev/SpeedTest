"""Vendored Fourier helper for the CompressNet benchmark modules.

Provides ``fourier_basis``, the design matrix used by ``fourier_compress``:
a constant column plus sin/cos harmonics sampled over one full period.
Smooth (band-limited) matrices are reconstructed exactly when enough
coefficients are kept.
"""

from __future__ import annotations

import numpy as np

__all__ = ["fourier_basis"]


def fourier_basis(n_points: int, num_coeffs: int = 4) -> np.ndarray:
    """Fourier design matrix of shape ``(n_points, 2 * num_coeffs + 1)``.

    Columns are ``[1, sin(t), cos(t), sin(2t), cos(2t), ..., sin(k t), cos(k t)]``
    with ``t = linspace(0, 2*pi, n_points, endpoint=False)`` and
    ``k = num_coeffs``.

    Args:
        n_points: Number of sample points (columns of the target matrix).
        num_coeffs: Number of harmonics; higher keeps more detail.

    Returns:
        Design matrix suitable for ``numpy.linalg.lstsq`` fitting.

    Raises:
        ValueError: If ``n_points`` or ``num_coeffs`` is less than 1.
    """
    if n_points < 1:
        raise ValueError("n_points must be >= 1")
    if num_coeffs < 1:
        raise ValueError("num_coeffs must be >= 1")
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    cols = [np.ones(n_points)]
    for k in range(1, num_coeffs + 1):
        cols.append(np.sin(k * t))
        cols.append(np.cos(k * t))
    return np.stack(cols, axis=1)
