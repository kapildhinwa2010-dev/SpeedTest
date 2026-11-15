import numpy as np
import pytest

from speedtest import (
    benchmark,
    fourier_compress,
    hybrid_compress,
    quantize,
    svd_compress,
)


def _smooth(rows=16, cols=64, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2.0 * np.pi, cols, endpoint=False)
    amps = rng.standard_normal((rows, 3)) * 0.5
    return amps[:, 0:1] + amps[:, 1:2] * np.sin(t) + amps[:, 2:3] * np.cos(2 * t)


def test_svd_low_rank_exact():
    A = _smooth()
    assert np.linalg.norm(A - svd_compress(A, rank=8)) < 1e-8


def test_svd_rejects_bad_rank():
    with pytest.raises(ValueError):
        svd_compress(np.eye(4), rank=0)


def test_quantize_error_bounded():
    A = _smooth()
    err = np.abs(A - quantize(A, bits=8)).max()
    span = A.max() - A.min()
    assert err <= span / 256 + 1e-12


def test_quantize_rejects_bad_bits():
    with pytest.raises(ValueError):
        quantize(np.ones((2, 2)), bits=0)


def test_fourier_exact_for_smooth():
    A = _smooth()
    err = np.linalg.norm(A - fourier_compress(A, num_coeffs=4)) / np.linalg.norm(A)
    assert err < 1e-6


def test_hybrid_reasonable():
    A = _smooth()
    err = np.linalg.norm(A - hybrid_compress(A, rank=8, bits=12)) / np.linalg.norm(A)
    assert err < 0.05


def test_benchmark_returns_all_methods():
    results = benchmark(_smooth())
    names = {r["method"] for r in results}
    assert names == {"svd_rank4", "quantize8", "fourier4", "hybrid"}


def test_benchmark_entries_have_error_and_pass():
    for entry in benchmark(_smooth()):
        assert 0.0 <= entry["relative_error"]
        assert isinstance(entry["passed"], bool)


def test_benchmark_rejects_1d():
    with pytest.raises(ValueError):
        benchmark(np.ones(5))


def test_benchmark_low_error_for_smooth():
    results = benchmark(_smooth(rows=32, cols=128))
    assert all(r["relative_error"] < 0.5 for r in results)


def test_time_methods_returns_all():
    timing = time_methods(_smooth(), repeats=2)
    assert set(timing) == {"svd_rank4", "quantize8", "fourier4", "hybrid"}
    for entry in timing.values():
        assert entry["median_seconds"] >= 0.0
        assert entry["min_seconds"] >= 0.0


def test_time_methods_rejects_bad_repeats():
    with pytest.raises(ValueError):
        time_methods(_smooth(), repeats=0)


def test_speed_report_has_fastest():
    report = speed_report(_smooth(), repeats=2)
    assert report["fastest"] in {"svd_rank4", "quantize8", "fourier4", "hybrid"}
    assert len(report["errors"]) == 4


def test_speed_report_rejects_1d():
    with pytest.raises(ValueError):
        speed_report(np.ones(5))