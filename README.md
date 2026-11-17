# SpeedTest

SpeedTest is a small Python library for comparing four matrix-compression approximations by reconstruction error and single-call wall-clock time. It is an experimental benchmarking utility, not a benchmark of end-to-end model or inference performance.

## What it implements

The functions in `speedtest.py` accept a non-empty, two-dimensional NumPy matrix (converted to `float64`) and return an approximation or report:

- `svd_compress(matrix, rank=4)` keeps a truncated singular-value decomposition.
- `quantize(matrix, bits=8)` uniformly quantizes values over the matrix's minimum-to-maximum range.
- `fourier_compress(matrix, num_coeffs=4)` fits a constant-plus-sine/cosine basis to each row by least squares.
- `hybrid_compress(matrix, rank=4, bits=8)` applies truncated SVD followed by quantization.
- `benchmark(matrix)` returns each method's relative Frobenius reconstruction error and a `passed` flag (error below the code's fixed `0.5` threshold).
- `time_methods(matrix, repeats=5)` reports median and minimum elapsed seconds per method; `speed_report(...)` combines these timings with the error results and names the lowest-median-time method.

The `fastest` field describes timing only; it does not indicate the most accurate method. `activation_compress_helpers.py` separately exposes `fourier_basis(...)`, a NumPy helper for constructing a Fourier design matrix.

## Requirements and usage

The project metadata declares Python 3.9 or newer and NumPy 1.24 or newer. From the repository root, install the declared dependencies:

```bash
python -m pip install -r requirements.txt
```

There is no configured command-line entry point. Use the functions from Python, for example from the repository root:

```python
import numpy as np
from speedtest import speed_report

matrix = np.random.default_rng(0).normal(size=(64, 32))
report = speed_report(matrix, repeats=5)
print("Fastest by median time:", report["fastest"])
for method, timing in report["timing"].items():
    print(method, timing["median_seconds"], "seconds")
for result in report["errors"]:
    print(result["method"], result["relative_error"], result["passed"])
```

The timings are measurements for this run and machine; the example does not imply a performance result.

## Tests and CI

Run the test suite from the repository root with:

```bash
python -m pytest -q
```

The repository's latest recorded GitHub Actions CI run for commit `c9d9af2f74ca9e2e34c2e209b23737d0d5b5349b` is marked successful. The CI workflow tests Python 3.10, 3.11, and 3.12, and also compiles the test/module files and performs an SVD smoke check. The latest CodeQL workflow run for that same commit is also marked successful. These are workflow results, not an independently confirmed test run on this machine.

## Limitations and scope

- Timing uses `time.perf_counter()` around individual compression calls. It is machine-dependent, single-process timing and excludes serialization and decompression; it does not measure GPU, batched, or end-to-end inference performance.
- `benchmark()` compares each approximation with the input matrix itself and uses a fixed relative-error threshold. It does not evaluate downstream model accuracy or establish a generally suitable quality threshold.
- Tests include synthetic smooth matrices and input-validation cases; the repository does not provide a published benchmark dataset, measured comparative results, plotting, or a benchmark-result export workflow.
- The code operates on in-memory two-dimensional NumPy matrices. It does not provide model/framework integration, data loading, a CLI, or a service/API.

## License

The repository includes an MIT License (`LICENSE`), copyright 2026 Kapildhinwa2010-dev.
