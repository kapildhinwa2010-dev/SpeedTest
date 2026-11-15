# SpeedTest

Compression benchmark suite.

**Part of the [Kapildhinwa2010-dev Toolkit](https://github.com/kapildhinwa2010-dev/Kapildhinwa2010-dev).**

![CI](https://github.com/kapildhinwa2010-dev/SpeedTest/actions/workflows/ci.yml/badge.svg)
![CodeQL](https://github.com/kapildhinwa2010-dev/SpeedTest/actions/workflows/codeql.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Features

- Four strategies: SVD rank, quantization, Fourier basis, hybrid
- Uniform relative-error reporting
- One-call `benchmark()` over all methods

## 📦 Install

```bash
pip install -r requirements.txt
```

## 🚀 Quick start

```python
from speedtest import benchmark

for result in benchmark(W):
    print(result['method'], result['relative_error'])
```

## 🧪 Tests

```bash
pytest tests/ -v
```

## 📄 License

MIT — see [LICENSE](LICENSE).
