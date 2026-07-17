# Installation

## Requirements

- Python 3.10 or newer
- The only runtime dependencies are **NumPy** and **Numba** — both are
  installed automatically.

## Install

```bash
pip install numba-utils
```

That is the whole story: no compilers to configure, no optional native
extensions. Numba brings its own LLVM-based compiler.

## Verify

```python
import numba_utils as nu
print(nu.__version__)
```

## The cache kill-switch

Numba caches compiled kernels to disk. On multi-process farms, network
filesystems, and ephemeral containers that cache can be dangerous — see
[Cache](../numba-cache.md) for the full story. If you hit crashes that
look like they come from a stale or shared cache, disable it globally:

```bash
NUMBA_UTILS_CACHE=0        # environment
```

```python
import numba_utils as nu
nu.configure(cache=False)  # or from code
```

This override wins even over per-call `cache=True`, by design: it exists
as an environment policy tool, and a policy any call site can defeat is
not a policy.

## Development install

```bash
git clone https://github.com/nicoseijas/numba-utils.git
cd numba-utils
python -m venv .venv
.venv/Scripts/pip install -e .[dev]     # POSIX: .venv/bin/pip
.venv/Scripts/python -m pytest
```

See [Contributing](../contributing.md) for the rules — benchmarks are
mandatory, honesty is policy.
