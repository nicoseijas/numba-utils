# Numba's on-disk cache

`cache=True` (the numba-utils default) stores compiled binaries next to
the source (`__pycache__/*.nbi` + `*.nbc`), keyed by signature and
compilation flags. Later runs load the binary instead of recompiling —
for scripts this turns a seconds-long warmup into milliseconds. This is
what Numba recommends, and it is the right default for most users.

This page exists because of the environments where it is NOT safe.

## Where the cache breaks

The cache assumes a stored binary is valid for any process that loads
it. In practice, these setups have produced **intermittent crashes**
(access violation `0xC0000005` on Windows, SIGSEGV on Linux) when a
process loads a binary compiled by a *different* process:

- **Multi-process worker farms** — many workers importing the same
  module, one of them compiled the cache, the rest load it. This has
  crashed even on pure, non-parallel kernels.
- **Network / shared filesystems** — the cache directory is shared
  between machines that don't share an ABI.
- **Ephemeral container storage** — stale caches surviving image layers
  or volume mounts they weren't compiled in.

## Recognizing it

The failure signature is distinctive:

- The **first** run (the one that compiled) works; later runs crash.
- Crashes look **random** — same code, same data, different outcome.
- Deleting `__pycache__` "fixes" it… until the next run repopulates it.
- The crash is a hard process death (no Python traceback).

If that pattern matches, stop debugging your kernel: it's the cache.

## The fix

Disable caching globally — this is a policy decision about the
environment, so it overrides everything, including explicit
`cache=True` arguments:

```python
import numba_utils as nu
nu.configure(cache=False)          # in code, before defining kernels
```

```
NUMBA_UTILS_CACHE=0                # or from the environment (CI, farms)
```

Cost: each process recompiles fresh (seconds to a couple of minutes for
large kernels). For a worker farm, amortize by running long-lived worker
processes rather than fanning out a process per task.

Hygiene for affected machines: clear `__pycache__` before runs, and
avoid chaining multiple Numba-heavy script invocations in one shell.

`diagnostics.check(fn)` warns when a function has caching enabled and
tells you exactly this.

## Trade-off summary

| Environment | Recommendation |
| --- | --- |
| Single-process scripts, notebooks, dev loops | Keep the default (`cache=True`) |
| Multi-process farms, shared FS, containers | `configure(cache=False)` / `NUMBA_UTILS_CACHE=0` |
