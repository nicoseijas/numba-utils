import importlib
import os
import sys

import numpy as np
import pytest
from numba import njit

from numba_utils import diagnostics, njit_fast, njit_parallel


def _make_compiled():
    @njit_fast
    def scaled_sum(arr):
        acc = 0.0
        for x in arr:
            acc += x
        return acc * 2.0

    scaled_sum(np.arange(10, dtype=np.float64))
    return scaled_sum


class TestInspect:
    def test_reports_flags_and_signatures(self):
        fn = _make_compiled()
        report = diagnostics.inspect(fn)
        assert report.name == "scaled_sum"
        assert len(report.signatures) == 1
        assert report.fastmath is True
        assert report.nogil is True
        assert report.parallel is False
        assert report.cache_enabled is True

    def test_uncompiled_has_no_signatures(self):
        @njit
        def untouched(x):
            return x

        report = diagnostics.inspect(untouched)
        assert report.signatures == ()

    def test_non_dispatcher_raises(self):
        with pytest.raises(TypeError):
            diagnostics.inspect(sorted)

    def test_report_is_immutable(self):
        report = diagnostics.inspect(_make_compiled())
        with pytest.raises(AttributeError):
            report.name = "other"


class TestShow:
    def test_contains_key_facts(self, capsys):
        fn = _make_compiled()
        text = diagnostics.show(fn, verbose=False)
        assert capsys.readouterr().out == ""
        assert "scaled_sum" in text
        assert "Cache: enabled" in text
        assert "Fastmath: yes" in text
        assert "Parallel: no" in text

    def test_verbose_prints(self, capsys):
        diagnostics.show(_make_compiled())
        assert "scaled_sum" in capsys.readouterr().out


class TestCheck:
    def test_cache_warning_with_recommendation(self):
        warnings = diagnostics.check(_make_compiled(), verbose=False)
        cache_warnings = [w for w in warnings if "NUMBA_UTILS_CACHE=0" in w]
        assert len(cache_warnings) == 1

    def test_fastmath_warning(self):
        warnings = diagnostics.check(_make_compiled(), verbose=False)
        assert any("fastmath" in w for w in warnings)

    def test_parallel_warning(self):
        @njit_parallel
        def pfn(arr):
            return arr.sum()

        warnings = diagnostics.check(pfn, verbose=False)
        assert any("prange" in w for w in warnings)

    def test_uncompiled_suggests_warmup(self):
        @njit
        def cold(x):
            return x

        warnings = diagnostics.check(cold, verbose=False)
        assert any("warmup" in w for w in warnings)

    def test_plain_njit_is_clean(self):
        @njit
        def plain(x):
            return x + 1

        plain(1.0)
        assert diagnostics.check(plain, verbose=False) == []

    def test_verbose_prints_warnings(self, capsys):
        diagnostics.check(_make_compiled())
        assert "diagnostics.check" in capsys.readouterr().out


_PROBE = "shadow_probe"
_PROBE_PKG = "shadow_probe_pkg"


@pytest.fixture
def two_copies(tmp_path):
    """Two directories holding the same module name; the first is imported."""
    first, second = tmp_path / "first", tmp_path / "second"
    for directory, marker in ((first, "first"), (second, "second")):
        directory.mkdir()
        (directory / f"{_PROBE}.py").write_text(f"VALUE = {marker!r}\n")
        package = directory / _PROBE_PKG
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "sub.py").write_text(f"VALUE = {marker!r}\n")
    original_path = list(sys.path)
    sys.path.insert(0, str(first))
    # FileFinder caches directory listings; without this the finder can miss
    # files created after the interpreter last looked at the directory.
    importlib.invalidate_caches()
    try:
        yield first, second
    finally:
        sys.path[:] = original_path
        for name in list(sys.modules):
            if name == _PROBE or name.split(".")[0] == _PROBE_PKG:
                del sys.modules[name]
        importlib.invalidate_caches()


def _repoint(entry):
    sys.path[0] = str(entry)
    importlib.invalidate_caches()


class TestShadowed:
    def test_detects_a_module_the_path_no_longer_resolves(self, two_copies):
        first, second = two_copies
        module = importlib.import_module(_PROBE)
        _repoint(second)

        findings = diagnostics.shadowed(module, verbose=False)

        assert len(findings) == 1
        assert findings[0].name == _PROBE
        assert os.path.realpath(findings[0].loaded_origin).startswith(
            os.path.realpath(str(first))
        )
        assert os.path.realpath(findings[0].resolved_origin).startswith(
            os.path.realpath(str(second))
        )

    def test_consistent_module_is_not_reported(self, two_copies):
        module = importlib.import_module(_PROBE)
        assert diagnostics.shadowed(module, verbose=False) == []

    def test_scan_finds_it_without_being_told_where_to_look(self, two_copies):
        _, second = two_copies
        importlib.import_module(_PROBE)
        _repoint(second)

        names = [finding.name for finding in diagnostics.shadowed(verbose=False)]

        assert _PROBE in names

    def test_a_differently_spelled_path_is_not_a_finding(self, two_copies):
        first, _ = two_copies
        module = importlib.import_module(_PROBE)
        _repoint(os.path.join(str(first), "sub", os.pardir))

        assert diagnostics.shadowed(module, verbose=False) == []

    def test_submodule_resolves_through_its_loaded_parent(self, two_copies):
        _, second = two_copies
        parent = importlib.import_module(_PROBE_PKG)
        submodule = importlib.import_module(f"{_PROBE_PKG}.sub")
        _repoint(second)

        # The parent still owns first/, so the submodule is still consistent.
        assert diagnostics.shadowed(submodule, verbose=False) == []

        parent.__path__ = [str(second / _PROBE_PKG)]
        findings = diagnostics.shadowed(submodule, verbose=False)

        assert [finding.name for finding in findings] == [f"{_PROBE_PKG}.sub"]

    def test_unresolvable_module_is_not_reported(self, two_copies):
        module = importlib.import_module(_PROBE)
        sys.path.pop(0)
        importlib.invalidate_caches()

        assert diagnostics.shadowed(module, verbose=False) == []

    def test_module_without_a_file_is_skipped(self):
        assert diagnostics.shadowed(sys, verbose=False) == []

    def test_accepts_an_imported_module_name(self, two_copies):
        _, second = two_copies
        importlib.import_module(_PROBE)
        _repoint(second)

        assert len(diagnostics.shadowed(_PROBE, verbose=False)) == 1

    def test_unimported_name_raises(self):
        with pytest.raises(ValueError):
            diagnostics.shadowed("nu_definitely_not_imported", verbose=False)

    def test_non_module_raises(self):
        with pytest.raises(TypeError):
            diagnostics.shadowed(42, verbose=False)

    def test_clean_scan_prints_nothing(self, capsys):
        diagnostics.shadowed(sys)
        assert capsys.readouterr().out == ""

    def test_verbose_prints_both_origins(self, two_copies, capsys):
        _, second = two_copies
        module = importlib.import_module(_PROBE)
        _repoint(second)

        diagnostics.shadowed(module)
        out = capsys.readouterr().out

        assert "diagnostics.shadowed()" in out
        assert "loaded from" in out and "resolves to" in out

    def test_finding_is_immutable(self, two_copies):
        _, second = two_copies
        module = importlib.import_module(_PROBE)
        _repoint(second)

        finding = diagnostics.shadowed(module, verbose=False)[0]
        with pytest.raises(AttributeError):
            finding.name = "other"
