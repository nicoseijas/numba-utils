"""Is a loaded module the one the import path would resolve today?

A module imported early outlives the path that found it. A frozen
snapshot copied into the tree, a stale build directory, a sibling
checkout ahead of the real one on ``sys.path`` — once the module is in
``sys.modules`` nothing re-checks it, and the symptom is not an import
error but wrong numbers from code that reads correctly.

This asks one question per module: would importing this name right now
resolve to a different file than the one currently loaded? It is
deliberately not "does a second copy exist somewhere on disk" — that
needs a directory walk and fires on every vendored tree.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from importlib.machinery import PathFinder
from types import ModuleType

_VENDOR_MARKERS = ("site-packages", "dist-packages")


@dataclass(frozen=True)
class ShadowedModule:
    """One module whose loaded file is not the file it now resolves to."""

    name: str
    loaded_origin: str
    resolved_origin: str

    def __str__(self) -> str:
        return (
            f"{self.name}\n"
            f"    loaded from: {self.loaded_origin}\n"
            f"    resolves to: {self.resolved_origin}"
        )


def shadowed(
    module: ModuleType | str | None = None, *, verbose: bool = True
) -> list[ShadowedModule]:
    """Report modules loaded from a file the current path no longer resolves.

    Call it with no argument to scan every loaded first-party module, or
    with a module (or its imported name) to check just that one. An empty
    list means nothing was detected. Read-only: nothing is imported,
    reloaded, or removed, and ``sys.path`` is never touched.

    A scan skips the stdlib, anything under ``site-packages`` /
    ``dist-packages``, ``__main__``, and modules without a ``__file__``
    (builtins, frozen and namespace packages) — a vendored dependency
    resolving oddly is not the failure this looks for. An explicit
    argument bypasses those filters and answers about whatever you pass.

    A module that no longer resolves AT ALL is not reported: a popped
    path entry, or an editable install whose finder sits on
    ``sys.meta_path`` where ``PathFinder`` cannot see it, would otherwise
    make the check cry wolf on healthy trees.

    A module imported from a sourceless ``.pyc`` is compared the same
    way as any other: quiet while that ``.pyc`` is what the path finds,
    reported once a ``.py`` appears beside it — which is a real
    divergence, since the source is what an import would pick up now.
    """
    if module is None:
        candidates = [
            loaded
            for name, loaded in list(sys.modules.items())
            # An alias entry (sys.modules["os.path"] is ntpath) is the same
            # module under a second key, not a second module to check.
            if isinstance(loaded, ModuleType)
            and getattr(loaded, "__name__", None) == name
            and _is_scannable(loaded)
        ]
    else:
        candidates = [_as_module(module)]

    findings = sorted(
        (
            finding
            for finding in (_check(candidate) for candidate in candidates)
            if finding is not None
        ),
        key=lambda finding: finding.name,
    )
    if verbose and findings:
        count = len(findings)
        noun = "module" if count == 1 else "modules"
        print(
            f"diagnostics.shadowed(): {count} {noun} loaded from a file the "
            f"import path no longer resolves:"
        )
        for index, finding in enumerate(findings, 1):
            print(f"  {index}. {finding}")
        print(
            "  Whatever is loaded is what ran. Check sys.path ordering and "
            "remove stale copies, then re-run in a fresh process."
        )
    return findings


def _as_module(module: ModuleType | str) -> ModuleType:
    if isinstance(module, ModuleType):
        return module
    if isinstance(module, str):
        loaded = sys.modules.get(module)
        if loaded is None:
            raise ValueError(
                f"diagnostics: module {module!r} is not imported — there is "
                f"nothing loaded to compare against"
            )
        return loaded
    raise TypeError(
        "diagnostics: expected a module or the name of an imported module, "
        f"got {type(module).__name__!r}"
    )


def _check(module: ModuleType) -> ShadowedModule | None:
    name = getattr(module, "__name__", None)
    loaded_origin = getattr(module, "__file__", None)
    if not name or not loaded_origin:
        return None
    resolved_origin = _origin_now(name)
    if resolved_origin is None or _same_file(loaded_origin, resolved_origin):
        return None
    return ShadowedModule(name, loaded_origin, resolved_origin)


def _origin_now(name: str) -> str | None:
    """Where the path would find ``name``, ignoring what is already loaded.

    ``importlib.util.find_spec`` is the wrong tool here: for a name in
    ``sys.modules`` it returns that module's own ``__spec__``, so the
    comparison is trivially equal and the check becomes a silent no-op.
    ``PathFinder`` re-resolves against the path entries without consulting
    ``sys.modules`` and without executing the module.
    """
    parent, _, _ = name.rpartition(".")
    if parent:
        search_path = getattr(sys.modules.get(parent), "__path__", None)
        if search_path is None:
            return None
        search = list(search_path)
    else:
        search = sys.path
    try:
        spec = PathFinder.find_spec(name, search)
    except (ImportError, OSError, ValueError):
        return None  # a path entry hook that cannot answer is not a finding
    return getattr(spec, "origin", None) if spec is not None else None


def _is_scannable(module: ModuleType) -> bool:
    name = getattr(module, "__name__", "") or ""
    if not name or name == "__main__":
        return False
    if name.split(".", 1)[0] in getattr(sys, "stdlib_module_names", ()):
        return False
    file = getattr(module, "__file__", None)
    if not file:
        return False
    # Match path COMPONENTS, not a substring: a checkout living under
    # ".../my-site-packages-fork/" is first-party and must still scan.
    parts = _canonical(file).replace("\\", "/").split("/")
    return not any(marker in parts for marker in _VENDOR_MARKERS)


def _same_file(loaded: str, resolved: str) -> bool:
    return _canonical(loaded) == _canonical(resolved)


def _canonical(path: str) -> str:
    """Compare paths, not spellings: symlinks, ``..`` and (on Windows) case."""
    try:
        return os.path.normcase(os.path.realpath(path))
    except (OSError, ValueError):
        return os.path.normcase(path)
