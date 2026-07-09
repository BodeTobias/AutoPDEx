"""setup.py kept alongside pyproject.toml to build the optional native cffi
extension ``autopdex.sim_state._native._topo_hash``.

If cffi or a C compiler is unavailable the build degrades gracefully: a short
warning is printed and the extension is skipped, so the package still installs
and falls back to the pure-NumPy topology path at runtime.
"""

from __future__ import annotations

import sys

from setuptools import setup
from setuptools.command.build_ext import build_ext


try:
    from setuptools.errors import (
        CCompilerError,
        CompileError,
        LinkError,
        LibError,
        PlatformError,
    )
except ImportError:  # pragma: no cover - compatibility with older setuptools
    from distutils.errors import (  # type: ignore[no-redef]
        CCompilerError,
        CompileError,
        LinkError,
        LibError,
        DistutilsPlatformError,
    )

    PlatformError = DistutilsPlatformError


OPTIONAL_BUILD_ERRORS = (
    CCompilerError,
    CompileError,
    LinkError,
    LibError,
    PlatformError,
    OSError,
)


class optional_build_ext(build_ext):
    """Build optional native extensions if possible, otherwise skip them."""

    def run(self):
        try:
            super().run()
        except OPTIONAL_BUILD_ERRORS as exc:
            if self._all_extensions_optional():
                self.warn(
                    "autopdex: optional native topology extension "
                    f"could not be built: {exc}. "
                    "Falling back to the pure-NumPy path."
                )
                return
            raise

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except OPTIONAL_BUILD_ERRORS as exc:
            if getattr(ext, "optional", False):
                self.warn(
                    "autopdex: optional native topology extension "
                    f"{ext.name!r} could not be built: {exc}. "
                    "Falling back to the pure-NumPy path."
                )
                return
            raise

    def _all_extensions_optional(self) -> bool:
        return bool(self.extensions) and all(
            getattr(ext, "optional", False) for ext in self.extensions
        )


ext_modules = []

try:
    sys.path.insert(0, "autopdex/sim_state/_native")
    from _topo_hash_cffi import ffi

    topo_hash_ext = ffi.distutils_extension()
    topo_hash_ext.optional = True
    ext_modules = [topo_hash_ext]

except Exception as exc:  # pragma: no cover - depends on build environment
    print(
        "autopdex: could not prepare native topology extension "
        f"(_topo_hash): {exc}. Falling back to the pure-NumPy path.",
        file=sys.stderr,
    )


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": optional_build_ext},
)