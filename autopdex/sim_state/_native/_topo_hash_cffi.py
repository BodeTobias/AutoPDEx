"""cffi out-of-line API builder for the native topology hash extension.

Run ``python autopdex/sim_state/_native/_topo_hash_cffi.py`` to compile
``autopdex.sim_state._native._topo_hash`` manually; normally this is invoked
from ``setup.py`` during the build.
"""

import os
import sys

from cffi import FFI

ffi = FFI()
ffi.cdef("""
    int64_t factorize_entity_rows(const uint32_t*, int64_t, int, int64_t*, uint32_t*);
    int64_t lookup_entity_rows(const uint32_t*, int64_t, int,
                               const uint32_t*, int64_t, int64_t*);
    int64_t factorize_u64_rows(const uint64_t*, int64_t, int, int64_t*, int64_t*);
""")

_c = os.path.join(os.path.dirname(__file__), "_topo_hash.c")
with open(_c) as f:
    _source = f.read()

# MSVC (the default Windows toolchain) rejects the GCC/Clang flags; clang on
# macOS and gcc/clang on Linux accept them. Anything not understood would fail
# the build and fall back to the pure-NumPy path, so keep this compiler-aware.
if sys.platform == "win32":
    _extra_compile_args = ["/O2"]
else:
    _extra_compile_args = ["-O3", "-std=c11"]

ffi.set_source("autopdex.sim_state._native._topo_hash", _source,
               extra_compile_args=_extra_compile_args)

if __name__ == "__main__":
    ffi.compile(verbose=True)
