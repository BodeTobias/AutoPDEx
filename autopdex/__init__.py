# autopdex/__init__.py
# Copyright (C) 2024 Tobias Bode
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.


import os


def enable_compilation_cache(path=None, min_compile_secs=0.0):
    """Aktiviert JAX' persistenten Compilation-Cache (geteilt über Läufe/Beispiele).

    Der Cache muss vor dem ersten JIT-Aufruf konfiguriert werden, daher wird dies
    beim Import von ``autopdex`` ausgeführt. Senkt Wiederholungsläufe desselben
    Problems (Notebook-Restart, CI, Tutorials mit fixem Netz) deutlich, da die
    XLA-Kompilation entfällt.

    Parameters
    ----------
    path:
        Cache-Verzeichnis. Default: ``$AUTOPDEX_JAX_CACHE_DIR`` bzw.
        ``$XDG_CACHE_HOME/autopdex/jax`` (Fallback ``~/.cache/autopdex/jax``).
    min_compile_secs:
        Nur Kernel cachen, deren Kompilation länger dauert. ``0.0`` cached alle
        (für AutoPDEx-Workloads nötig, da der Aufwand aus vielen billigen Kerneln
        besteht).

    Returns
    -------
    Der verwendete Cache-Pfad.
    """
    import jax

    if path is None:
        path = os.environ.get("AUTOPDEX_JAX_CACHE_DIR") or os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
            "autopdex",
            "jax",
        )
    jax.config.update("jax_compilation_cache_dir", path)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", float(min_compile_secs))
    return path


def _compilation_cache_disabled():
    return os.environ.get("AUTOPDEX_DISABLE_JAX_CACHE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


# Standardmäßig aktivieren, sofern nicht per Env-Var deaktiviert. Niemals den Import
# blockieren, falls der Cache nicht einrichtbar ist (read-only FS, fehlende Rechte).
if not _compilation_cache_disabled():
    try:
        enable_compilation_cache()
    except Exception:
        pass

from autopdex.sim_state import SimState


def run_tests():
    import pytest

    pytest.main(["-v", "-n", "auto", "tests/"])
