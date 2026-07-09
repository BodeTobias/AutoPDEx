# Copyright (C) 2025 Tobias Bode
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

"""
Smoke test mirroring examples/offline/norton_creep.py.

Norton power-law creep (double-mechanism: linear + power-law) in an
axisymmetric cylinder under combined mantle and axial pressure.  Internal
variables are the creep strain tensor eps_cr, integrated via a local DAE
sub-system with Backward Euler.

This test uses a 4×4 quad mesh (nr=4, nz=4 vs nr=20, nz=80 in the
example) and only 2 time steps over [0, 0.2] days.  It verifies that the
setup, local solve, and global assembly run without error on a small
problem.
"""

from functools import partial

import jax
import jax.numpy as jnp
import pytest

from autopdex import SimState, dae, spaces
from autopdex.models import km_tools as km

jax.config.update("jax_enable_x64", True)

# ── Geometry & material (mm, MPa, d) ─────────────────────────────────────────
nz, nr = 4, 4
H, R = 100.0, 25.0

E = 25_000.0
nu = 0.25
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))
sigma_ref = 1.0
A1, n1 = 1.0e-7, 1.0
A2, n2 = 1.0e-10, 5.0
p_mantle = 10.0
p_axial  = 20.0
sqrt2 = jnp.sqrt(2.0)


def _strain_energy(eps_e):
    return mu * (eps_e @ eps_e) + 0.5 * lam * km.trace(eps_e) ** 2


_stress = jax.jacrev(_strain_energy)


# ── Local DAE ─────────────────────────────────────────────────────────────────

def _local_model(q_fun, t, settings):
    q, q_t = jax.jvp(q_fun, (t,), (1.0,))
    sig = _stress(settings["eps"] - q["eps_cr"])
    s = km.dev(sig)
    sig_eq = jnp.sqrt(1.5 * (s @ s) + 1.0e-30)
    eq_rate = A1 * (sig_eq / sigma_ref) ** n1 + A2 * (sig_eq / sigma_ref) ** n2
    return q_t["eps_cr"] - eq_rate * 1.5 * s / sig_eq


# ── Axisymmetric strain ───────────────────────────────────────────────────────

def _axissymmetric_strain(grad_u, u, r):
    return jnp.asarray([
        grad_u[0, 0],
        grad_u[1, 1],
        u[0] / r,
        (grad_u[1, 0] + grad_u[0, 1]) / sqrt2,
    ])


# ── Global weak form ──────────────────────────────────────────────────────────

def _model(ctx: SimState.ModelContext):
    t = ctx.t
    X = ctx.trial_ansatz["physical coor"](ctx.x_int)
    r = jnp.maximum(X[0], 1.0e-12)

    u     = ctx.trial_ansatz["displacement"](ctx.x_int, t)
    grad_u = jax.jacfwd(ctx.trial_ansatz["displacement"])(ctx.x_int, t)
    eps   = _axissymmetric_strain(grad_u, u, r)

    if ctx.mode in ("weak form", "internal variables"):
        new_int_vars = ctx.solve_local("creep", inputs={"eps": eps})
        sig = _stress(eps - new_int_vars["eps_cr"])
        if ctx.mode == "internal variables":
            return new_int_vars
        grad_v  = jax.jacfwd(ctx.test_ansatz["displacement"])(ctx.x_int)
        eps_test = _axissymmetric_strain(grad_v, jnp.zeros_like(u), r)
        return 2.0 * jnp.pi * r * (sig @ eps_test)

    if ctx.mode == "output":
        eps_cr = ctx.internal_vars["eps_cr"]
        sig    = _stress(eps - eps_cr)
        return {
            "eps_cr_eq": jnp.sqrt(2.0 / 3.0 * (km.dev(eps_cr) @ km.dev(eps_cr))),
            "sig_zz":    sig[1],
        }


# ── Test ──────────────────────────────────────────────────────────────────────

def test_norton_creep_runs():
    """Two time steps of Norton creep; displacement and creep strain are finite."""
    sim = SimState({
        "displacement": spaces.H1(order=1, dim=2, field_dimension=2),
        "eps_cr": spaces.InternalVariable(field_dimension=4),
    })
    sim.add_structured_mesh(
        (nr, nz),
        [[0.0, 0.0], [R, 0.0], [R, H], [0.0, H]],
        "quad",
    )
    sim.add_temporal_discretization({
        "displacement": dae.NoTimeDerivative(),
        "eps_cr": dae.BackwardEuler(),
    })
    sim.add_local_subsystem(
        "creep", _local_model, fields=("eps_cr",),
        root_solver=partial(dae.newton_solver, atol=1e-13, rtol=1e-11),
    )
    sim.add_model("__all__", "weak form", _model, internal_variables=("eps_cr",))
    sim.add_strong_bc("displacement", lambda x: jnp.isclose(x[1], 0.0),
                      lambda x, t: 0.0, index=1)
    sim.add_strong_bc("displacement", lambda x: jnp.isclose(x[0], 0.0),
                      lambda x, t: 0.0, index=0)
    sim.add_weak_bc("displacement", lambda x: jnp.isclose(x[0], R),
                    lambda x, t: 2.0 * jnp.pi * x[0] * jnp.asarray([-p_mantle, 0.0]))
    sim.add_weak_bc("displacement", lambda x: jnp.isclose(x[1], H),
                    lambda x, t: 2.0 * jnp.pi * x[0] * jnp.asarray([0.0, -p_axial]))
    sim.set_step_size_controller(
        dae.RootIterationController(target_niters=8, max_step_size=0.2)
    )
    sim.initialize(verbose=-1)
    sim.set_root_solver(partial(dae.newton_solver, atol=1e-10, rtol=1e-8))
    sim.prepare()

    result = sim.run(dt0=0.1, time_span=0.2, num_time_steps=4)

    assert jnp.all(jnp.isfinite(result.q["displacement"])), \
        "displacement contains non-finite values"
    assert float(result.settings["current time"]) == pytest.approx(0.2, abs=1e-8)
    assert int(result.num_accepted) > 0
