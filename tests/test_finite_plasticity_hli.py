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
Smoke test mirroring the compression example in
examples/offline/plasticity/finite_plasticity_exampes.ipynb.

Finite-strain multiplicative plasticity (J2, nonlinear isotropic Voce+linear
hardening) on a unit cube.  Internal variables are stored via a Cholesky
decomposition of the plastic right Cauchy-Green tensor C_p.

This test uses n=2 (8 brick elements) instead of n=7 in the notebook, and
runs only to t=0.1 (vs t=0.5) to keep compile + solve time manageable.
The assertion is that the simulation completes and displacement is finite —
the physics correctness is covered by the notebook.
"""

from functools import partial

import jax
import jax.numpy as jnp
import pytest

from autopdex import SimState, dae, spaces, utility
from autopdex.models import aug_lag_potential

jax.config.update("jax_enable_x64", True)

# ── Material parameters ───────────────────────────────────────────────────────
E = 206.9
nu = 0.29
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))
kappa = lam + 2.0 * mu / 3.0
sigma_y = 0.45
K_h = 0.13
R_inf = 1.165
delta_h = 16.93


# ── Helper functions (identical to notebook) ──────────────────────────────────

def _project_dev(mat):
    sqrt2 = jnp.sqrt(2.0)
    sqrt6 = jnp.sqrt(6.0)
    return jnp.stack([
        (mat[0, 0] - mat[1, 1]) / sqrt2,
        (mat[0, 0] + mat[1, 1] - 2.0 * mat[2, 2]) / sqrt6,
        sqrt2 * mat[0, 1],
        sqrt2 * mat[0, 2],
        sqrt2 * mat[1, 2],
    ])


def _C_p_from_q(q):
    a, b, c, d, e = q
    L = jnp.stack([
        jnp.stack([jnp.exp(a), 0.0, 0.0]),
        jnp.stack([c, jnp.exp(b), 0.0]),
        jnp.stack([d, e, jnp.exp(-a - b)]),
    ])
    return L @ L.T


def _strain_energy(F, C_p):
    b_e = F @ utility.matrix_inv(C_p) @ F.T
    I1 = jnp.trace(b_e)
    I3 = utility.matrix_det(F) ** 2
    return (
        kappa / 4.0 * (I3 - 1.0 - jnp.log(I3))
        + mu / 2.0 * (jnp.pow(I3, -1.0 / 3.0) * I1 - 3.0)
    )


def _hardening_energy(gamma):
    return (
        0.5 * K_h * gamma**2
        + R_inf * (gamma + 1.0 / delta_h * jnp.exp(-delta_h * gamma))
    )


def _local_dae(q_fun, t, settings):
    q, q_t = jax.jvp(q_fun, (t,), (1.0,))
    F = settings["F"]
    rho_p_dot = q_t["rho_p"]
    C_p, C_p_dot = jax.jvp(_C_p_from_q, (q["C_p"],), (q_t["C_p"],))

    P = jax.jacrev(_strain_energy)(F, C_p)
    tau = P @ F.T
    s = utility.matrix_dev(tau)
    sigma_vm = jnp.sqrt(1.5 * jnp.sum(s * s) + 1e-16)
    r_p = sigma_y + jax.jacfwd(_hardening_energy)(q["rho_p"])
    yield_func = sigma_vm / r_p - 1.0
    e_p = 0.5 * C_p_dot - 1.5 * (rho_p_dot / sigma_vm) * F.T @ s @ F
    e_yield = jax.jacfwd(
        lambda lmb: aug_lag_potential(lmb, -yield_func, eps=1.0)
    )(-rho_p_dot)
    return jnp.concatenate([_project_dev(e_p), jnp.atleast_1d(e_yield)])


def _model(ctx: SimState.ModelContext):
    H = jax.jacfwd(ctx.trial_ansatz["displacement"])(ctx.x_int, ctx.t)
    F = jnp.eye(3) + H

    if ctx.mode in ("internal variables", "weak form"):
        new_int_vars = ctx.solve_local("plasticity", inputs={"F": F})
        if ctx.mode == "internal variables":
            return new_int_vars
        C_p = _C_p_from_q(new_int_vars["C_p"])
    elif ctx.mode == "output":
        C_p = _C_p_from_q(ctx.internal_vars["C_p"])

    P = jax.jacrev(_strain_energy)(F, C_p)
    if ctx.mode == "weak form":
        Grad_v = jax.jacfwd(ctx.test_ansatz["displacement"])(ctx.x_int)
        return jnp.sum(P * Grad_v)
    return {
        "accumulated plastic strain": ctx.internal_vars["rho_p"],
        "von Mises stress": jnp.sqrt(1.5 * jnp.sum(utility.matrix_dev(P @ F.T) ** 2)),
    }


# ── Test ──────────────────────────────────────────────────────────────────────

def test_finite_plasticity_compression_runs():
    """Compression of a unit cube reaches t=0.1 with finite displacement field."""
    n = 2
    sim = SimState({
        "displacement": spaces.H1(order=2, dim=3, field_dimension=3),
        "C_p":   spaces.InternalVariable(field_dimension=5),
        "rho_p": spaces.InternalVariable(field_dimension=1),
    })
    sim.add_structured_mesh(
        (n, n, n),
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
        "brick", 1,
    )
    sim.add_temporal_discretization({
        "displacement": dae.NoTimeDerivative(),
        "C_p":          dae.BackwardEuler(),
        "rho_p":        dae.BackwardEuler(),
    })
    sim.add_local_subsystem(
        "plasticity", _local_dae, fields=("C_p", "rho_p"),
        root_solver=partial(dae.newton_solver, atol=1e-8, rtol=1e-6),
    )
    sim.add_model("__all__", "weak form", _model, internal_variables=("C_p", "rho_p"))
    sim.add_strong_bc("displacement", lambda x: jnp.isclose(x[0], 0.0),
                      lambda x, t: 0.0, index=0)
    sim.add_strong_bc("displacement", lambda x: jnp.isclose(x[1], 0.0),
                      lambda x, t: 0.0, index=1)
    sim.add_strong_bc("displacement", lambda x: jnp.isclose(x[2], 0.0),
                      lambda x, t: 0.0, index=2)
    sim.add_strong_bc("displacement", lambda x: jnp.isclose(x[2], 1.0),
                      lambda x, t: jnp.array([0.0, 0.0, -t]))
    sim.set_step_size_controller(
        dae.RootIterationController(target_niters=12, max_step_size=0.05)
    )
    sim.initialize(verbose=-1)
    sim.manager.root_solver = partial(dae.newton_solver, atol=1e-7, rtol=1e-5)
    sim.prepare()

    result = sim.run(dt0=0.05, time_span=0.1, num_time_steps=4)

    assert jnp.all(jnp.isfinite(result.q["displacement"])), \
        "displacement contains non-finite values"
    assert float(result.settings["current time"]) == pytest.approx(0.1, abs=1e-8)
    assert int(result.num_accepted) > 0
