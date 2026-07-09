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
Tests mirroring examples/offline/sim_state_demo.ipynb.

Linear elasticity (plane strain) on a small quad mesh with a settings-
parameterised BC (scaler²).  Covers:

  1. Forward solve — displacement is finite and the run reaches t=1.
  2. First-order AD (jacfwd) vs. central FD — agreement within FD truncation error.
  3. Second-order AD — result is finite and matches finite differences.
  4. Chained .run() calls and quasistatic mode.

Mesh: 4×4 quad elements on [0,2]×[0,1] (instead of 1000×1000 in the notebook).
"""

import jax
import jax.numpy as jnp
import pytest

from autopdex import SimState, dae, spaces

jax.config.update("jax_enable_x64", True)

L, H = 2.0, 1.0
E, nu = 1000.0, 0.3
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu  = E / (2.0 * (1.0 + nu))

VERTICES = [[0.0, 0.0], [L, 0.0], [L, H], [0.0, H]]


def _model(ctx: SimState.ModelContext):
    grad_u = jax.jacfwd(ctx.trial_ansatz["displacement"], 0)(ctx.x_int, ctx.t)
    strain = 0.5 * (grad_u + grad_u.T)
    if ctx.mode == "output":
        return {"strain": strain}
    grad_v = jax.jacfwd(ctx.test_ansatz["displacement"])(ctx.x_int)
    virtual_strain = 0.5 * (grad_v + grad_v.T)
    stress = lam * jnp.trace(strain) * jnp.eye(2) + 2.0 * mu * strain
    return jnp.sum(stress * virtual_strain)


def _make_sim():
    sim = SimState({"displacement": spaces.H1(order=1, dim=2, field_dimension=2)})
    sim.add_structured_mesh((4, 4), VERTICES, "quad", order=1)
    sim.add_temporal_discretization({"displacement": dae.NoTimeDerivative()})
    sim.add_model("__all__", "weak form", _model)
    sim.add_strong_bc(
        "displacement",
        on_boundary_fun=lambda x: jnp.isclose(x[0], 0.0),
        value_fun=lambda x, t, s: s["scaler"] ** 2 * jnp.ones((2,)),
    )
    sim.add_weak_bc(
        "displacement",
        on_boundary_fun=lambda x: jnp.isclose(x[0], L),
        value_fun=lambda x, t, s: s["scaler"] ** 2 * jnp.asarray([0.2 * t, 0.0]),
    )
    sim.settings["scaler"] = 0.0
    sim.initialize(verbose=-1)
    sim.prepare()
    return sim


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_sim_state_demo_forward():
    """Forward solve reaches t=1 and displacement is finite."""
    sim = _make_sim()
    sim.settings["scaler"] = 1.0
    result = sim.run(dt0=0.5, time_span=1.0, num_time_steps=2)
    assert jnp.all(jnp.isfinite(result.q["displacement"]))
    assert float(result.settings["current time"]) == pytest.approx(1.0, abs=1e-10)


def test_sim_state_demo_first_order_sensitivity():
    """jacfwd vs central FD: for u(s)=s²·u₀, both give 2s·u₀ — agreement within FD error."""
    sim = _make_sim()
    settings_0 = dict(sim.settings)
    bc_scaler = 1.0

    def run_with(s):
        sim.settings = dict(settings_0)
        sim.settings["scaler"] = s
        return sim.run(dt0=0.5, time_span=1.0, num_time_steps=2).q["displacement"]

    dq_ad = jax.jit(jax.jacfwd(run_with))(bc_scaler)
    h = 1e-6
    dq_fd = (run_with(bc_scaler + h) - run_with(bc_scaler - h)) / (2 * h)

    # For u(s) = s²·u₀, the third derivative is 0 ⟹ FD error is near machine eps.
    assert jnp.allclose(dq_ad, dq_fd, atol=1e-6, rtol=0.0), (
        f"max diff AD vs FD = {float(jnp.max(jnp.abs(dq_ad - dq_fd))):.2e}"
    )


def test_sim_state_demo_second_order_sensitivity():
    """d²u/ds² is finite and equals 2·u(s=1) by linearity of the system."""
    sim = _make_sim()
    settings_0 = dict(sim.settings)
    bc_scaler = 1.0

    def run_with(s):
        sim.settings = dict(settings_0)
        sim.settings["scaler"] = s
        return sim.run(dt0=0.5, time_span=1.0, num_time_steps=2).q["displacement"]

    sim.settings["scaler"] = bc_scaler
    u_ref = run_with(bc_scaler)

    d2u = jax.jit(jax.jacfwd(jax.jacfwd(run_with)))(bc_scaler)
    assert jnp.all(jnp.isfinite(d2u)), "second-order sensitivity contains non-finite values"
    # u(s) = s²·u₀  ⟹  d²u/ds² = 2·u₀ = 2·u(s=1)
    assert jnp.allclose(d2u, 2.0 * u_ref, atol=1e-8, rtol=1e-8), (
        f"max diff d²u/ds² vs 2u = {float(jnp.max(jnp.abs(d2u - 2.0 * u_ref))):.2e}"
    )


def test_sim_state_demo_chained_runs():
    """Chained .run() calls advance time correctly; quasistatic run produces finite results."""
    sim = _make_sim()
    sim.settings["scaler"] = 1.0

    r1 = sim.run(dt0=0.5, time_span=1.0, num_time_steps=2)
    assert float(r1.settings["current time"]) == pytest.approx(1.0, abs=1e-10)

    r2 = r1.run(dt0=0.5, time_span=1.0, num_time_steps=2)
    assert float(r2.settings["current time"]) == pytest.approx(2.0, abs=1e-10)

    # quasistatic=True uses a different time accounting — just check it runs and is finite
    r3 = r2.run(dt0=0.5, time_span=1.0, num_time_steps=2, quasistatic=True)
    assert jnp.all(jnp.isfinite(r3.q["displacement"]))
