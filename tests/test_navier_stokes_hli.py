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
Smoke test for the incompressible Navier-Stokes + energy HLI example.

Mirrors docs/notebooks/navier_stokes_hli.ipynb.  Only 3 time steps are run to
keep the test cheap; the goal is to verify that setup, assembly, and the linear
solve all work correctly on the external mixed-order mesh, not to reproduce the
full 10-second transient.

Mesh: docs/notebooks/meshes/navier_stokes_mesh.msh (Taylor-Hood, 2D channel + cylinder)
Fields: velocity (H1 order 2), pressure (H1 order 1), temperature (H1 order 2)
Integrators: BDF(2) / BackwardEuler / AdamsMoulton(1)
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import meshio
import pytest

from autopdex import SimState, dae, spaces

jax.config.update("jax_enable_x64", True)

_MESH_PATH = (
    Path(__file__).parent.parent / "docs" / "notebooks" / "meshes" / "navier_stokes_mesh.msh"
)


# ── Boundary helpers ──────────────────────────────────────────────────────────

def _on_left(x):
    return jnp.isclose(x[0], 0.0)


def _on_cylinder(x):
    return jnp.isclose((x[0] - 0.2) ** 2 + (x[1] - 0.2) ** 2, 0.05 ** 2)


def _on_top_bottom_or_cylinder(x):
    return jnp.logical_or(
        jnp.logical_or(jnp.isclose(x[1], 0.0), jnp.isclose(x[1], 0.41)),
        _on_cylinder(x),
    )


def _on_right(x):
    return jnp.isclose(x[0], 2.2)


def _inflow_velocity(x, t):
    y = x[1]
    u = 4.0 * 1.5 * y * (0.41 - y) / 0.41 ** 2
    ramp = jnp.where(t <= 3.5, jnp.sin(jnp.pi * t / 7.0), 1.0)
    return jnp.asarray([u * ramp, 0.0])


def _cylinder_temperature(x, t):
    return jnp.where(t <= 3.5, 20.0 * jnp.sin(jnp.pi * t / 7.0), 20.0)


# ── Weak form ─────────────────────────────────────────────────────────────────

def _model(ctx: SimState.ModelContext):
    x, t = ctx.x_int, ctx.t
    trial, test = ctx.trial_ansatz, ctx.test_ansatz

    p_fun = trial["pressure"]
    u_fun = trial["velocity"]
    temp_fun = trial["temperature"]

    if ctx.mode == "output":
        return {
                "velocity": u_fun(x, t),
                "pressure": p_fun(x, t),
                "temperature": temp_fun(x, t),
            }

    q_fun = test["pressure"]
    v_fun = test["velocity"]
    phi_fun = test["temperature"]

    p, q = p_fun(x, t), q_fun(x)
    u, (grad_u, du_dt) = u_fun(x, t), jax.jacfwd(u_fun, (0, 1))(x, t)
    v, grad_v = v_fun(x), jax.jacfwd(v_fun)(x)

    temp, (grad_temp, dtemp_dt) = temp_fun(x, t), jax.jacfwd(temp_fun, (0, 1))(x, t)
    phi, grad_phi = jax.value_and_grad(phi_fun)(x)

    viscosity = 1e-3 * (1.0 + 0.05 * temp)
    strain_u = 0.5 * (grad_u + grad_u.T)
    strain_v = 0.5 * (grad_v + grad_v.T)
    momentum = (
        du_dt @ v
        + (grad_u @ u) @ v
        + 2.0 * viscosity * jnp.vdot(strain_u, strain_v)
        - p * jnp.trace(grad_v)
    )
    continuity = -q * jnp.trace(grad_u)
    energy = dtemp_dt * phi + (u @ grad_temp) * phi + 1e-3 * (grad_temp @ grad_phi)

    return momentum + continuity + energy


# ── Test ──────────────────────────────────────────────────────────────────────

def test_navier_stokes_hli_runs():
    """Three time steps of NS + energy on the full channel mesh; fields must be finite."""
    mesh = meshio.read(str(_MESH_PATH))
    sim = SimState({
        "velocity":    spaces.H1(order=2, dim=2, field_dimension=2),
        "pressure":    spaces.H1(order=1, dim=2, field_dimension=1),
        "temperature": spaces.H1(order=2, dim=2, field_dimension=1),
    })
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({
        "velocity":    dae.BackwardDiffFormula(2),
        "pressure":    dae.BackwardEuler(),
        "temperature": dae.AdamsMoulton(1),
    })
    sim.add_model("__all__", "weak form", _model)
    sim.add_strong_bc("velocity",    _on_left,                  _inflow_velocity)
    sim.add_strong_bc("velocity",    _on_top_bottom_or_cylinder, lambda x, t: jnp.zeros((2,)))
    sim.add_strong_bc("pressure",    _on_right,                 lambda x, t: 0.0)
    sim.add_strong_bc("temperature", _on_left,                  lambda x, t: 0.0)
    sim.add_strong_bc("temperature", _on_cylinder,              _cylinder_temperature)
    sim.initialize(verbose=-1)
    sim.prepare()

    result = sim.run(dt0=0.02, time_span=0.06, num_time_steps=3)

    for field in ("velocity", "pressure", "temperature"):
        assert jnp.all(jnp.isfinite(result.q[field])), f"{field} contains non-finite values"
    assert float(result.settings["current time"]) == pytest.approx(0.06, abs=1e-10)
