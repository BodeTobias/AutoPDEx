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
Tests mirroring examples/offline/subdomain_models_demo.py.

A field restricted to a subdomain: displacement lives everywhere, pressure only
on the right half. Two different models (compressible left, incompressible
Taylor-Hood right) on the two subdomains. Small mesh instead of the example's.

Covers:
  1. Forward solve runs and pressure has DOFs only on the right half.
  2. VTU output merges same-named fields across domains under one key, with the
     per-domain projections combined (displacement filled everywhere; pressure
     finite on the right, NaN on the left where the field does not live).
  3. Implicit output source discovery for a subdomain-restricted field emits no
     RuntimeWarning.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from autopdex import SimState, dae, spaces

jax.config.update("jax_enable_x64", True)

L, H = 2.0, 1.0
E, nu = 1000.0, 0.3
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))
VERTICES = [[0.0, 0.0], [L, 0.0], [L, H], [0.0, H]]
N_ELEMENTS = (4, 2)


def _scalar(fun, *args):
    return jnp.asarray(fun(*args)).reshape(-1)[0]


def _compressible(ctx):
    grad_u = jax.jacfwd(ctx.trial_ansatz["displacement"], 0)(ctx.x_int, ctx.t)
    strain = 0.5 * (grad_u + grad_u.T)
    if ctx.mode == "output":
        return {"displacement": ctx.trial_ansatz["displacement"](ctx.x_int, ctx.t)}
    grad_v = jax.jacfwd(ctx.test_ansatz["displacement"])(ctx.x_int)
    virtual_strain = 0.5 * (grad_v + grad_v.T)
    stress = lam * jnp.trace(strain) * jnp.eye(2) + 2.0 * mu * strain
    return jnp.sum(stress * virtual_strain)


def _incompressible(ctx):
    grad_u = jax.jacfwd(ctx.trial_ansatz["displacement"], 0)(ctx.x_int, ctx.t)
    strain = 0.5 * (grad_u + grad_u.T)
    p = _scalar(ctx.trial_ansatz["pressure"], ctx.x_int, ctx.t)
    if ctx.mode == "output":
        return {"displacement": ctx.trial_ansatz["displacement"](ctx.x_int, ctx.t), "pressure": p}
    grad_v = jax.jacfwd(ctx.test_ansatz["displacement"])(ctx.x_int)
    virtual_strain = 0.5 * (grad_v + grad_v.T)
    q = _scalar(ctx.test_ansatz["pressure"], ctx.x_int)
    dev = 2.0 * mu * (strain - jnp.trace(strain) / 2.0 * jnp.eye(2))
    momentum = jnp.sum((dev - p * jnp.eye(2)) * virtual_strain)
    return momentum + q * jnp.trace(strain)


def _make_sim(result_folder_name=None):
    sim = SimState({
        "displacement": spaces.H1(order=2, dim=2, field_dimension=2),  # Q2
        "pressure": spaces.H1(order=1, dim=2, field_dimension=1),      # continuous Q1
    })
    sim.add_structured_mesh(N_ELEMENTS, VERTICES, "quad", order=1)
    sim.add_domain("left", lambda x: x[0] <= L / 2 + 1e-9)
    sim.add_domain("right", lambda x: x[0] >= L / 2 - 1e-9)
    sim.set_active_domains({"displacement": "__all__", "pressure": "right"})
    sim.add_temporal_discretization({
        "displacement": dae.NoTimeDerivative(),
        "pressure": dae.NoTimeDerivative(),
    })
    sim.add_model("left", "weak form", _compressible)
    sim.add_model("right", "weak form", _incompressible)
    sim.add_strong_bc(
        "displacement",
        on_boundary_fun=lambda x: jnp.isclose(x[0], 0.0),
        value_fun=lambda x, t, s: jnp.zeros((2,)),
    )
    sim.add_weak_bc(
        "displacement",
        on_boundary_fun=lambda x: jnp.isclose(x[0], L),
        value_fun=lambda x, t, s: jnp.asarray([0.05 * t, 0.0]),
    )
    if result_folder_name is not None:
        sim.set_postprocessing_policy(
            dae.SaveEquidistantPolicy(num_points=2),
            result_folder_name=result_folder_name,
            domains=["left", "right"],
        )
    sim.initialize(verbose=-1)
    sim.prepare()
    return sim


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_subdomain_restricted_field_forward():
    """Forward solve is finite and pressure has DOFs only on the right half."""
    sim = _make_sim()
    result = sim.run(dt0=0.5, time_span=1.0, num_time_steps=2)

    u = result.q["displacement"]
    p = jnp.asarray(result.q["pressure"]).ravel()
    assert jnp.all(jnp.isfinite(u))
    assert jnp.all(jnp.isfinite(p))

    # Pressure is a continuous Q1 field on the right half only: it has strictly
    # fewer DOFs than the number of nodes it would occupy on the whole mesh.
    node_x = np.asarray(sim.settings["node coordinates"])[:, 0]
    n_nodes_total = int(node_x.shape[0])
    n_nodes_right = int(np.count_nonzero(node_x >= L / 2 - 1e-9))
    assert p.shape[0] == n_nodes_right
    assert p.shape[0] < n_nodes_total


def test_subdomain_vtu_fields_merged(tmp_path):
    """VTU merges same-named fields across domains under one key; the restricted
    field is written only where it lives (NaN elsewhere)."""
    meshio = pytest.importorskip("meshio")
    folder = str(tmp_path / "subdomain.res")
    sim = _make_sim(result_folder_name=folder)
    sim.run(dt0=0.5, time_span=1.0, num_time_steps=2)

    frames = sorted((tmp_path / "subdomain.res").glob("step_*.vtu"))
    assert frames, "no VTU frames were written"
    m = meshio.read(str(frames[-1]))

    # One key per field -- no "displacement__2" duplicate from the second domain.
    assert "displacement" in m.point_data
    assert "pressure" in m.point_data
    assert not any(name.startswith("displacement__") for name in m.point_data)

    u = np.asarray(m.point_data["displacement"])
    p = np.asarray(m.point_data["pressure"]).ravel()
    x = np.asarray(m.points)[:, 0]

    # displacement lives everywhere -> the two per-domain projections merge into a
    # gap-free field.
    assert not np.isnan(u).any()

    # pressure lives only on the right -> finite on the right-interior nodes,
    # undefined (NaN) on the left-interior nodes.
    left = x < L / 2 - 1e-9
    right = x > L / 2 + 1e-9
    assert np.isfinite(p[right]).all()
    assert np.isnan(p[left]).all()


def test_subdomain_output_discovery_no_warning(tmp_path):
    """Implicit output source discovery for a restricted field emits no warning."""
    folder = str(tmp_path / "subdomain_nowarn.res")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        sim = _make_sim(result_folder_name=folder)
        sim.run(dt0=0.5, time_span=1.0, num_time_steps=2)
