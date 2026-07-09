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
Tests mirroring examples/offline/checkpointing_reverse_transient_hli.py.

Transient heat conduction on a thin 2D beam driven by a spatially distributed
Gaussian source field.  The objective is the sensor-weighted final temperature.
Sensitivities of the objective w.r.t. the source coefficients are computed via
reverse-mode AD (jacrev on scipy backend) and verified against central FD.

Mesh: 6×2 quad elements (vs 60×6 in the example; governs DOF count only,
      not the character of the computation).
Time: 4 uniform Backward-Euler steps (vs 80 in the example).
Source: 2×2 Gaussian basis functions (source_side=2, 4 control parameters).
"""

import jax
import jax.numpy as jnp
import pytest

from autopdex import SimState, dae, spaces

jax.config.update("jax_enable_x64", True)

L, H = 1.0, 0.2
nx, ny = 6, 2
n_steps = 4
t_final = 0.2
kappa = 0.03
dt = t_final / n_steps

source_side = 2
xs = jnp.linspace(0.15, 0.85, source_side)
ys = jnp.linspace(0.25 * H, 0.75 * H, source_side)
xx, yy = jnp.meshgrid(xs, ys, indexing="xy")
source_centers = jnp.stack((xx.ravel(), yy.ravel()), axis=1)
source_sigma = 0.12
source_coeffs_0 = 0.2 * jnp.sin(
    jnp.linspace(0.0, 2.0 * jnp.pi, source_centers.shape[0], endpoint=False)
)


def _source_basis(x):
    r2 = jnp.sum((source_centers - x) ** 2, axis=1)
    return jnp.exp(-0.5 * r2 / source_sigma**2)


def _model_heat(ctx: SimState.ModelContext):
    if ctx.mode != "weak form":
        return {}
    T_dot  = jax.jacfwd(ctx.trial_ansatz["T"], 1)(ctx.x_int, ctx.t)
    grad_T = jax.jacfwd(ctx.trial_ansatz["T"], 0)(ctx.x_int, ctx.t)
    v      = ctx.test_ansatz["T"](ctx.x_int)
    grad_v = jax.jacfwd(ctx.test_ansatz["T"])(ctx.x_int)
    source = jnp.dot(ctx.settings["source_coeffs"], _source_basis(ctx.x_int))
    return T_dot * v + ctx.settings["kappa"] * jnp.dot(grad_T, grad_v) - source * v


def _make_ckpt_sim():
    sim = SimState({"T": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.add_structured_mesh((nx, ny), [[0.0, 0.0], [L, 0.0], [L, H], [0.0, H]], "quad", order=1)
    sim.add_temporal_discretization({"T": dae.BackwardEuler()})
    sim.add_model("__all__", "weak form", _model_heat)
    sim.add_strong_bc(
        "T",
        lambda x: (
            jnp.isclose(x[0], 0.0)
            | jnp.isclose(x[0], L)
            | jnp.isclose(x[1], 0.0)
            | jnp.isclose(x[1], H)
        ),
        lambda x, t: 0.0,
    )
    sim.settings["kappa"] = kappa
    sim.settings["source_coeffs"] = source_coeffs_0
    sim.static_settings["solver backend"] = "scipy"
    sim.static_settings["solver"] = "lapack"
    sim.static_settings["implicit diff mode"] = "reverse"
    sim.initialize(verbose=-1)
    sim.prepare()
    return sim


def test_checkpointing_reverse_runs():
    """Forward run completes and temperature is finite."""
    sim = _make_ckpt_sim()
    settings_0 = {**sim.settings, "current time": 0.0}
    sim.settings = dict(settings_0)
    result = sim.run(dt0=dt, time_span=t_final, num_time_steps=n_steps)
    assert jnp.all(jnp.isfinite(result.q["T"]))
    assert float(result.settings["current time"]) == pytest.approx(t_final, abs=1e-10)


def test_checkpointing_reverse_ad_vs_fd():
    """Reverse-mode gradient of sensor objective agrees with central FD."""
    sim = _make_ckpt_sim()
    settings_0 = {**sim.settings, "current time": 0.0}
    coords = settings_0["node coordinates"]
    sensor_center = jnp.array([0.75 * L, 0.5 * H])
    sensor_weight = jnp.exp(-0.5 * jnp.sum((coords - sensor_center) ** 2, axis=1) / 0.08**2)
    sensor_weight = sensor_weight / jnp.sum(sensor_weight)

    def objective(source_coeffs):
        sim.settings = dict(settings_0)
        sim.settings["source_coeffs"] = source_coeffs
        T_final = sim.run(dt0=dt, time_span=t_final, num_time_steps=n_steps).q["T"]
        return jnp.dot(sensor_weight, T_final)

    _, dJ_da = jax.jit(jax.value_and_grad(objective))(source_coeffs_0)

    idx = 0
    eps = 1e-5
    e_i = jnp.zeros_like(source_coeffs_0).at[idx].set(1.0)
    dJ_fd = (objective(source_coeffs_0 + eps * e_i) - objective(source_coeffs_0 - eps * e_i)) / (2 * eps)

    assert jnp.abs(dJ_da[idx] - dJ_fd) < 1e-5 * (jnp.abs(dJ_da[idx]) + 1e-12), (
        f"reverse AD dJ/da[{idx}] = {float(dJ_da[idx]):.6e}, FD = {float(dJ_fd):.6e}"
    )
