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

"""
Sensitivity tests: essential BCs, natural BCs, material parameters, initial conditions.

All comparisons exploit the exact linearity of the discrete problem — no FD, no
discretization error:
  - Essential/Natural:  T(p) = p · T(p=1)   =>  d(Σ Tᵢ)/dp = Σ Tᵢ / p
  - Material (k):       T(k) = T_ref / k · k_ref  =>  d(Σ Tᵢ)/dk = −Σ Tᵢ / k
  - IC (T0, transient): T_f(T0) = T0 · T_f(1)   =>  d(Tᵢ_f)/dT0 = Tᵢ_f / T0

Test problems (1D-like heat on thin 2D strip [0,1]×[0,0.1], quad mesh):
  1. Essential BC:      −∇²T = 0,  T(0) = g, T(1) = 0
  2. Natural BC:        −∇²T = 0,  T(0) = 0, ∂T/∂n|_{x=1} = q
  3. Material param:    −k ∇²T = 1, T(0) = T(1) = 0
  4. IC (transient):    ∂T/∂t = k ∇²T, T(x,0) = T0 sin(πx), T(0)=T(1)=0
"""

import pytest
import jax
import jax.numpy as jnp

from autopdex import SimState, dae, spaces

jax.config.update("jax_enable_x64", True)

L, H = 1.0, 0.1
N = 10  # elements in x


# ─── Shared model functions ────────────────────────────────────────────────────

def _poisson_model(ctx):
    if ctx.mode != "weak form":
        return {}
    k = ctx.settings.get("k", 1.0)
    grad_T = jax.jacfwd(ctx.trial_ansatz["T"], 0)(ctx.x_int, ctx.t)
    grad_v = jax.jacfwd(ctx.test_ansatz["T"])(ctx.x_int)
    return k * jnp.dot(grad_T, grad_v)


def _poisson_source_model(ctx):
    """Laplacian with constant unit source: −k ∇²T = 1."""
    if ctx.mode != "weak form":
        return {}
    k = ctx.settings.get("k", 1.0)
    grad_T = jax.jacfwd(ctx.trial_ansatz["T"], 0)(ctx.x_int, ctx.t)
    grad_v = jax.jacfwd(ctx.test_ansatz["T"])(ctx.x_int)
    v = ctx.test_ansatz["T"](ctx.x_int)
    return k * jnp.dot(grad_T, grad_v) - v


def _heat_model(ctx):
    """Transient heat equation: ∂T/∂t + k ∇T · ∇v = 0."""
    if ctx.mode != "weak form":
        return {}
    k = ctx.settings.get("k", 1.0)
    T_dot  = jax.jacfwd(ctx.trial_ansatz["T"], 1)(ctx.x_int, ctx.t)
    grad_T = jax.jacfwd(ctx.trial_ansatz["T"], 0)(ctx.x_int, ctx.t)
    v      = ctx.test_ansatz["T"](ctx.x_int)
    grad_v = jax.jacfwd(ctx.test_ansatz["T"])(ctx.x_int)
    return T_dot * v + k * jnp.dot(grad_T, grad_v)


# ─── SimState factory ─────────────────────────────────────────────────────────

def _strip_sim(model, temporal, backend, diff_mode="forward"):
    sim = SimState({"T": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.add_structured_mesh(
        (N, 1), [[0, 0], [L, 0], [L, H], [0, H]], "quad", order=1
    )
    sim.add_temporal_discretization({"T": temporal})
    sim.add_model("__all__", "weak form", model)
    sim.static_settings["solver backend"] = backend
    sim.static_settings["implicit diff mode"] = diff_mode
    return sim


# ─── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("backend,ad_fun,diff_mode", [
    ("dense", jax.jacfwd, "forward"),
    ("dense", jax.jacrev, "forward"),
    ("scipy", jax.jacfwd, "forward"),
    ("scipy", jax.jacrev, "reverse"),  # scipy needs implicit diff mode='reverse' for jacrev
], ids=["dense-jacfwd", "dense-jacrev", "scipy-jacfwd", "scipy-jacrev"])
def test_sensitivity_essential_bc(backend, ad_fun, diff_mode):
    """d(Σ T)/dg — Dirichlet BC parameter; exact by linearity T(g) = g·T(g=1)."""
    g_ref = 2.3

    sim = _strip_sim(_poisson_model, dae.NoTimeDerivative(), backend, diff_mode)
    sim.add_strong_bc("T", lambda x: jnp.isclose(x[0], 0.0),
                         lambda x, t, s: s["g"])
    sim.add_strong_bc("T", lambda x: jnp.isclose(x[0], L),
                         lambda x, t: 0.0)
    sim.settings["g"] = g_ref
    sim.initialize(verbose=-1)
    sim.prepare()
    settings_0 = dict(sim.settings)

    result_ref = sim.run(dt0=1.0, time_span=1.0, num_time_steps=1)
    grad_exact = result_ref.q["T"] / g_ref  # exact nodal gradient by linearity

    def compute(g):
        sim.settings = dict(settings_0)
        sim.settings["g"] = g
        return sim.run(dt0=1.0, time_span=1.0, num_time_steps=1).q["T"]

    grad_ad = ad_fun(compute)(g_ref)
    assert jnp.allclose(grad_ad, grad_exact, atol=1e-11, rtol=1e-11), (
        f"{ad_fun.__name__}/{backend}: max err = "
        f"{float(jnp.max(jnp.abs(grad_ad - grad_exact))):.3e}"
    )


@pytest.mark.parametrize("backend,ad_fun,diff_mode", [
    ("dense", jax.jacfwd, "forward"),
    ("dense", jax.jacrev, "forward"),
    ("scipy", jax.jacfwd, "forward"),
    ("scipy", jax.jacrev, "reverse"),  # scipy needs implicit diff mode='reverse' for jacrev
], ids=["dense-jacfwd", "dense-jacrev", "scipy-jacfwd", "scipy-jacrev"])
def test_sensitivity_natural_bc(backend, ad_fun, diff_mode):
    """d(Σ T)/dq — Neumann BC parameter; exact by linearity T(q) = q·T(q=1)."""
    q_ref = 1.7

    sim = _strip_sim(_poisson_model, dae.NoTimeDerivative(), backend, diff_mode)
    sim.add_strong_bc("T", lambda x: jnp.isclose(x[0], 0.0),
                         lambda x, t: 0.0)
    sim.add_weak_bc("T", lambda x: jnp.isclose(x[0], L),
                       lambda x, t, s: s["q"])
    sim.settings["q"] = q_ref
    sim.initialize(verbose=-1)
    sim.prepare()
    settings_0 = dict(sim.settings)

    result_ref = sim.run(dt0=1.0, time_span=1.0, num_time_steps=1)
    grad_exact = result_ref.q["T"] / q_ref

    def compute(q):
        sim.settings = dict(settings_0)
        sim.settings["q"] = q
        return sim.run(dt0=1.0, time_span=1.0, num_time_steps=1).q["T"]

    grad_ad = ad_fun(compute)(q_ref)
    assert jnp.allclose(grad_ad, grad_exact, atol=1e-11, rtol=1e-11), (
        f"{ad_fun.__name__}/{backend}: max err = "
        f"{float(jnp.max(jnp.abs(grad_ad - grad_exact))):.3e}"
    )


@pytest.mark.parametrize("backend,ad_fun,diff_mode", [
    ("dense", jax.jacfwd, "forward"),
    ("dense", jax.jacrev, "forward"),
    ("scipy", jax.jacfwd, "forward"),
    ("scipy", jax.jacrev, "reverse"),  # scipy needs implicit diff mode='reverse' for jacrev
], ids=["dense-jacfwd", "dense-jacrev", "scipy-jacfwd", "scipy-jacrev"])
def test_sensitivity_material_parameter(backend, ad_fun, diff_mode):
    """d(Σ T)/dk for −k∇²T = 1; T(k) = T_ref·k_ref/k => gradient = −Σ T / k."""
    k_ref = 0.8

    sim = _strip_sim(_poisson_source_model, dae.NoTimeDerivative(), backend, diff_mode)
    sim.add_strong_bc("T", lambda x: jnp.isclose(x[0], 0.0),
                         lambda x, t: 0.0)
    sim.add_strong_bc("T", lambda x: jnp.isclose(x[0], L),
                         lambda x, t: 0.0)
    sim.settings["k"] = k_ref
    sim.initialize(verbose=-1)
    sim.prepare()
    settings_0 = dict(sim.settings)

    result_ref = sim.run(dt0=1.0, time_span=1.0, num_time_steps=1)
    grad_exact = -result_ref.q["T"] / k_ref  # T(k) = C/k => dT/dk = -T/k

    def compute(k):
        sim.settings = dict(settings_0)
        sim.settings["k"] = k
        return sim.run(dt0=1.0, time_span=1.0, num_time_steps=1).q["T"]

    grad_ad = ad_fun(compute)(k_ref)
    assert jnp.allclose(grad_ad, grad_exact, atol=1e-11, rtol=1e-11), (
        f"{ad_fun.__name__}/{backend}: max err = "
        f"{float(jnp.max(jnp.abs(grad_ad - grad_exact))):.3e}"
    )


@pytest.mark.parametrize("backend,ad_fun,diff_mode", [
    ("dense", jax.jacfwd, "forward"),
    ("dense", jax.jacrev, "forward"),
    ("scipy", jax.jacfwd, "forward"),
    ("scipy", jax.jacrev, "reverse"),  # scipy needs implicit diff mode='reverse' for jacrev
], ids=["dense-jacfwd", "dense-jacrev", "scipy-jacfwd", "scipy-jacrev"])
def test_sensitivity_initial_condition(backend, ad_fun, diff_mode):
    """d(T_f)/dT0 for transient heat with IC T0·sin(πx); exact by linearity."""
    T0_ref, k_ref = 2.0, 0.5
    t_final, n_steps = 0.2, 20

    sim = _strip_sim(_heat_model, dae.BackwardEuler(), backend, diff_mode)
    sim.add_strong_bc("T", lambda x: jnp.isclose(x[0], 0.0),
                         lambda x, t: 0.0)
    sim.add_strong_bc("T", lambda x: jnp.isclose(x[0], L),
                         lambda x, t: 0.0)
    sim.settings["T0"] = T0_ref
    sim.settings["k"]  = k_ref
    sim.set_initial_conditions(values={
        "T": lambda x, s: s["T0"] * jnp.sin(jnp.pi * x[0])
    })
    sim.initialize(verbose=-1)
    sim.prepare()
    settings_0 = {**sim.settings, "current time": 0.0}

    result_ref = sim.run(dt0=t_final / n_steps, time_span=t_final, num_time_steps=n_steps)
    grad_exact = result_ref.q["T"] / T0_ref  # T_f(T0) = T0·T_f(1) by linearity

    def compute(T0):
        sim.settings = dict(settings_0)
        sim.settings["T0"] = T0
        return sim.run(dt0=t_final / n_steps, time_span=t_final, num_time_steps=n_steps).q["T"]

    grad_ad = ad_fun(compute)(T0_ref)
    assert jnp.allclose(grad_ad, grad_exact, atol=1e-11, rtol=1e-11), (
        f"{ad_fun.__name__}/{backend}: max err = "
        f"{float(jnp.max(jnp.abs(grad_ad - grad_exact))):.3e}"
    )


@pytest.mark.parametrize("num_stages", [1, 2])
def test_sparse_transient_heat_runs_with_decoupled_dirk_stages(num_stages):
    sim = _strip_sim(_heat_model, dae.DiagonallyImplicitRungeKutta(num_stages), "scipy")
    sim.add_strong_bc("T", lambda x: jnp.isclose(x[0], 0.0),
                         lambda x, t: 0.0)
    sim.add_strong_bc("T", lambda x: jnp.isclose(x[0], L),
                         lambda x, t: 0.0)
    sim.set_initial_conditions(values={
        "T": lambda x: jnp.sin(jnp.pi * x[0])
    })
    sim.initialize(verbose=-1)
    sim.prepare()

    result = sim.run(dt0=0.1, time_span=0.1, num_time_steps=1)

    assert int(result.num_accepted) == 1
    assert int(result.num_rejected) == 0
    assert result.q["T"].shape == sim.dofs["T"].shape
    assert bool(jnp.all(jnp.isfinite(result.q["T"])))
