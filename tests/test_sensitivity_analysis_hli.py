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
Tests mirroring examples/offline/sensitivity_analysis_hli.ipynb.

Two problems with exact analytical solutions:

Part 1 — Stationary heat conduction on a thin strip [0,1]×[0,0.1]:
    −k ∇²T = 0,  T(0) = T_D,  ∂T/∂n|_{x=1} = q_N
    Exact solution: T(x) = T_D + (q_N / k) · x

    Sensitivities via jacfwd are compared to the exact derivatives:
        dT/dk   = −(q_N / k²) · x
        dT/dq_N = x / k
        dT/dT_D = 1

Part 2 — Transient heat conduction on the same strip:
    ∂T/∂t = k ∇²T,  T(x,0) = T0 sin(πx),  T(0) = T(1) = 0
    Exact solution: T(x,t) = T0 exp(−k π² t) sin(πx)

    Tests: callable IC evaluation, forward solve accuracy, and that
    jacfwd agrees with central FD (both carry the same Backward-Euler
    truncation error, so they match to within floating-point noise).
"""

import jax
import jax.numpy as jnp
import pytest

from autopdex import SimState, dae, spaces

jax.config.update("jax_enable_x64", True)

L, H = 1.0, 0.1
k_ref, T_D_ref, q_N_ref = 2.0, 3.0, 4.0
T0_ref, k2_ref = 2.0, 0.5
t_final, n_steps = 0.2, 20  # reduced from 200; Backward-Euler error scales with dt


# ── Models ────────────────────────────────────────────────────────────────────

def _model_poisson(ctx: SimState.ModelContext):
    if ctx.mode != "weak form":
        return {}
    k = ctx.settings["k"]
    grad_T = jax.jacfwd(ctx.trial_ansatz["T"])(ctx.x_int, ctx.t)
    grad_v = jax.jacfwd(ctx.test_ansatz["T"])(ctx.x_int)
    return k * jnp.dot(grad_T, grad_v)


def _model_heat(ctx: SimState.ModelContext):
    if ctx.mode != "weak form":
        return {}
    k = ctx.settings["k"]
    T_dot  = jax.jacfwd(ctx.trial_ansatz["T"], 1)(ctx.x_int, ctx.t)
    grad_T = jax.jacfwd(ctx.trial_ansatz["T"], 0)(ctx.x_int, ctx.t)
    v      = ctx.test_ansatz["T"](ctx.x_int)
    grad_v = jax.jacfwd(ctx.test_ansatz["T"])(ctx.x_int)
    return T_dot * v + k * jnp.dot(grad_T, grad_v)


# ── Part 1 factory ────────────────────────────────────────────────────────────

def _make_steady_sim():
    sim = SimState({"T": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.add_structured_mesh((20, 2), [[0, 0], [L, 0], [L, H], [0, H]], "quad", order=1)
    sim.add_temporal_discretization({"T": dae.NoTimeDerivative()})
    sim.add_model("__all__", "weak form", _model_poisson)
    sim.add_strong_bc("T", lambda x: jnp.isclose(x[0], 0.0), lambda x, t, s: s["T_D"])
    sim.add_weak_bc("T",  lambda x: jnp.isclose(x[0], L),    lambda x, t, s: s["q_N"])
    sim.settings["k"]   = k_ref
    sim.settings["T_D"] = T_D_ref
    sim.settings["q_N"] = q_N_ref
    sim.initialize(verbose=-1)
    sim.prepare()
    return sim


# ── Part 1 tests ──────────────────────────────────────────────────────────────

def test_steady_forward_exact():
    """Forward solution matches T(x) = T_D + (q_N/k)·x to machine precision."""
    sim = _make_steady_sim()
    result = sim.run(dt0=1.0, time_span=1.0, num_time_steps=1)
    coords  = sim.settings["node coordinates"]
    T_exact = T_D_ref + q_N_ref / k_ref * coords[:, 0]
    assert jnp.max(jnp.abs(result.q["T"] - T_exact)) < 1e-11, (
        f"forward error = {float(jnp.max(jnp.abs(result.q['T'] - T_exact))):.2e}"
    )


def test_steady_sensitivity_k():
    """AD (jacfwd) dT/dk matches −(q_N/k²)·x to ≤1e-11."""
    sim = _make_steady_sim()
    settings_0 = dict(sim.settings)
    coords = settings_0["node coordinates"]

    def run_with(k):
        sim.settings = dict(settings_0)
        sim.settings["k"] = k
        return sim.run(dt0=1.0, time_span=1.0, num_time_steps=1).q["T"]

    dT_ad    = jax.jit(jax.jacfwd(run_with))(k_ref)
    dT_exact = -q_N_ref / k_ref**2 * coords[:, 0]
    assert jnp.max(jnp.abs(dT_ad - dT_exact)) < 1e-11, (
        f"dT/dk: AD vs exact = {float(jnp.max(jnp.abs(dT_ad - dT_exact))):.2e}"
    )


def test_steady_sensitivity_q_N():
    """AD (jacfwd) dT/dq_N matches x/k to ≤1e-11."""
    sim = _make_steady_sim()
    settings_0 = dict(sim.settings)
    coords = settings_0["node coordinates"]

    def run_with(q_N):
        sim.settings = dict(settings_0)
        sim.settings["q_N"] = q_N
        return sim.run(dt0=1.0, time_span=1.0, num_time_steps=1).q["T"]

    dT_ad    = jax.jit(jax.jacfwd(run_with))(q_N_ref)
    dT_exact = coords[:, 0] / k_ref
    assert jnp.max(jnp.abs(dT_ad - dT_exact)) < 1e-11, (
        f"dT/dq_N: AD vs exact = {float(jnp.max(jnp.abs(dT_ad - dT_exact))):.2e}"
    )


def test_steady_sensitivity_T_D():
    """AD (jacfwd) dT/dT_D matches 1 (everywhere) to ≤1e-11."""
    sim = _make_steady_sim()
    settings_0 = dict(sim.settings)
    coords = settings_0["node coordinates"]

    def run_with(T_D):
        sim.settings = dict(settings_0)
        sim.settings["T_D"] = T_D
        return sim.run(dt0=1.0, time_span=1.0, num_time_steps=1).q["T"]

    dT_ad    = jax.jit(jax.jacfwd(run_with))(T_D_ref)
    dT_exact = jnp.ones(len(coords))
    assert jnp.max(jnp.abs(dT_ad - dT_exact)) < 1e-11, (
        f"dT/dT_D: AD vs exact = {float(jnp.max(jnp.abs(dT_ad - dT_exact))):.2e}"
    )


# ── Part 2 factory ────────────────────────────────────────────────────────────

def _make_transient_sim():
    sim = SimState({"T": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.add_structured_mesh((20, 2), [[0, 0], [L, 0], [L, H], [0, H]], "quad", order=1)
    sim.add_temporal_discretization({"T": dae.BackwardEuler()})
    sim.add_model("__all__", "weak form", _model_heat)
    sim.add_strong_bc("T", lambda x: jnp.isclose(x[0], 0.0), lambda x, t: 0.0)
    sim.add_strong_bc("T", lambda x: jnp.isclose(x[0], L),   lambda x, t: 0.0)
    sim.settings["T0"] = T0_ref
    sim.settings["k"]  = k2_ref
    sim.set_initial_conditions(values={"T": lambda x, s: s["T0"] * jnp.sin(jnp.pi * x[0])})
    sim.initialize(verbose=-1)
    sim.prepare()
    return sim


# ── Part 2 tests ──────────────────────────────────────────────────────────────

def test_transient_callable_ic_applied():
    """Callable IC T0·sin(πx) is applied and close to the analytical value (< 1% error).

    set_initial_conditions uses L2 projection, not nodal interpolation, so the
    DOF values differ slightly from exact sin(πx) at node coordinates.
    """
    sim = _make_transient_sim()
    coords = sim.settings["node coordinates"]
    T_ic_exact = T0_ref * jnp.sin(jnp.pi * coords[:, 0])
    max_err = float(jnp.max(jnp.abs(sim.dofs["T"] - T_ic_exact)))
    assert max_err < 0.01, f"IC application error = {max_err:.2e}"


def test_transient_forward_accuracy():
    """Backward-Euler solution at t=0.2 is within 3% of the exact decay (n_steps=20)."""
    sim = _make_transient_sim()
    settings_0 = {**sim.settings, "current time": 0.0}
    sim.settings = dict(settings_0)
    result = sim.run(dt0=t_final / n_steps, time_span=t_final, num_time_steps=n_steps)
    coords = settings_0["node coordinates"]
    T_exact = T0_ref * jnp.exp(-k2_ref * jnp.pi**2 * t_final) * jnp.sin(jnp.pi * coords[:, 0])
    rel_err = jnp.max(jnp.abs(result.q["T"] - T_exact)) / (jnp.max(jnp.abs(T_exact)) + 1e-15)
    assert rel_err < 0.03, f"relative forward error = {float(rel_err):.2e}"


def test_transient_sensitivity_ic_ad_vs_fd():
    """jacfwd dT/dT0 agrees with central FD (both carry identical BE truncation error)."""
    sim = _make_transient_sim()
    settings_0 = {**sim.settings, "current time": 0.0}

    def compute(T0):
        sim.settings = dict(settings_0)
        sim.settings["T0"] = T0
        return sim.run(
            dt0=t_final / n_steps, time_span=t_final, num_time_steps=n_steps
        ).q["T"]

    dT_ad = jax.jit(jax.jacfwd(compute))(T0_ref)
    eps   = 1e-5
    dT_fd = (compute(T0_ref + eps) - compute(T0_ref - eps)) / (2 * eps)
    assert jnp.allclose(dT_ad, dT_fd, atol=1e-6, rtol=0.0), (
        f"max |AD − FD| = {float(jnp.max(jnp.abs(dT_ad - dT_fd))):.2e}"
    )
