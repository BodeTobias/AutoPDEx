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
Tests for the 'potential' domain_type in the high-level SimState API.

A scalar potential Π(u) = ∫(½|∇u|² − f·u)dΩ is stationary exactly when u
solves the Poisson equation −Δu = f with the given boundary conditions. We
exploit this to verify that the potential formulation produces the same discrete
solution as the corresponding weak form.

Test problems (2D structured quad mesh, H1 order-1 elements):

  1. Poisson with unit source on [0,1]²    — agreement with weak form
  2. Poisson with spatially varying source — agreement with weak form
  3. Poisson with material parameter k     — sensitivities via jacfwd / jacrev
"""

import pytest
import jax
import jax.numpy as jnp

from autopdex import SimState, dae, spaces

jax.config.update("jax_enable_x64", True)

# ── Shared geometry ────────────────────────────────────────────────────────────
N = 8
VERTICES = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
_ALL_BOUNDARY = lambda x: (
    jnp.isclose(x[0], 0.0) | jnp.isclose(x[0], 1.0)
    | jnp.isclose(x[1], 0.0) | jnp.isclose(x[1], 1.0)
)


# ── Model definitions ──────────────────────────────────────────────────────────

def _weak_form_model(source_fun):
    def integrand(ctx):
        if ctx.mode != "weak form":
            return {}
        k = ctx.settings.get("k", 1.0)
        grad_u = jax.jacfwd(ctx.trial_ansatz["u"], 0)(ctx.x_int, ctx.t).ravel()
        grad_v = jax.jacfwd(ctx.test_ansatz["u"])(ctx.x_int).ravel()
        v      = ctx.test_ansatz["u"](ctx.x_int).sum()
        f      = source_fun(ctx.x_int, ctx.settings)
        return k * jnp.dot(grad_u, grad_v) - f * v
    return integrand


def _potential_model(source_fun):
    def integrand(ctx):
        k = ctx.settings.get("k", 1.0)
        u      = ctx.trial_ansatz["u"](ctx.x_int, ctx.t).sum()
        grad_u = jax.jacfwd(ctx.trial_ansatz["u"], 0)(ctx.x_int, ctx.t).ravel()
        f      = source_fun(ctx.x_int, ctx.settings)
        return 0.5 * k * jnp.dot(grad_u, grad_u) - f * u
    return integrand


# ── SimState factory ───────────────────────────────────────────────────────────

def _make_sim(domain_type, integrand, backend="scipy", diff_mode="forward"):
    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.add_structured_mesh((N, N), VERTICES, "quad", order=1)
    sim.add_temporal_discretization({"u": dae.NoTimeDerivative()})
    sim.add_model("__all__", domain_type, integrand)
    sim.add_strong_bc("u", _ALL_BOUNDARY, lambda x, t, s: jnp.zeros(1))
    sim.static_settings["solver backend"] = backend
    sim.static_settings["implicit diff mode"] = diff_mode
    sim.initialize(verbose=-1)
    sim.prepare()
    return sim


def _solve(sim):
    return sim.run(dt0=1.0, time_span=1.0, num_time_steps=1).q["u"]


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_potential_agrees_with_weak_form_unit_source():
    """Potential and weak form produce identical solutions for -Δu = 1."""
    source = lambda x, s: 1.0
    u_wf = _solve(_make_sim("weak form", _weak_form_model(source)))
    u_pt = _solve(_make_sim("potential", _potential_model(source)))
    assert jnp.allclose(u_wf, u_pt, atol=1e-12), (
        f"max diff = {float(jnp.max(jnp.abs(u_wf - u_pt))):.3e}"
    )


def test_potential_agrees_with_weak_form_varying_source():
    """Potential and weak form agree for spatially varying f(x) = sin(πx)sin(πy)."""
    source = lambda x, s: jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])
    u_wf = _solve(_make_sim("weak form", _weak_form_model(source)))
    u_pt = _solve(_make_sim("potential", _potential_model(source)))
    assert jnp.allclose(u_wf, u_pt, atol=1e-12), (
        f"max diff = {float(jnp.max(jnp.abs(u_wf - u_pt))):.3e}"
    )


@pytest.mark.parametrize("ad_fun,diff_mode", [
    (jax.jacfwd, "forward"),
    (jax.jacrev, "reverse"),
], ids=["jacfwd", "scipy-jacrev"])
def test_potential_sensitivity_material_param(ad_fun, diff_mode):
    """d(Σu)/dk via AD matches exact derivative from linearity: u(k) = u(1)/k."""
    k_ref = 2.5
    source = lambda x, s: 1.0

    sim = _make_sim("potential", _potential_model(source), backend="scipy", diff_mode=diff_mode)
    settings_0 = dict(sim.settings)

    sim.settings["k"] = k_ref
    u_ref = _solve(sim)
    grad_exact = -u_ref / k_ref  # exact nodal derivative by linearity

    def compute(k):
        sim.settings = dict(settings_0)
        sim.settings["k"] = k
        return sim.run(dt0=1.0, time_span=1.0, num_time_steps=1).q["u"]

    grad_ad = ad_fun(compute)(k_ref)
    assert jnp.allclose(grad_ad, grad_exact, atol=1e-10, rtol=1e-10), (
        f"{ad_fun.__name__}: max err = "
        f"{float(jnp.max(jnp.abs(grad_ad - grad_exact))):.3e}"
    )
