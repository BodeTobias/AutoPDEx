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
Smoke test mirroring examples/offline/plasticity/plasticity_hli.py.

Small-strain J2 plasticity (plane strain) with linear isotropic hardening,
augmented-Lagrange return mapping via a local DAE sub-system.  The primary
field is displacement; internal variables are the plastic strain tensor eps_p
and accumulated plastic strain alpha.

This test uses a 4×2 quad mesh (vs 200×100 in the example) and only 2
time steps over [0, 0.5].  The goal is to verify that the local Newton
sub-solver, the internal variable update, and the global solve all run
without errors on a small problem.
"""

from functools import partial

import jax
import jax.numpy as jnp
import pytest

from autopdex import SimState, dae, spaces
from autopdex.models import aug_lag_potential
from autopdex.models import km_tools as km

jax.config.update("jax_enable_x64", True)

# ── Material parameters ───────────────────────────────────────────────────────
Lx, Ly = 10.0, 1.0
E = 30_000.0
nu = 0.2
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))
sigma_y0 = 24.0
H_iso = 100.0


def _strain_energy(eps_e):
    return mu * eps_e @ eps_e + 0.5 * lam * km.trace(eps_e) ** 2


_stress = jax.jacrev(_strain_energy)


# ── Local DAE (return mapping) ────────────────────────────────────────────────

def _local_dae(q_fun, t, settings):
    q, q_t = jax.jvp(q_fun, (t,), (1.0,))

    def potential(sigma, alpha_dot):
        s = km.dev(sigma)
        yield_func = (
            jnp.sqrt(3.0 / 2.0) * jnp.sqrt(s @ s + 1e-16)
            - (sigma_y0 + settings["H_iso"] * q["alpha"])
        )
        return sigma @ q_t["eps_p"] - aug_lag_potential(-alpha_dot, -yield_func)

    e_sig, e_lmb = jax.jacrev(potential, argnums=(0, 1))(
        _stress(settings["eps"] - q["eps_p"]), q_t["alpha"]
    )
    return jnp.concatenate([jnp.atleast_1d(e_lmb), e_sig.flatten()])


# ── Global weak form ──────────────────────────────────────────────────────────

def _model(ctx: SimState.ModelContext):
    if ctx.mode in ("weak form", "internal variables"):
        grad_u = jax.jacfwd(ctx.trial_ansatz["displacement"])(ctx.x_int, ctx.t)
        eps = km.pack_z_shear_zero(grad_u)
        new_int_vars = ctx.solve_local(
            "plasticity", inputs={"eps": eps, "H_iso": ctx.settings["H_iso"]}
        )
        if ctx.mode == "internal variables":
            return new_int_vars
        sigma = _stress(eps - new_int_vars["eps_p"])
        grad_v = jax.jacfwd(ctx.test_ansatz["displacement"])(ctx.x_int)
        eps_test = km.pack_z_shear_zero(grad_v)
        return sigma @ eps_test

    if ctx.mode == "output":
        return {
                "accumulated plastic strain": ctx.internal_vars["alpha"],
            }


# ── Test ──────────────────────────────────────────────────────────────────────

def test_plasticity_hli_runs():
    """Two time steps of J2 plasticity; displacement and internal vars are finite."""
    sim = SimState({
        "displacement": spaces.H1(order=1, dim=2, field_dimension=2),
        "eps_p":  spaces.InternalVariable(field_dimension=4),
        "alpha":  spaces.InternalVariable(field_dimension=1),
    })
    sim.add_structured_mesh(
        (4, 2),
        [[0.0, 0.0], [Lx, 0.0], [Lx, Ly], [0.0, Ly]],
        "quad",
    )
    sim.add_temporal_discretization({
        "displacement": dae.NoTimeDerivative(),
        "eps_p":  dae.BackwardEuler(),
        "alpha":  dae.BackwardEuler(),
    })
    sim.add_local_subsystem("plasticity", _local_dae, fields=("eps_p", "alpha"))
    sim.add_model("__all__", "weak form", _model, internal_variables=("eps_p", "alpha"))
    sim.add_strong_bc(
        "displacement",
        on_boundary_fun=lambda x: jnp.isclose(x[0], 0.0),
        value_fun=lambda x, t: jnp.zeros((2,)),
    )
    sim.add_weak_bc(
        "displacement",
        on_boundary_fun=lambda x: jnp.isclose(x[0], Lx),
        value_fun=lambda x, t: jnp.asarray([0.0, -2.0 * t]),
    )
    sim.set_step_size_controller(dae.RootIterationController(target_niters=8))
    sim.settings["H_iso"] = H_iso
    sim.initialize(verbose=-1)
    sim.prepare()

    result = sim.run(dt0=0.5, time_span=0.5, num_time_steps=2)

    assert jnp.all(jnp.isfinite(result.q["displacement"])), \
        "displacement contains non-finite values"
    assert float(result.settings["current time"]) == pytest.approx(0.5, abs=1e-8)
    assert int(result.num_accepted) > 0
