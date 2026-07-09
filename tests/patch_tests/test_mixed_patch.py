import pytest

import jax
import jax.numpy as jnp

from autopdex import SimState, dae, spaces

from conftest import (
    MAX_PATCH_SPACE_ORDER,
    affine_data,
    assert_field_matches_exact,
    exact_affine,
    load_patch_mesh,
)


@pytest.mark.parametrize("order", [1, MAX_PATCH_SPACE_ORDER], ids=lambda p: f"space-p{p}")
def test_mixed_scalar_vector_patch_problem_with_essential_and_natural_bcs(order):
    mesh = load_patch_mesh("quad_p1.vtu")
    dim = 2
    phi_offset, _ = affine_data(())
    u_offset, u_slope = affine_data((dim,))
    phi_exact = lambda x, t: t * phi_offset
    u_exact = exact_affine(u_offset, u_slope)

    sim = SimState(
        {
            "phi": spaces.H1(order=order, dim=dim, field_dimension=1),
            "u": spaces.H1(order=order, dim=dim, field_dimension=dim),
        }
    )
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({"phi": dae.NoTimeDerivative(), "u": dae.NoTimeDerivative()})

    def mixed_weak_form(ctx):
        t = ctx.settings.get("current time", 0.0)
        grad_phi = jax.jacfwd(lambda xi: ctx.trial_ansatz["phi"](xi, t))(ctx.x_int)
        grad_eta = jax.jacfwd(ctx.test_ansatz["phi"])(ctx.x_int)
        grad_u = jax.jacfwd(lambda xi: ctx.trial_ansatz["u"](xi, t))(ctx.x_int)
        grad_v = jax.jacfwd(ctx.test_ansatz["u"])(ctx.x_int)
        weak = jnp.sum(grad_phi * grad_eta) + jnp.sum(grad_u * grad_v)
        return weak

    sim.add_model("__all__", "weak form", mixed_weak_form, integration_order=2 * order + 1)
    sim.add_strong_bc(
        field="phi",
        on_boundary_fun=lambda x: jnp.isclose(x[0], 0.0),
        value_fun=phi_exact,
    )
    sim.add_strong_bc(
        field="u",
        on_boundary_fun=lambda x: jnp.isclose(x[0], 0.0),
        value_fun=u_exact,
    )
    sim.add_weak_bc(
        field="u",
        on_boundary_fun=lambda x: jnp.isclose(x[0], 1.0),
        value_fun=lambda x, t: t * u_slope,
    )

    sim.initialize(verbose=0)
    sim.prepare(clean_up=False)
    result = sim.run(dt0=1.0, time_span=1.0, num_time_steps=1)

    assert int(result.num_accepted) == 1
    assert_field_matches_exact(sim, result, "phi", phi_exact)
    assert_field_matches_exact(sim, result, "u", u_exact)
