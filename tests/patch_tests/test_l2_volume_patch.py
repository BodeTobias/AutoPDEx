import numpy as np
import pytest

from autopdex import SimState, dae, spaces

from conftest import (
    L2_MESH_CASES,
    affine_data,
    assert_field_matches_exact,
    exact_affine,
    load_patch_mesh,
)


def _effective_dim(mesh):
    return 2 if mesh.points.shape[1] == 3 and np.allclose(mesh.points[:, 2], 0.0) else mesh.points.shape[1]


@pytest.mark.parametrize("mesh_file, cell_type, order", L2_MESH_CASES)
def test_l2_scalar_affine_volume_patch_all_mesh_topologies(mesh_file, cell_type, order):
    mesh = load_patch_mesh(mesh_file)
    dim = int(_effective_dim(mesh))
    field = "p"
    offset, slope = affine_data(())
    exact = exact_affine(offset, slope)

    sim = SimState({field: spaces.L2(order=order, dim=dim, field_dimension=1)})
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({field: dae.NoTimeDerivative()})

    def mass_projection_patch(ctx):
        t = ctx.settings.get("current time", 0.0)
        x = ctx.trial_ansatz["physical coor"](ctx.x_int)
        value = ctx.trial_ansatz[field](ctx.x_int, t) - exact(x, t)
        return value * ctx.test_ansatz[field](ctx.x_int)

    sim.add_model("__all__", "weak form", mass_projection_patch, integration_order=2 * order)
    sim.initialize(verbose=0)
    sim.prepare(clean_up=False)
    result = sim.run(dt0=1.0, time_span=1.0, num_time_steps=1)

    assert int(result.num_accepted) == 1
    assert_field_matches_exact(sim, result, field, exact)
