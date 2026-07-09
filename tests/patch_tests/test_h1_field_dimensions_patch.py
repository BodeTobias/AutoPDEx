import pytest

from conftest import MAX_PATCH_SPACE_ORDER, assert_field_matches_exact, load_patch_mesh, run_h1_patch


@pytest.mark.parametrize("order", [1, MAX_PATCH_SPACE_ORDER], ids=lambda p: f"space-p{p}")
@pytest.mark.parametrize(
    "field_dimension",
    [pytest.param(1, id="scalar"), pytest.param(2, id="vector"), pytest.param((2, 2), id="tensor")],
)
def test_h1_laplace_patch_field_dimensions(field_dimension, order):
    use_natural_bc = field_dimension != 1
    sim, result, exact = run_h1_patch(
        load_patch_mesh("quad_p1.vtu"),
        order=order,
        field_dimension=field_dimension,
        use_natural_bc=use_natural_bc,
    )

    assert int(result.num_accepted) == 1
    assert_field_matches_exact(sim, result, "u", exact)
