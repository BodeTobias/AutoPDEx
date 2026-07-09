import pytest

from conftest import GEOMETRY_MESH_CASES, H1_MESH_CASES, assert_field_matches_exact, load_patch_mesh, run_h1_patch


@pytest.mark.parametrize(
    "mesh_file, cell_type",
    [
        pytest.param("triangle_p1.vtu", "triangle", id="tri"),
        pytest.param("quad_p1.vtu", "quad", id="quad"),
        pytest.param("tetra_p1.vtu", "tetra", id="tet"),
        pytest.param("hexahedron_p1.vtu", "hexahedron", id="hex"),
    ],
)
def test_h1_scalar_laplace_patch_natural_bc_supported_topologies(mesh_file, cell_type):
    sim, result, exact = run_h1_patch(
        load_patch_mesh(mesh_file),
        order=1,
        field_dimension=1,
        use_natural_bc=True,
    )

    assert int(result.num_accepted) == 1
    assert_field_matches_exact(sim, result, "u", exact)


@pytest.mark.parametrize("mesh_file, cell_type, order", H1_MESH_CASES)
def test_h1_scalar_laplace_patch_all_mesh_topologies_and_orders(mesh_file, cell_type, order):
    mesh = load_patch_mesh(mesh_file)
    # The topology/order sweep stays scalar to keep the higher-order suite tractable.
    # Essential+natural BCs are exercised in the field-dimension and mixed tests.
    sim, result, exact = run_h1_patch(
        mesh,
        order=order,
        field_dimension=1,
        use_natural_bc=False,
    )

    assert int(result.num_accepted) == 1
    assert_field_matches_exact(sim, result, "u", exact)


@pytest.mark.parametrize("mesh_file, cell_type, geometry_order", GEOMETRY_MESH_CASES)
def test_h1_constant_laplace_patch_all_higher_order_geometry_mesh_topologies(mesh_file, cell_type, geometry_order):
    sim, result, exact = run_h1_patch(
        load_patch_mesh(mesh_file),
        order=1,
        field_dimension=1,
        use_natural_bc=False,
        exact_solution="constant",
    )

    assert int(result.num_accepted) == 1
    assert_field_matches_exact(sim, result, "u", exact)
