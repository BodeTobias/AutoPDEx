from pathlib import Path

from flax.core import FrozenDict
import meshio
import numpy as np
import pytest

from jax import config
import jax
import jax.numpy as jnp

from autopdex import SimState, assembler, dae, seeder, spaces

config.update("jax_enable_x64", True)

MESH_DIR = Path(__file__).resolve().parent / "meshes"
MAX_PATCH_SPACE_ORDER = 5
MAX_PATCH_GEOMETRY_ORDER = 3

PATCH_MESHES = [
    ("triangle_p1.vtu", "triangle"),
    ("quad_p1.vtu", "quad"),
    ("tetra_p1.vtu", "tetra"),
    ("hexahedron_p1.vtu", "hexahedron"),
]

GEOMETRY_MESH_ORDER_LIMITS = {
    "triangle": 10,
    "quad": 20,
    "tetra": 5,
    "hexahedron": 2,
}


# The p1 meshes are already covered by PATCH_MESHES. These cases add
# higher-order geometry topologies up to MAX_PATCH_GEOMETRY_ORDER with a cheap
# linear H1 approximation space.
def _geometry_mesh_cases():
    cases = []
    for cell_type, max_order in GEOMETRY_MESH_ORDER_LIMITS.items():
        max_order = min(max_order, MAX_PATCH_GEOMETRY_ORDER)
        short_cell = "hex" if cell_type == "hexahedron" else cell_type
        for geometry_order in range(2, max_order + 1):
            mesh_file = f"{cell_type}_p{geometry_order}.vtu"
            if not (MESH_DIR / mesh_file).exists():
                continue
            cases.append(
                pytest.param(
                    mesh_file,
                    cell_type,
                    geometry_order,
                    id=f"{short_cell}-geometry-p{geometry_order}-space-p1",
                )
            )
    return cases


# The p1 meshes stay fixed; only the approximation space order is varied.
# Hex/brick bases are implemented up to order 2 in the current backend.
# H1 tetra p4/p5 need high-order face-DOF permutations and are too slow for
# the dense routine patch suite; L2 still covers tetra orders up to p5.
H1_MAX_SPACE_ORDER = {"triangle": 5, "quad": 5, "tetra": 3, "hexahedron": 2}
# L2 tetra p3+ is currently too expensive on the stored unstructured p1 mesh
# for the routine dense patch suite.
L2_MAX_SPACE_ORDER = {"triangle": 5, "quad": 5, "tetra": 2, "hexahedron": 2}


def _space_order_cases(max_order_by_cell):
    cases = []
    for mesh_file, cell_type in PATCH_MESHES:
        max_order = min(MAX_PATCH_SPACE_ORDER, max_order_by_cell[cell_type])
        short_cell = "hex" if cell_type == "hexahedron" else cell_type
        for order in range(1, max_order + 1):
            cases.append(
                pytest.param(
                    mesh_file,
                    cell_type,
                    order,
                    id=f"{short_cell}-space-p{order}-mesh-p1",
                )
            )
    return cases


H1_MESH_CASES = _space_order_cases(H1_MAX_SPACE_ORDER)
L2_MESH_CASES = _space_order_cases(L2_MAX_SPACE_ORDER)
GEOMETRY_MESH_CASES = _geometry_mesh_cases()


def load_patch_mesh(filename):
    return meshio.read(MESH_DIR / filename)


def base_type(cell_type):
    lower = cell_type.lower()
    if lower.startswith("triangle") or lower.startswith("vtk_lagrange_triangle"):
        return "triangle"
    if lower.startswith("quad") or lower.startswith("vtk_lagrange_quadrilateral"):
        return "quad"
    if lower.startswith("tetra") or lower.startswith("vtk_lagrange_tetrahedron"):
        return "tetra"
    if lower.startswith("hexahedron") or lower.startswith("vtk_lagrange_hexahedron"):
        return "hex"
    if lower.startswith("line"):
        return "line"
    return cell_type


def mesh_type(cell_type):
    base = base_type(cell_type)
    return "hexahedron" if base == "hex" else base


def topological_dim(cell_type):
    return {"line": 1, "triangle": 2, "quad": 2, "tetra": 3, "hex": 3}[base_type(cell_type)]


def quadrature(cell_type, order):
    mt = mesh_type(cell_type)
    if mt == "triangle":
        return seeder.int_pts_ref_tri(order=order)
    if mt == "tetra":
        return seeder.int_pts_ref_tet(order=order)
    if mt == "quad":
        return seeder.gauss_legendre_nd(dimension=2, order=order)
    if mt == "hexahedron":
        return seeder.gauss_legendre_nd(dimension=3, order=order)
    raise ValueError(cell_type)


def sample_points(cell_type):
    return quadrature(cell_type, order=2)[0]


def mapping_basis(cell_type):
    mt = mesh_type(cell_type)
    if mt in ("quad", "hexahedron"):
        return spaces.fem_iso_line_quad_brick
    if mt in ("triangle", "tetra"):
        return spaces.fem_iso_line_tri_tet
    raise ValueError(cell_type)


def field_shape_from_dimension(field_dimension):
    if field_dimension == 1:
        return ()
    if isinstance(field_dimension, tuple):
        return field_dimension
    return (int(field_dimension),)


def affine_data(field_shape):
    if field_shape == ():
        return jnp.asarray(0.35), jnp.asarray(0.4)
    n = int(np.prod(field_shape))
    offset = jnp.linspace(-0.2, 0.3, n).reshape(field_shape)
    slope = jnp.linspace(0.15, 0.45, n).reshape(field_shape)
    return offset, slope


def exact_affine(offset, slope):
    return lambda x, t: t * (offset + slope * x[0])


def laplace_patch_integrand(field):
    def weak_form(ctx):
        t = ctx.settings.get("current time", 0.0)
        grad_u = jax.jacfwd(lambda xi: ctx.trial_ansatz[field](xi, t))(ctx.x_int)
        grad_v = jax.jacfwd(ctx.test_ansatz[field])(ctx.x_int)
        return jnp.sum(grad_u * grad_v)

    return weak_form


def run_h1_patch(mesh, order, field_dimension, use_natural_bc=True, exact_solution="affine"):
    dim = int(mesh.points.shape[1])
    if dim == 3 and np.allclose(np.asarray(mesh.points)[:, 2], 0.0):
        dim = 2
    field = "u"
    field_shape = field_shape_from_dimension(field_dimension)
    offset, slope = affine_data(field_shape)
    if exact_solution == "constant":
        slope = jnp.zeros_like(slope)
    elif exact_solution != "affine":
        raise ValueError("exact_solution must be 'affine' or 'constant'.")
    exact = exact_affine(offset, slope)

    sim = SimState({field: spaces.H1(order=order, dim=dim, field_dimension=field_dimension)})
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({field: dae.NoTimeDerivative()})
    sim.add_model("__all__", "weak form", laplace_patch_integrand(field), integration_order=2 * order)

    if use_natural_bc:
        sim.add_strong_bc(
            field=field,
            on_boundary_fun=lambda x: jnp.isclose(x[0], 0.0),
            value_fun=exact,
        )
        sim.add_weak_bc(
            field=field,
            on_boundary_fun=lambda x: jnp.isclose(x[0], 1.0),
            value_fun=lambda x, t: t * slope,
        )
    else:
        sim.add_strong_bc(
            field=field,
            on_boundary_fun=lambda x: jnp.any(jnp.isclose(x, 0.0) | jnp.isclose(x, 1.0)),
            value_fun=exact,
        )

    sim.initialize(verbose=0)
    sim.prepare(clean_up=False)
    result = sim.run(dt0=1.0, time_span=1.0, num_time_steps=1)
    return sim, result, exact


def _measure_factor(jac, dim):
    jac = jnp.asarray(jac)
    if dim == 1:
        return jnp.linalg.norm(jac)
    if dim == 2:
        return jnp.sqrt(jnp.maximum(jnp.linalg.det(jac.T @ jac), 0.0))
    return jnp.abs(jnp.linalg.det(jac))


def _field_metric_potential(field, basis, mapping, dim, q_points, q_weights, exact, reference_only):
    q_points = jnp.asarray(q_points)
    q_weights = jnp.asarray(q_weights)

    def user_potential(fI, xI, elem_number, settings, static_settings, set):
        local_dofs = fI[field]
        t = settings.get("current time", 0.0)

        def field_fun(x):
            return basis(x, xI, local_dofs, settings, True, dim)

        def mapping_fun(x):
            return mapping(x, xI, xI, settings, False, dim)

        jacobian_fun = jax.jacfwd(mapping_fun)

        def functional(x_int, w_int):
            x_phys = mapping_fun(x_int)
            expected = jnp.asarray(exact(x_phys, t))
            quantity = expected if reference_only else jnp.asarray(field_fun(x_int)) - expected
            measure = _measure_factor(jacobian_fun(x_int), dim)
            return w_int * measure * jnp.sum(quantity * quantity)

        return jax.vmap(functional, (0, 0), 0)(q_points, q_weights).sum()

    return user_potential


def _field_metric_assembler_inputs(sim, result, field, exact, reference_only):
    field_space = sim.static_settings["spatial discretizations"][field]
    pack = sim._fe[field]
    dim = int(field_space.dim)
    q_order = max(2, 2 * int(getattr(field_space, "order", 1)))

    connectivity = []
    models = []
    for cell_type, cinfo in sim.mesh_info.cells.items():
        if topological_dim(cell_type) != dim or cell_type not in pack.volume:
            continue

        q_points, q_weights = quadrature(cell_type, order=q_order)
        models.append(
            _field_metric_potential(
                field,
                field_space.basis(base_type(cell_type)),
                mapping_basis(cell_type),
                dim,
                q_points,
                q_weights,
                exact,
                reference_only,
            )
        )
        connectivity.append(
            {
                field: pack.volume[cell_type],
                "physical coor": cinfo.nodes_of_cells,
            }
        )

    settings = dict(result.settings)
    settings["connectivity"] = tuple(connectivity)
    settings["node coordinates"] = {"physical coor": sim.mesh_info.points}
    static_settings = FrozenDict(
        {
            "assembling mode": ("user potential",) * len(models),
            "model": tuple(models),
        }
    )
    dofs = {field: result.q[field]}
    return dofs, settings, static_settings


def assert_field_matches_exact(sim, result, field, exact, atol=2.0e-7, rtol=2.0e-7):
    err_inputs = _field_metric_assembler_inputs(sim, result, field, exact, reference_only=False)
    ref_inputs = _field_metric_assembler_inputs(sim, result, field, exact, reference_only=True)

    err_sq = float(assembler.integrate_functional(*err_inputs))
    ref_sq = float(assembler.integrate_functional(*ref_inputs))

    abs_err = float(np.sqrt(max(err_sq, 0.0)))
    rel_err = abs_err / max(float(np.sqrt(max(ref_sq, 0.0))), 1.0)
    assert abs_err < atol or rel_err < rtol
