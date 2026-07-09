import numpy as np
import jax
import jax.numpy as jnp
import meshio

from autopdex import SimState, dae, spaces

jax.config.update("jax_enable_x64", True)


def _rt0_divergence_phys(fun, xi, mapping):
    d_dxi = jax.jacfwd(fun)(xi)
    jac = jax.jacfwd(mapping)(xi)
    return jnp.einsum("ai,ia->", d_dxi, jnp.linalg.inv(jac))


def test_rt0_reference_basis_fluxes_and_divergence():
    space = spaces.HDiv(order=0, dim=2, field_dimension=1)
    basis = space.basis("triangle")
    xI = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    edges = (
        (jnp.asarray([0.0, 0.0]), jnp.asarray([1.0, 0.0]), jnp.asarray([0.0, -1.0])),
        (jnp.asarray([1.0, 0.0]), jnp.asarray([0.0, 1.0]), jnp.asarray([1.0, 1.0]) / jnp.sqrt(2.0)),
        (jnp.asarray([0.0, 1.0]), jnp.asarray([0.0, 0.0]), jnp.asarray([-1.0, 0.0])),
    )

    for dof in range(3):
        coeffs = jnp.zeros((3,)).at[dof].set(1.0)
        ansatz = lambda xi: basis(xi, xI, coeffs, {}, False, 2, 0, 0)
        fluxes = []
        for a, b, normal in edges:
            length = jnp.linalg.norm(b - a)
            midpoint = 0.5 * (a + b)
            fluxes.append(jnp.dot(ansatz(midpoint), normal) * length)
        expected = jnp.eye(3)[dof]
        assert np.allclose(np.asarray(fluxes), np.asarray(expected), atol=1e-12)
        assert np.isclose(float(jnp.trace(jax.jacfwd(ansatz)(jnp.asarray([0.2, 0.3])))), 2.0)


def test_rt0_affine_piola_divergence_scales_with_area():
    space = spaces.HDiv(order=0, dim=2, field_dimension=1)
    basis = space.basis("triangle")
    xI = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    coeffs = jnp.asarray([1.0, 0.0, 0.0])
    mapping = lambda xi: spaces.fem_line_tri_tet(xi, xI, xI, {}, False, 2)
    ansatz = lambda xi: basis(xi, xI, coeffs, {}, False, 2, 0, 0)
    div = _rt0_divergence_phys(ansatz, jnp.asarray([0.25, 0.25]), mapping)
    assert np.isclose(float(div), 1.0)


def test_rt0_overwrite_diff_returns_physical_derivatives():
    space = spaces.HDiv(order=0, dim=2, field_dimension=1)
    basis = space.basis("triangle")
    xI = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    coeffs = jnp.asarray([1.0, 0.0, 0.0])
    ansatz = lambda xi: basis(xi, xI, coeffs, {}, True, 2, 0, 0)
    assert np.isclose(float(jnp.trace(jax.jacfwd(ansatz)(jnp.asarray([0.25, 0.25])))), 1.0)


def test_rt0_orientation_on_shared_edge():
    sim = SimState({"sigma": spaces.HDiv(order=0, dim=2, field_dimension=1)})
    sim.add_structured_mesh(
        (1, 1),
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        "tri",
        order=1,
    )
    sim.add_temporal_discretization({"sigma": dae.NoTimeDerivative()})
    sim.add_model(
        "__all__",
        "weak form",
        lambda ctx: 0.0 * jnp.sum(ctx.test_ansatz["sigma"](ctx.x_int)),
        integration_order=1,
    )
    sim.initialize(verbose=0)

    tri = np.asarray(sim.mesh_info.cells["triangle"].nodes_of_cells, dtype=int)[:, :3]
    conn = np.asarray(sim._fe["sigma"].volume["triangle"], dtype=int)
    signs = np.asarray(sim.settings["orientation"][0]["sigma"])
    local_edges = tri[:, np.asarray([[0, 1], [1, 2], [2, 0]], dtype=int)]

    seen = {}
    for cell in range(local_edges.shape[0]):
        for edge in range(3):
            key = tuple(sorted(local_edges[cell, edge]))
            if key in seen:
                other_cell, other_edge = seen[key]
                assert conn[cell, edge] == conn[other_cell, other_edge]
                assert signs[cell, edge] == -signs[other_cell, other_edge]
                return
            seen[key] = (cell, edge)
    raise AssertionError("Structured two-triangle mesh should contain one shared edge.")


def test_rt0_strong_bc_projects_oriented_flux():
    sim = SimState({"sigma": spaces.HDiv(order=0, dim=2, field_dimension=1)})
    sim.add_structured_mesh(
        (1, 1),
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        "tri",
        order=1,
    )
    sim.add_temporal_discretization({"sigma": dae.NoTimeDerivative()})
    sim.add_strong_bc("sigma", lambda x: jnp.isclose(x[0], 1.0), lambda x, t: 2.0)
    sim.add_model(
        "__all__",
        "weak form",
        lambda ctx: 0.0 * jnp.sum(ctx.test_ansatz["sigma"](ctx.x_int)),
        integration_order=1,
    )
    sim.initialize(verbose=0)
    sim.prepare(clean_up=False)

    points = np.asarray(sim.mesh_info.points)
    tri = np.asarray(sim.mesh_info.cells["triangle"].nodes_of_cells, dtype=int)[:, :3]
    conn = np.asarray(sim._fe["sigma"].volume["triangle"], dtype=int)
    mask = np.asarray(sim.settings["dirichlet dofs"]["sigma"], dtype=bool)
    values = np.asarray(sim.settings["dirichlet conditions"]["sigma"])

    for cell, nodes in enumerate(tri):
        for edge, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
            edge_nodes = nodes[[a, b]]
            edge_points = points[edge_nodes]
            if not np.allclose(edge_points[:, 0], 1.0):
                continue
            length = np.linalg.norm(edge_points[1] - edge_points[0])
            sign = 1.0 if edge_nodes[0] < edge_nodes[1] else -1.0
            dof = conn[cell, edge]
            assert mask[dof]
            assert np.isclose(values[dof], sign * 2.0 * length)
            return
    raise AssertionError("No right boundary RT0 edge found.")


def test_rt0_vtu_output_uses_volume_orientation(tmp_path):
    def u_exact(x):
        return 1.0 + jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])

    def sigma_exact(x):
        return -jnp.pi * jnp.array(
            [
                jnp.cos(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]),
                jnp.sin(jnp.pi * x[0]) * jnp.cos(jnp.pi * x[1]),
            ]
        )

    def source(x):
        return 2.0 * jnp.pi**2 * (u_exact(x) - 1.0)

    def neumann_flux(x):
        return jnp.where(
            jnp.isclose(x[0], 1.0),
            jnp.pi * jnp.sin(jnp.pi * x[1]),
            jnp.where(jnp.isclose(x[1], 1.0), jnp.pi * jnp.sin(jnp.pi * x[0]), 0.0),
        )

    def mixed_poisson(ctx):
        t = ctx.t
        x = ctx.trial_ansatz["physical coor"](ctx.x_int)
        sigma = ctx.trial_ansatz["sigma"](ctx.x_int, t)
        u = ctx.trial_ansatz["u"](ctx.x_int, t)
        if ctx.mode == "output":
            return {"sigma": sigma}
        tau = ctx.test_ansatz["sigma"](ctx.x_int)
        v = ctx.test_ansatz["u"](ctx.x_int)
        weak = jnp.dot(sigma, tau)
        weak -= u * jnp.trace(jax.jacfwd(ctx.test_ansatz["sigma"])(ctx.x_int))
        weak += v * (jnp.trace(jax.jacfwd(ctx.trial_ansatz["sigma"])(ctx.x_int, t)) - source(x))
        return weak

    sim = SimState(
        {
            "sigma": spaces.HDiv(order=0, dim=2, field_dimension=1),
            "u": spaces.L2(order=0, dim=2, field_dimension=1),
        }
    )
    sim.add_structured_mesh(
        (4, 4),
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        "tri",
        order=1,
    )
    sim.add_temporal_discretization({"sigma": dae.NoTimeDerivative(), "u": dae.NoTimeDerivative()})
    sim.add_weak_bc(
        "sigma",
        lambda x: jnp.logical_or(jnp.isclose(x[0], 0.0), jnp.isclose(x[1], 0.0)),
        lambda x, t: u_exact(x),
        trace="normal",
        sign=+1.0,
    )
    sim.add_strong_bc(
        "sigma",
        lambda x: jnp.logical_or(jnp.isclose(x[0], 1.0), jnp.isclose(x[1], 1.0)),
        neumann_flux,
    )
    sim.add_model("__all__", "weak form", mixed_poisson, integration_order=4)
    sim.static_settings["solver backend"] = "scipy"
    sim.set_postprocessing_policy(
        dae.SaveAllPolicy(),
        result_folder_name=str(tmp_path / "rt0_output.res"),
    )
    sim.initialize(verbose=0)
    sim.prepare(clean_up=False)
    sim.run(dt0=1.0, time_span=1.0, num_time_steps=1)

    mesh = meshio.read(tmp_path / "rt0_output.res" / "step_000001.vtu")
    points = np.asarray(mesh.points)[:, :2]
    sigma = np.asarray(mesh.point_data["sigma"])[:, :2]
    idx = np.where(np.isclose(points[:, 0], 1.0) & np.isclose(points[:, 1], 0.5))[0][0]
    assert sigma[idx, 0] > 0.0
    assert np.linalg.norm(sigma[idx] - np.asarray(sigma_exact(jnp.asarray(points[idx])))) < 1.0
