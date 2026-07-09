
from pathlib import Path
from types import SimpleNamespace

from jax import config
import jax
import jax.numpy as jnp
import meshio
import numpy as np
import pytest
from flax.core import FrozenDict

from autopdex import SimState, assembler, dae, mesher, spaces
from autopdex.sim_state.mesh_topology import (
    _add_facet_domain,
    _convert_degenerate_linear_quads_to_triangles,
)
from autopdex.sim_state.mesh_info import _build_mesh_info

config.update("jax_enable_x64", True)


def _make_quad_mesh(nx=4, ny=3):
    pts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    x_nodes, elements = mesher.structured_mesh((nx, ny), pts, "quad", order=1)
    return meshio.Mesh(points=np.asarray(x_nodes), cells=[("quad", np.asarray(elements))])


def _make_tri_mesh(nx=4, ny=3, order=1):
    if order == 1:
        pts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        x_nodes, elements = mesher.structured_mesh((nx, ny), pts, "tri", order)
        return meshio.Mesh(points=np.asarray(x_nodes), cells=[("triangle", np.asarray(elements))])

    if order == 2:
        points = np.asarray([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [1.0, 0.5],
            [0.5, 0.5],
            [0.5, 1.0],
            [0.0, 0.5],
        ])
        cells = np.asarray([
            [0, 1, 2, 4, 5, 6],
            [0, 2, 3, 6, 7, 8],
        ])
        return meshio.Mesh(points=points, cells=[("triangle6", cells)])

    raise ValueError("Unsupported test mesh order.")


def _make_hex27_mesh():
    offsets = np.asarray(
        [
            (0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0),
            (0, 0, 2), (2, 0, 2), (2, 2, 2), (0, 2, 2),
            (1, 0, 0), (2, 1, 0), (1, 2, 0), (0, 1, 0),
            (1, 0, 2), (2, 1, 2), (1, 2, 2), (0, 1, 2),
            (0, 0, 1), (2, 0, 1), (2, 2, 1), (0, 2, 1),
            (0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1),
            (1, 1, 0), (1, 1, 2), (1, 1, 1),
        ],
        dtype=float,
    )
    return meshio.Mesh(points=0.5 * offsets, cells=[("hexahedron27", np.arange(27)[None, :])])


def _make_quad_mesh_with_duplicate_interface_nodes():
    points = np.asarray([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [1.0, 1.0],
    ])
    cells = np.asarray([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ])
    point_data = {"marker": np.asarray([10.0, 20.0, 30.0, 40.0, 21.0, 50.0, 60.0, 31.0])}
    point_sets = {
        "shared": np.asarray([1, 4, 2, 7], dtype=int),
        "right": np.asarray([5, 6], dtype=int),
    }
    return meshio.Mesh(
        points=points,
        cells=[("quad", cells)],
        point_data=point_data,
        point_sets=point_sets,
    )


def _make_preexisting_degenerate_quad_mesh():
    return meshio.Mesh(
        points=np.asarray([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]),
        cells=[("quad", np.asarray([[0, 1, 2, 0]], dtype=int))],
    )


def _make_dedup_triggered_degenerate_quad_mesh():
    return meshio.Mesh(
        points=np.asarray([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]),
        cells=[("quad", np.asarray([[0, 1, 2, 3]], dtype=int))],
    )


def _make_split_quad_block_with_metadata():
    return meshio.Mesh(
        points=np.asarray([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ]),
        cells=[("quad", np.asarray([
            [0, 1, 2, 3],
            [1, 4, 5, 1],
        ], dtype=int))],
        cell_data={
            "region": [np.asarray([10, 20], dtype=int)],
            "marker": [np.asarray([100, 200], dtype=int)],
        },
        cell_sets={
            "all_cells": [np.asarray([0, 1], dtype=int)],
            "degenerate_only": [np.asarray([1], dtype=int)],
        },
    )


def _make_nonadjacent_degenerate_quad_mesh():
    return meshio.Mesh(
        points=np.asarray([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]),
        cells=[("quad", np.asarray([[0, 1, 0, 2]], dtype=int))],
    )


def _count_point_components(points, cells):
    parent = list(range(len(points)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for conn in np.asarray(cells, dtype=int):
        for i in range(len(conn)):
            for j in range(i + 1, len(conn)):
                union(int(conn[i]), int(conn[j]))

    return len({find(i) for i in range(len(points))})


def test_build_assembling_template_from_indices_consolidates_duplicates():
    data = jnp.asarray([10.0, 1.0, 5.0, 20.0, 3.0])
    indices = jnp.asarray(
        [
            [2, 2],
            [0, 1],
            [0, 2],
            [2, 2],
            [0, 1],
        ],
        dtype=jnp.int32,
    )

    settings, static_settings = assembler._build_assembling_template_from_indices(
        indices=indices,
        shape=(3, 3),
        settings={},
        static_settings=FrozenDict({}),
    )
    template = settings["assembling template"]
    assembled = static_settings["assembling kernel"](data, template.scatter)

    assert template.nnz == 3
    assert np.array_equal(np.asarray(template.scatter), np.asarray([2, 0, 1, 2, 0], dtype=np.int32))
    assert np.array_equal(np.asarray(template.col_sorted), np.asarray([1, 2, 2], dtype=np.int32))
    assert np.array_equal(np.asarray(template.indptr), np.asarray([0, 2, 2, 3], dtype=np.int32))
    assert np.array_equal(
        np.asarray(template.indices_unique),
        np.asarray([[0, 1], [0, 2], [2, 2]], dtype=np.int32),
    )
    assert np.allclose(np.asarray(assembled), np.asarray([4.0, 5.0, 30.0]))


def test_add_facet_domain_selects_only_boundary_edges_of_quad_mesh():
    mesh = _make_quad_mesh(nx=3, ny=2)
    _add_facet_domain(mesh, "right", point_selector=lambda points: np.isclose(points[:, 0], 1.0))

    facet_block = mesh.cells[-1]
    facet_points = np.asarray(mesh.points)[facet_block.data]

    assert facet_block.type == "line"
    assert facet_block.data.shape == (2, 2)
    assert np.allclose(facet_points[:, :, 0], 1.0, atol=1e-12, rtol=1e-12)
    assert all(block.size == 0 for block in mesh.cell_sets["right"][:-1])
    assert np.array_equal(mesh.cell_sets["right"][-1], np.arange(2))


def test_add_facet_domain_preserves_higher_order_triangle_boundary_edges():
    mesh = _make_tri_mesh(order=2)
    _add_facet_domain(mesh, "right", point_selector=lambda points: np.isclose(points[:, 0], 1.0))

    facet_block = mesh.cells[-1]
    line = facet_block.data[0]
    facet_points = np.asarray(mesh.points)[line]

    assert facet_block.type == "line3"
    assert facet_block.data.shape == (1, 3)
    assert tuple(line[:2]) == (1, 2)
    assert int(line[2]) == 5
    assert np.allclose(facet_points[:, 0], 1.0, atol=1e-12, rtol=1e-12)
    assert np.allclose(np.sort(facet_points[:, 1]), np.asarray([0.0, 0.5, 1.0]), atol=1e-12, rtol=1e-12)


def test_add_facet_domain_selects_quad9_face_of_hexahedron27_mesh():
    mesh = _make_hex27_mesh()
    _add_facet_domain(mesh, "bottom", point_selector=lambda points: np.isclose(points[:, 2], 0.0))

    facet_block = mesh.cells[-1]
    facet = facet_block.data[0]
    facet_points = np.asarray(mesh.points)[facet]

    assert facet_block.type == "quad9"
    assert facet_block.data.shape == (1, 9)
    assert np.array_equal(facet, np.asarray([0, 1, 2, 3, 8, 9, 10, 11, 24]))
    assert np.allclose(facet_points[:, 2], 0.0, atol=1e-12, rtol=1e-12)


def test_build_mesh_info_assigns_line3_entity_ids_from_volume_facets():
    mesh = _make_tri_mesh(order=2)
    _add_facet_domain(mesh, "right", point_selector=lambda points: np.isclose(points[:, 0], 1.0))
    mesh_info = _build_mesh_info(mesh)

    line_ids = np.asarray(mesh_info.domains["right"]["line3"].cell_indices, dtype=int)
    surface_entity_ids = np.asarray(mesh_info.cells["line3"].entity_id, dtype=int)[line_ids]
    surface_edge_nodes = np.sort(np.asarray(mesh_info.cells["line3"].nodes_of_cells, dtype=int)[line_ids, :2], axis=1)
    triangle_facets = np.asarray(mesh_info.cells["triangle6"].facets_of_cells["line3"], dtype=int)
    triangle_corners = np.asarray(mesh_info.cells["triangle6"].nodes_of_cells[:, :3], dtype=int)
    triangle_edge_nodes = np.sort(triangle_corners[:, np.asarray([[0, 1], [1, 2], [2, 0]], dtype=int)], axis=2)

    expected_ids = []
    for edge_nodes in surface_edge_nodes:
        matches = np.all(triangle_edge_nodes == edge_nodes, axis=2)
        cell_idx, local_edge_idx = np.nonzero(matches)
        assert len(cell_idx) == 1
        expected_ids.append(triangle_facets[cell_idx[0], local_edge_idx[0]])

    assert np.all(surface_entity_ids >= 0)
    assert np.array_equal(surface_entity_ids, np.asarray(expected_ids, dtype=int))


def test_sim_state_add_weak_bc_registers_surface_domain_and_model():
    mesh = _make_quad_mesh(nx=3, ny=2)
    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=2)})
    sim.import_mesh(mesh)

    result = sim.add_weak_bc(
        field="u",
        on_boundary_fun=lambda x: jnp.isclose(x[0], 1.0),
        value_fun=lambda x, t: jnp.asarray([t, 0.0]),
    )

    assert result is None
    domain_name = sim._raw_model_specs[0]["domain_name"]
    assert domain_name in sim.mesh_info.domains
    assert "line" in sim.mesh_info.domains[domain_name]
    assert int(np.asarray(sim.mesh_info.domains[domain_name]["line"].cell_indices).size) > 0
    assert len(sim._raw_model_specs) == 1


def _rows_as_set(a):
    return {tuple(np.asarray(r)) for r in np.asarray(a)}


def test_sim_state_import_mesh_deduplicates_exact_points_by_default():
    mesh = _make_quad_mesh_with_duplicate_interface_nodes()
    original_points = np.asarray(mesh.points)
    unique_coords = np.unique(original_points, axis=0)

    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)

    src_pts = np.asarray(sim._source_mesh.points)
    cells = np.asarray(sim.mesh_info.cells["quad"].nodes_of_cells, dtype=int)
    shared_nodes = np.intersect1d(cells[0], cells[1])

    # Deduplication merges coincident points; the numbering order is an
    # implementation detail, so compare by coordinate set.
    assert src_pts.shape[0] == unique_coords.shape[0]
    assert _rows_as_set(src_pts) == _rows_as_set(unique_coords)
    assert _rows_as_set(np.asarray(sim.mesh_info.points)) == _rows_as_set(unique_coords)
    assert shared_nodes.size == 2
    assert _count_point_components(sim._source_mesh.points, cells) == 1
    # point_data is intentionally not retained on the source-mesh snapshot
    # (not used internally; VTU output writes its own point_data).
    assert "marker" not in (sim._source_mesh.point_data or {})
    for set_name in ("shared", "right"):
        got = src_pts[np.asarray(sim._source_mesh.point_sets[set_name], dtype=int)]
        want = original_points[np.asarray(mesh.point_sets[set_name], dtype=int)]
        assert _rows_as_set(got) == _rows_as_set(want)


def test_sim_state_import_mesh_can_disable_exact_point_deduplication():
    mesh = _make_quad_mesh_with_duplicate_interface_nodes()

    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh, deduplicate_points=False)

    cells = np.asarray(sim.mesh_info.cells["quad"].nodes_of_cells, dtype=int)

    assert np.asarray(sim._source_mesh.points).shape[0] == 8
    assert np.intersect1d(cells[0], cells[1]).size == 0
    assert _count_point_components(sim._source_mesh.points, cells) == 2
    # point_data is intentionally not retained on the source-mesh snapshot.
    assert "marker" not in (sim._source_mesh.point_data or {})
    assert np.array_equal(np.asarray(sim._source_mesh.point_sets["shared"]), np.asarray(mesh.point_sets["shared"]))


def test_sim_state_import_mesh_converts_preexisting_degenerate_quad_to_triangle():
    mesh = _make_preexisting_degenerate_quad_mesh()

    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh, convert_degenerate_quads=True)

    assert [block.type for block in sim._source_mesh.cells] == ["triangle"]
    assert np.array_equal(
        np.asarray(sim._source_mesh.cells[0].data, dtype=int),
        np.asarray([[0, 1, 2]], dtype=int),
    )
    assert np.array_equal(
        np.asarray(sim.mesh_info.cells["triangle"].nodes_of_cells, dtype=int),
        np.asarray([[0, 1, 2]], dtype=int),
    )


def test_convert_degenerate_linear_quads_fast_path_keeps_regular_quad_mesh():
    mesh = _make_quad_mesh()

    converted = _convert_degenerate_linear_quads_to_triangles(mesh, enabled=False)

    assert converted is mesh


def test_sim_state_import_mesh_ignores_gmsh_bounding_entities():
    mesh = meshio.Mesh(
        points=np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        cells=[("quad", np.asarray([[0, 1, 2, 3]], dtype=int))],
        cell_sets={"gmsh:bounding_entities": [None]},
    )

    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)

    assert "gmsh:bounding_entities" not in sim._source_mesh.cell_sets


def test_sim_state_import_mesh_converts_quad_degenerated_by_deduplication():
    mesh = _make_dedup_triggered_degenerate_quad_mesh()
    # Expected triangle corner coordinates (in winding order), derived from the
    # quad by collapsing consecutive coincident corners — independent of the
    # point numbering produced by deduplication.
    quad_coords = np.asarray(mesh.points)[np.asarray(mesh.cells[0].data[0], dtype=int)]
    expected_coords = []
    for c in quad_coords:
        if not expected_coords or not np.array_equal(c, expected_coords[-1]):
            expected_coords.append(c)
    if len(expected_coords) > 1 and np.array_equal(expected_coords[0], expected_coords[-1]):
        expected_coords.pop()
    expected_coords = np.asarray(expected_coords)

    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh, convert_degenerate_quads=True)

    assert np.asarray(sim._source_mesh.points).shape[0] == 3
    assert [block.type for block in sim._source_mesh.cells] == ["triangle"]
    tri = np.asarray(sim._source_mesh.cells[0].data, dtype=int)
    tri_coords = np.asarray(sim._source_mesh.points)[tri[0]]
    assert np.array_equal(tri_coords, expected_coords)


def test_sim_state_import_mesh_preserves_cell_data_and_sets_when_splitting_quad_block():
    mesh = _make_split_quad_block_with_metadata()

    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh, convert_degenerate_quads=True)

    assert [block.type for block in sim._source_mesh.cells] == ["quad", "triangle"]
    assert np.array_equal(np.asarray(sim._source_mesh.cell_data["region"][0]), np.asarray([10]))
    assert np.array_equal(np.asarray(sim._source_mesh.cell_data["region"][1]), np.asarray([20]))
    assert np.array_equal(np.asarray(sim._source_mesh.cell_data["marker"][0]), np.asarray([100]))
    assert np.array_equal(np.asarray(sim._source_mesh.cell_data["marker"][1]), np.asarray([200]))
    assert np.array_equal(np.asarray(sim._source_mesh.cell_sets["all_cells"][0]), np.asarray([0]))
    assert np.array_equal(np.asarray(sim._source_mesh.cell_sets["all_cells"][1]), np.asarray([0]))
    assert np.asarray(sim._source_mesh.cell_sets["degenerate_only"][0]).size == 0
    assert np.array_equal(np.asarray(sim._source_mesh.cell_sets["degenerate_only"][1]), np.asarray([0]))
    assert np.array_equal(
        np.asarray(sim.mesh_info.domains["degenerate_only"]["triangle"].cell_indices, dtype=int),
        np.asarray([0], dtype=int),
    )


def test_sim_state_import_mesh_cell_data_keys_make_cell_data_available_per_model_set():
    mesh = _make_split_quad_block_with_metadata()

    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh, convert_degenerate_quads=True, cell_data_keys=("region", "marker"))

    def weak_form(ctx):
        region = ctx.settings["cell data"][ctx.set]["region"][ctx.elem_number]
        marker = ctx.settings["cell data"][ctx.set]["marker"][ctx.elem_number]
        return (region + marker) * ctx.test_ansatz["u"](ctx.x_int)

    sim.add_model("__all__", "weak form", weak_form, integration_order=2)
    sim._fe = {
        "u": SimpleNamespace(
            g2a={
                "cell_g2a": {
                    cell_type: np.arange(int(cinfo.nodes_of_cells.shape[0]), dtype=int)
                    for cell_type, cinfo in sim.mesh_info.cells.items()
                }
            },
            volume={
                cell_type: cinfo.nodes_of_cells.astype(int)
                for cell_type, cinfo in sim.mesh_info.cells.items()
            },
            surface={},
        )
    }
    sim._ensure_fe_built = lambda: None
    sim._materialize_model_specs()

    cell_data = sim.settings["cell data"]
    assert len(cell_data) == 2
    assert np.array_equal(np.asarray(cell_data[0]["region"]), np.asarray([10]))
    assert np.array_equal(np.asarray(cell_data[1]["region"]), np.asarray([20]))
    assert np.array_equal(np.asarray(cell_data[0]["marker"]), np.asarray([100]))
    assert np.array_equal(np.asarray(cell_data[1]["marker"]), np.asarray([200]))


def test_sim_state_import_mesh_can_disable_degenerate_quad_conversion():
    mesh = _make_preexisting_degenerate_quad_mesh()

    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})

    with pytest.raises(ValueError, match="requires convert_degenerate_quads=True"):
        sim.import_mesh(mesh, convert_degenerate_quads=False)


def test_sim_state_import_mesh_rejects_nonadjacent_degenerate_quad_pattern():
    mesh = _make_nonadjacent_degenerate_quad_mesh()

    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})

    with pytest.raises(ValueError, match="non-adjacent repeated corner pattern"):
        sim.import_mesh(mesh, convert_degenerate_quads=True)


def test_sim_state_model_output_exports_vector_and_tensor_point_data(tmp_path):
    mesh = _make_quad_mesh(nx=2, ny=1)
    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=2)})
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({"u": dae.NoTimeDerivative()})

    def model(ctx):
        if ctx.mode == "output":
            x = ctx.trial_ansatz["physical coor"](ctx.x_int)
            return {
                "coord_vec": x,
                "coord_tensor": jnp.asarray([
                    [x[0], x[1]],
                    [x[1], x[0]],
                ]),
            }
        return 0.0 * jnp.sum(ctx.test_ansatz["u"](ctx.x_int))

    sim.add_model("__all__", "weak form", model, integration_order=2)
    sim.set_postprocessing_policy(
        domains="__all__",
    )
    sim.initialize(verbose=0)
    sim.dofs["u"] = sim.mesh_info.points

    outpath = tmp_path / "formatted_outputs.vtu"
    sim._write_vtu(path=str(outpath))

    exported = meshio.read(outpath)
    points = np.asarray(sim.mesh_info.points)
    coord_vec = np.asarray(exported.point_data["coord_vec"])
    coord_tensor = np.asarray(exported.point_data["coord_tensor"])

    assert "u" not in exported.point_data
    assert coord_vec.shape == (points.shape[0], 3)
    assert np.allclose(coord_vec[:, :2], points, atol=1e-8, rtol=1e-8)
    assert np.allclose(coord_vec[:, 2], 0.0, atol=1e-12, rtol=1e-12)

    expected_tensor = np.zeros((points.shape[0], 3, 3))
    expected_tensor[:, 0, 0] = points[:, 0]
    expected_tensor[:, 0, 1] = points[:, 1]
    expected_tensor[:, 1, 0] = points[:, 1]
    expected_tensor[:, 1, 1] = points[:, 0]
    assert coord_tensor.shape == (points.shape[0], 9)
    assert np.allclose(coord_tensor, expected_tensor.reshape((points.shape[0], 9)), atol=1e-8, rtol=1e-8)


def test_sim_state_model_output_preserves_output_key_order():
    mesh = _make_quad_mesh(nx=1, ny=1)
    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({"u": dae.NoTimeDerivative()})

    def model(ctx):
        if ctx.mode == "output":
            x = ctx.trial_ansatz["physical coor"](ctx.x_int)
            return {
                "z_out": x[0],
                "a_out": x,
                "m_out": jnp.asarray([[x[0]]]),
            }
        return 0.0 * ctx.test_ansatz["u"](ctx.x_int)

    sim.add_model("__all__", "weak form", model, integration_order=2)
    sim.set_postprocessing_policy(
        domains="__all__",
    )

    sim.initialize(verbose=0)

    assert tuple(sim._prepared_output_specs[0]["output_keys"]) == ("z_out", "a_out", "m_out")


def test_sim_state_model_output_fails_for_non_traceable_shape_inference():
    mesh = _make_quad_mesh(nx=1, ny=1)
    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({"u": dae.NoTimeDerivative()})

    def model(ctx):
        if ctx.mode == "output":
            return {
                "coord_vec": np.asarray(ctx.trial_ansatz["physical coor"](ctx.x_int))
            }
        return 0.0 * ctx.test_ansatz["u"](ctx.x_int)

    sim.add_model("__all__", "weak form", model, integration_order=2)
    sim.set_postprocessing_policy(
        domains="__all__",
    )

    with pytest.raises(jax.errors.TracerArrayConversionError):
        sim.initialize(verbose=0)


def test_sim_state_add_strong_bc_projects_boundary_trace_dofs():
    mesh = _make_tri_mesh(order=2)
    sim = SimState({"u": spaces.H1(order=2, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({"u": dae.NoTimeDerivative()})

    def weak_form(ctx):
        t = ctx.settings.get("current time", 0.0)
        return ctx.trial_ansatz["u"](ctx.x_int, t) * ctx.test_ansatz["u"](ctx.x_int)

    sim.add_model("__all__", "weak form", weak_form, integration_order=2)
    sim.static_settings["solver backend"] = "dense"
    sim.initialize(verbose=0)
    sim.add_strong_bc(
        field="u",
        on_boundary_fun=lambda x: jnp.isclose(x[0], 0.0),
        value_fun=lambda x, t: x[1],
    )

    sim.prepare(clean_up=False)
    projected_settings = sim.settings

    assert len(sim._l2_projectors) == 1
    projector = next(iter(sim._l2_projectors.values()))
    constrained = projected_settings["dirichlet dofs"]["u"]
    constrained_values = projected_settings["dirichlet conditions"]["u"]
    unconstrained_ids = jnp.setdiff1d(jnp.arange(constrained.shape[0]), projector.global_scalar_ids)

    assert bool(jnp.all(constrained[projector.global_scalar_ids]))
    assert bool(jnp.all(~constrained[unconstrained_ids]))
    assert jnp.allclose(
        constrained_values[projector.global_scalar_ids],
        sim.mesh_info.points[projector.global_scalar_ids, 1],
        atol=1e-8,
        rtol=1e-8,
    )


def test_sim_state_initialize_materializes_projected_dirichlet_specs_registered_before_initialize():
    mesh = _make_tri_mesh(order=2)
    sim = SimState({"u": spaces.H1(order=2, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({"u": dae.NoTimeDerivative()})

    def weak_form(ctx):
        t = ctx.settings.get("current time", 0.0)
        return ctx.trial_ansatz["u"](ctx.x_int, t) * ctx.test_ansatz["u"](ctx.x_int)

    sim.add_model("__all__", "weak form", weak_form, integration_order=2)
    sim.static_settings["solver backend"] = "dense"
    sim.add_strong_bc(
        field="u",
        on_boundary_fun=lambda x: jnp.isclose(x[0], 0.0),
        value_fun=lambda x, t: x[1],
    )

    sim.initialize(verbose=0)

    assert len(sim._projected_dirichlet_specs) == 1
    assert len(sim._l2_projectors) == 1

    sim.prepare(clean_up=False)
    projected_settings = sim.settings
    projector = next(iter(sim._l2_projectors.values()))
    constrained = projected_settings["dirichlet dofs"]["u"]
    constrained_values = projected_settings["dirichlet conditions"]["u"]

    assert bool(jnp.all(constrained[projector.global_scalar_ids]))
    assert jnp.allclose(
        constrained_values[projector.global_scalar_ids],
        sim.mesh_info.points[projector.global_scalar_ids, 1],
        atol=1e-8,
        rtol=1e-8,
    )


def test_sim_state_add_structured_mesh_imports_meshio_mesh():
    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.add_structured_mesh(
        n_elements=(2, 1),
        vertices=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        element_type="quad",
        order=1,
    )

    assert sim.mesh_info is not None
    assert sim._source_mesh is not None
    assert "quad" in sim._source_mesh.cells_dict
    assert sim.mesh_info.points.shape == (6, 2)
    assert sim.mesh_info.cells["quad"].nodes_of_cells.shape == (2, 4)
    assert "__all__" in sim.mesh_info.domains


def test_sim_state_prepare_cleanup_releases_initialization_only_state_and_run_still_works():
    mesh = _make_quad_mesh(nx=1, ny=1)
    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({"u": dae.NoTimeDerivative()})

    def weak_form(ctx):
        t = ctx.settings.get("current time", 0.0)
        return (ctx.trial_ansatz["u"](ctx.x_int, t) - t) * ctx.test_ansatz["u"](ctx.x_int)

    sim.add_model("__all__", "weak form", weak_form, integration_order=2)
    sim.static_settings["solver backend"] = "dense"
    sim.initialize(verbose=-1)
    sim.prepare()

    assert sim._fe is None
    assert sim._fe_signature is None
    assert sim.mesh_info is None
    assert sim._source_mesh is None
    assert sim.dimension is None
    assert sim._l2_projectors == {}
    assert sim._geometry_projection_core_cache == {}
    assert sim._projection_source_cache == {}
    assert sim._output_projection_last_local == {}
    assert sim._output_projection_model_cache == {}
    assert sim._prepared_output_specs == tuple()
    assert "node coordinates" in sim.settings
    assert "connectivity" in sim.settings

    result = sim.run(dt0=1.0, time_span=1.0, num_time_steps=1)
    assert int(result.num_accepted) == 1
    assert jnp.allclose(result.q["u"], 1.0, atol=1e-8)


def test_sim_state_prepare_cleanup_keeps_projected_dirichlet_runtime_data():
    mesh = _make_tri_mesh(order=2)
    sim = SimState({"u": spaces.H1(order=2, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({"u": dae.NoTimeDerivative()})

    def weak_form(ctx):
        t = ctx.settings.get("current time", 0.0)
        return ctx.trial_ansatz["u"](ctx.x_int, t) * ctx.test_ansatz["u"](ctx.x_int)

    sim.add_model("__all__", "weak form", weak_form, integration_order=2)
    sim.add_strong_bc(
        field="u",
        on_boundary_fun=lambda x: jnp.isclose(x[0], 0.0),
        value_fun=lambda x, t: x[1],
    )
    sim.static_settings["solver backend"] = "dense"
    sim.initialize(verbose=-1)
    sim.prepare()

    assert sim.mesh_info is None
    assert len(sim._l2_projectors) == 1
    projector = next(iter(sim._l2_projectors.values()))
    apply_bcs = sim.manager.static_settings["apply dirichlet bcs"]
    projected_settings = apply_bcs(0.0, dict(sim.settings))

    constrained_values = projected_settings["dirichlet conditions"]["u"]
    coords = projector.settings_template["node coordinates"]
    assert jnp.allclose(
        constrained_values[projector.global_scalar_ids],
        coords[projector.global_scalar_ids, 1],
        atol=1e-8,
        rtol=1e-8,
    )


def test_sim_state_postprocessing_policy_writes_vtu_series(tmp_path):
    mesh = _make_quad_mesh(nx=2, ny=1)
    sim = SimState({"displacement": spaces.H1(order=1, dim=2, field_dimension=2)})
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({"displacement": dae.NoTimeDerivative()})

    lam = 576.9230769230769
    mu = 384.6153846153846

    def strain_trial(fun, x, t):
        grad_u = jax.jacfwd(lambda xi: fun(xi, t))(x)
        return 0.5 * (grad_u + grad_u.T)

    def strain_test(fun, x):
        grad_v = jax.jacfwd(fun)(x)
        return 0.5 * (grad_v + grad_v.T)

    def model(ctx):
        if ctx.mode == "output":
            t = ctx.settings.get("current time", 0.0)
            eps = strain_trial(ctx.trial_ansatz["displacement"], ctx.x_int, t)
            return {"strain_yy": eps[1, 1]}

        t = ctx.settings.get("current time", 0.0)
        eps_u = strain_trial(ctx.trial_ansatz["displacement"], ctx.x_int, t)
        eps_v = strain_test(ctx.test_ansatz["displacement"], ctx.x_int)
        sigma = lam * jnp.trace(eps_u) * jnp.eye(2) + 2.0 * mu * eps_u
        return jnp.sum(sigma * eps_v)

    sim.add_model("__all__", "weak form", model, integration_order=2)
    sim.add_strong_bc(
        field="displacement",
        on_boundary_fun=lambda x: jnp.isclose(x[0], 0.0),
        value_fun=lambda x, t: jnp.zeros((2,)),
    )
    sim.add_weak_bc(
        field="displacement",
        on_boundary_fun=lambda x: jnp.isclose(x[0], 1.0),
        value_fun=lambda x, t: jnp.asarray([0.05 * t, 0.0]),
    )

    output_dir = tmp_path / "simdemo.res"
    sim.set_postprocessing_policy(
        dae.SaveEquidistantPolicy(num_points=1),
        result_folder_name=str(output_dir),
        domains="__all__",
        integration_order=2,
    )

    sim.initialize(verbose=0)
    sim.prepare()
    assert sim._source_mesh is None
    assert "_vtu_output_runtime" in sim.settings
    sim.run(dt0=1.0, time_span=1.0, num_time_steps=1)

    vtu_files = sorted(output_dir.glob("step_*.vtu"))
    pvd_file = output_dir / "simdemo.pvd"

    assert len(vtu_files) == 2
    assert pvd_file.exists()
    pvd_text = pvd_file.read_text(encoding="utf-8")
    assert "step_000000.vtu" in pvd_text
    assert "step_000001.vtu" in pvd_text

    exported = meshio.read(vtu_files[-1])
    assert "displacement" not in exported.point_data
    assert "strain_yy" in exported.point_data


def test_sim_state_save_all_postprocessing_writes_initial_and_accepted_steps(tmp_path):
    mesh = _make_quad_mesh(nx=1, ny=1)
    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({"u": dae.NoTimeDerivative()})

    def model(ctx):
        if ctx.mode == "output":
            return {"x": ctx.trial_ansatz["physical coor"](ctx.x_int)[0]}
        x = ctx.trial_ansatz["physical coor"](ctx.x_int)
        value = ctx.trial_ansatz["u"](ctx.x_int, ctx.settings.get("current time", 0.0))
        return (value - x[0]) * ctx.test_ansatz["u"](ctx.x_int)

    sim.add_model("__all__", "weak form", model, integration_order=2)
    output_dir = tmp_path / "saveall.res"
    sim.set_postprocessing_policy(
        dae.SaveAllPolicy(),
        result_folder_name=str(output_dir),
        domains="__all__",
    )

    sim.initialize(verbose=0)
    sim.prepare()
    sim.run(dt0=0.5, time_span=1.0, num_time_steps=2)

    vtu_files = sorted(output_dir.glob("step_*.vtu"))
    assert [path.name for path in vtu_files] == [
        "step_000000.vtu",
        "step_000001.vtu",
        "step_000002.vtu",
    ]
    exported = meshio.read(vtu_files[-1])
    assert "u" not in exported.point_data
    assert "x" in exported.point_data


def test_sim_state_vtu_output_uses_time_derivatives_from_q_fun_state(tmp_path):
    mesh = _make_quad_mesh(nx=1, ny=1)
    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({"u": dae.BackwardEuler()})

    def model(ctx):
        t = ctx.settings.get("current time", 0.0)
        if ctx.mode == "output":
            u_dot = jax.jacfwd(ctx.trial_ansatz["u"], 1)(ctx.x_int, t)
            return {"u_dot": u_dot}
        u = ctx.trial_ansatz["u"](ctx.x_int, t)
        return (u - t) * ctx.test_ansatz["u"](ctx.x_int)

    sim.add_model("__all__", "weak form", model, integration_order=2)
    output_dir = tmp_path / "time_derivative.res"
    sim.set_postprocessing_policy(
        dae.SaveEquidistantPolicy(num_points=1),
        result_folder_name=str(output_dir),
        domains="__all__",
    )

    sim.initialize(verbose=-1)
    sim.prepare()
    sim.run(dt0=1.0, time_span=1.0, num_time_steps=1)

    exported = meshio.read(sorted(output_dir.glob("step_*.vtu"))[-1])
    assert np.allclose(exported.point_data["u_dot"], 1.0, atol=1e-10, rtol=1e-10)


def test_sim_state_postprocessing_policy_can_run_twice_after_prepare(tmp_path):
    mesh = _make_quad_mesh(nx=1, ny=1)
    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({"u": dae.NoTimeDerivative()})

    def model(ctx):
        if ctx.mode == "output":
            return {"time": ctx.settings.get("current time", 0.0)}
        value = ctx.trial_ansatz["u"](ctx.x_int, ctx.settings.get("current time", 0.0))
        return value * ctx.test_ansatz["u"](ctx.x_int)

    sim.add_model("__all__", "weak form", model, integration_order=2)
    output_dir = tmp_path / "twice.res"
    sim.set_postprocessing_policy(
        dae.SaveEquidistantPolicy(num_points=1),
        result_folder_name=str(output_dir),
        domains="__all__",
    )

    sim.initialize(verbose=0)
    sim.prepare()
    first = sim.run(dt0=1.0, time_span=1.0, num_time_steps=1)
    second = first.run(dt0=1.0, time_span=1.0, num_time_steps=1)

    assert int(first.num_accepted) == 1
    assert int(second.num_accepted) == 1
    vtu_files = sorted(output_dir.glob("step_*.vtu"))
    assert [path.name for path in vtu_files] == [
        "step_000000.vtu",
        "step_000001.vtu",
        "step_000002.vtu",
        "step_000003.vtu",
    ]
    pvd_text = (output_dir / "twice.pvd").read_text(encoding="utf-8")
    assert pvd_text.count("step_000000.vtu") == 1
    assert pvd_text.count("step_000001.vtu") == 1
    assert pvd_text.count("step_000002.vtu") == 1
    assert pvd_text.count("step_000003.vtu") == 1


def test_sim_state_postprocessing_policy_can_change_schedule_after_prepare(tmp_path):
    mesh = _make_quad_mesh(nx=1, ny=1)
    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({"u": dae.NoTimeDerivative()})

    def model(ctx):
        if ctx.mode == "output":
            return {"time": ctx.settings.get("current time", 0.0)}
        value = ctx.trial_ansatz["u"](ctx.x_int, ctx.settings.get("current time", 0.0))
        return value * ctx.test_ansatz["u"](ctx.x_int)

    sim.add_model("__all__", "weak form", model, integration_order=2)
    output_dir = tmp_path / "policy_switch.res"
    sim.set_postprocessing_policy(
        dae.SaveEquidistantPolicy(num_points=1),
        result_folder_name=str(output_dir),
        domains="__all__",
    )

    sim.initialize(verbose=0)
    sim.prepare()
    sim = sim.run(dt0=1.0, time_span=1.0, num_time_steps=1)
    sim.set_postprocessing_policy(dae.SaveAllPolicy())
    sim = sim.run(dt0=0.5, time_span=1.0, num_time_steps=2)

    vtu_files = sorted(output_dir.glob("step_*.vtu"))
    assert [path.name for path in vtu_files] == [
        "step_000000.vtu",
        "step_000001.vtu",
        "step_000002.vtu",
        "step_000003.vtu",
        "step_000004.vtu",
    ]


def test_sim_state_internal_variables_are_not_global_dofs_and_reject_bcs():
    mesh = _make_quad_mesh(nx=1, ny=1)
    sim = SimState({
        "alpha": spaces.InternalVariable(field_dimension=1),
        "u": spaces.H1(order=1, dim=2, field_dimension=1),
    })
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({
        "alpha": dae.BackwardEuler(),
        "u": dae.NoTimeDerivative(),
    })

    with pytest.raises(ValueError, match="internal variables"):
        sim.add_strong_bc(
            field="alpha",
            on_boundary_fun=lambda x: jnp.isclose(x[0], 0.0),
            value_fun=lambda x, t: 0.0,
        )

    with pytest.raises(ValueError, match="internal variables"):
        sim.add_weak_bc(
            field="alpha",
            on_boundary_fun=lambda x: jnp.isclose(x[0], 1.0),
            value_fun=lambda x, t: 1.0,
        )

    sim.initialize(verbose=0)

    assert tuple(sim.dofs.keys()) == ("u",)
    assert tuple(sim.manager.integrators.keys()) == ("u",)
    assert "alpha" in sim.manager.static_settings["internal variable time integrators"]


def test_sim_state_internal_variable_update_commits_per_gauss_point_state():
    mesh = _make_quad_mesh(nx=1, ny=1)
    sim = SimState({
        "alpha": spaces.InternalVariable(field_dimension=1),
        "u": spaces.H1(order=1, dim=2, field_dimension=1),
    })
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({
        "alpha": dae.BackwardEuler(),
        "u": dae.NoTimeDerivative(),
    })

    def weak_form(ctx):
        t = ctx.settings.get("current time", 0.0)
        value = ctx.trial_ansatz["u"](ctx.x_int, t) - ctx.internal_vars["alpha"]
        return value * ctx.test_ansatz["u"](ctx.x_int)

    def update_alpha(ctx):
        x = ctx.trial_ansatz["physical coor"](ctx.x_int)
        return {"alpha": ctx.internal_vars["alpha"] + 1.0 + x[0]}

    sim.add_model(
        "__all__",
        "weak form",
        weak_form,
        integration_order=2,
        internal_variables=("alpha",),
        internal_variable_update_fun=update_alpha,
    )
    sim.initialize(verbose=0)
    sim.prepare()
    result = sim.run(dt0=1.0, time_span=1.0, num_time_steps=1)

    alpha = result.settings["internal variables"][0]["alpha"]
    alpha_n = result.settings["internal variables n"][0]["alpha"]

    assert alpha.shape == (1, 4)
    assert jnp.all(alpha > 1.0)
    assert jnp.all(alpha < 2.0)
    assert jnp.allclose(alpha, alpha_n)


def test_sim_state_context_model_combines_weak_form_internal_updates_and_output(tmp_path):
    mesh = _make_quad_mesh(nx=1, ny=1)
    sim = SimState({
        "alpha": spaces.InternalVariable(field_dimension=1),
        "u": spaces.H1(order=1, dim=2, field_dimension=1),
    })
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({
        "alpha": dae.BackwardEuler(),
        "u": dae.NoTimeDerivative(),
    })

    def model(ctx):
        if ctx.mode == "weak form":
            t = ctx.settings.get("current time", 0.0)
            u = ctx.trial_ansatz["u"](ctx.x_int, t)
            v = ctx.test_ansatz["u"](ctx.x_int)
            return (u - ctx.internal_vars["alpha"]) * v
        if ctx.mode == "internal variables":
            x = ctx.trial_ansatz["physical coor"](ctx.x_int)
            return {"alpha": ctx.internal_vars["alpha"] + 1.0 + x[0]}
        if ctx.mode == "output":
            return {"alpha_out": ctx.internal_vars["alpha"]}
        raise ValueError(f"Unexpected model mode {ctx.mode}.")

    sim.add_model("__all__", "weak form", model, integration_order=2, internal_variables=("alpha",))
    sim.set_postprocessing_policy(
        domains="__all__",
        integration_order=2,
    )
    sim.initialize(verbose=0)
    sim.prepare(clean_up=False)
    result = sim.run(dt0=1.0, time_span=1.0, num_time_steps=1)

    alpha = result.settings["internal variables"][0]["alpha"]
    assert alpha.shape == (1, 4)
    assert jnp.all(alpha > 1.0)
    assert jnp.all(alpha < 2.0)

    outpath = tmp_path / "context_outputs.vtu"
    sim._write_vtu(path=str(outpath), dofs=result.q, settings=result.settings)
    exported = meshio.read(outpath)
    alpha_out = np.asarray(exported.point_data["alpha_out"])
    assert np.all(np.isfinite(alpha_out))
    assert np.isclose(alpha_out.mean(), np.asarray(alpha).mean(), atol=1e-8, rtol=1e-8)


def test_sim_state_local_subsystem_uses_global_time_interval():
    mesh = _make_quad_mesh(nx=1, ny=1)
    sim = SimState({
        "alpha": spaces.InternalVariable(field_dimension=1),
        "u": spaces.H1(order=1, dim=2, field_dimension=1),
    })
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({
        "alpha": dae.BackwardEuler(),
        "u": dae.NoTimeDerivative(),
    })

    def local_dae(q_fun, t, settings):
        _, q_t = jax.jvp(q_fun, (t,), (1.0,))
        return jnp.atleast_1d(q_t["alpha"] - settings["rate"] * t)

    def model(ctx):
        if ctx.mode == "internal variables":
            return ctx.solve_local("alpha_time", inputs={"rate": 2.0})
        t = ctx.settings.get("current time", 0.0)
        u = ctx.trial_ansatz["u"](ctx.x_int, t)
        v = ctx.test_ansatz["u"](ctx.x_int)
        return u * v

    sim.add_local_subsystem("alpha_time", local_dae, fields=("alpha",))
    sim.add_model("__all__", "weak form", model, integration_order=2, internal_variables=("alpha",))
    sim.set_initial_conditions(time=2.0)
    sim.initialize(verbose=-1)
    local_solver = sim.static_settings["local subsystem solvers"]["alpha_time"]
    assert local_solver.manager.static_settings["verbose"] == -1
    sim.prepare()
    result = sim.run(dt0=0.5, time_span=0.5, num_time_steps=1)

    alpha = result.settings["internal variables"][0]["alpha"]
    assert jnp.allclose(alpha, 2.5)


def test_sim_state_local_subsystem_selects_registered_key():
    mesh = _make_quad_mesh(nx=1, ny=1)
    sim = SimState({
        "alpha": spaces.InternalVariable(field_dimension=1),
        "beta": spaces.InternalVariable(field_dimension=1),
        "u": spaces.H1(order=1, dim=2, field_dimension=1),
    })
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({
        "alpha": dae.BackwardEuler(),
        "beta": dae.BackwardEuler(),
        "u": dae.NoTimeDerivative(),
    })

    def alpha_dae(q_fun, t, settings):
        _, q_t = jax.jvp(q_fun, (t,), (1.0,))
        return jnp.atleast_1d(q_t["alpha"] - 1.0)

    def beta_dae(q_fun, t, settings):
        _, q_t = jax.jvp(q_fun, (t,), (1.0,))
        return jnp.atleast_1d(q_t["beta"] - 3.0)

    def model(ctx):
        if ctx.mode == "internal variables":
            beta_np1 = ctx.solve_local("beta_update")
            return {"alpha": ctx.internal_vars["alpha"], "beta": beta_np1["beta"]}
        t = ctx.settings.get("current time", 0.0)
        u = ctx.trial_ansatz["u"](ctx.x_int, t)
        v = ctx.test_ansatz["u"](ctx.x_int)
        return u * v

    sim.add_local_subsystem("alpha_update", alpha_dae, fields=("alpha",))
    sim.add_local_subsystem("beta_update", beta_dae, fields=("beta",))
    sim.add_model("__all__", "weak form", model, integration_order=2, internal_variables=("alpha", "beta"))
    sim.initialize(verbose=-1)
    sim.prepare()
    result = sim.run(dt0=1.0, time_span=1.0, num_time_steps=1)

    block = result.settings["internal variables"][0]
    assert jnp.allclose(block["alpha"], 0.0)
    assert jnp.allclose(block["beta"], 3.0)


def test_sim_state_local_subsystem_validation():
    mesh = _make_quad_mesh(nx=1, ny=1)

    def local_dae(q_fun, t, settings):
        q, _ = jax.jvp(q_fun, (t,), (1.0,))
        return jnp.atleast_1d(q["alpha"])

    sim = SimState({
        "alpha": spaces.InternalVariable(field_dimension=1),
        "u": spaces.H1(order=1, dim=2, field_dimension=1),
    })
    sim.import_mesh(mesh)

    with pytest.raises(KeyError, match="Unknown local subsystem field"):
        sim.add_local_subsystem("missing", local_dae, fields=("missing",))

    with pytest.raises(ValueError, match="internal variables"):
        sim.add_local_subsystem("primary", local_dae, fields=("u",))

    ctx = SimState.ModelContext(
        mode="internal variables",
        x_int=None,
        trial_ansatz={},
        internal_vars={},
        local_solvers={},
        settings={},
    )
    with pytest.raises(KeyError, match="Unknown local subsystem"):
        ctx.solve_local("missing")

    ctx_time = SimState.ModelContext(
        mode="weak form",
        x_int=None,
        trial_ansatz={},
        settings={"current time": 2.5, "last time": 1.0},
    )
    assert jnp.allclose(ctx_time.t, 2.5)
    assert jnp.allclose(ctx_time.dt, 1.5)

    sim_missing_integrator = SimState({
        "alpha": spaces.InternalVariable(field_dimension=1),
        "u": spaces.H1(order=1, dim=2, field_dimension=1),
    })
    sim_missing_integrator.import_mesh(mesh)
    sim_missing_integrator.add_temporal_discretization({"u": dae.NoTimeDerivative()})
    sim_missing_integrator.add_local_subsystem("alpha_update", local_dae, fields=("alpha",))
    with pytest.raises(ValueError, match="add_temporal_discretization"):
        sim_missing_integrator.initialize(verbose=-1)


def test_sim_state_context_output_requires_dict():
    mesh = _make_quad_mesh(nx=1, ny=1)
    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)

    def model(ctx):
        if ctx.mode == "output":
            return ctx.trial_ansatz["physical coor"](ctx.x_int)[0]
        return 0.0 * ctx.test_ansatz["u"](ctx.x_int)

    sim.add_model("__all__", "weak form", model, integration_order=2)
    sim.set_postprocessing_policy(domains="__all__")
    with pytest.raises(ValueError, match="output"):
        sim.initialize(verbose=0)


def test_sim_state_rejects_non_context_model_functions_and_has_no_add_output():
    mesh = _make_quad_mesh(nx=1, ny=1)
    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)

    def old_weak_form(x_int, trial_ansatz, test_ansatz, settings, static_settings, elem_number, set):
        return test_ansatz["u"](x_int)

    with pytest.raises(ValueError, match="ModelContext"):
        sim.add_model("__all__", "weak form", old_weak_form)

    assert not hasattr(sim, "add_output")


def test_sim_state_initial_conditions_accept_only_field_shaped_values():
    mesh = _make_quad_mesh(nx=1, ny=1)
    sim = SimState({
        "u": spaces.H1(order=1, dim=2, field_dimension=1),
        "v": spaces.H1(order=1, dim=2, field_dimension=2),
    })
    sim.import_mesh(mesh)

    sim.set_initial_conditions(values={
        "u": 2.0,
        "v": jnp.asarray([1.0, 3.0]),
    })

    assert jnp.allclose(sim.dofs["u"], 2.0)
    assert jnp.allclose(sim.dofs["v"], jnp.asarray([1.0, 3.0]))

    with pytest.raises(ValueError, match="field shape"):
        sim.set_initial_conditions(values={"v": 1.0})

    with pytest.raises(ValueError, match="field shape"):
        sim.set_initial_conditions(values={"u": jnp.zeros_like(sim.dofs["u"])})


def test_sim_state_initial_conditions_apply_domain_overwrites():
    mesh = _make_quad_mesh(nx=2, ny=1)
    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)
    sim._add_domain("right", lambda x: x[0] >= 0.5)

    sim.set_initial_conditions(values={
        "__all__": {"u": 0.0},
        "right": {"u": 3.0},
    })

    x = sim.mesh_info.points[:, 0]
    assert jnp.allclose(sim.dofs["u"][x < 0.5], 0.0)
    assert jnp.allclose(sim.dofs["u"][x >= 0.5], 3.0)


def test_sim_state_initial_conditions_collocation_uses_lobatto_nodes():
    mesh = _make_quad_mesh(nx=1, ny=1)
    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)

    sim.set_initial_conditions(
        values={"u": lambda x: jnp.where(x[0] > 0.5, 1.0, -1.0)},
        integration_mode="collocation",
    )

    expected = jnp.where(sim.mesh_info.points[:, 0] > 0.5, 1.0, -1.0)
    assert jnp.allclose(sim.dofs["u"], expected)


def test_sim_state_dof_scaling_stores_initial_conditions_internal():
    mesh = _make_quad_mesh(nx=1, ny=1)
    sim = SimState({
        "u": spaces.H1(order=1, dim=2, field_dimension=1),
        "v": spaces.H1(order=1, dim=2, field_dimension=2),
    })
    sim.set_dof_scaling({"u": 2.0, "v": jnp.asarray([10.0, 100.0])})
    sim.import_mesh(mesh)

    sim.set_initial_conditions(values={
        "u": 4.0,
        "v": jnp.asarray([10.0, 200.0]),
    })

    assert jnp.allclose(sim.dofs["u"], 2.0)
    assert jnp.allclose(sim.dofs["v"], jnp.asarray([1.0, 2.0]))


def test_sim_state_dof_scaling_scales_projected_dirichlet_values():
    mesh = _make_tri_mesh(order=2)
    sim = SimState({"u": spaces.H1(order=2, dim=2, field_dimension=1)})
    sim.set_dof_scaling({"u": 5.0})
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({"u": dae.NoTimeDerivative()})

    def weak_form(ctx):
        t = ctx.settings.get("current time", 0.0)
        return ctx.trial_ansatz["u"](ctx.x_int, t) * ctx.test_ansatz["u"](ctx.x_int)

    sim.add_model("__all__", "weak form", weak_form, integration_order=2)
    sim.static_settings["solver backend"] = "dense"
    sim.add_strong_bc(
        field="u",
        on_boundary_fun=lambda x: jnp.isclose(x[0], 0.0),
        value_fun=lambda x, t: x[1] + 1.0,
    )

    sim.initialize(verbose=0)
    sim.prepare(clean_up=False)

    projector = next(iter(sim._l2_projectors.values()))
    ids = projector.global_scalar_ids
    constrained = sim.settings["dirichlet dofs"]["u"]
    constrained_values = sim.settings["dirichlet conditions"]["u"]

    assert bool(jnp.all(constrained[ids]))
    assert jnp.allclose(
        constrained_values[ids],
        (sim.mesh_info.points[ids, 1] + 1.0) / 5.0,
        atol=1e-8,
        rtol=1e-8,
    )


def test_sim_state_dof_scaling_scales_residual_and_tangent():
    mesh = _make_quad_mesh(nx=1, ny=1)

    def make_sim(scale=None):
        sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
        if scale is not None:
            sim.set_dof_scaling({"u": scale})
        sim.import_mesh(mesh)
        sim.add_temporal_discretization({"u": dae.NoTimeDerivative()})

        def weak_form(ctx):
            t = ctx.settings.get("current time", 0.0)
            return ctx.trial_ansatz["u"](ctx.x_int, t) * ctx.test_ansatz["u"](ctx.x_int)

        sim.add_model("__all__", "weak form", weak_form, integration_order=2)
        sim.set_initial_conditions(values={"u": 2.0})
        sim.initialize(verbose=0)
        return sim

    scale = 3.0
    unscaled = make_sim()
    scaled = make_sim(scale)

    r_unscaled = unscaled.manager._global_residual(unscaled.dofs, 1.0, unscaled.settings)
    r_scaled = scaled.manager._global_residual(scaled.dofs, 1.0, scaled.settings)
    k_unscaled = unscaled.manager._global_tangent(unscaled.dofs, 1.0, unscaled.settings).todense()
    k_scaled = scaled.manager._global_tangent(scaled.dofs, 1.0, scaled.settings).todense()

    assert jnp.allclose(scaled.dofs["u"], unscaled.dofs["u"] / scale)
    assert jnp.allclose(r_scaled, scale * r_unscaled, atol=1e-12, rtol=1e-12)
    assert jnp.allclose(k_scaled, scale**2 * k_unscaled, atol=1e-12, rtol=1e-12)


def test_sim_state_dof_scaling_explicit_vtu_output_is_physical(tmp_path):
    mesh = _make_quad_mesh(nx=1, ny=1)
    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.set_dof_scaling({"u": 4.0})
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({"u": dae.NoTimeDerivative()})

    def weak_form(ctx):
        t = ctx.settings.get("current time", 0.0)
        if ctx.mode == "output":
            return {"u": ctx.trial_ansatz["u"](ctx.x_int, t)}
        return ctx.trial_ansatz["u"](ctx.x_int, t) * ctx.test_ansatz["u"](ctx.x_int)

    sim.add_model("__all__", "weak form", weak_form, integration_order=2)
    sim.set_postprocessing_policy(domains="__all__")
    sim.set_initial_conditions(values={"u": 8.0})
    sim.initialize(verbose=0)
    sim.prepare(clean_up=False)

    path = tmp_path / "scaled_primary.vtu"
    sim._write_vtu(path=path)
    out_mesh = meshio.read(path)

    assert jnp.allclose(sim.dofs["u"], 2.0)
    assert np.allclose(out_mesh.point_data["u"], 8.0)


def test_sim_state_collocation_rejects_simplex_projection():
    mesh = _make_tri_mesh(nx=1, ny=1)
    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)

    with pytest.raises(ValueError, match="collocation"):
        sim.set_initial_conditions(
            values={"u": lambda x: x[0]},
            integration_mode="collocation",
        )


def test_sim_state_output_projection_accepts_collocation_mode(tmp_path):
    mesh = _make_quad_mesh(nx=1, ny=1)
    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)

    def model(ctx):
        if ctx.mode == "output":
            x = ctx.trial_ansatz["physical coor"](ctx.x_int)
            return {"jump": jnp.where(x[0] > 0.5, 1e-14, -1e-14)}
        return 0.0 * ctx.test_ansatz["u"](ctx.x_int)

    sim.add_model("__all__", "weak form", model, integration_order=2)
    sim.set_postprocessing_policy(
        domains="__all__",
        integration_mode="collocation",
        projection_atol=1e-20,
    )
    sim.initialize(verbose=-1)

    path = sim._write_vtu(path=tmp_path / "collocation_output.vtu")
    out = meshio.read(path)
    expected = np.where(np.asarray(sim.mesh_info.points)[:, 0] > 0.5, 1e-14, -1e-14)
    assert np.allclose(out.point_data["jump"], expected, atol=0.0, rtol=1e-8)


def test_sim_state_initial_derivatives_are_used_for_newmark_history():
    mesh = _make_quad_mesh(nx=1, ny=1)
    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=2)})
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({"u": dae.Newmark()})
    sim.set_initial_conditions(
        derivatives={"__all__": {"u": {
            1: jnp.asarray([1.0, 2.0]),
            2: jnp.asarray([3.0, 4.0]),
        }}}
    )
    sim.initialize(verbose=-1)

    _, q_der_n = sim.manager._initialize(sim.dofs, sim.settings)
    assert jnp.allclose(q_der_n["u"][0, 0], jnp.asarray([1.0, 2.0]))
    assert jnp.allclose(q_der_n["u"][0, 1], jnp.asarray([3.0, 4.0]))


def test_sim_state_initial_conditions_set_internal_variables_when_materialized():
    mesh = _make_quad_mesh(nx=1, ny=1)
    sim = SimState({
        "alpha": spaces.InternalVariable(field_dimension=1),
        "u": spaces.H1(order=1, dim=2, field_dimension=1),
    })
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({
        "alpha": dae.BackwardEuler(),
        "u": dae.NoTimeDerivative(),
    })

    def weak_form(ctx):
        t = ctx.settings.get("current time", 0.0)
        value = ctx.trial_ansatz["u"](ctx.x_int, t) - ctx.internal_vars["alpha"]
        return value * ctx.test_ansatz["u"](ctx.x_int)

    sim.add_model("__all__", "weak form", weak_form, integration_order=2, internal_variables=("alpha",))
    sim.set_initial_conditions(internal_variables={"__all__": {"alpha": 5.0}})
    sim.initialize(verbose=-1)

    alpha = sim.settings["internal variables"][0]["alpha"]
    alpha_n = sim.settings["internal variables n"][0]["alpha"]
    assert jnp.allclose(alpha, 5.0)
    assert jnp.allclose(alpha_n, 5.0)


def test_sim_state_step_size_controller_reaches_manager():
    mesh = _make_quad_mesh(nx=1, ny=1)
    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)
    controller = dae.RootIterationController(target_niters=3, max_step_size=0.25)

    sim.set_step_size_controller(controller)
    sim.initialize(verbose=-1)

    assert sim.manager.step_size_controller is controller


def test_sim_state_root_solver_reaches_initialized_manager():
    mesh = _make_quad_mesh(nx=1, ny=1)
    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)

    def root_solver(*args, **kwargs):
        return dae.newton_solver(*args, **kwargs)

    sim.initialize(verbose=-1)
    sim.set_root_solver(root_solver)

    assert sim.manager.root_solver is root_solver


def test_sim_state_root_solver_requires_initialized_phase():
    mesh = _make_quad_mesh(nx=1, ny=1)
    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)

    def weak_form(ctx):
        t = ctx.settings.get("current time", 0.0)
        return ctx.trial_ansatz["u"](ctx.x_int, t) * ctx.test_ansatz["u"](ctx.x_int)

    sim.add_model("__all__", "weak form", weak_form, integration_order=2)
    sim.static_settings["solver backend"] = "dense"

    with pytest.raises(RuntimeError, match="after initialize"):
        sim.set_root_solver(dae.newton_solver)

    sim.initialize(verbose=-1)
    sim.prepare()

    with pytest.raises(RuntimeError, match="before prepare"):
        sim.set_root_solver(dae.newton_solver)


def test_sim_state_root_solver_requires_callable():
    mesh = _make_quad_mesh(nx=1, ny=1)
    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)
    sim.initialize(verbose=-1)

    with pytest.raises(TypeError, match="callable"):
        sim.set_root_solver(None)


def test_sim_state_run_result_continues_time_and_state():
    mesh = _make_quad_mesh(nx=1, ny=1)

    def weak_form(ctx):
        t = ctx.settings.get("current time", 0.0)
        return (ctx.trial_ansatz["u"](ctx.x_int, t) - t) * ctx.test_ansatz["u"](ctx.x_int)

    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({"u": dae.NoTimeDerivative()})
    sim.add_model("__all__", "weak form", weak_form, integration_order=2)
    sim.static_settings["solver backend"] = "dense"
    sim.initialize(verbose=-1)
    sim.prepare()
    assert sim.mesh_info is None

    first = sim.run(dt0=1.0, time_span=1.0, num_time_steps=1)
    first_u = first.q["u"]
    second = first.run(dt0=1.0, time_span=1.0, num_time_steps=1)

    assert jnp.allclose(first_u, 1.0, atol=1e-8)
    assert jnp.allclose(second.q["u"], 2.0, atol=1e-8)
    assert jnp.allclose(second.settings["current time"], 2.0)


def test_sim_state_run_is_jittable_and_returns_sim_state():
    mesh = _make_quad_mesh(nx=1, ny=1)

    def weak_form(ctx):
        t = ctx.settings.get("current time", 0.0)
        return (ctx.trial_ansatz["u"](ctx.x_int, t) - t) * ctx.test_ansatz["u"](ctx.x_int)

    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({"u": dae.NoTimeDerivative()})
    sim.add_model("__all__", "weak form", weak_form, integration_order=2)
    sim.static_settings["solver backend"] = "dense"
    sim.initialize(verbose=-1)
    sim.prepare()

    run_once = jax.jit(lambda state: state.run(dt0=1.0, time_span=1.0, num_time_steps=1))
    result = run_once(sim)

    assert isinstance(result, SimState)
    assert jnp.allclose(result.q["u"], 1.0, atol=1e-8)
    assert int(result.num_accepted) == 1
    assert jnp.allclose(result.settings["current time"], 1.0)


def test_sim_state_quasistatic_run_resets_time_but_keeps_state():
    mesh = _make_quad_mesh(nx=1, ny=1)

    def weak_form(ctx):
        t = ctx.settings.get("current time", 0.0)
        return (ctx.trial_ansatz["u"](ctx.x_int, t) - t) * ctx.test_ansatz["u"](ctx.x_int)

    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({"u": dae.NoTimeDerivative()})
    sim.add_model("__all__", "weak form", weak_form, integration_order=2)
    sim.static_settings["solver backend"] = "dense"
    sim.initialize(verbose=-1)
    sim.prepare()

    result = sim.run(dt0=1.0, time_span=1.0, num_time_steps=1, quasistatic=True)

    assert jnp.allclose(result.q["u"], 1.0, atol=1e-8)
    assert jnp.allclose(result.settings["current time"], 0.0)
    assert jnp.allclose(result.settings["last time"], 0.0)


def test_sim_state_manager_controller_can_change_between_runs():
    mesh = _make_quad_mesh(nx=1, ny=1)

    def weak_form(ctx):
        t = ctx.settings.get("current time", 0.0)
        return (ctx.trial_ansatz["u"](ctx.x_int, t) - t) * ctx.test_ansatz["u"](ctx.x_int)

    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({"u": dae.NoTimeDerivative()})
    sim.add_model("__all__", "weak form", weak_form, integration_order=2)
    sim.static_settings["solver backend"] = "dense"
    sim.initialize(verbose=-1)
    sim.prepare()

    first = sim.run(dt0=0.5, time_span=1.0, num_time_steps=10)
    first.manager.step_size_controller = dae.RootIterationController(target_niters=3, max_step_size=0.25)
    second = first.run(dt0=0.5, time_span=1.0, num_time_steps=10)

    assert int(first.num_accepted) == 2
    assert int(second.num_accepted) == 3
    assert jnp.allclose(second.q["u"], 2.0, atol=1e-8)


def test_sim_state_context_dt_tracks_steps_without_internal_variables():
    mesh = _make_quad_mesh(nx=1, ny=1)

    def weak_form(ctx):
        value = ctx.trial_ansatz["u"](ctx.x_int, ctx.t) - ctx.dt
        return value * ctx.test_ansatz["u"](ctx.x_int)

    sim = SimState({"u": spaces.H1(order=1, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({"u": dae.NoTimeDerivative()})
    sim.add_model("__all__", "weak form", weak_form, integration_order=2)
    sim.static_settings["solver backend"] = "dense"
    sim.initialize(verbose=-1)
    sim.prepare()

    result = sim.run(dt0=0.5, time_span=0.5, num_time_steps=1)

    assert jnp.allclose(result.q["u"], 0.5, atol=1e-8)
    assert jnp.allclose(result.settings["current time"], 0.5)
    assert jnp.allclose(result.settings["last time"], 0.5)
