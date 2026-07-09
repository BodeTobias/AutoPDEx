"""Generate small VTU meshes used by the SimState patch tests.

Run from the repository root with:
    python tests/patch_tests/meshes/generate_patch_meshes.py
"""

from __future__ import annotations

from math import comb
from pathlib import Path
from typing import Callable

import gmsh
import meshio
import numpy as np
import pygmsh


MESH_DIR = Path(__file__).resolve().parent

# Gmsh and VTK/Lagrange use almost the same high-order numbering, with a few
# known exceptions. These are the same permutations meshio applies for the
# corresponding hard-coded element types.
_GMSH_TO_VTK_ORDER: dict[str, list[int]] = {
    "tetra10": [0, 1, 2, 3, 4, 5, 6, 7, 9, 8],
    "hexahedron20": [
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15,
    ],
    "hexahedron27": [
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 14, 15,
        22, 23, 21, 24, 20, 25, 26,
    ],
}

# Corner-node count per element family.
_PRIMARY_NODES = {
    "triangle": 3,
    "quad": 4,
    "tetra": 4,
    "hexahedron": 8,
}


def _normalize_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise RuntimeError("Expected a 2D points array.")
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(points.shape[0])])
    elif points.shape[1] > 3:
        points = points[:, :3]
    return points


def _node_data() -> tuple[np.ndarray, np.ndarray]:
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    if node_tags.size == 0:
        raise RuntimeError("Gmsh returned no nodes.")

    points = np.asarray(coords, dtype=float).reshape(-1, 3)
    order = np.argsort(node_tags)
    node_tags = node_tags[order]
    points = points[order]

    tag_to_local = np.full(int(node_tags.max()) + 1, -1, dtype=np.int64)
    tag_to_local[node_tags] = np.arange(node_tags.size, dtype=np.int64)
    return _normalize_points(points), tag_to_local


def _expected_num_nodes(family: str, order: int) -> int:
    if family == "triangle":
        return (order + 1) * (order + 2) // 2
    if family == "quad":
        return (order + 1) ** 2
    if family == "tetra":
        return comb(order + 3, 3)
    if family == "hexahedron":
        return (order + 1) ** 3
    raise ValueError(f"Unsupported family {family!r}.")


def _meshio_name_from_family_order(family: str, order: int) -> str:
    n_nodes = _expected_num_nodes(family, order)
    if family == "triangle":
        return "triangle" if order == 1 else f"triangle{n_nodes}"
    if family == "quad":
        return "quad" if order == 1 else f"quad{n_nodes}"
    if family == "tetra":
        return "tetra" if order == 1 else f"tetra{n_nodes}"
    if family == "hexahedron":
        return "hexahedron" if order == 1 else f"hexahedron{n_nodes}"
    raise ValueError(f"Unsupported family {family!r}.")


def _output_vtu_name(family: str, order: int) -> str:
    if family == "triangle":
        if order <= 2:
            return _meshio_name_from_family_order(family, order)
        return "VTK_LAGRANGE_TRIANGLE"
    if family == "quad":
        if order <= 2:
            return _meshio_name_from_family_order(family, order)
        return "VTK_LAGRANGE_QUADRILATERAL"
    if family == "tetra":
        if order <= 2:
            return _meshio_name_from_family_order(family, order)
        return "VTK_LAGRANGE_TETRAHEDRON"
    if family == "hexahedron":
        if order <= 2:
            return _meshio_name_from_family_order(family, order)
        return "VTK_LAGRANGE_HEXAHEDRON"
    raise ValueError(f"Unsupported family {family!r}.")


def _add_tuple_to_list(ary: list[tuple[int, ...]], x: tuple[int, ...]) -> list[tuple[int, ...]]:
    return [tuple(xv + yv for xv, yv in zip(x, y)) for y in ary]


def _vtk_lagrange_triangle_tuples(order: int) -> list[tuple[int, int]]:
    nodes: list[tuple[int, int]] = []
    offset = (0, 0)

    while order >= 0:
        if order == 0:
            nodes += [offset]
            break

        vertices = [(0, 0), (order, 0), (0, order)]
        nodes += _add_tuple_to_list(vertices, offset)

        if order == 1:
            break

        face_ids = range(1, order)
        faces = (
            [(i, 0) for i in face_ids]
            + [(order - i, i) for i in face_ids]
            + [(0, order - i) for i in face_ids]
        )
        nodes += _add_tuple_to_list(faces, offset)

        order -= 3
        offset = (offset[0] + 1, offset[1] + 1)

    return nodes


def _vtk_lagrange_tetra_tuples(order: int) -> list[tuple[int, int, int]]:
    nodes: list[tuple[int, int, int]] = []
    offset = (0, 0, 0)

    while order >= 0:
        if order == 0:
            nodes += [offset]
            break

        vertices = [(0, 0, 0), (order, 0, 0), (0, order, 0), (0, 0, order)]
        nodes += _add_tuple_to_list(vertices, offset)

        if order == 1:
            break

        edge_ids = range(1, order)
        edges = (
            [(i, 0, 0) for i in edge_ids]
            + [(order - i, i, 0) for i in edge_ids]
            + [(0, order - i, 0) for i in edge_ids]
            + [(0, 0, i) for i in edge_ids]
            + [(order - i, 0, i) for i in edge_ids]
            + [(0, order - i, i) for i in edge_ids]
        )
        nodes += _add_tuple_to_list(edges, offset)

        face_ids = _add_tuple_to_list(_vtk_lagrange_triangle_tuples(order - 3), (1, 1))
        faces = (
            [(i, 0, j) for i, j in face_ids]  # face (0, 2, 3)
            + [(j, order - (i + j), i) for i, j in face_ids]  # face (1, 2, 3)
            + [(0, j, i) for i, j in face_ids]  # face (0, 1, 3)
            + [(j, i, 0) for i, j in face_ids]  # face (0, 1, 2)
        )
        nodes += _add_tuple_to_list(faces, offset)

        order -= 4
        offset = (offset[0] + 1, offset[1] + 1, offset[2] + 1)

    return nodes


def _barycentric_coordinates(point: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    a = np.column_stack((vertices[1] - vertices[0], vertices[2] - vertices[0], vertices[3] - vertices[0]))
    rhs = point - vertices[0]
    coeffs = np.linalg.solve(a, rhs)
    l1, l2, l3 = coeffs
    l0 = 1.0 - l1 - l2 - l3
    return np.asarray([l0, l1, l2, l3], dtype=float)


def _gmsh_tetra_permutation_to_vtk(points: np.ndarray, cells: np.ndarray, order: int) -> np.ndarray:
    if cells.shape[0] == 0:
        raise RuntimeError("Expected at least one tetrahedron to determine node permutation.")

    first_cell = np.asarray(cells[0], dtype=np.int64)
    cell_points = points[first_cell]
    vertices = cell_points[:4]

    gmsh_tuple_to_local: dict[tuple[int, int, int], int] = {}
    for local_index, point in enumerate(cell_points):
        bary = _barycentric_coordinates(point, vertices)
        scaled = np.rint(order * bary[1:]).astype(int)
        key = tuple(int(v) for v in scaled)
        gmsh_tuple_to_local[key] = local_index

    vtk_tuples = _vtk_lagrange_tetra_tuples(order)
    try:
        return np.asarray([gmsh_tuple_to_local[key] for key in vtk_tuples], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError(
            f"Failed to determine Gmsh→VTK tetra permutation for order {order}. Missing tuple {exc.args[0]!r}."
        ) from exc


def _structured_quad_mesh(order: int, nx: int = 3, ny: int = 2) -> meshio.Mesh:
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1.")

    npx = nx * order + 1
    npy = ny * order + 1

    xs = np.linspace(0.0, 1.0, npx)
    ys = np.linspace(0.0, 1.0, npy)

    points = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=float)

    def point_index(ix: int, iy: int) -> int:
        return iy * npx + ix

    def square_indices(l: int, u: int) -> list[tuple[int, int]]:
        if l < u:
            indices: list[tuple[int, int]] = [(l, l), (u, l), (u, u), (l, u)]
            for k in range(l + 1, u):
                indices.append((k, l))
            for k in range(l + 1, u):
                indices.append((u, k))
            for k in range(u - 1, l, -1):
                indices.append((k, u))
            for k in range(u - 1, l, -1):
                indices.append((l, k))
            return indices
        return [(l, u)]

    def quad_ordering(p: int) -> list[tuple[int, int]]:
        ordering = square_indices(0, p)
        for k in range(1, p):
            l = k
            u = p - k
            if l <= u:
                ordering += square_indices(l, u)
        return ordering

    local_pattern = quad_ordering(order)
    cells = []
    for ey in range(ny):
        for ex in range(nx):
            ix0 = ex * order
            iy0 = ey * order
            cells.append([
                point_index(ix0 + i, iy0 + j)
                for (i, j) in local_pattern
            ])

    cell_type = _output_vtu_name("quad", order)
    return meshio.Mesh(points=points, cells=[(cell_type, np.asarray(cells, dtype=np.int64))])


def _extract_family_mesh(dim: int, family: str, order: int) -> meshio.Mesh:
    points, tag_to_local = _node_data()
    elem_types, elem_tags, node_tags = gmsh.model.mesh.getElements(dim)
    if not elem_types:
        raise RuntimeError(f"Gmsh returned no elements in dimension {dim}.")

    expected_num_nodes = _expected_num_nodes(family, order)
    expected_primary_nodes = _PRIMARY_NODES[family]
    meshio_name = _meshio_name_from_family_order(family, order)

    blocks: list[np.ndarray] = []
    seen: list[str] = []

    for elem_type, elem_tags_block, node_tags_block in zip(elem_types, elem_tags, node_tags):
        element_name, element_dim, element_order, n_nodes, _, n_primary_nodes = (
            gmsh.model.mesh.getElementProperties(int(elem_type))
        )
        normalized_name = str(element_name).lower().replace(" ", "")
        seen.append(
            f"{normalized_name}(dim={element_dim},order={element_order},"
            f"nodes={n_nodes},primary={n_primary_nodes})"
        )

        if int(element_dim) != dim:
            continue
        if int(element_order) != order:
            continue
        if int(n_nodes) != expected_num_nodes:
            continue
        if int(n_primary_nodes) != expected_primary_nodes:
            continue

        data = np.asarray(node_tags_block, dtype=np.int64).reshape(-1, int(n_nodes))
        local = tag_to_local[data]
        if np.any(local < 0):
            raise RuntimeError("Failed to map Gmsh node tags to local point indices.")

        perm = _GMSH_TO_VTK_ORDER.get(meshio_name)
        if perm is not None:
            local = local[:, perm]
        elif family == "tetra" and order >= 3:
            local = local[:, _gmsh_tetra_permutation_to_vtk(points, local, order)]
        blocks.append(local)

    if not blocks:
        raise RuntimeError(
            f"Generated mesh does not contain the requested family/order combination "
            f"for family={family!r}, order={order}. "
            f"Seen full-dimensional element types: {sorted(set(seen))!r}."
        )

    cells = np.concatenate(blocks, axis=0) if len(blocks) > 1 else blocks[0]
    return meshio.Mesh(points=points, cells=[(_output_vtu_name(family, order), cells)])


def _generate(
    build: Callable[[pygmsh.geo.Geometry], None],
    dim: int,
    family: str,
    order: int,
    recombine: bool = False,
) -> meshio.Mesh:
    with pygmsh.geo.Geometry() as geom:
        build(geom)
        geom.synchronize()

        if recombine:
            gmsh.option.setNumber("Mesh.RecombineAll", 1)

        gmsh.model.mesh.generate(dim)

        if recombine:
            gmsh.model.mesh.recombine()

        gmsh.model.mesh.setOrder(order)
        return _extract_family_mesh(dim=dim, family=family, order=order)


def _write(name: str, mesh: meshio.Mesh) -> None:
    meshio.write(MESH_DIR / name, mesh, file_format="vtu", binary=False)


def _triangle(order: int) -> meshio.Mesh:
    def build(geom: pygmsh.geo.Geometry) -> None:
        geom.add_rectangle(0.0, 1.0, 0.0, 1.0, 0.0, mesh_size=0.8)

    return _generate(build, dim=2, family="triangle", order=order, recombine=False)


def _quad(order: int) -> meshio.Mesh:
    return _structured_quad_mesh(order, nx=3, ny=2)


def _tetra(order: int) -> meshio.Mesh:
    def build(geom: pygmsh.geo.Geometry) -> None:
        geom.add_box(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, mesh_size=0.9)

    return _generate(build, dim=3, family="tetra", order=order, recombine=False)


def _hexahedron(order: int) -> meshio.Mesh:
    def build(geom: pygmsh.geo.Geometry) -> None:
        rect = geom.add_rectangle(0.0, 1.0, 0.0, 1.0, 0.0, mesh_size=0.9)
        for curve in rect.curves:
            geom.set_transfinite_curve(curve, 3, "Progression", 1.0)
        geom.set_transfinite_surface(rect.surface, "Left", rect.points)
        geom.set_recombined_surfaces([rect.surface])
        geom.extrude(rect.surface, [0.0, 0.0, 1.0], num_layers=2, recombine=True)

    return _generate(build, dim=3, family="hexahedron", order=order, recombine=True)


def main() -> None:
    cases: list[tuple[str, meshio.Mesh]] = []

    for order in range(1, 11):
        cases.append((f"triangle_p{order}.vtu", _triangle(order)))

    for order in range(1, 21):
        cases.append((f"quad_p{order}.vtu", _quad(order)))

    for order in range(1, 6):
        cases.append((f"tetra_p{order}.vtu", _tetra(order)))

    for order in range(1, 3):
        cases.append((f"hexahedron_p{order}.vtu", _hexahedron(order)))

    for name, mesh in cases:
        _write(name, mesh)
        print(f"wrote {name}: {mesh.cells[0].type}")


if __name__ == "__main__":
    main()
