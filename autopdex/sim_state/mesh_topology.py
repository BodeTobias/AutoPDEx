# autopdex/sim_state/mesh_topology.py
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

"""Pure topology/meshio helpers (no SimState coupling)."""

import copy
import math
import warnings
from typing import Callable, Dict, Optional, List, Tuple

import numpy as np
import meshio

try:
    from autopdex.sim_state._native._topo_hash import lib as _topo_lib, ffi as _topo_ffi
    _HAS_NATIVE_TOPO = True
except ImportError:
    _topo_lib = None
    _topo_ffi = None
    _HAS_NATIVE_TOPO = False
    warnings.warn(
        "autopdex: native topology extension (_topo_hash) is unavailable; "
        "falling back to the pure-NumPy O(n log n) mesh-build path. "
        "Reinstall with a C compiler (`pip install -e .`) to enable the O(n) path.",
        RuntimeWarning,
        stacklevel=2,
    )


def _source_mesh_snapshot(mesh: meshio.Mesh) -> meshio.Mesh:
    """Return a copy of the preprocessed ``mesh`` for retention through phases 0–3.

    Faithfully preserves points, connectivity (original high-order ordering, needed
    for VTU output and mesh rebuilds), ``cell_data``, ``cell_sets`` (domain
    definitions, also mutated by ``_add_facet_domain``), ``point_sets`` and
    ``field_data``. Deliberately drops ``point_data``: the nodal payload of the
    input mesh is never read internally (VTU output writes its own ``point_data``),
    so a full ``copy.deepcopy(mesh)`` would retain it needlessly through setup.
    """
    cells = [meshio.CellBlock(cb.type, np.array(cb.data, copy=True)) for cb in mesh.cells]
    cell_data = {
        name: [np.array(block, copy=True) for block in blocks]
        for name, blocks in (getattr(mesh, "cell_data", None) or {}).items()
    }
    cell_sets = {
        name: [None if blk is None else np.array(blk, copy=True) for blk in blocks]
        for name, blocks in (getattr(mesh, "cell_sets", None) or {}).items()
    }
    point_sets = {
        name: np.array(idx, copy=True)
        for name, idx in (getattr(mesh, "point_sets", None) or {}).items()
    }
    return meshio.Mesh(
        points=np.array(mesh.points, copy=True),
        cells=cells,
        cell_data=cell_data,
        cell_sets=cell_sets,
        point_sets=point_sets,
        field_data=copy.deepcopy(getattr(mesh, "field_data", None) or {}),
    )


def _cell_dim(cell_type: str) -> int:
    ct = cell_type.lower()
    if ct.startswith("vertex"):
        return 0
    if ct.startswith("line"):
        return 1
    if ct.startswith("triangle") or ct.startswith("quad") or ct.startswith("vtk_lagrange_triangle") or ct.startswith("vtk_lagrange_quadrilateral"):
        return 2
    if ct.startswith("tetra") or ct.startswith("hexahedron") or ct.startswith("vtk_lagrange_tetrahedron") or ct.startswith("vtk_lagrange_hexahedron"):
        return 3
    # add wedge/pyramid if needed
    return -1


def _triangle_order_from_node_count(n_nodes: int) -> Optional[int]:
    for p in range(1, 21):
        if (p + 1) * (p + 2) // 2 == n_nodes:
            return p
    return None


def _tetra_order_from_node_count(n_nodes: int) -> Optional[int]:
    for p in range(1, 21):
        if (p + 1) * (p + 2) * (p + 3) // 6 == n_nodes:
            return p
    return None


def _vtk_lagrange_triangle_ij_order(p: int) -> List[Tuple[int, int]]:
    """Ordering used by the high-level triangle/tetra Lagrange basis."""
    if p < 0:
        raise ValueError("p must be >= 0")
    if p == 0:
        return [(0, 0)]

    ij: List[Tuple[int, int]] = [(0, 0), (p, 0), (0, p)]
    ij += [(i, 0) for i in range(1, p)]
    ij += [(p - i, i) for i in range(1, p)]
    ij += [(0, p - i) for i in range(1, p)]

    if p >= 3:
        ij += [(i + 1, j + 1) for i, j in _vtk_lagrange_triangle_ij_order(p - 3)]
    return ij


def _line_nodes_from_geometric_order(nodes: List[int]) -> List[int]:
    if len(nodes) <= 2:
        return nodes
    return [nodes[0], nodes[-1], *nodes[1:-1]]


def _vtk_lagrange_tetra_ijk_order(p: int) -> List[Tuple[int, int, int]]:
    """Ordering used by the high-level tetrahedral Lagrange basis."""
    if p < 0:
        raise ValueError("p must be >= 0")
    if p == 0:
        return [(0, 0, 0)]

    vertices = [(0, 0, 0), (p, 0, 0), (0, p, 0), (0, 0, p)]
    ijk: List[Tuple[int, int, int]] = list(vertices)

    edge_vertices = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    for a_id, b_id in edge_vertices:
        a = vertices[a_id]
        b = vertices[b_id]
        for t in range(1, p):
            ijk.append(tuple(((p - t) * a[d] + t * b[d]) // p for d in range(3)))

    face_vertices = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
    if p >= 3:
        inner_face_ij = [(i + 1, j + 1) for i, j in _vtk_lagrange_triangle_ij_order(p - 3)]
        for a_id, b_id, c_id in face_vertices:
            a = vertices[a_id]
            b = vertices[b_id]
            c = vertices[c_id]
            for i, j in inner_face_ij:
                ijk.append(tuple(((p - i - j) * a[d] + i * b[d] + j * c[d]) // p for d in range(3)))

    if p >= 4:
        for i, j, k in _vtk_lagrange_tetra_ijk_order(p - 4):
            ijk.append((i + 1, j + 1, k + 1))
    return ijk


def _vtk_lagrange_tetra_ijk_order_meshio(p: int) -> List[Tuple[int, int, int]]:
    """VTK/Gmsh Lagrange tetrahedron node ordering used by meshio VTU files."""
    if p < 0:
        raise ValueError("p must be >= 0")

    ijk: List[Tuple[int, int, int]] = []
    offset = (0, 0, 0)
    order = p

    def shifted(nodes: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        return [tuple(x[d] + offset[d] for d in range(3)) for x in nodes]

    while order >= 0:
        if order == 0:
            ijk.append(offset)
            break

        vertices = [(0, 0, 0), (order, 0, 0), (0, order, 0), (0, 0, order)]
        ijk += shifted(vertices)

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
        ijk += shifted(edges)

        if order >= 3:
            face_ids = [(i + 1, j + 1) for i, j in _vtk_lagrange_triangle_ij_order(order - 3)]
            faces = (
                [(i, 0, j) for i, j in face_ids]
                + [(j, order - (i + j), i) for i, j in face_ids]
                + [(0, j, i) for i, j in face_ids]
                + [(j, i, 0) for i, j in face_ids]
            )
            ijk += shifted(faces)

        order -= 4
        offset = (offset[0] + 1, offset[1] + 1, offset[2] + 1)

    return ijk


def _vtk_lagrange_quad_ij_order(p: int) -> List[Tuple[int, int]]:
    """
    VTK Lagrange quadrilateral ordering for full tensor-product nodes.
    Returns list of (i,j) for each local node id, with i,j in [0..p].
    For p=1: 4 nodes, p=2: 9 nodes, p=3: 16 nodes, ...
    """
    if p < 1:
        raise ValueError("p must be >= 1")

    ij = []
    # vertices (0..3): (0,0), (p,0), (p,p), (0,p)
    ij += [(0, 0), (p, 0), (p, p), (0, p)]

    # edge points (excluding vertices)
    # bottom: (i,0), i=1..p-1
    for i in range(1, p):
        ij.append((i, 0))
    # right: (p,j), j=1..p-1
    for j in range(1, p):
        ij.append((p, j))
    # top: (i,p), i=p-1..1  (note reversed to follow edge (2->3))
    for i in range(p - 1, 0, -1):
        ij.append((i, p))
    # left: (0,j), j=p-1..1 (note reversed to follow edge (3->0))
    for j in range(p - 1, 0, -1):
        ij.append((0, j))

    # face interior (excluding edges): row-major in j, then i
    for j in range(1, p):
        for i in range(1, p):
            ij.append((i, j))

    return ij


def _quad_order_from_cell_type(cell_type: str) -> Optional[int]:
    """
    Return polynomial order p for full Lagrange quad types:
      quad      -> p=1
      quad9     -> p=2
      quad16    -> p=3
      quad25    -> p=4
      ...
    Returns None if not full-Lagrange.
    """
    ct = cell_type.lower()
    if ct == "quad":
        return 1
    if ct.startswith("quad") and ct[4:].isdigit():
        n = int(ct[4:])
        s = int(round(np.sqrt(n)))
        if s * s == n and s >= 2:
            return s - 1
    return None


def _line_type_for_order(p: int) -> str:
    # p=1 => 2 nodes => "line"
    # p=2 => 3 nodes => "line3", etc.
    return "line" if (p + 1) == 2 else f"line{p+1}"


def _quad_type_for_order(p: int) -> str:
    # p=1 => 4 nodes => "quad"
    # p=2 => 9 nodes => "quad9", p=3 => "quad16", ...
    return "quad" if p == 1 else f"quad{(p+1)**2}"


def _vtk_lagrange_hex_ijk_order(p: int) -> List[Tuple[int, int, int]]:
    """
    VTK Lagrange hexahedron ordering for full tensor-product nodes.
    Returns list of (i,j,k) for each local node id, i,j,k in [0..p].
    For p=1: 8 nodes, p=2: 27 nodes, p=3: 64 nodes, ...
    """
    if p < 1:
        raise ValueError("p must be >= 1")

    ijk = []

    # vertices (0..7) in standard hex order
    V = [
        (0, 0, 0), (p, 0, 0), (p, p, 0), (0, p, 0),
        (0, 0, p), (p, 0, p), (p, p, p), (0, p, p),
    ]
    ijk += V

    # edges in VTK order (see VTK docs for quadratic/triquadratic hex):
    # (0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)
    edges = [
        (V[0], V[1]), (V[1], V[2]), (V[2], V[3]), (V[3], V[0]),
        (V[4], V[5]), (V[5], V[6]), (V[6], V[7]), (V[7], V[4]),
        (V[0], V[4]), (V[1], V[5]), (V[2], V[6]), (V[3], V[7]),
    ]
    for a, b in edges:
        ax, ay, az = a
        bx, by, bz = b
        for t in range(1, p):
            ijk.append((
                ax + (bx - ax) * t // p,
                ay + (by - ay) * t // p,
                az + (bz - az) * t // p,
            ))

    # faces (6) interior points (excluding edges), matching the brick basis:
    # i=0, i=p, j=0, j=p, k=0, k=p
    # For each face, we iterate v=1..p-1, u=1..p-1 in row-major
    def add_face(u_to_ijk):
        for v in range(1, p):
            for u in range(1, p):
                ijk.append(u_to_ijk(u, v))

    # i=0, i=p, j=0, j=p, k=0, k=p
    add_face(lambda u, v: (0, u, v))
    add_face(lambda u, v: (p, u, v))
    add_face(lambda u, v: (u, 0, v))
    add_face(lambda u, v: (u, p, v))
    add_face(lambda u, v: (u, v, 0))
    add_face(lambda u, v: (u, v, p))

    # volume interior (excluding faces): k,j,i in row-major
    for k in range(1, p):
        for j in range(1, p):
            for i in range(1, p):
                ijk.append((i, j, k))

    return ijk


def _autopdex_hex_ijk_order(p: int) -> List[Tuple[int, int, int]]:
    """Ordering used by the currently implemented brick basis."""
    return _vtk_lagrange_hex_ijk_order(p)


def _permutation_from_source_to_target(
    source: List[Tuple[int, ...]], target: List[Tuple[int, ...]]
) -> np.ndarray:
    source_id = {node: i for i, node in enumerate(source)}
    return np.asarray([source_id[node] for node in target], dtype=int)


def _normalize_mesh_cell_node_ordering(mesh: meshio.Mesh) -> meshio.Mesh:
    """
    Convert supported VTK/Gmsh high-order cell node orderings to the local
    ordering expected by the currently implemented AutoPDEx basis functions.
    """
    normalized = copy.deepcopy(mesh)
    for block in normalized.cells:
        data = np.asarray(block.data, dtype=int)
        if data.ndim != 2 or data.shape[1] <= 0:
            continue

        base = _canonical_base_type(block.type)
        n_nodes = int(data.shape[1])

        if base == "tetra":
            p = _tetra_order_from_node_count(n_nodes)
            if p is not None and p >= 3:
                source = _vtk_lagrange_tetra_ijk_order_meshio(p)
                target = _vtk_lagrange_tetra_ijk_order(p)
                block.data = data[:, _permutation_from_source_to_target(source, target)]
            continue


    return normalized


def _deduplicate_mesh_points_exact(mesh: meshio.Mesh) -> meshio.Mesh:
    """
    Merge mesh points with exactly identical coordinates and remap all cell
    connectivities accordingly.

    Point-associated data is reduced consistently to the retained unique points.
    Cell-associated metadata is left unchanged because cell indices do not change.
    """
    deduplicated = copy.deepcopy(mesh)
    points = np.asarray(deduplicated.points)
    if points.ndim != 2 or points.shape[0] == 0:
        return deduplicated

    n, gdim = points.shape
    if _HAS_NATIVE_TOPO and gdim in (1, 2, 3, 4):
        # O(n) exact dedup: hash the coordinate bit patterns. Adding 0.0 maps
        # -0.0 to +0.0 so coincident nodes still merge (matching ==-semantics).
        pts = np.ascontiguousarray(points, dtype=np.float64) + 0.0
        key = pts.view(np.uint64)
        inverse = np.empty(n, dtype=np.int64)
        first = np.empty(n, dtype=np.int64)
        nu = _topo_lib.factorize_u64_rows(
            _topo_ffi.cast("uint64_t*", key.ctypes.data), n, gdim,
            _topo_ffi.cast("int64_t*", inverse.ctypes.data),
            _topo_ffi.cast("int64_t*", first.ctypes.data))
        if nu < 0:
            raise MemoryError("_topo_hash: allocation failed")
        if nu == n:
            return deduplicated
        first_idx = first[:nu]
        unique_points = points[first_idx]
    else:
        unique_points, first_idx, inverse = np.unique(
            points, axis=0, return_index=True, return_inverse=True
        )
        if unique_points.shape[0] == points.shape[0]:
            return deduplicated
        inverse = np.asarray(inverse).reshape(-1)

    deduplicated.points = unique_points

    for block in deduplicated.cells:
        data = np.asarray(block.data, dtype=int)
        if data.ndim != 2 or data.shape[1] == 0:
            block.data = inverse[data]
            continue

        block.data = inverse[data]

    if getattr(deduplicated, "point_data", None):
        deduplicated.point_data = {
            name: np.asarray(values)[first_idx]
            for name, values in deduplicated.point_data.items()
        }

    if getattr(deduplicated, "point_sets", None):
        deduplicated.point_sets = {
            name: np.unique(inverse[np.asarray(point_ids, dtype=int).reshape(-1)])
            for name, point_ids in deduplicated.point_sets.items()
        }

    return deduplicated


def _collapse_degenerate_linear_quad_to_triangle(conn: np.ndarray) -> Optional[np.ndarray]:
    """
    Collapse a linear quad with one repeated corner on the boundary to a triangle.

    Supported patterns are only adjacent/cyclic duplicates such as:
      [a,a,b,c], [a,b,b,c], [a,b,c,c], [a,b,c,a]
    """
    conn = np.asarray(conn, dtype=int).reshape(-1)
    if conn.shape != (4,):
        raise ValueError("Only 4-node linear quads can be collapsed to triangles.")

    collapsed: List[int] = []
    for node in conn.tolist():
        if not collapsed or node != collapsed[-1]:
            collapsed.append(node)

    if len(collapsed) > 1 and collapsed[0] == collapsed[-1]:
        collapsed.pop()

    if len(collapsed) == 3 and len(set(collapsed)) == 3:
        return np.asarray(collapsed, dtype=conn.dtype)
    return None


def _cell_duplicate_node_mask(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=int)
    n_cells = int(data.shape[0]) if data.ndim >= 1 else 0
    if data.ndim != 2 or data.shape[1] <= 1:
        return np.zeros((n_cells,), dtype=bool)

    if data.shape[1] == 4:
        a, b, c, d = data.T
        return (
            (a == b) | (a == c) | (a == d)
            | (b == c) | (b == d)
            | (c == d)
        )

    sorted_data = np.sort(data, axis=1)
    return np.any(sorted_data[:, 1:] == sorted_data[:, :-1], axis=1)


def _convert_degenerate_linear_quads_to_triangles(
    mesh: meshio.Mesh,
    *,
    enabled: bool = False,
) -> meshio.Mesh:
    """
    Replace degenerate 4-node quads with 3-node triangles when a single boundary
    corner collapse occurred.
    """
    if not any(block.type == "quad" for block in mesh.cells):
        return mesh

    needs_conversion = False
    for block in mesh.cells:
        data = np.asarray(block.data, dtype=int)
        duplicate_mask = _cell_duplicate_node_mask(data)
        if not np.any(duplicate_mask):
            continue

        cell_idx = int(np.flatnonzero(duplicate_mask)[0])
        if block.type != "quad":
            raise ValueError(
                f"Degenerate '{block.type}' cell {cell_idx} is not supported during mesh import."
            )

        conn = data[cell_idx]
        unique_count = int(np.unique(conn).size)
        if unique_count < 3:
            raise ValueError(
                f"Degenerate 'quad' cell {cell_idx} has fewer than 3 unique nodes and cannot be imported."
            )

        if unique_count > 3:
            raise ValueError(
                f"Degenerate 'quad' cell {cell_idx} has an unsupported repetition pattern and cannot be imported."
            )

        if not enabled:
            raise ValueError(
                f"Degenerate 'quad' cell {cell_idx} requires convert_degenerate_quads=True during import."
            )

        needs_conversion = True
        break

    if not needs_conversion:
        return mesh

    converted = copy.deepcopy(mesh)
    old_cells = list(converted.cells)
    old_cell_data = {key: list(values) for key, values in (converted.cell_data or {}).items()}
    old_cell_sets = {name: list(values) for name, values in (converted.cell_sets or {}).items()}
    cell_data_dtypes = {
        key: (np.asarray(values[0]).dtype if len(values) > 0 else float)
        for key, values in old_cell_data.items()
    }

    new_cells: List[meshio.CellBlock] = []
    new_cell_data = {key: [] for key in old_cell_data}
    new_cell_sets = {name: [] for name in old_cell_sets}

    def _block_cell_data(key: str, block_idx: int, n_cells: int) -> np.ndarray:
        values = old_cell_data.get(key, [])
        if block_idx < len(values):
            return np.asarray(values[block_idx])
        return np.zeros((n_cells,), dtype=cell_data_dtypes[key])

    def _block_cell_set(name: str, block_idx: int) -> np.ndarray:
        values = old_cell_sets.get(name, [])
        if block_idx < len(values):
            return np.asarray(values[block_idx], dtype=int).reshape(-1)
        return np.zeros((0,), dtype=int)

    def _append_block(
        cell_type: str,
        block_tags: List[str],
        block_data: np.ndarray,
        src_indices: np.ndarray,
        block_idx: int,
        n_old_cells: int,
    ) -> None:
        block_data = np.asarray(block_data, dtype=int)
        src_indices = np.asarray(src_indices, dtype=int).reshape(-1)
        new_cells.append(meshio.CellBlock(cell_type, block_data, tags=list(block_tags)))

        for key in new_cell_data:
            values = _block_cell_data(key, block_idx, n_old_cells)
            new_cell_data[key].append(np.asarray(values[src_indices]).copy())

        local_map = np.full((n_old_cells,), -1, dtype=int)
        local_map[src_indices] = np.arange(src_indices.size, dtype=int)
        for name in new_cell_sets:
            old_ids = _block_cell_set(name, block_idx)
            if old_ids.size == 0:
                new_cell_sets[name].append(np.zeros((0,), dtype=int))
                continue
            mapped = local_map[old_ids]
            new_cell_sets[name].append(mapped[mapped >= 0])

    for block_idx, block in enumerate(old_cells):
        data = np.asarray(block.data, dtype=int)
        n_cells = int(data.shape[0]) if data.ndim >= 1 else 0
        tags = list(getattr(block, "tags", []))

        if block.type != "quad":
            if data.ndim == 2 and data.shape[1] > 0:
                for cell_idx, conn in enumerate(data):
                    if np.unique(conn).size < conn.size:
                        raise ValueError(
                            f"Degenerate '{block.type}' cell {cell_idx} is not supported during mesh import."
                        )
            _append_block(block.type, tags, data, np.arange(n_cells, dtype=int), block_idx, n_cells)
            continue

        quad_rows: List[np.ndarray] = []
        quad_src: List[int] = []
        tri_rows: List[np.ndarray] = []
        tri_src: List[int] = []

        for cell_idx, conn in enumerate(data):
            unique_count = int(np.unique(conn).size)
            if unique_count == conn.size:
                quad_rows.append(np.asarray(conn, dtype=int))
                quad_src.append(cell_idx)
                continue

            if unique_count < 3:
                raise ValueError(
                    f"Degenerate 'quad' cell {cell_idx} has fewer than 3 unique nodes and cannot be imported."
                )

            if unique_count > 3:
                raise ValueError(
                    f"Degenerate 'quad' cell {cell_idx} has an unsupported repetition pattern and cannot be imported."
                )

            if not enabled:
                raise ValueError(
                    f"Degenerate 'quad' cell {cell_idx} requires convert_degenerate_quads=True during import."
                )

            triangle = _collapse_degenerate_linear_quad_to_triangle(conn)
            if triangle is None:
                raise ValueError(
                    f"Degenerate 'quad' cell {cell_idx} has a non-adjacent repeated corner pattern and cannot be converted."
                )

            tri_rows.append(triangle)
            tri_src.append(cell_idx)

        if quad_rows:
            _append_block(
                "quad",
                tags,
                np.asarray(quad_rows, dtype=int),
                np.asarray(quad_src, dtype=int),
                block_idx,
                n_cells,
            )
        if tri_rows:
            _append_block(
                "triangle",
                tags,
                np.asarray(tri_rows, dtype=int),
                np.asarray(tri_src, dtype=int),
                block_idx,
                n_cells,
            )

    converted.cells = new_cells
    converted.cell_data = new_cell_data
    converted.cell_sets = new_cell_sets
    return converted


def _hex_order_from_cell_type(cell_type: str) -> Optional[int]:
    """
    Full Lagrange hex:
      hexahedron    -> p=1
      hexahedron27  -> p=2
      hexahedron64  -> p=3
      hexahedron125 -> p=4
      ...
    Returns None if not full-Lagrange.
    """
    ct = cell_type.lower()
    if ct == "hexahedron":
        return 1
    if ct.startswith("hexahedron") and ct[10:].isdigit():
        n = int(ct[10:])
        c = int(round(n ** (1.0 / 3.0)))
        if c**3 == n and c >= 2:
            return c - 1
    return None


def _edge_templates_2d(cell_type: str, n_nodes: int) -> Tuple[str, List[List[int]]]:
    """
    Return (edge_cell_type, list_of_edges_as local index lists) for 2D cells.
    Line cells use the same node ordering as the 1D Lagrange basis: endpoints first,
    then interior points in geometric direction.
    """
    ct = cell_type.lower()
    base = _canonical_base_type(cell_type)

    if base == "triangle":
        p = _triangle_order_from_node_count(n_nodes)
        if p is None:
            raise NotImplementedError(f"2D facet extraction not implemented for '{cell_type}'.")
        ij_order = _vtk_lagrange_triangle_ij_order(p)
        id_of = {ij: idx for idx, ij in enumerate(ij_order)}
        e0 = [(i, 0) for i in range(0, p + 1)]
        e1 = [(p - i, i) for i in range(0, p + 1)]
        e2 = [(0, p - i) for i in range(0, p + 1)]
        edges = [_line_nodes_from_geometric_order([id_of[x] for x in e]) for e in (e0, e1, e2)]
        return _line_type_for_order(p), edges

    # quad serendipity (8) and full Lagrange quads
    if ct == "quad" and n_nodes >= 4:
        return "line", [[0, 1], [1, 2], [2, 3], [3, 0]]
    if ct == "quad8" and n_nodes >= 8:
        return "line3", [[0, 1, 4], [1, 2, 5], [2, 3, 6], [3, 0, 7]]

    p = _quad_order_from_cell_type(ct)
    if p is None and base == "quad":
        s = int(round(math.isqrt(n_nodes)))
        if s * s == n_nodes and s >= 2:
            p = s - 1
    if p is not None:
        ij_order = _vtk_lagrange_quad_ij_order(p)
        if len(ij_order) != (p + 1) ** 2 or n_nodes < (p + 1) ** 2:
            raise ValueError(f"{cell_type}: inconsistent node count (expected {(p+1)**2}, got {n_nodes}).")
        id_of = {ij: idx for idx, ij in enumerate(ij_order)}

        e0 = [(i, 0) for i in range(0, p + 1)]
        e1 = [(p, j) for j in range(0, p + 1)]
        e2 = [(i, p) for i in range(p, -1, -1)]
        e3 = [(0, j) for j in range(p, -1, -1)]
        edges = [_line_nodes_from_geometric_order([id_of[x] for x in e]) for e in (e0, e1, e2, e3)]
        return _line_type_for_order(p), edges

    raise NotImplementedError(f"2D facet extraction not implemented for '{cell_type}'.")


def _face_templates_3d(cell_type: str, n_nodes: int) -> Tuple[str, List[List[int]]]:
    """
    Return (face_cell_type, list_of_faces_as_local_index_lists)
    for 3D volume cells.
    """
    ct = cell_type.lower()

    # tetra / tetra10
    if ct == "tetra" and n_nodes >= 4:
        # faces as triangles (corner-only)
        return "triangle", [
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2],
        ]
    if ct == "tetra10" and n_nodes >= 10:
        # VTK quadratic tetra: 4-9 are midedge nodes on (0,1),(1,2),(2,0),(0,3),(1,3),(2,3)
        # Faces as triangle6 with node order (v0,v1,v2, e01,e12,e20)
        return "triangle6", [
            [0, 1, 2, 4, 5, 6],
            [0, 3, 1, 7, 8, 4],
            [0, 2, 3, 6, 9, 7],
            [1, 3, 2, 8, 9, 5],
        ]

    p_tet = _tetra_order_from_node_count(n_nodes) if _canonical_base_type(cell_type) == "tetra" else None
    if p_tet is not None:
        ijk_order = _vtk_lagrange_tetra_ijk_order(p_tet)
        if len(ijk_order) > n_nodes:
            raise ValueError(f"{cell_type}: inconsistent node count (expected at least {len(ijk_order)}, got {n_nodes}).")
        id_of = {ijk: idx for idx, ijk in enumerate(ijk_order)}
        vertices = [(0, 0, 0), (p_tet, 0, 0), (0, p_tet, 0), (0, 0, p_tet)]
        face_vertices = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
        tri_order = _vtk_lagrange_triangle_ij_order(p_tet)

        def face_from_vertices(a_id, b_id, c_id):
            a = vertices[a_id]
            b = vertices[b_id]
            c = vertices[c_id]
            nodes = []
            for i, j in tri_order:
                coord = tuple(((p_tet - i - j) * a[d] + i * b[d] + j * c[d]) // p_tet for d in range(3))
                nodes.append(id_of[coord])
            return nodes

        face_nodes = (p_tet + 1) * (p_tet + 2) // 2
        face_type = "triangle" if p_tet == 1 else f"triangle{face_nodes}"
        return face_type, [face_from_vertices(*face) for face in face_vertices]

    # hexahedron serendipity (20)
    if ct == "hexahedron" and n_nodes >= 8:
        return "quad", [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ]
    if ct == "hexahedron20" and n_nodes >= 20:
        # VTK quadratic hex: 8-19 are edge nodes on edges:
        # (0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)
        # Faces as quad8: (v0,v1,v2,v3, e01,e12,e23,e30)
        return "quad8", [
            [0, 1, 2, 3,  8,  9, 10, 11],
            [4, 5, 6, 7, 12, 13, 14, 15],
            [0, 1, 5, 4,  8, 17, 12, 16],
            [1, 2, 6, 5,  9, 18, 13, 17],
            [2, 3, 7, 6, 10, 19, 14, 18],
            [3, 0, 4, 7, 11, 16, 15, 19],
        ]

    p = _hex_order_from_cell_type(ct)
    if p is not None:
        ijk_order = _autopdex_hex_ijk_order(p)
        n_expected = (p + 1) ** 3
        if len(ijk_order) != n_expected or n_nodes < n_expected:
            raise ValueError(f"{cell_type}: inconsistent node count (expected {n_expected}, got {n_nodes}).")
        id_of = {ijk: idx for idx, ijk in enumerate(ijk_order)}

        # face is a full Lagrange quad of same order p
        face_type = _quad_type_for_order(p)
        ij_order = _vtk_lagrange_quad_ij_order(p)

        def face_from_map(map_ij_to_ijk) -> List[int]:
            return [id_of[map_ij_to_ijk(i, j)] for (i, j) in ij_order]

        faces = []
        # k=0 (0,1,2,3)
        faces.append(face_from_map(lambda i, j: (i, j, 0)))
        # k=p (4,5,6,7)
        faces.append(face_from_map(lambda i, j: (i, j, p)))
        # j=0 (0,1,5,4)
        faces.append(face_from_map(lambda i, j: (i, 0, j)))
        # i=p (1,2,6,5)
        faces.append(face_from_map(lambda i, j: (p, i, j)))
        # j=p (2,3,7,6) with i reversed
        faces.append(face_from_map(lambda i, j: (p - i, p, j)))
        # i=0 (3,0,4,7) with j reversed
        faces.append(face_from_map(lambda i, j: (0, p - i, j)))

        return face_type, faces

    raise NotImplementedError(f"3D facet extraction not implemented for '{cell_type}'.")


def _find_boundary_facets(
    mesh_info: "_MeshInfo",
    point_selector: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    cell_dim: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Return boundary facets (count==1) from an already-built _MeshInfo.

    Operates directly on ``mesh_info.cells`` so no meshio.Mesh rebuild is
    required.  Each value in the returned dict is the full node connectivity
    array of the selected facets for that facet cell type.

    ``cell_dim`` selects the dimension of the cells whose boundary is taken
    (default: the mesh's top dimension). Passing ``cell_dim`` < top dimension
    yields the boundary of a lower-dimensional field's cells (e.g. the edges of a
    membrane on a 3D block, or the endpoints of a wire in a 2D plate).
    """
    if cell_dim is None:
        cell_dim = max(
            (_cell_dim(ct) for ct, cinfo in mesh_info.cells.items()
             if int(cinfo.nodes_of_cells.shape[0]) > 0),
            default=-1,
        )
    if cell_dim not in (1, 2, 3):
        return {}

    cand_conns: Dict[str, List[np.ndarray]] = {}
    cand_corner_keys: Dict[str, List[np.ndarray]] = {}

    for cell_type, cinfo in mesh_info.cells.items():
        if _cell_dim(cell_type) != cell_dim:
            continue
        conn = np.asarray(cinfo.nodes_of_cells, dtype=int)
        if conn.size == 0:
            continue
        n_nodes = conn.shape[1]

        if cell_dim == 1:
            facet_type, templates = "vertex", [[0], [1]]
            corner_count = 1
        elif cell_dim == 2:
            facet_type, templates = _edge_templates_2d(cell_type, n_nodes)
            corner_count = 2
        else:
            facet_type, templates = _face_templates_3d(cell_type, n_nodes)
            corner_count = 3 if facet_type.startswith("triangle") else 4

        templates = np.asarray(templates, dtype=int)
        fconn = conn[:, templates].reshape(-1, templates.shape[1])
        cand_conns.setdefault(facet_type, []).append(fconn)
        cand_corner_keys.setdefault(facet_type, []).append(fconn[:, :corner_count])

    points = np.asarray(mesh_info.points)
    selected_by_type: Dict[str, np.ndarray] = {}

    for ftype, chunks in cand_conns.items():
        all_fconn = np.vstack(chunks)
        all_corners = np.vstack(cand_corner_keys[ftype])

        keep = _boundary_facet_row_indices(all_corners)

        if keep.size == 0:
            continue

        fconn = all_fconn[keep]

        if point_selector is not None:
            point_ids, inverse = np.unique(fconn.reshape(-1), return_inverse=True)
            point_mask = np.asarray(point_selector(points[point_ids]), dtype=bool)
            if point_mask.ndim != 1 or point_mask.shape[0] != point_ids.shape[0]:
                raise ValueError(
                    "point_selector must return a boolean mask with one entry per input point."
                )
            facet_mask = point_mask[inverse].reshape(fconn.shape).all(axis=1)
            fconn = fconn[facet_mask]

        if fconn.shape[0] > 0:
            selected_by_type[ftype] = fconn

    return selected_by_type


def _add_facet_domain(
    mesh: meshio.Mesh,
    domain_name: str,
    point_selector: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    cell_dim: Optional[int] = None,
) -> meshio.Mesh:
    """
    Add boundary facets to a meshio.Mesh as new cell blocks + mark them as a cell_set (domain).

    `point_selector(points)` receives an array of point coordinates with shape (n_points, gdim)
    and must return a boolean mask with length n_points.

    ``cell_dim`` selects the dimension of the cells whose boundary is taken
    (default: the mesh's top dimension); a lower value yields the boundary of a
    lower-dimensional field's cells.
    """
    if domain_name is None or domain_name == "":
        raise ValueError("domain_name must be non-empty")

    points = np.asarray(mesh.points)

    if cell_dim is None:
        cell_dim = max(_cell_dim(cb.type) for cb in mesh.cells) if mesh.cells else -1
    if cell_dim not in (1, 2, 3):
        raise ValueError(f"Unsupported topological dimension {cell_dim}. Expected 1, 2 or 3.")

    cand_conns: Dict[str, List[np.ndarray]] = {}
    cand_corner_keys: Dict[str, List[np.ndarray]] = {}

    for cb in mesh.cells:
        ct = cb.type
        if _cell_dim(ct) != cell_dim:
            continue

        conn = np.asarray(cb.data, dtype=int)
        if conn.size == 0:
            continue

        n_nodes = conn.shape[1]

        if cell_dim == 1:
            facet_type, templates = "vertex", [[0], [1]]
            corner_count = 1
        elif cell_dim == 2:
            facet_type, templates = _edge_templates_2d(ct, n_nodes)
            corner_count = 2
        else:
            facet_type, templates = _face_templates_3d(ct, n_nodes)
            if facet_type.startswith("triangle"):
                corner_count = 3
            elif facet_type.startswith("quad"):
                corner_count = 4
            else:
                raise RuntimeError(f"Unexpected facet_type '{facet_type}' in 3D.")

        templates = np.asarray(templates, dtype=int)
        fconn = conn[:, templates].reshape(-1, templates.shape[1])
        cand_conns.setdefault(facet_type, []).append(fconn)
        cand_corner_keys.setdefault(facet_type, []).append(fconn[:, :corner_count])

    selected_by_type: Dict[str, np.ndarray] = {}

    for ftype, chunks in cand_conns.items():
        all_fconn = np.vstack(chunks)
        all_corners = np.vstack(cand_corner_keys[ftype])

        keep = _boundary_facet_row_indices(all_corners)

        if keep.size == 0:
            continue

        fconn = all_fconn[keep]

        if point_selector is not None:
            point_ids, inverse = np.unique(fconn.reshape(-1), return_inverse=True)
            point_mask = np.asarray(point_selector(points[point_ids]), dtype=bool)
            if point_mask.ndim != 1 or point_mask.shape[0] != point_ids.shape[0]:
                raise ValueError("point_selector must return a boolean mask with one entry per input point.")
            facet_mask = point_mask[inverse].reshape(fconn.shape).all(axis=1)
            fconn = fconn[facet_mask]

        if fconn.shape[0] > 0:
            selected_by_type[ftype] = fconn

    if not selected_by_type:
        # Ensure the domain exists but empty
        if mesh.cell_sets is None:
            mesh.cell_sets = {}
        n_blocks_old = len(mesh.cells)
        mesh.cell_sets[domain_name] = [np.array([], dtype=int) for _ in range(n_blocks_old)]
        return mesh

    # Prepare cell_sets and cell_data for appending new blocks
    if mesh.cell_sets is None:
        mesh.cell_sets = {}

    # normalize existing sets: each must have length == len(mesh.cells)
    n_blocks_old = len(mesh.cells)
    for sname, lst in mesh.cell_sets.items():
        if len(lst) < n_blocks_old:
            mesh.cell_sets[sname] = list(lst) + [np.array([], dtype=int) for _ in range(n_blocks_old - len(lst))]

    if domain_name not in mesh.cell_sets:
        mesh.cell_sets[domain_name] = [np.array([], dtype=int) for _ in range(n_blocks_old)]

    if mesh.cell_data is None:
        mesh.cell_data = {}

    # normalize cell_data: each entry is a list aligned with mesh.cells
    for key, lst in mesh.cell_data.items():
        if len(lst) < n_blocks_old:
            # pad with empty arrays (best effort)
            mesh.cell_data[key] = list(lst) + [np.array([], dtype=lst[0].dtype if len(lst) else float)
                                               for _ in range(n_blocks_old - len(lst))]

    # Append new facet blocks (one block per facet type)
    for ftype, fconn in selected_by_type.items():
        fconn = np.asarray(fconn, dtype=int)

        # append CellBlock
        mesh.cells.append(meshio.CellBlock(ftype, fconn))

        # extend ALL sets by one empty array, then fill for domain_name
        for sname, lst in mesh.cell_sets.items():
            lst.append(np.array([], dtype=int))
        mesh.cell_sets[domain_name][-1] = np.arange(fconn.shape[0], dtype=int)

        # extend cell_data lists by one filler array (zeros) for consistency
        for dkey, lst in mesh.cell_data.items():
            # choose dtype-preserving fill
            dtype = lst[0].dtype if len(lst) > 0 and hasattr(lst[0], "dtype") else float
            lst.append(np.zeros((fconn.shape[0],), dtype=dtype))

    return mesh


_BASE_DIM = {
    "vertex": 0,
    "line": 1,
    "triangle": 2,
    "quad": 2,
    "tetra": 3,
    "hex": 3,
}


_PATTERNS = {
    # 2D: edges
    "triangle": np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int8),
    "quad":     np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int8),

    # 3D: faces
    "tetra":    np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int8),
    "hex":      np.array([
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [0, 1, 5, 4],
                    [1, 2, 6, 5],
                    [2, 3, 7, 6],
                    [3, 0, 4, 7],
                ], dtype=np.int8),
}


_EDGE_PATTERNS = {
    "triangle": np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int8),
    "quad":     np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int8),
    "tetra":    np.array([[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]], dtype=np.int8),
    "hex":      np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ], dtype=np.int8),
}


_CORNER_COUNT = {
    "vertex": 1,
    "line": 2,
    "triangle": 3,
    "quad": 4,
    "tetra": 4,
    "hex": 8,
}


def _canonical_base_type(raw: str) -> str:
    r = raw.lower()
    if r.startswith("vertex"):
        return "vertex"
    if r.startswith("line"):
        return "line"
    if r.startswith("triangle") or r.startswith("vtk_lagrange_triangle"):
        return "triangle"
    if r.startswith("quad") or r.startswith("vtk_lagrange_quadrilateral"):
        return "quad"
    if r == "tet" or r.startswith("tetrahedron") or r.startswith("tetra") or r.startswith("vtk_lagrange_tetrahedron"):
        return "tetra"
    if r.startswith("hexahedron") or r.startswith("hex") or r.startswith("vtk_lagrange_hexahedron"):
        return "hex"
    return r


def _int_row_keys(a: np.ndarray) -> np.ndarray:
    """Return a 1D row-key view for a contiguous 2D integer array."""
    a = np.ascontiguousarray(a)
    if a.ndim != 2:
        raise ValueError("a must be a 2D array.")
    return a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))).reshape(-1)


def _unique_rows_int(a: np.ndarray):
    """Fast unique for 2D arrays (unique rows). Returns (unique_rows, inverse).

    ``unique_rows[inverse] == a`` always holds. With the native extension the
    unique rows are returned in O(n) **first-occurrence** order; the pure-NumPy
    fallback returns them sorted. Either ordering is acceptable because the
    entity numbering is produced by a single ``_build_mesh_info`` pass: Pass 2
    orders chunks volume-cells-first, so the numbering is fixed by the volume
    cells and stays stable when boundary facets are folded in (see
    ``SimState._ensure_all_boundary_mesh_info``). The numbering is deterministic
    for a given extension availability; a checkpoint should therefore be reloaded
    in an environment with the same native/fallback configuration.
    """
    a = np.ascontiguousarray(a)
    if _HAS_NATIVE_TOPO and a.ndim == 2 and a.shape[1] in (2, 3, 4):
        # Node ids fit in 32 bits; reinterpret int32 input as uint32 for free and
        # otherwise cast. The unique rows are returned as uint32 so the Pass 3
        # native lookup can consume them without another copy.
        if a.dtype == np.uint32:
            a32 = a
        elif a.dtype == np.int32:
            a32 = a.view(np.uint32)
        else:
            a32 = np.ascontiguousarray(a.astype(np.uint32))
        n, w = a32.shape
        inv_fo = np.empty(n, dtype=np.int64)
        uniq_fo = np.empty_like(a32)
        nu = _topo_lib.factorize_entity_rows(
            _topo_ffi.cast("uint32_t*", a32.ctypes.data), n, w,
            _topo_ffi.cast("int64_t*", inv_fo.ctypes.data),
            _topo_ffi.cast("uint32_t*", uniq_fo.ctypes.data))
        if nu < 0:
            raise MemoryError("_topo_hash: allocation failed")
        # First-occurrence order is returned directly (O(n), no sort): the entity
        # numbering is defined by a single _build_mesh_info pass and consumers no
        # longer require a cross-build canonical order.
        return uniq_fo[:nu], inv_fo
    # NumPy fallback (sorted): keeps searchsorted-based Pass 3 lookup valid.
    v = _int_row_keys(a)
    _, idx, inv = np.unique(v, return_index=True, return_inverse=True)
    return a[idx], inv


def _boundary_facet_row_indices(corner_keys: np.ndarray) -> np.ndarray:
    """Row indices of facets whose (sorted) corner key occurs exactly once.

    A facet shared by two cells appears twice and is interior; a facet that
    appears once lies on the boundary. With the native extension this is an O(n)
    hash (deduplicate, then count occurrences via ``bincount``); otherwise it
    falls back to ``np.unique(..., return_counts=True)``.
    """
    key = np.sort(np.ascontiguousarray(corner_keys), axis=1)
    n = int(key.shape[0])
    if _HAS_NATIVE_TOPO and key.ndim == 2 and key.shape[1] in (2, 3, 4):
        uniq, inv = _unique_rows_int(key)
        counts = np.bincount(inv, minlength=uniq.shape[0])
        # Representative row per unique facet; for count==1 it is the only row.
        rep = np.empty(uniq.shape[0], dtype=np.int64)
        rep[inv] = np.arange(n, dtype=np.int64)
        return rep[counts == 1]
    v = key.view(np.dtype((np.void, key.dtype.itemsize * key.shape[1]))).reshape(-1)
    _, first_idx, counts = np.unique(v, return_index=True, return_counts=True)
    return first_idx[counts == 1]


def _line_corner_nodes(points: np.ndarray, line_conn: np.ndarray) -> np.ndarray:
    """Return the two endpoint node ids for linear or high-order line cells."""
    line_conn = np.asarray(line_conn, dtype=int)
    if line_conn.ndim != 2 or line_conn.shape[1] < 2:
        raise ValueError("line_conn must have shape (n_lines, n_nodes>=2).")
    if line_conn.shape[1] == 2:
        return line_conn[:, :2]

    pts = np.asarray(points, dtype=float)
    coords = pts[line_conn]
    pairs = [(i, j) for i in range(line_conn.shape[1]) for j in range(i + 1, line_conn.shape[1])]
    d2 = np.stack(
        [np.sum((coords[:, i, :] - coords[:, j, :]) ** 2, axis=1) for i, j in pairs],
        axis=1,
    )
    best = np.argmax(d2, axis=1)
    corners = np.empty((line_conn.shape[0], 2), dtype=int)
    for pair_id, (i, j) in enumerate(pairs):
        mask = best == pair_id
        corners[mask, 0] = line_conn[mask, i]
        corners[mask, 1] = line_conn[mask, j]
    return corners


def _celltype_with_count(base: str, n: int) -> str:
    if base == "line":
        return "line" if n == 2 else f"line{n}"
    if base == "triangle":
        return "triangle" if n == 3 else f"triangle{n}"
    if base == "quad":
        return "quad" if n == 4 else f"quad{n}"
    raise ValueError(f"Unsupported base for cell type key: {base}")


def _infer_p_triangle(n_nodes: int) -> Optional[int]:
    # n_nodes = (p+1)(p+2)/2
    disc = 1 + 8 * n_nodes
    s = int(math.isqrt(disc))
    if s * s != disc:
        return None
    num = -3 + s
    if num % 2 != 0:
        return None
    p = num // 2
    return p if (p + 1) * (p + 2) // 2 == n_nodes else None


def _infer_p_tetra(n_nodes: int, p_max: int = 20) -> Optional[int]:
    # n_nodes = (p+1)(p+2)(p+3)/6
    for p in range(1, p_max + 1):
        if (p + 1) * (p + 2) * (p + 3) // 6 == n_nodes:
            return p
    return None


def _facet_type_key_from_parent(base: str, n_nodes_per_cell: int, raw_cell_type: str) -> Optional[str]:
    """Infer face/edge cell type key for facets_of_cells (2D: edges, 3D: faces)."""
    if base in ("vertex", "line"):
        return None

    if base == "triangle":
        p = _infer_p_triangle(n_nodes_per_cell)
        if p is None:
            warnings.warn(
                f"Could not infer polynomial order for '{raw_cell_type}' (nodes_per_cell={n_nodes_per_cell}). "
                f"Falling back to linear facet type 'line'.",
                RuntimeWarning,
                stacklevel=3,
            )
            return "line"
        return _celltype_with_count("line", p + 1)

    if base == "quad":
        if n_nodes_per_cell == 4:
            return "line"
        if n_nodes_per_cell in (8, 9):
            return "line3"
        s = int(round(math.isqrt(n_nodes_per_cell)))
        if s * s == n_nodes_per_cell and s >= 2:
            return _celltype_with_count("line", s)
        warnings.warn(
            f"Could not infer edge type for '{raw_cell_type}' (nodes_per_cell={n_nodes_per_cell}). "
            f"Falling back to linear facet type 'line'.",
            RuntimeWarning,
            stacklevel=3,
        )
        return "line"

    if base == "tetra":
        if n_nodes_per_cell == 4:
            return "triangle"
        p = _infer_p_tetra(n_nodes_per_cell)
        if p is None:
            warnings.warn(
                f"Could not infer polynomial order for '{raw_cell_type}' (nodes_per_cell={n_nodes_per_cell}). "
                f"Falling back to linear facet type 'triangle'.",
                RuntimeWarning,
                stacklevel=3,
            )
            return "triangle"
        face_nodes = (p + 1) * (p + 2) // 2
        return _celltype_with_count("triangle", face_nodes)

    if base == "hex":
        if n_nodes_per_cell == 8:
            return "quad"
        if n_nodes_per_cell == 20:
            return "quad8"
        if n_nodes_per_cell == 27:
            return "quad9"
        c = round(n_nodes_per_cell ** (1.0 / 3.0))
        if int(c) ** 3 == n_nodes_per_cell and c >= 2:
            face_nodes = int(c) ** 2
            return _celltype_with_count("quad", face_nodes)
        warnings.warn(
            f"Could not infer face type for '{raw_cell_type}' (nodes_per_cell={n_nodes_per_cell}). "
            f"Falling back to linear facet type 'quad'.",
            RuntimeWarning,
            stacklevel=3,
        )
        return "quad"

    return None


def _edge_type_key_from_parent(base: str, n_nodes_per_cell: int, raw_cell_type: str) -> Optional[str]:
    """Infer edge cell type key for edges_of_cells (3D edges)."""
    if base in ("vertex", "line"):
        return None

    if base == "tetra":
        if n_nodes_per_cell == 4:
            return "line"
        p = _infer_p_tetra(n_nodes_per_cell)
        if p is None:
            warnings.warn(
                f"Could not infer polynomial order for '{raw_cell_type}' (nodes_per_cell={n_nodes_per_cell}). "
                f"Falling back to linear edge type 'line'.",
                RuntimeWarning,
                stacklevel=3,
            )
            return "line"
        return _celltype_with_count("line", p + 1)

    if base == "hex":
        if n_nodes_per_cell == 8:
            return "line"
        if n_nodes_per_cell in (20, 27):
            return "line3"
        c = round(n_nodes_per_cell ** (1.0 / 3.0))
        if int(c) ** 3 == n_nodes_per_cell and c >= 2:
            return _celltype_with_count("line", int(c))
        warnings.warn(
            f"Could not infer edge type for '{raw_cell_type}' (nodes_per_cell={n_nodes_per_cell}). "
            f"Falling back to linear edge type 'line'.",
            RuntimeWarning,
            stacklevel=3,
        )
        return "line"

    # In 2D, edges are already facets_of_cells.
    return None


def _surface_corner_count(surface_base: str) -> int:
    if surface_base == "line":
        return 2
    if surface_base == "triangle":
        return 3
    if surface_base == "quad":
        return 4
    raise ValueError(f"Unsupported surface base type '{surface_base}'.")
