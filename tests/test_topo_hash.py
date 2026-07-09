"""Tests for the native O(n) topology dedup/lookup extension and its migration.

The direct-C tests are skipped if the native extension was not built; the
end-to-end tests exercise both the native and the NumPy fallback path.
"""

import numpy as np
import pytest
import meshio

from autopdex import mesher
from autopdex.sim_state import mesh_info as mi
from autopdex.sim_state import mesh_topology as mt
from autopdex.sim_state.mesh_info import _build_mesh_info

_native = pytest.importorskip(
    "autopdex.sim_state._native._topo_hash", reason="native _topo_hash extension not built"
)
lib = _native.lib
ffi = _native.ffi


# --- thin Python wrappers around the C API ----------------------------------

def _factorize(rows):
    rows = np.ascontiguousarray(rows, dtype=np.uint32)
    n, w = rows.shape
    inv = np.empty(n, dtype=np.int64)
    uniq = np.empty_like(rows)
    nu = lib.factorize_entity_rows(
        ffi.cast("uint32_t*", rows.ctypes.data), n, w,
        ffi.cast("int64_t*", inv.ctypes.data),
        ffi.cast("uint32_t*", uniq.ctypes.data))
    assert nu >= 0
    return uniq[:nu].copy(), inv, int(nu)


def _lookup(entities, queries):
    ent = np.ascontiguousarray(entities, dtype=np.uint32)
    qry = np.ascontiguousarray(queries, dtype=np.uint32)
    out = np.empty(len(qry), dtype=np.int64)
    miss = lib.lookup_entity_rows(
        ffi.cast("uint32_t*", ent.ctypes.data), len(ent), ent.shape[1],
        ffi.cast("uint32_t*", qry.ctypes.data), len(qry),
        ffi.cast("int64_t*", out.ctypes.data))
    assert miss >= 0
    return out, int(miss)


def _canonical_rows(rng, n, width, vocab):
    rows = rng.integers(0, vocab, size=(n, width), dtype=np.uint32)
    return np.sort(rows, axis=1)


# --- factorize_entity_rows ---------------------------------------------------

@pytest.mark.parametrize("width", [2, 3, 4])
def test_factorize_bijective(width):
    rng = np.random.default_rng(width)
    rows = _canonical_rows(rng, 2000, width, vocab=40)
    uniq, inv, nu = _factorize(rows)

    # inverse reconstructs the input exactly
    assert np.array_equal(uniq[inv], rows)

    # count and uniqueness match NumPy ground truth
    ref = np.unique(rows, axis=0)
    assert nu == ref.shape[0]
    assert np.unique(uniq, axis=0).shape[0] == nu

    # ids are in first-occurrence order: first appearance of each id is increasing
    first_seen = {}
    for i, e in enumerate(inv):
        first_seen.setdefault(int(e), i)
    order = [first_seen[k] for k in range(nu)]
    assert order == sorted(order)
    assert list(first_seen.keys()) == list(range(nu))


def test_factorize_empty():
    uniq, inv, nu = _factorize(np.empty((0, 3), dtype=np.uint32))
    assert nu == 0
    assert inv.shape == (0,)


# --- lookup_entity_rows ------------------------------------------------------

@pytest.mark.parametrize("width", [2, 3, 4])
def test_lookup_found_and_missing(width):
    rng = np.random.default_rng(100 + width)
    entities = np.unique(_canonical_rows(rng, 500, width, vocab=50), axis=0)

    # every entity is found at its own index
    out, miss = _lookup(entities, entities)
    assert miss == 0
    assert np.array_equal(out, np.arange(len(entities)))

    # queries: mix of present and guaranteed-absent rows
    absent = np.full((20, width), 9999, dtype=np.uint32)  # outside vocab
    queries = np.concatenate([entities[:10], absent], axis=0)
    out, miss = _lookup(entities, queries)
    assert miss == 20
    assert np.array_equal(out[:10], np.arange(10))
    assert np.all(out[10:] == -1)


def test_lookup_empty_table():
    out, miss = _lookup(np.empty((0, 2), dtype=np.uint32),
                        np.array([[1, 2], [3, 4]], dtype=np.uint32))
    assert miss == 2
    assert np.all(out == -1)


# --- mixed keys (edges + facets of the same key, processed jointly) ----------

def test_factorize_mixed_chunks_consistent():
    # Simulate "line" edges coming from two different cell blocks that share rows.
    edges_a = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.uint32)
    edges_b = np.array([[2, 3], [3, 4], [0, 1]], dtype=np.uint32)
    rows = np.concatenate([edges_a, edges_b], axis=0)
    uniq, inv, nu = _factorize(rows)

    # shared rows must receive identical ids across the two chunks
    assert inv[0] == inv[5]  # [0,1]
    assert inv[2] == inv[3]  # [2,3]
    assert np.array_equal(uniq[inv], rows)


# --- end-to-end: native vs fallback over _build_mesh_info --------------------

def _mixed_tri_line_mesh():
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)
    return meshio.Mesh(points=points, cells=[("triangle", triangles), ("line", lines)])


def _build_with_native(flag, monkeypatch):
    monkeypatch.setattr(mt, "_HAS_NATIVE_TOPO", flag)
    monkeypatch.setattr(mi, "_HAS_NATIVE_TOPO", flag)
    return _build_mesh_info(_mixed_tri_line_mesh())


def test_entity_id_internally_consistent(monkeypatch):
    info = _build_with_native(True, monkeypatch)

    tri = info.cells["triangle"]
    line = info.cells["line"]

    # id -> sorted edge node-set, derived from triangle facets
    facet_ids = np.asarray(tri.facets_of_cells["line"], dtype=int)
    pat = np.array([[0, 1], [1, 2], [2, 0]])
    tri_nodes = np.asarray(tri.nodes_of_cells, dtype=int)[:, :3]
    id_to_nodes = {}
    for c in range(tri_nodes.shape[0]):
        for e in range(3):
            pair = tuple(np.sort(tri_nodes[c, pat[e]]))
            id_to_nodes.setdefault(int(facet_ids[c, e]), pair)

    # line entity_id must map to the same node-set
    line_nodes = np.sort(np.asarray(line.nodes_of_cells, dtype=int)[:, :2], axis=1)
    line_ids = np.asarray(line.entity_id, dtype=int)
    assert np.all(line_ids >= 0)
    for i, eid in enumerate(line_ids):
        assert id_to_nodes[int(eid)] == tuple(line_nodes[i])


def test_entity_sign_preserved_native_vs_fallback(monkeypatch):
    info_native = _build_with_native(True, monkeypatch)
    info_fallback = _build_with_native(False, monkeypatch)

    def sign_by_nodeset(info):
        line = info.cells["line"]
        nodes = np.sort(np.asarray(line.nodes_of_cells, dtype=int)[:, :2], axis=1)
        signs = np.asarray(line.entity_sign, dtype=float)
        return {tuple(nodes[i]): float(signs[i]) for i in range(len(signs))}

    nat = sign_by_nodeset(info_native)
    fb = sign_by_nodeset(info_fallback)
    assert nat.keys() == fb.keys()
    for k in nat:
        assert nat[k] == pytest.approx(fb[k])


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_factorize_u64_rows_matches_numpy_dedup(width):
    rng = np.random.default_rng(11 + width)
    base = rng.integers(0, 40, size=(50, width), dtype=np.uint64)
    rows = np.ascontiguousarray(base[rng.integers(0, 50, size=600)])
    n = len(rows)
    inv = np.empty(n, dtype=np.int64)
    first = np.empty(n, dtype=np.int64)
    nu = lib.factorize_u64_rows(
        ffi.cast("uint64_t*", rows.ctypes.data), n, width,
        ffi.cast("int64_t*", inv.ctypes.data),
        ffi.cast("int64_t*", first.ctypes.data))
    assert nu == np.unique(rows, axis=0).shape[0]
    # rows[first] are the unique rows in first-occurrence order; inv reconstructs.
    assert np.array_equal(rows[first[:nu]][inv], rows)
    # first[id] is the first row index producing that id
    seen = {}
    for i in range(n):
        seen.setdefault(int(inv[i]), i)
    assert all(first[k] == seen[k] for k in range(nu))


def test_point_dedup_merges_minus_zero():
    """Coincident nodes differing only by -0.0/+0.0 must still merge."""
    pts = np.array([[0.0, 0.0], [-0.0, 0.0], [1.0, 2.0], [0.0, -0.0]], dtype=float)
    mesh = meshio.Mesh(points=pts, cells=[("line", np.array([[0, 2], [1, 3]]))])
    from autopdex.sim_state.mesh_topology import _deduplicate_mesh_points_exact
    dedup = _deduplicate_mesh_points_exact(mesh)
    # rows 0,1,3 are all the origin -> 2 unique points remain
    assert np.asarray(dedup.points).shape[0] == 2


def test_volume_numbering_stable_when_boundary_facets_added():
    """Entity numbering must be fixed by the volume cells alone.

    The architecture relies on this: boundary-facet cells are appended after the
    volume cells and only reference already-seen entities, so folding them into
    the build must not renumber any volume edge/face. This is what lets strong
    BCs (which trigger such a fold, possibly after the FE is built) share the
    volume's entity numbering without a canonical (sorted) ordering.
    """
    # 8-hex block; appended boundary quad facets are subsets of existing faces.
    n = 3
    coords, elems = mesher.structured_mesh(
        (n, n, n),
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
        "brick", order=1,
    )
    elems = np.asarray(elems)
    vol = meshio.Mesh(points=np.asarray(coords), cells=[("hexahedron", elems)])
    info_vol = _build_mesh_info(vol)

    # Same volume cells first, plus a block of boundary quad facets afterwards.
    faces = elems[:, np.array([[0, 1, 2, 3], [4, 5, 6, 7]])].reshape(-1, 4)
    with_facets = meshio.Mesh(
        points=np.asarray(coords),
        cells=[("hexahedron", elems), ("quad", faces)],
    )
    info_facets = _build_mesh_info(with_facets)

    for key, a in info_vol.cells["hexahedron"].edges_of_cells.items():
        b = info_facets.cells["hexahedron"].edges_of_cells[key]
        assert np.array_equal(np.asarray(a, dtype=int), np.asarray(b, dtype=int))
    for key, a in info_vol.cells["hexahedron"].facets_of_cells.items():
        b = info_facets.cells["hexahedron"].facets_of_cells[key]
        assert np.array_equal(np.asarray(a, dtype=int), np.asarray(b, dtype=int))


def test_build_mesh_info_deterministic(monkeypatch):
    a = _build_with_native(True, monkeypatch)
    b = _build_with_native(True, monkeypatch)
    fa = np.asarray(a.cells["triangle"].facets_of_cells["line"], dtype=int)
    fb = np.asarray(b.cells["triangle"].facets_of_cells["line"], dtype=int)
    assert np.array_equal(fa, fb)
    ia = np.asarray(a.cells["line"].entity_id, dtype=int)
    ib = np.asarray(b.cells["line"].entity_id, dtype=int)
    assert np.array_equal(ia, ib)
