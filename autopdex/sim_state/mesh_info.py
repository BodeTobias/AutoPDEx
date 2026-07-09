# autopdex/sim_state/mesh_info.py
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

"""Topology/geometry model (_MeshInfo), including construction, domains, and quadrature helpers."""

import warnings
from fnmatch import fnmatch
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import jax
import jax.numpy as jnp

from autopdex import seeder, spaces

from .mesh_topology import _BASE_DIM, _CORNER_COUNT, _EDGE_PATTERNS, _PATTERNS, _canonical_base_type, _edge_type_key_from_parent, _facet_type_key_from_parent, _hex_order_from_cell_type, _int_row_keys, _line_corner_nodes, _quad_order_from_cell_type, _unique_rows_int
from .mesh_topology import _HAS_NATIVE_TOPO, _topo_lib, _topo_ffi

@dataclass
class _DomainInfo:
    cell_indices: jnp.ndarray


@dataclass
class _CellInfo:
    facets_of_cells: Optional[Dict[str, jnp.ndarray]]
    edges_of_cells: Optional[Dict[str, jnp.ndarray]]
    nodes_of_cells: jnp.ndarray
    # For sub-dimensional cell blocks:
    #   - line*    : global edge id in the "line*" entity space
    #   - triangle*/quad* : global face id in the "triangle*"/"quad*" entity space
    entity_id: Optional[jnp.ndarray]
    entity_sign: Optional[jnp.ndarray]


@dataclass
class _MeshInfo:
    points: jnp.ndarray
    domains: dict[str, dict[str, _DomainInfo]]
    cells: Dict[str, _CellInfo]


def _build_domains(mesh):
    # ranges_by_type: cell_type -> list[(g0,g1,loff)]
    # g0/g1 are "global" ONLY within the element DIMENSION (1D/2D/3D)
    ranges_by_type = {}
    dim_cursor = {0: 0, 1: 0, 2: 0, 3: 0}
    type_cursor = {}

    for block in mesh.cells:
        ct = block.type
        n = len(block.data)

        base = _canonical_base_type(ct)
        dim = _BASE_DIM.get(base, None)
        if dim is None:
            continue

        g0 = dim_cursor[dim]
        g1 = g0 + n
        dim_cursor[dim] = g1

        loff = type_cursor.get(ct, 0)
        type_cursor[ct] = loff + n

        ranges_by_type.setdefault(ct, []).append((g0, g1, loff))

    def _keep_cell_set(set_name: str) -> bool:
        """
        Decides whether a cell_set should be interpreted as an FE domain.

        Allow:
          - regular (non-gmsh-internal) sets
          - gmsh:physical
          - gmsh:geometrical

        Exclude:
          - gmsh:bounding_entities
          - other unknown gmsh:* metadata
        """
        if set_name == "gmsh:bounding_entities":
            return False

        if set_name.startswith("gmsh:"):
            return set_name in {"gmsh:physical", "gmsh:geometrical"}

        return True

    domains = {}
    for set_name, cell_set in mesh.cell_sets_dict.items():
        if not _keep_cell_set(set_name):
            continue

        sub = {}
        for cell_type, indices in cell_set.items():
            if cell_type not in mesh.cells_dict:
                continue

            idx = np.asarray(indices, dtype=np.int64).ravel()
            if idx.size == 0:
                continue

            # Negative values are not valid cell indices.
            # Such entries typically encode oriented/topological Gmsh
            # information and are not FE domains.
            if np.any(idx < 0):
                continue

            n_local = int(mesh.cells_dict[cell_type].shape[0])

            # Case 1: already local for mesh.cells_dict[cell_type]
            if idx.min() >= 0 and idx.max() < n_local:
                mapped = idx
            else:
                # Case 2: global within the dimension -> remap with searchsorted
                ranges = ranges_by_type.get(cell_type, [])
                starts = np.array([g0   for g0, _,  _    in ranges], dtype=np.int64)
                ends   = np.array([g1   for _,  g1, _    in ranges], dtype=np.int64)
                loffs  = np.array([loff for _,  _,  loff in ranges], dtype=np.int64)

                block_idx = np.searchsorted(starts, idx, side="right") - 1
                safe_bi   = np.clip(block_idx, 0, max(len(ranges) - 1, 0))
                valid     = (block_idx >= 0) & (idx < ends[safe_bi])
                if not valid.all():
                    bad = idx[~valid].tolist()
                    raise ValueError(
                        f"Cannot map gid(s)={bad} for type '{cell_type}' in set '{set_name}'. "
                        f"Known blocks for this type: {ranges}"
                    )

                mapped = (idx - starts[block_idx]) + loffs[block_idx]

                if mapped.min() < 0 or mapped.max() >= n_local:
                    raise ValueError(
                        f"Mapped ids OOB for set '{set_name}', type '{cell_type}': "
                        f"[{int(mapped.min())},{int(mapped.max())}] vs n_local={n_local}"
                    )

            sub[cell_type] = _DomainInfo(
                cell_indices=jnp.asarray(np.unique(mapped), dtype=jnp.int64)
            )

        # Only keep non-empty domains
        if sub:
            domains[set_name] = sub

    return domains


def _build_mesh_info(mesh) -> _MeshInfo:
    points = jnp.asarray(mesh.points)

    # Domains (cell sets)
    domains = _build_domains(mesh)

    # --- First pass: store connectivity + prepare registries ------------------
    facets_registry: Dict[str, list[np.ndarray]] = {}
    facets_sign_registry: Dict[str, list[np.ndarray]] = {}
    edges_registry: Dict[str, list[np.ndarray]] = {}

    facets_meta: Dict[str, list[tuple[str, int, int, int]]] = {}
    edges_meta: Dict[str, list[tuple[str, int, int, int]]] = {}

    tmp: Dict[str, dict] = {}

    for cell_type, conn in mesh.cells_dict.items():
        nodes_of_cells_np = np.asarray(conn, dtype=int)
        n_cells = int(nodes_of_cells_np.shape[0])

        base = _canonical_base_type(cell_type)

        tmp[cell_type] = {
            "nodes_of_cells": jnp.asarray(nodes_of_cells_np),
            "facets": {},
            "edges": {},
            "entity_id": None,  # filled later
            "entity_sign": None,
        }

        # Only topological extraction from volume cells
        if base in ("vertex", "line"):
            continue

        if base not in _CORNER_COUNT:
            warnings.warn(
                f"Could not interpret cell type '{cell_type}' (base='{base}'). "
                f"Facets/edges will not be available for this type.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        nodes64 = nodes_of_cells_np.astype(int, copy=False)
        n_corner = _CORNER_COUNT[base]
        if nodes64.shape[1] < n_corner:
            raise ValueError(
                f"{cell_type}: expected at least {n_corner} nodes per cell (corner nodes), got {nodes64.shape[1]}."
            )

        # int32 is enough for node ids and lets _unique_rows_int reinterpret the
        # facet/edge occurrence rows as uint32 for the native hash without a copy.
        corner_nodes = nodes64[:, :n_corner].astype(np.int32)

        # Facets (2D: edges, 3D: faces)
        if base in _PATTERNS:
            facet_key = _facet_type_key_from_parent(base, nodes64.shape[1], cell_type)
            if facet_key is None:
                warnings.warn(
                    f"Could not determine facet key for '{cell_type}'. Facets will be missing for this type.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                pat = _PATTERNS[base]
                k = int(pat.shape[1])
                n_per_cell = int(pat.shape[0])
                local_facets = corner_nodes[:, pat]
                occ = local_facets.reshape(-1, k)
                if k == 2:
                    signs = np.where(local_facets[:, :, 0] < local_facets[:, :, 1], 1.0, -1.0)
                    facets_sign_registry.setdefault(facet_key, []).append(signs.reshape(-1))
                occ = np.sort(occ, axis=1)

                facets_registry.setdefault(facet_key, []).append(occ)
                facets_meta.setdefault(facet_key, []).append((cell_type, n_cells, n_per_cell, occ.shape[0]))
        else:
            warnings.warn(
                f"Could not identify facets for cell type '{cell_type}' (base='{base}').",
                RuntimeWarning,
                stacklevel=2,
            )

        # 3D edges
        if base in ("tetra", "hex"):
            if base not in _EDGE_PATTERNS:
                warnings.warn(
                    f"Could not identify edges for cell type '{cell_type}' (base='{base}').",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                edge_key = _edge_type_key_from_parent(base, nodes64.shape[1], cell_type)
                if edge_key is None:
                    warnings.warn(
                        f"Could not determine edge key for '{cell_type}'. Edges will be missing for this type.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    epat = _EDGE_PATTERNS[base]
                    n_per_cell = int(epat.shape[0])
                    occ = corner_nodes[:, epat].reshape(-1, 2)
                    occ = np.sort(occ, axis=1)

                    edges_registry.setdefault(edge_key, []).append(occ)
                    edges_meta.setdefault(edge_key, []).append((cell_type, n_cells, n_per_cell, occ.shape[0]))

    # --- Second pass: global unique per key and remap back --------------------
    # A key such as "line" can occur both as a facet of surface cells and as an
    # edge of volume cells. Number such entities once per key, otherwise surface
    # connectivity and volume edge DOFs can silently refer to different IDs.
    entity_nodes_by_key: Dict[str, np.ndarray] = {}
    entity_row_keys_by_key: Dict[str, np.ndarray] = {}
    entity_sign_by_key: Dict[str, np.ndarray] = {}

    for key in sorted(set(facets_registry.keys()) | set(edges_registry.keys())):
        # Collect facet/edge occurrence chunks tagged with the topological
        # dimension of the producing cell, and order higher-dimensional (volume)
        # chunks first. This makes the first-occurrence entity numbering a
        # function of the volume cells alone: appended boundary-facet cells only
        # reference already-seen entities, so folding them in never renumbers a
        # volume edge/face. Strong-BC projections therefore share the volume's
        # numbering without needing a canonical (sorted) order.
        items = []  # (dim, kind, meta_idx, chunk, n_rows)
        for i, ch in enumerate(facets_registry.get(key, [])):
            meta = facets_meta[key][i]
            items.append((_BASE_DIM.get(_canonical_base_type(meta[0]), 0), "f", i, ch, meta[-1]))
        for i, ch in enumerate(edges_registry.get(key, [])):
            meta = edges_meta[key][i]
            items.append((_BASE_DIM.get(_canonical_base_type(meta[0]), 0), "e", i, ch, meta[-1]))
        if not items:
            continue

        order = sorted(range(len(items)), key=lambda j: -items[j][0])  # stable: volume first
        all_occ = (np.concatenate([items[j][3] for j in order], axis=0)
                   if len(order) > 1 else items[order[0]][3])
        uniq, inv = _unique_rows_int(all_occ)
        entity_nodes_by_key[key] = uniq
        entity_row_keys_by_key[key] = _int_row_keys(uniq)

        # Slice the inverse back to each chunk (in the concatenated order).
        inv_f: Dict[int, np.ndarray] = {}
        inv_e: Dict[int, np.ndarray] = {}
        cursor = 0
        for j in order:
            _, kind, midx, _, n_rows = items[j]
            (inv_f if kind == "f" else inv_e)[midx] = inv[cursor:cursor + n_rows]
            cursor += n_rows

        if key in facets_sign_registry:
            facet_inv = (np.concatenate([inv_f[i] for i in range(len(facets_meta[key]))], axis=0)
                         if facets_meta.get(key) else np.empty(0, dtype=inv.dtype))
            facet_signs = np.concatenate(facets_sign_registry[key], axis=0)
            entity_ids, first = np.unique(facet_inv, return_index=True)
            signs = np.ones((uniq.shape[0],), dtype=float)
            signs[entity_ids] = facet_signs[first]
            entity_sign_by_key[key] = signs

        if key in facets_registry:
            for i, (cell_type, n_cells, n_per_cell, n_rows) in enumerate(facets_meta[key]):
                ids = inv_f[i].reshape(n_cells, n_per_cell).astype(int, copy=False)
                tmp[cell_type]["facets"][key] = jnp.asarray(ids)

        if key in edges_registry:
            for i, (cell_type, n_cells, n_per_cell, n_rows) in enumerate(edges_meta[key]):
                ids = inv_e[i].reshape(n_cells, n_per_cell).astype(int, copy=False)
                tmp[cell_type]["edges"][key] = jnp.asarray(ids)


    # --- Third pass: assign entity_id for sub-dimensional cell blocks ----------
    def _assign_entity_id_for_cell_block(cell_type: str, nodes_of_cells: jnp.ndarray) -> Optional[jnp.ndarray]:
        base = _canonical_base_type(cell_type)
        key = cell_type.lower()

        if key not in entity_nodes_by_key:
            return None

        nodes_np = np.asarray(nodes_of_cells, dtype=int)

        if base == "line":
            if nodes_np.shape[1] < 2:
                return None
            occ = _line_corner_nodes(np.asarray(points), nodes_np)
            occ = np.sort(occ, axis=1)

        elif base == "triangle":
            if nodes_np.shape[1] < 3:
                return None
            occ = nodes_np[:, :3]
            occ = np.sort(occ, axis=1)

        elif base == "quad":
            if nodes_np.shape[1] < 4:
                return None
            occ = nodes_np[:, :4]
            occ = np.sort(occ, axis=1)

        else:
            return None

        # Match the int32 dtype of the entity table built in Pass 2 so the native
        # lookup reinterprets without a copy and the fallback void-key comparison
        # uses a consistent width.
        occ = np.ascontiguousarray(occ, dtype=np.int32)
        ids = np.full((occ.shape[0],), -1, dtype=np.int64)
        if _HAS_NATIVE_TOPO:
            # O(n) native hash lookup against the entity table (first-occurrence
            # ids). Both arrays are 32-bit, so reinterpret as uint32 without a copy.
            entity_nodes = np.ascontiguousarray(entity_nodes_by_key[key])
            ent32 = entity_nodes.view(np.uint32)
            qry32 = occ.view(np.uint32)
            _topo_lib.lookup_entity_rows(
                _topo_ffi.cast("uint32_t*", ent32.ctypes.data), len(ent32), ent32.shape[1],
                _topo_ffi.cast("uint32_t*", qry32.ctypes.data), len(qry32),
                _topo_ffi.cast("int64_t*", ids.ctypes.data))
            matches = ids >= 0
        else:
            # Fallback: searchsorted, valid because the NumPy path keeps uniq sorted.
            uniq_keys = entity_row_keys_by_key[key]
            occ_keys = _int_row_keys(occ)
            pos = np.searchsorted(uniq_keys, occ_keys)
            valid = pos < uniq_keys.size
            matches = np.zeros(occ.shape[0], dtype=bool)
            if np.any(valid):
                valid_pos = pos[valid]
                matches[valid] = uniq_keys[valid_pos] == occ_keys[valid]
            ids[matches] = pos[matches].astype(np.int64, copy=False)

        missing = int(np.count_nonzero(~matches))

        if missing > 0:
            warnings.warn(
                f"Could not map {missing}/{occ.shape[0]} '{cell_type}' elements to global entity ids for key '{key}'.",
                RuntimeWarning,
                stacklevel=2,
            )
            # keep -1 entries; caller can decide what to do
        return jnp.asarray(ids)

    for cell_type, d in tmp.items():
        # Only assign for potential surface blocks (line/triangle/quad)
        entity_id = _assign_entity_id_for_cell_block(cell_type, d["nodes_of_cells"])
        d["entity_id"] = entity_id
        signs = entity_sign_by_key.get(cell_type.lower(), None)
        if entity_id is not None and signs is not None:
            ids_np = np.asarray(entity_id, dtype=int)
            safe_ids = np.where(ids_np >= 0, ids_np, 0)
            d["entity_sign"] = jnp.asarray(np.where(ids_np >= 0, signs[safe_ids], 0.0))

    # --- Finalize _CellInfo objects -------------------------------------------
    cells: Dict[str, _CellInfo] = {}
    for cell_type, d in tmp.items():
        facets_dict = d["facets"]
        edges_dict = d["edges"]

        cells[cell_type] = _CellInfo(
            facets_of_cells=(facets_dict if len(facets_dict) > 0 else None),
            edges_of_cells=(edges_dict if len(edges_dict) > 0 else None),
            nodes_of_cells=d["nodes_of_cells"],
            entity_id=d["entity_id"],
            entity_sign=d["entity_sign"],
        )

    return _MeshInfo(points=points, domains=domains, cells=cells)


def _add_domain_to_mesh_info(
    mesh_info: _MeshInfo,
    domain_name: str,
    on_domain_fun: callable,
    element_type: Optional[Union[str, list[str]]] = None,
) -> _MeshInfo:
    """
    Add a domain to mesh_info by grouping existing cells via a predicate on element node coordinates.

    The domain is stored as:
        mesh_info.domains[domain_name][cell_type] = _DomainInfo(cell_indices=<matching cell ids>)

    Parameters
    ----------
    mesh_info:
        _MeshInfo object (modified in-place; also returned for convenience).
    domain_name:
        Name of the domain.
    on_domain_fun:
        Function applied per element with input shape (n_nodes_per_elem, gdim). It has to return a bool.
          - a scalar bool
    element_type:
        Filter for cell types:
          - None: consider all types
          - exact string: e.g. "triangle6"
          - wildcard: e.g. "triangle*", "quad*"
          - list of strings/patterns: e.g. ["triangle*", "quad4"]

    Returns
    -------
    mesh_info:
        The same object, modified in-place.
    """
    if domain_name in mesh_info.domains:
        warnings.warn(
            f"Domain name '{domain_name}' already exists in mesh_info.domains. "
            f"Existing domain will be overwritten.",
            RuntimeWarning,
            stacklevel=2,
        )

    def matches(t: str) -> bool:
        if element_type is None:
            return True
        if isinstance(element_type, (list, tuple, set)):
            return any(fnmatch(t, pat) for pat in element_type)
        if any(ch in element_type for ch in "*?[]"):
            return fnmatch(t, element_type)
        return t == element_type

    ids_by_type: dict[str, _DomainInfo] = {}
    points = jnp.asarray(mesh_info.points)  # (n_nodes, gdim)

    for cell_type, cinfo in mesh_info.cells.items():
        if not matches(cell_type):
            continue

        elem_nodes = jnp.asarray(cinfo.nodes_of_cells)  # (n_elem, n_nodes_per_elem)
        if elem_nodes.size == 0:
            continue

        # Element coordinates: (n_elem, n_nodes_per_elem, gdim)
        elem_coords = points[elem_nodes]

        # on_domain_fun is POINTWISE: expects (gdim,) and returns bool or bool-array
        def _node_pred(pcoords):
            out = jnp.asarray(on_domain_fun(pcoords))
            return out if out.ndim == 0 else jnp.all(out)

        def _elem_pred(ecoords):
            # ecoords: (n_nodes_per_elem, gdim)
            node_mask = jax.vmap(_node_pred)(ecoords)  # (n_nodes_per_elem,)
            return jnp.all(node_mask)                  # scalar

        mask = jax.vmap(_elem_pred)(elem_coords)  # (n_elem,), eager: no compile overhead
        mask = mask.astype(bool)

        local_ids = jnp.where(mask)[0]
        if local_ids.size == 0:
            continue

        local_ids_np = np.asarray(local_ids, dtype=np.int64)
        ids_by_type[cell_type] = _DomainInfo(cell_indices=local_ids_np)

    mesh_info.domains[domain_name] = ids_by_type
    return mesh_info


def _reduce_points(points: np.ndarray) -> tuple[jnp.ndarray, int]:
    """
    Reduce point coordinates if the mesh is embedded in a lower-dimensional subspace:
      - If all z == 0: drop z (3D -> 2D)
      - If all y == 0 and z == 0: drop y,z (-> 1D)
    Returns (reduced_points, geometric_dimension).
    """
    x = jnp.asarray(points)
    n_dim = int(x.shape[1]) if x.ndim == 2 else 1

    if x.ndim == 2 and x.shape[1] >= 3 and jnp.all(jnp.isclose(x[:, 2], 0.0)):
        x = x[:, :2]
        n_dim = 2
    if x.ndim == 2 and x.shape[1] >= 2 and jnp.all(jnp.isclose(x[:, 1:], 0.0)):
        x = x[:, :1]
        n_dim = 1

    return x, n_dim


def _mesh_type_from_topology(cell_type: str) -> str:
    """
    Map a meshio cell_type (e.g. 'triangle6', 'quad9', 'tetra10', 'hex27') to a base mesh type
    understood by your quadrature / mapping selection.
    """
    b = _canonical_base_type(cell_type)
    if b == "hex":
        return "hexahedron"
    if b == "tetra":
        return "tetra"
    return b  # line, triangle, quad, vertex


@jax.jit
def _unique1d_int_fast_core(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    x = jnp.sort(jnp.ravel(jnp.asarray(x)), stable=False)
    mask = jnp.concatenate(
        [jnp.array([True], dtype=bool), x[1:] != x[:-1]],
        axis=0,
    )
    return x, mask


def _unique1d_int_fast(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.ravel(jnp.asarray(x))
    if x.size == 0:
        return x

    x, mask = _unique1d_int_fast_core(x)
    return x[mask]


def _collect_elem_ids_by_type(mesh_info, domain_name):
    """
    Union element indices across one or multiple domains.

    Returns:
        dict[cell_type] -> unique element ids (local ids inside that cell_type block).
    """    
    dom_names = [domain_name] if isinstance(domain_name, str) else list(domain_name)
    buckets = defaultdict(list)

    for dn in dom_names:
        if dn not in mesh_info.domains:
            raise KeyError(f"Unknown domain '{dn}'. Available: {list(mesh_info.domains.keys())}")

        for ct, dinfo in mesh_info.domains[dn].items():
            ids = jnp.ravel(jnp.asarray(dinfo.cell_indices))
            if ids.size:
                buckets[ct].append(ids)

    out = {}
    for ct, parts in buckets.items():
        if len(parts) == 1:
            out[ct] = parts[0]
        else:
            out[ct] = _unique1d_int_fast(jnp.concatenate(parts, axis=0))
    return out


def _pick_quadrature(mesh_type: str, order: int):
    """Default quadrature selection (same logic as your previous version)."""
    match mesh_type:
        case "line":
            a,b = seeder.gauss_legendre_nd(dimension=1, order=order)
        case "triangle":
            a,b = seeder.int_pts_ref_tri(order=order)
        case "quad":
            a,b = seeder.gauss_legendre_nd(dimension=2, order=order)
        case "tetra":
            a,b = seeder.int_pts_ref_tet(order=order)
        case "hexahedron":
            a,b = seeder.gauss_legendre_nd(dimension=3, order=order)
        case _:
            raise ValueError(f"No integration scheme defined for mesh_type '{mesh_type}'.")

    return jnp.asarray(a), jnp.asarray(b)


def _normalize_integration_mode(mode: Optional[str]) -> str:
    if mode is None:
        return "auto"
    mode = str(mode).lower()
    if mode not in ("auto", "collocation"):
        raise ValueError("integration_mode must be 'auto' or 'collocation'.")
    return mode


def _collocation_integration_order_from_poly_order(poly_order: int) -> int:
    return max(0, 2 * int(poly_order) - 1)


def _line_order_from_cell_type(cell_type: str) -> Optional[int]:
    ct = cell_type.lower()
    if ct == "line":
        return 1
    if ct.startswith("line") and ct[4:].isdigit():
        return max(0, int(ct[4:]) - 1)
    return None


def _tensor_product_order_from_topology(mesh_topology: str) -> Optional[int]:
    base = _canonical_base_type(mesh_topology)
    if base == "line":
        return _line_order_from_cell_type(mesh_topology)
    if base == "quad":
        return _quad_order_from_cell_type(mesh_topology)
    if base == "hex":
        return _hex_order_from_cell_type(mesh_topology)
    return None


def _pick_projection_integration(
    mesh_type: str,
    order: int,
    integration_mode: Optional[str],
    *,
    collocation_order: Optional[int] = None,
):
    mode = _normalize_integration_mode(integration_mode)
    if mode == "auto":
        return _pick_quadrature(mesh_type, order)

    match mesh_type:
        case "line":
            dimension = 1
        case "quad":
            dimension = 2
        case "hexahedron":
            dimension = 3
        case _:
            raise ValueError(
                "Projection integration mode 'collocation' uses Gauss-Lobatto rules and is only "
                f"implemented for tensor-product elements (line, quad, hexahedron), got '{mesh_type}'."
            )

    rule_order = int(order if collocation_order is None else collocation_order)
    a, b = seeder.gauss_lobatto_nd(dimension=dimension, order=rule_order)
    return jnp.asarray(a), jnp.asarray(b)


def _default_integration_order_from_spaces(spaces_by_field: Dict[str, Any]) -> int:
    """
    Take as max from space.default_integration_order
    """
    vals = []
    for sp in spaces_by_field.values():
        vals.append(sp.default_integration_order)
    return max(vals)


def _infer_field_dimensions(spaces_by_field: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a field dimension descriptor table from the spaces.

    For backward compatibility with older parts of your code base, this returns:
      - int for scalar/vector fields (1,2,3,...)
      - tuple for tensor fields (e.g. (2,2))
    """
    out: Dict[str, Any] = {}
    for f, sp in spaces_by_field.items():
        fd = getattr(sp, "field_dimension", 1)
        if isinstance(fd, int):
            out[f] = int(fd)
        elif isinstance(fd, (tuple, list)):
            out[f] = tuple(int(x) for x in fd)
        else:
            raise TypeError(f"Unsupported field_dimension for '{f}': {type(fd)}")
    return out
