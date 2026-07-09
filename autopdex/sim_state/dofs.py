# autopdex/sim_state/dofs.py
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

"""DOF numbering engine: active maps, global DOF numbering, and connectivity."""

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, List, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp

from autopdex import spaces

from .mesh_topology import _BASE_DIM, _CORNER_COUNT, _EDGE_PATTERNS, _PATTERNS, _canonical_base_type, _surface_corner_count
from .mesh_info import _MeshInfo

@dataclass
class _ActiveG2AAndCounts:
    # global -> active id (inactive -> -1)
    node_g2a: np.ndarray
    cell_g2a: Dict[str, np.ndarray]   # cell_type -> (n_cells,)
    edge_g2a: Dict[str, np.ndarray]   # edge_key ("line3", ...) -> (n_edges,)
    face_g2a: Dict[str, np.ndarray]   # facet_key ("triangle6","quad8","line3", ...) -> (n_facets,)

    # counts (= sum(mask))
    n_active_nodes: int
    n_active_cells: Dict[str, int]     # per cell_type
    n_active_edges: Dict[str, int]     # per edge_key
    n_active_faces: Dict[str, int]     # per facet_key

    # totals (convenience)
    n_active_cells_total: int
    n_active_edges_total: int
    n_active_faces_total: int


def _g2a_from_mask(mask: Union[np.ndarray, jnp.ndarray]) -> np.ndarray:
    """Create global-to-active mapping: active ids are 0..(nnz-1), inactive -> -1."""
    mask_np = np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(mask_np)
    g2a = -np.ones(mask_np.shape[0], dtype=int)
    g2a[idx] = np.arange(idx.size, dtype=int)
    return g2a


def _as_list(x: Union[str, List[str], tuple[str, ...]]) -> List[str]:
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    return [x]


def _build_active_g2a_and_counts(
    mesh_info: _MeshInfo,
    domain_names: Union[str, List[str]],
    *,
    space_dim: int,
) -> _ActiveG2AAndCounts:
    """
    Build global->active maps (g2a) and active entity counts for a union of domains.

    Assumes UPDATED _build_mesh_info:
      - facets_of_cells / edges_of_cells are globally consistent per key ("line3", "triangle6", ...),
      - entity_id is available for surface blocks (line*/triangle*/quad*) when needed later.

    Notes
    -----
    This function builds masks internally (idempotent True-setting), then derives g2a and counts.
    No unique() is used here.
    """
    domain_names = _as_list(domain_names)

    n_nodes = int(mesh_info.points.shape[0])
    node_mask = np.zeros((n_nodes,), dtype=bool)

    cell_mask: Dict[str, np.ndarray] = {}
    for cell_type, cinfo in mesh_info.cells.items():
        base = _canonical_base_type(cell_type)
        if base in _BASE_DIM and _BASE_DIM[base] == space_dim:
            cell_mask[cell_type] = np.zeros((int(cinfo.nodes_of_cells.shape[0]),), dtype=bool)

    edge_mask: Dict[str, np.ndarray] = {}
    face_mask: Dict[str, np.ndarray] = {}

    for cell_type, cinfo in mesh_info.cells.items():
        base = _canonical_base_type(cell_type)
        if base not in _BASE_DIM or _BASE_DIM[base] != space_dim:
            continue

        if cinfo.edges_of_cells is not None:
            for ekey, emat in cinfo.edges_of_cells.items():
                if ekey not in edge_mask:
                    emat_np = np.asarray(emat, dtype=int)
                    n_e = int(np.max(emat_np)) + 1 if emat_np.size > 0 else 0
                    edge_mask[ekey] = np.zeros((n_e,), dtype=bool)

        if cinfo.facets_of_cells is not None:
            for fkey, fmat in cinfo.facets_of_cells.items():
                if fkey not in face_mask:
                    fmat_np = np.asarray(fmat, dtype=int)
                    n_f = int(np.max(fmat_np)) + 1 if fmat_np.size > 0 else 0
                    face_mask[fkey] = np.zeros((n_f,), dtype=bool)

    for dname in domain_names:
        dom = mesh_info.domains[dname]

        for cell_type, dinfo in dom.items():
            if cell_type not in mesh_info.cells:
                continue

            base = _canonical_base_type(cell_type)
            if base not in _BASE_DIM or _BASE_DIM[base] != space_dim:
                continue

            ids = np.asarray(dinfo.cell_indices, dtype=int)
            if ids.size == 0:
                continue

            cinfo = mesh_info.cells[cell_type]
            nodes_of_cells = np.asarray(cinfo.nodes_of_cells, dtype=int)

            cell_mask[cell_type][ids] = True

            n_corner = _CORNER_COUNT[base]
            corner_nodes = nodes_of_cells[ids, :n_corner].reshape(-1)
            node_mask[corner_nodes] = True

            if cinfo.edges_of_cells is not None:
                for ekey, emat in cinfo.edges_of_cells.items():
                    eids = np.asarray(emat, dtype=int)[ids].reshape(-1)
                    edge_mask[ekey][eids] = True

            if cinfo.facets_of_cells is not None:
                for fkey, fmat in cinfo.facets_of_cells.items():
                    fids = np.asarray(fmat, dtype=int)[ids].reshape(-1)
                    face_mask[fkey][fids] = True

    node_g2a = _g2a_from_mask(node_mask)
    cell_g2a = {ct: _g2a_from_mask(m) for ct, m in cell_mask.items()}
    edge_g2a = {k: _g2a_from_mask(m) for k, m in edge_mask.items()}
    face_g2a = {k: _g2a_from_mask(m) for k, m in face_mask.items()}

    n_active_nodes = int(np.count_nonzero(node_mask))
    n_active_cells = {ct: int(np.count_nonzero(m)) for ct, m in cell_mask.items()}
    n_active_edges = {k: int(np.count_nonzero(m)) for k, m in edge_mask.items()}
    n_active_faces = {k: int(np.count_nonzero(m)) for k, m in face_mask.items()}

    return _ActiveG2AAndCounts(
        node_g2a=node_g2a,
        cell_g2a=cell_g2a,
        edge_g2a=edge_g2a,
        face_g2a=face_g2a,
        n_active_nodes=n_active_nodes,
        n_active_cells=n_active_cells,
        n_active_edges=n_active_edges,
        n_active_faces=n_active_faces,
        n_active_cells_total=int(sum(n_active_cells.values())),
        n_active_edges_total=int(sum(n_active_edges.values())),
        n_active_faces_total=int(sum(n_active_faces.values())),
    )


@dataclass
class _FieldDofAndConnectivity:
    dofs: jnp.ndarray  # (n_scalar_dofs, *field_shape)  OR (n_scalar_dofs,) for scalar fields
    layout: dict       # offsets/blocks (scalar blocks!)
    volume: Dict[str, jnp.ndarray]                      # cell_type -> (n_active_cells, n_loc_scalar_dofs)
    surface: Dict[str, Dict[str, jnp.ndarray]]          # domain -> cell_type -> (n_elems, n_loc_scalar_dofs)
    g2a: dict          # g2a maps (node/edge/face/cell)
    counts: dict       # active counts (nodes/edges/faces/cells)


def _is_internal_variable_space(space: Any) -> bool:
    return bool(getattr(space, "is_internal_variable", False))


def _split_spatial_discretizations(spatial_discretizations: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    primary = {}
    internal = {}
    for field, space in spatial_discretizations.items():
        if _is_internal_variable_space(space):
            internal[field] = space
        else:
            primary[field] = space
    return primary, internal


def _field_shape_from_space(space) -> Tuple[int, ...]:
    return _field_shape_from_dimension(space.field_dimension)


def _field_shape_from_dimension(field_dimension: Any) -> Tuple[int, ...]:
    if isinstance(field_dimension, int):
        return () if field_dimension == 1 else (int(field_dimension),)
    if isinstance(field_dimension, (tuple, list)):
        return tuple(int(x) for x in field_dimension)
    raise TypeError(f"Unsupported field_dimension type: {type(field_dimension)}")


def _zero_internal_variable_block(
    internal_variable_spaces: Dict[str, Any],
    internal_variable_names: Tuple[str, ...],
    n_elems: int,
    n_qp: int,
    dtype: Any,
) -> Dict[str, jnp.ndarray]:
    block = {}
    for name in internal_variable_names:
        field_shape = _field_shape_from_space(internal_variable_spaces[name])
        block[name] = jnp.zeros((n_elems, n_qp) + field_shape, dtype=dtype)
    return block


def _dof_indices(active_ids: jnp.ndarray, start: int, block: int) -> jnp.ndarray:
    """
    active_ids: (...,) >=0
    returns: (..., block) global scalar dof indices (axis-0 indices into dofs container)
    """
    if block == 0:
        return jnp.zeros(active_ids.shape + (0,), dtype=int)
    offs = jnp.arange(block, dtype=int)
    return start + active_ids.astype(int)[..., None] * block + offs[None, :]


def _orient_edge_dof_indices(edge_dofs: jnp.ndarray, local_edges: jnp.ndarray) -> jnp.ndarray:
    """Orient edge-interior dofs from ascending global node ids to local edge direction."""
    if edge_dofs.shape[-1] <= 1:
        return edge_dofs
    reverse = jnp.asarray(local_edges[..., 0] > local_edges[..., 1])
    return jnp.where(reverse[..., None], jnp.flip(edge_dofs, axis=-1), edge_dofs)


def _np_dof_indices(active_ids: np.ndarray, start: int, block: int) -> np.ndarray:
    """NumPy equivalent of `_dof_indices` for FE setup outside JAX traces."""
    active_ids = np.asarray(active_ids, dtype=int)
    if block == 0:
        return np.zeros(active_ids.shape + (0,), dtype=int)
    return start + active_ids[..., None] * block + np.arange(block, dtype=int)


def _np_orient_edge_dof_indices(edge_dofs: np.ndarray, local_edges: np.ndarray) -> np.ndarray:
    """Orient edge-interior dofs from ascending global node ids to local edge direction."""
    if edge_dofs.shape[-1] <= 1:
        return edge_dofs
    reverse = np.asarray(local_edges[..., 0] > local_edges[..., 1], dtype=bool)
    return np.where(reverse[..., None], np.flip(edge_dofs, axis=-1), edge_dofs)


def _space_basis_for_field(space: Any, cell_type: str, field: str) -> Callable:
    factory = getattr(space, "_basis_for_field", None)
    if factory is not None:
        return factory(cell_type, field)
    return space.basis(cell_type)


def _space_surface_basis_for_field(space: Any, cell_type: str, field: str) -> Callable:
    factory = getattr(space, "_surface_basis_for_field", None)
    if factory is not None:
        return factory(cell_type, field)
    return space.surface_basis(cell_type)


def _space_has_surface_dofs(space: Any, surface_cell_type: str) -> bool:
    base = _canonical_base_type(surface_cell_type)
    node_block = int(space.num_node_dofs.get("line", 0))
    edge_block = int(space.num_edge_dofs.get("line", 0))
    if base == "line":
        return 2 * node_block + edge_block > 0
    if base in ("triangle", "quad"):
        n_corners = _surface_corner_count(base)
        edge_count = 3 if base == "triangle" else 4
        face_block = int(space.num_face_dofs.get(base, 0))
        return n_corners * node_block + edge_count * edge_block + face_block > 0
    return False


def _edge_signs_from_nodes(nodes: np.ndarray, pattern: np.ndarray) -> jnp.ndarray:
    local_edges = nodes[:, pattern]
    return jnp.asarray(np.where(local_edges[:, :, 0] < local_edges[:, :, 1], 1.0, -1.0))


def _orientation_for_space_on_topology(
    mesh_info: _MeshInfo,
    space: Any,
    mesh_topology: str,
    elem_ids: Any,
    *,
    is_surface: bool,
) -> Optional[jnp.ndarray]:
    orientation_kind = getattr(space, "orientation_kind", None)
    if orientation_kind is None:
        return None
    if orientation_kind != "edge":
        raise NotImplementedError(f"Unsupported orientation kind '{orientation_kind}'.")

    base = _canonical_base_type(mesh_topology)
    if is_surface:
        cinfo = mesh_info.cells[mesh_topology]
        if base != "line" or cinfo.entity_sign is None:
            raise NotImplementedError(f"Edge-oriented surface orientation is unavailable for '{mesh_topology}'.")
        return jnp.asarray(cinfo.entity_sign[elem_ids]).reshape((-1, 1))

    if base in ("triangle", "quad"):
        pattern = _PATTERNS[base]
    elif base in ("tetra", "hex"):
        pattern = _EDGE_PATTERNS[base]
    else:
        raise NotImplementedError(f"Edge orientation is unavailable for '{mesh_topology}'.")
    nodes = np.asarray(mesh_info.cells[mesh_topology].nodes_of_cells, dtype=int)[np.asarray(elem_ids), :_CORNER_COUNT[base]]
    return _edge_signs_from_nodes(nodes, pattern)


def _build_dofs_and_connectivity_all_fields(
    mesh_info: _MeshInfo,
    spaces: Dict[str, object],
    active_domains: Dict[str, Union[str, List[str]]],
) -> Dict[str, _FieldDofAndConnectivity]:
    """
    Builds DOF containers + connectivity for all fields.

    IMPORTANT:
      - dofs[field] indexes scalar basis dofs along axis 0, and stores field components in trailing axes.
      - connectivity arrays index axis-0 only (scalar dof ids).
    """
    out: Dict[str, _FieldDofAndConnectivity] = {}
    hex_face_perm = np.asarray([5, 3, 2, 4, 0, 1], dtype=int)

    for field, space in spaces.items():
        if field not in active_domains:
            raise KeyError(f"active_domains missing field '{field}'")

        dim = int(space.dim)
        field_shape = _field_shape_from_space(space)

        # --- active maps (from volume domains only) --------------------------
        maps = _build_active_g2a_and_counts(mesh_info, active_domains[field], space_dim=dim)

        node_g2a = maps.node_g2a
        cell_g2a = maps.cell_g2a

        # unify "edges" / "faces" g2a depending on dimension
        if dim == 2:
            edge_g2a = {k: v for k, v in maps.face_g2a.items() if _canonical_base_type(k) == "line"}
            face_g2a = {}
        else:
            edge_g2a = dict(maps.edge_g2a)
            face_g2a = dict(maps.face_g2a)

        # --- active counts ---------------------------------------------------
        n_active_nodes = int(maps.n_active_nodes)
        if dim == 2:
            n_active_edges = {k: int(maps.n_active_faces.get(k, 0)) for k in edge_g2a.keys()}
            n_active_faces = {}
        else:
            n_active_edges = {k: int(maps.n_active_edges.get(k, 0)) for k in edge_g2a.keys()}
            n_active_faces = {k: int(maps.n_active_faces.get(k, 0)) for k in face_g2a.keys()}
        n_active_cells = {ct: int(maps.n_active_cells.get(ct, 0)) for ct in cell_g2a.keys()}

        # --- scalar dof blocks from space -----------------------------------
        node_block = int(space.num_node_dofs.get("line", 1))               # scalar dofs per vertex
        edge_block = int(space.num_edge_dofs.get("line", 0))               # scalar dofs per edge

        def face_block_for_key(k: str) -> int:
            base = _canonical_base_type(k)
            if base == "triangle":
                return int(space.num_face_dofs.get("tetra", 0))
            if base == "quad":
                return int(space.num_face_dofs.get("hex", 0))
            return 0

        def cell_block_for_celltype(ct: str) -> int:
            base = _canonical_base_type(ct)
            return int(space.num_cell_dofs.get(base, 0))

        # --- offsets (compact numbering in scalar dof space) -----------------
        offset = 0
        node_offset = offset
        offset += n_active_nodes * node_block

        edge_offsets: Dict[str, int] = {}
        if edge_block > 0:
            for k in sorted(n_active_edges.keys()):
                edge_offsets[k] = offset
                offset += n_active_edges[k] * edge_block

        face_offsets: Dict[str, int] = {}
        face_blocks: Dict[str, int] = {}
        if dim == 3:
            for k in sorted(n_active_faces.keys()):
                blk = face_block_for_key(k)
                face_blocks[k] = blk
                if blk > 0:
                    face_offsets[k] = offset
                    offset += n_active_faces[k] * blk

        cell_offsets: Dict[str, int] = {}
        cell_blocks: Dict[str, int] = {}
        for ct in sorted(n_active_cells.keys()):
            blk = cell_block_for_celltype(ct)
            cell_blocks[ct] = blk
            if blk > 0:
                cell_offsets[ct] = offset
                offset += n_active_cells[ct] * blk

        n_scalar_dofs = int(offset)

        # DOF container (not flat in field components)
        dofs_shape = (n_scalar_dofs,) + field_shape
        dofs = np.zeros(dofs_shape, dtype=float)

        # --- volume connectivity (scalar dof indices) ------------------------
        volume_conn: Dict[str, np.ndarray] = {}

        for ct, cinfo in mesh_info.cells.items():
            base = _canonical_base_type(ct)
            if base not in _BASE_DIM or _BASE_DIM[base] != dim:
                continue
            if ct not in cell_g2a:
                continue

            ids = np.flatnonzero(cell_g2a[ct] >= 0)
            if ids.size == 0:
                continue

            nodes_of_cells = np.asarray(cinfo.nodes_of_cells, dtype=int)
            parts = []

            # corner nodes -> node dofs
            n_corner = _CORNER_COUNT[base]
            corner_nodes = nodes_of_cells[ids, :n_corner]
            v_active = node_g2a[corner_nodes]
            if np.any(v_active < 0):
                raise ValueError(f"[{field}] inactive vertex on active volume cells '{ct}'")

            vd = _np_dof_indices(v_active.reshape(-1), start=node_offset, block=node_block)
            parts.append(vd.reshape(ids.shape[0], -1))

            # edge dofs
            if edge_block > 0:
                if dim == 2:
                    if cinfo.facets_of_cells is not None:
                        local_edges = corner_nodes[:, _PATTERNS[base]]
                        for ekey, emat in cinfo.facets_of_cells.items():
                            if ekey not in edge_g2a or ekey not in edge_offsets:
                                continue
                            emat_np = np.asarray(emat, dtype=int)
                            e_active = edge_g2a[ekey][emat_np[ids]]
                            if np.any(e_active < 0):
                                raise ValueError(f"[{field}] inactive edge in volume '{ct}' key '{ekey}'")
                            ed = _np_dof_indices(e_active.reshape(-1), start=edge_offsets[ekey], block=edge_block)
                            ed = ed.reshape(ids.shape[0], -1, edge_block)
                            ed = _np_orient_edge_dof_indices(ed, local_edges)
                            parts.append(ed.reshape(ids.shape[0], -1))
                else:
                    if cinfo.edges_of_cells is not None:
                        local_edges = corner_nodes[:, _EDGE_PATTERNS[base]]
                        for ekey, emat in cinfo.edges_of_cells.items():
                            if ekey not in edge_g2a or ekey not in edge_offsets:
                                continue
                            emat_np = np.asarray(emat, dtype=int)
                            e_active = edge_g2a[ekey][emat_np[ids]]
                            if np.any(e_active < 0):
                                raise ValueError(f"[{field}] inactive edge in volume '{ct}' key '{ekey}'")
                            ed = _np_dof_indices(e_active.reshape(-1), start=edge_offsets[ekey], block=edge_block)
                            ed = ed.reshape(ids.shape[0], -1, edge_block)
                            ed = _np_orient_edge_dof_indices(ed, local_edges)
                            parts.append(ed.reshape(ids.shape[0], -1))

            # face dofs (3D)
            if dim == 3 and cinfo.facets_of_cells is not None:
                for fkey, fmat in cinfo.facets_of_cells.items():
                    if fkey not in face_g2a or fkey not in face_offsets:
                        continue
                    blk = face_blocks.get(fkey, 0)
                    if blk == 0:
                        continue
                    fmat_np = np.asarray(fmat, dtype=int)
                    f_active = face_g2a[fkey][fmat_np[ids]]
                    if np.any(f_active < 0):
                        raise ValueError(f"[{field}] inactive face in volume '{ct}' key '{fkey}'")
                    fdofs = _np_dof_indices(f_active.reshape(-1), start=face_offsets[fkey], block=blk)
                    fdofs = fdofs.reshape(ids.shape[0], -1, blk)
                    if base == "hex" and _canonical_base_type(fkey) == "quad":
                        fdofs = fdofs[:, hex_face_perm, :]
                    parts.append(fdofs.reshape(ids.shape[0], -1))

            # cell interior dofs
            cblk = cell_blocks.get(ct, 0)
            if cblk > 0:
                c_active = cell_g2a[ct][ids]
                cd = _np_dof_indices(c_active, start=cell_offsets[ct], block=cblk)
                parts.append(cd)

            volume_conn[ct] = np.concatenate(parts, axis=1).astype(int, copy=False)

        # --- surface connectivity (scalar dof indices) ------------------------
        surf_dim = dim - 1
        surface_conn: Dict[str, Dict[str, np.ndarray]] = {}

        if surf_dim >= 1:
            for dname, dom in mesh_info.domains.items():
                per_type: Dict[str, np.ndarray] = {}

                for sct, dinfo in dom.items():
                    sbase = _canonical_base_type(sct)
                    if sbase not in _BASE_DIM or _BASE_DIM[sbase] != surf_dim:
                        continue
                    if sct not in mesh_info.cells:
                        continue

                    ids = np.asarray(dinfo.cell_indices, dtype=int)
                    if ids.size == 0:
                        continue

                    sinfo = mesh_info.cells[sct]
                    surface_nodes = np.asarray(sinfo.nodes_of_cells, dtype=int)
                    parts = []
                    valid = np.ones((ids.shape[0],), dtype=bool)

                    # corner nodes
                    if sbase == "line":
                        ar = 2
                    elif sbase == "triangle":
                        ar = 3
                    elif sbase == "quad":
                        ar = 4
                    else:
                        continue

                    corners = surface_nodes[ids, :ar]
                    v_active = node_g2a[corners]
                    valid = valid & np.all(v_active >= 0, axis=1)

                    vd = _np_dof_indices(v_active.reshape(-1), start=node_offset, block=node_block)
                    parts.append(vd.reshape(ids.shape[0], -1))

                    # edge dofs on surface elements
                    if edge_block > 0:
                        if sbase == "line":
                            # use entity_id (maps line-cell -> global edge id in key sct)
                            if sinfo.entity_id is not None and sct in edge_g2a and sct in edge_offsets:
                                entity_id = np.asarray(sinfo.entity_id, dtype=int)
                                eid = entity_id[ids]
                                e_active = edge_g2a[sct][eid]
                                valid = valid & (eid >= 0) & (e_active >= 0)
                                ed = _np_dof_indices(e_active, start=edge_offsets[sct], block=edge_block)
                                ed = _np_orient_edge_dof_indices(ed, corners)
                                parts.append(ed)
                            else:
                                warnings.warn(
                                    f"[{field}] Surface '{dname}' has '{sct}' but cannot attach edge dofs "
                                    f"(missing entity_id or edge key not active). Using vertex dofs only.",
                                    RuntimeWarning,
                                    stacklevel=2,
                                )
                        else:
                            # triangle/quad surface in 3D: its edges are in facets_of_cells (keys line*)
                            if sinfo.facets_of_cells is not None:
                                local_edges = corners[:, _EDGE_PATTERNS[sbase]]
                                for ekey, emat in sinfo.facets_of_cells.items():
                                    if ekey not in edge_g2a or ekey not in edge_offsets:
                                        continue
                                    emat_np = np.asarray(emat, dtype=int)
                                    e_active = edge_g2a[ekey][emat_np[ids]]
                                    valid = valid & np.all(e_active >= 0, axis=1)
                                    ed = _np_dof_indices(e_active.reshape(-1), start=edge_offsets[ekey], block=edge_block)
                                    ed = ed.reshape(ids.shape[0], -1, edge_block)
                                    ed = _np_orient_edge_dof_indices(ed, local_edges)
                                    parts.append(ed.reshape(ids.shape[0], -1))

                    # face dofs on 3D surface elements (triangle*/quad*): use entity_id in key sct
                    if dim == 3 and sinfo.entity_id is not None and sct in face_g2a and sct in face_offsets:
                        blk = face_blocks.get(sct, 0)
                        if blk > 0:
                            entity_id = np.asarray(sinfo.entity_id, dtype=int)
                            fid = entity_id[ids]
                            f_active = face_g2a[sct][fid]
                            valid = valid & (fid >= 0) & (f_active >= 0)
                            fdofs = _np_dof_indices(f_active, start=face_offsets[sct], block=blk)
                            parts.append(fdofs)

                    conn = np.concatenate(parts, axis=1).astype(int, copy=False)

                    if not bool(np.all(valid)):
                        conn = conn[valid]

                    if conn.shape[0] > 0:
                        per_type[sct] = conn

                if per_type:
                    surface_conn[dname] = per_type

        out[field] = _FieldDofAndConnectivity(
            dofs=jnp.asarray(dofs),
            layout=dict(
                n_scalar_dofs=n_scalar_dofs,
                field_shape=field_shape,
                node_offset=node_offset,
                node_block=node_block,
                edge_offsets=edge_offsets,
                edge_block=edge_block,
                face_offsets=face_offsets,
                face_blocks=face_blocks,
                cell_offsets=cell_offsets,
                cell_blocks=cell_blocks,
            ),
            volume={ct: jnp.asarray(conn, dtype=int) for ct, conn in volume_conn.items()},
            surface={
                dname: {sct: jnp.asarray(conn, dtype=int) for sct, conn in per_type.items()}
                for dname, per_type in surface_conn.items()
            },
            g2a=dict(
                node_g2a=jnp.asarray(node_g2a, dtype=int),
                edge_g2a={k: jnp.asarray(v, dtype=int) for k, v in edge_g2a.items()},
                face_g2a={k: jnp.asarray(v, dtype=int) for k, v in face_g2a.items()},
                cell_g2a={k: jnp.asarray(v, dtype=int) for k, v in cell_g2a.items()},
            ),
            counts=dict(
                n_active_nodes=n_active_nodes,
                n_active_edges=n_active_edges,
                n_active_faces=n_active_faces,
                n_active_cells=n_active_cells,
            ),
        )

    return out
