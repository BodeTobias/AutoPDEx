# autopdex/sim_state/projection.py
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

"""Shared L2 projection, output, and VTU helpers (free functions/classes)."""

import os
import warnings
from pathlib import Path
from inspect import signature
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, List, Tuple, Union

import numpy as np
import meshio
import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from autopdex import assembler, dae, models, utility

from .mesh_topology import _EDGE_PATTERNS, _canonical_base_type, _surface_corner_count
from .mesh_info import _MeshInfo
from .dofs import _FieldDofAndConnectivity, _as_list, _dof_indices, _orient_edge_dof_indices

_PROJECTION_KEY = "projection"


_PROJECTED_DIRICHLET_RUNTIME_KEY = "_projected_dirichlet_runtime"


_PROJECTED_IC_RUNTIME_KEY = "_projected_ic_runtime"


_VTU_OUTPUT_SOURCE_DOFS_KEY = "_vtu_output_source_dofs"


@dataclass(frozen=True)
class _L2ProjectionSetInfo:
    domain_name: str
    mesh_topology: str
    mesh_type: str
    basis_cell_type: str
    domain_dim: int
    is_surface: bool
    target_basis: Callable
    mapping_basis: Callable
    physical_connectivity: jnp.ndarray
    target_global_connectivity: jnp.ndarray
    volume_element_ids: Optional[jnp.ndarray]
    surface_domains: Tuple[str, ...]
    surface_element_ids: Optional[jnp.ndarray]
    ref_int_coor: jnp.ndarray
    ref_int_weights: jnp.ndarray


@dataclass
class _PreparedL2Projection:
    name: str
    target_name: Optional[str]
    field_shape: Tuple[int, ...]
    flat_components: int
    global_scalar_ids: jnp.ndarray
    local_shape: Tuple[int, ...]
    full_shape: Tuple[int, ...]
    set_infos: Tuple[_L2ProjectionSetInfo, ...]
    settings_template: Dict[str, Any]
    mass_matrix: Any
    jacobi_diag: jnp.ndarray
    jacobi_inv_diag: jnp.ndarray
    outside_value: float = 0.0
    last_local_solution: Optional[jnp.ndarray] = None


@dataclass(frozen=True)
class _OutputComponentLayout:
    key: str
    export_name: str
    field_shape: Tuple[int, ...]
    start: int
    stop: int


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _OrderedOutputValues:
    keys: Tuple[str, ...]
    arrays: Tuple[Any, ...]

    def tree_flatten(self):
        return self.arrays, self.keys

    @classmethod
    def tree_unflatten(cls, keys, arrays):
        return cls(tuple(keys), tuple(arrays))


@jax.tree_util.register_dataclass
@dataclass
class _VtuOutputRuntime:
    global_scalar_ids: jnp.ndarray
    node_coordinates: jnp.ndarray
    connectivity: Tuple[Dict[str, jnp.ndarray], ...]
    source_connectivity: Tuple[Dict[str, jnp.ndarray], ...]
    mass_matrix: Any
    jacobi_inv_diag: jnp.ndarray


@jax.tree_util.register_dataclass
@dataclass
class _ProjectedDirichletRuntime:
    global_scalar_ids: jnp.ndarray
    node_coordinates: jnp.ndarray
    connectivity: Tuple[Dict[str, jnp.ndarray], ...]
    orientation: Tuple[Dict[str, jnp.ndarray], ...]
    source_connectivity: Tuple[Dict[str, jnp.ndarray], ...]
    mass_matrix: Any
    jacobi_inv_diag: jnp.ndarray


@jax.tree_util.register_dataclass
@dataclass
class _ProjectedICRuntime:
    global_scalar_ids: jnp.ndarray
    node_coordinates: jnp.ndarray
    connectivity: Tuple[Dict[str, jnp.ndarray], ...]
    mass_matrix: Any
    jacobi_inv_diag: jnp.ndarray


@jax.tree_util.register_dataclass
@dataclass
class _VtuCallbackPolicyState:
    target_times: jnp.ndarray
    current_save_idx: jnp.ndarray
    next_target_idx: jnp.ndarray
    output_locals: Tuple[jnp.ndarray, ...]


@dataclass(frozen=True)
class _VtuOutputGroup:
    static_settings: Any
    layouts: Tuple[_OutputComponentLayout, ...]
    cache_key: Tuple[Any, ...]
    n_local: int
    n_points: int
    total_components: int
    outside_value: float
    projection_tol: float
    projection_atol: float
    projection_maxiter: int


def _solve_l2_projector_components_impl(
    mass_matrix,
    jacobi_inv_diag: jnp.ndarray,
    rhs: jnp.ndarray,
    *,
    tol: float,
    atol: float,
    maxiter: int,
) -> jnp.ndarray:
    rhs = jnp.asarray(rhs)
    inv_sqrt_diag = jnp.sqrt(jacobi_inv_diag)

    def mass_matvec(x):
        if callable(mass_matrix):
            return mass_matrix(x)
        return mass_matrix @ x

    def scaled_mass_matvec(x):
        return inv_sqrt_diag * mass_matvec(inv_sqrt_diag * x)

    def solve_component(rhs_component):
        scaled_delta, _ = jax.scipy.sparse.linalg.cg(
            scaled_mass_matvec,
            inv_sqrt_diag * rhs_component,
            tol=tol,
            atol=atol,
            maxiter=maxiter,
        )
        return inv_sqrt_diag * scaled_delta

    if rhs.ndim == 1:
        return solve_component(rhs)
    return jax.vmap(solve_component, in_axes=1, out_axes=1)(rhs)


_solve_l2_projector_components = jax.jit(
    _solve_l2_projector_components_impl,
    static_argnames=("tol", "atol", "maxiter"),
)


def _num_flat_components(field_shape: Tuple[int, ...]) -> int:
    return 1 if len(field_shape) == 0 else int(np.prod(field_shape))


def _reshape_projector_coefficients(coeffs: jnp.ndarray, field_shape: Tuple[int, ...]) -> jnp.ndarray:
    coeffs = jnp.asarray(coeffs)
    if len(field_shape) == 0:
        return coeffs.reshape((coeffs.shape[0],))
    return coeffs.reshape((coeffs.shape[0],) + field_shape)


def _flatten_projector_values(values: jnp.ndarray) -> jnp.ndarray:
    values = jnp.asarray(values)
    if values.ndim <= 1:
        return values
    return values.reshape((values.shape[0], -1))


def _coerce_projected_value(value: Any, field_shape: Tuple[int, ...], flat_components: int) -> jnp.ndarray:
    arr = jnp.asarray(value)
    if len(field_shape) == 0:
        if arr.ndim != 0:
            arr = jnp.reshape(arr, ())
        return arr
    if tuple(arr.shape) == field_shape:
        return arr
    if arr.size != flat_components:
        raise ValueError(
            f"Projected quantity has shape {arr.shape}, expected {field_shape} or a flat array with {flat_components} entries."
        )
    return jnp.reshape(arr, field_shape)


def _component_id(field_shape: Tuple[int, ...], index: Optional[Union[int, Tuple[int, ...]]]) -> int:
    if len(field_shape) == 0:
        if index is not None:
            raise ValueError("index must be None for scalar fields.")
        return 0
    if index is None:
        raise ValueError("index is required when prescribing a scalar value on a non-scalar field.")
    idx_t = (index,) if isinstance(index, int) else tuple(index)
    if len(idx_t) != len(field_shape):
        raise ValueError(f"index must have length {len(field_shape)}, got {idx_t}.")
    for ax, (ii, dim) in enumerate(zip(idx_t, field_shape)):
        if (not isinstance(ii, int)) or ii < 0 or ii >= dim:
            raise ValueError(f"index[{ax}]={ii} out of range [0, {dim}).")
    return int(np.ravel_multi_index(idx_t, field_shape))


def _component_mask(field_shape: Tuple[int, ...], index: Optional[Union[int, Tuple[int, ...]]]) -> jnp.ndarray:
    n_comp = _num_flat_components(field_shape)
    if len(field_shape) == 0:
        return jnp.ones((1,), dtype=bool)
    if index is None:
        return jnp.ones((n_comp,), dtype=bool)
    comp_id = _component_id(field_shape, index)
    return jnp.arange(n_comp) == comp_id


def _merge_projector_scalar_ids(connectivities: list[jnp.ndarray], n_scalar_dofs: int) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
    merged = np.unique(
        np.concatenate([np.asarray(conn, dtype=int).reshape(-1) for conn in connectivities], axis=0)
    )
    g2l = -np.ones((n_scalar_dofs,), dtype=int)
    g2l[merged] = np.arange(merged.size, dtype=int)
    local_connectivities = tuple(
        jnp.asarray(g2l[np.asarray(conn, dtype=int)], dtype=int)
        for conn in connectivities
    )
    return jnp.asarray(merged, dtype=int), local_connectivities


def _integration_scaling_from_jacobian(jacobian: jnp.ndarray, domain_dim: int) -> jnp.ndarray:
    problem_dim = jacobian.shape[0]
    if domain_dim == problem_dim:
        return utility.matrix_det(jacobian)
    if domain_dim == 2 and problem_dim == 3:
        g_1 = jacobian[:, 0]
        g_2 = jacobian[:, 1]
        return jnp.linalg.norm(jnp.cross(g_1, g_2))
    if domain_dim == 1 and problem_dim in (2, 3):
        return jnp.linalg.norm(jacobian)
    raise ValueError(
        f"Unsupported combination of domain and problem dimension: {domain_dim}D domain in {problem_dim}D space."
    )


def _build_surface_connectivity_for_field(
    mesh_info: _MeshInfo,
    pack: _FieldDofAndConnectivity,
    domain_name: str,
    surface_cell_type: str,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if domain_name not in mesh_info.domains or surface_cell_type not in mesh_info.domains[domain_name]:
        return jnp.zeros((0,), dtype=int), jnp.zeros((0, 0), dtype=int)

    ids = jnp.asarray(mesh_info.domains[domain_name][surface_cell_type].cell_indices, dtype=int)
    if ids.size == 0:
        return ids, jnp.zeros((0, 0), dtype=int)

    sinfo = mesh_info.cells[surface_cell_type]
    sbase = _canonical_base_type(surface_cell_type)
    node_offset = int(pack.layout["node_offset"])
    node_block = int(pack.layout["node_block"])
    edge_offsets = pack.layout["edge_offsets"]
    edge_block = int(pack.layout["edge_block"])
    face_offsets = pack.layout["face_offsets"]
    face_blocks = pack.layout["face_blocks"]
    node_g2a = pack.g2a["node_g2a"]
    edge_g2a = pack.g2a["edge_g2a"]
    face_g2a = pack.g2a["face_g2a"]

    parts = []
    valid = jnp.ones((ids.shape[0],), dtype=bool)

    n_corners = _surface_corner_count(sbase)
    corners = sinfo.nodes_of_cells[ids, :n_corners]
    v_active = node_g2a[corners]
    valid = valid & jnp.all(v_active >= 0, axis=1)
    if node_block > 0:
        vd = _dof_indices(v_active.reshape(-1), start=node_offset, block=node_block)
        parts.append(vd.reshape(ids.shape[0], -1))

    if edge_block > 0:
        if sbase == "line":
            if sinfo.entity_id is not None and surface_cell_type in edge_g2a and surface_cell_type in edge_offsets:
                eid = sinfo.entity_id[ids]
                e_active = edge_g2a[surface_cell_type][eid]
                valid = valid & (eid >= 0) & (e_active >= 0)
                ed = _dof_indices(e_active, start=edge_offsets[surface_cell_type], block=edge_block)
                ed = _orient_edge_dof_indices(ed, corners)
                parts.append(ed)
        elif sinfo.facets_of_cells is not None:
            for ekey, emat in sinfo.facets_of_cells.items():
                if ekey not in edge_g2a or ekey not in edge_offsets:
                    continue
                e_active = edge_g2a[ekey][emat[ids]]
                valid = valid & jnp.all(e_active >= 0, axis=1)
                ed = _dof_indices(e_active.reshape(-1), start=edge_offsets[ekey], block=edge_block)
                ed = ed.reshape(ids.shape[0], -1, edge_block)
                local_edges = corners[:, _EDGE_PATTERNS[sbase]]
                ed = _orient_edge_dof_indices(ed, local_edges)
                parts.append(ed.reshape(ids.shape[0], -1))

    if sinfo.entity_id is not None and surface_cell_type in face_g2a and surface_cell_type in face_offsets:
        blk = int(face_blocks.get(surface_cell_type, 0))
        if blk > 0:
            fid = sinfo.entity_id[ids]
            f_active = face_g2a[surface_cell_type][fid]
            valid = valid & (fid >= 0) & (f_active >= 0)
            fd = _dof_indices(f_active, start=face_offsets[surface_cell_type], block=blk)
            parts.append(fd)

    if len(parts) == 0:
        return jnp.zeros((0,), dtype=int), jnp.zeros((0, 0), dtype=int)

    conn = jnp.concatenate(parts, axis=1).astype(int)
    conn = conn[valid]
    ids = ids[valid]
    return ids, conn


def _make_projection_quantity_caller(
    quantity_fun: Optional[Callable],
    source_static_settings: Any,
    field_shape: Tuple[int, ...],
    flat_components: int,
) -> Callable:
    if quantity_fun is None:
        zero = jnp.zeros(field_shape if len(field_shape) > 0 else ())
        return lambda x_int, trial_ansatz, settings, elem_number, int_num, set: zero

    num_args = len(signature(quantity_fun).parameters)
    if num_args not in (6, 7):
        raise ValueError("quantity_fun must take 6 or 7 arguments.")

    def call_quantity(x_int, trial_ansatz, settings, elem_number, int_num, set):
        if num_args == 6:
            value = quantity_fun(x_int, trial_ansatz, settings, source_static_settings, elem_number, set)
        else:
            value = quantity_fun(
                x_int,
                trial_ansatz,
                settings,
                source_static_settings,
                elem_number,
                int_num,
                set,
            )
        return _coerce_projected_value(value, field_shape, flat_components)

    return call_quantity


def _make_l2_projection_model(
    *,
    projection_key: str,
    target_basis: Callable,
    mapping_basis: Callable,
    field_shape: Tuple[int, ...],
    flat_components: int,
    domain_dim: int,
    ref_int_coor: jnp.ndarray,
    ref_int_weights: jnp.ndarray,
    quantity_fun: Optional[Callable],
    source_fields: Tuple[str, ...],
    source_bases: Dict[str, Callable],
    source_sets: Optional[Dict[str, int]],
    source_static_settings: Any,
) -> Callable:
    quantity_caller = _make_projection_quantity_caller(
        quantity_fun,
        source_static_settings,
        field_shape,
        flat_components,
    )
    target_caller = models._make_caller_adapter(target_basis)
    mapping_caller = models._make_caller_adapter(mapping_basis)
    source_callers = {
        field: models._make_caller_adapter(basis)
        for field, basis in source_bases.items()
    }
    source_sets = {} if source_sets is None else dict(source_sets)

    def user_residual(fI, xI, elem_number, settings, static_settings, set):
        current_time = settings.get("current time", 0.0)

        def mapping(x):
            return mapping_caller(x, xI, xI, settings, False, domain_dim, elem_number, set)

        def projector_from_local(coeffs_flat):
            coeffs = _reshape_projector_coefficients(coeffs_flat, field_shape)
            return lambda x: target_caller(
                x, xI, coeffs, settings, False, domain_dim, elem_number, set
            )

        trial_ansatz = {"physical coor": mapping}
        if len(source_fields) > 0:
            source_connectivity = settings["projection source connectivity"][set]
            source_dofs = settings["projection source dofs"]
            for field in source_fields:
                local_conn = source_connectivity[field][elem_number]
                basis = source_bases[field]
                if isinstance(source_dofs, assembler._q_fun_state):
                    q_tup = source_dofs.q_ts[field]
                    local_q = q_tup[0].at[local_conn].get(wrap_negative_indices=False)
                    local_q_derivs = tuple(
                        q_deriv.at[local_conn].get(wrap_negative_indices=False)
                        for q_deriv in q_tup[1:]
                    )

                    def local_coeffs_fun(t, local_q=local_q, local_q_derivs=local_q_derivs):
                        return dae.discrete_value_with_derivatives(t, local_q, local_q_derivs)
                else:
                    local_coeffs = source_dofs[field][local_conn]

                    def local_coeffs_fun(t, local_coeffs=local_coeffs):
                        return local_coeffs

                basis = source_callers[field]
                source_set = int(source_sets.get(field, set))
                trial_ansatz[field] = (
                    lambda x, t, field=field, local_coeffs_fun=local_coeffs_fun, basis=basis, source_set=source_set:
                    utility.scale_dof_value(
                        settings, field,
                        basis(x, xI, local_coeffs_fun(t), settings, True, domain_dim, elem_number, source_set)
                    )
                )

        def integrated_weak_form(test_coeffs_flat):
            trial_coeffs = fI(current_time)[projection_key] if callable(fI) else fI[projection_key]
            trial_projector = projector_from_local(trial_coeffs)
            test_projector = projector_from_local(test_coeffs_flat)

            def functional(x_int, w_int, int_num):
                quantity = quantity_caller(x_int, trial_ansatz, settings, elem_number, int_num, set)
                weak_form = jnp.sum((quantity - trial_projector(x_int)) * test_projector(x_int))
                jacobian = jax.jacfwd(mapping)(x_int)
                return weak_form * w_int * _integration_scaling_from_jacobian(jacobian, domain_dim)

            int_nums = jnp.arange(ref_int_coor.shape[0])
            return jax.vmap(functional, (0, 0, 0), 0)(ref_int_coor, ref_int_weights, int_nums).sum(axis=0)

        test0 = jnp.zeros_like(fI(current_time)[projection_key] if callable(fI) else fI[projection_key])
        return {projection_key: jax.jacrev(integrated_weak_form)(test0)}

    return user_residual


def _wrap_projected_value_fun(
    field_shape: Tuple[int, ...],
    value_fun: Callable,
    index: Optional[Union[int, Tuple[int, ...]]],
    field: Optional[str] = None,
) -> Callable:
    flat_components = _num_flat_components(field_shape)
    comp_id = None if (len(field_shape) == 0 or index is None) else _component_id(field_shape, index)

    def quantity_fun(x_int, trial_ansatz, settings, static_settings, elem_number, set):
        x = trial_ansatz["physical coor"](x_int)
        value = _call_natural_value_fun(value_fun, x, settings.get("current time", 0.0), settings)
        if len(field_shape) == 0:
            coerced = jnp.asarray(value)
            return utility.unscale_dof_value(settings, field, coerced) if field is not None else coerced
        if comp_id is None:
            coerced = _coerce_projected_value(value, field_shape, flat_components)
            return utility.unscale_dof_value(settings, field, coerced) if field is not None else coerced
        flat = jnp.zeros((flat_components,), dtype=jnp.asarray(value).dtype)
        flat = flat.at[comp_id].set(jnp.asarray(value))
        coerced = flat.reshape(field_shape)
        return utility.unscale_dof_value(settings, field, coerced) if field is not None else coerced

    return quantity_fun


def _make_initial_value_quantity_fun(field_shape: Tuple[int, ...], value_fun: Callable, field: Optional[str] = None) -> Callable:
    flat_components = _num_flat_components(field_shape)

    def quantity_fun(x_int, trial_ansatz, settings, static_settings, elem_number, set):
        x = trial_ansatz["physical coor"](x_int)
        value = _call_ic_value_fun(value_fun, x, settings)
        coerced = _coerce_projected_value(value, field_shape, flat_components)
        return utility.unscale_dof_value(settings, field, coerced) if field is not None else coerced

    return quantity_fun


def _apply_callable_ic_projections(ic_specs, settings, dofs):
    """Re-evaluate callable ICs via L2 projection under current settings. JAX-differentiable."""
    dofs = dict(dofs)
    runtimes = settings.get(_PROJECTED_IC_RUNTIME_KEY, None)
    if runtimes is None:
        return dofs
    for spec, runtime in zip(ic_specs, runtimes):
        field = spec["field"]
        proj_settings = dict(settings)
        proj_settings.pop(_PROJECTED_IC_RUNTIME_KEY, None)
        proj_settings.pop(_PROJECTED_DIRICHLET_RUNTIME_KEY, None)
        proj_settings["node coordinates"] = runtime.node_coordinates
        proj_settings["connectivity"] = runtime.connectivity
        proj_settings["projection source connectivity"] = {}
        proj_settings["projection source dofs"] = {}
        local_guess = dofs[field][runtime.global_scalar_ids]
        projection_dofs = {_PROJECTION_KEY: local_guess}
        residual = assembler.assemble_residual(projection_dofs, proj_settings, spec["projection_static_settings"])[_PROJECTION_KEY]
        rhs = residual.reshape(-1)
        delta = _solve_l2_projector_components(runtime.mass_matrix, runtime.jacobi_inv_diag, rhs, tol=1e-8, atol=1e-10, maxiter=1000)
        local_solution = local_guess + delta.reshape(local_guess.shape)
        values = _reshape_projector_coefficients(local_solution, tuple(spec["field_shape"]))
        dofs[field] = dofs[field].at[runtime.global_scalar_ids].set(values)
    return dofs


def _call_boundary_selector(on_boundary_fun: Callable, x: jnp.ndarray) -> Any:
    num_args = len(signature(on_boundary_fun).parameters)
    if num_args == 2:
        warnings.warn(
            "on_boundary_fun(x, t) is deprecated in SimState.add_strong_bc(...); use on_boundary_fun(x) instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    if x.ndim == 1:
        if num_args == 1:
            return on_boundary_fun(x)
        if num_args == 2:
            return on_boundary_fun(x, 0.0)
    elif x.ndim == 2:
        if num_args == 1:
            return jax.vmap(on_boundary_fun)(x)
        if num_args == 2:
            return jax.vmap(lambda xi: on_boundary_fun(xi, 0.0))(x)
    raise ValueError("on_boundary_fun must take 1 argument (x) or, deprecated, 2 arguments (x, t).")


def _select_output_internal_variables(settings, static_settings, set, elem_number, int_num):
    names_by_set = tuple(static_settings.get("internal variable names", tuple()))
    if set >= len(names_by_set):
        return {}
    names = tuple(names_by_set[set])
    if len(names) == 0:
        return {}
    store = settings.get("internal variables", None)
    if store is None:
        raise KeyError("settings must contain 'internal variables' for context output using internal_vars.")
    block = store[set] if isinstance(store, (tuple, list)) else store
    return {name: block[name][elem_number, int_num] for name in names}


def _require_model_context_fun(fun: Callable, name: str) -> None:
    if len(signature(fun).parameters) != 1:
        raise ValueError(f"{name} must take exactly one ModelContext argument in the high-level interface.")


def _normalize_output_domains(
    output_domains: Optional[Union[str, List[str], Tuple[str, ...]]],
) -> Optional[Tuple[str, ...]]:
    if output_domains is None:
        return None
    domains = tuple(_as_list(output_domains))
    if len(domains) == 0 or any((not isinstance(name, str)) or name == "" for name in domains):
        raise ValueError("domains must be None, a non-empty string, or a non-empty sequence of strings.")
    return domains


def _selected_output_domains_for_model(
    output_domains: Optional[Tuple[str, ...]],
    requested_domains: Tuple[str, ...],
) -> Tuple[str, ...]:
    if output_domains is None:
        return tuple()
    requested = set(requested_domains)
    return tuple(domain for domain in output_domains if domain in requested)


def _call_output_dict_fun(
    output_fun: Callable,
    x_int: jnp.ndarray,
    trial_ansatz: Dict[str, Callable],
    settings: Dict[str, Any],
    source_static_settings: Any,
    elem_number: int,
    int_num: int,
    set: int,
) -> Dict[str, Any]:
    ctx = models.ModelContext(
        mode="output",
        x_int=x_int,
        trial_ansatz=trial_ansatz,
        internal_vars=_select_output_internal_variables(
            settings, source_static_settings, set, elem_number, int_num
        ),
        local_solvers=source_static_settings.get("local subsystem solvers", None),
        settings=settings,
        static_settings=source_static_settings,
        elem_number=elem_number,
        int_point_number=int_num,
        set=set,
    )
    values = output_fun(ctx)
    if not isinstance(values, dict):
        raise ValueError("ModelContext output mode must return a dict of output quantities.")
    return values


def _output_component_layouts(
    spec: FrozenDict,
    selected: Optional[set[str]],
) -> Tuple[_OutputComponentLayout, ...]:
    layouts = []
    start = 0
    shapes = dict(spec["output_shapes"])
    for key in tuple(spec["output_keys"]):
        preferred_name = f"{key}__{spec['domain_name']}" if spec["append_domain"] else key
        if selected is not None and preferred_name not in selected and key not in selected:
            continue
        field_shape = tuple(shapes[key])
        stop = start + _num_flat_components(field_shape)
        layouts.append(
            _OutputComponentLayout(
                key=key,
                export_name=preferred_name,
                field_shape=field_shape,
                start=start,
                stop=stop,
            )
        )
        start = stop
    return tuple(layouts)


def _make_grouped_output_quantity_fun(
    output_fun: Callable,
    layouts: Tuple[_OutputComponentLayout, ...],
) -> Callable:
    def quantity_fun(x_int, trial_ansatz, settings, static_settings, elem_number, int_num, set):
        values = _call_output_dict_fun(
            output_fun,
            x_int,
            trial_ansatz,
            settings,
            static_settings,
            elem_number,
            int_num,
            set,
        )
        flat_values = []
        for layout in layouts:
            if layout.key not in values:
                raise KeyError(f"output_fun did not return the requested output key '{layout.key}'.")
            flat_components = layout.stop - layout.start
            coerced = _coerce_projected_value(values[layout.key], layout.field_shape, flat_components)
            flat_values.append(jnp.reshape(coerced, (flat_components,)))
        if len(flat_values) == 1:
            return flat_values[0]
        return jnp.concatenate(flat_values, axis=0)

    return quantity_fun


def _scatter_output_components(
    projector: _PreparedL2Projection,
    local_solution: jnp.ndarray,
    layout: _OutputComponentLayout,
) -> jnp.ndarray:
    n_local = int(projector.global_scalar_ids.shape[0])
    component_values = local_solution[:, layout.start:layout.stop]
    if len(layout.field_shape) == 0:
        local_values = component_values.reshape((n_local,))
    else:
        local_values = component_values.reshape((n_local,) + layout.field_shape)
    full_shape = (int(projector.full_shape[0]),) if len(layout.field_shape) == 0 else (int(projector.full_shape[0]),) + layout.field_shape
    full = jnp.full(full_shape, projector.outside_value, dtype=local_values.dtype)
    return full.at[projector.global_scalar_ids].set(local_values)


def _raise_nonpositive_time_span(_value) -> None:
    raise ValueError("time_span must be positive.")


def _vtu_step_file_name(step_index: int) -> str:
    return f"step_{int(step_index):06d}.vtu"


def _vtu_output_layout(result_folder_name: Optional[str]) -> tuple[str, str]:
    if result_folder_name is None:
        raise ValueError("set_postprocessing_policy(...) requires result_folder_name for VTU/PVD output.")
    folder = Path(result_folder_name)
    stem = folder.stem if folder.suffix == ".res" else folder.name
    if not stem:
        raise ValueError("result_folder_name must name a non-empty output folder.")
    return stem, str(folder)


def _prepare_mesh_for_vtu(mesh: meshio.Mesh) -> meshio.Mesh:
    points = np.asarray(mesh.points)
    if points.ndim == 2 and points.shape[1] == 2:
        mesh.points = np.concatenate([points, np.zeros((points.shape[0], 1), dtype=points.dtype)], axis=1)
    mesh.cell_sets = {}
    mesh.point_sets = {}
    return mesh


def _format_vtu_point_data(values: Any, n_points: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.shape[0] != n_points:
        raise ValueError(f"Expected point data with leading dimension {n_points}, got {arr.shape}.")
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if arr.shape[1] == 1:
            return arr.reshape((n_points,))
        if arr.shape[1] == 2:
            zeros = np.zeros((n_points, 1), dtype=arr.dtype)
            return np.concatenate([arr, zeros], axis=1)
        return arr.reshape((n_points, -1))
    if arr.ndim == 3:
        rows, cols = arr.shape[1], arr.shape[2]
        if rows <= 3 and cols <= 3:
            tensor = np.zeros((n_points, 3, 3), dtype=arr.dtype)
            tensor[:, :rows, :cols] = arr
            return tensor.reshape((n_points, 9))
        return arr.reshape((n_points, -1))
    return arr.reshape((n_points, -1))


def _format_vtu_point_data_jax(values: Any, n_points: int) -> jnp.ndarray:
    arr = jnp.asarray(values)
    if arr.shape[0] != n_points:
        raise ValueError(f"Expected point data with leading dimension {n_points}, got {arr.shape}.")
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if arr.shape[1] == 1:
            return arr.reshape((n_points,))
        if arr.shape[1] == 2:
            zeros = jnp.zeros((n_points, 1), dtype=arr.dtype)
            return jnp.concatenate([arr, zeros], axis=1)
        return arr.reshape((n_points, -1))
    if arr.ndim == 3:
        rows, cols = arr.shape[1], arr.shape[2]
        if rows <= 3 and cols <= 3:
            tensor = jnp.zeros((n_points, 3, 3), dtype=arr.dtype)
            tensor = tensor.at[:, :rows, :cols].set(arr)
            return tensor.reshape((n_points, 9))
        return arr.reshape((n_points, -1))
    return arr.reshape((n_points, -1))


def _write_pvd_file(path: str, records: list[tuple[float, str]]) -> None:
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>',
    ]
    for time_value, filename in records:
        lines.append(f'    <DataSet timestep="{time_value:.16g}" group="" part="0" file="{filename}"/>')
    lines += ['  </Collection>', '</VTKFile>', '']
    Path(path).write_text("\n".join(lines), encoding="utf-8")


class _PreparedVtuHostWriter:
    def __init__(
        self,
        *,
        mesh: meshio.Mesh,
        folder: str,
        file_stem: str,
        point_data_names: Tuple[str, ...],
        point_data_templates: Tuple[np.ndarray, ...],
    ):
        self.mesh = mesh
        self.folder = folder
        self.file_stem = file_stem
        self.point_data_names = tuple(point_data_names)
        self.records: list[tuple[float, str]] = []
        self.next_index = 0
        os.makedirs(self.folder, exist_ok=True)

        existing_point_data = getattr(self.mesh, "point_data", None)
        self.mesh.point_data = {} if existing_point_data is None else dict(existing_point_data)
        for name, template in zip(self.point_data_names, point_data_templates):
            self.mesh.point_data[name] = np.empty_like(np.asarray(template))

    def reset(self, _token=None) -> None:
        self.records = []
        self.next_index = 0

    def write(self, file_index, t, *point_data) -> None:
        written: set = set()
        for name, values in zip(self.point_data_names, point_data):
            arr = np.asarray(values)
            if name not in written:
                target = self.mesh.point_data.get(name)
                if target is None or target.shape != arr.shape or target.dtype != arr.dtype:
                    self.mesh.point_data[name] = np.array(arr, copy=True)
                else:
                    np.copyto(target, arr)
                written.add(name)
            else:
                # A later domain contributing to the same field: each per-domain
                # array is NaN outside its own domain, so fill the still-unset
                # (NaN) entries from this domain and keep already-filled ones.
                target = self.mesh.point_data[name]
                if np.issubdtype(arr.dtype, np.floating) and np.issubdtype(target.dtype, np.floating):
                    self.mesh.point_data[name] = np.where(np.isnan(target), arr, target)
                else:
                    np.copyto(target, arr)

        filename = _vtu_step_file_name(self.next_index)
        self.next_index += 1
        path = os.path.join(self.folder, filename)
        meshio.write(path, self.mesh, binary=True, compression=False)
        self.records.append((float(np.asarray(t)), filename))
        _write_pvd_file(os.path.join(self.folder, f"{self.file_stem}.pvd"), self.records)


def _scatter_vtu_output_component(
    global_scalar_ids: jnp.ndarray,
    n_points: int,
    outside_value: float,
    local_solution: jnp.ndarray,
    layout: _OutputComponentLayout,
) -> jnp.ndarray:
    n_local = int(global_scalar_ids.shape[0])
    component_values = local_solution[:, layout.start:layout.stop]
    if len(layout.field_shape) == 0:
        local_values = component_values.reshape((n_local,))
        full_shape = (n_points,)
    else:
        local_values = component_values.reshape((n_local,) + layout.field_shape)
        full_shape = (n_points,) + layout.field_shape
    full = jnp.full(full_shape, outside_value, dtype=local_values.dtype)
    return full.at[global_scalar_ids].set(local_values)


def _solve_vtu_projection_group(
    group: _VtuOutputGroup,
    runtime: _VtuOutputRuntime,
    q: Dict[str, jnp.ndarray],
    settings: Dict[str, Any],
    local_guess: jnp.ndarray,
    t,
) -> jnp.ndarray:
    projection_settings = dict(settings)
    source_dofs = projection_settings.pop(_VTU_OUTPUT_SOURCE_DOFS_KEY, q)
    projection_settings["current time"] = t
    projection_settings["node coordinates"] = runtime.node_coordinates
    projection_settings["connectivity"] = runtime.connectivity
    projection_settings["projection source connectivity"] = runtime.source_connectivity
    projection_settings["projection source dofs"] = source_dofs

    projection_dofs = {_PROJECTION_KEY: local_guess}
    residual = assembler.assemble_residual(
        projection_dofs,
        projection_settings,
        group.static_settings,
    )[_PROJECTION_KEY]
    rhs = residual.reshape((group.n_local, group.total_components))
    delta = _solve_l2_projector_components(
        runtime.mass_matrix,
        runtime.jacobi_inv_diag,
        rhs,
        tol=float(group.projection_tol),
        atol=float(group.projection_atol),
        maxiter=int(group.projection_maxiter),
    )
    if group.total_components == 1:
        delta = delta.reshape((group.n_local, 1))
    return local_guess + delta


def _vtu_group_point_data(
    group: _VtuOutputGroup,
    runtime: _VtuOutputRuntime,
    local_solution: jnp.ndarray,
) -> Tuple[jnp.ndarray, ...]:
    values = []
    for layout in group.layouts:
        component_values = _scatter_vtu_output_component(
            runtime.global_scalar_ids,
            group.n_points,
            group.outside_value,
            local_solution,
            layout,
        )
        values.append(_format_vtu_point_data_jax(component_values, group.n_points))
    return tuple(values)


class _VtuCallbackPolicy(dae.SavePolicy):
    def __init__(
        self,
        source_policy,
        writer: _PreparedVtuHostWriter,
        output_groups: Tuple[_VtuOutputGroup, ...],
    ):
        self.writer = writer
        self.output_groups = tuple(output_groups)
        self.kind = "all"
        self.num_points = None
        self.tol = 0.0

        if isinstance(source_policy, dae.SaveEquidistantPolicy):
            self.kind = "equidistant"
            self.num_points = source_policy.num_points
            self.tol = float(source_policy.tol)
        elif isinstance(source_policy, dae.SaveAllPolicy):
            self.kind = "all"
        elif not isinstance(source_policy, dae.SaveNothingPolicy):
            warnings.warn(
                f"Unsupported postprocessing policy '{type(source_policy).__name__}'. Falling back to dae.SaveAllPolicy semantics.",
                RuntimeWarning,
                stacklevel=3,
            )

    def initialize(self, q, t_start, t_final, num_time_steps, user_data={}):
        if self.kind == "equidistant":
            num_points = int(self.num_points) if self.num_points is not None else int(num_time_steps)
            target_times = jnp.linspace(t_start, t_final, num_points + 1)
        else:
            target_times = jnp.zeros((1,), dtype=jnp.asarray(t_final).dtype)

        output_locals = tuple(
            jnp.zeros((group.n_local, group.total_components), dtype=float)
            for group in self.output_groups
        )
        return _VtuCallbackPolicyState(
            target_times=target_times,
            current_save_idx=jnp.asarray(0, dtype=jnp.int32),
            next_target_idx=jnp.asarray(0, dtype=jnp.int32),
            output_locals=output_locals,
        )

    def _should_save(self, state: _VtuCallbackPolicyState, t):
        if self.kind == "all":
            return jnp.asarray(True)
        last_target_idx = state.target_times.shape[0] - 1
        safe_idx = jnp.minimum(state.next_target_idx, last_target_idx)
        in_range = state.next_target_idx <= last_target_idx
        return jnp.logical_and(in_range, t >= state.target_times[safe_idx] - self.tol)

    def _point_data(
        self,
        q: Dict[str, jnp.ndarray],
        settings: Dict[str, Any],
        output_locals: Tuple[jnp.ndarray, ...],
        t,
    ) -> Tuple[Tuple[jnp.ndarray, ...], Tuple[jnp.ndarray, ...]]:
        point_data = []

        new_output_locals = []
        runtimes = settings.get("_vtu_output_runtime", tuple())
        for group, runtime, local_guess in zip(self.output_groups, runtimes, output_locals):
            local_solution = _solve_vtu_projection_group(group, runtime, q, settings, local_guess, t)
            new_output_locals.append(local_solution)
            point_data.extend(_vtu_group_point_data(group, runtime, local_solution))

        return tuple(point_data), tuple(new_output_locals)

    def save_step(self, state, t, q, user_data={}):
        should_save = self._should_save(state, t)

        def do_save(current_state):
            point_data, output_locals = self._point_data(q, user_data, current_state.output_locals, t)
            jax.debug.callback(
                self.writer.write,
                current_state.current_save_idx,
                t,
                *point_data,
                ordered=True,
            )
            return _VtuCallbackPolicyState(
                target_times=current_state.target_times,
                current_save_idx=current_state.current_save_idx + 1,
                next_target_idx=current_state.next_target_idx + 1,
                output_locals=output_locals,
            )

        return jax.lax.cond(should_save, do_save, lambda current_state: current_state, state)

    def finalize(self, state):
        return None


def _call_natural_value_fun(value_fun: Callable, x: jnp.ndarray, t: Any, settings: Dict[str, Any]) -> Any:
    num_args = len(signature(value_fun).parameters)
    if num_args == 1:
        return value_fun(x)
    if num_args == 2:
        return value_fun(x, t)
    if num_args == 3:
        return value_fun(x, t, settings)
    raise ValueError("value_fun must take 1 to 3 arguments: x, optionally t and settings.")


def _call_ic_value_fun(value_fun: Callable, x: jnp.ndarray, settings: Dict[str, Any]) -> Any:
    num_args = len(signature(value_fun).parameters)
    if num_args == 1:
        return value_fun(x)
    if num_args == 2:
        return value_fun(x, settings)
    raise ValueError("IC value_fun must take 1 or 2 arguments: x, or x and settings.")


def _coerce_weak_bc_load(raw_value: Any, test_value: jnp.ndarray, index: Optional[int]) -> jnp.ndarray:
    test_value = jnp.asarray(test_value)
    raw = jnp.asarray(raw_value)

    if test_value.ndim == 0:
        if raw.shape != ():
            raise ValueError("Weak BC value_fun must return a scalar for scalar fields.")
        return raw

    if index is not None:
        if raw.shape != ():
            raise ValueError("Weak BC value_fun must return a scalar when index is specified.")
        load = jnp.zeros_like(test_value)
        return load.at[index].set(raw)

    if raw.shape != test_value.shape:
        raise ValueError(
            f"Weak BC value shape {raw.shape} does not match field/test-function shape {test_value.shape}. "
            "Return a matching traction/load vector or pass index=... for a scalar component load."
        )
    return raw


def _make_weak_bc_integrand(
    field: str,
    value_fun: Callable,
    index: Optional[int],
    *,
    trace: str,
    sign: float,
) -> Callable:
    if trace not in ("value", "normal"):
        raise ValueError("trace must be 'value' or 'normal'.")

    def integrand(ctx):
        x_phys = ctx.trial_ansatz["physical coor"](ctx.x_int)
        t = ctx.settings.get("current time", 0.0)
        raw_value = _call_natural_value_fun(value_fun, x_phys, t, ctx.settings)
        test_value = ctx.test_ansatz[field](ctx.x_int)
        if trace == "normal":
            if index is not None:
                raise ValueError("index is not supported for trace='normal'.")
            if jnp.asarray(test_value).ndim != 0:
                raise ValueError("trace='normal' requires a scalar normal trace basis.")
            load = jnp.asarray(raw_value)
            if load.shape != ():
                raise ValueError("Weak normal-trace BC value_fun must return a scalar.")
        else:
            load = _coerce_weak_bc_load(raw_value, test_value, index)
        return sign * jnp.sum(test_value * load)

    return integrand


def _solve_projected_dirichlet_runtime(
    runtime: _ProjectedDirichletRuntime,
    static_settings: Any,
    settings: Dict[str, Any],
    local_guess: jnp.ndarray,
    t: Any,
    *,
    tol: float,
    atol: float,
    maxiter: int,
) -> jnp.ndarray:
    projection_settings = dict(settings)
    projection_settings.pop(_PROJECTED_DIRICHLET_RUNTIME_KEY, None)
    projection_settings["current time"] = t
    projection_settings["node coordinates"] = runtime.node_coordinates
    projection_settings["connectivity"] = runtime.connectivity
    projection_settings["orientation"] = runtime.orientation
    projection_settings["projection source connectivity"] = runtime.source_connectivity
    projection_settings["projection source dofs"] = {}

    projection_dofs = {_PROJECTION_KEY: local_guess}
    residual = assembler.assemble_residual(
        projection_dofs,
        projection_settings,
        static_settings,
    )[_PROJECTION_KEY]
    rhs = residual.reshape(-1)
    delta = _solve_l2_projector_components(
        runtime.mass_matrix,
        runtime.jacobi_inv_diag,
        rhs,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
    )
    return local_guess + delta.reshape(local_guess.shape)
