# autopdex/sim_state/state.py
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

"""Orchestrator class :class:`SimState` (AutoPDEx high-level interface).

The free topology/DOF/projection helpers live in the sibling modules
``mesh_topology``, ``mesh_info``, ``dofs`` and ``projection``; the full
``SimState`` state machine with its methods remains here.
"""

from fnmatch import fnmatch
from pathlib import Path
import os
import copy
from inspect import signature
from typing import Any, Callable, Dict, Optional, List, Tuple, Union
from zlib import crc32
from dataclasses import dataclass, replace
import math
import warnings
from collections import defaultdict

import numpy as np
import meshio
import jax
import jax.numpy as jnp

from flax.core import FrozenDict

from autopdex import assembler, dae, mesher, models, seeder, spaces, utility

from .mesh_topology import (
    _BASE_DIM,
    _source_mesh_snapshot,
    _add_facet_domain,
    _canonical_base_type,
    _cell_dim,
    _convert_degenerate_linear_quads_to_triangles,
    _deduplicate_mesh_points_exact,
    _find_boundary_facets,
    _normalize_mesh_cell_node_ordering,
)
from .mesh_info import (
    _CellInfo,
    _DomainInfo,
    _MeshInfo,
    _add_domain_to_mesh_info,
    _build_mesh_info,
    _collect_elem_ids_by_type,
    _collocation_integration_order_from_poly_order,
    _default_integration_order_from_spaces,
    _infer_field_dimensions,
    _mesh_type_from_topology,
    _normalize_integration_mode,
    _pick_projection_integration,
    _pick_quadrature,
    _reduce_points,
    _tensor_product_order_from_topology,
    _unique1d_int_fast,
)
from .dofs import (
    _as_list,
    _build_dofs_and_connectivity_all_fields,
    _field_shape_from_space,
    _orientation_for_space_on_topology,
    _space_basis_for_field,
    _space_has_surface_dofs,
    _space_surface_basis_for_field,
    _split_spatial_discretizations,
    _zero_internal_variable_block,
)
from .projection import (
    _L2ProjectionSetInfo,
    _OrderedOutputValues,
    _OutputComponentLayout,
    _PROJECTED_DIRICHLET_RUNTIME_KEY,
    _PROJECTED_IC_RUNTIME_KEY,
    _PROJECTION_KEY,
    _PreparedL2Projection,
    _PreparedVtuHostWriter,
    _ProjectedDirichletRuntime,
    _ProjectedICRuntime,
    _VTU_OUTPUT_SOURCE_DOFS_KEY,
    _VtuCallbackPolicy,
    _VtuOutputGroup,
    _VtuOutputRuntime,
    _apply_callable_ic_projections,
    _build_surface_connectivity_for_field,
    _call_boundary_selector,
    _call_output_dict_fun,
    _component_mask,
    _flatten_projector_values,
    _format_vtu_point_data,
    _make_grouped_output_quantity_fun,
    _make_initial_value_quantity_fun,
    _make_l2_projection_model,
    _make_weak_bc_integrand,
    _merge_projector_scalar_ids,
    _normalize_output_domains,
    _num_flat_components,
    _output_component_layouts,
    _prepare_mesh_for_vtu,
    _raise_nonpositive_time_span,
    _require_model_context_fun,
    _reshape_projector_coefficients,
    _scatter_output_components,
    _selected_output_domains_for_model,
    _solve_l2_projector_components,
    _solve_projected_dirichlet_runtime,
    _vtu_output_layout,
    _vtu_step_file_name,
    _wrap_projected_value_fun,
    _call_natural_value_fun,
    _coerce_projected_value,
)


# Name of the internal domain holding *all* auto-detected boundary facets. Built
# once from the source mesh and shared across strong-BC projectors (see
# SimState._ensure_all_boundary_mesh_info).
_ALL_BOUNDARY_FACET_DOMAIN = "__all_boundary_facets__"


def _is_restart_state(obj: Any) -> bool:
    return isinstance(obj, dae.TimeSteppingManagerState)


def _clear_restart_settings(settings: Dict[str, Any]) -> None:
    for key in (
        dae._RESTART_Q_N_KEY,
        dae._RESTART_Q_DER_N_KEY,
        dae._RESTART_CONTROLLER_STATE_KEY,
        dae._RESTART_DT_KEY,
    ):
        settings.pop(key, None)

def _validate_field_shaped_value(
    value: Any,
    field_shape: Tuple[int, ...],
    field_name: str,
    arg_name: str,
) -> jnp.ndarray:
    arr = jnp.asarray(value)
    if tuple(arr.shape) != tuple(field_shape):
        raise ValueError(
            f"{arg_name} for field '{field_name}' must have field shape {field_shape}, "
            f"got {arr.shape}."
        )
    return arr

def _normalize_dof_scaling(static_settings: Dict[str, Any], scales: Optional[Dict[str, Any]]) -> Dict[str, jnp.ndarray]:
    if scales is None:
        return {}
    if not isinstance(scales, dict):
        raise TypeError("scales must be a dict mapping field names to positive scale factors.")

    spaces_by_field = dict(static_settings.get("spatial discretizations", {}))
    unknown = sorted(set(scales.keys()) - set(spaces_by_field.keys()))
    if unknown:
        raise KeyError(f"dof scaling contains unknown field(s) {unknown}. Available: {sorted(spaces_by_field)}")

    normalized = {}
    for field in sorted(scales.keys()):
        field_shape = _field_shape_from_space(spaces_by_field[field])
        arr = np.asarray(scales[field], dtype=float)
        if tuple(arr.shape) not in ((), field_shape):
            raise ValueError(
                f"dof scaling for field '{field}' must be scalar or have field shape {field_shape}, got {arr.shape}."
            )
        if not np.all(np.isfinite(arr)) or not np.all(arr > 0.0):
            raise ValueError(f"dof scaling for field '{field}' must contain positive finite values.")
        normalized[field] = jnp.asarray(arr.item() if arr.shape == () else arr, dtype=float)
    return normalized

def _eval_nodal_bc_values(value_fun, coords, t, settings, field, field_shape, unscale=True):
    """Evaluate a nodal BC ``value_fun`` at each node coordinate.

    Returns an array of shape ``(n_nodes, *field_shape)``. With ``unscale=True``
    the result is in stored-DOF units (Dirichlet values, matching the projected
    path); with ``unscale=False`` it is left in physical units (nodal loads,
    which are added to the physical residual).
    """
    flat = _num_flat_components(field_shape)

    def one(x):
        v = _call_natural_value_fun(value_fun, x, t, settings)
        coerced = _coerce_projected_value(v, field_shape, flat)
        return utility.unscale_dof_value(settings, field, coerced) if unscale else coerced

    return jax.vmap(one)(coords)

def _normalize_domain_field_map(
    values: Optional[Dict[str, Any]],
    field_names: Tuple[str, ...],
    arg_name: str,
) -> Tuple[Tuple[str, Dict[str, Any]], ...]:
    if values is None:
        return tuple()
    if not isinstance(values, dict):
        raise TypeError(f"{arg_name} must be a dict.")

    field_name_set = set(field_names)
    if set(values.keys()).issubset(field_name_set):
        return (("__all__", values),)

    normalized = []
    for domain_name, per_field in values.items():
        if not isinstance(domain_name, str):
            raise TypeError(f"{arg_name} domain names must be strings.")
        if not isinstance(per_field, dict):
            raise TypeError(f"{arg_name}['{domain_name}'] must be a dict of field values.")
        unknown = sorted(set(per_field.keys()) - field_name_set)
        if unknown:
            raise KeyError(
                f"{arg_name} for domain '{domain_name}' contains unknown field(s) {unknown}. "
                f"Available: {sorted(field_name_set)}"
            )
        normalized.append((domain_name, per_field))
    return tuple(normalized)

def _init_ephemeral_state(obj: "SimState") -> None:
    """Set all ephemeral (non-pytree) attributes of a SimState to their defaults.

    Called by both SimState.__init__ and SimState._tree_unflatten so that the
    two code paths cannot diverge.  Attributes whose values differ between the
    two paths (settings, dofs, manager, _phase, …) are set by the respective
    caller after this function returns.
    """
    obj._fe = None
    obj._fe_signature = None
    obj.mesh_info = None
    obj.dimension = None
    obj._source_mesh = None
    obj._requested_cell_data_keys: Tuple[str, ...] = tuple()
    obj._l2_projectors: Dict[str, _PreparedL2Projection] = {}
    obj._geometry_projection_core_cache: Dict[Tuple[Any, ...], str] = {}
    obj._projection_source_cache: Dict[Tuple[Any, ...], Any] = {}
    obj._output_projection_last_local: Dict[Tuple[Any, ...], jnp.ndarray] = {}
    obj._output_projection_model_cache: Dict[Tuple[Any, ...], Any] = {}
    obj._raw_local_subsystem_specs: Tuple[FrozenDict, ...] = tuple()
    obj._raw_model_specs: Tuple[FrozenDict, ...] = tuple()
    obj._materialized_model_specs = 0
    obj._raw_projected_dirichlet_specs: Tuple[FrozenDict, ...] = tuple()
    obj._projected_dirichlet_specs: Tuple[FrozenDict, ...] = tuple()
    obj._nodal_dirichlet_specs: Tuple[FrozenDict, ...] = tuple()
    obj._nodal_load_specs: Tuple[FrozenDict, ...] = tuple()
    obj._raw_output_specs: Tuple[FrozenDict, ...] = tuple()
    obj._prepared_output_specs: Tuple[FrozenDict, ...] = tuple()
    obj._base_apply_dirichlet_bcs = None
    obj._output_domains: Optional[Tuple[str, ...]] = None
    obj._output_integration_order: Optional[int] = None
    obj._output_int_pts_and_weights: Optional[Dict[str, Any]] = None
    obj._output_integration_mode = "auto"
    obj._output_projection_tol = 1e-8
    obj._output_projection_atol = 1e-10
    obj._output_projection_maxiter = 1000
    obj._vtu_postprocessing_policy = None
    obj._vtu_postprocessing_file_stem = None
    obj._vtu_postprocessing_result_folder_name = None
    obj._vtu_postprocessing_output_folder = None
    obj._manual_vtu_file_index = 0
    obj._base_post_step_updates = None
    obj._vtu_postprocessing_callback = None
    obj._vtu_postprocessing_callback_installed = False
    obj._active_vtu_postprocessing_state = None
    obj._step_size_controller = dae.ConstantStepSizeController()
    obj._pending_initial_condition_specs = []
    obj._pending_restart_state = None
    obj._run_result = None
    obj._phase = "setup"
    obj._source_mesh_modified = False


class SimState:
    """AutoPDEx high-level interface (HLI): orchestrator of the simulation workflow.

    ``SimState`` encapsulates the complete simulation workflow (mesh import ->
    domain/BC/model registration -> DOF numbering -> assembly setup ->
    time integration) as a state machine. The object is
    registered as a JAX pytree and the run method is transformable via ``jit``, ``jacfwd`` or ``jacrev``.
    """

    ModelContext = models.ModelContext

    def __init__(self, spatial_discretizations: Dict[str, Any]):
        """
        Parameters
        ----------
        spatial_discretizations:
            Dict mapping field name -> FE space object (e.g. Lagrange(...)).
            These are treated as global / fixed for the SimState instance.
        """
        if "physical coor" in spatial_discretizations:
            raise ValueError("'physical coor' is reserved for the geometry mapping and must not be used as a field name.")

        spatial_discretizations = dict(sorted(spatial_discretizations.items()))
        primary_spaces, internal_variable_spaces = _split_spatial_discretizations(spatial_discretizations)
        self.settings: Dict[str, Any] = {}
        self.static_settings: Dict[str, Any] = {
            "spatial discretizations": primary_spaces,
            "internal variable discretizations": internal_variable_spaces,
            "field dimensions": _infer_field_dimensions(primary_spaces),
            "internal variable dimensions": _infer_field_dimensions(internal_variable_spaces),
        }
        self.dofs: Dict[str, jnp.ndarray] = {}
        self.manager = None
        _init_ephemeral_state(self)

    @staticmethod
    def _tree_flatten(obj):
        children = (obj.settings, obj.dofs, getattr(obj, "_run_result", None))
        manager = obj.manager
        manager_signature = None if manager is None else tuple(
            id(getattr(manager, name, None))
            for name in ("root_solver", "save_policy", "step_size_controller")
        )
        aux_data = (
            obj.static_settings,
            manager,
            manager_signature,
            getattr(obj, "_phase", "setup"),
            getattr(obj, "_requested_cell_data_keys", tuple()),
            getattr(obj, "_step_size_controller", dae.ConstantStepSizeController()),
            )
        return children, aux_data

    @staticmethod
    def _tree_unflatten(aux_data, children):
        settings, dofs, run_result = children
        (
            static_settings,
            manager,
            _manager_signature,
            phase,
            requested_cell_data_keys,
            step_size_controller,
        ) = aux_data
        obj = object.__new__(SimState)
        _init_ephemeral_state(obj)
        obj.settings = settings
        obj.static_settings = static_settings
        obj.dofs = dofs
        obj.manager = manager
        obj._phase = phase
        obj._requested_cell_data_keys = requested_cell_data_keys
        obj._step_size_controller = step_size_controller
        obj._run_result = run_result
        return obj

    @property
    def q(self): return self.dofs

    def _run_attr(self, name, default=None):
        return default if self._run_result is None else getattr(self._run_result, name)

    @property
    def history(self): return self._run_attr("history")

    @property
    def dt(self): return self._run_attr("dt")

    @property
    def controller_state(self): return self._run_attr("controller_state")

    @property
    def num_steps(self): return self._run_attr("num_steps", 0)

    @property
    def num_accepted(self): return self._run_attr("num_accepted", 0)

    @property
    def num_rejected(self): return self._run_attr("num_rejected", 0)

    def set_active_domains(self, active_domains: Dict[str, Union[str, list[str]]]):
        """
        Restrict each field to a subset of volume domains.

        A field only lives (has DOFs) on the domains listed for it, and only
        participates in models registered on those domains. Fields not listed
        default to '__all__'.

        Example
        -------
        sim.set_active_domains({
            "displacement": "__all__",   # lives everywhere
            "pressure": "right",         # lives only on the 'right' subdomain
        })

        Notes
        -----
        A field must be either fully present or fully absent on the domain of
        every model it enters (align model domains with active domains).
        """
        return self._set_active_domains(active_domains)

    def _set_active_domains(self, active_domains: Dict[str, Union[str, list[str]]]):
        """
        Set per-field active *volume* domains used to build global DOF numbering.

        Example
        -------
        sim._set_active_domains({
            "displacement": ["set-A", "set-B"],
            "temperature": "set-A",
        })

        Notes
        -----
        If not set, the default is '__all__' for each field.
        """
        self._require_mesh_editable("_set_active_domains")
        internal_fields = set(self.static_settings.get("internal variable discretizations", {}).keys())
        requested_internal = sorted(internal_fields.intersection(active_domains.keys()))
        if requested_internal:
            raise ValueError(
                "Internal variables do not use active volume domains: "
                f"{requested_internal}."
            )
        self.static_settings["active domains"] = dict(active_domains)
        self._reset_discretization_dependent_caches()

    def import_mesh(
        self,
        mesh,
        deduplicate_points: bool = True,
        convert_degenerate_quads: bool = False,
        cell_data_keys: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
    ):
        """
        Load mesh data from a meshio mesh object.

          - creates a default domain '__all__' containing all cells of the highest
            topological dimension
          - if `cell_data_keys` is given, makes those entries available later as
            `settings["cell data"][set][key][elem_number]` inside weak forms
        """
        # Side effects:
        #   - stores reduced node coordinates in `self.settings["node coordinates"]`
        #   - stores _MeshInfo in `self.mesh_info`

        self._require_mesh_editable("import_mesh")
        if self.mesh_info is not None:
            raise NotImplementedError("Multiple meshes are not supported.")

        mesh = _normalize_mesh_cell_node_ordering(mesh)
        mesh.cell_sets.pop("gmsh:bounding_entities", None)
        if deduplicate_points:
            mesh = _deduplicate_mesh_points_exact(mesh)
        mesh = _convert_degenerate_linear_quads_to_triangles(
            mesh,
            enabled=convert_degenerate_quads,
        )
        if cell_data_keys is None:
            requested_cell_data_keys = tuple()
        elif isinstance(cell_data_keys, str):
            requested_cell_data_keys = (cell_data_keys,)
        else:
            requested_cell_data_keys = tuple(cell_data_keys)
        if any((not isinstance(key, str)) or (key == "") for key in requested_cell_data_keys):
            raise TypeError("cell_data_keys must be a string or a sequence of non-empty strings.")
        requested_cell_data_keys = tuple(dict.fromkeys(requested_cell_data_keys))
        cell_data_dict = getattr(mesh, "cell_data_dict", {}) or {}
        for key in requested_cell_data_keys:
            if key not in cell_data_dict:
                raise KeyError(f"Cell data key '{key}' not found in mesh.")
        mesh_info = _build_mesh_info(mesh)
        pts_red, n_dim = _reduce_points(np.asarray(mesh.points))
        self.dimension = n_dim
        mesh_info = type(mesh_info)(points=pts_red, domains=mesh_info.domains, cells=mesh_info.cells)

        cell_dims = {
            ct: _cell_dim(ct)
            for ct, cinfo in mesh_info.cells.items()
            if int(cinfo.nodes_of_cells.shape[0]) > 0
        }
        top_dim = max(cell_dims.values()) if cell_dims else -1
        all_dom: Dict[str, _DomainInfo] = {}
        for ct, cinfo in mesh_info.cells.items():
            n = int(cinfo.nodes_of_cells.shape[0])
            if n > 0 and cell_dims.get(ct) == top_dim:
                all_dom[ct] = _DomainInfo(cell_indices=jnp.arange(n, dtype=int))
        mesh_info.domains.setdefault("__all__", all_dom)

        self.mesh_info = mesh_info
        self._source_mesh = _source_mesh_snapshot(mesh)
        self._requested_cell_data_keys = requested_cell_data_keys
        self.settings["node coordinates"] = mesh_info.points
        self.settings.pop("cell data", None)
        self._reset_discretization_dependent_caches()
        self._materialized_model_specs = 0

    def add_structured_mesh(self, n_elements, vertices, element_type, order=1):
        """
            Generate a structured 2D or 3D mesh over a quadrilateral or hexahedral domain,
            with an optional subdivision into simplex (triangular/tetrahedral) elements.

            Parameters
            ----------
            n_elements : tuple of int
                For 2D, a tuple (nx, ny) specifying the number of elements along the x-
                and y-directions. For 3D, a tuple (nx, ny, nz).
            vertices : array-like
                For 2D, a 4x2 array; for 3D, an 8x3 array defining the coordinates of the
                domain's corner points.
            element_type : str
                For 2D, either "quad" or "tri"; for 3D, either "brick" or "tet".
            order : int, optional
                The polynomial order of the elements. Currently, only linear elements
                (order == 1) are supported. Default is 1.

            Returns
            -------
            Updated SimState object.

            Notes
            -----
            For 2D:
                - The vertices should be provided as a 4x2 array in anti-clockwise order.
                - The mapping from a reference square ([-1, 1] x [-1, 1]) to the physical
                  domain is performed using bilinear interpolation.
            
            For 3D:
                - The vertices should be provided as an 8x3 array, with the ordering corresponding
                  to a standard hexahedron (e.g., starting with (-1, -1, -1) and proceeding in an
                  anti-clockwise fashion on the bottom face, then defining the top face).
                - The mapping from a reference cube ([-1, 1]^3) to the physical domain is done via
                  trilinear interpolation.

            Currently, only linear (order == 1) elements are supported.
        """
        meshio_cell_type_by_element_type = {
            "quad": "quad",
            "tri": "triangle",
            "brick": "hexahedron",
            "tet": "tetra",
        }
        if element_type not in meshio_cell_type_by_element_type:
            raise ValueError(
                "element_type must be one of "
                f"{sorted(meshio_cell_type_by_element_type)}."
            )

        coords, elements = mesher.structured_mesh(
            n_elements, vertices, element_type, order=order
        )
        mesh = meshio.Mesh(
            points=np.asarray(coords),
            cells=[(
                meshio_cell_type_by_element_type[element_type],
                np.asarray(elements, dtype=int),
            )],
        )
        self.import_mesh(mesh)

    def add_domain(
        self,
        domain_name: str,
        on_domain_fun: callable,
        element_type: Optional[Union[str, list[str]]] = None,
    ):
        """
        Group existing cells into a named (sub)domain via a coordinate predicate.

        ``on_domain_fun(p)`` is a pointwise predicate on a node coordinate; a cell
        joins the domain when the predicate holds for all its nodes. The resulting
        domain can be targeted by :meth:`add_model` and :meth:`set_active_domains`.

        See :meth:`_add_domain` for the full parameter description.
        """
        return self._add_domain(domain_name, on_domain_fun, element_type=element_type)

    def _add_domain(
        self,
        domain_name: str,
        on_domain_fun: callable,
        element_type: Optional[Union[str, list[str]]] = None,
    ):
        """
        Add a domain by grouping existing cells via a predicate on node coordinates.

        Parameters
        ----------
        domain_name:
            Name of the domain to be added.
        on_domain_fun:
            Pointwise predicate that is called with a single node coordinate p of shape (gdim,).
            It must return a boolean (or a boolean array reducible to a scalar).
            Internally, the predicate is vectorized over nodes and elements.
        element_type:
            Filter for cell types:
              - None: consider all types
              - exact string: e.g. "triangle6"
              - wildcard: e.g. "triangle*", "quad*"
              - list of strings/patterns: e.g. ["triangle*", "quad4"]

        Notes
        -----
        This only *groups* cells; it does not modify connectivity or cell data.
        """
        self._require_mesh_editable("_add_domain")
        if self.mesh_info is None:
            raise RuntimeError("Call import_mesh(...) before _add_domain(...).")

        self.mesh_info = _add_domain_to_mesh_info(
            self.mesh_info,
            domain_name=domain_name,
            on_domain_fun=on_domain_fun,
            element_type=element_type,
        )

    def _ensure_fe_built(self):
        """
        Build global DOF containers and connectivity tables for all fields once.

        Uses:
          - `self.mesh_info` (must exist)
          - `self.static_settings["spatial discretizations"]` (fixed at construction)
          - `self.static_settings["active domains"]` (optional; defaults to '__all__' for each field)

        Stores:
          - per-field dof arrays in `self.dofs[field]`
          - full FE pack in `self._fe`
        """
        if self.mesh_info is None:
            raise RuntimeError("Call import_mesh(...) before building FE data.")

        spaces_by_field: Dict[str, Any] = dict(self.static_settings.get("spatial discretizations", {}))
        if not spaces_by_field:
            raise RuntimeError("No spatial discretizations are configured.")

        active_domains = dict(self.static_settings.get("active domains", {}))
        if not active_domains:
            active_domains = {f: "__all__" for f in spaces_by_field.keys()}

        # Signature for cache invalidation.
        sig = tuple(
            sorted(
                (
                    f,
                    int(getattr(sp, "order", 0)),
                    int(getattr(sp, "dim", 0)),
                    str(getattr(sp, "field_dimension", "")),
                    tuple(active_domains[f]) if isinstance(active_domains.get(f), list) else active_domains.get(f),
                )
                for f, sp in spaces_by_field.items()
            )
        )
        if self._fe_signature == sig:
            return

        self._fe = _build_dofs_and_connectivity_all_fields(self.mesh_info, spaces_by_field, active_domains)

        # Publish dof containers.
        for f, pack in self._fe.items():
            self.dofs[f] = pack.dofs

        self._fe_signature = sig

    def add_local_subsystem(
        self,
        key: str,
        local_dae: Callable,
        fields: Union[str, List[str], Tuple[str, ...]],
        root_solver: Callable = dae.newton_solver,
    ):
        """
        Register a local (element-level) DAE subsystem for internal variables.

        The subsystem is solved on demand from a model integrand via
        ``ModelContext.solve_local(key, ...)`` and is materialized in
        ``initialize(...)``.

        Parameters
        ----------
        key:
            Unique, non-empty name used to invoke the subsystem.
        local_dae:
            Callable ``(q_fun, t, settings)`` defining the local residual/DAE.
        fields:
            Internal variable field name(s) the subsystem solves for. All must
            be registered as internal variables (not primary fields) and have a
            time integrator set via ``add_temporal_discretization(...)``.
        root_solver:
            Root solver for the local system (default: ``dae.newton_solver``).
        """
        self._enter_registration_phase("add_local_subsystem")
        if not isinstance(key, str) or key == "":
            raise ValueError("Local subsystem key must be a non-empty string.")
        if any(spec["key"] == key for spec in self._raw_local_subsystem_specs):
            raise ValueError(f"Local subsystem '{key}' is already registered.")
        if len(signature(local_dae).parameters) != 3:
            raise ValueError("local_dae must take exactly (q_fun, t, settings).")
        if not callable(root_solver):
            raise TypeError("root_solver must be callable.")

        field_names = tuple(_as_list(fields))
        if len(field_names) == 0:
            raise ValueError("fields must contain at least one internal variable.")
        if len(set(field_names)) != len(field_names):
            raise ValueError("fields must not contain duplicates.")

        internal_spaces = dict(self.static_settings.get("internal variable discretizations", {}))
        primary_spaces = dict(self.static_settings.get("spatial discretizations", {}))
        unknown = sorted(set(field_names) - set(internal_spaces.keys()) - set(primary_spaces.keys()))
        if unknown:
            raise KeyError(
                f"Unknown local subsystem field(s) {unknown}. "
                f"Available internal variables: {sorted(internal_spaces.keys())}"
            )
        non_internal = sorted(set(field_names).intersection(primary_spaces.keys()))
        if non_internal:
            raise ValueError(f"Local subsystem fields must be internal variables, got {non_internal}.")

        spec = FrozenDict({
            "key": key,
            "local_dae": local_dae,
            "fields": field_names,
            "root_solver": root_solver,
        })
        self._raw_local_subsystem_specs = tuple(self._raw_local_subsystem_specs) + (spec,)

    def _materialize_local_subsystem_solvers(self) -> None:
        specs = tuple(self._raw_local_subsystem_specs)
        if len(specs) == 0:
            return
        internal_integrators = dict(self.static_settings.get("internal variable time integrators", {}))
        solvers = {}
        for spec in specs:
            fields = tuple(spec["fields"])
            missing = [field for field in fields if field not in internal_integrators]
            if missing:
                raise ValueError(
                    f"Local subsystem '{spec['key']}' requires time integrators for {missing}. "
                    "Set them with add_temporal_discretization(...)."
                )
            solvers[spec["key"]] = dae._make_local_subsystem_solver(
                spec["local_dae"],
                fields,
                internal_integrators,
                root_solver=spec["root_solver"],
            )
        self.static_settings["local subsystem solvers"] = FrozenDict(solvers)

    def add_model(
        self,
        domain_name: Union[str, list[str]],
        domain_type: str,
        integrand_fun: Callable,
        integration_order: Optional[int] = None,
        int_pts_and_weights: Optional[Dict[str, Any]] = None,
        internal_variables: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
        internal_variable_update_fun: Optional[Callable] = None,
    ):
        """
        Register a model (weak form or potential) on one or more domains.

        The contribution is integrated over the named domain(s); connectivity and
        quadrature kernels are built lazily in ``initialize(...)``.

        Parameters
        ----------
        domain_name:
            Domain name or list of names, looked up in ``mesh_info.domains``.
        domain_type:
            ``"weak form"`` for a residual integrand, or ``"potential"`` for a
            scalar potential/energy integrand that is differentiated automatically.
        integrand_fun:
            ``ModelContext`` callable returning the integrand (see ``models``).
        integration_order:
            Quadrature order; if None a default is derived from the field spaces.
        int_pts_and_weights:
            Optional quadrature: None (defaults), a dict keyed by cell type
            (e.g. ``"triangle6"`` or base ``"triangle"``), or an explicit
            ``(xi, w)`` pair.
        internal_variables:
            Internal variable field name(s) this model reads/updates, if any.
        internal_variable_update_fun:
            Optional ``ModelContext`` callable that updates those internal
            variables after a step; requires ``internal_variables``.
        """
        self._enter_registration_phase("add_model")
        _require_model_context_fun(integrand_fun, "integrand_fun")
        if internal_variable_update_fun is not None:
            _require_model_context_fun(internal_variable_update_fun, "internal_variable_update_fun")
        spec = FrozenDict({
            "domain_name": domain_name,
            "domain_type": domain_type,
            "integrand_fun": integrand_fun,
            "integration_order": integration_order,
            "int_pts_and_weights": int_pts_and_weights,
            "internal_variables": None if internal_variables is None else tuple(_as_list(internal_variables)),
            "internal_variable_update_fun": internal_variable_update_fun,
        })
        self._raw_model_specs = tuple(self._raw_model_specs) + (spec,)

    def _materialize_model_specs(self) -> None:
        raw_specs = tuple(self._raw_model_specs)
        start = int(self._materialized_model_specs)
        if start >= len(raw_specs):
            if self._output_domains is not None and len(self._raw_output_specs) == 0:
                raise ValueError("domains must match at least one add_model(...) domain.")
            return
        self._ensure_fe_built()
        for spec in raw_specs[start:]:
            self._add_model_materialized(
                spec["domain_name"],
                spec["domain_type"],
                spec["integrand_fun"],
                integration_order=spec["integration_order"],
                int_pts_and_weights=spec["int_pts_and_weights"],
                internal_variables=spec["internal_variables"],
                internal_variable_update_fun=spec["internal_variable_update_fun"],
            )
            self._materialized_model_specs += 1
        if self._output_domains is not None and len(self._raw_output_specs) == 0:
            raise ValueError("domains must match at least one add_model(...) domain.")

    def _add_model_materialized(
        self,
        domain_name: Union[str, list[str]],
        domain_type: str,
        integrand_fun: Callable,
        integration_order: Optional[int] = None,
        int_pts_and_weights: Optional[Dict[str, Any]] = None,
        internal_variables: Optional[Tuple[str, ...]] = None,
        internal_variable_update_fun: Optional[Callable] = None,
    ):
        """
        Register a new model on a given domain (volume or surface).

        Parameters
        ----------
        domain_name:
            Domain name or list of domain names. Domains are looked up in mesh_info.domains.
        domain_type:
            Type of the model ('potential' or 'weak form').
        integrand_fun:
            User-defined integrand function used by the model generator (e.g. models.weak_form).
        integration_order:
            If None, a default order is chosen from the configured spaces.
        int_pts_and_weights:
            Optional quadrature schemes, either:
            - None: use defaults
            - dict: keyed by mesh_topology (e.g. "triangle6") or base mesh_type (e.g. "triangle")
            - otherwise: treated as already being (xi, w)

        Notes
        -----
        Connectivity is obtained from the prebuilt FE pack:
        - Volume: `fe[field].volume[cell_type]` subset by active element ids
        - Surface: `fe[field].surface[domain][cell_type]` (built using surface blocks and entity_id)

        The mapping connectivity for 'physical coor' always uses the raw mesh nodes for the selected elements.
        """
        if self.mesh_info is None:
            raise RuntimeError("Call import_mesh(...) before add_model(...).")
        _require_model_context_fun(integrand_fun, "integrand_fun")
        if internal_variable_update_fun is not None:
            _require_model_context_fun(internal_variable_update_fun, "internal_variable_update_fun")

        # Build FE cache if needed.
        self._ensure_fe_built()

        spaces_by_field: Dict[str, Any] = dict(self.static_settings["spatial discretizations"])
        if not spaces_by_field:
            raise RuntimeError("No spatial discretizations are configured.")
        internal_variable_spaces: Dict[str, Any] = dict(
            self.static_settings.get("internal variable discretizations", {})
        )
        internal_variable_names = tuple(() if internal_variables is None else internal_variables)
        if internal_variable_update_fun is not None and len(internal_variable_names) == 0:
            raise ValueError("internal_variables must be specified when internal_variable_update_fun is provided.")
        unknown_internal = sorted(set(internal_variable_names) - set(internal_variable_spaces.keys()))
        if unknown_internal:
            raise KeyError(
                f"Unknown internal variable field(s) {unknown_internal}. "
                f"Available: {sorted(internal_variable_spaces.keys())}"
            )

        # Normalize domain selection.
        # Important: static_settings["domains"] must be a flat tuple of hashable items
        # (JAX static args). Therefore, if multiple domains are passed, we create a
        # single flat string key for the assembled model block.
        if isinstance(domain_name, str):
            requested_domains = (domain_name,)
            domain_selection = domain_name
            domain_key = domain_name
        else:
            requested_domains = tuple(domain_name)
            if len(requested_domains) == 0:
                raise ValueError("domain_name must not be empty.")
            domain_selection = list(requested_domains)
            domain_key = " + ".join(requested_domains)

        # Collect element ids per cell type for the selected domain(s).
        elem_ids_by_type = _collect_elem_ids_by_type(self.mesh_info, domain_selection)
        if len(elem_ids_by_type) == 0:
            raise ValueError(f"No elements found for domain '{domain_name}'.")

        if integration_order is None:
            integration_order = _default_integration_order_from_spaces(spaces_by_field)
        selected_output_domains = _selected_output_domains_for_model(
            getattr(self, "_output_domains", None),
            requested_domains,
        )
        materialized_any = False

        # Set up one model per mesh topology in the selected domain(s).
        #
        # The integration dimension is a property of the *domain*: it is the
        # dimension of the domain's own cells (``elem_dim``). Each field then
        # participates relative to that dimension (see the per-field
        # classification in the field loop below), which is what allows fields on
        # genuinely different dimensions (e.g. a 2D membrane on the face of a 3D
        # block, or a 1D wire embedded in a 2D plate) to live on one mesh.
        for mesh_topology in sorted(elem_ids_by_type.keys()):
            mesh_type = _mesh_type_from_topology(mesh_topology)
            elem_ids = jnp.asarray(elem_ids_by_type[mesh_topology], dtype=int)

            base = _canonical_base_type(mesh_topology)
            if base not in _BASE_DIM:
                warnings.warn(f"Skipping unknown cell type '{mesh_topology}'.", RuntimeWarning, stacklevel=2)
                continue

            elem_dim = int(_BASE_DIM[base])

            # Quadrature scheme selection.
            if int_pts_and_weights is None:
                current_int_pts_and_weights = _pick_quadrature(mesh_type, int(integration_order))
            elif isinstance(int_pts_and_weights, dict):
                if mesh_topology in int_pts_and_weights:
                    current_int_pts_and_weights = int_pts_and_weights[mesh_topology]
                elif mesh_type in int_pts_and_weights:
                    current_int_pts_and_weights = int_pts_and_weights[mesh_type]
                else:
                    raise ValueError(f"No quadrature entry for '{mesh_topology}' or '{mesh_type}'.")
            else:
                current_int_pts_and_weights = int_pts_and_weights

            # Geometry mapping basis (autopdex mapping kernels).
            if mesh_type in ["line", "quad", "hexahedron"]:
                mapping_fun = spaces.fem_line_quad_brick
            elif mesh_type in ["triangle", "tetra"]:
                mapping_fun = spaces.fem_line_tri_tet
            else:
                raise NotImplementedError(f"Unknown mesh type '{mesh_type}'.")

            ansatz_funs = {"physical coor": mapping_fun}
            connectivity: Dict[str, Any] = {}
            orientation: Dict[str, Any] = {}
            cell_data: Dict[str, Any] = {}
            if self._requested_cell_data_keys:
                cell_data_dict = getattr(self._source_mesh, "cell_data_dict", {}) or {}
                elem_ids_np = np.asarray(elem_ids, dtype=int)
                for key in self._requested_cell_data_keys:
                    values_by_topology = cell_data_dict.get(key, {})
                    if mesh_topology in values_by_topology:
                        values = np.asarray(values_by_topology[mesh_topology])
                        cell_data[key] = jnp.asarray(values[elem_ids_np])

            # Geometry connectivity: raw nodes for selected elements (supports higher-order geometry).
            connectivity["physical coor"] = self.mesh_info.cells[mesh_topology].nodes_of_cells[elem_ids].astype(int)

            # Field connectivity: pulled from the FE pack. A field participates
            # in this model relative to the domain's integration dimension:
            #   space.dim == elem_dim      -> volume field (DOFs live on these cells)
            #   space.dim == elem_dim + 1  -> trace/surface field (weak-BC style)
            # Otherwise the field does not belong to this domain and is skipped.
            for field, space in spaces_by_field.items():
                pack = self._fe[field]
                field_dim = int(getattr(space, "dim", elem_dim))

                if field_dim == elem_dim:
                    cell_g2a = pack.g2a["cell_g2a"]
                    if mesh_topology not in cell_g2a or mesh_topology not in pack.volume:
                        # Field has no cells of this topology (it lives on a
                        # different-dimensional domain) -> not part of this model.
                        continue

                    # Check that elem_ids are in bounds
                    min_id = int(elem_ids.min())
                    max_id = int(elem_ids.max())
                    assert 0 <= min_id and max_id < cell_g2a[mesh_topology].shape[0], f"elem_ids out of bounds: min={min_id}, max={max_id}, valid=0..{cell_g2a[mesh_topology].shape[0]-1}"

                    rows = cell_g2a[mesh_topology][elem_ids]
                    valid = rows >= 0
                    n_active = int(jnp.count_nonzero(valid))
                    if n_active == 0:
                        # The field is inactive on the whole model domain (its
                        # active domains do not overlap this model's domain). It
                        # does not participate in this model block and is omitted
                        # from the assembled field coupling.
                        continue
                    if n_active != int(valid.shape[0]):
                        # Partial overlap: some elements of the model domain carry
                        # this field, others do not. Per-element blocks require a
                        # consistent field set, so this must be all-or-nothing.
                        raise ValueError(
                            f"[{field}] is active on only {n_active} of "
                            f"{int(valid.shape[0])} elements of model domain "
                            f"'{domain_name}'. A field must be either fully present "
                            f"or fully absent on a model's domain; align the model "
                            f"domain with the field's active domains."
                        )
                    vol = pack.volume[mesh_topology]
                    n_vol = vol.shape[0]
                    if rows.shape[0] == n_vol and bool(jnp.all(rows == jnp.arange(n_vol))):
                        # Fully-active domain: pass the reference instead of gathering a full copy
                        # (JAX arrays are immutable, so sharing is safe).
                        connectivity[field] = vol
                    else:
                        connectivity[field] = vol[rows]

                    # Basis wrapper: map base cell type to volume basis.
                    b = _canonical_base_type(mesh_topology)
                    signs = _orientation_for_space_on_topology(
                        self.mesh_info,
                        space,
                        mesh_topology,
                        elem_ids,
                        is_surface=False,
                    )
                    if signs is not None:
                        orientation[field] = signs

                    ansatz_funs[field] = _space_basis_for_field(space, b, field)

                elif field_dim == elem_dim + 1:
                    # Trace of a higher-dimensional field over this domain (surface
                    # integral, e.g. weak boundary conditions).
                    b = _canonical_base_type(mesh_topology)
                    chunks = []
                    for dn in requested_domains:
                        if dn in pack.surface and mesh_topology in pack.surface[dn]:
                            chunks.append(pack.surface[dn][mesh_topology])
                    if len(chunks) == 0:
                        if not _space_has_surface_dofs(space, mesh_topology):
                            connectivity[field] = jnp.zeros((elem_ids.shape[0], 0), dtype=int)
                        else:
                            # No registered trace connectivity for this field on
                            # this domain -> the field is not coupled here; skip it.
                            continue
                    else:
                        connectivity[field] = jnp.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]

                    signs = _orientation_for_space_on_topology(
                        self.mesh_info,
                        space,
                        mesh_topology,
                        elem_ids,
                        is_surface=True,
                    )
                    if signs is not None:
                        orientation[field] = signs

                    ansatz_funs[field] = _space_surface_basis_for_field(space, b, field)

                else:
                    # Field dimension unrelated to this domain -> not part of it.
                    continue

            # No field lives on (or traces onto) cells of this topology -> nothing
            # to assemble here; skip this topology.
            if not any(f in connectivity for f in spaces_by_field.keys()):
                continue

            # Element kernel selection.
            if domain_type == "potential":
                element_fun = models.potential
            elif domain_type == "weak form":
                element_fun = models.weak_form
            else:
                raise ValueError("domain_type must be 'potential' or 'weak form'.")

            model_fun = element_fun(
                integrand_fun,
                ansatz_funs,
                *current_int_pts_and_weights,
                internal_variable_names=internal_variable_names,
            )
            int_var_update_model = None
            if len(internal_variable_names) > 0:
                update_source = internal_variable_update_fun if internal_variable_update_fun is not None else integrand_fun
                int_var_update_model = models.internal_variable_updates(
                    update_source,
                    ansatz_funs,
                    *current_int_pts_and_weights,
                    internal_variable_names=internal_variable_names,
                )

            primary_connectivity_fields = [f for f in spaces_by_field.keys() if f in connectivity]
            if not primary_connectivity_fields:
                raise RuntimeError("Model connectivity contains no primary fields.")
            n_model_elems = int(connectivity[primary_connectivity_fields[0]].shape[0])
            n_qp = int(jnp.asarray(current_int_pts_and_weights[0]).shape[0])
            int_var_block = _zero_internal_variable_block(
                internal_variable_spaces,
                internal_variable_names,
                n_model_elems,
                n_qp,
                self.mesh_info.points.dtype,
            )

            # Static settings append.
            first_domain = len(self.static_settings.get("domains", tuple())) == 0
            if first_domain:
                self.static_settings["dae"] = "call pde"
                self.static_settings["domains"] = (domain_key,)
                self.static_settings["assembling mode"] = ("user residual",)
                self.static_settings["model"] = (model_fun,)
                self.static_settings["internal variable names"] = (internal_variable_names,)
                self.static_settings["int var updates"] = (int_var_update_model,)
            else:
                self.static_settings["domains"] += (domain_key,)
                self.static_settings["assembling mode"] += ("user residual",)
                self.static_settings["model"] += (model_fun,)
                self.static_settings["internal variable names"] += (internal_variable_names,)
                self.static_settings["int var updates"] += (int_var_update_model,)

            # Default time integrators.
            time_integrators = dict(self.static_settings.get("time integrators", {}))
            for f in sorted(self.static_settings["spatial discretizations"].keys()):
                time_integrators.setdefault(f, dae.NoTimeDerivative())
            self.static_settings["time integrators"] = time_integrators

            # Dynamic settings append.
            if first_domain:
                self.settings["connectivity"] = (connectivity,)
                self.settings["orientation"] = (orientation,)
                self.settings["internal variables"] = (int_var_block,)
                self.settings["internal variables n"] = (int_var_block,)
                if self._requested_cell_data_keys:
                    self.settings["cell data"] = (cell_data,)
                self.settings["current time"] = 0.0
                self.settings.setdefault("last time", 0.0)
            else:
                self.settings["connectivity"] += (connectivity,)
                self.settings["orientation"] = self.settings.get("orientation", tuple()) + (orientation,)
                self.settings["internal variables"] = self.settings.get("internal variables", tuple()) + (int_var_block,)
                self.settings["internal variables n"] = self.settings.get("internal variables n", tuple()) + (int_var_block,)
                if self._requested_cell_data_keys:
                    self.settings["cell data"] = self.settings.get("cell data", tuple()) + (cell_data,)
                self.settings.setdefault("last time", 0.0)
            materialized_any = True

        if materialized_any and len(selected_output_domains) > 0:
            output_integration_mode = _normalize_integration_mode(self._output_integration_mode)
            output_integration_order = (
                self._output_integration_order
                if self._output_integration_order is not None
                else (None if output_integration_mode == "collocation" else integration_order)
            )
            output_int_pts_and_weights = (
                self._output_int_pts_and_weights
                if self._output_int_pts_and_weights is not None
                else (None if output_integration_mode == "collocation" else int_pts_and_weights)
            )
            self._raw_output_specs = tuple(self._raw_output_specs) + (
                FrozenDict({
                    "domains": selected_output_domains,
                    "output_fun": integrand_fun,
                    "integration_order": output_integration_order,
                    "int_pts_and_weights": output_int_pts_and_weights,
                    "integration_mode": output_integration_mode,
                    "projection_tol": self._output_projection_tol,
                    "projection_atol": self._output_projection_atol,
                    "projection_maxiter": self._output_projection_maxiter,
                }),
            )

    def set_step_size_controller(self, controller=None):
        """
        Configure the dae.StepSizeController used by the global time stepping manager.
        """
        if controller is None:
            controller = dae.ConstantStepSizeController()
        if not isinstance(controller, dae.StepSizeController):
            raise TypeError("controller must be a dae.StepSizeController instance or None.")
        if self._phase in ("setup", "registering"):
            self._step_size_controller = controller
            return
        if self._phase not in ("initialized", "prepared", "ran"):
            raise RuntimeError("set_step_size_controller(...) cannot be called in the current SimState phase.")
        if self.manager is None:
            raise RuntimeError("Call initialize(...) before set_step_size_controller(...).")
        self._step_size_controller = controller
        self.manager.step_size_controller = controller
        self.settings.pop(dae._RESTART_CONTROLLER_STATE_KEY, None)
        self.settings.pop(dae._RESTART_DT_KEY, None)

    def set_dof_scaling(self, scales: Optional[Dict[str, Any]] = None):
        """
        Configure constant per-field DOF scaling.

        Values passed to models, initial conditions, essential boundary conditions,
        and output stay physical. Stored primary DOFs are divided by these factors.
        Missing fields use scale 1.0.
        """
        if self._phase not in ("setup", "registering"):
            raise RuntimeError("set_dof_scaling(...) must be called before initialize(...).")
        normalized = _normalize_dof_scaling(self.static_settings, scales)
        if normalized:
            self.settings[utility.DOF_SCALING_KEY] = normalized
        else:
            self.settings.pop(utility.DOF_SCALING_KEY, None)

    def set_root_solver(self, root_solver):
        """
        Configure the root solver used by the global time stepping manager. Default is dae.newton_solver.
        """
        if not callable(root_solver):
            raise TypeError("root_solver must be callable.")
        if self._phase != "initialized":
            raise RuntimeError("set_root_solver(...) must be called after initialize(...) and before prepare(...).")
        if self.manager is None:
            raise RuntimeError("Call initialize(...) before set_root_solver(...).")
        self.manager.root_solver = root_solver

    def set_initial_conditions(
        self,
        values=None,
        *,
        derivatives=None,
        internal_variables=None,
        time=None,
        integration_mode: Optional[str] = None,
    ):
        """
        Set domain-wise initial values or restart from a TimeSteppingManagerState.

        ``values`` accepts either field-shaped arrays or callables L2-projected into
        the FE space.  Callable signatures: ``lambda x`` or ``lambda x, settings``.
        The 2-argument form receives the full ``settings`` dict, allowing access to
        user-defined parameters (e.g. ``sim.settings["u0"] = ...``) for sensitivity
        analysis.

        Args:
            values: Field-shaped initial values, callables ``f(x)`` / ``f(x, settings)``.
            derivatives: Optional field-shaped initial time derivatives by order.
            internal_variables: Optional initial values for internal variables.
            time: Absolute initial or restart time. When provided, both
                ``settings["current time"]`` and ``settings["last time"]`` are set to this
                value. ``run(..., time_span=...)`` expects a relative time interval, so
                ``time=2.0`` and ``time_span=0.5`` runs the interval from 2.0 to 2.5.
            integration_mode: ``"auto"`` (default) or ``"collocation"``.
        """
        self._ensure_projector_cache()
        if (
            _is_restart_state(values)
            and derivatives is None
            and internal_variables is None
            and time is None
            and integration_mode is None
        ):
            if self._phase in ("setup", "registering"):
                self._pending_restart_state = values
            else:
                self._apply_restart_state(values)
            return
        if _is_restart_state(values):
            raise ValueError("Restart state must be the only argument to set_initial_conditions(...).")

        spec = {
            "values": values,
            "derivatives": derivatives,
            "internal_variables": internal_variables,
            "time": time,
            "integration_mode": _normalize_integration_mode(integration_mode),
        }
        if values is None and derivatives is None and internal_variables is None and time is None:
            return
        if self._phase in ("setup", "registering"):
            self._apply_manual_initial_conditions(
                values=values,
                derivatives=derivatives,
                internal_variables=internal_variables,
                time=time,
                integration_mode=integration_mode,
                allow_missing_internal_blocks=True,
            )
            self._pending_initial_condition_specs.append(spec)
        elif self._phase in ("initialized", "prepared", "ran"):
            self._apply_manual_initial_conditions(
                values=values,
                derivatives=derivatives,
                internal_variables=internal_variables,
                time=time,
                integration_mode=integration_mode,
                allow_missing_internal_blocks=False,
            )
        else:
            raise RuntimeError("set_initial_conditions(...) cannot be called in the current SimState phase.")

    def set_postprocessing_policy(
        self,
        policy=None,
        *,
        result_folder_name: Optional[str] = None,
        domains: Optional[Union[str, list[str], Tuple[str, ...]]] = '__all__',
        integration_order: Optional[int] = None,
        int_pts_and_weights: Optional[Dict[str, Any]] = None,
        integration_mode: Optional[str] = None,
        projection_tol: float = 1e-8,
        projection_atol: float = 1e-10,
        projection_maxiter: int = 1000,
    ):
        """
        Configure dae.SavePolicy for VTU/PVD output during `run(...)`.

        `result_folder_name` is used as the output folder path.
        If `domains` is provided, matching `add_model(...)` functions are evaluated in
        ModelContext mode "output" and projected to VTU point data.
        `integration_mode="collocation"` uses Gauss-Lobatto points for tensor-product elements.
        """
        if self._phase in ("prepared", "ran"):
            if self.manager is None:
                raise RuntimeError("Call initialize(...) before set_postprocessing_policy(...).")
            current = self.manager.save_policy
            if policy is None or isinstance(policy, dae.SaveNothingPolicy):
                self.manager.save_policy = None
                self._vtu_postprocessing_policy = policy
                return
            if not isinstance(current, _VtuCallbackPolicy):
                raise RuntimeError(
                    "Changing VTU postprocessing after prepare(...) requires an existing prepared VTU policy."
                )
            output_changed = (
                domains != '__all__'
                or integration_order is not None
                or int_pts_and_weights is not None
                or integration_mode is not None
                or projection_tol != 1e-8
                or projection_atol != 1e-10
                or projection_maxiter != 1000
            )
            if result_folder_name is not None and _vtu_output_layout(result_folder_name)[1] != current.writer.folder:
                raise RuntimeError("Changing the VTU output folder after prepare(...) is not supported.")
            if output_changed:
                raise RuntimeError(
                    "After prepare(...), only the save policy can be changed; output fields/domains stay fixed."
                )
            self.manager.save_policy = _VtuCallbackPolicy(policy, current.writer, current.output_groups)
            self._vtu_postprocessing_policy = policy
            return

        self._enter_registration_phase("set_postprocessing_policy")
        self._ensure_projector_cache()
        self._vtu_postprocessing_policy = policy
        self._vtu_postprocessing_result_folder_name = result_folder_name
        self._vtu_postprocessing_file_stem = None
        self._vtu_postprocessing_output_folder = None
        self._output_domains = _normalize_output_domains(domains)
        self._output_integration_order = integration_order
        self._output_int_pts_and_weights = int_pts_and_weights
        self._output_integration_mode = _normalize_integration_mode(integration_mode)
        self._output_projection_tol = float(projection_tol)
        self._output_projection_atol = float(projection_atol)
        self._output_projection_maxiter = int(projection_maxiter)
        self._raw_output_specs = tuple()
        self._clear_output_projection_state()

    def add_temporal_discretization(self, temporal_discretizations: Dict[str, Any]):
        """
        Assign a time-integration scheme to each field.

        Parameters
        ----------
        temporal_discretizations:
            Mapping ``{field: integrator}`` where ``integrator`` is a ``dae`` time
            discretization (e.g. ``dae.NoTimeDerivative()`` for a stationary field,
            or a Backward/Forward Euler integrator). Each field is routed to the
            primary or internal-variable integrator set depending on how it was
            discretized; an unknown field name raises ``KeyError``.
        """
        self._enter_registration_phase("add_temporal_discretization")
        primary_spaces = dict(self.static_settings.get("spatial discretizations", {}))
        internal_spaces = dict(self.static_settings.get("internal variable discretizations", {}))
        primary_integrators = {}
        internal_integrators = {}
        for field, integrator in temporal_discretizations.items():
            if field in primary_spaces:
                primary_integrators[field] = integrator
            elif field in internal_spaces:
                internal_integrators[field] = integrator
            else:
                raise KeyError(f"Unknown field '{field}'.")
        self.static_settings['time integrators'] = primary_integrators
        self.static_settings['internal variable time integrators'] = internal_integrators

    def add_strong_bc(
        self,
        field,
        on_boundary_fun,
        value_fun,
        index=None,
        reset=False,
        integration_mode: Optional[str] = None,
    ):
        """
        Register a strong boundary condition via an internal L2 projection on the selected boundary trace.

        Required signatures:
          - `on_boundary_fun(x) -> bool` for the static geometric boundary selection
          - `value_fun(x)`, `value_fun(x, t)`, or `value_fun(x, t, settings)` for the prescribed value.
            The 3-argument form receives the full `settings` dict, allowing access to user-defined
            parameters (e.g. ``sim.settings["user key"] = ...``) for sensitivity analysis.
          - `integration_mode="collocation"` uses Gauss-Lobatto points on tensor-product facets.
        """
        if getattr(self, "_phase", "setup") == "initialized":
            if self.mesh_info is None:
                raise RuntimeError("Call import_mesh(...) before add_strong_bc(...).")
        else:
            self._enter_registration_phase("add_strong_bc")
        self._ensure_projector_cache()
        if field in self.static_settings.get("internal variable discretizations", {}):
            raise ValueError("Strong boundary conditions are not defined for internal variables.")

        # A field whose intrinsic dimension is below the mesh's top dimension has
        # its Dirichlet trace on a lower-dimensional boundary than the mesh
        # facets (e.g. the edges of a 2D membrane on a 3D block). Build that
        # field-dimensional boundary domain explicitly and project onto it; the
        # standard (top-dimensional) path is left untouched for ordinary fields.
        field_dim = int(getattr(self.static_settings["spatial discretizations"][field], "dim", 0))
        top_dim = self._mesh_top_dim()

        # A 1D field's boundary is 0-dimensional (its endpoints). Integration over
        # a point degenerates to evaluation, so instead of an L2 projection the
        # Dirichlet value is set directly on the field's DOFs at the selected
        # nodes (a point Dirichlet constrains exactly its own node DOF).
        if field_dim == 1:
            self._add_nodal_dirichlet_spec(field, on_boundary_fun, value_fun, index, reset)
            if self.manager is not None:
                self._install_projected_dirichlet_callback()
            return

        boundary_domain = None
        if 0 < field_dim < top_dim:
            base_name = f"__strong_bc__{field}__{len(self._raw_projected_dirichlet_specs)}"
            boundary_domain = base_name
            counter = 2
            while boundary_domain in self.mesh_info.domains:
                boundary_domain = f"{base_name}__{counter}"
                counter += 1
            self._add_boundary_domain_to_state(on_boundary_fun, boundary_domain, cell_dim=field_dim)

        spec = FrozenDict({
            "field": field,
            "on_boundary_fun": on_boundary_fun,
            "boundary_domain": boundary_domain,
            "value_fun": value_fun,
            "index": index,
            "reset": reset,
            "projector_name": None,
            "integration_order": None,
            "int_pts_and_weights": None,
            "integration_mode": _normalize_integration_mode(integration_mode),
            "tol": 1e-8,
            "atol": 1e-10,
            "maxiter": 1000,
        })
        self._raw_projected_dirichlet_specs = tuple(self._raw_projected_dirichlet_specs) + (spec,)
        if self.manager is not None:
            self._materialize_projected_dirichlet_specs()
            if len(self._projected_dirichlet_specs) > 0:
                self._install_projected_dirichlet_callback()

    def add_weak_bc(
        self,
        field,
        on_boundary_fun,
        value_fun,
        index=None,
        *,
        trace: str = "value",
        sign: float = -1.0,
    ):
        """
        Add a weak boundary condition as a convenience wrapper around a surface weak form.

        With `trace="value"`, `value_fun` must return a load/traction with the same shape as the selected field.
        With `index`, `value_fun` must return a scalar that is applied only to that component.
        With `trace="normal"` (experimental), the selected field's surface basis must return a scalar normal trace.
        """
        self._enter_registration_phase("add_weak_bc")

        self._ensure_projector_cache()
        if field in self.static_settings.get("internal variable discretizations", {}):
            raise ValueError("Weak boundary conditions are not defined for internal variables.")
        if field not in self.static_settings.get("spatial discretizations", {}):
            raise KeyError(f"Unknown field '{field}'.")

        # A 1D field's boundary is 0-dimensional (its endpoints); a weak BC there
        # is a point load. Integration over a point degenerates to evaluation, so
        # the load is added directly to the residual at the field's node DOFs.
        field_dim = int(getattr(self.static_settings["spatial discretizations"][field], "dim", 0))
        if field_dim == 1:
            if trace != "value":
                raise NotImplementedError("Point (0-dimensional) weak BCs support trace='value' only.")
            self._add_nodal_load_spec(field, on_boundary_fun, value_fun, index, float(sign))
            return

        base_name = f"__weak_bc__{field}__{len(self._raw_model_specs)}"
        domain_name = base_name
        counter = 2
        while self.mesh_info is not None and domain_name in self.mesh_info.domains:
            domain_name = f"{base_name}__{counter}"
            counter += 1

        # The BC lives on the boundary of the field's own domain, whose dimension
        # is the field's space dimension. For a lower-dimensional field this is
        # below the mesh's top dimension (e.g. the edges of a 2D membrane).
        self._add_boundary_domain_to_state(on_boundary_fun, domain_name, cell_dim=field_dim)
        self.add_model(
            domain_name,
            "weak form",
            _make_weak_bc_integrand(field, value_fun, index, trace=trace, sign=float(sign)),
        )

    def _validate_integrator_compatibility(self):
        """
        Check that the configured time integrators are mutually compatible.

        This is a cheap, pure-Python inspection of integrator metadata
        (``stage_types``, ``num_stages``, ``num_derivs``, ``num_steps``) and the
        step size controller type. It runs once in ``initialize(...)`` after the
        default ``NoTimeDerivative`` integrators have been filled in, so it sees
        the complete set of primary and internal-variable integrators.

        Current limitations enforced here:
          1. Time integrators with coupled stages (e.g. Gauss-Legendre
             Runge-Kutta) are not supported for primary fields.
          2. Algebraic constraints don't work together with explicit stages.
          3. The stage positions must be the same for all (time-integrated)
             fields. Stationary ``NoTimeDerivative`` fields are exempt.
          4. Multi-step methods (BDF, AdamsMoulton, AdamsBashforth) are only
             compatible with the ConstantStepSizeController.
        """
        primary = self.static_settings["time integrators"]
        internal = self.static_settings["internal variable time integrators"]
        all_integrators = {**primary, **internal}

        # 1. Coupled stages are not supported for primary fields. A coupled
        #    block groups several stages, so it contributes fewer stage_types
        #    entries than num_stages (diagonally implicit methods keep one
        #    entry per stage).
        for field, integ in primary.items():
            if len(integ.stage_types) < integ.num_stages:
                raise NotImplementedError(
                    f"Field '{field}': time integrators with coupled stages "
                    f"(e.g. Gauss-Legendre Runge-Kutta) are not supported for "
                    f"primary fields."
                )

        # 2. Algebraic constraints (NoTimeDerivative, num_derivs == 0) don't
        #    work together with explicit integration stages.
        has_explicit = any(
            "explicit" in integ.stage_types for integ in all_integrators.values()
        )
        algebraic = [f for f, integ in primary.items() if integ.num_derivs == 0]
        if has_explicit and algebraic:
            raise NotImplementedError(
                f"Algebraic constraints (fields {algebraic}) are not supported "
                f"together with explicit time integration stages."
            )

        # 3. The stage positions must match across all time-integrated fields.
        #    Stationary NoTimeDerivative fields (num_derivs == 0) have no real
        #    stage structure and are exempt.
        ref_field = None
        ref_positions = None
        for field, integ in all_integrators.items():
            if integ.num_derivs == 0:
                continue
            positions = jnp.asarray(integ.stage_positions)
            if ref_positions is None:
                ref_field, ref_positions = field, positions
            elif ref_positions.shape != positions.shape or not bool(
                jnp.allclose(ref_positions, positions)
            ):
                raise ValueError(
                    f"Incompatible stage positions: field '{field}' vs "
                    f"'{ref_field}'. The time integrators of all fields must use "
                    f"the same stage positions."
                )

        # 4. Multi-step methods (num_steps > 1) only work with a constant step
        #    size.
        multistep = [f for f, integ in all_integrators.items() if integ.num_steps > 1]
        if multistep and not isinstance(
            self._step_size_controller, dae.ConstantStepSizeController
        ):
            raise ValueError(
                f"Multi-step methods (fields {multistep}, e.g. BDF / Adams) are "
                f"only compatible with the ConstantStepSizeController."
            )

    def initialize(self, verbose: int = 0):
        """
        Finalize the input state and set up the time stepping manager.

        Parameters
        ----------
        verbose:
            Verbosity level forwarded to the time stepping manager.
        """
        # This version assumes that global DOF containers and connectivity are built via the new
        # _MeshInfo/FE pipeline:
        # - DOF containers are created in `self._ensure_fe_built()` and stored in `self.dofs`.
        # - The FE pack (connectivity/layout/g2a) is stored in `self._fe`.

        if self._phase not in ("setup", "registering"):
            raise RuntimeError("initialize(...) may only be called once and before prepare(...).")
        if self.mesh_info is None:
            raise RuntimeError("Call import_mesh(...) before initialize(...).")
        # Strong (projected Dirichlet) BCs project onto the boundary facets and
        # then scatter into the volume DOF vector, so their entity numbering must
        # match the volume's. Fold the full boundary-facet set into the single
        # mesh build (instead of a separate _build_mesh_info) so both share one
        # numbering; this also removes the redundant second topology build.
        if len(getattr(self, "_raw_projected_dirichlet_specs", tuple())) > 0 and self._source_mesh is not None:
            existing_sets = getattr(self._source_mesh, "cell_sets", None) or {}
            if _ALL_BOUNDARY_FACET_DOMAIN not in existing_sets:
                _add_facet_domain(self._source_mesh, _ALL_BOUNDARY_FACET_DOMAIN, point_selector=None)
                self._source_mesh_modified = True
        # Rebuild mesh_info once to include all boundary domains added via
        # add_weak_bc and the all-boundary facet set above.
        if getattr(self, "_source_mesh_modified", False):
            self._rebuild_mesh_info_from_source()
            self._source_mesh_modified = False
        self._ensure_fe_built()
        self._materialize_model_specs()
        self._materialize_local_subsystem_solvers()
        self._materialize_output_specs()
        self._materialize_projected_dirichlet_specs()
        time_integrators = dict(self.static_settings.get("time integrators", {}))
        for f in sorted(self.static_settings["spatial discretizations"].keys()):
            time_integrators.setdefault(f, dae.NoTimeDerivative())
        self.static_settings["time integrators"] = time_integrators
        internal_time_integrators = dict(self.static_settings.get("internal variable time integrators", {}))
        for f in sorted(self.static_settings.get("internal variable discretizations", {}).keys()):
            internal_time_integrators.setdefault(f, dae.NoTimeDerivative())
        self.static_settings["internal variable time integrators"] = internal_time_integrators
        self._validate_integrator_compatibility()
        self._apply_pending_initial_conditions()
        self.static_settings["verbose"] = int(verbose)
        self.static_settings = FrozenDict(self.static_settings)
        self.manager = dae.TimeSteppingManager(
            self.static_settings,
            step_size_controller=self._step_size_controller,
        )
        self._base_apply_dirichlet_bcs = None
        if len(self._projected_dirichlet_specs) > 0 or len(self._nodal_dirichlet_specs) > 0:
            self._install_projected_dirichlet_callback()
        self._install_nodal_load_callback()
        self._install_internal_variable_update_callback()
        self._materialize_projected_ic_specs()
        ic_specs = getattr(self, "_projected_ic_specs", tuple())
        if ic_specs:
            self.manager.static_settings = self.manager.static_settings.copy(
                add_or_replace={"_ic_projection_specs": ic_specs}
            )
        self._phase = "initialized"

    def prepare(self, clean_up: bool = True):
        """
        Prepare the initialized simulation for run(...).

        With clean_up=True, mesh/FE construction data and initialization-only
        caches are released after the runtime data has been materialized.
        """
        self._require_initialized_phase("prepare")
        self.settings = self.manager.initialize(self.dofs, self.settings)
        self._materialize_projected_dirichlet_runtime()
        self._materialize_projected_ic_runtime()
        apply_bcs = self.manager.static_settings.get("apply dirichlet bcs", None)
        if apply_bcs is not None:
            current_time = float(self.settings.get("current time", 0.0))
            self.settings = apply_bcs(current_time, dict(self.settings))
        if self._has_active_vtu_postprocessing_policy():
            self._prepare_vtu_postprocessing_policy()
        self._raw_model_specs = tuple()
        self._raw_output_specs = tuple()
        self._raw_projected_dirichlet_specs = tuple()
        self._phase = "prepared"
        if clean_up:
            self._cleanup_after_prepare()

    @utility.jit_with_docstring(static_argnames=["num_time_steps", "quasistatic"])
    def run(self, dt0: float, time_span, num_time_steps: int, *, quasistatic: bool = False):
        """
        Run the prepared simulation over a time span and return the updated SimState.

        Args:
            dt0 (float): Initial time step size.
            time_span (float): Time interval to compute from the current time.
            num_time_steps (int): Maximum number of time steps to perform.
            quasistatic (bool): If True, reset the time to the start time after the run.

        Returns:
            SimState: A new SimState object containing the final state, history, and step statistics.
        """
        self._require_prepared_phase("run")
        jax.lax.cond(
            jnp.any(jnp.asarray(time_span) <= 0.0),
            lambda value: jax.debug.callback(_raise_nonpositive_time_span, value, ordered=True),
            lambda value: None,
            time_span,
        )
        t_start = self.settings.get("current time", 0.0)
        t_final = t_start + time_span
        ic_specs = self.manager.static_settings.get("_ic_projection_specs", None)
        dofs = _apply_callable_ic_projections(ic_specs, self.settings, self.dofs) if ic_specs else self.dofs
        run_settings = dict(self.settings)
        run_settings.pop(dae._RESTART_CONTROLLER_STATE_KEY, None)
        run_settings.pop(dae._RESTART_DT_KEY, None)
        result = self.manager.run(dofs, dt0, t_final, num_time_steps, run_settings)
        self.manager._global_template = None

        settings = dict(result.settings)
        settings[dae._RESTART_Q_N_KEY] = result.q_n
        settings[dae._RESTART_Q_DER_N_KEY] = result.q_der_n
        settings.pop(dae._RESTART_CONTROLLER_STATE_KEY, None)
        settings.pop(dae._RESTART_DT_KEY, None)
        if quasistatic:
            settings["current time"] = t_start
            settings["last time"] = t_start

        run_result = replace(result, settings=settings)
        controller = getattr(self.manager, "step_size_controller", self._step_size_controller)
        return SimState._tree_unflatten(
            (
                self.static_settings,
                self.manager,
                None,
                "ran",
                getattr(self, "_requested_cell_data_keys", tuple()),
                controller,
            ),
            (
                settings,
                result.q,
                run_result,
            ),
        )

    def _write_vtu(self, path=None, dofs=None, settings=None, output_fields=None):
        """
        Write the current mesh together with registered projected outputs to a VTU file.

        Parameters
        ----------
        path:
            Output path. If omitted, a timestamp-based path inside `sim_state.res/` is used.
        dofs:
            Optional state dictionary to be used for output evaluation.
        settings:
            Optional settings dictionary to be used for output evaluation.
        output_fields:
            Optional subset of registered projected output field names.
        """
        if getattr(self, "_phase", "setup") not in ("initialized", "prepared", "ran"):
            raise RuntimeError("_write_vtu(...) must be called after initialize(...).")
        self._ensure_projector_cache()
        if self._source_mesh is None or self.mesh_info is None:
            raise RuntimeError("Call import_mesh(...) before _write_vtu(...).")

        dofs_use = self.dofs if dofs is None else dofs
        settings_use = dict(self.settings if settings is None else settings)
        settings_use.setdefault("current time", 0.0)

        if path is None:
            _, output_folder = _vtu_output_layout("results")
            os.makedirs(output_folder, exist_ok=True)
            step_index = int(getattr(self, "_manual_vtu_file_index", 0))
            path = os.path.join(output_folder, _vtu_step_file_name(step_index))
            self._manual_vtu_file_index = step_index + 1
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

        mesh = _prepare_mesh_for_vtu(copy.deepcopy(self._source_mesh))
        existing_point_data = getattr(mesh, "point_data", None)
        mesh.point_data = {} if existing_point_data is None else dict(existing_point_data)

        n_points = int(self.mesh_info.points.shape[0])

        for name, values in self._evaluate_registered_outputs(dofs_use, settings_use, output_fields).items():
            export_name = name
            counter = 2
            while export_name in mesh.point_data:
                export_name = f"{name}__output" if counter == 2 else f"{name}__output__{counter}"
                counter += 1
            mesh.point_data[export_name] = _format_vtu_point_data(values, n_points)

        meshio.write(path, mesh, binary=True, compression=False)
        return path


    def _ensure_projector_cache(self) -> None:
        # Fast path: _init_ephemeral_state guarantees all attributes are present.
        if hasattr(self, "_l2_projectors"):
            return
        # Fallback for legacy pickled SimState instances predating _init_ephemeral_state.
        _init_ephemeral_state(self)

    def _build_projection_source_connectivity(
        self,
        field: str,
        set_info: _L2ProjectionSetInfo,
        *,
        strict: bool,
    ) -> Optional[jnp.ndarray]:
        pack = self._fe[field]
        if not set_info.is_surface:
            if set_info.mesh_topology not in pack.volume or set_info.mesh_topology not in pack.g2a["cell_g2a"]:
                if strict:
                    raise ValueError(f"[{field}] No volume connectivity for '{set_info.mesh_topology}'.")
                return None
            rows = pack.g2a["cell_g2a"][set_info.mesh_topology][set_info.volume_element_ids]
            if not bool(jnp.all(rows >= 0)):
                if strict:
                    raise ValueError(f"[{field}] Projector domain contains inactive volume cells for '{set_info.mesh_topology}'.")
                return None
            return pack.volume[set_info.mesh_topology][rows].astype(int)

        chunks = []
        ids_all = []
        for domain_name in set_info.surface_domains:
            ids_chunk, conn_chunk = _build_surface_connectivity_for_field(
                self.mesh_info,
                pack,
                domain_name,
                set_info.mesh_topology,
            )
            if conn_chunk.shape[0] == 0:
                if strict:
                    raise ValueError(f"[{field}] No surface connectivity for domain '{domain_name}' and type '{set_info.mesh_topology}'.")
                return None
            chunks.append(conn_chunk)
            ids_all.append(ids_chunk)

        if len(chunks) == 0:
            if strict:
                raise ValueError(f"[{field}] No surface connectivity for '{set_info.mesh_topology}'.")
            return None

        ids = jnp.concatenate(ids_all, axis=0) if len(ids_all) > 1 else ids_all[0]
        if not np.array_equal(np.asarray(ids), np.asarray(set_info.surface_element_ids)):
            if strict:
                raise ValueError(
                    f"[{field}] Surface connectivity on '{set_info.mesh_topology}' is not aligned with the target projector domain."
                )
            return None

        return jnp.concatenate(chunks, axis=0).astype(int) if len(chunks) > 1 else chunks[0].astype(int)

    def _resolve_source_fields_and_connectivity(
        self,
        projector: _PreparedL2Projection,
        source_fields: Optional[Tuple[str, ...]],
    ) -> Tuple[Tuple[str, ...], Tuple[Dict[str, jnp.ndarray], ...]]:
        available_fields = tuple(sorted(self.dofs.keys()))
        requested = tuple(source_fields) if source_fields is not None else available_fields
        explicit = source_fields is not None

        per_field_connectivity: Dict[str, list[jnp.ndarray]] = {}
        resolved_fields = []

        for field in requested:
            if field not in self.dofs:
                if explicit:
                    raise KeyError(f"Unknown source field '{field}'.")
                continue

            field_ok = True
            conn_per_set = []
            for set_info in projector.set_infos:
                conn = self._build_projection_source_connectivity(field, set_info, strict=explicit)
                if conn is None:
                    field_ok = False
                    break
                conn_per_set.append(conn)

            if field_ok:
                resolved_fields.append(field)
                per_field_connectivity[field] = conn_per_set
            # If not field_ok and not explicit, the field simply does not live on
            # this projector's domain (e.g. a subdomain-restricted field). During
            # implicit source discovery that is expected, so it is skipped
            # silently. The explicit path (source_fields given) instead raises via
            # the strict connectivity build above.

        resolved = tuple(resolved_fields)
        per_set = []
        for set_idx in range(len(projector.set_infos)):
            per_set.append({field: per_field_connectivity[field][set_idx] for field in resolved})
        return resolved, tuple(per_set)

    def _resolve_source_fields_and_connectivity_cached(
        self,
        projector: _PreparedL2Projection,
        source_fields: Optional[Tuple[str, ...]],
    ) -> Tuple[Tuple[str, ...], Tuple[Dict[str, jnp.ndarray], ...]]:
        self._ensure_projector_cache()
        requested = None if source_fields is None else tuple(source_fields)
        cache_key = (projector.name, requested)
        cached = self._projection_source_cache.get(cache_key, None)
        if cached is None:
            cached = self._resolve_source_fields_and_connectivity(projector, source_fields)
            self._projection_source_cache[cache_key] = cached
        return cached

    def _source_bases_for_projection(
        self,
        projector: _PreparedL2Projection,
        resolved_source_fields: Tuple[str, ...],
    ) -> Tuple[Dict[str, Callable], ...]:
        bases_by_set = []
        for set_info in projector.set_infos:
            source_bases = {}
            for source_field in resolved_source_fields:
                space = self.static_settings["spatial discretizations"][source_field]
                if set_info.is_surface:
                    source_bases[source_field] = _space_surface_basis_for_field(
                        space, set_info.basis_cell_type, source_field
                    )
                else:
                    source_bases[source_field] = _space_basis_for_field(
                        space, set_info.basis_cell_type, source_field
                    )
            bases_by_set.append(source_bases)
        return tuple(bases_by_set)

    def _source_sets_for_projection(
        self,
        projector: _PreparedL2Projection,
        resolved_source_fields: Tuple[str, ...],
    ) -> Tuple[Dict[str, int], ...]:
        domains = tuple(self.static_settings.get("domains", tuple()))
        sets_by_set = []
        for set_info in projector.set_infos:
            domain_name = set_info.surface_domains[0] if set_info.is_surface and len(set_info.surface_domains) == 1 else set_info.domain_name
            source_sets = {}
            if domain_name in domains:
                source_set = domains.index(domain_name)
                source_sets = {field: source_set for field in resolved_source_fields}
            sets_by_set.append(source_sets)
        return tuple(sets_by_set)

    def _prepare_l2_projection(
        self,
        field: str,
        domain_name: Union[str, list[str]],
        *,
        name: Optional[str] = None,
        integration_order: Optional[int] = None,
        int_pts_and_weights: Optional[Dict[str, Any]] = None,
        integration_mode: Optional[str] = None,
    ) -> _PreparedL2Projection:
        self._ensure_projector_cache()
        if self.mesh_info is None:
            raise RuntimeError("Call import_mesh(...) before _prepare_l2_projection(...).")

        self._ensure_fe_built()
        spaces_by_field = dict(self.static_settings.get("spatial discretizations", {}))
        if field not in spaces_by_field:
            raise KeyError(f"Unknown field '{field}'.")

        requested_domains = tuple(_as_list(domain_name))
        projector_name = name or f"{field}:{' + '.join(requested_domains)}"
        if projector_name in self._l2_projectors:
            return self._l2_projectors[projector_name]

        target_space = spaces_by_field[field]
        field_shape = _field_shape_from_space(target_space)
        flat_components = _num_flat_components(field_shape)
        pack = self._fe[field]

        elem_ids_by_type = _collect_elem_ids_by_type(self.mesh_info, list(requested_domains))
        if len(elem_ids_by_type) == 0:
            raise ValueError(f"No elements found for domain '{domain_name}'.")

        integration_mode = _normalize_integration_mode(integration_mode)
        integration_order_was_none = integration_order is None
        if integration_order is None:
            integration_order = _default_integration_order_from_spaces(spaces_by_field)
        target_collocation_order = None
        if integration_mode == "collocation" and integration_order_was_none:
            target_collocation_order = _collocation_integration_order_from_poly_order(target_space.order)

        set_infos = []
        global_connectivities = []

        for mesh_topology in sorted(elem_ids_by_type.keys()):
            elem_ids = jnp.asarray(elem_ids_by_type[mesh_topology], dtype=int)
            base = _canonical_base_type(mesh_topology)
            if base not in _BASE_DIM:
                continue

            elem_dim = int(_BASE_DIM[base])
            is_surface = elem_dim == int(target_space.dim) - 1
            is_volume = elem_dim == int(target_space.dim)
            if not (is_surface or is_volume):
                continue

            mesh_type = _mesh_type_from_topology(mesh_topology)
            basis_cell_type = _canonical_base_type(mesh_topology)

            if int_pts_and_weights is None:
                current_int = _pick_projection_integration(
                    mesh_type,
                    int(integration_order),
                    integration_mode,
                    collocation_order=target_collocation_order,
                )
            elif isinstance(int_pts_and_weights, dict):
                if mesh_topology in int_pts_and_weights:
                    current_int = int_pts_and_weights[mesh_topology]
                elif mesh_type in int_pts_and_weights:
                    current_int = int_pts_and_weights[mesh_type]
                else:
                    raise ValueError(f"No quadrature entry for '{mesh_topology}' or '{mesh_type}'.")
            else:
                current_int = int_pts_and_weights

            if mesh_type in ("line", "quad", "hexahedron"):
                mapping_basis = spaces.fem_line_quad_brick
            elif mesh_type in ("triangle", "tetra"):
                mapping_basis = spaces.fem_line_tri_tet
            else:
                raise NotImplementedError(f"Unknown mesh type '{mesh_type}'.")

            if is_volume:
                if mesh_topology not in pack.volume or mesh_topology not in pack.g2a["cell_g2a"]:
                    raise ValueError(f"[{field}] No volume connectivity for '{mesh_topology}'.")
                rows = pack.g2a["cell_g2a"][mesh_topology][elem_ids]
                if not bool(jnp.all(rows >= 0)):
                    raise ValueError(f"[{field}] Projector domain contains inactive cells for '{mesh_topology}'.")
                target_conn = pack.volume[mesh_topology][rows].astype(int)
                phys_conn = self.mesh_info.cells[mesh_topology].nodes_of_cells[elem_ids].astype(int)
                surface_domains = tuple()
                surface_element_ids = None
                target_basis = _space_basis_for_field(target_space, basis_cell_type, _PROJECTION_KEY)
            else:
                target_chunks = []
                phys_chunks = []
                ids_chunks = []
                used_domains = []
                for dn in requested_domains:
                    ids_chunk, conn_chunk = _build_surface_connectivity_for_field(self.mesh_info, pack, dn, mesh_topology)
                    if conn_chunk.shape[0] == 0:
                        continue
                    target_chunks.append(conn_chunk.astype(int))
                    phys_chunks.append(self.mesh_info.cells[mesh_topology].nodes_of_cells[ids_chunk].astype(int))
                    ids_chunks.append(ids_chunk)
                    used_domains.append(dn)

                if len(target_chunks) == 0:
                    continue

                target_conn = jnp.concatenate(target_chunks, axis=0) if len(target_chunks) > 1 else target_chunks[0]
                phys_conn = jnp.concatenate(phys_chunks, axis=0) if len(phys_chunks) > 1 else phys_chunks[0]
                surface_element_ids = jnp.concatenate(ids_chunks, axis=0) if len(ids_chunks) > 1 else ids_chunks[0]
                surface_domains = tuple(used_domains)
                target_basis = _space_surface_basis_for_field(target_space, basis_cell_type, _PROJECTION_KEY)

            set_infos.append(
                _L2ProjectionSetInfo(
                    domain_name=str(domain_name),
                    mesh_topology=mesh_topology,
                    mesh_type=mesh_type,
                    basis_cell_type=basis_cell_type,
                    domain_dim=int(current_int[0].shape[1]) if current_int[0].ndim >= 2 else 1,
                    is_surface=is_surface,
                    target_basis=target_basis,
                    mapping_basis=mapping_basis,
                    physical_connectivity=phys_conn,
                    target_global_connectivity=target_conn,
                    volume_element_ids=elem_ids if is_volume else None,
                    surface_domains=surface_domains,
                    surface_element_ids=surface_element_ids,
                    ref_int_coor=current_int[0],
                    ref_int_weights=current_int[1],
                )
            )
            global_connectivities.append(target_conn)

        if len(set_infos) == 0:
            raise ValueError(f"No compatible projection topology found for field '{field}' on domain '{domain_name}'.")

        global_scalar_ids, local_connectivities = _merge_projector_scalar_ids(global_connectivities, int(pack.dofs.shape[0]))
        local_shape = (int(global_scalar_ids.shape[0]),) if flat_components == 1 else (int(global_scalar_ids.shape[0]), flat_components)
        orientation_template = []
        for info in set_infos:
            info_elem_ids = info.surface_element_ids if info.is_surface else info.volume_element_ids
            signs = _orientation_for_space_on_topology(
                self.mesh_info,
                target_space,
                info.mesh_topology,
                info_elem_ids,
                is_surface=info.is_surface,
            )
            orientation_template.append({} if signs is None else {_PROJECTION_KEY: signs})

        settings_template = {
            "node coordinates": self.mesh_info.points,
            "connectivity": tuple(
                {
                    _PROJECTION_KEY: local_conn.astype(int),
                    "physical coor": info.physical_connectivity.astype(int),
                }
                for info, local_conn in zip(set_infos, local_connectivities)
            ),
            "orientation": tuple(orientation_template),
        }

        zero_local = jnp.zeros(local_shape, dtype=float)
        zero_dofs = {_PROJECTION_KEY: zero_local}
        mass_models = tuple(
            _make_l2_projection_model(
                projection_key=_PROJECTION_KEY,
                target_basis=info.target_basis,
                mapping_basis=info.mapping_basis,
                field_shape=field_shape,
                flat_components=flat_components,
                domain_dim=info.domain_dim,
                ref_int_coor=info.ref_int_coor,
                ref_int_weights=info.ref_int_weights,
                quantity_fun=None,
                source_fields=tuple(),
                source_bases={},
                source_sets={},
                source_static_settings=self.static_settings,
            )
            for info in set_infos
        )
        mass_static_settings = FrozenDict({
            "assembling mode": tuple("user residual" for _ in set_infos),
            "model": mass_models,
        })

        mass_matrix = (-assembler.assemble_tangent(zero_dofs, settings_template, mass_static_settings))
        jacobi_diag = -assembler.assemble_tangent_diagonal(zero_dofs, settings_template, mass_static_settings)
        jacobi_inv_diag = jnp.where(jnp.abs(jacobi_diag) > 1e-14, 1.0 / jacobi_diag, 1.0)


        projector = _PreparedL2Projection(
            name=projector_name,
            target_name=field,
            field_shape=field_shape,
            flat_components=flat_components,
            global_scalar_ids=global_scalar_ids,
            local_shape=local_shape,
            full_shape=((int(pack.dofs.shape[0]),) if len(field_shape) == 0 else (int(pack.dofs.shape[0]),) + field_shape),
            set_infos=tuple(set_infos),
            settings_template=settings_template,
            mass_matrix=mass_matrix,
            jacobi_diag=jacobi_diag,
            jacobi_inv_diag=jacobi_inv_diag,
            outside_value=0.0,
        )

        self._l2_projectors[projector_name] = projector
        return projector

    def _coerce_local_initial_guess(self, projector: _PreparedL2Projection, initial_guess: Optional[jnp.ndarray]) -> jnp.ndarray:
        if initial_guess is None:
            if projector.last_local_solution is not None:
                return projector.last_local_solution
            return jnp.zeros(projector.local_shape, dtype=float)

        arr = jnp.asarray(initial_guess)
        if tuple(arr.shape) == projector.local_shape:
            return arr

        local_field_shape = ((int(projector.global_scalar_ids.shape[0]),) if len(projector.field_shape) == 0
                             else (int(projector.global_scalar_ids.shape[0]),) + projector.field_shape)
        if tuple(arr.shape) == local_field_shape:
            local = _flatten_projector_values(arr)
            return local.reshape(projector.local_shape)

        if tuple(arr.shape) == tuple(projector.full_shape):
            local = arr[projector.global_scalar_ids]
            local = _flatten_projector_values(local)
            return local.reshape(projector.local_shape)

        raise ValueError(
            f"initial_guess has shape {arr.shape}, expected {projector.local_shape} or {projector.full_shape}."
        )

    def _scatter_local_projection(self, projector: _PreparedL2Projection, local_solution: jnp.ndarray) -> jnp.ndarray:
        values = _reshape_projector_coefficients(local_solution, projector.field_shape)
        full = jnp.full(projector.full_shape, projector.outside_value, dtype=values.dtype)
        return full.at[projector.global_scalar_ids].set(values)

    def _l2_project(
        self,
        *,
        field: Optional[str] = None,
        domain_name: Optional[Union[str, list[str]]] = None,
        quantity_fun: Callable,
        projector_name: Optional[str] = None,
        source_fields: Optional[Tuple[str, ...]] = None,
        source_dofs: Optional[Dict[str, jnp.ndarray]] = None,
        source_settings: Optional[Dict[str, Any]] = None,
        current_time: Optional[float] = None,
        integration_order: Optional[int] = None,
        int_pts_and_weights: Optional[Dict[str, Any]] = None,
        integration_mode: Optional[str] = None,
        initial_guess: Optional[jnp.ndarray] = None,
        tol: float = 1e-8,
        atol: float = 1e-10,
        maxiter: int = 1000,
        return_local: bool = False,
    ):
        self._ensure_projector_cache()
        if projector_name is None:
            if field is None or domain_name is None:
                raise ValueError("field and domain_name are required when projector_name is not provided.")
            projector = self._prepare_l2_projection(
                field,
                domain_name,
                name=None,
                integration_order=integration_order,
                int_pts_and_weights=int_pts_and_weights,
                integration_mode=integration_mode,
            )
        else:
            if projector_name not in self._l2_projectors:
                if field is None or domain_name is None:
                    raise KeyError(f"Unknown projector '{projector_name}'.")
                projector = self._prepare_l2_projection(
                    field,
                    domain_name,
                    name=projector_name,
                    integration_order=integration_order,
                    int_pts_and_weights=int_pts_and_weights,
                    integration_mode=integration_mode,
                )
            else:
                projector = self._l2_projectors[projector_name]

        source_dofs_use = self.dofs if source_dofs is None else source_dofs
        if source_settings is None:
            projection_settings = dict(self.settings)
        else:
            projection_settings = dict(source_settings)

        if current_time is None:
            current_time = float(projection_settings.get("current time", self.settings.get("current time", 0.0)))
        projection_settings["current time"] = current_time

        resolved_source_fields, source_connectivity = self._resolve_source_fields_and_connectivity_cached(projector, source_fields)
        projection_settings["node coordinates"] = projector.settings_template["node coordinates"]
        projection_settings["connectivity"] = projector.settings_template["connectivity"]
        projection_settings["projection source connectivity"] = source_connectivity
        projection_settings["projection source dofs"] = source_dofs_use

        source_bases_by_set = self._source_bases_for_projection(projector, resolved_source_fields)
        source_sets_by_set = self._source_sets_for_projection(projector, resolved_source_fields)
        models_for_sets = []
        for set_info, source_bases, source_sets in zip(projector.set_infos, source_bases_by_set, source_sets_by_set):
            models_for_sets.append(
                _make_l2_projection_model(
                    projection_key=_PROJECTION_KEY,
                    target_basis=set_info.target_basis,
                    mapping_basis=set_info.mapping_basis,
                    field_shape=projector.field_shape,
                    flat_components=projector.flat_components,
                    domain_dim=set_info.domain_dim,
                    ref_int_coor=set_info.ref_int_coor,
                    ref_int_weights=set_info.ref_int_weights,
                    quantity_fun=quantity_fun,
                    source_fields=resolved_source_fields,
                    source_bases=source_bases,
                    source_sets=source_sets,
                    source_static_settings=self.static_settings,
                )
            )

        static_settings = FrozenDict({
            "assembling mode": tuple("user residual" for _ in projector.set_infos),
            "model": tuple(models_for_sets),
        })

        local_guess = self._coerce_local_initial_guess(projector, initial_guess)
        projection_dofs = {_PROJECTION_KEY: local_guess}
        residual = assembler.assemble_residual(projection_dofs, projection_settings, static_settings)[_PROJECTION_KEY]
        rhs = residual.reshape(-1)
        delta = _solve_l2_projector_components(
            projector.mass_matrix,
            projector.jacobi_inv_diag,
            rhs,
            tol=tol,
            atol=atol,
            maxiter=maxiter,
        )
        local_solution = local_guess + delta.reshape(projector.local_shape)
        projector.last_local_solution = local_solution

        full_solution = self._scatter_local_projection(projector, local_solution)
        if return_local:
            return full_solution, local_solution
        return full_solution

    def _store_raw_callable_ic_spec(
        self,
        projector_name: str,
        domain_name: str,
        field: str,
        field_shape: Tuple,
        value_fun: Callable,
        integration_mode: Optional[str],
    ) -> None:
        if not hasattr(self, "_raw_callable_ic_specs"):
            self._raw_callable_ic_specs = []
        self._raw_callable_ic_specs = [s for s in self._raw_callable_ic_specs if s["projector_name"] != projector_name]
        self._raw_callable_ic_specs.append(FrozenDict({
            "domain_name": domain_name,
            "field": field,
            "field_shape": tuple(field_shape),
            "value_fun": value_fun,
            "projector_name": projector_name,
            "integration_mode": _normalize_integration_mode(integration_mode),
        }))

    def _materialize_projected_ic_specs(self) -> None:
        raw_specs = getattr(self, "_raw_callable_ic_specs", [])
        if not raw_specs:
            return
        specs = []
        for raw in raw_specs:
            projector = self._l2_projectors.get(raw["projector_name"])
            if projector is None:
                continue
            field_shape = tuple(raw["field_shape"])
            quantity_fun = _make_initial_value_quantity_fun(field_shape, raw["value_fun"], raw["field"])
            models = tuple(
                _make_l2_projection_model(
                    projection_key=_PROJECTION_KEY,
                    target_basis=si.target_basis,
                    mapping_basis=si.mapping_basis,
                    field_shape=projector.field_shape,
                    flat_components=projector.flat_components,
                    domain_dim=si.domain_dim,
                    ref_int_coor=si.ref_int_coor,
                    ref_int_weights=si.ref_int_weights,
                    quantity_fun=quantity_fun,
                    source_fields=tuple(),
                    source_bases={},
                    source_sets={},
                    source_static_settings=self.static_settings,
                )
                for si in projector.set_infos
            )
            specs.append(FrozenDict({
                "field": raw["field"],
                "field_shape": field_shape,
                "projector_name": raw["projector_name"],
                "projection_static_settings": FrozenDict({
                    "assembling mode": tuple("user residual" for _ in projector.set_infos),
                    "model": models,
                }),
            }))
        self._projected_ic_specs = tuple(specs)

    def _materialize_projected_ic_runtime(self) -> None:
        specs = getattr(self, "_projected_ic_specs", tuple())
        if not specs:
            self.settings.pop(_PROJECTED_IC_RUNTIME_KEY, None)
            return
        runtimes = []
        for spec in specs:
            p = self._l2_projectors[spec["projector_name"]]
            runtimes.append(_ProjectedICRuntime(
                global_scalar_ids=p.global_scalar_ids,
                node_coordinates=p.settings_template["node coordinates"],
                connectivity=p.settings_template["connectivity"],
                mass_matrix=p.mass_matrix,
                jacobi_inv_diag=p.jacobi_inv_diag,
            ))
        self.settings[_PROJECTED_IC_RUNTIME_KEY] = tuple(runtimes)

    def _make_boundary_point_selector(self, on_boundary_fun: Callable) -> Callable[[np.ndarray], np.ndarray]:
        def point_selector(points: np.ndarray) -> np.ndarray:
            points = np.asarray(points, dtype=float)
            if points.ndim != 2:
                raise ValueError("Boundary point selector expects points with shape (n_points, gdim).")
            if self.dimension is not None and points.shape[1] > int(self.dimension):
                points = points[:, : int(self.dimension)]
            values = np.asarray(_call_boundary_selector(on_boundary_fun, jnp.asarray(points)))
            return values.reshape((points.shape[0], -1)).all(axis=1)

        return point_selector

    def _build_geometry_output_set_infos(
        self,
        domain_name: str,
        *,
        integration_order: Optional[int] = None,
        int_pts_and_weights: Optional[Dict[str, Any]] = None,
        integration_mode: Optional[str] = None,
    ) -> Tuple[Tuple[_L2ProjectionSetInfo, ...], list[jnp.ndarray]]:
        if self.mesh_info is None:
            raise RuntimeError("Call import_mesh(...) before configuring output.")

        elem_ids_by_type = _collect_elem_ids_by_type(self.mesh_info, domain_name)
        if len(elem_ids_by_type) == 0:
            raise ValueError(f"No elements found for output domain '{domain_name}'.")

        spaces_by_field = dict(self.static_settings.get("spatial discretizations", {}))
        integration_mode = _normalize_integration_mode(integration_mode)
        integration_order_was_none = integration_order is None
        if integration_order is None:
            integration_order = _default_integration_order_from_spaces(spaces_by_field) if spaces_by_field else 2

        set_infos = []
        global_connectivities = []

        for mesh_topology in sorted(elem_ids_by_type.keys()):
            elem_ids = jnp.asarray(elem_ids_by_type[mesh_topology], dtype=int)
            base = _canonical_base_type(mesh_topology)
            if base not in _BASE_DIM:
                continue

            elem_dim = int(_BASE_DIM[base])
            if elem_dim == 0:
                continue

            mesh_type = _mesh_type_from_topology(mesh_topology)
            collocation_order = None
            if integration_mode == "collocation" and integration_order_was_none:
                topology_order = _tensor_product_order_from_topology(mesh_topology)
                if topology_order is None:
                    raise ValueError(
                        "Projection integration mode 'collocation' requires a full tensor-product "
                        f"Lagrange element topology, got '{mesh_topology}'."
                    )
                collocation_order = _collocation_integration_order_from_poly_order(topology_order)
            if int_pts_and_weights is None:
                current_int = _pick_projection_integration(
                    mesh_type,
                    int(integration_order),
                    integration_mode,
                    collocation_order=collocation_order,
                )
            elif isinstance(int_pts_and_weights, dict):
                if mesh_topology in int_pts_and_weights:
                    current_int = int_pts_and_weights[mesh_topology]
                elif mesh_type in int_pts_and_weights:
                    current_int = int_pts_and_weights[mesh_type]
                else:
                    raise ValueError(f"No quadrature entry for '{mesh_topology}' or '{mesh_type}'.")
            else:
                current_int = int_pts_and_weights

            if mesh_type in ("line", "quad", "hexahedron"):
                mapping_basis = spaces.fem_line_quad_brick
            elif mesh_type in ("triangle", "tetra"):
                mapping_basis = spaces.fem_line_tri_tet
            else:
                raise NotImplementedError(f"Unknown mesh type '{mesh_type}'.")

            is_surface = elem_dim != int(self.dimension)
            phys_conn = self.mesh_info.cells[mesh_topology].nodes_of_cells[elem_ids].astype(int)
            set_infos.append(
                _L2ProjectionSetInfo(
                    domain_name=domain_name,
                    mesh_topology=mesh_topology,
                    mesh_type=mesh_type,
                    basis_cell_type=_canonical_base_type(mesh_topology),
                    domain_dim=int(current_int[0].shape[1]) if current_int[0].ndim >= 2 else 1,
                    is_surface=is_surface,
                    target_basis=mapping_basis,
                    mapping_basis=mapping_basis,
                    physical_connectivity=phys_conn,
                    target_global_connectivity=phys_conn,
                    volume_element_ids=elem_ids if not is_surface else None,
                    surface_domains=(domain_name,) if is_surface else tuple(),
                    surface_element_ids=elem_ids if is_surface else None,
                    ref_int_coor=current_int[0],
                    ref_int_weights=current_int[1],
                )
            )
            global_connectivities.append(phys_conn)

        if len(set_infos) == 0:
            raise ValueError(f"No compatible projection topology found for output domain '{domain_name}'.")
        return tuple(set_infos), global_connectivities

    def _geometry_projection_core_cache_key(
        self,
        domain_name: str,
        integration_order: Optional[int],
        int_pts_and_weights: Optional[Dict[str, Any]],
        integration_mode: Optional[str],
    ) -> Tuple[Any, ...]:
        mode = _normalize_integration_mode(integration_mode)
        spaces_by_field = dict(self.static_settings.get("spatial discretizations", {}))
        resolved_order = integration_order
        if resolved_order is None and mode == "auto":
            resolved_order = _default_integration_order_from_spaces(spaces_by_field) if spaces_by_field else 2
        order_key = None if resolved_order is None else int(resolved_order)
        return ("geometry", str(domain_name), mode, order_key, id(int_pts_and_weights) if int_pts_and_weights is not None else None)

    def _prepare_geometry_output_core(
        self,
        domain_name: str,
        *,
        name: str,
        integration_order: Optional[int] = None,
        int_pts_and_weights: Optional[Dict[str, Any]] = None,
        integration_mode: Optional[str] = None,
    ) -> _PreparedL2Projection:
        self._ensure_projector_cache()
        integration_mode = _normalize_integration_mode(integration_mode)
        cache_key = self._geometry_projection_core_cache_key(domain_name, integration_order, int_pts_and_weights, integration_mode)
        cached_name = self._geometry_projection_core_cache.get(cache_key, None)
        if cached_name is not None and cached_name in self._l2_projectors:
            return self._l2_projectors[cached_name]
        if name in self._l2_projectors:
            self._geometry_projection_core_cache[cache_key] = name
            return self._l2_projectors[name]

        set_infos, global_connectivities = self._build_geometry_output_set_infos(
            domain_name,
            integration_order=integration_order,
            int_pts_and_weights=int_pts_and_weights,
            integration_mode=integration_mode,
        )

        n_points = int(self.mesh_info.points.shape[0])
        global_scalar_ids, local_connectivities = _merge_projector_scalar_ids(global_connectivities, n_points)
        local_shape = (int(global_scalar_ids.shape[0]),)
        full_shape = (n_points,)

        settings_template = {
            "node coordinates": self.mesh_info.points,
            "connectivity": tuple(
                {
                    _PROJECTION_KEY: local_conn.astype(int),
                    "physical coor": info.physical_connectivity.astype(int),
                }
                for info, local_conn in zip(set_infos, local_connectivities)
            ),
        }

        zero_local = jnp.zeros(local_shape, dtype=float)
        zero_dofs = {_PROJECTION_KEY: zero_local}
        mass_models = tuple(
            _make_l2_projection_model(
                projection_key=_PROJECTION_KEY,
                target_basis=info.target_basis,
                mapping_basis=info.mapping_basis,
                field_shape=(),
                flat_components=1,
                domain_dim=info.domain_dim,
                ref_int_coor=info.ref_int_coor,
                ref_int_weights=info.ref_int_weights,
                quantity_fun=None,
                source_fields=tuple(),
                source_bases={},
                source_sets={},
                source_static_settings=self.static_settings,
            )
            for info in set_infos
        )
        mass_static_settings = FrozenDict({
            "assembling mode": tuple("user residual" for _ in set_infos),
            "model": mass_models,
        })

        mass_matrix = (-assembler.assemble_tangent(zero_dofs, settings_template, mass_static_settings))
        jacobi_diag = -assembler.assemble_tangent_diagonal(zero_dofs, settings_template, mass_static_settings)
        jacobi_inv_diag = jnp.where(jnp.abs(jacobi_diag) > 1e-14, 1.0 / jacobi_diag, 1.0)

        projector = _PreparedL2Projection(
            name=name,
            target_name="physical coor",
            field_shape=(),
            flat_components=1,
            global_scalar_ids=global_scalar_ids,
            local_shape=local_shape,
            full_shape=full_shape,
            set_infos=set_infos,
            settings_template=settings_template,
            mass_matrix=mass_matrix,
            jacobi_diag=jacobi_diag,
            jacobi_inv_diag=jacobi_inv_diag,
            outside_value=float("nan"),
        )
        self._l2_projectors[name] = projector
        self._geometry_projection_core_cache[cache_key] = name
        return projector

    def _build_preview_trial_ansatz(
        self,
        set_info: _L2ProjectionSetInfo,
        source_dofs: Dict[str, jnp.ndarray],
        source_settings: Dict[str, Any],
    ) -> Dict[str, Callable]:
        xI = self.mesh_info.points[set_info.physical_connectivity[0]]
        trial_ansatz = {
            "physical coor": lambda x: set_info.mapping_basis(x, xI, xI, source_settings, False, set_info.domain_dim)
        }
        for field in sorted(source_dofs.keys()):
            if field not in self.static_settings.get("spatial discretizations", {}):
                continue
            conn = self._build_projection_source_connectivity(field, set_info, strict=False)
            if conn is None or conn.shape[0] == 0:
                continue
            local_coeffs = source_dofs[field][conn[0]]
            space = self.static_settings["spatial discretizations"][field]
            basis = (
                _space_surface_basis_for_field(space, set_info.basis_cell_type, field)
                if set_info.is_surface
                else _space_basis_for_field(space, set_info.basis_cell_type, field)
            )
            basis_caller = models._make_caller_adapter(basis)
            trial_ansatz[field] = (
                lambda x, t, field=field, local_coeffs=local_coeffs, basis_caller=basis_caller:
                utility.scale_dof_value(
                    source_settings, field,
                    basis_caller(
                        x, xI, local_coeffs, source_settings, True, set_info.domain_dim, 0, 0
                    )
                )
            )
        return trial_ansatz

    def _infer_output_shapes(
        self,
        core_projector: _PreparedL2Projection,
        output_fun: Callable,
    ) -> Dict[str, Tuple[int, ...]]:
        if self.mesh_info is None:
            raise RuntimeError("Call import_mesh(...) before configuring output.")
        self._ensure_fe_built()

        set_info = core_projector.set_infos[0]
        source_settings = dict(self.settings)
        source_settings.setdefault("current time", 0.0)
        trial_ansatz = self._build_preview_trial_ansatz(set_info, self.dofs, source_settings)

        def preview_values():
            values = _call_output_dict_fun(
                output_fun,
                set_info.ref_int_coor[0],
                trial_ansatz,
                source_settings,
                self.static_settings,
                0,
                0,
                0,
            )
            return _OrderedOutputValues(
                tuple(values.keys()),
                tuple(jnp.asarray(value) for value in values.values()),
            )

        inferred = jax.eval_shape(preview_values)
        return {
            key: (() if value.ndim == 0 else tuple(value.shape))
            for key, value in zip(inferred.keys, inferred.arrays)
        }

    def _solve_grouped_output_projection(
        self,
        spec: FrozenDict,
        layouts: Tuple[_OutputComponentLayout, ...],
        dofs: Dict[str, jnp.ndarray],
        settings: Dict[str, Any],
        current_time: float,
    ) -> Dict[str, jnp.ndarray]:
        if len(layouts) == 0:
            return {}

        projector = self._l2_projectors[spec["core_projector_name"]]
        source_dofs_use = self.dofs if dofs is None else dofs
        total_components = layouts[-1].stop
        cache_key = (projector.name, id(spec["output_fun"]), tuple(layout.key for layout in layouts))
        cached_model = self._output_projection_model_cache.get(cache_key, None)
        if cached_model is None:
            resolved_source_fields, source_connectivity = self._resolve_source_fields_and_connectivity_cached(projector, None)
            quantity_fun = _make_grouped_output_quantity_fun(spec["output_fun"], layouts)
            source_bases_by_set = self._source_bases_for_projection(projector, resolved_source_fields)
            source_sets_by_set = self._source_sets_for_projection(projector, resolved_source_fields)
            models_for_sets = []
            for set_info, source_bases, source_sets in zip(projector.set_infos, source_bases_by_set, source_sets_by_set):
                models_for_sets.append(
                    _make_l2_projection_model(
                        projection_key=_PROJECTION_KEY,
                        target_basis=set_info.target_basis,
                        mapping_basis=set_info.mapping_basis,
                        field_shape=(total_components,),
                        flat_components=total_components,
                        domain_dim=set_info.domain_dim,
                        ref_int_coor=set_info.ref_int_coor,
                        ref_int_weights=set_info.ref_int_weights,
                        quantity_fun=quantity_fun,
                        source_fields=resolved_source_fields,
                        source_bases=source_bases,
                        source_sets=source_sets,
                        source_static_settings=self.static_settings,
                    )
                )

            static_settings = FrozenDict({
                "assembling mode": tuple("user residual" for _ in projector.set_infos),
                "model": tuple(models_for_sets),
            })
            cached_model = (static_settings, source_connectivity)
            self._output_projection_model_cache[cache_key] = cached_model
        else:
            static_settings, source_connectivity = cached_model

        projection_settings = dict(settings)
        projection_settings["current time"] = current_time
        projection_settings["node coordinates"] = projector.settings_template["node coordinates"]
        projection_settings["connectivity"] = projector.settings_template["connectivity"]
        projection_settings["projection source connectivity"] = source_connectivity
        projection_settings["projection source dofs"] = source_dofs_use

        n_local = int(projector.global_scalar_ids.shape[0])
        local_guess = self._output_projection_last_local.get(cache_key, None)
        if local_guess is None or tuple(local_guess.shape) != (n_local, total_components):
            local_guess = jnp.zeros((n_local, total_components), dtype=float)

        projection_dofs = {_PROJECTION_KEY: local_guess}
        residual = assembler.assemble_residual(projection_dofs, projection_settings, static_settings)[_PROJECTION_KEY]
        rhs = residual.reshape((n_local, total_components))
        delta = _solve_l2_projector_components(
            projector.mass_matrix,
            projector.jacobi_inv_diag,
            rhs,
            tol=float(spec.get("projection_tol", 1e-8)),
            atol=float(spec.get("projection_atol", 1e-10)),
            maxiter=int(spec.get("projection_maxiter", 1000)),
        )
        if total_components == 1:
            delta = delta.reshape((n_local, 1))
        local_solution = local_guess + delta
        self._output_projection_last_local[cache_key] = local_solution

        return {
            layout.export_name: _scatter_output_components(projector, local_solution, layout)
            for layout in layouts
        }

    def _materialize_output_specs(self) -> None:
        self._ensure_projector_cache()
        if self.mesh_info is None:
            raise RuntimeError("Call import_mesh(...) before configuring output.")

        raw_specs = tuple(self._raw_output_specs)
        start = len(self._prepared_output_specs)
        if start >= len(raw_specs):
            return
        self._ensure_fe_built()

        prepared = []
        for spec_id, raw_spec in enumerate(raw_specs[start:], start=start):
            domains = tuple(raw_spec["domains"])
            for domain_idx, domain_name in enumerate(domains):
                core_name = f"__output_core__{spec_id}__{domain_idx}"
                core_projector = self._prepare_geometry_output_core(
                    domain_name,
                    name=core_name,
                    integration_order=raw_spec["integration_order"],
                    int_pts_and_weights=raw_spec["int_pts_and_weights"],
                    integration_mode=raw_spec.get("integration_mode", None),
                )
                shapes = self._infer_output_shapes(
                    core_projector,
                    raw_spec["output_fun"],
                )
                prepared.append(FrozenDict({
                    "domain_name": domain_name,
                    "output_fun": raw_spec["output_fun"],
                    "core_projector_name": core_projector.name,
                    "output_keys": tuple(shapes.keys()),
                    "output_shapes": FrozenDict({key: tuple(shape) for key, shape in shapes.items()}),
                    "append_domain": bool(len(domains) > 1),
                    "projection_tol": float(raw_spec.get("projection_tol", 1e-8)),
                    "projection_atol": float(raw_spec.get("projection_atol", 1e-10)),
                    "projection_maxiter": int(raw_spec.get("projection_maxiter", 1000)),
                }))

        self._prepared_output_specs = tuple(self._prepared_output_specs) + tuple(prepared)

    def _clear_output_projection_state(self) -> None:
        self._ensure_projector_cache()
        output_projector_names = tuple(
            name for name in self._l2_projectors.keys()
            if str(name).startswith("__output_core__")
        )
        for name in output_projector_names:
            self._l2_projectors.pop(name, None)
        self._geometry_projection_core_cache = {
            key: value
            for key, value in self._geometry_projection_core_cache.items()
            if not str(value).startswith("__output_core__")
        }
        self._prepared_output_specs = tuple()
        self._output_projection_last_local = {}
        self._output_projection_model_cache = {}

    def _evaluate_registered_outputs(
        self,
        dofs: Dict[str, jnp.ndarray],
        settings: Dict[str, Any],
        output_fields: Optional[Union[str, list[str], tuple[str, ...]]] = None,
    ) -> Dict[str, jnp.ndarray]:
        self._materialize_output_specs()
        selected = None if output_fields is None else set(_as_list(output_fields))
        current_time = float(settings.get("current time", self.settings.get("current time", 0.0)))
        values: Dict[str, jnp.ndarray] = {}

        for spec in self._prepared_output_specs:
            layouts = _output_component_layouts(spec, selected)
            projected_values = self._solve_grouped_output_projection(
                spec,
                layouts,
                dofs,
                settings,
                current_time,
            )
            for preferred_name, projected in projected_values.items():
                if preferred_name in values:
                    # Same field on another domain: merge under one key. Each
                    # per-domain projection is NaN outside its domain, so fill
                    # the still-unset entries and keep the already-filled ones.
                    existing = values[preferred_name]
                    if jnp.issubdtype(existing.dtype, jnp.floating) and jnp.issubdtype(projected.dtype, jnp.floating):
                        values[preferred_name] = jnp.where(jnp.isnan(existing), projected, existing)
                    else:
                        values[preferred_name] = projected
                else:
                    values[preferred_name] = projected

        return values

    def _has_active_vtu_postprocessing_policy(self) -> bool:
        self._ensure_projector_cache()
        policy = self._vtu_postprocessing_policy
        return policy is not None and not isinstance(policy, dae.SaveNothingPolicy)

    def _prepare_vtu_output_group(
        self,
        projector: _PreparedL2Projection,
        layouts: Tuple[_OutputComponentLayout, ...],
        quantity_fun: Callable,
        source_fields: Optional[Tuple[str, ...]],
        cache_key: Tuple[Any, ...],
        projection_tol: float,
        projection_atol: float,
        projection_maxiter: int,
    ) -> Tuple[_VtuOutputGroup, _VtuOutputRuntime]:
        total_components = layouts[-1].stop
        cached_model = self._output_projection_model_cache.get(cache_key, None)
        if cached_model is None:
            resolved_source_fields, source_connectivity = self._resolve_source_fields_and_connectivity_cached(projector, source_fields)
            source_bases_by_set = self._source_bases_for_projection(projector, resolved_source_fields)
            source_sets_by_set = self._source_sets_for_projection(projector, resolved_source_fields)
            models_for_sets = []
            for set_info, source_bases, source_sets in zip(projector.set_infos, source_bases_by_set, source_sets_by_set):
                models_for_sets.append(
                    _make_l2_projection_model(
                        projection_key=_PROJECTION_KEY,
                        target_basis=set_info.target_basis,
                        mapping_basis=set_info.mapping_basis,
                        field_shape=(total_components,),
                        flat_components=total_components,
                        domain_dim=set_info.domain_dim,
                        ref_int_coor=set_info.ref_int_coor,
                        ref_int_weights=set_info.ref_int_weights,
                        quantity_fun=quantity_fun,
                        source_fields=resolved_source_fields,
                        source_bases=source_bases,
                        source_sets=source_sets,
                        source_static_settings=self.static_settings,
                    )
                )

            static_settings = FrozenDict({
                "assembling mode": tuple("user residual" for _ in projector.set_infos),
                "model": tuple(models_for_sets),
            })
            cached_model = (static_settings, source_connectivity)
            self._output_projection_model_cache[cache_key] = cached_model
        else:
            static_settings, source_connectivity = cached_model

        runtime = _VtuOutputRuntime(
            global_scalar_ids=projector.global_scalar_ids,
            node_coordinates=projector.settings_template["node coordinates"],
            connectivity=projector.settings_template["connectivity"],
            source_connectivity=source_connectivity,
            mass_matrix=projector.mass_matrix,
            jacobi_inv_diag=projector.jacobi_inv_diag,
        )
        group = _VtuOutputGroup(
            static_settings=static_settings,
            layouts=tuple(layouts),
            cache_key=cache_key,
            n_local=int(projector.global_scalar_ids.shape[0]),
            n_points=int(projector.full_shape[0]),
            total_components=int(total_components),
            outside_value=float(projector.outside_value),
            projection_tol=float(projection_tol),
            projection_atol=float(projection_atol),
            projection_maxiter=int(projection_maxiter),
        )
        return group, runtime

    def _prepare_vtu_model_output_group(
        self,
        spec: FrozenDict,
        layouts: Tuple[_OutputComponentLayout, ...],
    ) -> Tuple[_VtuOutputGroup, _VtuOutputRuntime]:
        projector = self._l2_projectors[spec["core_projector_name"]]
        cache_key = (projector.name, id(spec["output_fun"]), tuple(layout.key for layout in layouts))
        return self._prepare_vtu_output_group(
            projector,
            layouts,
            _make_grouped_output_quantity_fun(spec["output_fun"], layouts),
            None,
            cache_key,
            spec.get("projection_tol", 1e-8),
            spec.get("projection_atol", 1e-10),
            spec.get("projection_maxiter", 1000),
        )

    def _prepare_vtu_postprocessing_policy(self) -> None:
        self._ensure_projector_cache()
        if self.manager is None:
            raise RuntimeError("Call initialize(...) before preparing VTU postprocessing.")

        policy = self._vtu_postprocessing_policy
        if policy is None or isinstance(policy, dae.SaveNothingPolicy):
            self.manager.save_policy = None
            self.settings.pop("_vtu_output_runtime", None)
            return
        if self._source_mesh is None or self.mesh_info is None:
            raise RuntimeError("Call import_mesh(...) before configuring VTU postprocessing.")

        self._materialize_output_specs()
        file_stem, folder = _vtu_output_layout(self._vtu_postprocessing_result_folder_name)
        mesh = _prepare_mesh_for_vtu(copy.deepcopy(self._source_mesh))
        existing_point_data = getattr(mesh, "point_data", None)
        mesh.point_data = {} if existing_point_data is None else dict(existing_point_data)

        n_points = int(self.mesh_info.points.shape[0])

        point_data_names = []
        point_data_templates = []
        point_data_name_set = set(mesh.point_data.keys())
        output_groups = []
        output_runtimes = []

        export_to_final: Dict[str, str] = {}
        for spec in self._prepared_output_specs:
            layouts = _output_component_layouts(spec, None)
            if len(layouts) == 0:
                continue

            for layout in layouts:
                zero_shape = (n_points,) if len(layout.field_shape) == 0 else (n_points,) + layout.field_shape
                zero_values = np.zeros(zero_shape, dtype=np.asarray(self.mesh_info.points).dtype)
                template = _format_vtu_point_data(zero_values, n_points)

                # The same output field on different domains is written under one
                # VTU key: the per-domain L2 projections run separately (each is
                # NaN outside its own domain) and are merged at write time. Only
                # genuinely different fields get distinct names.
                if layout.export_name in export_to_final:
                    final_name = export_to_final[layout.export_name]
                else:
                    final_name = layout.export_name
                    counter = 2
                    while final_name in point_data_name_set:
                        final_name = f"{layout.export_name}__output" if counter == 2 else f"{layout.export_name}__output__{counter}"
                        counter += 1
                    point_data_name_set.add(final_name)
                    export_to_final[layout.export_name] = final_name

                point_data_names.append(final_name)
                point_data_templates.append(template)

            group, runtime = self._prepare_vtu_model_output_group(
                spec,
                layouts,
            )
            output_groups.append(group)
            output_runtimes.append(runtime)

        writer = _PreparedVtuHostWriter(
            mesh=mesh,
            folder=folder,
            file_stem=file_stem,
            point_data_names=tuple(point_data_names),
            point_data_templates=tuple(point_data_templates),
        )
        self.settings["_vtu_output_runtime"] = tuple(output_runtimes)
        self.manager.save_policy = _VtuCallbackPolicy(
            policy,
            writer,
            tuple(output_groups),
        )
        def vtu_postprocessing_fun(q_fun, t, settings):
            output_settings = dict(settings)
            output_settings[_VTU_OUTPUT_SOURCE_DOFS_KEY] = q_fun
            return output_settings

        self.manager.postprocessing_fun = vtu_postprocessing_fun
        self._vtu_postprocessing_output_folder = folder
        self._vtu_postprocessing_file_stem = file_stem
        self._vtu_postprocessing_callback = writer
        self._vtu_postprocessing_callback_installed = True

    def _rebuild_mesh_info_from_source(self) -> None:
        # Domains added via _add_domain(...) live only in the derived mesh_info,
        # not in the source mesh cell_sets, so they would be dropped by the
        # rebuild. Cell ordering is deterministic and identical to the source
        # build, so their cell_indices stay valid; carry them over below.
        preserved_domains = dict(self.mesh_info.domains) if self.mesh_info is not None else {}

        mesh_info = _build_mesh_info(self._source_mesh)
        pts_red, n_dim = _reduce_points(np.asarray(self._source_mesh.points))
        self.dimension = n_dim
        mesh_info = type(mesh_info)(points=pts_red, domains=mesh_info.domains, cells=mesh_info.cells)

        all_dom: Dict[str, _DomainInfo] = {}
        for ct, cinfo in mesh_info.cells.items():
            n = int(cinfo.nodes_of_cells.shape[0])
            if n > 0 and _cell_dim(ct) == int(self.dimension):
                all_dom[ct] = _DomainInfo(cell_indices=jnp.arange(n, dtype=int))
        mesh_info.domains.setdefault("__all__", all_dom)

        # Re-apply user-defined domains; source-derived domains (facet sets,
        # "__all__") take precedence, so only genuinely user-added ones survive.
        for name, dom in preserved_domains.items():
            mesh_info.domains.setdefault(name, dom)

        self.mesh_info = mesh_info
        self.settings["node coordinates"] = mesh_info.points

    def _clear_projection_caches(self) -> None:
        """Drop the geometry-derived projection/boundary caches.

        Shared by both cache-lifecycle operations
        (:meth:`_reset_discretization_dependent_caches` and
        :meth:`_cleanup_after_prepare`) so the set stays in sync in one place.
        """
        self._geometry_projection_core_cache = {}
        self._projection_source_cache = {}
        self._output_projection_last_local = {}
        self._output_projection_model_cache = {}

    def _clear_postprocessing_callbacks(self) -> None:
        self._base_apply_dirichlet_bcs = None
        self._base_post_step_updates = None
        self._vtu_postprocessing_callback = None
        self._vtu_postprocessing_callback_installed = False
        self._active_vtu_postprocessing_state = None

    def _reset_discretization_dependent_caches(self) -> None:
        """Invalidate everything that depends on the FE discretization.

        Called whenever the discretization changes (mesh import, active-domain
        changes); the runtime data lives on until :meth:`_cleanup_after_prepare`.
        """
        self._fe_signature = None
        self._fe = None
        self._l2_projectors = {}
        self._projected_dirichlet_specs = tuple()
        self.settings.pop(_PROJECTED_DIRICHLET_RUNTIME_KEY, None)
        self._prepared_output_specs = tuple()
        self._clear_projection_caches()
        self._clear_postprocessing_callbacks()

    def _cleanup_after_prepare(self) -> None:
        """Release the mesh/FE build data once the runtime is materialized.

        Distinct scope from :meth:`_reset_discretization_dependent_caches`: the
        Dirichlet projectors are kept (still needed at runtime), and the mesh
        build inputs are freed.
        """
        dirichlet_projector_names = {
            spec["projector_name"]
            for spec in tuple(getattr(self, "_projected_dirichlet_specs", tuple()))
        }
        self._l2_projectors = {
            name: projector
            for name, projector in self._l2_projectors.items()
            if name in dirichlet_projector_names
        }
        self._fe_signature = None
        self._fe = None
        self.mesh_info = None
        self.dimension = None
        self._source_mesh = None
        self._raw_local_subsystem_specs = tuple()
        self._raw_model_specs = tuple()
        self._raw_output_specs = tuple()
        self._raw_projected_dirichlet_specs = tuple()
        self._nodal_dirichlet_specs = tuple()
        self._nodal_load_specs = tuple()
        self._prepared_output_specs = tuple()
        self._pending_initial_condition_specs = []
        self._pending_restart_state = None
        self._clear_projection_caches()
        self._clear_postprocessing_callbacks()

    def _require_mesh_editable(self, method_name: str) -> None:
        self._ensure_projector_cache()
        if self._phase != "setup":
            raise RuntimeError(
                f"{method_name}(...) must be called before model/output/boundary-condition "
                "registration starts."
            )

    def _enter_registration_phase(self, method_name: str, *, require_mesh: bool = True) -> None:
        self._ensure_projector_cache()
        if self._phase not in ("setup", "registering"):
            raise RuntimeError(f"{method_name}(...) must be called before initialize(...).")
        if require_mesh and self.mesh_info is None:
            raise RuntimeError(f"Call import_mesh(...) before {method_name}(...).")
        self._phase = "registering"

    def _require_initialized_phase(self, method_name: str) -> None:
        self._ensure_projector_cache()
        if self._phase != "initialized":
            raise RuntimeError(f"{method_name}(...) must be called after initialize(...) and before run(...).")

    def _require_prepared_phase(self, method_name: str) -> None:
        self._ensure_projector_cache()
        if self._phase not in ("prepared", "ran"):
            raise RuntimeError(f"{method_name}(...) must be called after prepare(...).")

    def _mesh_top_dim(self) -> int:
        if self.mesh_info is None:
            return -1
        return max(
            (_cell_dim(ct) for ct, cinfo in self.mesh_info.cells.items()
             if int(cinfo.nodes_of_cells.shape[0]) > 0),
            default=-1,
        )

    def _add_boundary_domain_to_state(self, on_boundary_fun: Callable, domain_name: str,
                                      cell_dim: Optional[int] = None) -> str:
        if getattr(self, "_source_mesh", None) is None:
            raise RuntimeError("Call import_mesh(...) before adding boundary conditions.")

        point_selector = self._make_boundary_point_selector(on_boundary_fun)

        # Find boundary facets from already-built mesh_info (no rebuild needed for validation).
        found = _find_boundary_facets(self.mesh_info, point_selector, cell_dim=cell_dim)
        if not found:
            raise ValueError("Boundary predicate did not select any boundary facets.")

        # Register domain + placeholder cells in mesh_info immediately so that:
        #   - mesh_info.domains[domain_name] is visible to callers right away
        #   - subsequent add_weak_bc name-collision checks work correctly
        # entity_id / facets_of_cells are left None here; they are filled correctly
        # by _rebuild_mesh_info_from_source() which is called once in initialize().
        domain_entry: Dict[str, _DomainInfo] = {}
        for facet_type, fconn in found.items():
            n_facets = int(fconn.shape[0])
            if facet_type not in self.mesh_info.cells:
                self.mesh_info.cells[facet_type] = _CellInfo(
                    facets_of_cells=None,
                    edges_of_cells=None,
                    nodes_of_cells=jnp.asarray(fconn),
                    entity_id=None,
                    entity_sign=None,
                )
                cell_indices = jnp.arange(n_facets, dtype=jnp.int64)
            else:
                # Additional BC domain for an already-present facet cell type.
                # Track offset into the existing cell block.
                existing_n = int(self.mesh_info.cells[facet_type].nodes_of_cells.shape[0])
                cell_indices = jnp.arange(existing_n, existing_n + n_facets, dtype=jnp.int64)
            domain_entry[facet_type] = _DomainInfo(cell_indices=cell_indices)
        self.mesh_info.domains[domain_name] = domain_entry

        # Update source mesh for VTU output; defer full rebuild to initialize().
        _add_facet_domain(self._source_mesh, domain_name, point_selector=point_selector, cell_dim=cell_dim)
        self._source_mesh_modified = True
        self._reset_discretization_dependent_caches()
        return domain_name

    def _ensure_all_boundary_mesh_info(self) -> _MeshInfo:
        """Return the single mesh_info that already carries *all* boundary facets.

        The all-boundary facet set is folded into the one topology build in
        :meth:`initialize` (see there), so strong-BC projectors share the
        volume's entity numbering directly. Per-BC selection then only filters
        this facet set (see :meth:`_temporary_boundary_mesh_info`), and no
        separate ``_build_mesh_info`` is needed.
        """
        if self.mesh_info is None:
            raise RuntimeError("Call import_mesh(...) before add_strong_bc(...).")
        if _ALL_BOUNDARY_FACET_DOMAIN not in self.mesh_info.domains:
            # Strong BC registered after initialize(): the all-boundary facets
            # were not folded into the build yet. Add them to the source and
            # rebuild once. Under the canonical (key-sorted) entity numbering the
            # volume ids are stable when facet cells are added, so the FE system
            # built in initialize() stays valid.
            if self._source_mesh is None:
                raise RuntimeError("Call import_mesh(...) before add_strong_bc(...).")
            existing_sets = getattr(self._source_mesh, "cell_sets", None) or {}
            if _ALL_BOUNDARY_FACET_DOMAIN not in existing_sets:
                _add_facet_domain(self._source_mesh, _ALL_BOUNDARY_FACET_DOMAIN, point_selector=None)
            self._rebuild_mesh_info_from_source()
        return self.mesh_info

    def _temporary_boundary_mesh_info(self, on_boundary_fun: Callable, domain_name: str) -> _MeshInfo:
        mesh_info = self._ensure_all_boundary_mesh_info()
        all_boundary = mesh_info.domains.get(_ALL_BOUNDARY_FACET_DOMAIN, {})
        selector = self._make_boundary_point_selector(on_boundary_fun)
        points = np.asarray(mesh_info.points)

        selected: Dict[str, _DomainInfo] = {}
        for ftype, dinfo in all_boundary.items():
            cell_ids = np.asarray(dinfo.cell_indices, dtype=int)
            if cell_ids.size == 0:
                continue
            facet_nodes = np.asarray(mesh_info.cells[ftype].nodes_of_cells)[cell_ids]
            coords = points[facet_nodes]  # (n_facets, n_nodes_per_facet, gdim)
            node_mask = np.asarray(
                selector(coords.reshape(-1, coords.shape[-1])), dtype=bool
            ).reshape(coords.shape[0], coords.shape[1])
            facet_mask = node_mask.all(axis=1)
            chosen = cell_ids[facet_mask]
            if chosen.size > 0:
                selected[ftype] = _DomainInfo(cell_indices=chosen)

        if not selected:
            raise ValueError("Boundary predicate did not select any boundary facets.")
        mesh_info.domains[domain_name] = selected
        return mesh_info

    def _prepare_boundary_l2_projection(
        self,
        field: str,
        on_boundary_fun: Callable,
        *,
        name: str,
        integration_order: Optional[int] = None,
        int_pts_and_weights: Optional[Dict[str, Any]] = None,
        integration_mode: Optional[str] = None,
    ) -> _PreparedL2Projection:
        self._ensure_projector_cache()
        if name in self._l2_projectors:
            return self._l2_projectors[name]

        boundary_domain = f"__dirichlet_boundary__{crc32(name.encode('utf-8')) & 0xFFFFFFFF:08x}"
        boundary_mesh_info = self._temporary_boundary_mesh_info(on_boundary_fun, boundary_domain)
        saved_mesh_info = self.mesh_info
        try:
            self.mesh_info = boundary_mesh_info
            return self._prepare_l2_projection(
                field,
                boundary_domain,
                name=name,
                integration_order=integration_order,
                int_pts_and_weights=int_pts_and_weights,
                integration_mode=integration_mode,
            )
        finally:
            self.mesh_info = saved_mesh_info

    def _add_nodal_dirichlet_spec(self, field, on_boundary_fun, value_fun, index, reset) -> None:
        """Register a point (0-dimensional-boundary) Dirichlet condition.

        Used for 1D fields, whose boundary is a set of endpoints. The value is
        set directly on the field's node DOFs at the selected points (no L2
        projection). DOF ids are resolved lazily once the FE numbering is final.
        """
        if getattr(self, "_source_mesh", None) is None:
            raise RuntimeError("Call import_mesh(...) before add_strong_bc(...).")
        self._nodal_dirichlet_specs = tuple(self._nodal_dirichlet_specs) + (FrozenDict({
            "field": field,
            "on_boundary_fun": on_boundary_fun,
            "value_fun": value_fun,
            "index": index,
            "reset": bool(reset),
        }),)
        self._reset_discretization_dependent_caches()

    def _add_nodal_load_spec(self, field, on_boundary_fun, value_fun, index, sign) -> None:
        """Register a point (0-dimensional-boundary) weak BC / nodal load.

        Used for 1D fields, whose boundary is a set of endpoints. The load is
        added directly to the residual at the field's node DOFs (no surface
        integral). DOF ids are resolved lazily once the FE numbering is final.
        """
        if getattr(self, "_source_mesh", None) is None:
            raise RuntimeError("Call import_mesh(...) before add_weak_bc(...).")
        self._nodal_load_specs = tuple(self._nodal_load_specs) + (FrozenDict({
            "field": field,
            "on_boundary_fun": on_boundary_fun,
            "value_fun": value_fun,
            "index": index,
            "sign": float(sign),
        }),)
        self._reset_discretization_dependent_caches()

    def _install_nodal_load_callback(self) -> None:
        """Install external nodal loads into the manager's residual (point weak BCs)."""
        if self.manager is None:
            return
        loads = tuple(
            (self._resolve_nodal_dirichlet(s), s["value_fun"], float(s["sign"]))
            for s in tuple(self._nodal_load_specs)
        )
        if not loads:
            return

        def nodal_residual_fun(residual, settings, t):
            residual = dict(residual)
            for (field, scalar_ids, coords, field_shape, comp_mask), value_fun, sign in loads:
                vals = _eval_nodal_bc_values(value_fun, coords, t, settings, field, field_shape, unscale=False)
                m = comp_mask.reshape(field_shape)[None, ...]  # (1, *field_shape)
                contrib = jnp.where(m, sign * vals, jnp.zeros_like(vals))
                residual[field] = residual[field].at[scalar_ids].add(contrib)
            return residual

        self.manager.static_settings = self.manager.static_settings.copy(add_or_replace={
            "nodal residual fun": nodal_residual_fun,
        })

    def _resolve_nodal_dirichlet(self, spec):
        """Resolve a nodal Dirichlet spec to (field, scalar_dof_ids, coords, mask)."""
        self._ensure_fe_built()
        field = spec["field"]
        pack = self._fe[field]
        space = self.static_settings["spatial discretizations"][field]
        field_shape = _field_shape_from_space(space)

        points = np.asarray(self.mesh_info.points)
        selector = self._make_boundary_point_selector(spec["on_boundary_fun"])
        node_mask = np.asarray(selector(points), dtype=bool)
        global_nodes = np.flatnonzero(node_mask)

        node_g2a = np.asarray(pack.g2a["node_g2a"])
        active = node_g2a[global_nodes]
        keep = active >= 0
        if not np.any(keep):
            raise ValueError(f"[{field}] point Dirichlet predicate selected no active nodes.")
        global_nodes = global_nodes[keep]
        active = active[keep]

        node_offset = int(pack.layout["node_offset"])
        node_block = int(pack.layout["node_block"])
        scalar_ids = jnp.asarray(node_offset + active * node_block, dtype=int)
        coords = jnp.asarray(points[global_nodes])
        mask = jnp.asarray(_component_mask(field_shape, spec["index"]), dtype=bool)
        return field, scalar_ids, coords, field_shape, mask

    def _materialize_projected_dirichlet_specs(self) -> None:
        self._ensure_projector_cache()
        raw_specs = tuple(self._raw_projected_dirichlet_specs)
        start = len(self._projected_dirichlet_specs)
        if start >= len(raw_specs):
            return

        materialized = []
        for spec_id, raw_spec in enumerate(raw_specs[start:], start=start):
            projector_name = raw_spec.get("projector_name", None)
            if projector_name is None:
                projector_name = f"__dirichlet__{raw_spec['field']}__{spec_id}"

            boundary_domain = raw_spec.get("boundary_domain", None)
            if boundary_domain is not None:
                # Lower-dimensional field: project onto its own (already-built)
                # field-dimensional boundary domain (see add_strong_bc).
                projector = self._prepare_l2_projection(
                    raw_spec["field"],
                    boundary_domain,
                    name=projector_name,
                    integration_order=raw_spec["integration_order"],
                    int_pts_and_weights=raw_spec["int_pts_and_weights"],
                    integration_mode=raw_spec.get("integration_mode", None),
                )
            else:
                projector = self._prepare_boundary_l2_projection(
                    raw_spec["field"],
                    raw_spec["on_boundary_fun"],
                    name=projector_name,
                    integration_order=raw_spec["integration_order"],
                    int_pts_and_weights=raw_spec["int_pts_and_weights"],
                    integration_mode=raw_spec.get("integration_mode", None),
                )
            quantity_fun = _wrap_projected_value_fun(
                projector.field_shape,
                raw_spec["value_fun"],
                raw_spec["index"],
                raw_spec["field"],
            )
            models_for_sets = tuple(
                _make_l2_projection_model(
                    projection_key=_PROJECTION_KEY,
                    target_basis=set_info.target_basis,
                    mapping_basis=set_info.mapping_basis,
                    field_shape=projector.field_shape,
                    flat_components=projector.flat_components,
                    domain_dim=set_info.domain_dim,
                    ref_int_coor=set_info.ref_int_coor,
                    ref_int_weights=set_info.ref_int_weights,
                    quantity_fun=quantity_fun,
                    source_fields=tuple(),
                    source_bases={},
                    source_sets={},
                    source_static_settings=self.static_settings,
                )
                for set_info in projector.set_infos
            )
            materialized.append(FrozenDict({
                "field": raw_spec["field"],
                "projector_name": projector.name,
                "value_fun": raw_spec["value_fun"],
                "index": raw_spec["index"],
                "reset": raw_spec["reset"],
                "field_shape": tuple(projector.field_shape),
                "component_mask": _component_mask(projector.field_shape, raw_spec["index"]),
                "projection static settings": FrozenDict({
                    "assembling mode": tuple("user residual" for _ in projector.set_infos),
                    "model": models_for_sets,
                }),
                "tol": float(raw_spec["tol"]),
                "atol": float(raw_spec["atol"]),
                "maxiter": int(raw_spec["maxiter"]),
            }))

        self._projected_dirichlet_specs = tuple(self._projected_dirichlet_specs) + tuple(materialized)

    def _materialize_projected_dirichlet_runtime(self) -> None:
        specs = tuple(getattr(self, "_projected_dirichlet_specs", tuple()))
        if len(specs) == 0:
            self.settings.pop(_PROJECTED_DIRICHLET_RUNTIME_KEY, None)
            return

        runtimes = []
        for spec in specs:
            projector = self._l2_projectors[spec["projector_name"]]
            _, source_connectivity = self._resolve_source_fields_and_connectivity_cached(projector, tuple())
            runtimes.append(_ProjectedDirichletRuntime(
                global_scalar_ids=projector.global_scalar_ids,
                node_coordinates=projector.settings_template["node coordinates"],
                connectivity=projector.settings_template["connectivity"],
                orientation=projector.settings_template.get("orientation", tuple({} for _ in projector.set_infos)),
                source_connectivity=source_connectivity,
                mass_matrix=projector.mass_matrix,
                jacobi_inv_diag=projector.jacobi_inv_diag,
            ))
        self.settings[_PROJECTED_DIRICHLET_RUNTIME_KEY] = tuple(runtimes)

    def _install_projected_dirichlet_callback(self) -> None:
        self._ensure_projector_cache()
        if self.manager is None:
            raise RuntimeError("Call initialize(...) before projected Dirichlet conditions are applied.")

        if self._base_apply_dirichlet_bcs is None:
            self._base_apply_dirichlet_bcs = self.manager.static_settings.get("apply dirichlet bcs", None)

        specs = tuple(self._projected_dirichlet_specs)
        base_apply = self._base_apply_dirichlet_bcs

        # Resolve nodal (point) Dirichlet specs now that the FE numbering is
        # final: (field, scalar dof ids, node coords, field shape, comp mask).
        nodal = tuple(
            (self._resolve_nodal_dirichlet(s), s["value_fun"], bool(s["reset"]))
            for s in tuple(self._nodal_dirichlet_specs)
        )

        def apply_dirichlet_bcs(t, settings):
            settings = dict(settings)
            settings["current time"] = t

            if base_apply is not None:
                settings = base_apply(t, settings)

            if settings.get("dirichlet dofs") is None:
                settings["dirichlet dofs"] = {field: jnp.zeros_like(vals, dtype=bool) for field, vals in self.dofs.items()}
            if settings.get("dirichlet conditions") is None:
                settings["dirichlet conditions"] = utility.dict_zeros_like(settings["dirichlet dofs"], dtype=float)

            # Point (nodal) Dirichlet: set the value directly on the node DOFs.
            for (field, scalar_ids, coords, field_shape, comp_mask), value_fun, reset in nodal:
                if reset:
                    settings["dirichlet dofs"][field] = jnp.zeros_like(settings["dirichlet dofs"][field])
                    settings["dirichlet conditions"][field] = jnp.zeros_like(settings["dirichlet conditions"][field])
                vals = _eval_nodal_bc_values(value_fun, coords, t, settings, field, field_shape)
                cond_f = settings["dirichlet conditions"][field]
                dofs_f = settings["dirichlet dofs"][field]
                cur_cond = cond_f[scalar_ids]
                cur_dofs = dofs_f[scalar_ids]
                m = comp_mask.reshape(field_shape)[None, ...]  # (1, *field_shape)
                new_cond = jnp.where(m, vals, cur_cond)
                new_dofs = jnp.where(m, jnp.asarray(True), cur_dofs)
                settings["dirichlet conditions"][field] = cond_f.at[scalar_ids].set(new_cond)
                settings["dirichlet dofs"][field] = dofs_f.at[scalar_ids].set(new_dofs)

            if not specs:
                return settings

            runtimes = settings.get(_PROJECTED_DIRICHLET_RUNTIME_KEY, None)
            if runtimes is None:
                raise RuntimeError("Projected Dirichlet runtime data is missing from settings. Call prepare(...) before run(...).")
            if len(runtimes) != len(specs):
                raise RuntimeError("Projected Dirichlet runtime data does not match the registered boundary conditions.")

            for spec, runtime in zip(specs, runtimes):
                field = spec["field"]
                if spec.get("reset", False):
                    settings["dirichlet dofs"][field] = jnp.zeros_like(settings["dirichlet dofs"][field])
                    settings["dirichlet conditions"][field] = jnp.zeros_like(settings["dirichlet conditions"][field])

                existing_full = settings["dirichlet conditions"][field]
                initial_guess = existing_full[runtime.global_scalar_ids]
                local_solution = _solve_projected_dirichlet_runtime(
                    runtime,
                    spec["projection static settings"],
                    settings,
                    initial_guess,
                    t,
                    tol=spec["tol"],
                    atol=spec["atol"],
                    maxiter=spec["maxiter"],
                )

                local_values = _reshape_projector_coefficients(local_solution, spec["field_shape"])
                comp_mask = jnp.asarray(spec["component_mask"], dtype=bool)
                cond_f = settings["dirichlet conditions"][field]
                dofs_f = settings["dirichlet dofs"][field]
                cond_rows = cond_f[runtime.global_scalar_ids]
                dofs_rows = dofs_f[runtime.global_scalar_ids]
                cond_rows_flat = _flatten_projector_values(cond_rows)
                dofs_rows_flat = _flatten_projector_values(dofs_rows)
                values_flat = _flatten_projector_values(local_values)
                if values_flat.ndim == 1:
                    values_flat = values_flat[:, None]
                    cond_rows_flat = cond_rows_flat[:, None]
                    dofs_rows_flat = dofs_rows_flat[:, None]

                apply_mask = comp_mask[None, :]
                cond_rows_flat = jnp.where(apply_mask, values_flat, cond_rows_flat)
                dofs_rows_flat = jnp.where(apply_mask, True, dofs_rows_flat)

                cond_rows = cond_rows_flat.reshape(cond_rows.shape)
                dofs_rows = dofs_rows_flat.reshape(dofs_rows.shape)
                settings["dirichlet conditions"][field] = settings["dirichlet conditions"][field].at[runtime.global_scalar_ids].set(cond_rows)
                settings["dirichlet dofs"][field] = settings["dirichlet dofs"][field].at[runtime.global_scalar_ids].set(dofs_rows)

            return settings

        self.manager.static_settings = self.manager.static_settings.copy(add_or_replace={
            "apply dirichlet bcs": apply_dirichlet_bcs,
        })

    def _has_internal_variable_updates(self) -> bool:
        names_by_set = tuple(self.static_settings.get("internal variable names", tuple()))
        return any(len(tuple(names)) > 0 for names in names_by_set)

    def _install_internal_variable_update_callback(self) -> None:
        if self.manager is None:
            return
        if self._base_post_step_updates is None:
            self._base_post_step_updates = self.manager.post_step_updates

        base_post_step_updates = self._base_post_step_updates
        static_settings = self.manager.static_settings
        names_by_set = tuple(static_settings.get("internal variable names", tuple()))
        has_internal_variable_updates = self._has_internal_variable_updates()

        def post_step_updates(q_fun, t, settings):
            settings = base_post_step_updates(q_fun, t, settings)
            if has_internal_variable_updates:
                current_blocks = list(settings.get("internal variables", tuple()))
                committed_blocks = list(settings.get("internal variables n", tuple(current_blocks)))

                for set_index, names in enumerate(names_by_set):
                    if len(tuple(names)) == 0:
                        continue
                    current_blocks[set_index] = assembler.get_int_var_updates(
                        q_fun, settings, static_settings, set_index
                    )
                    committed_blocks[set_index] = current_blocks[set_index]

                settings["internal variables"] = tuple(current_blocks)
                settings["internal variables n"] = tuple(committed_blocks)
            settings["last time"] = settings.get("current time", t)
            return settings

        self.manager.post_step_updates = post_step_updates

    def _primary_domain_scalar_ids(self, field: str, domain_name: str) -> jnp.ndarray:
        if self.mesh_info is None:
            raise RuntimeError("Call import_mesh(...) before set_initial_conditions(...).")
        self._ensure_fe_built()
        pack = self._fe[field]
        parts = []

        elem_ids_by_type = _collect_elem_ids_by_type(self.mesh_info, domain_name)
        for mesh_topology, elem_ids in elem_ids_by_type.items():
            cell_g2a = pack.g2a["cell_g2a"]
            if mesh_topology not in cell_g2a or mesh_topology not in pack.volume:
                continue
            rows = cell_g2a[mesh_topology][jnp.asarray(elem_ids, dtype=int)]
            rows = rows[rows >= 0]
            if rows.size:
                parts.append(jnp.ravel(pack.volume[mesh_topology][rows]))

        if domain_name in pack.surface:
            for conn in pack.surface[domain_name].values():
                if conn.size:
                    parts.append(jnp.ravel(conn))

        if not parts:
            raise ValueError(f"Domain '{domain_name}' selects no active DOFs for field '{field}'.")
        return _unique1d_int_fast(jnp.concatenate(parts, axis=0)).astype(int)

    def _apply_primary_initial_values(
        self,
        values: Dict[str, Any],
        integration_mode: Optional[str],
    ) -> None:
        integration_mode = _normalize_integration_mode(integration_mode)
        spaces_by_field = dict(self.static_settings.get("spatial discretizations", {}))
        field_names = tuple(spaces_by_field.keys())
        for domain_name, per_field in _normalize_domain_field_map(values, field_names, "values"):
            for field, value in per_field.items():
                field_shape = _field_shape_from_space(spaces_by_field[field])
                if callable(value):
                    mode_suffix = "" if integration_mode == "auto" else f"__{integration_mode}"
                    projector_name = f"__initial__{field}__{crc32(str(domain_name).encode('utf-8')) & 0xFFFFFFFF:08x}{mode_suffix}"
                    projector = self._prepare_l2_projection(
                        field,
                        domain_name,
                        name=projector_name,
                        integration_mode=integration_mode,
                    )
                    projected = self._l2_project(
                        projector_name=projector_name,
                        quantity_fun=_make_initial_value_quantity_fun(field_shape, value, field),
                        source_fields=tuple(),
                        source_dofs={},
                        source_settings=self.settings,
                        current_time=self.settings.get("current time", 0.0),
                        integration_mode=integration_mode,
                        initial_guess=self.dofs[field],
                    )
                    ids = projector.global_scalar_ids
                    self.dofs[field] = self.dofs[field].at[ids].set(projected[ids])
                    self._store_raw_callable_ic_spec(
                        projector_name,
                        domain_name,
                        field,
                        field_shape,
                        value,
                        integration_mode,
                    )
                else:
                    ids = self._primary_domain_scalar_ids(field, domain_name)
                    value = _validate_field_shaped_value(value, field_shape, field, "values")
                    value = utility.unscale_dof_value(self.settings, field, value)
                    fill = jnp.broadcast_to(value, (ids.shape[0],) + field_shape)
                    self.dofs[field] = self.dofs[field].at[ids].set(fill)

    def _apply_initial_derivatives(self, derivatives: Dict[str, Any]) -> None:
        spaces_by_field = dict(self.static_settings.get("spatial discretizations", {}))
        field_names = tuple(spaces_by_field.keys())
        stored = {
            field: {int(order): arr for order, arr in by_order.items()}
            for field, by_order in self.settings.get(dae._INITIAL_DERIVATIVES_KEY, {}).items()
        }
        for domain_name, per_field in _normalize_domain_field_map(derivatives, field_names, "derivatives"):
            for field, by_order in per_field.items():
                if not isinstance(by_order, dict):
                    raise TypeError(f"derivatives for field '{field}' must be a dict mapping derivative order to value.")
                field_shape = _field_shape_from_space(spaces_by_field[field])
                ids = self._primary_domain_scalar_ids(field, domain_name)
                stored.setdefault(field, {})
                for order, value in by_order.items():
                    order = int(order)
                    if order < 1:
                        raise ValueError("Derivative order must be >= 1.")
                    value = _validate_field_shaped_value(value, field_shape, field, "derivatives")
                    value = utility.unscale_dof_value(self.settings, field, value)
                    current = stored[field].get(order, jnp.zeros_like(self.dofs[field]))
                    fill = jnp.broadcast_to(value, (ids.shape[0],) + field_shape)
                    stored[field][order] = current.at[ids].set(fill)
        self.settings[dae._INITIAL_DERIVATIVES_KEY] = stored

    def _apply_internal_variable_initial_values(
        self,
        internal_variables: Dict[str, Any],
        *,
        allow_missing_blocks: bool,
    ) -> None:
        internal_spaces = dict(self.static_settings.get("internal variable discretizations", {}))
        field_names = tuple(internal_spaces.keys())
        normalized = _normalize_domain_field_map(internal_variables, field_names, "internal_variables")
        if not normalized:
            return
        if "internal variables" not in self.settings:
            if allow_missing_blocks:
                return
            raise RuntimeError("Internal variables are not materialized yet. Call add_model(...) and initialize(...).")

        current_blocks = list(self.settings.get("internal variables", tuple()))
        committed_blocks = list(self.settings.get("internal variables n", tuple(current_blocks)))
        domain_names = tuple(self.static_settings.get("domains", tuple()))
        names_by_set = tuple(self.static_settings.get("internal variable names", tuple()))

        for domain_name, per_field in normalized:
            matching_sets = tuple(i for i, name in enumerate(domain_names) if name == domain_name)
            if not matching_sets:
                if allow_missing_blocks:
                    continue
                raise KeyError(f"Unknown model domain '{domain_name}'. Available: {list(domain_names)}")
            for set_index in matching_sets:
                block = dict(current_blocks[set_index])
                committed = dict(committed_blocks[set_index])
                active_names = set(names_by_set[set_index])
                for field, value in per_field.items():
                    if field not in active_names:
                        raise ValueError(f"Internal variable '{field}' is not active on model domain '{domain_name}'.")
                    field_shape = _field_shape_from_space(internal_spaces[field])
                    value = _validate_field_shaped_value(value, field_shape, field, "internal_variables")
                    fill = jnp.broadcast_to(value, block[field].shape)
                    block[field] = fill
                    committed[field] = fill
                current_blocks[set_index] = block
                committed_blocks[set_index] = committed

        self.settings["internal variables"] = tuple(current_blocks)
        self.settings["internal variables n"] = tuple(committed_blocks)

    def _apply_restart_state(self, result: dae.TimeSteppingManagerState) -> None:
        if self.mesh_info is not None:
            self._ensure_fe_built()
        elif not self.dofs:
            raise RuntimeError("Call import_mesh(...) before set_initial_conditions(...).")
        self.dofs = {field: jnp.asarray(value) for field, value in result.q.items()}
        self.settings = dict(result.settings)
        self.settings[dae._RESTART_Q_N_KEY] = result.q_n
        self.settings[dae._RESTART_Q_DER_N_KEY] = result.q_der_n
        self.settings[dae._RESTART_CONTROLLER_STATE_KEY] = result.controller_state
        self.settings[dae._RESTART_DT_KEY] = result.dt
        self.settings["current time"] = self.settings.get("current time", 0.0)
        self.settings.setdefault("last time", self.settings["current time"])

    def _apply_manual_initial_conditions(
        self,
        *,
        values=None,
        derivatives=None,
        internal_variables=None,
        time=None,
        integration_mode: Optional[str] = None,
        allow_missing_internal_blocks=False,
    ) -> None:
        integration_mode = _normalize_integration_mode(integration_mode)
        if self.mesh_info is None:
            raise RuntimeError("Call import_mesh(...) before set_initial_conditions(...).")
        _clear_restart_settings(self.settings)
        if values is not None:
            self._apply_primary_initial_values(values, integration_mode)
        if derivatives is not None:
            self._apply_initial_derivatives(derivatives)
        if internal_variables is not None:
            self._apply_internal_variable_initial_values(
                internal_variables,
                allow_missing_blocks=allow_missing_internal_blocks,
            )
        if time is not None:
            self.settings["current time"] = time
            self.settings["last time"] = time

    def _apply_pending_initial_conditions(self) -> None:
        restart_state = getattr(self, "_pending_restart_state", None)
        if restart_state is not None:
            self._apply_restart_state(restart_state)
            self._pending_restart_state = None

        pending = tuple(getattr(self, "_pending_initial_condition_specs", tuple()))
        for spec in pending:
            self._apply_manual_initial_conditions(
                values=spec["values"],
                derivatives=spec["derivatives"],
                internal_variables=spec["internal_variables"],
                time=spec["time"],
                integration_mode=spec.get("integration_mode", None),
                allow_missing_internal_blocks=False,
            )
        self._pending_initial_condition_specs = []
