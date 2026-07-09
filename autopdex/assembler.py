# assembler.py
# Copyright (C) 2024 Tobias Bode
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

"""
Module for assembling and integrating functionals, residuals and tangents, supporting both 
'sparse' and 'dense' modes and 'user potentials', 'user residuals' and 'user elements'.

The assembly modes 'dense' and 'sparse' use JAX's automatic differentiation capabilities to 
determine the global residual and the tangent for the given values of the degrees of freedom 
and the models and variational schemes specified in static_settings.
In the 'dense' mode, the tangent matrix is returned as a jnp.ndarray, while in the 'sparse' 
mode, it is returned as a jax.experimental.BCOO matrix with duplicates. 
In the 'user potential', 'user residual' and 'user element' modes, the global residual and 
tangent can be assembled based on user-defined element-wise contributions. 
Depending on the execution location, the entries are calculated on CPU or GPU. 
For an efficient calculation, JAX's automatic vectorization transformation (vmap) is used.
The summation of duplicates is then carried out within the solver module. Currently, 
SciPy is used for this on the CPU.
"""

from functools import partial
from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp
import jax
from jax import vmap, jacrev, jacfwd, hessian, jvp
from jax.tree import map as treemap
from jax.experimental import sparse

from autopdex import variational_schemes
from autopdex.dae import discrete_value_with_derivatives
from autopdex.utility import jit_with_docstring, dict_zeros_like, dict_flatten, reshape_as


@jax.tree_util.register_dataclass
@dataclass
class AssemblingTemplate:
    """
    Container for all data needed to map raw element-wise tangent entries
    to a consolidated sparse matrix representation.
    """
    nnz: int
    col_sorted: jnp.ndarray
    indptr: jnp.ndarray
    scatter: jnp.ndarray
    indices_unique: jnp.ndarray


@jax.tree_util.register_dataclass
@dataclass
class _q_fun_state:
    q_ts: dict[str, tuple]

    def __call__(self, t):
        return {
            key: discrete_value_with_derivatives(t, q_tup[0], q_tup[1:])
            for key, q_tup in self.q_ts.items()
        }


def _dofs_reference(dofs):
    return {key: q_tup[0] for key, q_tup in dofs.q_ts.items()} if isinstance(dofs, _q_fun_state) else dofs


def _num_dofs(dofs):
    dofs_ref = _dofs_reference(dofs)
    return dofs_ref.size if not isinstance(dofs_ref, dict) else sum(v.size for v in dofs_ref.values())


def _block_field_keys(connectivity, dofs):
    """DOF fields present in this element block's connectivity, in global DOF order.

    A model registered on a subdomain only couples the fields that are active on
    that subdomain, so a per-block connectivity dict may hold a subset of the
    global fields. Iterating this subset (instead of the global ``dofs.keys()``)
    keeps assembled values and indices aligned, while global DOF offsets/sizes
    are still computed from the full ``dofs``.
    """
    dofs_ref = _dofs_reference(dofs)
    return [k for k in dofs_ref.keys() if k in connectivity]


def _make_assembling_kernel(nnz: int):
    """
    nnz : Number of unique matrix entries after consolidation.
    """
    @jax.jit
    def kernel(values, scatter):
        out_shape = (nnz,) + values.shape[1:]
        data = jnp.zeros(out_shape, dtype=values.dtype)
        return data.at[scatter].add(values)

    return kernel


## Helper functions
def _get_indices(connectivity, dofs):
    """
    Constructs the global indices for the assembly of the tangent matrix.

    Args:
        connectivity (array or dict): Connectivity array or dictionary of connectivity arrays.
        dofs (array or dict): DOFs array or dictionary of DOFs.

    Returns:
        indices (jnp.ndarray): Array of indices for the sparse matrix.
    """
    dofs = _dofs_reference(dofs)

    if isinstance(dofs, dict):
        # Field-Offsets are global (span the full DOF vector) ...
        field_offsets = {}
        current_offset = 0
        for field in dofs.keys():
            field_offsets[field] = current_offset
            field_size = dofs[field].size
            current_offset += field_size

        # ... but the coupling is only over the fields present in this block.
        keys = _block_field_keys(connectivity, dofs)

        indices_list = []

        num_elems = connectivity[keys[0]].shape[0]
        elem_indices = jnp.arange(num_elems, dtype=int)

        for field_i in keys:
            for field_j in keys:

                def one_elem_indices(elem_idx):
                    # Global DOFs for field_i
                    conn_i = connectivity[field_i][elem_idx]
                    if dofs[field_i].ndim == 1:
                        dofs_per_node_i = 1
                    else:
                        dofs_per_node_i = int(dofs[field_i].size // dofs[field_i].shape[0])
                    field_offset_i = field_offsets[field_i]
                    dof_local_i = jnp.arange(dofs_per_node_i, dtype=int)
                    dof_indices_i = (
                        field_offset_i + conn_i[:, None] * dofs_per_node_i + dof_local_i
                    )
                    global_dofs_i = jnp.asarray(dof_indices_i, dtype=int).flatten()

                    # Global DOFs for field_j
                    conn_j = connectivity[field_j][elem_idx]
                    if dofs[field_j].ndim == 1:
                        dofs_per_node_j = 1
                    else:
                        dofs_per_node_j = int(dofs[field_j].size // dofs[field_j].shape[0])

                    field_offset_j = field_offsets[field_j]
                    dof_local_j = jnp.arange(dofs_per_node_j, dtype=int)
                    dof_indices_j = (
                        field_offset_j + conn_j[:, None] * dofs_per_node_j + dof_local_j
                    )
                    global_dofs_j = jnp.asarray(dof_indices_j, dtype=int).flatten()

                    # Generate indices
                    row_indices = jnp.repeat(global_dofs_i, global_dofs_j.size)
                    col_indices = jnp.tile(global_dofs_j, global_dofs_i.size)
                    indices = jnp.stack([row_indices, col_indices], axis=-1)
                    return indices

                # Vectorize over elements
                all_elem_indices = vmap(one_elem_indices)(elem_indices)
                indices = all_elem_indices.reshape(-1, 2)
                indices_list.append(indices)

        # Concatenate all indices
        indices = jnp.concatenate(indices_list, axis=0)
        return indices
    else:
        # dofs is array
        if dofs.ndim == 1:
            dofs_per_node = 1
        else:
            dofs_per_node = dofs.shape[-1]

        def one_elem_idx(neighb):
            global_dofs = neighb[:, None] * dofs_per_node + jnp.arange(dofs_per_node)
            global_dofs = global_dofs.flatten()
            n_dofs_element = global_dofs.size

            row_indices = jnp.repeat(global_dofs, n_dofs_element)
            col_indices = jnp.tile(global_dofs, n_dofs_element)
            indices = jnp.stack([row_indices, col_indices], axis=-1)
            return indices.astype(int)

        all_elem_indices = vmap(one_elem_idx)(connectivity)
        indices = all_elem_indices.reshape(-1, 2)
        return indices

def _get_element_quantities(dofs, settings, static_settings, set):
    """
    Extracts element-dependent quantities for the specified set.

    Args:
        dofs (jnp.ndarray, dict or _q_fun_state): Degrees of freedom.
        settings (dict): Settings dictionary.
        static_settings (dict or flax.core.FrozenDict): Static settings dictionary.
        set (int): The domain number.

    Returns:
        tuple: (model_fun, local_dofs, local_node_coor, elem_numbers, connectivity)
    """
    model_fun = static_settings["model"][set]
    x_nodes = settings["node coordinates"]
    dofs_ref = _dofs_reference(dofs)
    dofs_is_dict = isinstance(dofs_ref, dict)

    # Warning if it was defined in static_settings
    assert "connectivity" not in static_settings, \
        "'connectivity' has been moved to 'settings' in order to reduce compile time. \
        Further, you should not transform it to a tuple of tuples anymore."

    connectivity = settings["connectivity"][set]

    if dofs_is_dict:
        # assert isinstance(
        #     x_nodes, dict
        # ), "If 'dofs' is a dict, 'settings['node coordinates']' must also be a dict."
        assert isinstance(
            connectivity, dict
        ), "If 'dofs' is a dict, 'settings['connectivity'][set]' must also be a dict."

        first_field = _block_field_keys(connectivity, dofs_ref)[0]
        elem_numbers = jnp.arange(connectivity[first_field].shape[0])
    else:
        elem_numbers = jnp.arange(connectivity.shape[0])

    return model_fun, x_nodes, elem_numbers, connectivity

def _get_element_quantities_2(dofs, settings, static_settings, set):
    dofs = _dofs_reference(dofs)
    assert isinstance(
        dofs, jnp.ndarray
    ), "Variational schemes do currently not support dofs as dicts."

    connectivity = settings["connectivity"][set]
    variational_scheme = static_settings["variational scheme"][set]
    x_int = settings["integration coordinates"][set]
    w_int = settings["integration weights"][set]
    int_point_numbers = jnp.arange(0, x_int.shape[0], 1)
    return connectivity, variational_scheme, x_int, w_int, int_point_numbers

def _extract_local_dofs_and_coor(dofs, node_list, x_nodes):
    if isinstance(dofs, _q_fun_state):

        def _local_field_values(key):
            q_tup = dofs.q_ts[key]
            local_q = q_tup[0].at[node_list[key]].get(wrap_negative_indices=False)
            local_q_derivs = tuple(
                q_deriv.at[node_list[key]].get(wrap_negative_indices=False)
                for q_deriv in q_tup[1:]
            )
            return local_q, local_q_derivs

        local_q_ts = {
            key: _local_field_values(key)
            for key in dofs.q_ts.keys() if key in node_list
        }

        def local_dofs(t):
            return {
                key: discrete_value_with_derivatives(t, local_q, local_q_derivs)
                for key, (local_q, local_q_derivs) in local_q_ts.items()
            }

    else:
        if isinstance(dofs, dict):
            local_dofs = {
                key: dofs[key].at[node_list[key]].get(wrap_negative_indices=False)
                for key in dofs.keys() if key in node_list
            }
        else:
            local_dofs = treemap(lambda x, y: x.at[y].get(wrap_negative_indices=False), dofs, node_list)

    if isinstance(node_list, dict):
        if 'physical coor' in node_list:
            if isinstance(x_nodes, dict):
                x_source = x_nodes['physical coor'] if 'physical coor' in x_nodes else x_nodes[next(iter(x_nodes.keys()))]
            else:
                x_source = x_nodes
            local_node_coor = x_source.at[node_list['physical coor']].get(wrap_negative_indices=False)
        else:
            if isinstance(x_nodes, dict):
                local_node_coor = {key: x_nodes[key].at[node_list[key]].get(wrap_negative_indices=False) for key in node_list.keys()}
            else:
                local_node_coor = {key: x_nodes.at[node_list[key]].get(wrap_negative_indices=False) for key in node_list.keys()}
    else:
        if isinstance(x_nodes, dict):
            x_source = x_nodes['physical coor'] if 'physical coor' in x_nodes else x_nodes[next(iter(x_nodes.keys()))]
            local_node_coor = x_source.at[node_list].get(wrap_negative_indices=False)
        else:
            local_node_coor = x_nodes.at[node_list].get(wrap_negative_indices=False)
    return local_dofs, local_node_coor

def _extract_local_dofs_and_coor_2(dofs, int_point_number, x_int, w_int, connectivity):
    x_i = x_int[int_point_number]
    w_i = w_int[int_point_number]
    dofs = _dofs_reference(dofs)
    local_dofs = treemap(lambda x, y: x.at[y].get(), dofs, connectivity[int_point_number])
    return x_i, w_i, local_dofs

def _get_tangent_diagonal(tangent_contributions, connectivity, dofs):
    """
    Assembles the diagonal entries of the tangent matrix from the tangent contributions,
    connectivity, and degrees of freedom.

    Args:
        tangent_contributions (dict or jnp.ndarray): Tangent contributions from the model function.
        connectivity (dict or jnp.ndarray): Connectivity information for elements.
        dofs (dict or jnp.ndarray): Degrees of freedom.

    Returns:
        jnp.ndarray: The assembled diagonal of the tangent matrix.
    """
    dofs = _dofs_reference(dofs)

    # Total number of DOFs
    num_dofs = _num_dofs(dofs)

    if isinstance(dofs, dict):
        # Field offsets are global; the block only carries its present fields.
        field_offsets = {}
        current_offset = 0
        for key in dofs.keys():
            field_offsets[key] = current_offset
            field_size = dofs[key].size  # Total DOFs in the field
            current_offset += field_size

        keys = _block_field_keys(connectivity, dofs)
        n_elems = connectivity[keys[0]].shape[0]  # Number of elements

        # Initialize the global diagonal vector
        diag = jnp.zeros(num_dofs)

        # Iterate over fields to assemble diagonal contributions
        for key in keys:
            # Extract tangent contributions for field [key][key]
            tc = tangent_contributions[key][key]
            # tc has shape: (n_elems, nodes_per_element_i, dofs_per_node_i, nodes_per_element_j, dofs_per_node_j)

            # Reshape tc to (n_elems, element_dofs, element_dofs)
            conn = connectivity[key]  # Shape: (n_elems, nodes_per_element)
            nodes_per_element = conn.shape[1]
            if dofs[key].ndim == 1:
                dofs_per_node = 1
            else:
                dofs_per_node = int(dofs[key].size // dofs[key].shape[0])
            element_dofs = nodes_per_element * dofs_per_node
            if element_dofs == 0:
                continue

            # Reshape tc
            tc = tc.reshape(n_elems, element_dofs, element_dofs)

            # Extract diagonal contributions
            diagonal_contributions = jnp.diagonal(
                tc, axis1=1, axis2=2
            )  # Shape: (n_elems, element_dofs)

            # Compute global DOF indices
            dof_local = jnp.arange(dofs_per_node)
            dof_indices = (
                field_offsets[key]
                + conn[:, :, None] * dofs_per_node
                + dof_local[None, None, :]
            )  # Shape: (n_elems, nodes_per_element, dofs_per_node)
            global_dofs = dof_indices.reshape(
                n_elems, element_dofs
            )  # Shape: (n_elems, element_dofs)

            # Flatten indices and values
            diag_indices = global_dofs.flatten().astype(int)
            diag_values = diagonal_contributions.flatten()

            # Sum into the global diagonal vector
            diag = diag.at[diag_indices].add(diag_values)

    else:
        # For the array case
        n_elems = connectivity.shape[0]
        conn = connectivity  # Shape: (n_elems, nodes_per_element)
        nodes_per_element = conn.shape[1]
        dofs_per_node = dofs.shape[-1]
        element_dofs = nodes_per_element * dofs_per_node

        # Reshape tangent_contributions
        tc = tangent_contributions.reshape(n_elems, element_dofs, element_dofs)

        # Extract diagonal contributions
        diagonal_contributions = jnp.diagonal(
            tc, axis1=1, axis2=2
        )  # Shape: (n_elems, element_dofs)

        # Compute global DOF indices
        dof_local = jnp.arange(dofs_per_node)
        dof_indices = (
            conn[:, :, None] * dofs_per_node + dof_local[None, None, :]
        )  # Shape: (n_elems, nodes_per_element, dofs_per_node)
        global_dofs = dof_indices.reshape(
            n_elems, element_dofs
        )  # Shape: (n_elems, element_dofs)

        # Flatten indices and values
        diag_indices = global_dofs.flatten().astype(int)
        diag_values = diagonal_contributions.flatten()

        # Sum into the global diagonal vector
        diag = jnp.zeros(num_dofs)
        diag = diag.at[diag_indices].add(diag_values)

    return diag

def _get_residual(residual_contributions, connectivity, dofs):
    """
    Assembles the global residual vector from the residual contributions,
    connectivity, and degrees of freedom, returning a residual with the same
    structure as dofs.

    Args:
        residual_contributions (dict or jnp.ndarray): Residual contributions from the model function.
        connectivity (dict or jnp.ndarray): Connectivity information for elements.
        dofs (dict or jnp.ndarray): Degrees of freedom.

    Returns:
        dict or jnp.ndarray: The assembled residual vector with the same structure as dofs.
    """
    dofs = _dofs_reference(dofs)

    if isinstance(dofs, dict):
        # Only the fields present in this block contribute a residual here; the
        # global accumulation (see assemble_residual) sums per field key.
        keys = _block_field_keys(connectivity, dofs)
        n_elems = connectivity[keys[0]].shape[0]

        # Initialize the residual dictionary
        residual = {}

        # Iterate over fields to assemble residual contributions
        for key in keys:
            # Extract residual contributions for field [key]
            rc = residual_contributions[
                key
            ]  # Shape: (n_elems, nodes_per_element, dofs_per_node)

            # Reshape rc to (n_elems, element_dofs)
            conn = connectivity[key]  # Shape: (n_elems, nodes_per_element)
            nodes_per_element = conn.shape[1]
            if dofs[key].ndim == 1:
                dofs_per_node = 1
            else:
                dofs_per_node = int(dofs[key].size // dofs[key].shape[0])
            element_dofs = nodes_per_element * dofs_per_node
            if element_dofs == 0:
                residual[key] = jnp.zeros_like(dofs[key])
                continue

            # Reshape rc
            rc = rc.reshape(n_elems, element_dofs)  # Shape: (n_elems, element_dofs)

            # Compute global DOF indices
            dof_local = jnp.arange(dofs_per_node)
            dof_indices = (
                conn[:, :, None] * dofs_per_node + dof_local[None, None, :]
            )  # Shape: (n_elems, nodes_per_element, dofs_per_node)
            global_dofs = dof_indices.reshape(
                n_elems, element_dofs
            )  # Shape: (n_elems, element_dofs)

            # Flatten indices and values
            residual_indices = global_dofs.flatten().astype(int)
            residual_values = rc.flatten()

            # Initialize the residual array for this field
            field_residual = jnp.zeros_like(dofs[key]).flatten()

            # Sum into the field residual vector
            field_residual = field_residual.at[residual_indices].add(residual_values)

            # Reshape back to the original shape
            field_residual = field_residual.reshape(dofs[key].shape)

            # Assign to the residual dictionary
            residual[key] = field_residual

        return residual

    else:
        # For the array case
        n_elems = connectivity.shape[0]
        conn = connectivity  # Shape: (n_elems, nodes_per_element)
        nodes_per_element = conn.shape[1]
        dofs_per_node = dofs.shape[-1]
        element_dofs = nodes_per_element * dofs_per_node

        # Reshape residual_contributions
        rc = residual_contributions.reshape(
            n_elems, element_dofs
        )  # Shape: (n_elems, element_dofs)

        # Compute global DOF indices
        dof_local = jnp.arange(dofs_per_node)
        dof_indices = (
            conn[:, :, None] * dofs_per_node + dof_local[None, None, :]
        )  # Shape: (n_elems, nodes_per_element, dofs_per_node)
        global_dofs = dof_indices.reshape(
            n_elems, element_dofs
        )  # Shape: (n_elems, element_dofs)

        # Flatten indices and values
        residual_indices = global_dofs.flatten().astype(int)
        residual_values = rc.flatten()

        # Initialize the residual array
        residual = jnp.zeros_like(dofs).flatten()

        # Sum into the residual vector
        residual = residual.at[residual_indices].add(residual_values)

        # Reshape back to the original shape
        residual = residual.reshape(dofs.shape)

        return residual

def _get_num_local_dofs(connectivity, dofs):
    """
    Determine the number of local degrees of freedom per element.

    This is used by `_batched_map` to estimate a memory-aware batch size.
    The result is a Python int and is derived purely from static shape
    information of `connectivity` and `dofs`.

    Args:
        connectivity (jnp.ndarray or dict): Element connectivity. For dict DOFs,
            this must contain one connectivity array per DOF field.
        dofs (jnp.ndarray, dict or _q_fun_state): Global degrees of freedom.

    Returns:
        int: Number of local DOFs on one element.
    """
    dofs = _dofs_reference(dofs)

    if isinstance(dofs, dict):
        assert isinstance(
            connectivity, dict
        ), "If 'dofs' is a dict, 'connectivity' must also be a dict."

        num_local_dofs = 0

        for key in _block_field_keys(connectivity, dofs):
            field_dofs = dofs[key]
            conn = connectivity[key]

            # Expected shape: (num_elems, nodes_per_element)
            nodes_per_element = int(conn.shape[1])

            if field_dofs.ndim == 1:
                dofs_per_node = 1
            else:
                dofs_per_node = int(field_dofs.size // field_dofs.shape[0])

            num_local_dofs += nodes_per_element * dofs_per_node

        return int(num_local_dofs)

    else:
        # Expected shape: (num_elems, nodes_per_element)
        nodes_per_element = int(connectivity.shape[1])

        if dofs.ndim == 1:
            dofs_per_node = 1
        else:
            dofs_per_node = int(dofs.size // dofs.shape[0])
            # Equivalent for usual shape (num_nodes, n_components):
            # dofs_per_node = int(dofs.shape[-1])

        return int(nodes_per_element * dofs_per_node)

# TODO: check performance for different element orders and numbers for residual and tangent assembly
def _batched_map(fun, elem_numbers, connectivity, num_local_dofs):
    # In order to save memory map over batches
    body_fun = lambda i: fun(elem_numbers[i], jax.tree.map(lambda x: x[i], connectivity))
    # num_local_dofs = jax.eval_shape(lambda i: dict_flatten(body_fun(i)), 0).shape[0]
    if any(0 in getattr(leaf, "shape", ()) for leaf in jax.tree_util.tree_leaves(connectivity)):
        return vmap(fun, (0, 0), (0))(elem_numbers, connectivity)
    num_local_dofs = max(int(num_local_dofs), 1)
    batch_size = int(2097152/num_local_dofs)
    batch_size = 2 ** int(np.ceil(np.log2(batch_size)))
    batch_size = min(elem_numbers.shape[0], batch_size)
    # print("batch size: ", batch_size)
    return jax.lax.map(body_fun, jnp.arange(elem_numbers.shape[0]), batch_size=batch_size)

### General assembling functions
@jit_with_docstring(static_argnames=["static_settings"])
def integrate_functional(dofs, settings, static_settings):
    """
    Integrate functional as sum over set of domains.

    Args:
      dofs (jnp.ndarray or dict): Degrees of freedom.
      settings (dict): Settings dictionary.
      static_settings (flax.core.FrozenDict): Static settings as frozen dictionary.

    Returns:
      float: The integrated functional value for current dofs.
    """
    # Loop over all sets of integration points/ domains
    num_sets = len(static_settings["assembling mode"])
    integrated_functional = 0
    for set in range(num_sets):
        assembling_mode = static_settings["assembling mode"][set]

        if assembling_mode == "dense":
            integrated_functional += dense_integrate_functional(
                dofs, settings, static_settings, set
            )
        elif assembling_mode == "sparse":
            integrated_functional += sparse_integrate_functional(
                dofs, settings, static_settings, set
            )
        elif assembling_mode == "user potential":
            integrated_functional += user_potential_integrate_functional(
                dofs, settings, static_settings, set
            )
        else:
            assert (
                False
            ), "Assembling mode can be either 'sparse' or 'dense' in integrate_functional"

    return integrated_functional

@jit_with_docstring(static_argnames=["static_settings"])
def assemble_residual(dofs, settings, static_settings):
    """
    Assemble residuals over set of domains.

    Args:
      dofs (jnp.ndarray, dict or _q_fun_state): Degrees of freedom.
      settings (dict): Settings dictionary.
      static_settings (flax.core.FrozenDict): Static settings as frozen dictionary.

    Returns:
      jnp.ndarray: The assembled residual.
    """
    # Loop over all sets of integration points/ domains
    num_sets = len(static_settings["assembling mode"])
    dofs_ref = _dofs_reference(dofs)

    if isinstance(dofs_ref, dict):
        assert isinstance(settings['connectivity'][0], dict), \
                    "If the DOFs are a dict, the connectivity, dirichlet dofs, and dirichlet conditions must also be dicts."
        # assert all([isinstance(settings['connectivity'][0], dict),
        #             isinstance(settings['node coordinates'], dict)]), \
        #             "If the DOFs are a dict, the connectivity, node coordinates, dirichlet dofs, and dirichlet conditions must also be dicts."
        
    integrated_residual = dict_zeros_like(dofs_ref)
    for set in range(num_sets):
        assembling_mode = static_settings["assembling mode"][set]

        if assembling_mode == "dense":
            add = dense_assemble_residual(dofs, settings, static_settings, set)
        elif assembling_mode == "sparse":
            add = sparse_assemble_residual(dofs, settings, static_settings, set)
        elif assembling_mode == "user potential":
            add = user_potential_assemble_residual(dofs, settings, static_settings, set)
        elif assembling_mode == "user residual":
            add = user_residual_assemble_residual(dofs, settings, static_settings, set)
        elif assembling_mode == "user element":
            add = user_element_assemble_residual(dofs, settings, static_settings, set)
        else:
            assert (
                False
            ), "Assembling mode can be either 'sparse', 'dense' or 'user element'"

        # Handle both cases dict and jnp.ndarray
        if isinstance(add, dict):
            for key in add.keys():
                integrated_residual[key] = integrated_residual[key] + add[key]
        else:
            integrated_residual += add

    return integrated_residual

@jit_with_docstring(static_argnames=["static_settings"])
def assemble_tangent_diagonal(dofs, settings, static_settings):
    """
    Assemble the diagonal of the tangent matrix.

    Args:
      dofs (jnp.ndarray or dict): Degrees of freedom.
      settings (dict): Settings dictionary.
      static_settings (flax.core.FrozenDict): Static settings as frozen dictionary.

    Returns:
      jnp.ndarray: The diagonal of the assembled tangent matrix.
    """
    # Loop over all sets of integration points/ domains
    num_sets = len(static_settings["assembling mode"])
    tangent_diagonal = jnp.zeros_like(dict_flatten(_dofs_reference(dofs)))
    for set in range(num_sets):
        assembling_mode = static_settings["assembling mode"][set]
        if assembling_mode == "sparse":
            tangent_diagonal += sparse_assemble_tangent_diagonal(
                dofs, settings, static_settings, set
            )
        elif assembling_mode == "user potential":
            tangent_diagonal += user_potential_assemble_tangent_diagonal(
                dofs, settings, static_settings, set
            )
        elif assembling_mode == "user residual":
            tangent_diagonal += user_residual_assemble_tangent_diagonal(
                dofs, settings, static_settings, set
            )
        elif assembling_mode == "user element":
            tangent_diagonal += user_element_assemble_tangent_diagonal(
                dofs, settings, static_settings, set
            )
        else:
            assert (
                False
            ), "Assembling mode for assembling tangent diagonal supports currently only 'sparse' and 'user element'"
    return tangent_diagonal

@jit_with_docstring(static_argnames=["static_settings"])
def assemble_tangent(dofs, settings, static_settings):
    """
    Assemble the full (possibly sparse) tangent matrix.

    Args:
      dofs (jnp.ndarray or dict): Degrees of freedom.
      settings (dict): Settings dictionary.
      static_settings (flax.core.FrozenDict): Static settings as frozen dictionary.

    Returns:
      jnp.ndarray or sparse matrix: The assembled tangent matrix.
    """
    # ToDo: add symmetric mode

    num_sets = len(static_settings["assembling mode"])
    one_dense = "dense" in static_settings["assembling mode"]
    dofs_ref = _dofs_reference(dofs)

    if isinstance(dofs_ref, dict):
        num_dofs = sum(v.size for v in dofs_ref.values())
        float_dtype = list(dofs_ref.values())[0].dtype
    else:
        num_dofs = dofs_ref.size
        float_dtype = dofs_ref.dtype
    try:
        sparsity_pattern = static_settings["known sparsity pattern"]
    except KeyError:
        sparsity_pattern = "none"

    match sparsity_pattern:
        case "none":
            if one_dense:
                integrated_tangent = jnp.zeros((num_dofs, num_dofs), dtype=float_dtype)
            else:
                integrated_tangent = sparse.empty(
                    (num_dofs, num_dofs), dtype=float_dtype, index_dtype=int
                )

            # Loop over all sets of integration points/ domains
            for set in range(num_sets):
                assembling_mode = static_settings["assembling mode"][set]

                if assembling_mode == "dense":
                    integrated_tangent += dense_assemble_tangent(
                        dofs, settings, static_settings, set
                    )
                else:
                    if assembling_mode == "sparse":
                        add = sparse_assemble_tangent(
                            dofs, settings, static_settings, set
                        )
                    elif assembling_mode == "user potential":
                        add = user_potential_assemble_tangent(
                            dofs, settings, static_settings, set
                        )
                    elif assembling_mode == "user residual":
                        add = user_residual_assemble_tangent(
                            dofs, settings, static_settings, set
                        )
                    elif assembling_mode == "user element":
                        add = user_element_assemble_tangent(
                            dofs, settings, static_settings, set
                        )
                    else:
                        assert (
                            False
                        ), "Assembling mode can be either 'sparse', 'dense' or 'user element'"

                    if one_dense:
                        integrated_tangent += add.todense()
                    else:
                        integrated_tangent += add
        case "diagonal":
            # # Compute the diagonal tangent with sparsejac (not the diagonal of a tangent that is not diagonal)
            # residual_fun = lambda flat_dofs: assemble_residual(flat_dofs.reshape(dofs.shape), settings, static_settings).flatten()
            # with jax.ensure_compile_time_eval():
            #   data_and_indices = (jnp.ones((num_dofs,)),
            #                       vmap(lambda i: jnp.asarray([i, i]))(jnp.arange(0, num_dofs)))
            #   mat_shape = (num_dofs,num_dofs)
            #   sparsity = sparse.BCOO(data_and_indices, shape=mat_shape)
            #   sparse_diag_fun = sparsejac.jacfwd(residual_fun, sparsity=sparsity)
            # diag = sparse_diag_fun(dofs.flatten())
            # return diag

            diag = dict_flatten(
                assemble_tangent_diagonal(dofs, settings, static_settings)
            )
            indices = vmap(lambda i: jnp.asarray([i, i]))(jnp.arange(0, num_dofs))
            data_and_indices = (diag, indices)
            matrix_shape = (num_dofs, num_dofs)
            diag_mat = sparse.BCOO(data_and_indices, shape=matrix_shape)
            return diag_mat

        case _:
            assert False, "'known sparsity pattern' mode is not implemented."

    return integrated_tangent

### Initialization of template for deleting duplicates in the tangent
def _build_assembling_template_from_indices(indices, shape, settings, static_settings):
    """
    Build the assembling template from sparse matrix indices with duplicate entries.
    """
    n_rows, n_cols = shape

    if indices.size == 0:
        assembling_template = AssemblingTemplate(
            nnz=0,
            col_sorted=jnp.zeros((0,), dtype=int),
            indptr=jnp.zeros((int(n_rows) + 1,), dtype=int),
            scatter=jnp.zeros((0,), dtype=int),
            indices_unique=jnp.zeros((0, 2), dtype=int),
        )
    else:
        def _np_sort_key_val(keys, raw_pos):
            perm = np.argsort(np.asarray(keys), kind="stable")
            return np.asarray(keys)[perm], np.asarray(raw_pos)[perm]
        def sort_key_val_with_cpu_fallback(keys, raw_pos):
            if jax.default_backend() != "cpu":
                return jax.lax.sort_key_val(keys, raw_pos, is_stable=False)
            out = (
                jax.ShapeDtypeStruct(keys.shape, keys.dtype),
                jax.ShapeDtypeStruct(raw_pos.shape, raw_pos.dtype),
            )
            return jax.pure_callback(_np_sort_key_val, out, keys, raw_pos)

        @partial(jax.jit, static_argnames=("n_rows", "n_cols"))
        def _assemble_stage1(indices, *, n_rows: int, n_cols: int):
            n = indices.shape[0]

            max_key = np.int64(n_rows - 1) * np.int64(n_cols) + np.int64(n_cols - 1)
            key_dtype = jnp.uint32 if max_key <= np.iinfo(np.uint32).max else jnp.uint64

            row = indices[:, 0].astype(key_dtype)
            col = indices[:, 1].astype(key_dtype)
            keys = row * jnp.asarray(n_cols, dtype=key_dtype) + col

            raw_pos = jnp.arange(n, dtype=int)

            # keys_sorted, perm = jax.lax.sort_key_val(keys, raw_pos, is_stable=False)
            keys_sorted, perm = sort_key_val_with_cpu_fallback(keys, raw_pos)

            is_new = jnp.concatenate([
                jnp.array([True], dtype=bool),
                keys_sorted[1:] != keys_sorted[:-1],
            ])
            scatter_sorted = jnp.cumsum(is_new.astype(int)) - 1

            scatter = jnp.empty(n, dtype=int)
            scatter = scatter.at[perm].set(scatter_sorted)

            return keys_sorted, is_new, scatter

        @partial(jax.jit, static_argnames=("n_rows", "n_cols"))
        def _assemble_stage2(unique_keys, *, n_rows: int, n_cols: int):
            key_dtype = unique_keys.dtype
            n_cols_t = jnp.asarray(n_cols, dtype=key_dtype)

            row_unique = (unique_keys // n_cols_t).astype(int)
            col_unique = (unique_keys %  n_cols_t).astype(int)

            counts_row = jnp.bincount(row_unique, length=n_rows).astype(int)

            indptr = jnp.empty(n_rows + 1, dtype=int)
            indptr = indptr.at[0].set(0)
            indptr = indptr.at[1:].set(jnp.cumsum(counts_row))

            indices_unique = jnp.stack([row_unique, col_unique], axis=1)

            return col_unique, indptr, indices_unique

        n_cols = int(n_cols)
        keys_sorted, is_new, scatter = _assemble_stage1(
            indices,
            n_rows=n_rows,
            n_cols=n_cols,
        )
        unique_keys = keys_sorted[is_new]
        col_unique, indptr, indices_unique = _assemble_stage2(
            unique_keys,
            n_rows=n_rows,
            n_cols=n_cols,
        )
        assembling_template = AssemblingTemplate(
            nnz=int(unique_keys.shape[0]),
            col_sorted=col_unique,
            indptr=indptr,
            scatter=scatter,
            indices_unique=indices_unique,
        )

    settings["assembling template"] = assembling_template
    static_settings = static_settings.copy(
        add_or_replace={"assembling kernel": _make_assembling_kernel(assembling_template.nnz)}
    )
    return settings, static_settings

def _build_assembling_template_from_connectivity(initial_guess, settings, static_settings):
    """
    Build the sparse assembly template from connectivity only.

    The template depends on the global DOF layout and element connectivity, not on
    model values. Avoiding tangent assembly here saves an expensive prepare-time
    trace/compile.
    """
    connectivity = settings["connectivity"]
    if not isinstance(connectivity, tuple):
        connectivity = (connectivity,)
    indices = jnp.concatenate(
        [_get_indices(conn, initial_guess) for conn in connectivity],
        axis=0,
    )
    num_dofs = sum(v.size for v in initial_guess.values()) if isinstance(initial_guess, dict) else initial_guess.size
    return _build_assembling_template_from_indices(
        indices,
        (num_dofs, num_dofs),
        settings,
        static_settings,
    )

def build_assembling_template(initial_guess, settings, static_settings):
    """
    Builds a template for summing duplicate entries directly from connectivity
    and stores the result in `settings` and `static_settings`.

    Input
    -----
    initial_guess : dict
        Initial guess for the DOFs.
    settings : dict
        Runtime settings dictionary. Will be returned with an added entry
        "assembling template".
    static_settings : flax.core.FrozenDict
        Static settings dictionary. Will be returned with an added entry
        "assembling kernel".

    Returns
    -------
    settings : dict
        Updated settings including the assembling template.
    static_settings : flax.core.FrozenDict
        Updated static_settings including the assembling kernel.
    """

    return _build_assembling_template_from_connectivity(initial_guess, settings, static_settings)


### Dense assembling
@jit_with_docstring(static_argnames=["static_settings", "set"])
def dense_integrate_functional(dofs, settings, static_settings, set):
    """
    Dense integration of functional of specified domain.

    Args:
      dofs (jnp.ndarray or dict): Degrees of freedom.
      settings (dict): Settings dictionary.
      static_settings (flax.core.FrozenDict): Static settings as frozen dictionary.
      set (int): The domain number.

    Returns:
      float: The integrated functional value.
    """
    x_int = settings["integration coordinates"][set]
    w_int = settings["integration weights"][set]
    int_point_numbers = jnp.arange(0, x_int.shape[0], 1)

    def at_int_point(int_point_number):
        x_i = x_int[int_point_number]
        w_i = w_int[int_point_number]
        return variational_schemes.functional_at_int_point(x_i, w_i, int_point_number, dofs, settings, static_settings, set)

    functional_at_int_point_vj = vmap(at_int_point, (0,))
    integrated_functional = functional_at_int_point_vj(int_point_numbers).sum()

    return integrated_functional

@jit_with_docstring(static_argnames=["static_settings", "set"])
def dense_assemble_residual(dofs, settings, static_settings, set):
    """
    Dense assembly of residual of specified domain.

    Args:
      dofs (jnp.ndarray or dict): Degrees of freedom.
      settings (dict): Settings dictionary.
      static_settings (flax.core.FrozenDict): Static settings as frozen dictionary.
      set (int): The domain number.

    Returns:
      jnp.ndarray: The assembled residual.
    """
    return jacrev(dense_integrate_functional)(dofs, settings, static_settings, set)

@jit_with_docstring(static_argnames=["static_settings", "set"])
def dense_assemble_tangent(dofs, settings, static_settings, set):
    """
    Dense assembly of tangent of specified domain.

    Args:
      dofs (jnp.ndarray or dict): Degrees of freedom.
      settings (dict): Settings dictionary.
      static_settings (flax.core.FrozenDict): Static settings as frozen dictionary.
      set (int): The domain number.

    Returns:
      jnp.ndarray: The assembled tangent matrix.
    """
    assert isinstance(
        dofs, jnp.ndarray
    ), "Dense mode of tangent assembly does currently not support dofs as dicts."

    size = dict_flatten(dofs).size
    tangent = hessian(dense_integrate_functional)(dofs, settings, static_settings, set)
    return dict_flatten(tangent).reshape((size, size))


### Sparse assembling
@jit_with_docstring(static_argnames=["static_settings", "set"])
def sparse_integrate_functional(dofs, settings, static_settings, set):
    """
    Sparse integration of functional of specified domain.

    Args:
      dofs (jnp.ndarray or dict): Degrees of freedom.
      settings (dict): Settings dictionary.
      static_settings (flax.core.FrozenDict): Static settings as frozen dictionary.
      set (int): The domain number.

    Returns:
      float: The integrated functional value.
    """
    connectivity, variational_scheme, x_int, w_int, int_point_numbers = _get_element_quantities_2(dofs, settings, static_settings, set)

    def func_at_int_pt(int_point_number):
        x_i, w_i, local_dofs = _extract_local_dofs_and_coor_2(dofs, int_point_number, x_int, w_int, connectivity)
        return variational_schemes.functional_at_int_point(
            x_i, w_i, int_point_number, local_dofs, settings, static_settings, set
        )

    functional_at_int_point_vj = vmap(func_at_int_pt, (0,))
    return functional_at_int_point_vj(int_point_numbers).sum()

@jit_with_docstring(static_argnames=["static_settings", "set"])
def sparse_assemble_residual(dofs, settings, static_settings, set):
    """
    Sparse assembly of residual of specified domain.

    Args:
      dofs (jnp.ndarray or dict): Degrees of freedom.
      settings (dict): Settings dictionary.
      static_settings (flax.core.FrozenDict): Static settings as frozen dictionary.
      set (int): The domain number.

    Returns:
      jnp.ndarray: The assembled residual.
    """
    connectivity, variational_scheme, x_int, w_int, int_point_numbers = _get_element_quantities_2(dofs, settings, static_settings, set)

    if (
        variational_scheme == "least square pde loss"
        or variational_scheme == "least square function approximation"
    ):
        def func_at_int_pt(int_point_number):
            x_i, w_i, local_dofs = _extract_local_dofs_and_coor_2(dofs, int_point_number, x_int, w_int, connectivity)
            return jacrev(variational_schemes.functional_at_int_point, argnums=3)(
                x_i, w_i, int_point_number, local_dofs, settings, static_settings, set
            )
        residual_at_int_point_vj = vmap(func_at_int_pt, (0,))
        residual_contributions = residual_at_int_point_vj(int_point_numbers)

    elif variational_scheme == "strong form galerkin":
        # Direct implementation of residual, e.g. for Galerkin method
        def func_at_int_pt(int_point_number):
            x_i, w_i, local_dofs = _extract_local_dofs_and_coor_2(dofs, int_point_number, x_int, w_int, connectivity)
            return variational_schemes.direct_residual_at_int_point(
                x_i, w_i, int_point_number, local_dofs, settings, static_settings, set
            )
        residual_at_int_point_vj = vmap(func_at_int_pt, (0,))
        residual_contributions = residual_at_int_point_vj(int_point_numbers)

    elif variational_scheme == "weak form galerkin":
        # Pass local_dofs twice (assuming Bubnov Galerkin...)
        def func_at_int_pt(int_point_number):
            x_i, w_i, local_dofs = _extract_local_dofs_and_coor_2(dofs, int_point_number, x_int, w_int, connectivity)
            return variational_schemes.residual_from_deriv_at_int_point(
                x_i, w_i, int_point_number, local_dofs, local_dofs, settings, static_settings, set
            )
        residual_at_int_point_vj = vmap(func_at_int_pt, (0,))
        residual_contributions = residual_at_int_point_vj(int_point_numbers)

    else:
        raise KeyError("Variational scheme not or wrongly specified!")
    return _get_residual(residual_contributions, connectivity, dofs)

@jit_with_docstring(static_argnames=["static_settings", "set"])
def sparse_assemble_tangent_diagonal(dofs, settings, static_settings, set):
    """
    Sparse assembly of the diagonal of the tangent matrix for specified set.

    Args:
      dofs (jnp.ndarray or dict): Degrees of freedom.
      settings (dict): Settings dictionary.
      static_settings (flax.core.FrozenDict): Static settings as frozen dictionary.
      set (int): The domain number.

    Returns:
      jnp.ndarray: The diagonal of the assembled tangent matrix.
    """
    connectivity, variational_scheme, x_int, w_int, int_point_numbers = _get_element_quantities_2(dofs, settings, static_settings, set)

    # Compute tangent contributions
    if (
        variational_scheme == "least square pde loss"
        or variational_scheme == "least square function approximation"
    ):
        def func_at_int_pt(int_point_number):
            x_i, w_i, local_dofs = _extract_local_dofs_and_coor_2(dofs, int_point_number, x_int, w_int, connectivity)
            return jacfwd(jacrev(variational_schemes.functional_at_int_point, argnums=3), argnums=3)(
                x_i, w_i, int_point_number, local_dofs, settings, static_settings, set
            )
        at_int_point_vj = vmap(func_at_int_pt, (0,))
        tangent_contributions = at_int_point_vj(int_point_numbers)

    elif variational_scheme == "strong form galerkin":
        def func_at_int_pt(int_point_number):
            x_i, w_i, local_dofs = _extract_local_dofs_and_coor_2(dofs, int_point_number, x_int, w_int, connectivity)
            return jacfwd(variational_schemes.direct_residual_at_int_point, argnums=3)(
                x_i, w_i, int_point_number, local_dofs, settings, static_settings, set
            )
        at_int_point_vj = vmap(func_at_int_pt, (0,))
        tangent_contributions = at_int_point_vj(int_point_numbers)

    elif variational_scheme == "weak form galerkin":
        def func_at_int_pt(int_point_number):
            x_i, w_i, local_dofs = _extract_local_dofs_and_coor_2(dofs, int_point_number, x_int, w_int, connectivity)
            return jacfwd(variational_schemes.residual_from_deriv_at_int_point, argnums=3)(
                x_i, w_i, int_point_number, local_dofs, local_dofs, settings, static_settings, set
            )
        at_int_point_vj = vmap(func_at_int_pt, (0,))
        tangent_contributions = at_int_point_vj(int_point_numbers)

    else:
        raise KeyError("Variational scheme mode not or wrongly specified!")

    return _get_tangent_diagonal(tangent_contributions, connectivity, dofs)

@jit_with_docstring(static_argnames=["static_settings", "set"])
def sparse_assemble_tangent(dofs, settings, static_settings, set):
    """
    Sparse assembly of the full tangent matrix of specified domain.

    Args:
      dofs (jnp.ndarray or dict): Degrees of freedom.
      settings (dict): Settings dictionary.
      static_settings (flax.core.FrozenDict): Static settings as frozen dictionary.
      set (int): The domain number.

    Returns:
      jax.experimental.sparse.BCOO: The assembled tangent matrix.
    """
    connectivity, variational_scheme, x_int, w_int, int_point_numbers = _get_element_quantities_2(dofs, settings, static_settings, set)

    # Compute tangent contributions
    if (
        variational_scheme == "least square pde loss"
        or variational_scheme == "least square function approximation"
    ):
        def func_at_int_pt(int_point_number):
            x_i, w_i, local_dofs = _extract_local_dofs_and_coor_2(dofs, int_point_number, x_int, w_int, connectivity)
            return jacfwd(jacrev(variational_schemes.functional_at_int_point, argnums=3), argnums=3)(
                x_i, w_i, int_point_number, local_dofs, settings, static_settings, set
            )
        at_int_point_vj = vmap(func_at_int_pt, (0,))
        tangent_contributions = at_int_point_vj(int_point_numbers)

    elif variational_scheme == "strong form galerkin":
        def func_at_int_pt(int_point_number):
            x_i, w_i, local_dofs = _extract_local_dofs_and_coor_2(dofs, int_point_number, x_int, w_int, connectivity)
            return jacfwd(variational_schemes.direct_residual_at_int_point, argnums=3)(
                x_i, w_i, int_point_number, local_dofs, settings, static_settings, set
            )
        at_int_point_vj = vmap(func_at_int_pt, (0,))
        tangent_contributions = at_int_point_vj(int_point_numbers)

    elif variational_scheme == "weak form galerkin":
        def func_at_int_pt(int_point_number):
            x_i, w_i, local_dofs = _extract_local_dofs_and_coor_2(dofs, int_point_number, x_int, w_int, connectivity)
            return jacfwd(variational_schemes.residual_from_deriv_at_int_point, argnums=3)(
                x_i, w_i, int_point_number, local_dofs, local_dofs, settings, static_settings, set
            )
        at_int_point_vj = vmap(func_at_int_pt, (0,))
        tangent_contributions = at_int_point_vj(int_point_numbers)

    else:
        raise KeyError("Variational scheme mode not or wrongly specified!")

    # Assembling (without summing duplicates)
    data = dict_flatten(tangent_contributions)
    indices = _get_indices(connectivity, dofs)
    num_dofs = _num_dofs(dofs)
    tangent_matrix = sparse.BCOO((data, indices), shape=(num_dofs, num_dofs))
    return tangent_matrix


### Assembling for user potentials
@jit_with_docstring(static_argnames=["static_settings", "set"])
def user_potential_integrate_functional(dofs, settings, static_settings, set):
    """
    Assembly of potential for custom user definition of specified domain.

    Args:
      dofs (jnp.ndarray, dict or _q_fun_state): Degrees of freedom.
      settings (dict): Settings dictionary.
      static_settings (flax.core.FrozenDict): Static settings as frozen dictionary.
      set (int): The domain number.

    Returns:
      float: value of functional integrated over set of elements
    """
    model_fun, x_nodes, elem_numbers, connectivity = _get_element_quantities(dofs, settings, static_settings, set)

    # Modify the model_fun such that it extracts the DOFs from the global dofs and vmap only over connectivity
    def element_residual(elem_number, node_list):
        local_dofs, local_node_coor = _extract_local_dofs_and_coor(dofs, node_list, x_nodes)
        return  model_fun(local_dofs, local_node_coor, elem_number, settings, static_settings, set)

    functional_contributions = vmap(element_residual, (0, 0), (0))(elem_numbers, connectivity)

    return functional_contributions.sum()

@jit_with_docstring(static_argnames=["static_settings", "set"])
def user_potential_assemble_residual(dofs, settings, static_settings, set):
    """
    Assembly of residual for custom user potential of specified domain.

    Args:
      dofs (jnp.ndarray or dict): Degrees of freedom.
      settings (dict): Settings dictionary.
      static_settings (flax.core.FrozenDict): Static settings as frozen dictionary.
      set (int): The domain number.

    Returns:
      jnp.ndarray: The assembled residual.
    """
    # return jacrev(user_potential_integrate_functional)(dofs, settings, static_settings, set)

    model_fun, x_nodes, elem_numbers, connectivity = _get_element_quantities(dofs, settings, static_settings, set)

    # Modify the model_fun such that it extracts the DOFs from the global dofs and vmap only over connectivity
    def element_residual(elem_number, node_list):
        local_dofs, local_node_coor = _extract_local_dofs_and_coor(dofs, node_list, x_nodes)
        return  jacrev(model_fun)(local_dofs, local_node_coor, elem_number, settings, static_settings, set)

    no_local_dofs = _get_num_local_dofs(connectivity, dofs)

    # residual_contributions = vmap(element_residual, (0, 0))(elem_numbers, connectivity)
    residual_contributions = _batched_map(element_residual, elem_numbers, connectivity, no_local_dofs)

    # return residual_contributions
    return _get_residual(residual_contributions, connectivity, dofs)

@jit_with_docstring(static_argnames=["static_settings", "set"])
def user_potential_assemble_tangent_diagonal(dofs, settings, static_settings, set):
    """
    Assembly of the diagonal of the tangent matrix for custom user potential of specified domain.

    Args:
      dofs (jnp.ndarray or dict): Degrees of freedom.
      settings (dict): Settings dictionary.
      static_settings (flax.core.FrozenDict): Static settings as frozen dictionary.
      set (int): The domain number.

    Returns:
      jnp.ndarray: The diagonal of the assembled tangent matrix.
    """
    model_fun, x_nodes, elem_numbers, connectivity = _get_element_quantities(dofs, settings, static_settings, set)

    # Modify the model_fun such that it extracts the dofs from the global dofs and vmap only over connectivity
    def element_tangent(elem_number, node_list):
        local_dofs, local_node_coor = _extract_local_dofs_and_coor(dofs, node_list, x_nodes)
        return  jacfwd(jacrev(model_fun))(local_dofs, local_node_coor, elem_number, settings, static_settings, set)

    tangent_contributions = vmap(element_tangent, (0, 0), (0))(elem_numbers, connectivity)

    return _get_tangent_diagonal(tangent_contributions, connectivity, dofs)

@jit_with_docstring(static_argnames=["static_settings", "set"])
def user_potential_assemble_tangent(dofs, settings, static_settings, set):
    """
    Assembly of the full (sparse) tangent matrix for custom user potential of specified domain.

    Args:
      dofs (jnp.ndarray or dict): Degrees of freedom.
      settings (dict): Settings dictionary.
      static_settings (flax.core.FrozenDict): Static settings as frozen dictionary.
      set (int): The domain number.

    Returns:
      jax.experimental.sparse.BCOO: The assembled tangent matrix.
    """
    model_fun, x_nodes, elem_numbers, connectivity = _get_element_quantities(dofs, settings, static_settings, set)

    # Modify the model_fun such that it extracts the dofs from the global dofs and vmap only over connectivity
    def element_tangent(elem_number, node_list):
        local_dofs, local_node_coor = _extract_local_dofs_and_coor(dofs, node_list, x_nodes)
        return  jacfwd(jacrev(model_fun))(local_dofs, local_node_coor, elem_number, settings, static_settings, set)

    tangent_contributions = vmap(element_tangent, (0, 0), (0))(elem_numbers, connectivity)

    data = dict_flatten(tangent_contributions)
    indices = _get_indices(connectivity, dofs)
    num_dofs = _num_dofs(dofs)

    tangent_matrix = sparse.BCOO((data, indices), shape=(num_dofs, num_dofs))
    return tangent_matrix

@jit_with_docstring(static_argnames=["static_settings", "set"])
def _user_potential_assemble_r_and_t(dofs, settings, static_settings, set):
    model_fun, x_nodes, elem_numbers, connectivity = _get_element_quantities(dofs, settings, static_settings, set)

    # Modify the model_fun such that it extracts the dofs from the global dofs and vmap only over connectivity
    def element_r_and_t(elem_number, node_list):
        local_dofs, local_node_coor = _extract_local_dofs_and_coor(dofs, node_list, x_nodes)
        residual_fun = lambda x: jacrev(model_fun)(x, local_node_coor, elem_number, settings, static_settings, set)

        # residual_fun = jax.jit(residual_fun)
        # elem_res = residual_fun(local_dofs)
        # elem_tan = jacfwd(residual_fun)(local_dofs)

        # def residual_and_tangent_linearize(residual_fun, local_dofs):
        #     primals, lin_fun = linearize(residual_fun, local_dofs)            
        #     flat_local_dofs = dict_flatten(local_dofs)
        #     n = flat_local_dofs.shape[0]            
        #     identity = jnp.eye(n)            
        #     jacobian = vmap(lambda v: lin_fun(reshape_as(v, local_dofs)))(identity)            
        #     return primals, jacobian
        # elem_res, elem_tan = residual_and_tangent_linearize(residual_fun, local_dofs)

        def residual_and_tangent(residual_fun, local_dofs):
            flat_local_dofs = dict_flatten(local_dofs)
            n = flat_local_dofs.shape[0]
            identity = jnp.eye(n)
            def jvp_with_flat_tangent(v):
                tangent_pytree = reshape_as(v, local_dofs)
                return jvp(residual_fun, (local_dofs,), (tangent_pytree,))
            primals, elem_tan = vmap(jvp_with_flat_tangent)(identity)
            elem_res = treemap(lambda x: x[0], primals)
            return elem_res, elem_tan        
        elem_res, elem_tan = residual_and_tangent(residual_fun, local_dofs)

        # def residual_and_tangent_vjp(residual_fun, local_dofs):
        #     primals, vjp_fun = vjp(residual_fun, local_dofs)          
        #     flat_res = dict_flatten(primals)
        #     m = flat_res.shape[0]
        #     flat_local = dict_flatten(local_dofs)
        #     n = flat_local.shape[0]
        #     identity = jnp.eye(m)
        #     jacobian_rows = vmap(
        #         lambda v: dict_flatten(vjp_fun(reshape_as(v, primals))[0])
        #     )(identity)
        #     jacobian = jacobian_rows.T
        #     return primals, jacobian
        # elem_res, elem_tan = residual_and_tangent_vjp(residual_fun, local_dofs)

        return elem_res, elem_tan

    all_contributions = vmap(element_r_and_t, (0, 0), (0, 0))
    residual_contributions, tangent_contributions = all_contributions(elem_numbers, connectivity)

    residual = _get_residual(residual_contributions, connectivity, dofs)

    data = dict_flatten(tangent_contributions)
    indices = _get_indices(connectivity, dofs)
    num_dofs = _num_dofs(dofs)

    tangent_matrix = sparse.BCOO((data, indices), shape=(num_dofs, num_dofs))
    return residual, tangent_matrix


### Assembling for user residuals
@jit_with_docstring(static_argnames=["static_settings", "set"])
def user_residual_assemble_residual(dofs, settings, static_settings, set):
    """
    Assembly of residual for custom user residual of specified domain.

    Args:
      dofs (jnp.ndarray, dict or _q_fun_state): Degrees of freedom.
      settings (dict): Settings dictionary.
      static_settings (flax.core.FrozenDict): Static settings as frozen dictionary.
      set (int): The domain number.

    Returns:
      jnp.ndarray: The assembled residual.
    """
    model_fun, x_nodes, elem_numbers, connectivity = _get_element_quantities(dofs, settings, static_settings, set)

    # Modify the model_fun such that it extracts the DOFs from the global dofs and vmap only over connectivity
    def element_residual(elem_number, node_list):
        local_dofs, local_node_coor = _extract_local_dofs_and_coor(dofs, node_list, x_nodes)
        return  model_fun(local_dofs, local_node_coor, elem_number, settings, static_settings, set)
    
    no_local_dofs = _get_num_local_dofs(connectivity, dofs)

    # residual_contributions = vmap(element_residual, (0, 0), (0))(elem_numbers, connectivity)
    residual_contributions = _batched_map(element_residual, elem_numbers, connectivity, no_local_dofs)

    return _get_residual(residual_contributions, connectivity, dofs)

@jit_with_docstring(static_argnames=["static_settings", "set"])
def user_residual_assemble_tangent_diagonal(dofs, settings, static_settings, set):
    """
    Assembly of the diagonal of the tangent matrix for custom user residual of specified domain.

    Args:
      dofs (jnp.ndarray or dict): Degrees of freedom.
      settings (dict): Settings dictionary.
      static_settings (flax.core.FrozenDict): Static settings as frozen dictionary.
      set (int): The domain number.

    Returns:
      jnp.ndarray: The diagonal of the assembled tangent matrix.
    """
    model_fun, x_nodes, elem_numbers, connectivity = _get_element_quantities(dofs, settings, static_settings, set)

    # Modify the model_fun such that it extracts the DOFs from the global dofs and vmap only over connectivity
    def element_residual(elem_number, node_list):
        local_dofs, local_node_coor = _extract_local_dofs_and_coor(dofs, node_list, x_nodes)
        return  jacfwd(model_fun)(local_dofs, local_node_coor, elem_number, settings, static_settings, set)

    tangent_contributions = vmap(element_residual, (0, 0), (0))(elem_numbers, connectivity)

    return _get_tangent_diagonal(tangent_contributions, connectivity, dofs)

@jit_with_docstring(static_argnames=["static_settings", "set"])
def user_residual_assemble_tangent(dofs, settings, static_settings, set):
    """
    Assembly of the full (sparse) tangent matrix for custom user residual of specified domain.

    Args:
      dofs (jnp.ndarray or dict): Degrees of freedom.
      settings (dict): Settings dictionary.
      static_settings (flax.core.FrozenDict): Static settings as frozen dictionary.
      set (int): The domain number.

    Returns:
      jax.experimental.sparse.BCOO: The assembled tangent matrix.
    """
    model_fun, x_nodes, elem_numbers, connectivity = _get_element_quantities(dofs, settings, static_settings, set)

    # Modify the model_fun such that it extracts the DOFs from the global dofs and vmap only over connectivity
    def element_residual(elem_number, node_list):
        local_dofs, local_node_coor = _extract_local_dofs_and_coor(dofs, node_list, x_nodes)
        return  jacfwd(model_fun)(local_dofs, local_node_coor, elem_number, settings, static_settings, set)

    tangent_contributions = vmap(element_residual, (0, 0), (0))(elem_numbers, connectivity)

    data = dict_flatten(tangent_contributions)
    indices = _get_indices(connectivity, dofs)

    num_dofs = _num_dofs(dofs)
    tangent_matrix = sparse.BCOO((data, indices), shape=(num_dofs, num_dofs))
    return tangent_matrix


### Assembling for user elements
@jit_with_docstring(static_argnames=["static_settings", "set"])
def user_element_assemble_residual(dofs, settings, static_settings, set):
    """
    Assembly of residual for custom user element of specified domain.

    Args:
      dofs (jnp.ndarray or dict): Degrees of freedom.
      settings (dict): Settings dictionary.
      static_settings (flax.core.FrozenDict): Static settings as frozen dictionary.
      set (int): The domain number.

    Returns:
      jnp.ndarray: The assembled residual.
    """
    model_fun, x_nodes, elem_numbers, connectivity = _get_element_quantities(dofs, settings, static_settings, set)

    # Modify the model_fun such that it extracts the DOFs from the global dofs and vmap only over connectivity
    def element_residual(elem_number, node_list):
        local_dofs, local_node_coor = _extract_local_dofs_and_coor(dofs, node_list, x_nodes)
        return  model_fun(local_dofs, local_node_coor, elem_number, settings, static_settings, "residual", set)

    residual_contributions = vmap(element_residual, (0, 0), (0))(elem_numbers, connectivity)

    return _get_residual(residual_contributions, connectivity, dofs)

@jit_with_docstring(static_argnames=["static_settings", "set"])
def user_element_assemble_tangent_diagonal(dofs, settings, static_settings, set):
    """
    Assembly of the diagonal of the tangent matrix for custom user element of specified domain.

    Args:
      dofs (jnp.ndarray or dict): Degrees of freedom.
      settings (dict): Settings dictionary.
      static_settings (flax.core.FrozenDict): Static settings as frozen dictionary.
      set (int): The domain number.

    Returns:
      jnp.ndarray: The diagonal of the assembled tangent matrix.
    """
    model_fun, x_nodes, elem_numbers, connectivity = _get_element_quantities(dofs, settings, static_settings, set)

    # Modify the model_fun such that it extracts the DOFs from the global dofs and vmap only over connectivity
    def element_residual(elem_number, node_list):
        local_dofs, local_node_coor = _extract_local_dofs_and_coor(dofs, node_list, x_nodes)
        return  model_fun(local_dofs, local_node_coor, elem_number, settings, static_settings, "tangent", set)

    tangent_contributions = vmap(element_residual, (0, 0), (0))(elem_numbers, connectivity)

    return _get_tangent_diagonal(tangent_contributions, connectivity, dofs)

@jit_with_docstring(static_argnames=["static_settings", "set"])
def user_element_assemble_tangent(dofs, settings, static_settings, set):
    """
    Assembly of the full (sparse) tangent matrix for custom user element of specified domain.

    Args:
      dofs (jnp.ndarray or dict): Degrees of freedom.
      settings (dict): Settings dictionary.
      static_settings (flax.core.FrozenDict): Static settings as frozen dictionary.
      set (int): The domain number.

    Returns:
      jax.experimental.sparse.BCOO: The assembled tangent matrix.
    """
    model_fun, x_nodes, elem_numbers, connectivity = _get_element_quantities(dofs, settings, static_settings, set)

    # Modify the model_fun such that it extracts the DOFs from the global dofs and vmap only over connectivity
    def element_residual(elem_number, node_list):
        local_dofs, local_node_coor = _extract_local_dofs_and_coor(dofs, node_list, x_nodes)
        return  model_fun(local_dofs, local_node_coor, elem_number, settings, static_settings, "tangent", set)

    tangent_contributions = vmap(element_residual, (0, 0))(elem_numbers, connectivity)

    data = dict_flatten(tangent_contributions)
    indices = _get_indices(connectivity, dofs)
    num_dofs = _num_dofs(dofs)
    tangent_matrix = sparse.BCOO((data, indices), shape=(num_dofs, num_dofs))
    return tangent_matrix


### Internal variable update function
@jit_with_docstring(static_argnames=["static_settings", "set"])
def get_int_var_updates(dofs, settings, static_settings, set):
    """
    Get the internal variables for a specified domain for all elements and integration points.

    Similar Structure as the assemble_residual functions, but uses 'int var updates' instead of 'model' in order to compute the per element and Gauss point internal variables.
    """
    _, x_nodes, elem_numbers, connectivity = _get_element_quantities(dofs, settings, static_settings, set)
    int_var_updates = static_settings['int var updates'][set]
    def local_update_fun(elem_number, node_list):
        local_dofs, local_node_coor = _extract_local_dofs_and_coor(dofs, node_list, x_nodes)
        return  int_var_updates(local_dofs, local_node_coor, elem_number, settings, static_settings, set)
    return jax.vmap(local_update_fun, (0, 0), (0))(elem_numbers, connectivity)
