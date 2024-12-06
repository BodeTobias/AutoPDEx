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

import jax
import jax.numpy as jnp
from jax.experimental import sparse
from flax.core import FrozenDict

from autopdex import variational_schemes
from autopdex.utility import jit_with_docstring, dict_zeros_like, dict_flatten, reshape_as


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
    if isinstance(dofs, dict):
        field_keys = list(
            dofs.keys()
        )  # Verwenden Sie die natürliche Reihenfolge der Schlüssel

        # Feld-Offsets berechnen
        field_offsets = {}
        current_offset = 0
        for field in field_keys:
            field_offsets[field] = current_offset
            field_size = dofs[field].size  # Gesamtzahl der DOFs im Feld
            current_offset += field_size

        indices_list = []

        num_elems = connectivity[field_keys[0]].shape[0]
        elem_indices = jnp.arange(num_elems, dtype=int)

        # Für jedes Feldpaar in der gewünschten Reihenfolge Indizes generieren
        for field_i in field_keys:
            for field_j in field_keys:

                def one_elem_indices(elem_idx):
                    # Globale DOFs für field_i
                    conn_i = connectivity[field_i][elem_idx]
                    if dofs[field_i].ndim == 1:
                        dofs_per_node_i = 1
                    else:
                        dofs_per_node_i = dofs[field_i].shape[-1]
                    field_offset_i = field_offsets[field_i]
                    dof_local_i = jnp.arange(dofs_per_node_i, dtype=int)
                    dof_indices_i = (
                        field_offset_i + conn_i[:, None] * dofs_per_node_i + dof_local_i
                    )
                    # global_dofs_i = dof_indices_i.flatten()
                    global_dofs_i = jnp.asarray(dof_indices_i, dtype=int).flatten()

                    # Globale DOFs für field_j
                    conn_j = connectivity[field_j][elem_idx]
                    if dofs[field_i].ndim == 1:
                        dofs_per_node_j = 1
                    else:
                        dofs_per_node_j = dofs[field_j].shape[-1]
                    field_offset_j = field_offsets[field_j]
                    dof_local_j = jnp.arange(dofs_per_node_j, dtype=int)
                    dof_indices_j = (
                        field_offset_j + conn_j[:, None] * dofs_per_node_j + dof_local_j
                    )
                    # global_dofs_j = dof_indices_j.flatten()
                    global_dofs_j = jnp.asarray(dof_indices_j, dtype=int).flatten()

                    # Indizes generieren
                    row_indices = jnp.repeat(global_dofs_i, global_dofs_j.size)
                    col_indices = jnp.tile(global_dofs_j, global_dofs_i.size)
                    indices = jnp.stack([row_indices, col_indices], axis=-1)
                    return indices

                # Vektorisieren über Elemente
                all_elem_indices = jax.vmap(one_elem_indices)(elem_indices)
                indices = all_elem_indices.reshape(-1, 2)
                indices_list.append(indices)

        # Alle Indizes in der Reihenfolge zusammenfügen
        indices = jnp.concatenate(indices_list, axis=0)
        return indices
    else:
        # dofs ist ein Array
        if dofs.ndim == 1:
            dofs_per_node = 1
        else:
            dofs_per_node = dofs.shape[-1]

        def one_elem_idx(neighb):
            global_dofs = neighb[:, None] * dofs_per_node + jnp.arange(dofs_per_node)
            global_dofs = global_dofs.flatten()
            n_dofs_element = global_dofs.size

            # Indizes generieren
            row_indices = jnp.repeat(global_dofs, n_dofs_element)
            col_indices = jnp.tile(global_dofs, n_dofs_element)
            indices = jnp.stack([row_indices, col_indices], axis=-1)
            return indices

        all_elem_indices = jax.vmap(one_elem_idx)(connectivity)
        indices = all_elem_indices.reshape(-1, 2)
        return indices

def _get_element_quantities(dofs, settings, static_settings, set):
    """
    Extracts element-dependent quantities for the specified set.

    Args:
        dofs (jnp.ndarray or dict): Degrees of freedom.
        settings (dict): Settings dictionary.
        static_settings (dict or flax.core.FrozenDict): Static settings dictionary.
        set (int): The domain number.

    Returns:
        tuple: (model_fun, local_dofs, local_node_coor, elem_numbers, connectivity)
    """
    model_fun = static_settings["model"][set]
    x_nodes = settings["node coordinates"]

    if isinstance(dofs, dict):
        assert isinstance(
            x_nodes, dict
        ), "If 'dofs' is a dict, 'settings['node coordinates']' must also be a dict."

        # connectivity_dict = static_settings["connectivity"][set]
        connectivity_dict = settings["connectivity"][set] # Fixme
        assert isinstance(
            connectivity_dict, (dict, FrozenDict)
        ), "If 'dofs' is a dict, 'static_settings['connectivity'][set]' must be a FrozenDict."

        field_keys = list(dofs.keys())
        connectivity = {key: jnp.asarray(connectivity_dict[key]) for key in connectivity_dict}
        elem_numbers = jnp.arange(connectivity[field_keys[0]].shape[0])
        local_dofs = jax.tree.map(lambda x, y: x.at[y].get(), dofs, connectivity)
        local_node_coor = jax.tree.map(lambda x, y: x.at[y].get(), x_nodes, connectivity)

    else:
        # connectivity_list = jnp.asarray(static_settings["connectivity"][set])
        connectivity_list = jnp.asarray(settings["connectivity"][set]) # Fixme
        local_dofs = dofs[connectivity_list]
        local_node_coor = x_nodes[connectivity_list]
        elem_numbers = jnp.arange(connectivity_list.shape[0])
        connectivity = connectivity_list

    return model_fun, local_dofs, local_node_coor, elem_numbers, connectivity

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
    import jax.numpy as jnp

    # Total number of DOFs
    num_dofs = (
        dofs.size if not isinstance(dofs, dict) else sum(v.size for v in dofs.values())
    )

    if isinstance(dofs, dict):
        field_keys = list(dofs.keys())
        n_elems = next(iter(connectivity.values())).shape[0]  # Number of elements

        # Compute field offsets for global DOF numbering
        field_offsets = {}
        current_offset = 0
        for key in field_keys:
            field_offsets[key] = current_offset
            field_size = dofs[key].size  # Total DOFs in the field
            current_offset += field_size

        # Initialize the global diagonal vector
        diag = jnp.zeros(num_dofs)

        # Iterate over fields to assemble diagonal contributions
        for key in field_keys:
            # Extract tangent contributions for field [key][key]
            tc = tangent_contributions[key][key]
            # tc has shape: (n_elems, nodes_per_element_i, dofs_per_node_i, nodes_per_element_j, dofs_per_node_j)

            # Reshape tc to (n_elems, element_dofs, element_dofs)
            conn = connectivity[key]  # Shape: (n_elems, nodes_per_element)
            nodes_per_element = conn.shape[1]
            if dofs[key].ndim == 1:
                dofs_per_node = 1
            else:
                dofs_per_node = dofs[key].shape[-1]
            element_dofs = nodes_per_element * dofs_per_node

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
    if isinstance(dofs, dict):
        field_keys = list(dofs.keys())
        n_elems = connectivity[field_keys[0]].shape[0]
        
        # Initialize the residual dictionary
        residual = {}

        # Iterate over fields to assemble residual contributions
        for key in field_keys:
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
                dofs_per_node = dofs[key].shape[-1]
            element_dofs = nodes_per_element * dofs_per_node

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
      dofs (jnp.ndarray or dict): Degrees of freedom.
      settings (dict): Settings dictionary.
      static_settings (flax.core.FrozenDict): Static settings as frozen dictionary.

    Returns:
      jnp.ndarray: The assembled residual.
    """
    # Loop over all sets of integration points/ domains
    num_sets = len(static_settings["assembling mode"])
    integrated_residual = dict_zeros_like(dofs)
    # integrated_residual = jnp.zeros_like(dofs)
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
            integrated_residual = jax.tree.map(lambda x, y: x + y, integrated_residual, add)
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
    tangent_diagonal = jnp.zeros_like(dict_flatten(dofs))
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

    if isinstance(dofs, dict):
        num_dofs = sum(v.size for v in dofs.values())
    else:
        num_dofs = dofs.size

    try:
        sparsity_pattern = static_settings["known sparsity pattern"]
    except KeyError:
        sparsity_pattern = "none"

    match sparsity_pattern:
        case "none":
            if one_dense:
                integrated_tangent = jnp.zeros((num_dofs, num_dofs))
            else:
                integrated_tangent = sparse.empty(
                    (num_dofs, num_dofs), dtype=jnp.float64, index_dtype=jnp.int_
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
            #                       jax.vmap(lambda i: jnp.asarray([i, i]))(jnp.arange(0, num_dofs)))
            #   mat_shape = (num_dofs,num_dofs)
            #   sparsity = sparse.BCOO(data_and_indices, shape=mat_shape)
            #   sparse_diag_fun = sparsejac.jacfwd(residual_fun, sparsity=sparsity)
            # diag = sparse_diag_fun(dofs.flatten())
            # return diag

            diag = dict_flatten(
                assemble_tangent_diagonal(dofs, settings, static_settings)
            )
            indices = jax.vmap(lambda i: jnp.asarray([i, i]))(jnp.arange(0, num_dofs))
            data_and_indices = (diag, indices)
            matrix_shape = (num_dofs, num_dofs)
            diag_mat = sparse.BCOO(data_and_indices, shape=matrix_shape)
            return diag_mat

        case _:
            assert False, "'known sparsity pattern' mode is not implemented."

    return integrated_tangent


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

    functional_at_int_point_vj = jax.vmap(
        variational_schemes.functional_at_int_point,
        (0, 0, 0, None, None, None, None),
        0,
    )
    integrated_functional = functional_at_int_point_vj(
        x_int, w_int, int_point_numbers, dofs, settings, static_settings, set
    ).sum()

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
    return jax.jacrev(dense_integrate_functional)(dofs, settings, static_settings, set)


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
    hessian = jax.hessian(dense_integrate_functional)(
        dofs, settings, static_settings, set
    )
    return dict_flatten(hessian).reshape((size, size))


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
    assert isinstance(
        dofs, jnp.ndarray
    ), "Variational schemes do currently not support dofs as dicts."

    _, local_dofs, _, _, _ = _get_element_quantities(
        dofs, settings, static_settings, set
    )
    x_int = settings["integration coordinates"][set]
    w_int = settings["integration weights"][set]
    int_point_numbers = jnp.arange(0, x_int.shape[0], 1)

    functional_at_int_point_vj = jax.vmap(
        variational_schemes.functional_at_int_point, (0, 0, 0, 0, None, None, None), 0
    )
    return functional_at_int_point_vj(
        x_int, w_int, int_point_numbers, local_dofs, settings, static_settings, set
    ).sum()


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
    assert isinstance(
        dofs, jnp.ndarray
    ), "Variational schemes do currently not support dofs as dicts."

    _, local_dofs, _, _, connectivity = _get_element_quantities(
        dofs, settings, static_settings, set
    )
    variational_scheme = static_settings["variational scheme"][set]
    x_int = settings["integration coordinates"][set]
    w_int = settings["integration weights"][set]
    int_point_numbers = jnp.arange(0, x_int.shape[0], 1)

    if (
        variational_scheme == "least square pde loss"
        or variational_scheme == "least square function approximation"
    ):
        residual_at_int_point_vj = jax.vmap(
            jax.jacrev(variational_schemes.functional_at_int_point, argnums=3),
            (0, 0, 0, 0, None, None, None),
            0,
        )
        residual_contributions = residual_at_int_point_vj(
            x_int, w_int, int_point_numbers, local_dofs, settings, static_settings, set
        )
    elif variational_scheme == "strong form galerkin":
        # Direct implementation of residual, e.g. for Galerkin method
        residual_at_int_point_vj = jax.vmap(
            variational_schemes.direct_residual_at_int_point,
            (0, 0, 0, 0, None, None, None),
            0,
        )
        residual_contributions = residual_at_int_point_vj(
            x_int, w_int, int_point_numbers, local_dofs, settings, static_settings, set
        )
    elif variational_scheme == "weak form galerkin":
        # Pass local_dofs twice (assuming Bubnov Galerkin...)
        residual_at_int_point_vj = jax.vmap(
            variational_schemes.residual_from_deriv_at_int_point,
            (0, 0, 0, 0, 0, None, None, None),
            0,
        )
        residual_contributions = residual_at_int_point_vj(
            x_int,
            w_int,
            int_point_numbers,
            local_dofs,
            local_dofs,
            settings,
            static_settings,
            set,
        )
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
    assert isinstance(
        dofs, jnp.ndarray
    ), "Variational schemes do currently not support dofs as dicts."

    _, local_dofs, _, _, connectivity = _get_element_quantities(
        dofs, settings, static_settings, set
    )
    variational_scheme = static_settings["variational scheme"][set]
    x_int = settings["integration coordinates"][set]
    w_int = settings["integration weights"][set]
    int_point_numbers = jnp.arange(0, x_int.shape[0], 1)

    # Compute tangent contributions
    if (
        variational_scheme == "least square pde loss"
        or variational_scheme == "least square function approximation"
    ):
        tangent_at_int_point = jax.jacfwd(
            jax.jacrev(variational_schemes.functional_at_int_point, argnums=3),
            argnums=3,
        )  # argnum=3 are unknowns with local support
        tangent_at_int_point_vj = jax.vmap(
            tangent_at_int_point, (0, 0, 0, 0, None, None, None), 0
        )
        tangent_contributions = tangent_at_int_point_vj(
            x_int, w_int, int_point_numbers, local_dofs, settings, static_settings, set
        )
    elif variational_scheme == "strong form galerkin":
        tangent_at_int_point = jax.jacfwd(
            variational_schemes.direct_residual_at_int_point, argnums=3
        )
        tangent_at_int_point_vj = jax.vmap(
            tangent_at_int_point, (0, 0, 0, 0, None, None, None), 0
        )
        tangent_contributions = tangent_at_int_point_vj(
            x_int, w_int, int_point_numbers, local_dofs, settings, static_settings, set
        )
    elif variational_scheme == "weak form galerkin":
        tangent_at_int_point = jax.jacfwd(
            variational_schemes.residual_from_deriv_at_int_point, argnums=3
        )
        tangent_at_int_point_vj = jax.vmap(
            tangent_at_int_point, (0, 0, 0, 0, 0, None, None, None), 0
        )
        tangent_contributions = tangent_at_int_point_vj(
            x_int,
            w_int,
            int_point_numbers,
            local_dofs,
            local_dofs,
            settings,
            static_settings,
            set,
        )
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
    assert isinstance(
        dofs, jnp.ndarray
    ), "Variational schemes do currently not support dofs as dicts."

    _, local_dofs, _, _, connectivity = _get_element_quantities(
        dofs, settings, static_settings, set
    )
    variational_scheme = static_settings["variational scheme"][set]
    x_int = settings["integration coordinates"][set]
    w_int = settings["integration weights"][set]
    int_point_numbers = jnp.arange(0, x_int.shape[0], 1)

    # Compute tangent contributions
    if (
        variational_scheme == "least square pde loss"
        or variational_scheme == "least square function approximation"
    ):
        tangent_at_int_point = jax.jacfwd(
            jax.jacrev(variational_schemes.functional_at_int_point, argnums=3),
            argnums=3,
        )  # argnum=3 are unknowns with local support
        tangent_at_int_point_vj = jax.vmap(
            tangent_at_int_point, (0, 0, 0, 0, None, None, None), 0
        )
        tangent_contributions = tangent_at_int_point_vj(
            x_int, w_int, int_point_numbers, local_dofs, settings, static_settings, set
        )
    elif variational_scheme == "strong form galerkin":
        tangent_at_int_point = jax.jacfwd(
            variational_schemes.direct_residual_at_int_point, argnums=3
        )
        tangent_at_int_point_vj = jax.vmap(
            tangent_at_int_point, (0, 0, 0, 0, None, None, None), 0
        )
        tangent_contributions = tangent_at_int_point_vj(
            x_int, w_int, int_point_numbers, local_dofs, settings, static_settings, set
        )
    elif variational_scheme == "weak form galerkin":
        tangent_at_int_point = jax.jacfwd(
            variational_schemes.residual_from_deriv_at_int_point, argnums=3
        )
        tangent_at_int_point_vj = jax.vmap(
            tangent_at_int_point, (0, 0, 0, 0, 0, None, None, None), 0
        )
        tangent_contributions = tangent_at_int_point_vj(
            x_int,
            w_int,
            int_point_numbers,
            local_dofs,
            local_dofs,
            settings,
            static_settings,
            set,
        )
    else:
        raise KeyError("Variational scheme mode not or wrongly specified!")

    # Assembling (without summing duplicates)
    data = dict_flatten(tangent_contributions)
    indices = _get_indices(connectivity, dofs)
    num_dofs = (
        dofs.size if not isinstance(dofs, dict) else sum(v.size for v in dofs.values())
    )
    tangent_matrix = sparse.BCOO((data, indices), shape=(num_dofs, num_dofs))
    return tangent_matrix


### Assembling for user potentials
@jit_with_docstring(static_argnames=["static_settings", "set"])
def user_potential_integrate_functional(dofs, settings, static_settings, set):
    """
    Assembly of potential for custom user definition of specified domain.

    Args:
      dofs (jnp.ndarray or dict): Degrees of freedom.
      settings (dict): Settings dictionary.
      static_settings (flax.core.FrozenDict): Static settings as frozen dictionary.
      set (int): The domain number.

    Returns:
      float: value of functional integrated over set of elements
    """
    model_fun, local_dofs, local_node_coor, elem_numbers, _ = _get_element_quantities(
        dofs, settings, static_settings, set
    )

    # Vmapping over all elements
    functional_contributions = jax.vmap(model_fun, (0, 0, 0, None, None, None), (0))(
        local_dofs, local_node_coor, elem_numbers, settings, static_settings, set
    )

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
    model_fun, local_dofs, local_node_coor, elem_numbers, connectivity = (
        _get_element_quantities(dofs, settings, static_settings, set)
    )

    residual_contributions = jax.vmap(
        jax.jacrev(model_fun), (0, 0, 0, None, None, None), (0)
    )(local_dofs, local_node_coor, elem_numbers, settings, static_settings, set)

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
    model_fun, local_dofs, local_node_coor, elem_numbers, connectivity = (
        _get_element_quantities(dofs, settings, static_settings, set)
    )

    tangent_contributions = jax.vmap(
        jax.jacfwd(jax.jacrev(model_fun)), (0, 0, 0, None, None, None), (0)
    )(local_dofs, local_node_coor, elem_numbers, settings, static_settings, set)

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
    model_fun, local_dofs, local_node_coor, elem_numbers, connectivity = (
        _get_element_quantities(dofs, settings, static_settings, set)
    )

    tangent_contributions = jax.vmap(
        jax.jacfwd(jax.jacrev(model_fun)), (0, 0, 0, None, None, None), (0)
    )(local_dofs, local_node_coor, elem_numbers, settings, static_settings, set)

    data = dict_flatten(tangent_contributions)
    indices = _get_indices(connectivity, dofs)
    num_dofs = (
        dofs.size if not isinstance(dofs, dict) else sum(v.size for v in dofs.values())
    )

    tangent_matrix = sparse.BCOO((data, indices), shape=(num_dofs, num_dofs))
    return tangent_matrix


### Assembling for user residuals
@jit_with_docstring(static_argnames=["static_settings", "set"])
def user_residual_assemble_residual(dofs, settings, static_settings, set):
    """
    Assembly of residual for custom user residual of specified domain.

    Args:
      dofs (jnp.ndarray or dict): Degrees of freedom.
      settings (dict): Settings dictionary.
      static_settings (flax.core.FrozenDict): Static settings as frozen dictionary.
      set (int): The domain number.

    Returns:
      jnp.ndarray: The assembled residual.
    """
    model_fun, local_dofs, local_node_coor, elem_numbers, connectivity = (
        _get_element_quantities(dofs, settings, static_settings, set)
    )

    residual_contributions = jax.vmap(model_fun, (0, 0, 0, None, None, None), (0))(
        local_dofs, local_node_coor, elem_numbers, settings, static_settings, set
    )

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
    model_fun, local_dofs, local_node_coor, elem_numbers, connectivity = (
        _get_element_quantities(dofs, settings, static_settings, set)
    )

    tangent_contributions = jax.vmap(
        jax.jacfwd(model_fun), (0, 0, 0, None, None, None), (0)
    )(local_dofs, local_node_coor, elem_numbers, settings, static_settings, set)

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
    model_fun, local_dofs, local_node_coor, elem_numbers, connectivity = (
        _get_element_quantities(dofs, settings, static_settings, set)
    )

    tangent_contributions = jax.vmap(
        jax.jacfwd(model_fun), (0, 0, 0, None, None, None), (0)
    )(local_dofs, local_node_coor, elem_numbers, settings, static_settings, set)

    data = dict_flatten(tangent_contributions)
    indices = _get_indices(connectivity, dofs)
    num_dofs = (
        dofs.size if not isinstance(dofs, dict) else sum(v.size for v in dofs.values())
    )
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

    model_fun, local_dofs, local_node_coor, elem_numbers, connectivity = (
        _get_element_quantities(dofs, settings, static_settings, set)
    )

    residual_contributions = jax.vmap(
        model_fun, (0, 0, 0, None, None, None, None), (0)
    )(
        local_dofs,
        local_node_coor,
        elem_numbers,
        settings,
        static_settings,
        "residual",
        set,
    )

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
    model_fun, local_dofs, local_node_coor, elem_numbers, connectivity = (
        _get_element_quantities(dofs, settings, static_settings, set)
    )

    tangent_contributions = jax.vmap(model_fun, (0, 0, 0, None, None, None, None), (0))(
        local_dofs,
        local_node_coor,
        elem_numbers,
        settings,
        static_settings,
        "tangent",
        set,
    )

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

    model_fun, local_dofs, local_node_coor, elem_numbers, connectivity = (
        _get_element_quantities(dofs, settings, static_settings, set)
    )

    tangent_contributions = jax.vmap(model_fun, (0, 0, 0, None, None, None, None), (0))(
        local_dofs,
        local_node_coor,
        elem_numbers,
        settings,
        static_settings,
        "tangent",
        set,
    )

    data = dict_flatten(tangent_contributions)
    indices = _get_indices(connectivity, dofs)
    num_dofs = (
        dofs.size if not isinstance(dofs, dict) else sum(v.size for v in dofs.values())
    )
    tangent_matrix = sparse.BCOO((data, indices), shape=(num_dofs, num_dofs))
    return tangent_matrix
