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

import jax.numpy as jnp
import jax
from jax import vmap, jacrev, jacfwd, hessian, jvp, linearize, vjp, custom_jvp
from jax.tree import map as treemap
from jax.experimental import sparse

from autopdex import variational_schemes
from autopdex.utility import jit_with_docstring, dict_zeros_like, dict_flatten, reshape_as


# TODO: change vmaps to _batched_map in order to reduce memory consumption


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
    if callable(dofs):
        dofs = dofs(0.)

    if isinstance(dofs, dict):
        keys = dofs.keys()

        # Field-Offsets
        field_offsets = {}
        current_offset = 0
        for field in keys:
            field_offsets[field] = current_offset
            field_size = dofs[field].size
            current_offset += field_size

        indices_list = []

        num_elems = connectivity[next(iter(keys))].shape[0]
        elem_indices = jnp.arange(num_elems, dtype=int)

        for field_i in keys:
            for field_j in keys:

                def one_elem_indices(elem_idx):
                    # Global DOFs for field_i
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
                    global_dofs_i = jnp.asarray(dof_indices_i, dtype=int).flatten()

                    # Global DOFs for field_j
                    conn_j = connectivity[field_j][elem_idx]
                    if dofs[field_j].ndim == 1:
                        dofs_per_node_j = 1
                    else:
                        dofs_per_node_j = dofs[field_j].shape[-1]

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
        dofs (jnp.ndarray or dict or callable): Degrees of freedom. Can be a function of time for transient problems.
        settings (dict): Settings dictionary.
        static_settings (dict or flax.core.FrozenDict): Static settings dictionary.
        set (int): The domain number.

    Returns:
        tuple: (model_fun, local_dofs, local_node_coor, elem_numbers, connectivity)
    """
    model_fun = static_settings["model"][set]
    x_nodes = settings["node coordinates"]
    dofs_is_fun = True if callable(dofs) else False
    if dofs_is_fun:
        dofs_is_dict = True if isinstance(dofs(0.), dict) else False
    else:
        dofs_is_dict = True if isinstance(dofs, dict) else False

    # Warning if it was defined in static_settings
    assert "connectivity" not in static_settings, \
        "'connectivity' has been moved to 'settings' in order to reduce compile time. \
        Further, you should not transform it to a tuple of tuples anymore."

    connectivity = settings["connectivity"][set]

    if dofs_is_dict:
        assert isinstance(
            x_nodes, dict
        ), "If 'dofs' is a dict, 'settings['node coordinates']' must also be a dict."
        assert isinstance(
            connectivity, dict
        ), "If 'dofs' is a dict, 'settings['connectivity'][set]' must also be a dict."

        elem_numbers = jnp.arange(connectivity[next(iter(connectivity))].shape[0])
    else:
        elem_numbers = jnp.arange(connectivity.shape[0])

    return model_fun, x_nodes, elem_numbers, connectivity

def _get_element_quantities_2(dofs, settings, static_settings, set):
    if callable(dofs):
        dofs = dofs(0.)
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
    # If DOFs are a function of time (for transient problems, forward them as a function of time)
    if callable(dofs):
        local_dofs = lambda t: treemap(lambda x, y: x.at[y].get(), dofs(t), node_list)
        # local_dofs = _make_elem_dofs_fun(dofs, node_list)
    else:
        local_dofs = treemap(lambda x, y: x.at[y].get(), dofs, node_list)
    local_node_coor = treemap(lambda x, y: x.at[y].get(), x_nodes, node_list)
    return local_dofs, local_node_coor

def _extract_local_dofs_and_coor_2(dofs, int_point_number, x_int, w_int, connectivity):
    x_i = x_int[int_point_number]
    w_i = w_int[int_point_number]
    if callable(dofs):
        local_dofs = lambda t: treemap(lambda x, y: x.at[y].get(), dofs(t), connectivity[int_point_number])
        # local_dofs = _make_elem_dofs_fun(dofs, connectivity[int_point_number])
    else:
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
    if callable(dofs):
        dofs = dofs(0.)

    # Total number of DOFs
    num_dofs = (
        dofs.size if not isinstance(dofs, dict) else sum(v.size for v in dofs.values())
    )

    if isinstance(dofs, dict):
        keys = dofs.keys()
        n_elems = next(iter(connectivity.values())).shape[0]  # Number of elements

        # Compute field offsets for global DOF numbering
        field_offsets = {}
        current_offset = 0
        for key in keys:
            field_offsets[key] = current_offset
            field_size = dofs[key].size  # Total DOFs in the field
            current_offset += field_size

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
    if callable(dofs):
        dofs = dofs(0.)

    if isinstance(dofs, dict):
        keys = dofs.keys()
        n_elems = connectivity[next(iter(keys))].shape[0]
        
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

def _make_elem_dofs_fun(dofs, elem):
    """This function takes the function `dofs(t)` and returns basically lambda t: dofs(t)[elem].
    
    The difference is, that when this function is used under jacrev, it will allocate less memory.
    Supports only as many derivatives as are defined via the custom_jvp decorators
    """
    # @custom_jvp
    # def elem_dofs_ttt_f(t):
    #     dofs_ttt_ = jacfwd(jacfwd(jacfwd(dofs)))(t)
    #     elem_dofs_ttt_ = treemap(lambda x, y: x.at[y].get(), dofs_ttt_, elem)
    #     return elem_dofs_ttt_
    # @elem_dofs_ttt_f.defjvp
    # def elem_dofs_ttt_jvp(primals, tangents):
    #     t, = primals
    #     t_dot, = tangents
    #     elem_dofs_tttt = jacfwd(jacfwd(jacfwd(jacfwd(dofs))))(t)[elem]
    #     return elem_dofs_ttt_f(t), treemap(lambda x: x * t_dot, elem_dofs_tttt)
    # @custom_jvp
    # def elem_dofs_tt_f(t):
    #     dofs_tt_ = jacfwd(jacfwd(dofs))(t)
    #     elem_dofs_tt_ = treemap(lambda x, y: x.at[y].get(), dofs_tt_, elem)
    #     return elem_dofs_tt_
    # @elem_dofs_tt_f.defjvp
    # def elem_dofs_tt_jvp(primals, tangents):
    #     t, = primals
    #     t_dot, = tangents
    #     elem_dofs_ttt = elem_dofs_ttt_f(t)
    #     return elem_dofs_tt_f(t), treemap(lambda x: x * t_dot, elem_dofs_ttt)
    # @custom_jvp
    # def elem_dofs_t_f(t):
    #     dofs_t_ = jacfwd(dofs)(t)
    #     elem_dofs_t_ = treemap(lambda x, y: x.at[y].get(), dofs_t_, elem)
    #     return elem_dofs_t_
    # @elem_dofs_t_f.defjvp
    # def elem_dofs_t_jvp(primals, tangents):
    #     t, = primals
    #     t_dot, = tangents
    #     elem_dofs_tt = elem_dofs_tt_f(t)
    #     return elem_dofs_t_f(t), treemap(lambda x: x * t_dot, elem_dofs_tt)
    # @custom_jvp
    # def elem_dofs_f(t):
    #     dofs_ = dofs(t)
    #     elem_dofs_ = treemap(lambda x, y: x.at[y].get(), dofs_, elem)
    #     return elem_dofs_
    # @elem_dofs_f.defjvp
    # def elem_dofs_jvp(primals, tangents):
    #     t, = primals
    #     t_dot, = tangents
    #     elem_dofs_t = elem_dofs_t_f(t)
    #     return elem_dofs_f(t), treemap(lambda x: x * t_dot, elem_dofs_t)
    # return elem_dofs_f


    # @partial(custom_jvp, nondiff_argnums=(1, 2))
    # def elem_dofs_ttt_f(t, dofs, elem):
    #     dofs_ttt_ = jacfwd(jacfwd(jacfwd(dofs)))(t)
    #     elem_dofs_ttt_ = treemap(lambda x, y: x.at[y].get(), dofs_ttt_, elem)
    #     return elem_dofs_ttt_
    # @elem_dofs_ttt_f.defjvp
    # def elem_dofs_ttt_jvp(dofs, elem, primals, tangents):
    #     t, = primals
    #     t_dot, = tangents
    #     elem_dofs_tttt = jacfwd(jacfwd(jacfwd(jacfwd(dofs))))(t)[elem]
    #     return elem_dofs_ttt_f(t, dofs, elem), treemap(lambda x: x * t_dot, elem_dofs_tttt)
    # @partial(custom_jvp, nondiff_argnums=(1, 2))
    # def elem_dofs_tt_f(t, dofs, elem):
    #     dofs_tt_ = jacfwd(jacfwd(dofs))(t)
    #     elem_dofs_tt_ = treemap(lambda x, y: x.at[y].get(), dofs_tt_, elem)
    #     return elem_dofs_tt_
    # @elem_dofs_tt_f.defjvp
    # def elem_dofs_tt_jvp(dofs, elem, primals, tangents):
    #     t, = primals
    #     t_dot, = tangents
    #     elem_dofs_ttt = elem_dofs_ttt_f(t, dofs, elem)
    #     return elem_dofs_tt_f(t, dofs, elem), treemap(lambda x: x * t_dot, elem_dofs_ttt)
    @partial(custom_jvp, nondiff_argnums=(1, 2))
    def elem_dofs_f(t, dofs, elem):
        dofs_ = dofs(t)
        elem_dofs_ = treemap(lambda x, y: x[y], dofs_, elem)
        return elem_dofs_
    @elem_dofs_f.defjvp
    def elem_dofs_jvp(dofs, elem, primals, tangents):
        t, = primals
        t_dot, = tangents

        # @partial(custom_jvp, nondiff_argnums=(1, 2))
        # def elem_dofs_t_f(t, dofs, elem):
        #     dofs_t_ = jacfwd(dofs)(t)
        #     elem_dofs_t_ = treemap(lambda x, y: x.at[y].get(), dofs_t_, elem)
        #     return elem_dofs_t_
        # @elem_dofs_t_f.defjvp
        # def elem_dofs_t_jvp(dofs, elem, primals, tangents):
        #     t, = primals
        #     t_dot, = tangents
        #     # elem_dofs_tt = elem_dofs_tt_f(t, dofs, elem)
        #     elem_dofs_tt = jacfwd(jacfwd(dofs))(t)[elem]
        #     return elem_dofs_t_f(t, dofs, elem), treemap(lambda x: x * t_dot, elem_dofs_tt)

        # elem_dofs_t = elem_dofs_t_f(t, dofs, elem)
        elem_dofs_t = treemap(lambda a, b: a[b], jacfwd(dofs)(t), elem)
        return elem_dofs_f(t, dofs, elem), treemap(lambda x: x * t_dot, elem_dofs_t)
    local_dofs_fun = lambda t: elem_dofs_f(t, dofs, elem)
    return local_dofs_fun

def _batched_map(fun, elem_numbers, connectivity):
    body_fun = lambda i: fun(elem_numbers[i], jax.tree.map(lambda x: x[i], connectivity))
    num_dofs_per_elem = jax.eval_shape(lambda i: dict_flatten(body_fun(i)), 0).shape[0]
    return jax.lax.map(body_fun, jnp.arange(elem_numbers.shape[0]), batch_size=int(64000/num_dofs_per_elem))


### General assembling functions
@jit_with_docstring(static_argnames=["static_settings"], possibly_static_argnames=['dofs'])
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

@jit_with_docstring(static_argnames=["static_settings"], possibly_static_argnames=['dofs'])
def assemble_residual(dofs, settings, static_settings):
    """
    Assemble residuals over set of domains.

    Args:
      dofs (jnp.ndarray or dict or callable): Degrees of freedom. 
        Can be a function of time for transient problems in combination with user_residuals.
      settings (dict): Settings dictionary.
      static_settings (flax.core.FrozenDict): Static settings as frozen dictionary.

    Returns:
      jnp.ndarray: The assembled residual.
    """
    # Loop over all sets of integration points/ domains
    num_sets = len(static_settings["assembling mode"])

    if isinstance(dofs(0.), dict) if callable(dofs) else isinstance(dofs, dict):
        assert all([isinstance(settings['connectivity'][0], dict),
                    isinstance(settings['node coordinates'], dict)]), \
                    "If the DOFs are a dict, the connectivity, node coordinates, dirichlet dofs, and dirichlet conditions must also be dicts."
        
    if callable(dofs):
        integrated_residual = dict_zeros_like(dofs(0.))
    else:
        integrated_residual = dict_zeros_like(dofs)
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
            integrated_residual = treemap(lambda x, y: x + y, integrated_residual, add)
        else:
            integrated_residual += add

    return integrated_residual

@jit_with_docstring(static_argnames=["static_settings"], possibly_static_argnames=['dofs'])
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
    if callable(dofs):
        tangent_diagonal = jnp.zeros_like(dict_flatten(dofs(0.)))
    else:
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

@jit_with_docstring(static_argnames=["static_settings"], possibly_static_argnames=['dofs'])
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
                    (num_dofs, num_dofs), dtype=float, index_dtype=jnp.int_
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


### Dense assembling
@jit_with_docstring(static_argnames=["static_settings", "set"], possibly_static_argnames=['dofs'])
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

@jit_with_docstring(static_argnames=["static_settings", "set"], possibly_static_argnames=['dofs'])
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

@jit_with_docstring(static_argnames=["static_settings", "set"], possibly_static_argnames=['dofs'])
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
@jit_with_docstring(static_argnames=["static_settings", "set"], possibly_static_argnames=['dofs'])
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

@jit_with_docstring(static_argnames=["static_settings", "set"], possibly_static_argnames=['dofs'])
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

@jit_with_docstring(static_argnames=["static_settings", "set"], possibly_static_argnames=['dofs'])
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

@jit_with_docstring(static_argnames=["static_settings", "set"], possibly_static_argnames=['dofs'])
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
    num_dofs = (
        dofs.size if not isinstance(dofs, dict) else sum(v.size for v in dofs.values())
    )
    tangent_matrix = sparse.BCOO((data, indices), shape=(num_dofs, num_dofs))
    return tangent_matrix


### Assembling for user potentials
@jit_with_docstring(static_argnames=["static_settings", "set"], possibly_static_argnames=['dofs'])
def user_potential_integrate_functional(dofs, settings, static_settings, set):
    """
    Assembly of potential for custom user definition of specified domain.

    Args:
      dofs (jnp.ndarray or dict or callable): Degrees of freedom. Can be a function of time for transient problems.
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

@jit_with_docstring(static_argnames=["static_settings", "set"], possibly_static_argnames=['dofs'])
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
    model_fun, x_nodes, elem_numbers, connectivity = _get_element_quantities(dofs, settings, static_settings, set)

    # Modify the model_fun such that it extracts the DOFs from the global dofs and vmap only over connectivity
    def element_residual(elem_number, node_list):
        local_dofs, local_node_coor = _extract_local_dofs_and_coor(dofs, node_list, x_nodes)
        return  jacrev(model_fun)(local_dofs, local_node_coor, elem_number, settings, static_settings, set)

    # residual_contributions = vmap(element_residual, (0, 0))(elem_numbers, connectivity)
    residual_contributions = _batched_map(element_residual, elem_numbers, connectivity)

    return _get_residual(residual_contributions, connectivity, dofs)

@jit_with_docstring(static_argnames=["static_settings", "set"], possibly_static_argnames=['dofs'])
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

@jit_with_docstring(static_argnames=["static_settings", "set"], possibly_static_argnames=['dofs'])
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

    # tangent_contributions = vmap(element_tangent, (0, 0), (0))(elem_numbers, connectivity)

    body_fun = lambda i: element_tangent(elem_numbers[i], jax.tree.map(lambda x: x[i], connectivity))
    num_dofs_per_elem = jax.eval_shape(lambda i: dict_flatten(body_fun(i)), 0).shape[0]
    tangent_contributions = jax.lax.map(body_fun, jnp.arange(elem_numbers.shape[0]), batch_size=int(64000/num_dofs_per_elem))

    data = dict_flatten(tangent_contributions)
    indices = _get_indices(connectivity, dofs)
    num_dofs = (
        dofs.size if not isinstance(dofs, dict) else sum(v.size for v in dofs.values())
    )

    tangent_matrix = sparse.BCOO((data, indices), shape=(num_dofs, num_dofs))
    return tangent_matrix

@jit_with_docstring(static_argnames=["static_settings", "set"], possibly_static_argnames=['dofs'])
def _user_potential_assemble_r_and_t(dofs, settings, static_settings, set):
    model_fun, x_nodes, elem_numbers, connectivity = _get_element_quantities(dofs, settings, static_settings, set)

    # Modify the model_fun such that it extracts the dofs from the global dofs and vmap only over connectivity
    def element_r_and_t(elem_number, node_list):
        local_dofs, local_node_coor = _extract_local_dofs_and_coor(dofs, node_list, x_nodes)
        residual_fun = lambda x: jacrev(model_fun)(x, local_node_coor, elem_number, settings, static_settings, set)

        elem_res = residual_fun(local_dofs)
        elem_tan = jacfwd(residual_fun)(local_dofs)

        # def residual_and_tangent_linearize(residual_fun, local_dofs):
        #     primals, lin_fun = linearize(residual_fun, local_dofs)            
        #     flat_local_dofs = dict_flatten(local_dofs)
        #     n = flat_local_dofs.shape[0]            
        #     identity = jnp.eye(n)            
        #     jacobian = vmap(lambda v: lin_fun(reshape_as(v, local_dofs)))(identity)            
        #     return primals, jacobian
        # elem_res, elem_tan = residual_and_tangent_linearize(residual_fun, local_dofs)

        # def residual_and_tangent(residual_fun, local_dofs):
        #     flat_local_dofs = dict_flatten(local_dofs)
        #     n = flat_local_dofs.shape[0]
        #     identity = jnp.eye(n)
        #     def jvp_with_flat_tangent(v):
        #         tangent_pytree = reshape_as(v, local_dofs)
        #         return jvp(residual_fun, (local_dofs,), (tangent_pytree,))
        #     primals, elem_tan = vmap(jvp_with_flat_tangent)(identity)
        #     elem_res = treemap(lambda x: x[0], primals)
        #     return elem_res, elem_tan        
        # elem_res, elem_tan = residual_and_tangent(residual_fun, local_dofs)

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
    num_dofs = (
        dofs.size if not isinstance(dofs, dict) else sum(v.size for v in dofs.values())
    )

    tangent_matrix = sparse.BCOO((data, indices), shape=(num_dofs, num_dofs))
    return residual, tangent_matrix


### Assembling for user residuals
@jit_with_docstring(static_argnames=["static_settings", "set"], possibly_static_argnames=['dofs'])
def user_residual_assemble_residual(dofs, settings, static_settings, set):
    """
    Assembly of residual for custom user residual of specified domain.

    Args:
      dofs (jnp.ndarray or dict or callable): Degrees of freedom. Can be a function of time for transient problems.
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
    
    # residual_contributions = vmap(element_residual, (0, 0), (0))(elem_numbers, connectivity)

    body_fun = lambda i: element_residual(elem_numbers[i], jax.tree.map(lambda x: x[i], connectivity))
    num_dofs_per_elem = jax.eval_shape(lambda i: dict_flatten(body_fun(i)), 0).shape[0]
    residual_contributions = jax.lax.map(body_fun, jnp.arange(elem_numbers.shape[0]), batch_size=int(64000/num_dofs_per_elem))

    return _get_residual(residual_contributions, connectivity, dofs(0.) if callable(dofs) else dofs)

@jit_with_docstring(static_argnames=["static_settings", "set"], possibly_static_argnames=['dofs'])
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

@jit_with_docstring(static_argnames=["static_settings", "set"], possibly_static_argnames=['dofs'])
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

    if callable(dofs):
        dofs0 = dofs(0.)
        num_dofs = (
        dofs0.size if not isinstance(dofs0, dict) else sum(v.size for v in dofs0.values())
        )
    else:
        num_dofs = (
            dofs.size if not isinstance(dofs, dict) else sum(v.size for v in dofs.values())
        )
    tangent_matrix = sparse.BCOO((data, indices), shape=(num_dofs, num_dofs))
    return tangent_matrix


### Assembling for user elements
@jit_with_docstring(static_argnames=["static_settings", "set"], possibly_static_argnames=['dofs'])
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

@jit_with_docstring(static_argnames=["static_settings", "set"], possibly_static_argnames=['dofs'])
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

@jit_with_docstring(static_argnames=["static_settings", "set"], possibly_static_argnames=['dofs'])
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
    num_dofs = (
        dofs.size if not isinstance(dofs, dict) else sum(v.size for v in dofs.values())
    )
    tangent_matrix = sparse.BCOO((data, indices), shape=(num_dofs, num_dofs))
    return tangent_matrix


### Internal variable update function
@jit_with_docstring(static_argnames=["static_settings", "set"], possibly_static_argnames=['dofs'])
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
