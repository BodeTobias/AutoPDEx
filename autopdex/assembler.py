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
Module for assembling and integrating functionals, residuals and tangents, supporting both 'sparse' and 'dense' modes and 'user potentials', 'user residuals' and 'user elements'.

The assembly modes 'dense' and 'sparse' use JAX's automatic differentiation capabilities to determine the global residual and the tangent for the given values of the degrees of freedom and the models and variational schemes specified in static_settings.
In the 'dense' mode, the tangent matrix is returned as a jnp.ndarray, while in the 'sparse' mode, it is returned as a jax.experimental.BCOO matrix with duplicates. 
In the 'user potential', 'user residual' and 'user element' modes, the global residual and tangent can be assembled based on user-defined element-wise contributions. 
Depending on the execution location, the entries are calculated on CPU or GPU. 
For an efficient calculation, JAX's automatic vectorization transformation (vmap) is used.
The summation of duplicates is then carried out within the solver module. Currently, SciPy is used for this on the CPU.
"""

import jax
import jax.experimental
import jax.experimental.pjit
import jax.experimental.shard_map
import jax.numpy as jnp
from jax.experimental import sparse
import sparsejac
import numpy as np

from autopdex import variational_schemes
from autopdex.utility import jit_with_docstring


### Assembling switches: sparse/dense
@jit_with_docstring(static_argnames=['static_settings'])
def integrate_functional(dofs, settings, static_settings):
  """
  Integrate functional as sum over set of domains.
  
  Parameters:
    dofs (jnp.ndarray): Degrees of freedom.
    settings (dict): Settings for the integration.
    static_settings (flax.core.FrozenDict): Static settings including assembling modes.
  
  Returns:
    float: The integrated functional value for current dofs.
  """
  # ToDo: add a function that checks the entries of settings and static_settings for inconsistencies

  # Loop over all sets of integration points/ domains
  num_sets = len(static_settings['assembling mode'])
  integrated_functional = 0
  for set in range(num_sets):
    assembling_mode = static_settings['assembling mode'][set]

    if assembling_mode == 'dense':
      integrated_functional += dense_integrate_functional(dofs, settings, static_settings, set)
    elif assembling_mode == 'sparse':
      integrated_functional += sparse_integrate_functional(dofs, settings, static_settings, set)
    elif assembling_mode == 'user potential':
      integrated_functional += user_potential_integrate_functional(dofs, settings, static_settings, set)
    else:
      assert False, 'Assembling mode can be either \'sparse\' or \'dense\' in integrate_functional'

  return integrated_functional

@jit_with_docstring(static_argnames=['static_settings'])
def assemble_residual(dofs, settings, static_settings):
  """
  Assemble residuals over set of domains.
  
  Parameters:
    dofs (jnp.ndarray): Degrees of freedom.
    settings (dict): Settings for the integration.
    static_settings (flax.core.FrozenDict): Static settings including assembling modes.
  
  Returns:
    jnp.ndarray: The assembled residual.
  """
  # Loop over all sets of integration points/ domains
  num_sets = len(static_settings['assembling mode'])
  integrated_residual = jnp.zeros_like(dofs)
  for set in range(num_sets):
    assembling_mode = static_settings['assembling mode'][set]

    if assembling_mode == 'dense':
      integrated_residual += dense_assemble_residual(dofs, settings, static_settings, set)
    elif assembling_mode == 'sparse':
      integrated_residual += sparse_assemble_residual(dofs, settings, static_settings, set)
    elif assembling_mode == 'user potential':
      integrated_residual += user_potential_assemble_residual(dofs, settings, static_settings, set)
    elif assembling_mode == 'user residual':
      integrated_residual += user_residual_assemble_residual(dofs, settings, static_settings, set)
    elif assembling_mode == 'user element':
      integrated_residual += user_element_assemble_residual(dofs, settings, static_settings, set)
    else:
      assert False, 'Assembling mode can be either \'sparse\', \'dense\' or \'user element\''

  return integrated_residual

@jit_with_docstring(static_argnames=['static_settings'])
def assemble_tangent_diagonal(dofs, settings, static_settings):
  """
  Assemble the diagonal of the tangent matrix.
  
  Parameters:
    dofs (jnp.ndarray): Degrees of freedom.
    settings (dict): Settings for the integration.
    static_settings (flax.core.FrozenDict): Static settings including assembling modes.
  
  Returns:
    jnp.ndarray: The diagonal of the assembled tangent matrix.
  """
  # Loop over all sets of integration points/ domains
  num_sets = len(static_settings['assembling mode'])
  tangent_diagonal = jnp.zeros_like(dofs.flatten())
  for set in range(num_sets):
    assembling_mode = static_settings['assembling mode'][set]
    if assembling_mode == 'sparse':
      tangent_diagonal += sparse_assemble_tangent_diagonal(dofs, settings, static_settings, set)
    elif assembling_mode == 'user potential':
      tangent_diagonal += user_potential_assemble_tangent_diagonal(dofs, settings, static_settings, set)
    elif assembling_mode == 'user residual':
      tangent_diagonal += user_residual_assemble_tangent_diagonal(dofs, settings, static_settings, set)
    elif assembling_mode == 'user element':
      tangent_diagonal += user_element_assemble_tangent_diagonal(dofs, settings, static_settings, set)
    else:
      assert False, 'Assembling mode for assembling tangent diagonal supports currently only \'sparse\' and \'user element\''
  return tangent_diagonal

@jit_with_docstring(static_argnames=['static_settings'])
def assemble_tangent(dofs, settings, static_settings):
  """
  Assemble the full (possibly sparse) tangent matrix.
  
  Parameters:
    dofs (jnp.ndarray): Degrees of freedom.
    settings (dict): Settings for the integration.
    static_settings (flax.core.FrozenDict): Static settings including assembling modes.
  
  Returns:
    jnp.ndarray or sparse matrix: The assembled tangent matrix.
  """
  # ToDo: add symmetric mode

  num_sets = len(static_settings['assembling mode'])
  one_dense = 'dense' in static_settings['assembling mode']
  num_dofs = np.prod(dofs.shape)
  try:
    sparsity_pattern = static_settings['known sparsity pattern']
  except KeyError:
    sparsity_pattern = 'none'

  match sparsity_pattern:
    case 'none':
      if one_dense:
        integrated_tangent = jnp.zeros((num_dofs, num_dofs))
      else:
        integrated_tangent = sparse.empty((num_dofs, num_dofs), dtype=jnp.float_, index_dtype=jnp.int_)

      # Loop over all sets of integration points/ domains
      for set in range(num_sets):
        assembling_mode = static_settings['assembling mode'][set]

        if assembling_mode == 'dense':
          integrated_tangent += dense_assemble_tangent(dofs, settings, static_settings, set)
        else:
          if assembling_mode == 'sparse':
            add = sparse_assemble_tangent(dofs, settings, static_settings, set)
          elif assembling_mode == 'user potential':
            add = user_potential_assemble_tangent(dofs, settings, static_settings, set)
          elif assembling_mode == 'user residual':
            add = user_residual_assemble_tangent(dofs, settings, static_settings, set)
          elif assembling_mode == 'user element':
            add = user_element_assemble_tangent(dofs, settings, static_settings, set)
          else:
            assert False, 'Assembling mode can be either \'sparse\', \'dense\' or \'user element\''

          if one_dense:
            integrated_tangent += add.todense()
          else:
            integrated_tangent += add
    case 'diagonal':
      # Compute the diagonal tangent (not the diagonal of a tangent that is not diagonal)
      residual_fun = lambda flat_dofs: assemble_residual(flat_dofs.reshape(dofs.shape), settings, static_settings).flatten()
      with jax.ensure_compile_time_eval():
        data_and_indices = (jnp.ones((num_dofs,)),
                            jax.vmap(lambda i: jnp.asarray([i, i]))(jnp.arange(0, num_dofs)))
        mat_shape = (num_dofs,num_dofs)
        sparsity = jax.experimental.sparse.BCOO(data_and_indices, shape=mat_shape) # ToDo: with unique and sorted indices more efficient?
        sparse_diag_fun = sparsejac.jacfwd(residual_fun, sparsity=sparsity)
      diag = sparse_diag_fun(dofs.flatten())
      return diag

    case _:
      assert False, '\'known sparsity pattern\' mode is not implemented.'

  return integrated_tangent


### Dense assembling
@jit_with_docstring(static_argnames=['static_settings', 'set'])
def dense_integrate_functional(dofs, settings, static_settings, set):
  """
  Dense integration of functional of specified set.
  
  Parameters:
    dofs (jnp.ndarray): Degrees of freedom.
    settings (dict): Settings for the integration.
    static_settings (flax.core.FrozenDict): Static settings including assembling modes.
    set (int): The set index.
  
  Returns:
    float: The integrated functional value.
  """
  x_int = settings['integration coordinates'][set]
  w_int = settings['integration weights'][set]

  functional_at_int_point_vj = jax.vmap(variational_schemes.functional_at_int_point, (0,0,0,None,None,None,None), 0)
  int_point_numbers = jnp.arange(0, x_int.shape[0], 1)
  integrated_functional = functional_at_int_point_vj(x_int, w_int, int_point_numbers, dofs, settings, static_settings, set).sum()

  return integrated_functional

@jit_with_docstring(static_argnames=['static_settings', 'set'])
def dense_assemble_residual(dofs, settings, static_settings, set):
  """
  Dense assembly of residual of specified set.
  
  Parameters:
    dofs (jnp.ndarray): Degrees of freedom.
    settings (dict): Settings for the integration.
    static_settings (flax.core.FrozenDict): Static settings including assembling modes.
    set (int): The set index.
  
  Returns:
    jnp.ndarray: The assembled residual.
  """
  grad = jax.grad(dense_integrate_functional, set)
  return grad(dofs, settings, static_settings).flatten()

@jit_with_docstring(static_argnames=['static_settings', 'set'])
def dense_assemble_tangent(dofs, settings, static_settings, set):
  """
  Dense assembly of tangent of specified set.
  
  Parameters:
    dofs (jnp.ndarray): Degrees of freedom.
    settings (dict): Settings for the integration.
    static_settings (flax.core.FrozenDict): Static settings including assembling modes.
    set (int): The set index.
  
  Returns:
    jnp.ndarray: The assembled tangent matrix.
  """
  hessian = jax.hessian(dense_integrate_functional)
  return jnp.reshape(hessian(dofs, settings, static_settings, set),
                      (dofs.shape[0]*dofs.shape[1],dofs.shape[0]*dofs.shape[1]))


### Sparse assembling
@jit_with_docstring(static_argnames=['static_settings', 'set'])
def sparse_integrate_functional(dofs, settings, static_settings, set):
  """
  Sparse integration of functional of specified set.
  
  Parameters:
    dofs (jnp.ndarray): Degrees of freedom.
    settings (dict): Settings for the integration.
    static_settings (flax.core.FrozenDict): Static settings including assembling modes.
    set (int): The set index.
  
  Returns:
    float: The integrated functional value.
  """
  neighbor_list = jnp.asarray(static_settings['connectivity'][set])
  local_dofs = dofs[neighbor_list]
  x_int = settings['integration coordinates'][set]
  w_int = settings['integration weights'][set]

  functional_at_int_point_vj = jax.vmap(variational_schemes.functional_at_int_point, (0,0,0,0,None,None,None), 0)
  int_point_numbers = jnp.arange(0, x_int.shape[0], 1)
  return functional_at_int_point_vj(x_int, w_int, int_point_numbers, local_dofs, settings, static_settings, set).sum()

@jit_with_docstring(static_argnames=['static_settings', 'set'])
def sparse_assemble_residual(dofs, settings, static_settings, set):
  """
  Sparse assembly of residual of specified set.
  
  Parameters:
    dofs (jnp.ndarray): Degrees of freedom.
    settings (dict): Settings for the integration.
    static_settings (flax.core.FrozenDict): Static settings including assembling modes.
    set (int): The set index.
  
  Returns:
    jnp.ndarray: The assembled residual.
  """
  neighbor_list = jnp.asarray(static_settings['connectivity'][set])
  local_dofs = dofs[neighbor_list]
  variational_scheme = static_settings['variational scheme'][set]
  n_fields = static_settings['number of fields'][set]
  x_int = settings['integration coordinates'][set]
  w_int = settings['integration weights'][set]

  int_point_numbers = jnp.arange(0, x_int.shape[0], 1)

  if variational_scheme == 'least square pde loss' or variational_scheme == 'least square function approximation':
    grad = jax.jacrev(sparse_integrate_functional)
    return grad(dofs, settings, static_settings, set)

  elif variational_scheme == 'strong form galerkin':
    # Direct implementation of residual, e.g. for Galerkin method
    residual_at_int_point_vj = jax.vmap(variational_schemes.direct_residual_at_int_point, (0,0,0,0,None,None,None), 0)
    residual_contributions = residual_at_int_point_vj(x_int, w_int, int_point_numbers, local_dofs, settings, static_settings, set)

    flat_neighbors = neighbor_list.flatten()
    tmp = jnp.outer(jnp.ones(flat_neighbors.shape, dtype=jnp.int_), jnp.arange(n_fields, dtype=jnp.int_))
    indices = tmp + jnp.outer(flat_neighbors, n_fields * jnp.ones(n_fields, dtype=jnp.int_)).astype(jnp.int_)

    res = jnp.zeros(dofs.shape).flatten()
    res = res.at[indices.flatten()].add(residual_contributions.flatten())
    res = res.reshape(dofs.shape)
  elif variational_scheme == 'weak form galerkin':
    # Pass local_dofs twice (assuming Bubnov Galerkin...)
    residual_at_int_point_vj = jax.vmap(variational_schemes.residual_from_deriv_at_int_point, (0,0,0,0,0,None,None,None), 0)
    residual_contributions = residual_at_int_point_vj(x_int, w_int, int_point_numbers, local_dofs, local_dofs, settings, static_settings, set)

    # Remaining stuff like for strong form galerkin
    flat_neighbors = neighbor_list.flatten()
    tmp = jnp.outer(jnp.ones(flat_neighbors.shape, dtype=jnp.int_), jnp.arange(n_fields, dtype=jnp.int_))
    indices = tmp + jnp.outer(flat_neighbors, n_fields * jnp.ones(n_fields, dtype=jnp.int_)).astype(jnp.int_)

    res = jnp.zeros(dofs.shape).flatten()
    res = res.at[indices.flatten()].add(residual_contributions.flatten())
    res = res.reshape(dofs.shape)
    res = jnp.reshape(res, dofs.shape)
  else:
    assert False, 'Residual mode not or wrongly specified!'

  # Assembled residual
  return res

@jit_with_docstring(static_argnames=['static_settings', 'set'])
def sparse_assemble_tangent_diagonal(dofs, settings, static_settings, set):
  """
  Sparse assembly of the diagonal of the tangent matrix for specified set.
  
  Parameters:
    dofs (jnp.ndarray): Degrees of freedom.
    settings (dict): Settings for the integration.
    static_settings (flax.core.FrozenDict): Static settings including assembling modes.
    set (int): The set index.
  
  Returns:
    jnp.ndarray: The diagonal of the assembled tangent matrix.
  """

  neighbor_list = jnp.asarray(static_settings['connectivity'][set])
  local_dofs = dofs[neighbor_list]
  maximum = static_settings['maximal number of neighbors'][set]
  n_fields = static_settings['number of fields'][set]
  variational_scheme = static_settings['variational scheme'][set]
  x_int = settings['integration coordinates'][set]
  w_int = settings['integration weights'][set]

  # Compute tangent contributions
  if variational_scheme == 'least square pde loss' or variational_scheme == 'least square function approximation':
    tangent_at_int_point = jax.jacfwd(jax.jacrev(variational_schemes.functional_at_int_point, argnums=3), argnums=3) # argnum=3 are unknowns with local support
  elif variational_scheme == 'strong form galerkin':
    tangent_at_int_point = jax.jacfwd(variational_schemes.direct_residual_at_int_point, argnums=3)
  elif variational_scheme == 'weak form galerkin':
    tangent_at_int_point = jax.jacfwd(variational_schemes.residual_from_deriv_at_int_point, argnums=3)
  else:
    assert False, 'Residual mode not or wrongly specified!'

  int_point_numbers = jnp.arange(0, x_int.shape[0], 1)

  # Same as for assembling the whole matrix, but just take the diagonal on the local level
  if variational_scheme == 'weak form galerkin':
    tangent_at_int_point_vj = jax.vmap(tangent_at_int_point, (0,0,0,0,0,None,None,None), 0)
    tangent_contributions = tangent_at_int_point_vj(x_int, w_int, int_point_numbers, local_dofs, local_dofs, settings, static_settings, set)
  else:
    tangent_at_int_point_vj = jax.vmap(tangent_at_int_point, (0,0,0,0,None,None,None), 0)
    tangent_contributions = tangent_at_int_point_vj(x_int, w_int, int_point_numbers, local_dofs, settings, static_settings, set)
  tangent_contributions = jnp.reshape(tangent_contributions, (x_int.shape[0], maximum*n_fields, maximum*n_fields))
  diagonal_contributions = jnp.diagonal(tangent_contributions, axis1=1, axis2=2)

  # Assemble diagonal entries like a residual
  flat_neighbors = neighbor_list.flatten()
  tmp = jnp.outer(jnp.ones(flat_neighbors.shape, dtype=jnp.int_), jnp.arange(n_fields, dtype=jnp.int_))
  indices = tmp + jnp.outer(flat_neighbors, n_fields * jnp.ones(n_fields, dtype=jnp.int_)).astype(jnp.int_)
  diag = jnp.zeros(dofs.shape).flatten()
  diag = diag.at[indices.flatten()].add(diagonal_contributions.flatten())
  return diag

# ToDo: Try row-wise assembling with automatic treatment of duplicates (maybe with sparsejac and pre-calculation of sparsity pattern with ints)
@jit_with_docstring(static_argnames=['static_settings', 'set'])
def sparse_assemble_tangent(dofs, settings, static_settings, set):
  """
  Sparse assembly of the full tangent matrix of specified set.
  
  Parameters:
    dofs (jnp.ndarray): Degrees of freedom.
    settings (dict): Settings for the integration.
    static_settings (flax.core.FrozenDict): Static settings including assembling modes.
    set (int): The set index.
  
  Returns:
    jax.experimental.sparse.BCOO: The assembled tangent matrix.
  """
  neighbor_list = jnp.asarray(static_settings['connectivity'][set])
  local_dofs = dofs[neighbor_list]
  maximum = static_settings['maximal number of neighbors'][set]
  n_fields = static_settings['number of fields'][set]
  variational_scheme = static_settings['variational scheme'][set]
  x_int = settings['integration coordinates'][set]
  w_int = settings['integration weights'][set]

  # Compute tangent contributions
  if variational_scheme == 'least square pde loss' or variational_scheme == 'least square function approximation':
    tangent_at_int_point = jax.jacfwd(jax.jacrev(variational_schemes.functional_at_int_point, argnums=3), argnums=3) # argnum=3 are unknowns with local support
  elif variational_scheme == 'strong form galerkin':
    tangent_at_int_point = jax.jacfwd(variational_schemes.direct_residual_at_int_point, argnums=3)
  elif variational_scheme == 'weak form galerkin':
    tangent_at_int_point = jax.jacfwd(variational_schemes.residual_from_deriv_at_int_point, argnums=3)
  else:
    assert False, 'Residual mode not or wrongly specified!'
    
  int_point_numbers = jnp.arange(0, x_int.shape[0], 1)
  if variational_scheme == 'weak form galerkin':
    tangent_at_int_point_vj = jax.vmap(tangent_at_int_point, (0,0,0,0,0,None,None,None), 0)
    tangent_contributions = tangent_at_int_point_vj(x_int, w_int, int_point_numbers, local_dofs, local_dofs, settings, static_settings, set)
  else:
    tangent_at_int_point_vj = jax.vmap(tangent_at_int_point, (0,0,0,0,None,None,None), 0)
    tangent_contributions = tangent_at_int_point_vj(x_int, w_int, int_point_numbers, local_dofs, settings, static_settings, set)
  tangent_contributions = jnp.reshape(tangent_contributions, (x_int.shape[0], maximum*n_fields, maximum*n_fields))

  # Assembling (without summing duplicates)
  def get_indices(list):
    def one_elem_idx(neighb):
      def row_indices(a, row):
        return jnp.asarray([[a,row[i]] for i in range(row.shape[0])])
      colrow = jax.vmap(row_indices, (0, None), 0)
      tmp = jnp.outer(jnp.ones(neighb.shape, dtype=jnp.int_), jnp.arange(n_fields, dtype=jnp.int_)).astype(jnp.int_)
      indices = tmp + jnp.outer(neighb, n_fields * jnp.ones(n_fields, dtype=jnp.int_)).astype(jnp.int_)
      flat_indices = indices.flatten()
      return colrow(flat_indices,flat_indices)
    all_elem_idx = jax.vmap(one_elem_idx, (0), 0)
    return jnp.reshape(all_elem_idx(list), shape=(list.shape[0]*(maximum*n_fields)**2,2))
  indices = get_indices(neighbor_list)
  data = tangent_contributions.flatten()
  n_dofs = n_fields * dofs.shape[0]

  tangent = sparse.BCOO((data, indices), shape=(n_dofs,n_dofs))
  return tangent


### Assembling for user potentials
@jit_with_docstring(static_argnames=['static_settings', 'set'])
def user_potential_integrate_functional(dofs, settings, static_settings, set):
  """
  Assembly of potential for custom user definition of specified set.
  
  Parameters:
    dofs (jnp.ndarray): Degrees of freedom.
    settings (dict): Settings for the integration.
    static_settings (flax.core.FrozenDict): Static settings including assembling modes.
    set (int): The set index.
  
  Returns:
    float: value of functional integrated over set of elements
  """

  user_elem_fun = static_settings['model'][set]
  connectivity_list = jnp.asarray(static_settings['connectivity'][set])
  x_nodes = settings['node coordinates']
  local_dofs = dofs[connectivity_list]
  local_node_coor = x_nodes[connectivity_list]
  elem_numbers = jnp.arange(0, connectivity_list.shape[0], 1)

  # For each element call user element routine (as the one in pde-module with mode='residual') and get contributions
  functional_contributions = jax.vmap(user_elem_fun, (0, 0, 0, None, None, None), (0)
                                      )(local_dofs, local_node_coor, elem_numbers, settings, static_settings, set)

  return functional_contributions.sum()

@jit_with_docstring(static_argnames=['static_settings', 'set'])
def user_potential_assemble_residual(dofs, settings, static_settings, set):
  """
  Assembly of residual for custom user potential of specified set.
  
  Parameters:
    dofs (jnp.ndarray): Degrees of freedom.
    settings (dict): Settings for the integration.
    static_settings (flax.core.FrozenDict): Static settings including assembling modes.
    set (int): The set index.
  
  Returns:
    jnp.ndarray: The assembled residual.
  """
  n_fields = static_settings['number of fields'][set]
  user_potential_fun = static_settings['model'][set]
  connectivity_list = jnp.asarray(static_settings['connectivity'][set])
  x_nodes = settings['node coordinates']
  local_dofs = dofs[connectivity_list]
  local_node_coor = x_nodes[connectivity_list]
  elem_numbers = jnp.arange(0, connectivity_list.shape[0], 1)

  # For each element call user residual routine (as the one in pde-module with mode='residual') and get contributions
  residual_contributions = jax.vmap(jax.jacrev(user_potential_fun), (0, 0, 0, None, None, None), (0))(local_dofs, local_node_coor, elem_numbers, settings, static_settings, set)

  # Assembling
  flat_neighbors = connectivity_list.flatten()
  tmp = jnp.outer(jnp.ones(flat_neighbors.shape, dtype=jnp.int_), jnp.arange(n_fields, dtype=jnp.int_))
  indices = tmp + jnp.outer(flat_neighbors, n_fields * jnp.ones(n_fields, dtype=jnp.int_)).astype(jnp.int_)

  res = jnp.zeros(dofs.shape).flatten()
  res = res.at[indices.flatten()].add(residual_contributions.flatten())
  res = res.reshape(dofs.shape)
  res = jnp.reshape(res, dofs.shape)
  return res

@jit_with_docstring(static_argnames=['static_settings', 'set'])
def user_potential_assemble_tangent_diagonal(dofs, settings, static_settings, set):
  """
  Assembly of the diagonal of the tangent matrix for custom user potential of specified set.
  
  Parameters:
    dofs (jnp.ndarray): Degrees of freedom.
    settings (dict): Settings for the integration.
    static_settings (flax.core.FrozenDict): Static settings including assembling modes.
    set (int): The set index.
  
  Returns:
    jnp.ndarray: The diagonal of the assembled tangent matrix.
  """
  n_fields = static_settings['number of fields'][set]
  user_potential_fun = static_settings['model'][set]
  connectivity_list = jnp.asarray(static_settings['connectivity'][set])
  x_nodes = settings['node coordinates']
  local_dofs = dofs[connectivity_list]
  local_node_coor = x_nodes[connectivity_list]
  elem_numbers = jnp.arange(0, connectivity_list.shape[0], 1)

  # For each element call user element routine (as the one in pde-module with mode='residual') and get contributions
  tangent_contributions = jax.vmap(jax.jacfwd(jax.jacrev(user_potential_fun)), (0, 0, 0, None, None, None), (0))(local_dofs, local_node_coor, elem_numbers, settings, static_settings, set)
  diagonal_contributions = jnp.diagonal(tangent_contributions, axis1=1, axis2=2)

  # Assemble diagonal entries like a residual
  flat_neighbors = connectivity_list.flatten()
  tmp = jnp.outer(jnp.ones(flat_neighbors.shape, dtype=jnp.int_), jnp.arange(n_fields, dtype=jnp.int_))
  indices = tmp + jnp.outer(flat_neighbors, n_fields * jnp.ones(n_fields, dtype=jnp.int_)).astype(jnp.int_)
  diag = jnp.zeros(dofs.shape).flatten()
  diag = diag.at[indices.flatten()].add(diagonal_contributions.flatten())
  return diag

@jit_with_docstring(static_argnames=['static_settings', 'set'])
def user_potential_assemble_tangent(dofs, settings, static_settings, set):
  """
  Assembly of the full (sparse) tangent matrix for custom user potential of specified set.
  
  Parameters:
    dofs (jnp.ndarray): Degrees of freedom.
    settings (dict): Settings for the integration.
    static_settings (flax.core.FrozenDict): Static settings including assembling modes.
    set (int): The set index.
  
  Returns:
    jax.experimental.sparse.BCOO: The assembled tangent matrix.
  """
  n_fields = static_settings['number of fields'][set]
  user_potential_fun = static_settings['model'][set]
  connectivity_list = jnp.asarray(static_settings['connectivity'][set])
  x_nodes = settings['node coordinates']
  local_dofs = dofs[connectivity_list]
  local_node_coor = x_nodes[connectivity_list]
  dofs_per_elem = local_dofs[0].flatten().shape[0]
  elem_numbers = jnp.arange(0, connectivity_list.shape[0], 1)

  # For each element call user element routine (as the one in pde-module with mode='residual') and get contributions
  tangent_contributions = jax.vmap(jax.jacfwd(jax.jacrev(user_potential_fun)), (0, 0, 0, None, None, None), (0)
                                   )(local_dofs, local_node_coor, elem_numbers, settings, static_settings, set)

  # Assembling (without summing duplicates)
  def get_indices(list):
    def one_elem_idx(neighb):
      def row_indices(a, row):
        return jnp.asarray([[a,row[i]] for i in range(row.shape[0])])
      colrow = jax.vmap(row_indices, (0, None), 0)
      tmp = jnp.outer(jnp.ones(neighb.shape, dtype=jnp.int_), jnp.arange(n_fields, dtype=jnp.int_)).astype(jnp.int_)
      indices = tmp + jnp.outer(neighb, n_fields * jnp.ones(n_fields, dtype=jnp.int_)).astype(jnp.int_)
      flat_indices = indices.flatten()
      return colrow(flat_indices,flat_indices)
    all_elem_idx = jax.vmap(one_elem_idx, (0), 0)
    return jnp.reshape(all_elem_idx(list), shape=(list.shape[0]*(dofs_per_elem)**2,2))
  
  indices = get_indices(connectivity_list)
  data = tangent_contributions.flatten()
  n_dofs = n_fields * dofs.shape[0]

  tangent = sparse.BCOO((data, indices), shape=(n_dofs,n_dofs))
  return tangent


### Assembling for user residuals
@jit_with_docstring(static_argnames=['static_settings', 'set'])
def user_residual_assemble_residual(dofs, settings, static_settings, set):
  """
  Assembly of residual for custom user residual of specified set.
  
  Parameters:
    dofs (jnp.ndarray): Degrees of freedom.
    settings (dict): Settings for the integration.
    static_settings (flax.core.FrozenDict): Static settings including assembling modes.
    set (int): The set index.
  
  Returns:
    jnp.ndarray: The assembled residual.
  """
  n_fields = static_settings['number of fields'][set]
  user_residual_fun = static_settings['model'][set]
  connectivity_list = jnp.asarray(static_settings['connectivity'][set])
  x_nodes = settings['node coordinates']
  local_dofs = dofs[connectivity_list]
  local_node_coor = x_nodes[connectivity_list]
  elem_numbers = jnp.arange(0, connectivity_list.shape[0], 1)

  # For each element call user residual routine (as the one in pde-module with mode='residual') and get contributions
  residual_contributions = jax.vmap(user_residual_fun, (0, 0, 0, None, None, None), (0))(local_dofs, local_node_coor, elem_numbers, settings, static_settings, set)

  # Assembling
  flat_neighbors = connectivity_list.flatten()
  tmp = jnp.outer(jnp.ones(flat_neighbors.shape, dtype=jnp.int_), jnp.arange(n_fields, dtype=jnp.int_))
  indices = tmp + jnp.outer(flat_neighbors, n_fields * jnp.ones(n_fields, dtype=jnp.int_)).astype(jnp.int_)

  res = jnp.zeros(dofs.shape).flatten()
  res = res.at[indices.flatten()].add(residual_contributions.flatten())
  res = res.reshape(dofs.shape)
  res = jnp.reshape(res, dofs.shape)
  return res

@jit_with_docstring(static_argnames=['static_settings', 'set'])
def user_residual_assemble_tangent_diagonal(dofs, settings, static_settings, set):
  """
  Assembly of the diagonal of the tangent matrix for custom user residual of specified set.
  
  Parameters:
    dofs (jnp.ndarray): Degrees of freedom.
    settings (dict): Settings for the integration.
    static_settings (flax.core.FrozenDict): Static settings including assembling modes.
    set (int): The set index.
  
  Returns:
    jnp.ndarray: The diagonal of the assembled tangent matrix.
  """
  n_fields = static_settings['number of fields'][set]
  user_residual_fun = static_settings['model'][set]
  connectivity_list = jnp.asarray(static_settings['connectivity'][set])
  x_nodes = settings['node coordinates']
  local_dofs = dofs[connectivity_list]
  local_node_coor = x_nodes[connectivity_list]
  elem_numbers = jnp.arange(0, connectivity_list.shape[0], 1)

  # For each element call user element routine (as the one in pde-module with mode='residual') and get contributions
  tangent_contributions = jax.vmap(jax.jacfwd(user_residual_fun), (0, 0, 0, None, None, None), (0))(local_dofs, local_node_coor, elem_numbers, settings, static_settings, set)
  diagonal_contributions = jnp.diagonal(tangent_contributions, axis1=1, axis2=2)

  # Assemble diagonal entries like a residual
  flat_neighbors = connectivity_list.flatten()
  tmp = jnp.outer(jnp.ones(flat_neighbors.shape, dtype=jnp.int_), jnp.arange(n_fields, dtype=jnp.int_))
  indices = tmp + jnp.outer(flat_neighbors, n_fields * jnp.ones(n_fields, dtype=jnp.int_)).astype(jnp.int_)
  diag = jnp.zeros(dofs.shape).flatten()
  diag = diag.at[indices.flatten()].add(diagonal_contributions.flatten())
  return diag

@jit_with_docstring(static_argnames=['static_settings', 'set'])
def user_residual_assemble_tangent(dofs, settings, static_settings, set):
  """
  Assembly of the full (sparse) tangent matrix for custom user residual of specified set.
  
  Parameters:
    dofs (jnp.ndarray): Degrees of freedom.
    settings (dict): Settings for the integration.
    static_settings (flax.core.FrozenDict): Static settings including assembling modes.
    set (int): The set index.
  
  Returns:
    jax.experimental.sparse.BCOO: The assembled tangent matrix.
  """
  n_fields = static_settings['number of fields'][set]
  user_residual_fun = static_settings['model'][set]
  connectivity_list = jnp.asarray(static_settings['connectivity'][set])
  x_nodes = settings['node coordinates']
  local_dofs = dofs[connectivity_list]
  local_node_coor = x_nodes[connectivity_list]
  dofs_per_elem = local_dofs[0].flatten().shape[0]
  elem_numbers = jnp.arange(0, connectivity_list.shape[0], 1)

  # For each element call user element routine (as the one in pde-module with mode='residual') and get contributions
  tangent_contributions = jax.vmap(jax.jacfwd(user_residual_fun), (0, 0, 0, None, None, None), (0)
                                   )(local_dofs, local_node_coor, elem_numbers, settings, static_settings, set)

  # Assembling (without summing duplicates)
  def get_indices(list):
    def one_elem_idx(neighb):
      def row_indices(a, row):
        return jnp.asarray([[a,row[i]] for i in range(row.shape[0])])
      colrow = jax.vmap(row_indices, (0, None), 0)
      tmp = jnp.outer(jnp.ones(neighb.shape, dtype=jnp.int_), jnp.arange(n_fields, dtype=jnp.int_)).astype(jnp.int_)
      indices = tmp + jnp.outer(neighb, n_fields * jnp.ones(n_fields, dtype=jnp.int_)).astype(jnp.int_)
      flat_indices = indices.flatten()
      return colrow(flat_indices,flat_indices)
    all_elem_idx = jax.vmap(one_elem_idx, (0), 0)
    return jnp.reshape(all_elem_idx(list), shape=(list.shape[0]*(dofs_per_elem)**2,2))
  
  indices = get_indices(connectivity_list)
  data = tangent_contributions.flatten()
  n_dofs = n_fields * dofs.shape[0]

  tangent = sparse.BCOO((data, indices), shape=(n_dofs,n_dofs))
  return tangent


### Assembling for user elements
@jit_with_docstring(static_argnames=['static_settings', 'set'])
def user_element_assemble_residual(dofs, settings, static_settings, set):
  """
  Assembly of residual for custom user element of specified set.
  
  Parameters:
    dofs (jnp.ndarray): Degrees of freedom.
    settings (dict): Settings for the integration.
    static_settings (flax.core.FrozenDict): Static settings including assembling modes.
    set (int): The set index.
  
  Returns:
    jnp.ndarray: The assembled residual.
  """

  n_fields = static_settings['number of fields'][set]
  user_elem_fun = static_settings['model'][set]
  connectivity_list = jnp.asarray(static_settings['connectivity'][set])
  x_nodes = settings['node coordinates']
  local_dofs = dofs[connectivity_list]
  local_node_coor = x_nodes[connectivity_list]
  elem_numbers = jnp.arange(0, connectivity_list.shape[0], 1)


  # For each element call user element routine (as the one in pde-module with mode='residual') and get contributions
  residual_contributions = jax.vmap(user_elem_fun, (0, 0, 0, None, None, None, None), (0))(local_dofs, local_node_coor, elem_numbers, settings, static_settings, 'residual', set)

  # Assembling
  flat_neighbors = connectivity_list.flatten()
  tmp = jnp.outer(jnp.ones(flat_neighbors.shape, dtype=jnp.int_), jnp.arange(n_fields, dtype=jnp.int_))
  indices = tmp + jnp.outer(flat_neighbors, n_fields * jnp.ones(n_fields, dtype=jnp.int_)).astype(jnp.int_)

  res = jnp.zeros(dofs.shape).flatten()
  res = res.at[indices.flatten()].add(residual_contributions.flatten())
  res = res.reshape(dofs.shape)
  res = jnp.reshape(res, dofs.shape)

  return res

@jit_with_docstring(static_argnames=['static_settings', 'set'])
def user_element_assemble_tangent_diagonal(dofs, settings, static_settings, set):
  """
  Assembly of the diagonal of the tangent matrix for custom user element of specified set.
  
  Parameters:
    dofs (jnp.ndarray): Degrees of freedom.
    settings (dict): Settings for the integration.
    static_settings (flax.core.FrozenDict): Static settings including assembling modes.
    set (int): The set index.
  
  Returns:
    jnp.ndarray: The diagonal of the assembled tangent matrix.
  """

  n_fields = static_settings['number of fields'][set]
  user_elem_fun = static_settings['model'][set]
  connectivity_list = jnp.asarray(static_settings['connectivity'][set])
  x_nodes = settings['node coordinates']
  local_dofs = dofs[connectivity_list]
  local_node_coor = x_nodes[connectivity_list]
  elem_numbers = jnp.arange(0, connectivity_list.shape[0], 1)

  # For each element call user element routine (as the one in pde-module with mode='residual') and get contributions
  tangent_contributions = jax.vmap(user_elem_fun, (0, 0, 0, None, None, None, None), (0))(local_dofs, local_node_coor, elem_numbers, settings, static_settings, 'tangent', set)
  diagonal_contributions = jnp.diagonal(tangent_contributions, axis1=1, axis2=2)

  # Assemble diagonal entries like a residual
  flat_neighbors = connectivity_list.flatten()
  tmp = jnp.outer(jnp.ones(flat_neighbors.shape, dtype=jnp.int_), jnp.arange(n_fields, dtype=jnp.int_))
  indices = tmp + jnp.outer(flat_neighbors, n_fields * jnp.ones(n_fields, dtype=jnp.int_)).astype(jnp.int_)
  diag = jnp.zeros(dofs.shape).flatten()
  diag = diag.at[indices.flatten()].add(diagonal_contributions.flatten())
  return diag

@jit_with_docstring(static_argnames=['static_settings', 'set'])
def user_element_assemble_tangent(dofs, settings, static_settings, set):
  """
  Assembly of the full (sparse) tangent matrix for custom user element of specified set.
  
  Parameters:
    dofs (jnp.ndarray): Degrees of freedom.
    settings (dict): Settings for the integration.
    static_settings (flax.core.FrozenDict): Static settings including assembling modes.
    set (int): The set index.
  
  Returns:
    jax.experimental.sparse.BCOO: The assembled tangent matrix.
  """
  
  n_fields = static_settings['number of fields'][set]
  user_elem_fun = static_settings['model'][set]
  connectivity_list = jnp.asarray(static_settings['connectivity'][set])
  x_nodes = settings['node coordinates']
  local_dofs = dofs[connectivity_list]
  local_node_coor = x_nodes[connectivity_list]
  dofs_per_elem = local_dofs[0].flatten().shape[0]
  elem_numbers = jnp.arange(0, connectivity_list.shape[0], 1)

  # For each element call user element routine (as the one in pde-module with mode='residual') and get contributions
  tangent_contributions = jax.vmap(user_elem_fun, (0, 0, 0, None, None, None, None), (0)
                                   )(local_dofs, local_node_coor, elem_numbers, settings, static_settings, 'tangent', set)

  # Assembling (without summing duplicates)
  def get_indices(list):
    def one_elem_idx(neighb):
      def row_indices(a, row):
        return jnp.asarray([[a,row[i]] for i in range(row.shape[0])])
      colrow = jax.vmap(row_indices, (0, None), 0)
      tmp = jnp.outer(jnp.ones(neighb.shape, dtype=jnp.int_), jnp.arange(n_fields, dtype=jnp.int_)).astype(jnp.int_)
      indices = tmp + jnp.outer(neighb, n_fields * jnp.ones(n_fields, dtype=jnp.int_)).astype(jnp.int_)
      flat_indices = indices.flatten()
      return colrow(flat_indices,flat_indices)
    all_elem_idx = jax.vmap(one_elem_idx, (0), 0)
    return jnp.reshape(all_elem_idx(list), shape=(list.shape[0]*(dofs_per_elem)**2,2))
  
  indices = get_indices(connectivity_list)
  data = tangent_contributions.flatten()
  n_dofs = n_fields * dofs.shape[0]

  tangent = sparse.BCOO((data, indices), shape=(n_dofs,n_dofs))
  return tangent
