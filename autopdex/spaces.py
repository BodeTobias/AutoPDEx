# spaces.py
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
Definition of different kinds of solution spaces, including

- Moving Least Squares (MLS) methods
- Simplex-shaped finite elements with shape functions defined via least squares
- Isoparametric line, quadrilateral and brick shape functions for user elements

Supports both direct and compiled modes (with precomputation of discrete shape functions 
and derivatives) for MLS and simplex-shaped elements.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

from autopdex.utility import jit_with_docstring


@jit_with_docstring(static_argnames=['static_settings', 'set'])
def solution_space(x, int_point_number, local_dofs, settings, static_settings, set):
  """
  Compute the solution space for a given integration point and local degrees of freedom.

  This function determines the type of solution space based on the provided settings
  and computes it accordingly. The supported types of solution spaces include 
  moving least squares (mls), finite element simplices (fem simplex), nodal values, 
  and user-defined solution spaces.

  Parameters:
    x (jnp.ndarray): The coordinates of the evaluation point.
    int_point_number (int): The index of the integration point.
    local_dofs (jnp.ndarray): The local degrees of freedom.
    settings (dict): A dictionary containing various settings required for the computation.
    static_settings (dict): A dictionary containing static settings that define the solution space and other parameters.
    set (int): The index of the current set of settings being used.

  Returns:
    jnp.ndarray: The computed solution space value or shape functions at the evaluation point.
  """
  space_type = static_settings['solution space'][set]
  if space_type == 'mls':
    beta = settings['beta'][set]
    x_nodes = settings['node coordinates']
    neighbor_list = jnp.asarray(static_settings['connectivity'][set])
    support_radius = settings['support radius'][set]
    x_local_nodes = x_nodes[neighbor_list[int_point_number]]
    return moving_least_squares(x, x_local_nodes, local_dofs, beta, support_radius, static_settings, set)
  elif space_type == 'fem simplex':
    x_nodes = settings['node coordinates']
    connectivity_list = jnp.asarray(static_settings['connectivity'][set])
    x_local_nodes = x_nodes[connectivity_list[int_point_number]]
    return fem_ini_simplex(x, x_local_nodes, local_dofs, static_settings, set)
  elif space_type == 'nodal values':
    mode = static_settings['shape function mode']
    if mode == 'direct':
      return local_dofs
    elif mode == 'compiled':
      return jnp.asarray([1.])
  elif space_type == 'user':
    return static_settings['user solution space function'][set](x, int_point_number, local_dofs, settings, static_settings, set)
  else:
    assert False, 'Solution space not defined!'

### Shape function definitions

@jit_with_docstring(static_argnames=['static_settings', 'set'])
def moving_least_squares(x, xI, fI, beta, support_radius, static_settings, set):
  """
  Compute the moving least squares (MLS) approximation for a given set of points and data.

  Parameters:
    x (jnp.ndarray): The position of the evaluation point.
    xI (jnp.ndarray): The positions of neighboring nodes.
    fI (jnp.ndarray): The data at neighboring nodes.
    beta (float): The hyperparameter for smoothness, typically in the range [3, 5].
    support_radius (float): The radius within which neighboring nodes are considered.
    static_settings (dict): Dictionary containing static settings that define the solution space and other parameters.
        Keywords used:
        - 'order of basis functions': Order of polynomial basis functions.
        - 'shape function mode': Mode of shape function computation ('direct' or 'compiled').
        - 'weight function type': Type of weight function ('gaussian', 'bump', 'gaussian perturbed kronecker', 'bump perturbed kronecker').
    set (int): The index of the current set of settings being used.
    settings (dict): Dictionary containing dynamic settings.
        Keywords used:
        - 'beta': Hyperparameter for smoothness.
        - 'node coordinates': Coordinates of nodes.
        - 'connectivity': Connectivity information of integration points with respect to nodes.
        - 'support radius': Support radius for weight function.

  Returns:
    jnp.ndarray: 
      The computed MLS approximation at the evaluation point, either as shape functions or the evaluated function, depending on wether the compiled mode or direct mode is chosen.
  """
  order = static_settings['order of basis functions'][set]
  n_dim = x.shape[0]
  mode = static_settings['shape function mode']
  basis_length = _compute_poly_basis_length(n_dim, order)

  # Initial coefficients
  a_0 = jnp.zeros(basis_length)

  # Radial weigth function, smooth Gauß Kernel
  def weight_function(r_squared):
    scaled_r_squared = r_squared / (support_radius**2)
    cond = jnp.where(scaled_r_squared < 1, 1, 0)
    
    weight_function_type = static_settings['weight function type'][set]
    if weight_function_type == 'gaussian':
      return (jnp.exp(-beta**2 * scaled_r_squared) - jnp.exp(-beta**2))/(1 - jnp.exp(-beta**2)) * cond # Gauss kernel
    elif weight_function_type == 'bump':
      return (jnp.exp(beta**2 * scaled_r_squared / (scaled_r_squared - 1.))) * cond # Bump function
    elif weight_function_type == 'gaussian perturbed kronecker':
      return ((jnp.exp(-beta**2 * scaled_r_squared) - jnp.exp(-beta**2))/(1 - jnp.exp(-beta**2)) / (jnp.sqrt(scaled_r_squared) + 1e-6)) * cond # Gauss kernel
    elif weight_function_type == 'bump perturbed kronecker':
      return ((jnp.exp(beta**2 * scaled_r_squared / (scaled_r_squared - 1.))) / (jnp.sqrt(scaled_r_squared) + 1e-6)) * cond # Bump function
    else:
      assert False, 'Weight function type has to be either \'gaussian\', \'bump\', \'gaussian perturbed kronecker\' or \'bump perturbed kronecker\'.'

  # Ansatz of approximation, evaluated at shifted and scaled xi
  def polynomial_ansatz(a, xi):
    return jnp.dot(a, _polynomial_basis((xi-x)/support_radius, order))

  # Squared error at node i
  def squared_error(a, xi, fi):
    e = polynomial_ansatz(a, xi) - fi
    return e**2

  # Weighted squared error at node i
  def weighted_squared_error(a, xi, fi):
    d = x - xi
    r_squared = jnp.dot(d, d)
    w = weight_function(r_squared)
    e2 = squared_error(a, xi, fi)
    return w * e2

  # Summing weighted squared errors
  def mls_error_functional(a, fI0):
    weighted_squared_error_vmap = jax.vmap(weighted_squared_error, (None, 0, 0), 0)
    eWLS = weighted_squared_error_vmap(a, xI, fI0).sum() # For computation of shape functions one field is sufficient
    return eWLS

  # Compute coefficients via one Newton step
  def compute_mls(fI0):
    residual = jax.jacrev(mls_error_functional)
    tangent = jax.jacfwd(residual)
    residual_0 = residual(a_0, fI0)
    tangent_0 = tangent(a_0, fI0)

    chol, lower = jax.scipy.linalg.cho_factor(tangent_0)
    a_mls = - jax.scipy.linalg.cho_solve((chol, lower), residual_0)
    return polynomial_ansatz(a_mls, x)

  # Filter the shape functions
  tmpI = jnp.ones(fI.shape[0])
  shape_functions = jax.jacrev(compute_mls)(tmpI)

  if mode == 'direct':
    # Use shape functions for all fields
    return jnp.dot(shape_functions, fI)
  elif mode == 'compiled':
    return shape_functions
  else:
    assert False, 'Wrong mode of shape function computation'

@jit_with_docstring(static_argnames=['static_settings', 'set'])
def fem_ini_simplex(x, xI, fI, static_settings, set):
  """
  Compute finite element shape functions directly in the initial/physical configuration.

  Parameters:
    x (jnp.ndarray): The position of the evaluation point.
    xI (jnp.ndarray): The positions of neighboring nodes.
    fI (jnp.ndarray): The data at neighboring nodes.
    static_settings (dict): Dictionary containing static settings that define the solution space and other parameters.
        Keywords used:
        - 'shape function mode': Mode of shape function computation ('direct' or 'compiled').
    set (int): The index of the current set of settings being used.

  Returns:
    jnp.ndarray: The computed finite element shape functions at the evaluation point, either as shape functions or the evaluated function.

  Notes:
    - This method computes the polynomial order based on the number of nodes per element.
    - The method supports different dimensions and orders for the polynomial basis functions.
  """
  mode = static_settings['shape function mode']
  n_dim = xI.shape[-1]
  n_nodes = xI.shape[0]
  match n_dim: # Compute the polynomial order based on the number of nodes per element
    case 1:
      order = n_nodes -1
    case 2: 
      match n_nodes:
        case 3: 
          order = 1
        case 6:
          order = 2
        case _:
          assert False, "Order of shape functions not implemented or number of nodes not adequat"
    case 3:
      match n_nodes:
        case 4: 
          order = 1
        case 10:
          order = 2
        case _:
          assert False, "Order of shape functions not implemented or number of nodes not adequat"
    case _:
      order = 1
      assert n_nodes==n_dim+1, "Order of shape functions not implemented or number of nodes not adequat"
  basis_length = n_nodes

  # Initial coefficients
  a_0 = jnp.zeros(basis_length)

  # Ansatz of approximation, evaluated at shifted and scaled xi
  xc = jnp.mean(xI, axis=0)
  scaling_length = jnp.mean(xI-xc, axis=0)
  def polynomial_ansatz(a, xi):
    scaling_length = 1
    return jnp.dot(a, _polynomial_basis((xi-x)/scaling_length, order))

  # Squared error at node i
  def squared_error(a, xi, fi):
    e = polynomial_ansatz(a, xi) - fi
    return e**2

  # Summing weighted squared errors
  def error_functional(a, fI0):
    squared_error_vmap = jax.vmap(squared_error, (None, 0, 0), 0)
    eWLS = squared_error_vmap(a, xI, fI0).sum() # For computation of shape functions one field is sufficient
    return eWLS

  # Compute coefficients via one Newton step
  def compute_ls(fI0):
    residual = jax.jacrev(error_functional)
    tangent = jax.jacfwd(residual)
    residual_0 = residual(a_0, fI0)
    tangent_0 = tangent(a_0, fI0)
  
    chol, lower = jax.scipy.linalg.cho_factor(tangent_0)
    a_mls = - jax.scipy.linalg.cho_solve((chol, lower), residual_0)
    return polynomial_ansatz(a_mls, x)

  # Filter the shape functions
  tmpI = jnp.ones(fI.shape[0])
  shape_functions = jax.jacrev(compute_ls)(tmpI)
  
  if mode == 'direct':
    # Use shape functions for all fields
    return jnp.dot(shape_functions, fI)
  elif mode == 'compiled':
    return shape_functions
  else:
    assert False, 'Wrong mode of shape function computation'

def fem_iso_line_quad_brick(x, xI, fI, settings, overwrite_diff, n_dim):
  """
  Compute finite element shape functions for line, quadrilateral, and brick elements in the initial/physical configuration.

  Parameters:
    x (jnp.ndarray): The position of the evaluation point.
    xI (jnp.ndarray): The positions of neighboring nodes.
    fI (jnp.ndarray): The data at neighboring nodes.
    settings (dict): Dictionary containing various settings (not directly used in this function but passed for compatibility).
    overwrite_diff (bool): If True, overwrites the derivative to be with respect to the initial configuration instead of the reference configuration.
    n_dim (int): The dimensionality of the elements (1 for line, 2 for quadrilateral, 3 for brick).

  Returns:
    float:
      The computed finite element approximation (sum_i shape_fun_i nodal_values_i)

  Notes:
    - This function currently supports line elements up to order 20, quadrilateral elements up to order 20, and brick elements up to order 2.
    - Though the input `x` is a reference coordinate, its derivative is replaced with respect to the initial configuration when `overwrite_diff` is True.
    - Warning: Only first-order derivatives are supported in the custom JVP implementation with overwritten derivatives.
    - Warning: The derivatives with respect to xI are set to zero.
  """
  n_nodes = xI.shape[0]
  match n_dim:
    case 1:
      """ The following shape functions were generated using the code below:

      # Line elements

      import numpy as np
      import sympy as sp

      for n in range(1,21):
          #n = 8  # Order of shape functions

          print("case " + str((n+1)) + ":")

          # Lagrange polynomials
          x = sp.Symbol('x')
          xI = np.concatenate(([-1, 1], np.array([-1 + 2 * sp.Rational(i, n) for i in range(1, n)])))
          polys = [x - xi for xi in xI]
          denom = sp.prod(polys)
          num = [denom / poly for poly in polys]
          numI = [num[i].subs(x, xI[i]) for i in range(len(num))]
          Ln = [sp.simplify(num[i] / numI[i]).subs(x, sp.Symbol('xi')).n(20) for i in range(len(num))]

          # Common subexpression elimination
          subexpr, reduced_expr = sp.cse(Ln)
          utility.to_jax_function(subexpr, reduced_expr)
      """
      match n_nodes:
        case 2:
          def shape_functions(xi):
            x0 = 0.5*xi
            return jnp.asarray([0.5 - x0, x0 + 0.5])

        case 3:
          def shape_functions(xi):
            x0 = 0.5*xi
            return jnp.asarray([x0*(xi - 1.0), x0*(xi + 1.0), 1.0 - xi**2])

        case 4:
          def shape_functions(xi):
            x0 = 3.0*xi
            x1 = x0 + 1.0
            x2 = xi + 1.0
            x3 = x2*(x0 - 1.0)
            x4 = 0.5625*xi - 0.5625
            return jnp.asarray([-0.5625*xi**3 + 0.5625*xi**2 + 0.0625*xi - 0.0625, 0.0625*x1*x3, x3*x4, -x1*x2*x4])

        case 5:
          def shape_functions(xi):
            x0 = xi - 1.0
            x1 = 2.0*xi
            x2 = x1 - 1.0
            x3 = x0*x2*xi
            x4 = x1 + 1.0
            x5 = 0.16666666666666666667*x4
            x6 = xi + 1.0
            x7 = x6*xi
            return jnp.asarray([x3*x5, x2*x5*x7, -1.3333333333333333333*x3*x6, 4.0*xi**4 - 5.0*xi**2 + 1.0, -1.3333333333333333333*x0*x4*x7])

        case 6:
          def shape_functions(xi):
            x0 = 5.0*xi
            x1 = x0 + 1.0
            x2 = xi - 1.0
            x3 = x0 - 1.0
            x4 = x0 - 3.0
            x5 = x1*x2*x3*x4
            x6 = x0 + 3.0
            x7 = 0.0013020833333333333333*x6
            x8 = xi + 1.0
            x9 = x3*x4*x8
            x10 = 0.032552083333333333333*x8
            x11 = x2*x6
            x12 = 0.065104166666666666667*x11
            return jnp.asarray([-x5*x7, x1*x7*x9, x10*x5, -x12*x9, x1*x12*x4*x8, -x1*x10*x11*x3])

        case 7:
          def shape_functions(xi):
            x0 = 3.0*xi
            x1 = x0 + 1.0
            x2 = xi - 1.0
            x3 = x0 - 1.0
            x4 = x0 - 2.0
            x5 = x1*x2*x3*x4*xi
            x6 = x0 + 2.0
            x7 = 0.0125*x6
            x8 = xi + 1.0
            x9 = x3*x4*x8*xi
            x10 = 0.225*x8
            x11 = x2*x6
            x12 = 0.5625*x11
            x13 = x1*xi
            return jnp.asarray([x5*x7, x1*x7*x9, -x10*x5, x12*x9, -20.25*xi**6 + 31.5*xi**4 - 12.25*xi**2 + 1.0, x12*x13*x4*x8, -x10*x11*x13*x3])

        case 8:
          def shape_functions(xi):
            x0 = 7.0*xi
            x1 = x0 + 1.0
            x2 = x0 + 3.0
            x3 = xi - 1.0
            x4 = x0 - 1.0
            x5 = x0 - 3.0
            x6 = x0 - 5.0
            x7 = x1*x2*x3*x4*x5*x6
            x8 = x0 + 5.0
            x9 = 0.000010850694444444444444*x8
            x10 = xi + 1.0
            x11 = x1*x10*x4*x5*x6
            x12 = 0.00053168402777777777778*x10
            x13 = x3*x8
            x14 = 0.0015950520833333333333*x13
            x15 = x10*x2*x4*x6
            x16 = x13*x5
            x17 = 0.0026584201388888888889*x16
            x18 = x1*x2
            return jnp.asarray([-x7*x9, x11*x2*x9, x12*x7, -x11*x14, x15*x17, -x10*x17*x18*x6, x1*x14*x15, -x12*x16*x18*x4])

        case 9:
          def shape_functions(xi):
            x0 = 2.0*xi
            x1 = x0 + 1.0
            x2 = 4.0*xi
            x3 = x2 + 1.0
            x4 = xi - 1.0
            x5 = x0 - 1.0
            x6 = x2 - 1.0
            x7 = x2 - 3.0
            x8 = x1*x3*x4*x5*x6*x7*xi
            x9 = x2 + 3.0
            x10 = 0.0015873015873015873016*x9
            x11 = xi + 1.0
            x12 = x11*x3*x5*x6*x7*xi
            x13 = 0.050793650793650793651*x11
            x14 = x4*x9
            x15 = 0.088888888888888888889*x14
            x16 = x1*x11*x6*x7*xi
            x17 = x14*x5
            x18 = 0.35555555555555555556*x17
            x19 = x1*x3*xi
            return jnp.asarray([x10*x8, x1*x10*x12, -x13*x8, x12*x15, -x16*x18, 113.77777777777777778*xi**8 - 213.33333333333333333*xi**6 + 121.33333333333333333*xi**4 - 22.777777777777777778*xi**2 + 1.0, -x11*x18*x19*x7, x15*x16*x3, -x13*x17*x19*x6])

        case 10:
          def shape_functions(xi):
            x0 = 3.0*xi
            x1 = x0 + 1.0
            x2 = 9.0*xi
            x3 = x2 + 1.0
            x4 = x2 + 5.0
            x5 = xi - 1.0
            x6 = x0 - 1.0
            x7 = x2 - 1.0
            x8 = x2 - 5.0
            x9 = x2 - 7.0
            x10 = x1*x3*x4*x5*x6*x7*x8*x9
            x11 = x2 + 7.0
            x12 = 4.3596540178571428571e-7*x11
            x13 = xi + 1.0
            x14 = x1*x13*x3*x6*x7*x8*x9
            x15 = 0.000035313197544642857143*x13
            x16 = x11*x5
            x17 = 0.00014125279017857142857*x16
            x18 = x13*x3*x4*x6*x7*x9
            x19 = x16*x8
            x20 = 0.00010986328125*x19
            x21 = x1*x13*x4*x7*x9
            x22 = x19*x6
            x23 = 0.000494384765625*x22
            x24 = x1*x3*x4
            return jnp.asarray([-x10*x12, x12*x14*x4, x10*x15, -x14*x17, x18*x20, -x21*x23, x13*x23*x24*x9, -x20*x21*x3, x1*x17*x18, -x15*x22*x24*x7])

        case 11:
          def shape_functions(xi):
            x0 = 5.0*xi
            x1 = x0 + 1.0
            x2 = x0 + 2.0
            x3 = x0 + 3.0
            x4 = xi - 1.0
            x5 = x0 - 1.0
            x6 = x0 - 2.0
            x7 = x0 - 4.0
            x8 = x0 - 3.0
            x9 = x1*x2*x3*x4*x5*x6*x7*x8*xi
            x10 = x0 + 4.0
            x11 = 6.8893298059964726631e-6*x10
            x12 = xi + 1.0
            x13 = x1*x12*x2*x5*x6*x7*x8*xi
            x14 = 0.00034446649029982363316*x12
            x15 = x10*x4
            x16 = 0.0015500992063492063492*x15
            x17 = x1*x12*x3*x5*x6*x7*xi
            x18 = x15*x8
            x19 = 0.0041335978835978835979*x18
            x20 = x12*x2*x3*x5*x7*xi
            x21 = x18*x6
            x22 = 0.0072337962962962962963*x21
            x23 = x1*x2*x3*xi
            return jnp.asarray([x11*x9, x11*x13*x3, -x14*x9, x13*x16, -x17*x19, x20*x22, -678.16840277777777778*xi**10 + 1491.9704861111111111*xi**8 - 1110.0260416666666667*xi**6 + 331.81423611111111111*xi**4 - 36.590277777777777778*xi**2 + 1.0, x12*x22*x23*x7, -x1*x19*x20, x16*x17*x2, -x14*x21*x23*x5])

        case 12:
          def shape_functions(xi):
            x0 = 11.0*xi
            x1 = x0 + 1.0
            x2 = x0 + 3.0
            x3 = x0 + 5.0
            x4 = x0 + 7.0
            x5 = xi - 1.0
            x6 = x0 - 1.0
            x7 = x0 - 3.0
            x8 = x0 - 5.0
            x9 = x0 - 7.0
            x10 = x0 - 9.0
            x11 = x1*x10*x2*x3*x4*x5*x6*x7*x8*x9
            x12 = x0 + 9.0
            x13 = 1.345572227733686067e-10*x12
            x14 = xi + 1.0
            x15 = x1*x10*x14*x2*x3*x6*x7*x8*x9
            x16 = 1.6281423955577601411e-8*x14
            x17 = x12*x5
            x18 = 8.1407119777888007055e-8*x17
            x19 = x1*x10*x14*x2*x4*x6*x7*x8
            x20 = x17*x9
            x21 = 2.4422135933366402116e-7*x20
            x22 = x1*x10*x14*x3*x4*x6*x7
            x23 = x20*x8
            x24 = 4.8844271866732804233e-7*x23
            x25 = x10*x14*x2*x3*x4*x6
            x26 = x23*x7
            x27 = 6.8381980613425925926e-7*x26
            x28 = x1*x2*x3*x4
            return jnp.asarray([-x11*x13, x13*x15*x4, x11*x16, -x15*x18, x19*x21, -x22*x24, x25*x27, -x10*x14*x27*x28, x1*x24*x25, -x2*x21*x22, x18*x19*x3, -x16*x26*x28*x6])

        case 13:
          def shape_functions(xi):
            x0 = 2.0*xi
            x1 = x0 + 1.0
            x2 = 3.0*xi
            x3 = x2 + 1.0
            x4 = 6.0*xi
            x5 = x4 + 1.0
            x6 = x2 + 2.0
            x7 = xi - 1.0
            x8 = x0 - 1.0
            x9 = x2 - 1.0
            x10 = x4 - 1.0
            x11 = x2 - 2.0
            x12 = x4 - 5.0
            x13 = x1*x10*x11*x12*x3*x5*x6*x7*x8*x9*xi
            x14 = x4 + 5.0
            x15 = 0.000010822510822510822511*x14
            x16 = xi + 1.0
            x17 = x1*x10*x11*x12*x16*x3*x5*x8*x9*xi
            x18 = 0.00077922077922077922078*x16
            x19 = x14*x7
            x20 = 0.0021428571428571428571*x19
            x21 = x10*x12*x16*x3*x5*x6*x8*x9*xi
            x22 = x11*x19
            x23 = 0.0047619047619047619048*x22
            x24 = x1*x10*x12*x16*x5*x6*x9*xi
            x25 = x22*x8
            x26 = 0.016071428571428571429*x25
            x27 = x1*x10*x12*x16*x3*x6*xi
            x28 = x25*x9
            x29 = 0.051428571428571428571*x28
            x30 = x1*x3*x5*x6*xi
            return jnp.asarray([x13*x15, x15*x17*x6, -x13*x18, x17*x20, -x21*x23, x24*x26, -x27*x29, 4199.04*xi**12 - 10614.24*xi**10 + 9729.72*xi**8 - 4002.57*xi**6 + 740.74*xi**4 - 53.69*xi**2 + 1.0, -x12*x16*x29*x30, x26*x27*x5, -x23*x24*x3, x1*x20*x21, -x10*x18*x28*x30])

        case 14:
          def shape_functions(xi):
            x0 = 13.0*xi
            x1 = x0 + 1.0
            x2 = x0 + 3.0
            x3 = x0 + 5.0
            x4 = x0 + 7.0
            x5 = x0 + 9.0
            x6 = xi - 1.0
            x7 = x0 - 1.0
            x8 = x0 - 3.0
            x9 = x0 - 5.0
            x10 = x0 - 7.0
            x11 = x0 - 9.0
            x12 = x0 - 11.0
            x13 = x1*x10*x11*x12*x2*x3*x4*x5*x6*x7*x8*x9
            x14 = x0 + 11.0
            x15 = 2.5484322494956175512e-13*x14
            x16 = xi + 1.0
            x17 = x1*x10*x11*x12*x16*x2*x3*x4*x7*x8*x9
            x18 = 4.3068505016475936615e-11*x16
            x19 = x14*x6
            x20 = 2.5841103009885561969e-10*x19
            x21 = x1*x10*x12*x16*x2*x3*x5*x7*x8*x9
            x22 = x11*x19
            x23 = 9.4750711036247060553e-10*x22
            x24 = x1*x12*x16*x2*x4*x5*x7*x8*x9
            x25 = x10*x22
            x26 = 2.3687677759061765138e-9*x25
            x27 = x1*x12*x16*x3*x4*x5*x7*x8
            x28 = x25*x9
            x29 = 4.2637819966311177249e-9*x28
            x30 = x12*x16*x2*x3*x4*x5*x7
            x31 = x28*x8
            x32 = 5.6850426621748236332e-9*x31
            x33 = x1*x2*x3*x4*x5
            return jnp.asarray([-x13*x15, x15*x17*x5, x13*x18, -x17*x20, x21*x23, -x24*x26, x27*x29, -x30*x32, x12*x16*x32*x33, -x1*x29*x30, x2*x26*x27, -x23*x24*x3, x20*x21*x4, -x18*x31*x33*x7])

        case 15:
          def shape_functions(xi):
            x0 = 7.0*xi
            x1 = x0 + 1.0
            x2 = x0 + 2.0
            x3 = x0 + 4.0
            x4 = x0 + 3.0
            x5 = x0 + 5.0
            x6 = xi - 1.0
            x7 = x0 - 1.0
            x8 = x0 - 2.0
            x9 = x0 - 4.0
            x10 = x0 - 3.0
            x11 = x0 - 6.0
            x12 = x0 - 5.0
            x13 = x1*x10*x11*x12*x2*x3*x4*x5*x6*x7*x8*x9*xi
            x14 = x0 + 6.0
            x15 = 5.6206653428875651098e-10*x14
            x16 = xi + 1.0
            x17 = x1*x10*x11*x12*x16*x2*x3*x4*x7*x8*x9*xi
            x18 = 5.5082520360298138076e-8*x16
            x19 = x14*x6
            x20 = 3.5803638234193789749e-7*x19
            x21 = x1*x10*x11*x16*x2*x4*x5*x7*x8*x9*xi
            x22 = x12*x19
            x23 = 1.43214552936775159e-6*x22
            x24 = x1*x10*x11*x16*x2*x3*x5*x7*x8*xi
            x25 = x22*x9
            x26 = 3.9384002057613168724e-6*x25
            x27 = x1*x11*x16*x3*x4*x5*x7*x8*xi
            x28 = x10*x25
            x29 = 7.8768004115226337449e-6*x28
            x30 = x11*x16*x2*x3*x4*x5*x7*xi
            x31 = x28*x8
            x32 = 0.000011815200617283950617*x31
            x33 = x1*x2*x3*x4*x5*xi
            return jnp.asarray([x13*x15, x15*x17*x5, -x13*x18, x17*x20, -x21*x23, x24*x26, -x27*x29, x30*x32, -26700.013890817901235*xi**14 + 76285.753973765432099*xi**12 - 82980.21809799382716*xi**10 + 43487.464081790123457*xi**8 - 11465.29836612654321*xi**6 + 1445.3903549382716049*xi**4 - 74.078055555555555556*xi**2 + 1.0, x11*x16*x32*x33, -x1*x29*x30, x2*x26*x27, -x23*x24*x4, x20*x21*x3, -x18*x31*x33*x7])

        case 16:
          def shape_functions(xi):
            x0 = 3.0*xi
            x1 = x0 + 1.0
            x2 = 5.0*xi
            x3 = x2 + 1.0
            x4 = 15.0*xi
            x5 = x4 + 1.0
            x6 = x2 + 3.0
            x7 = x4 + 7.0
            x8 = x4 + 11.0
            x9 = xi - 1.0
            x10 = x0 - 1.0
            x11 = x2 - 1.0
            x12 = x4 - 1.0
            x13 = x2 - 3.0
            x14 = x4 - 7.0
            x15 = x4 - 11.0
            x16 = x4 - 13.0
            x17 = x1*x10*x11*x12*x13*x14*x15*x16*x3*x5*x6*x7*x8*x9
            x18 = x4 + 13.0
            x19 = 7.0887023423470131059e-13*x18
            x20 = xi + 1.0
            x21 = x1*x10*x11*x12*x13*x14*x15*x16*x20*x3*x5*x6*x7
            x22 = 1.5949580270280779488e-10*x20
            x23 = x18*x9
            x24 = 1.1164706189196545642e-9*x23
            x25 = x1*x10*x11*x12*x13*x14*x16*x20*x3*x5*x7*x8
            x26 = x15*x23
            x27 = 1.6126797828839454816e-9*x26
            x28 = x1*x10*x11*x12*x14*x16*x20*x3*x5*x6*x8
            x29 = x13*x26
            x30 = 1.4514118045955509334e-8*x29
            x31 = x10*x11*x12*x16*x20*x3*x5*x6*x7*x8
            x32 = x14*x29
            x33 = 6.3862119402204241071e-9*x32
            x34 = x1*x11*x12*x16*x20*x5*x6*x7*x8
            x35 = x10*x32
            x36 = 1.7739477611723400298e-8*x35
            x37 = x1*x12*x16*x20*x3*x6*x7*x8
            x38 = x11*x35
            x39 = 6.8423699359504544005e-8*x38
            x40 = x1*x3*x5*x6*x7*x8
            return jnp.asarray([-x17*x19, x19*x21*x8, x17*x22, -x21*x24, x25*x27, -x28*x30, x31*x33, -x34*x36, x37*x39, -x16*x20*x39*x40, x36*x37*x5, -x3*x33*x34, x1*x30*x31, -x27*x28*x7, x24*x25*x6, -x12*x22*x38*x40])

        case 17:
          def shape_functions(xi):
            x0 = 2.0*xi
            x1 = x0 + 1.0
            x2 = 4.0*xi
            x3 = x2 + 1.0
            x4 = 8.0*xi
            x5 = x4 + 1.0
            x6 = x2 + 3.0
            x7 = x4 + 3.0
            x8 = x4 + 5.0
            x9 = xi - 1.0
            x10 = x0 - 1.0
            x11 = x2 - 1.0
            x12 = x4 - 1.0
            x13 = x2 - 3.0
            x14 = x4 - 3.0
            x15 = x4 - 5.0
            x16 = x4 - 7.0
            x17 = x1*x10*x11*x12*x13*x14*x15*x16*x3*x5*x6*x7*x8*x9*xi
            x18 = x4 + 7.0
            x19 = 7.8306956613834920713e-10*x18
            x20 = xi + 1.0
            x21 = x1*x10*x11*x12*x13*x14*x15*x16*x20*x3*x5*x7*x8*xi
            x22 = 1.0023290446570869851e-7*x20
            x23 = x18*x9
            x24 = 3.7587339174640761942e-7*x23
            x25 = x1*x10*x11*x12*x14*x15*x16*x20*x3*x5*x6*x7*xi
            x26 = x13*x23
            x27 = 3.508151656299804448e-6*x26
            x28 = x10*x11*x12*x14*x16*x20*x3*x5*x6*x7*x8*xi
            x29 = x15*x26
            x30 = 2.850373220743591114e-6*x29
            x31 = x1*x11*x12*x14*x16*x20*x3*x5*x6*x8*xi
            x32 = x10*x29
            x33 = 0.000027363582919138474694*x32
            x34 = x1*x11*x12*x16*x20*x5*x6*x7*x8*xi
            x35 = x14*x32
            x36 = 0.000025083284342543601803*x35
            x37 = x1*x12*x16*x20*x3*x6*x7*x8*xi
            x38 = x11*x35
            x39 = 0.000071666526692981719437*x38
            x40 = x1*x3*x5*x6*x7*x8*xi
            return jnp.asarray([x17*x19, x19*x21*x6, -x17*x22, x21*x24, -x25*x27, x28*x30, -x31*x33, x34*x36, -x37*x39, 173140.53095490047871*xi**16 - 551885.44241874527589*xi**14 + 694168.40804232804233*xi**12 - 441984.42698916603679*xi**10 + 152107.76187452758881*xi**8 - 28012.603597883597884*xi**6 + 2562.5271453766691862*xi**4 - 97.755011337868480726*xi**2 + 1.0, -x16*x20*x39*x40, x36*x37*x5, -x3*x33*x34, x30*x31*x7, -x1*x27*x28, x24*x25*x8, -x12*x22*x38*x40])

        case 18:
          def shape_functions(xi):
            x0 = 17.0*xi
            x1 = x0 + 1.0
            x2 = x0 + 3.0
            x3 = x0 + 5.0
            x4 = x0 + 7.0
            x5 = x0 + 9.0
            x6 = x0 + 11.0
            x7 = x0 + 13.0
            x8 = xi - 1.0
            x9 = x0 - 1.0
            x10 = x0 - 3.0
            x11 = x0 - 5.0
            x12 = x0 - 7.0
            x13 = x0 - 9.0
            x14 = x0 - 11.0
            x15 = x0 - 13.0
            x16 = x0 - 15.0
            x17 = x1*x10*x11*x12*x13*x14*x15*x16*x2*x3*x4*x5*x6*x7*x8*x9
            x18 = x0 + 15.0
            x19 = 3.6464518221949655895e-19*x18
            x20 = xi + 1.0
            x21 = x1*x10*x11*x12*x13*x14*x15*x16*x2*x20*x3*x4*x5*x6*x9
            x22 = 1.0538245766143450554e-16*x20
            x23 = x18*x8
            x24 = 8.4305966129147604429e-16*x23
            x25 = x1*x10*x11*x12*x13*x14*x16*x2*x20*x3*x4*x5*x7*x9
            x26 = x15*x23
            x27 = 4.2152983064573802214e-15*x26
            x28 = x1*x10*x11*x12*x13*x16*x2*x20*x3*x4*x6*x7*x9
            x29 = x14*x26
            x30 = 1.4753544072600830775e-14*x29
            x31 = x1*x10*x11*x12*x16*x2*x20*x3*x5*x6*x7*x9
            x32 = x13*x29
            x33 = 3.8359214588762160015e-14*x32
            x34 = x1*x10*x11*x16*x2*x20*x4*x5*x6*x7*x9
            x35 = x12*x32
            x36 = 7.671842917752432003e-14*x35
            x37 = x1*x10*x16*x20*x3*x4*x5*x6*x7*x9
            x38 = x11*x35
            x39 = 1.2055753156468107433e-13*x38
            x40 = x16*x2*x20*x3*x4*x5*x6*x7*x9
            x41 = x10*x38
            x42 = 1.5069691445585134292e-13*x41
            x43 = x1*x2*x3*x4*x5*x6*x7
            return jnp.asarray([-x17*x19, x19*x21*x7, x17*x22, -x21*x24, x25*x27, -x28*x30, x31*x33, -x34*x36, x37*x39, -x40*x42, x16*x20*x42*x43, -x1*x39*x40, x2*x36*x37, -x3*x33*x34, x30*x31*x4, -x27*x28*x5, x24*x25*x6, -x22*x41*x43*x9])

        case 19:
          def shape_functions(xi):
            x0 = 3.0*xi
            x1 = x0 + 1.0
            x2 = 9.0*xi
            x3 = x2 + 1.0
            x4 = x0 + 2.0
            x5 = x2 + 2.0
            x6 = x2 + 4.0
            x7 = x2 + 5.0
            x8 = x2 + 7.0
            x9 = xi - 1.0
            x10 = x0 - 1.0
            x11 = x2 - 1.0
            x12 = x0 - 2.0
            x13 = x2 - 2.0
            x14 = x2 - 4.0
            x15 = x2 - 8.0
            x16 = x2 - 5.0
            x17 = x2 - 7.0
            x18 = x1*x10*x11*x12*x13*x14*x15*x16*x17*x3*x4*x5*x6*x7*x8*x9*xi
            x19 = x2 + 8.0
            x20 = 1.0247761692089423182e-12*x19
            x21 = xi + 1.0
            x22 = x1*x10*x11*x12*x13*x14*x15*x16*x17*x21*x3*x4*x5*x6*x7*xi
            x23 = 1.6601373941184865555e-10*x21
            x24 = x19*x9
            x25 = 1.4111167850007135721e-9*x24
            x26 = x1*x10*x11*x12*x13*x14*x15*x16*x21*x3*x5*x6*x7*x8*xi
            x27 = x17*x24
            x28 = 2.5086520622234907949e-9*x27
            x29 = x1*x10*x11*x13*x14*x15*x16*x21*x3*x4*x5*x6*x8*xi
            x30 = x12*x27
            x31 = 2.8222335700014271443e-8*x30
            x32 = x1*x10*x11*x13*x14*x15*x21*x3*x4*x5*x7*x8*xi
            x33 = x16*x30
            x34 = 7.902253996003996004e-8*x33
            x35 = x10*x11*x13*x15*x21*x3*x4*x5*x6*x7*x8*xi
            x36 = x14*x33
            x37 = 5.7071834415584415584e-8*x36
            x38 = x1*x11*x13*x15*x21*x3*x4*x6*x7*x8*xi
            x39 = x10*x36
            x40 = 2.9351229128014842301e-7*x39
            x41 = x1*x11*x15*x21*x4*x5*x6*x7*x8*xi
            x42 = x13*x39
            x43 = 4.0357940051020408163e-7*x42
            x44 = x1*x3*x4*x5*x6*x7*x8*xi
            return jnp.asarray([x18*x20, x20*x22*x8, -x18*x23, x22*x25, -x26*x28, x29*x31, -x32*x34, x35*x37, -x38*x40, x41*x43, -1139827.4301937679369*xi**18 + 4010503.9210521464445*xi**16 - 5723632.7564645448023*xi**14 + 4288221.5882976921237*xi**12 - 1825541.0625608358578*xi**10 + 447065.31380067163584*xi**8 - 60894.246929607780612*xi**6 + 4228.3941844706632653*xi**4 - 124.72118622448979592*xi**2 + 1.0, x15*x21*x43*x44, -x3*x40*x41, x37*x38*x5, -x1*x34*x35, x31*x32*x6, -x28*x29*x7, x25*x26*x4, -x11*x23*x42*x44])

        case 20:
          def shape_functions(xi):
            x0 = 19.0*xi
            x1 = x0 + 1.0
            x2 = x0 + 3.0
            x3 = x0 + 5.0
            x4 = x0 + 7.0
            x5 = x0 + 9.0
            x6 = x0 + 11.0
            x7 = x0 + 13.0
            x8 = x0 + 15.0
            x9 = xi - 1.0
            x10 = x0 - 1.0
            x11 = x0 - 3.0
            x12 = x0 - 5.0
            x13 = x0 - 7.0
            x14 = x0 - 9.0
            x15 = x0 - 11.0
            x16 = x0 - 13.0
            x17 = x0 - 15.0
            x18 = x0 - 17.0
            x19 = x1*x10*x11*x12*x13*x14*x15*x16*x17*x18*x2*x3*x4*x5*x6*x7*x8*x9
            x20 = x0 + 17.0
            x21 = 2.9791273057148411679e-22*x20
            x22 = xi + 1.0
            x23 = x1*x10*x11*x12*x13*x14*x15*x16*x17*x18*x2*x22*x3*x4*x5*x6*x7
            x24 = 1.0754649573630576616e-19*x22
            x25 = x20*x9
            x26 = 9.6791846162675189544e-19*x25
            x27 = x1*x10*x11*x12*x13*x14*x15*x16*x18*x2*x22*x3*x4*x5*x6*x8
            x28 = x17*x25
            x29 = 5.4848712825515940742e-18*x28
            x30 = x1*x10*x11*x12*x13*x14*x15*x18*x2*x22*x3*x4*x5*x7*x8
            x31 = x16*x28
            x32 = 2.1939485130206376297e-17*x31
            x33 = x1*x10*x11*x12*x13*x14*x18*x2*x22*x3*x4*x6*x7*x8
            x34 = x15*x31
            x35 = 6.581845539061912889e-17*x34
            x36 = x1*x10*x11*x12*x13*x18*x2*x22*x3*x5*x6*x7*x8
            x37 = x14*x34
            x38 = 1.5357639591144463408e-16*x37
            x39 = x1*x10*x11*x12*x18*x2*x22*x4*x5*x6*x7*x8
            x40 = x13*x37
            x41 = 2.8521330669268289186e-16*x40
            x42 = x1*x10*x11*x18*x22*x3*x4*x5*x6*x7*x8
            x43 = x12*x40
            x44 = 4.2781996003902433779e-16*x43
            x45 = x10*x18*x2*x22*x3*x4*x5*x6*x7*x8
            x46 = x11*x43
            x47 = 5.2289106226991863507e-16*x46
            x48 = x1*x2*x3*x4*x5*x6*x7*x8
            return jnp.asarray([-x19*x21, x21*x23*x8, x19*x24, -x23*x26, x27*x29, -x30*x32, x33*x35, -x36*x38, x39*x41, -x42*x44, x45*x47, -x18*x22*x47*x48, x1*x44*x45, -x2*x41*x42, x3*x38*x39, -x35*x36*x4, x32*x33*x5, -x29*x30*x6, x26*x27*x7, -x10*x24*x46*x48])

        case 21:
          def shape_functions(xi):
            x0 = 2.0*xi
            x1 = x0 + 1.0
            x2 = 5.0*xi
            x3 = x2 + 1.0
            x4 = 10.0*xi
            x5 = x4 + 1.0
            x6 = x2 + 2.0
            x7 = x2 + 4.0
            x8 = x2 + 3.0
            x9 = x4 + 3.0
            x10 = x4 + 7.0
            x11 = xi - 1.0
            x12 = x0 - 1.0
            x13 = x2 - 1.0
            x14 = x4 - 1.0
            x15 = x2 - 2.0
            x16 = x2 - 4.0
            x17 = x2 - 3.0
            x18 = x4 - 3.0
            x19 = x4 - 7.0
            x20 = x4 - 9.0
            x21 = x1*x10*x11*x12*x13*x14*x15*x16*x17*x18*x19*x20*x3*x5*x6*x7*x8*x9*xi
            x22 = x4 + 9.0
            x23 = 2.6306032789197855094e-13*x22
            x24 = xi + 1.0
            x25 = x1*x10*x12*x13*x14*x15*x16*x17*x18*x19*x20*x24*x3*x5*x6*x8*x9*xi
            x26 = 5.2612065578395710189e-11*x24
            x27 = x11*x22
            x28 = 2.499073114973796234e-10*x27
            x29 = x1*x12*x13*x14*x15*x17*x18*x19*x20*x24*x3*x5*x6*x7*x8*x9*xi
            x30 = x16*x27
            x31 = 2.9988877379685554807e-9*x30
            x32 = x1*x10*x12*x13*x14*x15*x17*x18*x20*x24*x3*x5*x6*x7*x9*xi
            x33 = x19*x30
            x34 = 6.3726364431831803966e-9*x33
            x35 = x10*x12*x13*x14*x15*x18*x20*x24*x3*x5*x6*x7*x8*x9*xi
            x36 = x17*x33
            x37 = 8.1569746472744709076e-9*x36
            x38 = x1*x10*x13*x14*x15*x18*x20*x24*x3*x5*x7*x8*x9*xi
            x39 = x12*x36
            x40 = 5.0981091545465443173e-8*x39
            x41 = x1*x10*x13*x14*x18*x20*x24*x3*x5*x6*x7*x8*xi
            x42 = x15*x39
            x43 = 2.0392436618186177269e-7*x42
            x44 = x1*x10*x13*x14*x20*x24*x5*x6*x7*x8*x9*xi
            x45 = x18*x42
            x46 = 1.6568854752276269031e-7*x45
            x47 = x1*x10*x14*x20*x24*x3*x6*x7*x8*x9*xi
            x48 = x13*x45
            x49 = 4.4183612672736717416e-7*x48
            x50 = x1*x10*x3*x5*x6*x7*x8*x9*xi
            return jnp.asarray([x21*x23, x23*x25*x7, -x21*x26, x25*x28, -x29*x31, x32*x34, -x35*x37, x38*x40, -x41*x43, x44*x46, -x47*x49, 7594058.4281266233059*xi**20 - 29237124.948287499728*xi**18 + 46662451.417466849566*xi**16 - 40202717.496749499983*xi**14 + 20418933.234909475907*xi**12 - 6274158.9818744284408*xi**10 + 1153141.6151619398331*xi**8 - 121028.00916570045942*xi**6 + 6598.7171853566529492*xi**4 - 154.97677311665406904*xi**2 + 1.0, -x20*x24*x49*x50, x46*x47*x5, -x3*x43*x44, x40*x41*x9, -x37*x38*x6, x1*x34*x35, -x31*x32*x8, x10*x28*x29, -x14*x26*x48*x50])

        case _:
          assert False, "Order of shape functions not implemented or number of nodes not adequat"
    case 2:
      """ The following shape functions were generated using the code below:
      
      # Quadrilateral elements via tensor product line elements
      import numpy as np
      import sympy as sp

      for n in range(1,21):
          #n = 8  # Order of shape functions

          print("case " + str((n+1)**2) + ":")
          
          # Lagrange polynomials
          x = sp.Symbol('x')
          xI = np.concatenate(([-1], np.array([-1 + 2 * sp.Rational(i, n) for i in range(1, n)]), [1]))
          polys = [x - xi for xi in xI]
          denom = sp.prod(polys)
          num = [denom / poly for poly in polys]
          numI = [num[i].subs(x, xI[i]) for i in range(len(num))]
          Ln = [sp.simplify(num[i] / numI[i]).subs(x, sp.Symbol('xi[0]')) for i in range(len(num))]

          def square_indices(l, u):
              indices = []
              if l < u:
                  indices.extend([[l, l], [u, l], [u, u], [l, u]])
                  for k in range(l+1, u, 1):
                      indices.append([k, l])
                  for k in range(l+1, u, 1):
                      indices.append([u, k])
                  for k in range(u-1, l, -1):
                      indices.append([k, u])
                  for k in range(u-1, l, -1):
                      indices.append([l, k])
              else:
                  indices = [[l, u]]
              return indices

          ordering = square_indices(0, n)
          for k in range(1, n):
              l = k
              u = n - k
              if l <= u:
                  ordering += square_indices(l, u)

          # Tensor product shape functions
          QnTensorProduct = sp.Matrix([[Ln_i * Ln_j.subs(sp.Symbol('xi[0]'), sp.Symbol('xi[1]')).simplify() for Ln_j in Ln] for Ln_i in Ln])

          # Node ordering
          Qn = [QnTensorProduct[i, j].n(20) for (i,j) in ordering]

          # Common subexpression elimination
          subexpr, reduced_expr = sp.cse(Qn)
          utility.to_jax_function(subexpr, reduced_expr)
      """
      match n_nodes:
        case 4:
          def shape_functions(xi):
            x0 = 0.5*xi[0]
            x1 = 0.5 - x0
            x2 = 0.5*xi[1]
            x3 = 0.5 - x2
            x4 = x0 + 0.5
            x5 = x2 + 0.5
            return jnp.asarray([x1*x3, x3*x4, x4*x5, x1*x5])

        case 9:
          def shape_functions(xi):
            x0 = xi[0]*(xi[0] - 1.0)
            x1 = xi[1]*(xi[1] - 1.0)
            x2 = 0.25*x1
            x3 = xi[0]*(xi[0] + 1.0)
            x4 = xi[1]*(xi[1] + 1.0)
            x5 = 0.25*x4
            x6 = 1.0 - xi[0]**2
            x7 = 0.5*x6
            x8 = 1.0 - xi[1]**2
            x9 = 0.5*x8
            return jnp.asarray([x0*x2, x2*x3, x3*x5, x0*x5, x1*x7, x3*x9, x4*x7, x0*x9, x6*x8])

        case 16:
          def shape_functions(xi):
            x0 = -0.5625*xi[0]**3 + 0.5625*xi[0]**2 + 0.0625*xi[0] - 0.0625
            x1 = -0.5625*xi[1]**3 + 0.5625*xi[1]**2 + 0.0625*xi[1] - 0.0625
            x2 = xi[0] + 1.0
            x3 = 3.0*xi[0]
            x4 = x2*(x3 + 1.0)
            x5 = x3 - 1.0
            x6 = x1*x5
            x7 = xi[1] + 1.0
            x8 = 3.0*xi[1]
            x9 = x7*(x8 - 1.0)
            x10 = x5*x9
            x11 = x8 + 1.0
            x12 = x11*x4
            x13 = x0*x11
            x14 = xi[0] - 1.0
            x15 = 0.5625*x14
            x16 = xi[1] - 1.0
            x17 = 0.03515625*x16
            x18 = x12*x7
            x19 = 0.03515625*x14
            x20 = x10*x2
            x21 = 0.5625*x16
            x22 = 0.31640625*x14*x16
            return jnp.asarray([x0*x1, 0.0625*x4*x6, 0.00390625*x10*x12, 0.0625*x13*x9, x15*x2*x6, -x1*x15*x4, x10*x17*x4, -x17*x18*x5, -x12*x19*x9, x11*x19*x20, -x13*x21*x7, x0*x21*x9, x20*x22, -x22*x4*x9, x18*x22, -x11*x2*x22*x5*x7])

        case 25:
          def shape_functions(xi):
            x0 = xi[0] - 1.0
            x1 = xi[1] - 1.0
            x2 = 2.0*xi[1]
            x3 = x2 + 1.0
            x4 = x2 - 1.0
            x5 = x3*x4*xi[1]
            x6 = x1*x5
            x7 = 2.0*xi[0]
            x8 = x7 + 1.0
            x9 = x7 - 1.0
            x10 = x8*x9*xi[0]
            x11 = 0.027777777777777777778*x10
            x12 = x11*x6
            x13 = xi[0] + 1.0
            x14 = xi[1] + 1.0
            x15 = x11*x14*x5
            x16 = x0*xi[0]
            x17 = x16*x9
            x18 = 0.22222222222222222222*x13
            x19 = x18*x6
            x20 = 4.0*xi[0]**4 - 5.0*xi[0]**2 + 1.0
            x21 = x1*x20
            x22 = 0.16666666666666666667*x5
            x23 = x16*x8
            x24 = x14*xi[1]
            x25 = x24*x4
            x26 = x1*x10
            x27 = x18*x26
            x28 = 4.0*xi[1]**4 - 5.0*xi[1]**2 + 1.0
            x29 = x13*x28
            x30 = 0.16666666666666666667*x10
            x31 = x24*x3
            x32 = x14*x18*x5
            x33 = 0.22222222222222222222*x0*x26
            x34 = 1.7777777777777777778*x1*x13
            x35 = x25*x34
            x36 = x31*x34
            x37 = 1.3333333333333333333*x21
            x38 = 1.3333333333333333333*x29
            return jnp.asarray([x0*x12, x12*x13, x13*x15, x0*x15, -x17*x19, x21*x22, -x19*x23, -x25*x27, x29*x30, -x27*x31, -x23*x32, x14*x20*x22, -x17*x32, -x31*x33, x0*x28*x30, -x25*x33, x17*x35, x23*x35, x23*x36, x17*x36, -x25*x37, -x23*x38, -x31*x37, -x17*x38, x20*x28])

        case 36:
          def shape_functions(xi):
            x0 = 5.0*xi[1]
            x1 = x0 + 3.0
            x2 = xi[0] - 1.0
            x3 = x1*x2
            x4 = 5.0*xi[0]
            x5 = x4 + 1.0
            x6 = x0 + 1.0
            x7 = x4 + 3.0
            x8 = xi[1] - 1.0
            x9 = x4 - 1.0
            x10 = x0 - 1.0
            x11 = x4 - 3.0
            x12 = x0 - 3.0
            x13 = x10*x11*x12*x5*x6*x7*x8*x9
            x14 = 1.6954210069444444444e-6*x13
            x15 = xi[0] + 1.0
            x16 = x1*x15
            x17 = xi[1] + 1.0
            x18 = x17*x7
            x19 = x16*x18
            x20 = x10*x11*x12*x5*x6*x9
            x21 = 1.6954210069444444444e-6*x20
            x22 = x18*x3
            x23 = 0.000042385525173611111111*x15
            x24 = x3*x8
            x25 = x23*x24
            x26 = x10*x11*x6*x9
            x27 = x24*x26
            x28 = x12*x15
            x29 = 0.000084771050347222222222*x28
            x30 = x29*x7
            x31 = x10*x5
            x32 = x11*x6
            x33 = x31*x32
            x34 = x12*x9
            x35 = x31*x34
            x36 = x35*x6
            x37 = x17*x23
            x38 = x19*x8
            x39 = 0.000084771050347222222222*x38
            x40 = x11*x35
            x41 = x32*x5
            x42 = x34*x41
            x43 = 0.000042385525173611111111*x26*x5
            x44 = x22*x29
            x45 = x22*x8
            x46 = 0.000084771050347222222222*x45
            x47 = x17*x2
            x48 = 0.0010596381293402777778*x8
            x49 = x28*x9
            x50 = x31*x6
            x51 = x18*x2
            x52 = 0.0010596381293402777778*x15
            x53 = 0.0021192762586805555556*x26
            x54 = x28*x51*x8
            x55 = 0.0021192762586805555556*x33
            x56 = 0.0021192762586805555556*x49
            x57 = x31*x45
            x58 = x15*x45
            x59 = x17*x24*x56
            x60 = 0.0042385525173611111111*x11
            x61 = x45*x49
            return jnp.asarray([x14*x3, -x14*x16, x19*x21, -x21*x22, -x20*x25, x27*x30, -x24*x30*x33, x25*x36*x7, x13*x37, -x39*x40, x39*x42, -x38*x43, -x22*x23*x36, x33*x44, -x26*x44, x20*x3*x37, x43*x45, -x42*x46, x40*x46, -0.000042385525173611111111*x13*x47, x15*x20*x47*x48, -x48*x49*x50*x51, x45*x50*x52*x9, -x17*x27*x5*x52, -x53*x54, x54*x55, x56*x57, -x45*x5*x56*x6, -x55*x58, x53*x58, x41*x59, -x11*x31*x59, x10*x60*x61, -x28*x57*x60, 0.0042385525173611111111*x28*x41*x45, -0.0042385525173611111111*x32*x61])

        case 49:
          def shape_functions(xi):
            x0 = xi[0] - 1.0
            x1 = xi[1] - 1.0
            x2 = 3.0*xi[1]
            x3 = x2 + 1.0
            x4 = x2 + 2.0
            x5 = x2 - 1.0
            x6 = x2 - 2.0
            x7 = x3*x4*x5*x6*xi[1]
            x8 = x1*x7
            x9 = 3.0*xi[0]
            x10 = x9 + 1.0
            x11 = x9 + 2.0
            x12 = x9 - 1.0
            x13 = x9 - 2.0
            x14 = x10*x11*x12*x13*xi[0]
            x15 = 0.00015625*x14
            x16 = x15*x8
            x17 = xi[0] + 1.0
            x18 = xi[1] + 1.0
            x19 = x15*x18*x7
            x20 = x17*x8
            x21 = x0*x10*x12*xi[0]
            x22 = 0.0028125*x21
            x23 = x20*x22
            x24 = x0*x13*xi[0]
            x25 = x12*x24
            x26 = 0.00703125*x11
            x27 = x20*x26
            x28 = -20.25*xi[0]**6 + 31.5*xi[0]**4 - 12.25*xi[0]**2 + 1.0
            x29 = x1*x28
            x30 = 0.0125*x7
            x31 = x10*x24
            x32 = x18*x5*x6*xi[1]
            x33 = x1*x14
            x34 = x17*x33
            x35 = x32*x34
            x36 = 0.0028125*x3
            x37 = 0.00703125*x4
            x38 = -20.25*xi[1]**6 + 31.5*xi[1]**4 - 12.25*xi[1]**2 + 1.0
            x39 = x17*x38
            x40 = 0.0125*x14
            x41 = x18*x4*xi[1]
            x42 = x34*x41
            x43 = x3*x6
            x44 = 0.00703125*x43
            x45 = x36*x5
            x46 = x11*x17
            x47 = x18*x7
            x48 = x22*x47
            x49 = x17*x26*x47
            x50 = x13*x17
            x51 = x0*x33
            x52 = x41*x51
            x53 = x32*x51
            x54 = x1*x21
            x55 = x50*x54
            x56 = x3*x32
            x57 = 0.050625*x56
            x58 = x46*x54
            x59 = x41*x58
            x60 = x3*x5
            x61 = 0.050625*x60
            x62 = x41*x55
            x63 = x1*x46
            x64 = 0.1265625*x63
            x65 = x56*x64
            x66 = x29*x32
            x67 = 0.225*x3
            x68 = x32*x4
            x69 = 0.1265625*x68
            x70 = x11*x39
            x71 = 0.225*x21
            x72 = 0.1265625*x43
            x73 = x31*x41
            x74 = x60*x64
            x75 = x29*x41
            x76 = x25*x41
            x77 = 0.31640625*x63
            x78 = x68*x77
            x79 = x43*x77
            x80 = 0.5625*x70
            return jnp.asarray([x0*x16, x16*x17, x17*x19, x0*x19, -x13*x23, x25*x27, x29*x30, x27*x31, -x11*x23, -x35*x36, x35*x37, x39*x40, x42*x44, -x42*x45, -x46*x48, x31*x49, x18*x28*x30, x25*x49, -x48*x50, -x45*x52, x44*x52, x0*x38*x40, x37*x53, -x36*x53, x55*x57, x57*x58, x59*x61, x61*x62, -x25*x65, -x66*x67, -x31*x65, -x58*x69, -x70*x71, -x59*x72, -x73*x74, -x5*x67*x75, -x74*x76, -x62*x72, -x13*x39*x71, -x55*x69, x25*x78, x31*x78, x73*x79, x76*x79, 0.5625*x4*x66, x31*x80, 0.5625*x43*x75, x25*x80, x28*x38])

        case 64:
          def shape_functions(xi):
            x0 = 7.0*xi[1]
            x1 = x0 + 5.0
            x2 = xi[0] - 1.0
            x3 = x1*x2
            x4 = 7.0*xi[0]
            x5 = x4 + 1.0
            x6 = x0 + 1.0
            x7 = x4 + 3.0
            x8 = x0 + 3.0
            x9 = x4 + 5.0
            x10 = xi[1] - 1.0
            x11 = x4 - 1.0
            x12 = x0 - 1.0
            x13 = x4 - 3.0
            x14 = x0 - 3.0
            x15 = x4 - 5.0
            x16 = x0 - 5.0
            x17 = x10*x11*x12*x13*x14*x15*x16*x5*x6*x7*x8*x9
            x18 = 1.1773756992669753086e-10*x17
            x19 = xi[0] + 1.0
            x20 = x1*x19
            x21 = xi[1] + 1.0
            x22 = x21*x9
            x23 = x20*x22
            x24 = x11*x12*x13*x14*x15*x16*x5*x6*x7*x8
            x25 = 1.1773756992669753086e-10*x24
            x26 = x22*x3
            x27 = 5.7691409264081790123e-9*x19
            x28 = x10*x3
            x29 = x27*x28
            x30 = x11*x12*x13*x14*x15*x16*x5*x8
            x31 = x28*x30
            x32 = x19*x6
            x33 = 1.7307422779224537037e-8*x32
            x34 = x33*x9
            x35 = x11*x12*x13*x14*x15*x16
            x36 = x28*x35
            x37 = 2.8845704632040895062e-8*x7
            x38 = x32*x8
            x39 = x37*x38
            x40 = x39*x9
            x41 = x12*x13*x14*x5
            x42 = x15*x16
            x43 = x28*x42
            x44 = x11*x8
            x45 = x14*x44
            x46 = x5*x7
            x47 = x12*x46
            x48 = x43*x47
            x49 = x44*x6
            x50 = x41*x7
            x51 = x16*x50
            x52 = x49*x51
            x53 = x21*x27
            x54 = x10*x23
            x55 = x54*x6
            x56 = 1.7307422779224537037e-8*x35*x46
            x57 = x30*x37
            x58 = x13*x42
            x59 = x5*x58
            x60 = x37*x45*x59
            x61 = x49*x54
            x62 = x47*x58
            x63 = 1.7307422779224537037e-8*x62
            x64 = x15*x50
            x65 = 5.7691409264081790123e-9*x64
            x66 = x26*x42
            x67 = x47*x66
            x68 = x45*x67
            x69 = x26*x35
            x70 = x26*x30
            x71 = x10*x26
            x72 = x49*x71
            x73 = x6*x71
            x74 = x2*x21
            x75 = 2.826879053940007716e-7*x10
            x76 = x11*x38
            x77 = x2*x22
            x78 = x76*x77
            x79 = 2.826879053940007716e-7*x76
            x80 = x10*x77
            x81 = 8.4806371618200231481e-7*x32
            x82 = 1.413439526970003858e-6*x38
            x83 = x80*x82
            x84 = 8.4806371618200231481e-7*x47
            x85 = x10*x14
            x86 = x51*x71
            x87 = 1.413439526970003858e-6*x19
            x88 = x13*x16
            x89 = x71*x76
            x90 = x14*x89
            x91 = 1.413439526970003858e-6*x90
            x92 = x12*x7
            x93 = x13*x21*x76
            x94 = x14*x46
            x95 = 2.5441911485460069444e-6*x32
            x96 = x10*x69
            x97 = 4.2403185809100115741e-6*x32
            x98 = x7*x96
            x99 = x10*x50*x66
            x100 = 4.2403185809100115741e-6*x10*x19
            x101 = x38*x71
            x102 = 7.0671976348500192901e-6*x19*x8
            x103 = 7.0671976348500192901e-6*x58
            return jnp.asarray([x18*x3, -x18*x20, x23*x25, -x25*x26, -x24*x29, x31*x34, -x36*x40, x40*x41*x43, -x34*x45*x48, x29*x52*x9, x17*x53, -x55*x56, x54*x57, -x55*x60, x61*x63, -x61*x65, -x26*x27*x52, x33*x68, -x39*x41*x66, x39*x69, -x33*x70, x24*x3*x53, x65*x72, -x63*x72, x60*x73, -x57*x71, x56*x73, -5.7691409264081790123e-9*x17*x74, x19*x24*x74*x75, -x51*x75*x78, x50*x71*x79, -x21*x28*x64*x79, -x30*x80*x81, x35*x7*x83, -x42*x50*x83, x42*x78*x84*x85, x11*x81*x86, -x44*x86*x87, x46*x88*x91, -x84*x88*x89, -x15*x84*x90, x64*x71*x82, -x13*x15*x91*x92, 8.4806371618200231481e-7*x15*x41*x89, 8.4806371618200231481e-7*x48*x93, -1.413439526970003858e-6*x43*x93*x94, x21*x31*x7*x87, -x21*x36*x46*x81, x5*x95*x96, -x11*x67*x85*x95, 2.5441911485460069444e-6*x10*x67*x76, -2.5441911485460069444e-6*x12*x59*x89, -x97*x98, x97*x99, x100*x68, -4.2403185809100115741e-6*x46*x66*x76*x85, -4.2403185809100115741e-6*x101*x62, 4.2403185809100115741e-6*x58*x89*x92, 4.2403185809100115741e-6*x59*x90, -x100*x70, x102*x98, -x102*x99, x101*x103*x94, -x103*x7*x90])

        case 81:
          def shape_functions(xi):
            x0 = xi[0] - 1.0
            x1 = xi[1] - 1.0
            x2 = 2.0*xi[1]
            x3 = x2 + 1.0
            x4 = 4.0*xi[1]
            x5 = x4 + 1.0
            x6 = x4 + 3.0
            x7 = x2 - 1.0
            x8 = x4 - 1.0
            x9 = x4 - 3.0
            x10 = x3*x5*x6*x7*x8*x9*xi[1]
            x11 = x1*x10
            x12 = 2.0*xi[0]
            x13 = x12 + 1.0
            x14 = 4.0*xi[0]
            x15 = x14 + 1.0
            x16 = x14 + 3.0
            x17 = x12 - 1.0
            x18 = x14 - 1.0
            x19 = x14 - 3.0
            x20 = x13*x15*x16*x17*x18*x19*xi[0]
            x21 = 2.5195263290501385739e-6*x20
            x22 = x11*x21
            x23 = xi[0] + 1.0
            x24 = xi[1] + 1.0
            x25 = x10*x21*x24
            x26 = x11*x23
            x27 = x0*x13*x15*x17*x18*xi[0]
            x28 = 0.000080624842529604434366*x27
            x29 = x26*x28
            x30 = x16*x26
            x31 = x0*x15*x18*x19*xi[0]
            x32 = 0.00014109347442680776014*x31
            x33 = x30*x32
            x34 = x0*x17*x19*xi[0]
            x35 = x18*x34
            x36 = 0.00056437389770723104056*x13
            x37 = x30*x36
            x38 = 113.77777777777777778*xi[0]**8 - 213.33333333333333333*xi[0]**6 + 121.33333333333333333*xi[0]**4 - 22.777777777777777778*xi[0]**2 + 1.0
            x39 = x1*x38
            x40 = 0.0015873015873015873016*x10
            x41 = x15*x34
            x42 = x24*x5*x7*x8*x9*xi[1]
            x43 = x1*x20
            x44 = x23*x43
            x45 = x42*x44
            x46 = 0.000080624842529604434366*x3
            x47 = 0.00014109347442680776014*x6
            x48 = x24*x6*x7*x8*xi[1]
            x49 = x3*x9
            x50 = x48*x49
            x51 = 0.00056437389770723104056*x44
            x52 = 113.77777777777777778*xi[1]**8 - 213.33333333333333333*xi[1]**6 + 121.33333333333333333*xi[1]**4 - 22.777777777777777778*xi[1]**2 + 1.0
            x53 = x23*x52
            x54 = 0.0015873015873015873016*x20
            x55 = x6*x7
            x56 = x24*x49*xi[1]
            x57 = x5*x56
            x58 = x55*x57
            x59 = x57*x8
            x60 = x47*x59
            x61 = x48*x5
            x62 = x46*x61
            x63 = x16*x23
            x64 = x10*x24*x63
            x65 = x32*x64
            x66 = x36*x64
            x67 = x19*x23
            x68 = x0*x43
            x69 = 0.00056437389770723104056*x68
            x70 = x42*x68
            x71 = x1*x27
            x72 = x67*x71
            x73 = x3*x42
            x74 = 0.0025799949609473418997*x73
            x75 = x63*x71
            x76 = x3*x61
            x77 = 0.0025799949609473418997*x76
            x78 = x1*x63
            x79 = x73*x78
            x80 = 0.0045149911816578483245*x31
            x81 = x79*x80
            x82 = 0.018059964726631393298*x13
            x83 = x79*x82
            x84 = x39*x42
            x85 = 0.050793650793650793651*x3
            x86 = 0.0045149911816578483245*x6
            x87 = x75*x86
            x88 = 0.018059964726631393298*x75
            x89 = x16*x53
            x90 = 0.050793650793650793651*x27
            x91 = x76*x78
            x92 = x80*x91
            x93 = x82*x91
            x94 = x39*x5
            x95 = x72*x86
            x96 = 0.018059964726631393298*x72
            x97 = x31*x78
            x98 = x17*x97
            x99 = x42*x6
            x100 = 0.0079012345679012345679*x99
            x101 = x13*x97
            x102 = x59*x6
            x103 = 0.0079012345679012345679*x102
            x104 = x13*x78
            x105 = 0.031604938271604938272*x104
            x106 = x105*x99
            x107 = 0.088888888888888888889*x6
            x108 = 0.031604938271604938272*x101
            x109 = x13*x89
            x110 = 0.088888888888888888889*x31
            x111 = x102*x105
            x112 = x56*x94
            x113 = 0.031604938271604938272*x98
            x114 = 0.12641975308641975309*x104
            x115 = x114*x50
            x116 = x114*x58
            x117 = 0.35555555555555555556*x109
            return jnp.asarray([x0*x22, x22*x23, x23*x25, x0*x25, -x19*x29, x17*x33, -x35*x37, x39*x40, -x37*x41, x13*x33, -x16*x29, -x45*x46, x45*x47, -x50*x51, x53*x54, -x51*x58, x44*x60, -x44*x62, -x28*x64, x13*x65, -x41*x66, x24*x38*x40, -x35*x66, x17*x65, -x10*x24*x28*x67, -x62*x68, x60*x68, -x58*x69, x0*x52*x54, -x50*x69, x47*x70, -x46*x70, x72*x74, x74*x75, x75*x77, x72*x77, -x17*x81, x35*x83, -x84*x85, x41*x83, -x13*x81, -x42*x87, x50*x88, -x89*x90, x58*x88, -x59*x87, -x13*x92, x41*x93, -x48*x85*x94, x35*x93, -x17*x92, -x59*x95, x58*x96, -x19*x53*x90, x50*x96, -x42*x95, x100*x98, x100*x101, x101*x103, x103*x98, -x106*x35, x107*x84, -x106*x41, -x108*x50, x109*x110, -x108*x58, -x111*x41, x107*x112*x8, -x111*x35, -x113*x58, x110*x17*x89, -x113*x50, x115*x35, x115*x41, x116*x41, x116*x35, -0.35555555555555555556*x39*x50, -x117*x41, -0.35555555555555555556*x112*x55, -x117*x35, x38*x52])

        case 100:
          def shape_functions(xi):
            x0 = 9.0*xi[1]
            x1 = x0 + 7.0
            x2 = xi[0] - 1.0
            x3 = x1*x2
            x4 = 3.0*xi[0]
            x5 = x4 + 1.0
            x6 = 3.0*xi[1]
            x7 = x6 + 1.0
            x8 = 9.0*xi[0]
            x9 = x8 + 1.0
            x10 = x0 + 1.0
            x11 = x8 + 5.0
            x12 = x0 + 5.0
            x13 = x8 + 7.0
            x14 = xi[1] - 1.0
            x15 = x4 - 1.0
            x16 = x6 - 1.0
            x17 = x8 - 1.0
            x18 = x0 - 1.0
            x19 = x8 - 5.0
            x20 = x0 - 5.0
            x21 = x8 - 7.0
            x22 = x0 - 7.0
            x23 = x10*x11*x12*x13*x14*x15*x16*x17*x18*x19*x20*x21*x22*x5*x7*x9
            x24 = 1.900658315541792889e-13*x23
            x25 = xi[0] + 1.0
            x26 = x1*x25
            x27 = xi[1] + 1.0
            x28 = x13*x27
            x29 = x26*x28
            x30 = x10*x11*x12*x15*x16*x17*x18*x19*x20*x21*x22*x5*x7*x9
            x31 = 1.900658315541792889e-13*x30
            x32 = x28*x3
            x33 = 1.5395332355888522401e-11*x25
            x34 = x14*x3
            x35 = x33*x34
            x36 = x10*x12*x15*x16*x17*x18*x19*x20*x21*x22*x5*x9
            x37 = x34*x36
            x38 = x25*x7
            x39 = 6.1581329423554089605e-11*x38
            x40 = x13*x39
            x41 = x10*x15*x16*x17*x18*x19*x20*x21*x22*x9
            x42 = x34*x41
            x43 = 4.7896589551653180804e-11*x11
            x44 = x12*x38
            x45 = x43*x44
            x46 = x13*x45
            x47 = x10*x15*x17*x18*x19*x20*x21*x22
            x48 = x11*x5
            x49 = x44*x48
            x50 = x47*x49
            x51 = 2.1553465298243931362e-10*x50
            x52 = x16*x34
            x53 = x13*x52
            x54 = x10*x18*x19*x20*x21*x22*x9
            x55 = x15*x49
            x56 = x54*x55
            x57 = 2.1553465298243931362e-10*x56
            x58 = x17*x52
            x59 = x5*x54
            x60 = x10*x18*x20
            x61 = x21*x60
            x62 = x58*x61
            x63 = x12*x9
            x64 = x15*x48
            x65 = x22*x64
            x66 = x63*x65
            x67 = x19*x60
            x68 = x16*x17
            x69 = x66*x68
            x70 = x69*x7
            x71 = x67*x70
            x72 = x27*x33
            x73 = x14*x29
            x74 = x7*x73
            x75 = 6.1581329423554089605e-11*x41*x48
            x76 = x36*x43
            x77 = x18*x21
            x78 = x70*x77
            x79 = x73*x78
            x80 = x19*x20
            x81 = 2.1553465298243931362e-10*x80
            x82 = x10*x21
            x83 = x70*x81*x82
            x84 = x63*x74
            x85 = x47*x5
            x86 = x43*x85
            x87 = x10*x19
            x88 = 6.1581329423554089605e-11*x87
            x89 = x61*x68
            x90 = x19*x89
            x91 = 1.5395332355888522401e-11*x64*x90
            x92 = x32*x39
            x93 = x61*x69
            x94 = x32*x45
            x95 = x16*x32
            x96 = x14*x32
            x97 = x7*x96
            x98 = x63*x97
            x99 = x78*x96
            x100 = x2*x27
            x101 = 1.2470219208269703145e-9*x14
            x102 = x2*x28
            x103 = x102*x68
            x104 = x55*x9
            x105 = x104*x22
            x106 = x105*x67
            x107 = 1.2470219208269703145e-9*x104
            x108 = x67*x96
            x109 = x108*x68
            x110 = x102*x14
            x111 = 4.988087683307881258e-9*x38
            x112 = 3.8796237536839076451e-9*x11
            x113 = x112*x44
            x114 = 1.7458306891577584403e-8*x110*x16
            x115 = 3.8796237536839076451e-9*x49
            x116 = x14*x54
            x117 = 4.988087683307881258e-9*x105
            x118 = x65*x9
            x119 = x18*x68
            x120 = 1.7458306891577584403e-8*x96
            x121 = x105*x80
            x122 = x120*x121
            x123 = x10*x68
            x124 = x17*x96
            x125 = x117*x87
            x126 = 4.988087683307881258e-9*x96
            x127 = x9*x90
            x128 = x127*x96
            x129 = x14*x95
            x130 = x15*x44
            x131 = x130*x5
            x132 = x27*x58
            x133 = x132*x77
            x134 = 1.7458306891577584403e-8*x121
            x135 = 1.9952350733231525032e-8*x96
            x136 = x135*x38
            x137 = x135*x77
            x138 = x22*x68
            x139 = x138*x87
            x140 = x139*x9
            x141 = 1.551849501473563058e-8*x96
            x142 = x141*x38
            x143 = x11*x41
            x144 = 6.9833227566310337612e-8*x38
            x145 = x129*x47*x48
            x146 = x116*x64*x95
            x147 = x48*x54*x68
            x148 = x141*x25
            x149 = x105*x77
            x150 = 6.9833227566310337612e-8*x96
            x151 = x150*x20*x68
            x152 = x49*x77
            x153 = x140*x141
            x154 = x150*x77
            x155 = x11*x130
            x156 = x155*x77
            x157 = x44*x9
            x158 = x138*x80
            x159 = x158*x9
            x160 = x131*x159
            x161 = 1.2069940567016601562e-8*x96
            x162 = x12*x25
            x163 = x161*x162
            x164 = 5.4314732551574707031e-8*x162
            x165 = 5.4314732551574707031e-8*x96
            x166 = x159*x165
            x167 = x166*x82
            x168 = 2.4441629648208618164e-7*x77
            x169 = x158*x55*x96
            x170 = x121*x129
            x171 = 2.4441629648208618164e-7*x82
            return jnp.asarray([x24*x3, -x24*x26, x29*x31, -x31*x32, -x30*x35, x37*x40, -x42*x46, x51*x53, -x53*x57, x46*x58*x59, -x40*x62*x66, x13*x35*x71, x23*x72, -x74*x75, x73*x76, -x79*x81, x73*x83, -x84*x86, x79*x88, -x84*x91, -x32*x33*x71, x92*x93, -x59*x68*x94, x57*x95, -x51*x95, x41*x94, -x36*x92, x3*x30*x72, x91*x98, -x88*x99, x86*x98, -x83*x96, x81*x99, -x76*x96, x75*x97, -1.5395332355888522401e-11*x100*x23, x100*x101*x25*x30, -x101*x103*x106, x107*x109, -x107*x19*x27*x62, -x110*x111*x36, x110*x113*x41, -x114*x50, x114*x56, -x103*x115*x116, x110*x117*x89, x109*x111*x118, -3.8796237536839076451e-9*x108*x25*x69, x119*x122, -x122*x123, 3.8796237536839076451e-9*x106*x124, -x119*x125*x96, -x104*x126*x89, x115*x128, -1.7458306891577584403e-8*x104*x129*x19*x61, x120*x55*x90, -x113*x128*x15, x126*x127*x131, x125*x133, -3.8796237536839076451e-9*x27*x34*x50*x9, x132*x134*x82, -x133*x134, x112*x25*x27*x37, -x111*x27*x42*x48, x136*x41*x5, -x118*x136*x89, x105*x123*x137, -x131*x137*x140, -x142*x143, x144*x145, -x144*x146, x142*x147, x148*x93, -x149*x151, x105*x151*x82, -1.551849501473563058e-8*x105*x124*x61, -x152*x153, 6.9833227566310337612e-8*x129*x149*x87, -x139*x154*x55, x153*x156, x141*x157*x85, -x150*x160*x82, x154*x160, -x148*x36, x143*x163, -x147*x163, 1.2069940567016601562e-8*x124*x49*x54, -x11*x157*x161*x47, -x145*x164, x146*x164, x152*x166, -x167*x49, -x165*x56, x165*x50, x155*x167, -x156*x166, x168*x169, -x168*x170, x170*x171, -x169*x171])

        case 121:
          def shape_functions(xi):
            x0 = xi[0] - 1.0
            x1 = xi[1] - 1.0
            x2 = 5.0*xi[1]
            x3 = x2 + 1.0
            x4 = x2 + 2.0
            x5 = x2 + 4.0
            x6 = x2 + 3.0
            x7 = x2 - 1.0
            x8 = x2 - 2.0
            x9 = x2 - 4.0
            x10 = x2 - 3.0
            x11 = x10*x3*x4*x5*x6*x7*x8*x9*xi[1]
            x12 = x1*x11
            x13 = 5.0*xi[0]
            x14 = x13 + 1.0
            x15 = x13 + 2.0
            x16 = x13 + 4.0
            x17 = x13 + 3.0
            x18 = x13 - 1.0
            x19 = x13 - 2.0
            x20 = x13 - 4.0
            x21 = x13 - 3.0
            x22 = x14*x15*x16*x17*x18*x19*x20*x21*xi[0]
            x23 = 4.7462865175791395662e-11*x22
            x24 = x12*x23
            x25 = xi[0] + 1.0
            x26 = xi[1] + 1.0
            x27 = x11*x23*x26
            x28 = x12*x25
            x29 = x0*x14*x15*x17*x18*x19*x21*xi[0]
            x30 = 2.3731432587895697831e-9*x29
            x31 = x28*x30
            x32 = x16*x28
            x33 = x0*x14*x15*x18*x19*x20*xi[0]
            x34 = 1.0679144664553064024e-8*x33
            x35 = x32*x34
            x36 = x17*x32
            x37 = x0*x14*x18*x20*x21*xi[0]
            x38 = 2.8477719105474837397e-8*x37
            x39 = x36*x38
            x40 = x0*x19*x20*x21*xi[0]
            x41 = x18*x40
            x42 = 4.9836008434580965445e-8*x15
            x43 = x36*x42
            x44 = -678.16840277777777778*xi[0]**10 + 1491.9704861111111111*xi[0]**8 - 1110.0260416666666667*xi[0]**6 + 331.81423611111111111*xi[0]**4 - 36.590277777777777778*xi[0]**2 + 1.0
            x45 = x1*x44
            x46 = 6.8893298059964726631e-6*x11
            x47 = x14*x40
            x48 = x10*x26*x3*x4*x7*x8*x9*xi[1]
            x49 = x1*x22
            x50 = x25*x49
            x51 = x48*x50
            x52 = 2.3731432587895697831e-9*x6
            x53 = 1.0679144664553064024e-8*x5
            x54 = x10*x26*x3*x5*x7*x8*xi[1]
            x55 = x50*x54
            x56 = x6*x9
            x57 = 2.8477719105474837397e-8*x56
            x58 = x26*x56*x7*x8*xi[1]
            x59 = x10*x5
            x60 = x58*x59
            x61 = x4*x50
            x62 = 4.9836008434580965445e-8*x61
            x63 = -678.16840277777777778*xi[1]**10 + 1491.9704861111111111*xi[1]**8 - 1110.0260416666666667*xi[1]**6 + 331.81423611111111111*xi[1]**4 - 36.590277777777777778*xi[1]**2 + 1.0
            x64 = x25*x63
            x65 = 6.8893298059964726631e-6*x22
            x66 = x56*x8
            x67 = x26*x59*xi[1]
            x68 = x3*x67
            x69 = x66*x68
            x70 = x68*x7
            x71 = x57*x70
            x72 = x3*x58
            x73 = x53*x72
            x74 = x4*x52
            x75 = x16*x25
            x76 = x11*x26*x75
            x77 = x17*x76
            x78 = x38*x77
            x79 = x42*x77
            x80 = x20*x25
            x81 = x0*x49
            x82 = x54*x81
            x83 = x4*x81
            x84 = 4.9836008434580965445e-8*x83
            x85 = x48*x81
            x86 = x1*x29
            x87 = x80*x86
            x88 = x48*x6
            x89 = 1.1865716293947848916e-7*x88
            x90 = x75*x86
            x91 = x54*x90
            x92 = x4*x6
            x93 = 1.1865716293947848916e-7*x92
            x94 = x54*x87
            x95 = x1*x75
            x96 = x88*x95
            x97 = 5.339572332276532012e-7*x33
            x98 = x96*x97
            x99 = x17*x96
            x100 = 1.4238859552737418699e-6*x37
            x101 = x100*x99
            x102 = 2.4918004217290482723e-6*x15
            x103 = x102*x99
            x104 = x45*x48
            x105 = 0.00034446649029982363316*x6
            x106 = 5.339572332276532012e-7*x5
            x107 = x106*x90
            x108 = 1.4238859552737418699e-6*x56
            x109 = x4*x90
            x110 = 2.4918004217290482723e-6*x109
            x111 = x16*x64
            x112 = 0.00034446649029982363316*x29
            x113 = x108*x70
            x114 = x4*x72
            x115 = x17*x95
            x116 = x115*x54
            x117 = x116*x92
            x118 = x100*x117
            x119 = x102*x117
            x120 = x4*x45
            x121 = x21*x95
            x122 = x121*x54
            x123 = x106*x87
            x124 = x4*x87
            x125 = 2.4918004217290482723e-6*x124
            x126 = x121*x33
            x127 = x48*x5
            x128 = 2.4028075495244394054e-6*x127
            x129 = x115*x33
            x130 = x114*x5
            x131 = 2.4028075495244394054e-6*x130
            x132 = x19*x37
            x133 = x115*x127
            x134 = 6.4074867987318384144e-6*x133
            x135 = 0.000011213101897780717225*x15
            x136 = x133*x135
            x137 = 0.0015500992063492063492*x5
            x138 = x15*x37
            x139 = 6.4074867987318384144e-6*x56
            x140 = x139*x33
            x141 = x129*x4
            x142 = 0.000011213101897780717225*x141
            x143 = x111*x17
            x144 = 0.0015500992063492063492*x33
            x145 = x139*x70
            x146 = x115*x130
            x147 = 6.4074867987318384144e-6*x146
            x148 = x135*x146
            x149 = x120*x3
            x150 = x126*x4
            x151 = 0.000011213101897780717225*x150
            x152 = 0.000017086631463284902438*x56
            x153 = x116*x152
            x154 = x115*x4
            x155 = x138*x154
            x156 = x152*x70
            x157 = x132*x154
            x158 = x15*x41
            x159 = 0.000029901605060748579267*x56
            x160 = x116*x159
            x161 = 0.0041335978835978835979*x56
            x162 = x15*x47
            x163 = 0.000029901605060748579267*x155
            x164 = x143*x15
            x165 = 0.0041335978835978835979*x37
            x166 = x154*x162
            x167 = x159*x70
            x168 = x149*x67
            x169 = x154*x158
            x170 = 0.000029901605060748579267*x157
            x171 = 0.000052327808856310013717*x60
            x172 = 0.000052327808856310013717*x69
            x173 = 0.0072337962962962962963*x164
            return jnp.asarray([x0*x24, x24*x25, x25*x27, x0*x27, -x20*x31, x21*x35, -x19*x39, x41*x43, x45*x46, x43*x47, -x15*x39, x17*x35, -x16*x31, -x51*x52, x51*x53, -x55*x57, x60*x62, x64*x65, x62*x69, -x61*x71, x61*x73, -x55*x74, -x30*x76, x34*x77, -x15*x78, x47*x79, x26*x44*x46, x41*x79, -x19*x78, x21*x34*x76, -x11*x26*x30*x80, -x74*x82, x73*x83, -x71*x83, x69*x84, x0*x63*x65, x60*x84, -x57*x82, x53*x85, -x52*x85, x87*x89, x89*x90, x91*x93, x93*x94, -x21*x98, x101*x19, -x103*x41, -x104*x105, -x103*x47, x101*x15, -x17*x98, -x107*x48, x108*x91, -x110*x60, -x111*x112, -x110*x69, x109*x113, -x107*x114, -x117*x97, x118*x15, -x119*x47, -x105*x120*x54, -x119*x41, x118*x19, -x122*x92*x97, -x114*x123, x113*x124, -x125*x69, -x112*x20*x64, -x125*x60, x108*x94, -x123*x48, x126*x128, x128*x129, x129*x131, x126*x131, -x132*x134, x136*x41, x104*x137, x136*x47, -x134*x138, -x116*x140, x142*x60, x143*x144, x142*x69, -x141*x145, -x138*x147, x148*x47, x137*x149*x58, x148*x41, -x132*x147, -x145*x150, x151*x69, x111*x144*x21, x151*x60, -x122*x140, x132*x153, x138*x153, x155*x156, x156*x157, -x158*x160, -x161*x45*x54, -x160*x162, -x163*x60, -x164*x165, -x163*x69, -x166*x167, -x161*x168*x7, -x167*x169, -x170*x69, -x143*x165*x19, -x170*x60, x169*x171, x166*x171, x166*x172, x169*x172, 0.0072337962962962962963*x120*x60, x173*x47, 0.0072337962962962962963*x168*x66, x173*x41, x44*x63])

        case 144:
          def shape_functions(xi):
            x0 = 11.0*xi[1]
            x1 = x0 + 9.0
            x2 = xi[0] - 1.0
            x3 = x1*x2
            x4 = 11.0*xi[0]
            x5 = x4 + 1.0
            x6 = x0 + 1.0
            x7 = x4 + 3.0
            x8 = x0 + 3.0
            x9 = x4 + 5.0
            x10 = x0 + 5.0
            x11 = x4 + 7.0
            x12 = x0 + 7.0
            x13 = x4 + 9.0
            x14 = xi[1] - 1.0
            x15 = x4 - 1.0
            x16 = x0 - 1.0
            x17 = x4 - 3.0
            x18 = x0 - 3.0
            x19 = x4 - 5.0
            x20 = x0 - 5.0
            x21 = x4 - 7.0
            x22 = x0 - 7.0
            x23 = x4 - 9.0
            x24 = x0 - 9.0
            x25 = x10*x11*x12*x13*x14*x15*x16*x17*x18*x19*x20*x21*x22*x23*x24*x5*x6*x7*x8*x9
            x26 = 1.8105646200481947198e-20*x25
            x27 = xi[0] + 1.0
            x28 = x1*x27
            x29 = xi[1] + 1.0
            x30 = x13*x29
            x31 = x28*x30
            x32 = x10*x11*x12*x15*x16*x17*x18*x19*x20*x21*x22*x23*x24*x5*x6*x7*x8*x9
            x33 = 1.8105646200481947198e-20*x32
            x34 = x3*x30
            x35 = 2.1907831902583156109e-18*x27
            x36 = x14*x3
            x37 = x35*x36
            x38 = x12*x15*x16*x17*x18*x19*x20*x21*x22*x23*x24*x5*x6*x7*x8*x9
            x39 = x36*x38
            x40 = x10*x27
            x41 = 1.0953915951291578055e-17*x40
            x42 = x13*x41
            x43 = x15*x16*x17*x18*x19*x20*x21*x22*x23*x24*x5*x6*x7*x8
            x44 = x36*x43
            x45 = 3.2861747853874734164e-17*x11
            x46 = x12*x40
            x47 = x45*x46
            x48 = x13*x47
            x49 = x20*x36
            x50 = x13*x49
            x51 = x15*x16*x17*x18*x19*x21*x22*x23*x24*x5*x6*x8
            x52 = x11*x9
            x53 = x46*x52
            x54 = x51*x53
            x55 = 6.5723495707749468328e-17*x54
            x56 = x15*x16*x17*x18*x21*x22*x23*x24*x6*x7*x8
            x57 = x19*x53
            x58 = x50*x57
            x59 = 9.2012893990849255659e-17*x58
            x60 = x16*x17*x18*x21*x22*x5*x6*x7*x8
            x61 = x23*x24
            x62 = x60*x61
            x63 = x61*x8
            x64 = x15*x16*x18*x21*x22*x5*x6*x7
            x65 = 6.5723495707749468328e-17*x64
            x66 = x63*x65
            x67 = x49*x5
            x68 = x56*x9
            x69 = x16*x17*x18*x22
            x70 = x6*x69
            x71 = x63*x70
            x72 = x12*x7
            x73 = x15*x19
            x74 = x52*x73
            x75 = x72*x74
            x76 = x71*x75
            x77 = x10*x20
            x78 = x12*x77
            x79 = x24*x60
            x80 = x74*x79
            x81 = x78*x80
            x82 = x29*x35
            x83 = x14*x31
            x84 = x52*x83
            x85 = 1.0953915951291578055e-17*x10*x43
            x86 = x45*x83
            x87 = x17*x61
            x88 = x19*x87
            x89 = x65*x78*x88
            x90 = x21*x5
            x91 = x69*x90
            x92 = x63*x91
            x93 = x75*x77
            x94 = x83*x93
            x95 = 9.2012893990849255659e-17*x94
            x96 = x17*x22*x6*x90
            x97 = x18*x63
            x98 = x96*x97
            x99 = x16*x94
            x100 = x63*x96
            x101 = 6.5723495707749468328e-17*x100
            x102 = x51*x9
            x103 = x10*x102*x72
            x104 = x17*x6
            x105 = x90*x97
            x106 = x104*x105
            x107 = 1.0953915951291578055e-17*x106
            x108 = x23*x60
            x109 = 2.1907831902583156109e-18*x108*x74*x78
            x110 = x20*x34
            x111 = x110*x5
            x112 = x111*x76
            x113 = x110*x57
            x114 = 9.2012893990849255659e-17*x113
            x115 = x34*x43
            x116 = x34*x38
            x117 = x14*x34
            x118 = x117*x93
            x119 = x118*x16
            x120 = 9.2012893990849255659e-17*x118
            x121 = x117*x52
            x122 = x116*x14
            x123 = x2*x29
            x124 = 2.6508476602125618892e-16*x14
            x125 = x124*x15
            x126 = x2*x30
            x127 = x126*x20
            x128 = x57*x79
            x129 = x15*x57
            x130 = x129*x29
            x131 = x126*x14
            x132 = 1.3254238301062809446e-15*x40
            x133 = 3.9762714903188428338e-15*x11
            x134 = x133*x46
            x135 = x127*x14
            x136 = 7.9525429806376856677e-15*x135
            x137 = x135*x56
            x138 = 1.1133560172892759935e-14*x57
            x139 = x57*x64
            x140 = x139*x63
            x141 = 3.9762714903188428338e-15*x53
            x142 = 1.3254238301062809446e-15*x7
            x143 = x129*x142
            x144 = x5*x71
            x145 = x110*x14
            x146 = x145*x80
            x147 = x12*x27
            x148 = x113*x14
            x149 = x148*x24
            x150 = 7.9525429806376856677e-15*x149
            x151 = 1.1133560172892759935e-14*x7
            x152 = x15*x8
            x153 = x149*x152
            x154 = x151*x153
            x155 = x16*x7
            x156 = x104*x142*x16
            x157 = x23*x8
            x158 = x111*x14
            x159 = x158*x70
            x160 = x108*x145
            x161 = 7.9525429806376856677e-15*x148
            x162 = x151*x21
            x163 = x148*x70
            x164 = x152*x23
            x165 = x160*x73
            x166 = x46*x9
            x167 = x130*x67
            x168 = x21*x97
            x169 = x167*x63
            x170 = x104*x155
            x171 = x170*x22
            x172 = x171*x21
            x173 = x104*x22
            x174 = x173*x97
            x175 = 6.6271191505314047231e-15*x40
            x176 = x115*x14
            x177 = x158*x7
            x178 = x129*x158
            x179 = x106*x155
            x180 = x145*x166
            x181 = x180*x73
            x182 = 1.9881357451594214169e-14*x40
            x183 = x11*x176
            x184 = x145*x52
            x185 = x184*x40
            x186 = 3.9762714903188428338e-14*x185
            x187 = x19*x56
            x188 = 5.5667800864463799674e-14*x185
            x189 = x19*x62
            x190 = x19*x63*x64
            x191 = x158*x52*x56
            x192 = 1.9881357451594214169e-14*x27
            x193 = x129*x7
            x194 = x63*x69
            x195 = 5.5667800864463799674e-14*x129*x177
            x196 = 1.9881357451594214169e-14*x117
            x197 = 1.9881357451594214169e-14*x179
            x198 = x145*x53
            x199 = x15*x198
            x200 = 3.9762714903188428338e-14*x155
            x201 = x148*x15
            x202 = x201*x6
            x203 = x105*x202
            x204 = x168*x201
            x205 = x16*x201
            x206 = x11*x46
            x207 = x145*x206
            x208 = x207*x73
            x209 = x7*x98
            x210 = 5.5667800864463799674e-14*x181
            x211 = x7*x92
            x212 = x64*x88
            x213 = 5.9644072354782642507e-14*x147
            x214 = 5.9644072354782642507e-14*x117
            x215 = x147*x184
            x216 = 1.1928814470956528501e-13*x215
            x217 = 1.6700340259339139902e-13*x215
            x218 = 1.6700340259339139902e-13*x199
            x219 = x100*x155
            x220 = 1.1928814470956528501e-13*x219
            x221 = 1.1928814470956528501e-13*x117
            x222 = 1.6700340259339139902e-13*x117*x57
            x223 = 1.6700340259339139902e-13*x208
            x224 = 2.3857628941913057003e-13*x61
            x225 = x15*x163
            x226 = x155*x22
            x227 = 3.3400680518678279804e-13*x7
            x228 = x227*x61
            x229 = 3.3400680518678279804e-13*x201
            x230 = 4.6760952726149591726e-13*x7
            x231 = 4.6760952726149591726e-13*x148
            return jnp.asarray([x26*x3, -x26*x28, x31*x33, -x33*x34, -x32*x37, x39*x42, -x44*x48, x50*x55, -x56*x59, x59*x62, -x58*x66, x48*x67*x68, -x42*x67*x76, x13*x37*x81, x25*x82, -x84*x85, x38*x86, -x84*x89, x92*x95, -x95*x98, x101*x99, -x103*x86, x107*x99, -x109*x83, -x34*x35*x81, x112*x41, -x111*x47*x68, x113*x66, -x114*x62, x114*x56, -x110*x55, x115*x47, -x116*x41, x3*x32*x82, x109*x117, -x107*x119, x103*x117*x45, -x101*x119, x120*x98, -x120*x92, x121*x89, -x122*x45, x121*x85, -2.1907831902583156109e-18*x123*x25, x123*x124*x27*x32, -x125*x127*x128, x113*x125*x60, -2.6508476602125618892e-16*x108*x130*x49, -x131*x132*x38, x131*x134*x43, -x136*x54, x137*x138, -x135*x138*x62, x136*x140, -x137*x141*x5, x135*x143*x144, x132*x146, -3.9762714903188428338e-15*x146*x147, x150*x17*x64, -x154*x91, x154*x18*x96, -x150*x152*x155*x96, 3.9762714903188428338e-15*x117*x128*x15, -x153*x156*x18*x90, -x143*x157*x159, x141*x15*x160, -x157*x161*x64, 1.1133560172892759935e-14*x108*x148, -x162*x163*x164, x161*x164*x70*x90, -x134*x165, 1.3254238301062809446e-15*x165*x166, x156*x167*x168, -3.9762714903188428338e-15*x29*x36*x54*x7, 7.9525429806376856677e-15*x169*x172, -x162*x167*x174, x162*x169*x69, -7.9525429806376856677e-15*x139*x29*x49*x87, x133*x27*x29*x39, -x132*x29*x44*x52, x175*x176*x9, -x175*x177*x71*x74, 6.6271191505314047231e-15*x170*x178*x97, -6.6271191505314047231e-15*x179*x181, -x182*x183, x186*x51, -x187*x188, x188*x189, -x186*x190, x182*x191, x112*x14*x192, -3.9762714903188428338e-14*x159*x193*x61, x194*x195, -x174*x195, 3.9762714903188428338e-14*x171*x178*x63, -x144*x193*x196, -x197*x199, x200*x203, -5.5667800864463799674e-14*x148*x179, 5.5667800864463799674e-14*x170*x204, -3.9762714903188428338e-14*x106*x205, x197*x208, x102*x196*x46*x7, -x100*x181*x200, x209*x210, -x210*x211, 3.9762714903188428338e-14*x180*x212, -x122*x192, x183*x213, -x191*x213, x214*x5*x53*x56, -x206*x214*x51*x7, -x216*x51, x187*x217, -x189*x217, x190*x216, 1.1928814470956528501e-13*x198*x64*x87, -x211*x218, x209*x218, -x199*x220, -x140*x221, x222*x62, -x222*x56, x221*x54, x208*x220, -x209*x223, x211*x223, -1.1928814470956528501e-13*x207*x212, x224*x225*x90, -x148*x224*x64, 2.3857628941913057003e-13*x202*x226*x63*x90, -2.3857628941913057003e-13*x100*x205, -x21*x225*x228, x163*x228*x90, x105*x226*x229, -x203*x22*x227, -3.3400680518678279804e-13*x148*x219, x172*x229*x63, x229*x98, -x229*x92, x194*x201*x21*x230, -x211*x231, x209*x231, -x173*x204*x230])

        case 169:
          def shape_functions(xi):
            x0 = xi[0] - 1.0
            x1 = xi[1] - 1.0
            x2 = 2.0*xi[1]
            x3 = x2 + 1.0
            x4 = 3.0*xi[1]
            x5 = x4 + 1.0
            x6 = 6.0*xi[1]
            x7 = x6 + 1.0
            x8 = x4 + 2.0
            x9 = x6 + 5.0
            x10 = x2 - 1.0
            x11 = x4 - 1.0
            x12 = x6 - 1.0
            x13 = x4 - 2.0
            x14 = x6 - 5.0
            x15 = x10*x11*x12*x13*x14*x3*x5*x7*x8*x9*xi[1]
            x16 = x1*x15
            x17 = 2.0*xi[0]
            x18 = x17 + 1.0
            x19 = 3.0*xi[0]
            x20 = x19 + 1.0
            x21 = 6.0*xi[0]
            x22 = x21 + 1.0
            x23 = x19 + 2.0
            x24 = x21 + 5.0
            x25 = x17 - 1.0
            x26 = x19 - 1.0
            x27 = x21 - 1.0
            x28 = x19 - 2.0
            x29 = x21 - 5.0
            x30 = x18*x20*x22*x23*x24*x25*x26*x27*x28*x29*xi[0]
            x31 = 1.1712674050336387999e-10*x30
            x32 = x16*x31
            x33 = xi[0] + 1.0
            x34 = xi[1] + 1.0
            x35 = x15*x31*x34
            x36 = x16*x33
            x37 = x0*x18*x20*x22*x23*x25*x26*x27*x28*xi[0]
            x38 = 8.4331253162421993591e-9*x37
            x39 = x36*x38
            x40 = x24*x36
            x41 = x0*x18*x20*x22*x25*x26*x27*x29*xi[0]
            x42 = 2.3191094619666048237e-8*x41
            x43 = x40*x42
            x44 = x23*x40
            x45 = x0*x20*x22*x26*x27*x28*x29*xi[0]
            x46 = 5.1535765821480107194e-8*x45
            x47 = x44*x46
            x48 = x18*x44
            x49 = x0*x22*x25*x27*x28*x29*xi[0]
            x50 = 1.7393320964749536178e-7*x49
            x51 = x48*x50
            x52 = x0*x25*x26*x28*x29*xi[0]
            x53 = x27*x52
            x54 = 5.565862708719851577e-7*x20
            x55 = x48*x54
            x56 = 4199.04*xi[0]**12 - 10614.24*xi[0]**10 + 9729.72*xi[0]**8 - 4002.57*xi[0]**6 + 740.74*xi[0]**4 - 53.69*xi[0]**2 + 1.0
            x57 = x1*x56
            x58 = 0.000010822510822510822511*x15
            x59 = x22*x52
            x60 = x10*x11*x12*x13*x14*x3*x34*x5*x7*xi[1]
            x61 = x1*x30
            x62 = x33*x61
            x63 = x60*x62
            x64 = 8.4331253162421993591e-9*x8
            x65 = 2.3191094619666048237e-8*x9
            x66 = x10*x11*x12*x13*x34*x5*x7*x9*xi[1]
            x67 = x62*x66
            x68 = x14*x8
            x69 = 5.1535765821480107194e-8*x68
            x70 = x10*x11*x12*x34*x68*x7*xi[1]
            x71 = x3*x62
            x72 = x70*x71
            x73 = x13*x9
            x74 = 1.7393320964749536178e-7*x73
            x75 = x11*x12*x34*x73*xi[1]
            x76 = x10*x68
            x77 = x75*x76
            x78 = x5*x71
            x79 = 5.565862708719851577e-7*x78
            x80 = 4199.04*xi[1]**12 - 10614.24*xi[1]**10 + 9729.72*xi[1]**8 - 4002.57*xi[1]**6 + 740.74*xi[1]**4 - 53.69*xi[1]**2 + 1.0
            x81 = x33*x80
            x82 = 0.000010822510822510822511*x30
            x83 = x11*x73
            x84 = x34*x76*xi[1]
            x85 = x7*x84
            x86 = x83*x85
            x87 = x12*x85
            x88 = x74*x87
            x89 = x7*x75
            x90 = x69*x89
            x91 = x5*x65
            x92 = x3*x64
            x93 = x24*x33
            x94 = x15*x34*x93
            x95 = x23*x94
            x96 = x18*x95
            x97 = x50*x96
            x98 = x54*x96
            x99 = x29*x33
            x100 = x0*x61
            x101 = x100*x66
            x102 = x100*x3
            x103 = x102*x70
            x104 = x102*x5
            x105 = 5.565862708719851577e-7*x104
            x106 = x100*x60
            x107 = x1*x37
            x108 = x107*x99
            x109 = x60*x8
            x110 = 6.0718502276943835385e-7*x109
            x111 = x107*x93
            x112 = x111*x66
            x113 = x3*x8
            x114 = 6.0718502276943835385e-7*x113
            x115 = x108*x66
            x116 = x1*x93
            x117 = x109*x116
            x118 = 1.6697588126159554731e-6*x41
            x119 = x117*x118
            x120 = x117*x23
            x121 = 3.710575139146567718e-6*x45
            x122 = x120*x121
            x123 = x120*x18
            x124 = 0.000012523191094619666048*x49
            x125 = x123*x124
            x126 = 0.000040074211502782931354*x20
            x127 = x123*x126
            x128 = x57*x60
            x129 = 0.00077922077922077922078*x8
            x130 = 1.6697588126159554731e-6*x9
            x131 = x111*x130
            x132 = 3.710575139146567718e-6*x68
            x133 = x111*x3
            x134 = 0.000012523191094619666048*x73
            x135 = x133*x134
            x136 = x133*x5
            x137 = 0.000040074211502782931354*x136
            x138 = x24*x81
            x139 = 0.00077922077922077922078*x37
            x140 = x5*x87
            x141 = x132*x89
            x142 = x3*x5
            x143 = x142*x70
            x144 = x116*x23
            x145 = x144*x66
            x146 = x113*x145
            x147 = x146*x18
            x148 = x124*x147
            x149 = x126*x147
            x150 = x3*x57
            x151 = x116*x28
            x152 = x151*x66
            x153 = x108*x130
            x154 = x108*x142
            x155 = x140*x3
            x156 = x108*x134
            x157 = 0.000040074211502782931354*x154
            x158 = x3*x70
            x159 = x151*x41
            x160 = x60*x9
            x161 = 4.591836734693877551e-6*x160
            x162 = x144*x41
            x163 = x143*x9
            x164 = 4.591836734693877551e-6*x163
            x165 = x25*x45
            x166 = x144*x160
            x167 = 0.000010204081632653061224*x166
            x168 = x166*x18
            x169 = 0.000034438775510204081633*x49
            x170 = x168*x169
            x171 = 0.00011020408163265306122*x20
            x172 = x168*x171
            x173 = 0.0021428571428571428571*x9
            x174 = x18*x45
            x175 = 0.000010204081632653061224*x68
            x176 = x175*x41
            x177 = 0.000034438775510204081633*x73
            x178 = x162*x177
            x179 = x142*x162
            x180 = 0.00011020408163265306122*x179
            x181 = x138*x23
            x182 = 0.0021428571428571428571*x41
            x183 = x175*x89
            x184 = x144*x163
            x185 = 0.000010204081632653061224*x184
            x186 = x18*x184
            x187 = x169*x186
            x188 = x171*x186
            x189 = x150*x5
            x190 = x142*x159
            x191 = x159*x177
            x192 = 0.00011020408163265306122*x190
            x193 = 0.000022675736961451247166*x68
            x194 = x145*x193
            x195 = x144*x174
            x196 = x142*x195
            x197 = x193*x89
            x198 = x142*x144
            x199 = x165*x198
            x200 = x26*x49
            x201 = x18*x68
            x202 = x145*x201
            x203 = 0.000076530612244897959184*x202
            x204 = 0.00024489795918367346939*x20
            x205 = x202*x204
            x206 = 0.0047619047619047619048*x68
            x207 = x20*x49
            x208 = 0.000076530612244897959184*x73
            x209 = x195*x208
            x210 = 0.00024489795918367346939*x196
            x211 = x18*x181
            x212 = 0.0047619047619047619048*x45
            x213 = x198*x207
            x214 = x201*x89
            x215 = 0.000076530612244897959184*x214
            x216 = x198*x59
            x217 = x204*x214
            x218 = x189*x7
            x219 = x198*x53
            x220 = x198*x200
            x221 = x144*x155
            x222 = x165*x208
            x223 = 0.00024489795918367346939*x199
            x224 = x144*x158
            x225 = x18*x73
            x226 = 0.00025829081632653061224*x225
            x227 = x224*x226
            x228 = x221*x226
            x229 = 0.00082653061224489795918*x20*x225
            x230 = x224*x229
            x231 = 0.016071428571428571429*x73
            x232 = 0.00082653061224489795918*x18
            x233 = x213*x232
            x234 = x20*x211
            x235 = 0.016071428571428571429*x49
            x236 = x221*x229
            x237 = x218*x84
            x238 = x220*x232
            x239 = 0.0026448979591836734694*x18*x20
            x240 = x239*x77
            x241 = x239*x86
            x242 = 0.051428571428571428571*x234
            return jnp.asarray([x0*x32, x32*x33, x33*x35, x0*x35, -x29*x39, x28*x43, -x25*x47, x26*x51, -x53*x55, x57*x58, -x55*x59, x20*x51, -x18*x47, x23*x43, -x24*x39, -x63*x64, x63*x65, -x67*x69, x72*x74, -x77*x79, x81*x82, -x79*x86, x78*x88, -x78*x90, x72*x91, -x67*x92, -x38*x94, x42*x95, -x46*x96, x20*x97, -x59*x98, x34*x56*x58, -x53*x98, x26*x97, -x25*x46*x95, x28*x42*x94, -x15*x34*x38*x99, -x101*x92, x103*x91, -x104*x90, x104*x88, -x105*x86, x0*x80*x82, -x105*x77, x103*x74, -x101*x69, x106*x65, -x106*x64, x108*x110, x110*x111, x112*x114, x114*x115, -x119*x28, x122*x25, -x125*x26, x127*x53, -x128*x129, x127*x59, -x125*x20, x122*x18, -x119*x23, -x131*x60, x112*x132, -x135*x70, x137*x77, -x138*x139, x137*x86, -x135*x140, x136*x141, -x131*x143, -x118*x146, x121*x147, -x148*x20, x149*x59, -x129*x150*x66, x149*x53, -x148*x26, x121*x146*x25, -x113*x118*x152, -x143*x153, x141*x154, -x155*x156, x157*x86, -x139*x29*x81, x157*x77, -x156*x158, x115*x132, -x153*x60, x159*x161, x161*x162, x162*x164, x159*x164, -x165*x167, x170*x26, -x172*x53, x128*x173, -x172*x59, x170*x20, -x167*x174, -x145*x176, x158*x178, -x180*x77, x181*x182, -x180*x86, x155*x178, -x179*x183, -x174*x185, x187*x20, -x188*x59, x173*x189*x70, -x188*x53, x187*x26, -x165*x185, -x183*x190, x155*x191, -x192*x86, x138*x182*x28, -x192*x77, x158*x191, -x152*x176, x165*x194, x174*x194, x196*x197, x197*x199, -x200*x203, x205*x53, -x206*x57*x66, x205*x59, -x203*x207, -x158*x209, x210*x77, -x211*x212, x210*x86, -x155*x209, -x213*x215, x216*x217, -x206*x218*x75, x217*x219, -x215*x220, -x221*x222, x223*x86, -x181*x212*x25, x223*x77, -x222*x224, x200*x227, x207*x227, x207*x228, x200*x228, -x230*x53, x150*x231*x70, -x230*x59, -x233*x77, x234*x235, -x233*x86, -x236*x59, x12*x231*x237, -x236*x53, -x238*x86, x211*x235*x26, -x238*x77, x219*x240, x216*x240, x216*x241, x219*x241, -0.051428571428571428571*x189*x77, -x242*x59, -0.051428571428571428571*x237*x83, -x242*x53, x56*x80])

        case 196:
          def shape_functions(xi):
            x0 = 13.0*xi[1]
            x1 = x0 + 11.0
            x2 = xi[0] - 1.0
            x3 = x1*x2
            x4 = 13.0*xi[0]
            x5 = x4 + 1.0
            x6 = x0 + 1.0
            x7 = x4 + 3.0
            x8 = x0 + 3.0
            x9 = x4 + 5.0
            x10 = x0 + 5.0
            x11 = x4 + 7.0
            x12 = x0 + 7.0
            x13 = x4 + 9.0
            x14 = x0 + 9.0
            x15 = x4 + 11.0
            x16 = xi[1] - 1.0
            x17 = x4 - 1.0
            x18 = x0 - 1.0
            x19 = x4 - 3.0
            x20 = x0 - 3.0
            x21 = x4 - 5.0
            x22 = x0 - 5.0
            x23 = x4 - 7.0
            x24 = x0 - 7.0
            x25 = x4 - 9.0
            x26 = x0 - 9.0
            x27 = x4 - 11.0
            x28 = x0 - 11.0
            x29 = x10*x11*x12*x13*x14*x15*x16*x17*x18*x19*x20*x21*x22*x23*x24*x25*x26*x27*x28*x5*x6*x7*x8*x9
            x30 = 6.4945069302692935024e-26*x29
            x31 = xi[0] + 1.0
            x32 = x1*x31
            x33 = xi[1] + 1.0
            x34 = x15*x33
            x35 = x32*x34
            x36 = x10*x11*x12*x13*x14*x17*x18*x19*x20*x21*x22*x23*x24*x25*x26*x27*x28*x5*x6*x7*x8*x9
            x37 = 6.4945069302692935024e-26*x36
            x38 = x3*x34
            x39 = 1.0975716712155106019e-23*x31
            x40 = x16*x3
            x41 = x39*x40
            x42 = x10*x11*x14*x17*x18*x19*x20*x21*x22*x23*x24*x25*x26*x27*x28*x5*x6*x7*x8*x9
            x43 = x40*x42
            x44 = x12*x31
            x45 = 6.5854300272930636114e-23*x44
            x46 = x15*x45
            x47 = x10*x17*x18*x19*x20*x21*x22*x23*x24*x25*x26*x27*x28*x5*x6*x7*x8*x9
            x48 = x40*x47
            x49 = 2.4146576766741233242e-22*x13
            x50 = x14*x44
            x51 = x49*x50
            x52 = x15*x51
            x53 = x24*x40
            x54 = x15*x53
            x55 = x10*x17*x18*x19*x20*x21*x22*x23*x25*x26*x27*x28*x5*x6*x7*x8
            x56 = x11*x13
            x57 = x50*x56
            x58 = x55*x57
            x59 = 6.0366441916853083105e-22*x58
            x60 = x10*x17*x18*x19*x20*x21*x22*x25*x26*x27*x28*x5*x6*x8*x9
            x61 = x23*x57
            x62 = x54*x61
            x63 = 1.0865959545033554959e-21*x62
            x64 = x10*x17*x18*x19*x20*x21*x22*x25*x26*x6*x7*x8*x9
            x65 = x27*x28
            x66 = x62*x65
            x67 = 1.4487946060044739945e-21*x66
            x68 = x10*x18*x19*x20*x21*x22*x5*x6*x7*x8*x9
            x69 = x25*x26
            x70 = x68*x69
            x71 = x10*x18*x20*x22*x5*x6*x69*x7*x8*x9
            x72 = x17*x65
            x73 = x21*x72
            x74 = x71*x73
            x75 = 6.0366441916853083105e-22*x19
            x76 = x17*x71
            x77 = x60*x7
            x78 = x11*x77
            x79 = x23*x56
            x80 = x14*x79
            x81 = x26*x68
            x82 = x72*x81
            x83 = x80*x82
            x84 = x12*x80
            x85 = x24*x64
            x86 = x5*x85
            x87 = x28*x86
            x88 = x84*x87
            x89 = x33*x39
            x90 = x16*x35
            x91 = x12*x90
            x92 = 6.5854300272930636114e-23*x47*x56
            x93 = x42*x49
            x94 = x18*x20*x22*x5*x6*x69*x7*x9
            x95 = x73*x8
            x96 = x94*x95
            x97 = x84*x90
            x98 = x24*x75
            x99 = x97*x98
            x100 = x73*x94
            x101 = x24*x97
            x102 = x10*x19
            x103 = x101*x102
            x104 = 1.0865959545033554959e-21*x103
            x105 = x18*x95
            x106 = x20*x22*x5*x69*x7*x9
            x107 = 1.4487946060044739945e-21*x106
            x108 = x103*x107
            x109 = x6*x95
            x110 = x6*x69
            x111 = x22*x5*x7*x9
            x112 = x105*x111
            x113 = x110*x112
            x114 = x10*x20
            x115 = x110*x9
            x116 = x5*x7
            x117 = x105*x116
            x118 = x115*x117
            x119 = x114*x118
            x120 = x55*x9
            x121 = x11*x120
            x122 = x121*x14*x49
            x123 = x25*x68
            x124 = x123*x72
            x125 = 6.5854300272930636114e-23*x124
            x126 = x27*x86
            x127 = 1.0975716712155106019e-23*x126
            x128 = x24*x38
            x129 = x128*x83
            x130 = x38*x61
            x131 = x71*x72
            x132 = x130*x131
            x133 = x128*x61
            x134 = 1.0865959545033554959e-21*x133
            x135 = 1.4487946060044739945e-21*x65
            x136 = x133*x70
            x137 = x130*x85
            x138 = x38*x47
            x139 = x38*x42
            x140 = x16*x84
            x141 = x140*x38
            x142 = x128*x140
            x143 = x16*x38
            x144 = x12*x143
            x145 = x141*x98
            x146 = x102*x142
            x147 = 1.0865959545033554959e-21*x146
            x148 = x107*x146
            x149 = x2*x33
            x150 = 1.8548961243542129172e-21*x16
            x151 = x2*x34
            x152 = x151*x61
            x153 = x5*x64
            x154 = x33*x53*x61
            x155 = x151*x16
            x156 = 1.1129376746125277503e-20*x44
            x157 = 4.0807714735792684179e-20*x155
            x158 = x13*x50
            x159 = 1.0201928683948171045e-19*x24
            x160 = x152*x16
            x161 = x160*x24
            x162 = 1.836347163110670788e-19*x161
            x163 = 2.4484628841475610507e-19*x65
            x164 = x131*x19
            x165 = x57*x77
            x166 = x143*x87
            x167 = 4.0807714735792684179e-20*x31
            x168 = 1.0201928683948171045e-19*x19
            x169 = x21*x8
            x170 = x16*x28
            x171 = x133*x17
            x172 = x170*x171
            x173 = x172*x94
            x174 = 1.836347163110670788e-19*x102
            x175 = x169*x18
            x176 = x172*x175
            x177 = x102*x106
            x178 = 2.4484628841475610507e-19*x177
            x179 = x169*x6
            x180 = x110*x111
            x181 = x115*x20
            x182 = 1.0201928683948171045e-19*x102
            x183 = x116*x182
            x184 = x16*x27
            x185 = x171*x184
            x186 = x126*x143
            x187 = 4.0807714735792684179e-20*x186
            x188 = x133*x184*x76
            x189 = 2.4484628841475610507e-19*x184
            x190 = x22*x5
            x191 = x175*x181
            x192 = x20*x22
            x193 = x110*x192
            x194 = x158*x23
            x195 = x11*x23*x50
            x196 = x118*x20
            x197 = x154*x174
            x198 = x154*x178
            x199 = 6.677626047675166502e-20*x16
            x200 = x199*x44
            x201 = x128*x79
            x202 = x133*x72
            x203 = x124*x128
            x204 = x16*x44
            x205 = 2.4484628841475610507e-19*x204
            x206 = x13*x138
            x207 = 6.1211572103689026268e-19*x204
            x208 = x128*x56
            x209 = x208*x55
            x210 = x201*x204
            x211 = 1.1018082978664024728e-18*x210
            x212 = 1.4690777304885366304e-18*x65
            x213 = x143*x85
            x214 = x208*x77
            x215 = 2.4484628841475610507e-19*x16
            x216 = x215*x31
            x217 = 6.1211572103689026268e-19*x19
            x218 = x20*x26
            x219 = x133*x16
            x220 = x219*x6
            x221 = x112*x220
            x222 = x102*x218
            x223 = x111*x222
            x224 = 1.1018082978664024728e-18*x220
            x225 = x18*x73
            x226 = 1.4690777304885366304e-18*x219
            x227 = 1.1018082978664024728e-18*x221
            x228 = x220*x9
            x229 = 6.1211572103689026268e-19*x117
            x230 = x203*x215
            x231 = x102*x25
            x232 = x20*x231
            x233 = 6.1211572103689026268e-19*x16
            x234 = x202*x8
            x235 = x18*x234
            x236 = x192*x231
            x237 = 1.4690777304885366304e-18*x105
            x238 = x105*x190
            x239 = x238*x9
            x240 = x128*x195
            x241 = x102*x240
            x242 = x113*x16
            x243 = 1.1018082978664024728e-18*x241
            x244 = x16*x240
            x245 = x177*x244
            x246 = x100*x16
            x247 = x16*x31
            x248 = x14*x247
            x249 = 8.9776972418743905194e-19*x248
            x250 = 8.9776972418743905194e-19*x143
            x251 = 4.0399637588434757337e-18*x128
            x252 = x247*x80
            x253 = x251*x252
            x254 = 5.3866183451246343116e-18*x65
            x255 = x128*x252
            x256 = x254*x70
            x257 = x128*x16
            x258 = x257*x57
            x259 = 2.2444243104685976298e-18*x19
            x260 = x259*x96
            x261 = x102*x251
            x262 = x261*x57
            x263 = 5.3866183451246343116e-18*x177
            x264 = x258*x263
            x265 = 2.2444243104685976298e-18*x102*x196
            x266 = x130*x16
            x267 = 4.0399637588434757337e-18*x266
            x268 = x194*x257
            x269 = x194*x261
            x270 = x263*x268
            x271 = 5.6110607761714940746e-18*x19
            x272 = x117*x219
            x273 = x16*x94
            x274 = 5.6110607761714940746e-18*x102
            x275 = x16*x235
            x276 = x110*x272
            x277 = x19*x219
            x278 = x181*x277
            x279 = x22*x7
            x280 = 1.3466545862811585779e-17*x105
            x281 = 1.3466545862811585779e-17*x65
            x282 = 1.0099909397108689334e-17*x219
            x283 = 1.0099909397108689334e-17*x102
            x284 = 1.3466545862811585779e-17*x177
            x285 = x102*x219
            x286 = x116*x285
            x287 = x181*x285
            x288 = x102*x282
            x289 = x109*x69
            x290 = 1.3466545862811585779e-17*x192
            x291 = x225*x287
            x292 = x10*x219
            x293 = 1.8179836914795640802e-17*x292
            x294 = x115*x285
            x295 = 2.4239782553060854402e-17*x279
            x296 = 2.4239782553060854402e-17*x285
            x297 = x296*x65
            x298 = 2.4239782553060854402e-17*x106*x292
            x299 = x20*x296
            x300 = x289*x9
            x301 = 3.231971007074780587e-17*x192*x285*x7
            x302 = 3.231971007074780587e-17*x177*x219*x65
            return jnp.asarray([x3*x30, -x30*x32, x35*x37, -x37*x38, -x36*x41, x43*x46, -x48*x52, x54*x59, -x60*x63, x64*x67, -x67*x70, x63*x74, -x66*x75*x76, x52*x53*x78, -x46*x53*x83, x15*x41*x88, x29*x89, -x91*x92, x90*x93, -x96*x99, x100*x104, -x105*x108, x108*x109, -x104*x113, x119*x99, -x122*x91, x101*x125, -x127*x97, -x38*x39*x88, x129*x45, -x128*x51*x78, x132*x98, -x134*x74, x135*x136, -x135*x137, x134*x60, -x128*x59, x138*x51, -x139*x45, x3*x36*x89, x127*x141, -x125*x142, x122*x144, -x119*x145, x113*x147, -x109*x148, x105*x148, -x100*x147, x145*x96, -x143*x93, x144*x92, -1.0975716712155106019e-23*x149*x29, x149*x150*x31*x36, -x150*x152*x87, x130*x150*x86, -1.8548961243542129172e-21*x153*x154*x27, -x155*x156*x42, x157*x158*x47, -x155*x159*x58, x162*x60, -x160*x163*x85, x161*x163*x70, -x162*x74, x159*x160*x164, -x157*x165*x24, 1.1129376746125277503e-20*x161*x82, x156*x166*x79, -x166*x167*x80, x168*x169*x173, -x173*x174*x21, x176*x178, -x172*x178*x179, x174*x176*x180, -x176*x181*x183, 4.0807714735792684179e-20*x130*x153*x170, -1.1129376746125277503e-20*x123*x172, -1.1129376746125277503e-20*x185*x81, x187*x57, -x168*x188, 1.836347163110670788e-19*x188*x21, -x136*x189, x137*x189, -x174*x185*x190*x191, x175*x183*x185*x193, -x187*x194, 1.1129376746125277503e-20*x186*x195, 1.1129376746125277503e-20*x124*x154, -4.0807714735792684179e-20*x33*x40*x58*x9, x154*x182*x196, -x113*x197, x109*x198, -x105*x198, x100*x197, -x154*x168*x96, x13*x167*x33*x43, -x156*x33*x48*x56, x11*x138*x200, -x200*x201*x82, x199*x202*x68, -x195*x199*x203, -x205*x206, x207*x209, -x211*x60, x212*x213*x44*x79, -x210*x212*x70, x211*x74, -x164*x201*x207, x205*x214, x129*x216, -x217*x218*x221, x223*x224*x225, -x112*x222*x226, x109*x223*x226, -x102*x227*x26, x222*x228*x229, -x130*x215*x82, -x230*x57, x111*x232*x233*x235*x6, -x114*x227*x25, x123*x212*x219, -x228*x236*x237*x7, x224*x232*x239, -x220*x229*x236, x194*x230, 2.4484628841475610507e-19*x121*x143*x50, -x196*x233*x241, x242*x243, -1.4690777304885366304e-18*x109*x245, x237*x245, -x243*x246, x217*x244*x96, -x139*x216, x206*x249, -x214*x249, x165*x250, -x120*x158*x250, -2.2444243104685976298e-18*x209*x248, x253*x60, -x213*x254*x31*x80, x255*x256, -x253*x74, 2.2444243104685976298e-18*x164*x255, x258*x260, -x246*x262, x105*x264, -x109*x264, x242*x262, -x258*x265, -x132*x16*x259, x267*x74, -x256*x266, x254*x266*x64, -x267*x60, 2.2444243104685976298e-18*x143*x58, x265*x268, -x242*x269, x109*x270, -x105*x270, x246*x269, -x260*x268, x193*x271*x272, -x234*x271*x273, x116*x181*x274*x275, -x20*x274*x276, -1.0099909397108689334e-17*x238*x278, x278*x279*x280, -x169*x277*x281*x94, x282*x96, x202*x273*x283, -x275*x284, x16*x234*x284*x6, -x180*x275*x283, -x119*x282, x191*x281*x286, -x280*x287*x7, x105*x181*x288*x5, x22*x276*x283, -x286*x289*x290, x102*x272*x290*x69, -x116*x193*x225*x288, 1.8179836914795640802e-17*x190*x291, -x100*x293, x113*x293, -1.8179836914795640802e-17*x238*x294, -x291*x295, x21*x297*x94, x105*x298, -x109*x298, -x175*x180*x297, x105*x294*x295, x190*x299*x300, -x239*x299*x69, x105*x301*x69*x9, -x175*x302, x179*x302, -x300*x301])

        case 225:
          def shape_functions(xi):
            x0 = xi[0] - 1.0
            x1 = xi[1] - 1.0
            x2 = 7.0*xi[1]
            x3 = x2 + 1.0
            x4 = x2 + 2.0
            x5 = x2 + 4.0
            x6 = x2 + 3.0
            x7 = x2 + 6.0
            x8 = x2 + 5.0
            x9 = x2 - 1.0
            x10 = x2 - 2.0
            x11 = x2 - 4.0
            x12 = x2 - 3.0
            x13 = x2 - 6.0
            x14 = x2 - 5.0
            x15 = x10*x11*x12*x13*x14*x3*x4*x5*x6*x7*x8*x9*xi[1]
            x16 = x1*x15
            x17 = 7.0*xi[0]
            x18 = x17 + 1.0
            x19 = x17 + 2.0
            x20 = x17 + 4.0
            x21 = x17 + 3.0
            x22 = x17 + 6.0
            x23 = x17 + 5.0
            x24 = x17 - 1.0
            x25 = x17 - 2.0
            x26 = x17 - 4.0
            x27 = x17 - 3.0
            x28 = x17 - 6.0
            x29 = x17 - 5.0
            x30 = x18*x19*x20*x21*x22*x23*x24*x25*x26*x27*x28*x29*xi[0]
            x31 = 3.1591878896737389867e-19*x30
            x32 = x16*x31
            x33 = xi[0] + 1.0
            x34 = xi[1] + 1.0
            x35 = x15*x31*x34
            x36 = x16*x33
            x37 = x0*x18*x19*x20*x21*x23*x24*x25*x26*x27*x29*xi[0]
            x38 = 3.096004131880264207e-17*x37
            x39 = x36*x38
            x40 = x22*x36
            x41 = x0*x18*x19*x20*x21*x24*x25*x26*x27*x28*xi[0]
            x42 = 2.0124026857221717346e-16*x41
            x43 = x40*x42
            x44 = x23*x40
            x45 = x0*x18*x19*x21*x24*x25*x27*x28*x29*xi[0]
            x46 = 8.0496107428886869382e-16*x45
            x47 = x44*x46
            x48 = x20*x44
            x49 = x0*x18*x19*x24*x25*x26*x28*x29*xi[0]
            x50 = 2.213642954294388908e-15*x49
            x51 = x48*x50
            x52 = x21*x48
            x53 = x0*x18*x24*x26*x27*x28*x29*xi[0]
            x54 = 4.427285908588777816e-15*x53
            x55 = x52*x54
            x56 = x0*x25*x26*x27*x28*x29*xi[0]
            x57 = x24*x56
            x58 = 6.640928862883166724e-15*x19
            x59 = x52*x58
            x60 = -26700.013890817901235*xi[0]**14 + 76285.753973765432099*xi[0]**12 - 82980.21809799382716*xi[0]**10 + 43487.464081790123457*xi[0]**8 - 11465.29836612654321*xi[0]**6 + 1445.3903549382716049*xi[0]**4 - 74.078055555555555556*xi[0]**2 + 1.0
            x61 = x1*x60
            x62 = 5.6206653428875651098e-10*x15
            x63 = x18*x56
            x64 = x10*x11*x12*x13*x14*x3*x34*x4*x5*x6*x9*xi[1]
            x65 = x1*x30
            x66 = x33*x65
            x67 = x64*x66
            x68 = 3.096004131880264207e-17*x8
            x69 = 2.0124026857221717346e-16*x7
            x70 = x10*x11*x12*x14*x3*x34*x4*x6*x7*x9*xi[1]
            x71 = x66*x70
            x72 = x13*x8
            x73 = 8.0496107428886869382e-16*x72
            x74 = x10*x11*x12*x3*x34*x4*x72*x9*xi[1]
            x75 = x5*x66
            x76 = x74*x75
            x77 = x14*x7
            x78 = 2.213642954294388908e-15*x77
            x79 = x10*x12*x3*x34*x77*x9*xi[1]
            x80 = x6*x75
            x81 = x79*x80
            x82 = x11*x72
            x83 = 4.427285908588777816e-15*x82
            x84 = x10*x34*x82*x9*xi[1]
            x85 = x12*x77
            x86 = x84*x85
            x87 = x4*x80
            x88 = 6.640928862883166724e-15*x87
            x89 = -26700.013890817901235*xi[1]**14 + 76285.753973765432099*xi[1]**12 - 82980.21809799382716*xi[1]**10 + 43487.464081790123457*xi[1]**8 - 11465.29836612654321*xi[1]**6 + 1445.3903549382716049*xi[1]**4 - 74.078055555555555556*xi[1]**2 + 1.0
            x90 = x33*x89
            x91 = 5.6206653428875651098e-10*x30
            x92 = x10*x82
            x93 = x34*x85*xi[1]
            x94 = x3*x93
            x95 = x92*x94
            x96 = x9*x94
            x97 = x83*x96
            x98 = x3*x84
            x99 = x78*x98
            x100 = x4*x73
            x101 = x6*x69
            x102 = x5*x68
            x103 = x22*x33
            x104 = x103*x15*x34
            x105 = x104*x23
            x106 = x105*x20
            x107 = x106*x21
            x108 = x107*x54
            x109 = x107*x58
            x110 = x28*x33
            x111 = x0*x65
            x112 = x111*x70
            x113 = x111*x5
            x114 = x113*x74
            x115 = x113*x6
            x116 = x115*x79
            x117 = x115*x4
            x118 = 6.640928862883166724e-15*x117
            x119 = x111*x64
            x120 = x1*x37
            x121 = x110*x120
            x122 = x64*x8
            x123 = 3.0340840492426589229e-15*x122
            x124 = x103*x120
            x125 = x124*x70
            x126 = x5*x8
            x127 = 3.0340840492426589229e-15*x126
            x128 = x121*x70
            x129 = x1*x103
            x130 = x122*x129
            x131 = 1.9721546320077282999e-14*x41
            x132 = x130*x131
            x133 = x130*x23
            x134 = 7.8886185280309131995e-14*x45
            x135 = x133*x134
            x136 = x133*x20
            x137 = 2.1693700952085011299e-13*x49
            x138 = x136*x137
            x139 = x136*x21
            x140 = 4.3387401904170022597e-13*x53
            x141 = x139*x140
            x142 = 6.5081102856255033896e-13*x19
            x143 = x139*x142
            x144 = x61*x64
            x145 = 5.5082520360298138076e-8*x8
            x146 = 1.9721546320077282999e-14*x7
            x147 = x124*x146
            x148 = 7.8886185280309131995e-14*x72
            x149 = x124*x5
            x150 = 2.1693700952085011299e-13*x77
            x151 = x149*x150
            x152 = x149*x6
            x153 = x152*x79
            x154 = 4.3387401904170022597e-13*x82
            x155 = x152*x4
            x156 = 6.5081102856255033896e-13*x155
            x157 = x22*x90
            x158 = 5.5082520360298138076e-8*x37
            x159 = x154*x96
            x160 = x4*x6
            x161 = x160*x98
            x162 = x5*x6
            x163 = x162*x74
            x164 = x129*x23
            x165 = x164*x70
            x166 = x126*x165
            x167 = x166*x20
            x168 = x167*x21
            x169 = x140*x168
            x170 = x142*x168
            x171 = x5*x61
            x172 = x129*x29
            x173 = x172*x70
            x174 = x121*x146
            x175 = x121*x5
            x176 = x160*x175
            x177 = x150*x175
            x178 = 6.5081102856255033896e-13*x176
            x179 = x162*x79
            x180 = x172*x41
            x181 = x64*x7
            x182 = 1.2819005108050233949e-13*x181
            x183 = x164*x41
            x184 = x163*x7
            x185 = 1.2819005108050233949e-13*x184
            x186 = x26*x45
            x187 = x164*x181
            x188 = 5.1276020432200935797e-13*x187
            x189 = x187*x20
            x190 = 1.4100905618855257344e-12*x49
            x191 = x189*x190
            x192 = x189*x21
            x193 = 2.8201811237710514688e-12*x53
            x194 = x192*x193
            x195 = 4.2302716856565772032e-12*x19
            x196 = x192*x195
            x197 = 3.5803638234193789749e-7*x7
            x198 = x20*x45
            x199 = 5.1276020432200935797e-13*x72
            x200 = x199*x41
            x201 = x183*x5
            x202 = 1.4100905618855257344e-12*x77
            x203 = x201*x202
            x204 = 2.8201811237710514688e-12*x82
            x205 = x179*x204
            x206 = x160*x201
            x207 = 4.2302716856565772032e-12*x206
            x208 = x157*x23
            x209 = 3.5803638234193789749e-7*x41
            x210 = x204*x96
            x211 = x199*x79
            x212 = x164*x184
            x213 = 5.1276020432200935797e-13*x212
            x214 = x20*x21
            x215 = x212*x214
            x216 = x193*x215
            x217 = x195*x215
            x218 = x171*x6
            x219 = x20*x27
            x220 = x180*x5
            x221 = x160*x220
            x222 = x202*x220
            x223 = 4.2302716856565772032e-12*x221
            x224 = 2.0510408172880374319e-12*x72
            x225 = x165*x224
            x226 = x164*x198
            x227 = x226*x5
            x228 = x160*x227
            x229 = x224*x79
            x230 = x164*x186
            x231 = x160*x5
            x232 = x230*x231
            x233 = x219*x49
            x234 = x165*x72
            x235 = 5.6403622475421029376e-12*x234
            x236 = x214*x234
            x237 = 1.1280724495084205875e-11*x53
            x238 = x236*x237
            x239 = 1.6921086742626308813e-11*x19
            x240 = x236*x239
            x241 = 1.43214552936775159e-6*x72
            x242 = x214*x49
            x243 = 5.6403622475421029376e-12*x77
            x244 = x227*x243
            x245 = 1.1280724495084205875e-11*x82
            x246 = x179*x245
            x247 = 1.6921086742626308813e-11*x228
            x248 = x20*x208
            x249 = 1.43214552936775159e-6*x45
            x250 = x245*x96
            x251 = x164*x242
            x252 = x231*x251
            x253 = x72*x79
            x254 = 5.6403622475421029376e-12*x253
            x255 = x19*x214
            x256 = x164*x231
            x257 = x253*x256
            x258 = x237*x257
            x259 = x214*x239*x257
            x260 = x218*x4
            x261 = x214*x25
            x262 = x233*x256
            x263 = x161*x5
            x264 = x230*x243
            x265 = 1.6921086742626308813e-11*x232
            x266 = x5*x74
            x267 = x164*x233
            x268 = 1.5510996180740783078e-11*x77
            x269 = x266*x268
            x270 = x263*x268
            x271 = x261*x53
            x272 = x164*x77
            x273 = x266*x272
            x274 = 3.1021992361481566157e-11*x273
            x275 = 4.6532988542222349235e-11*x255
            x276 = x273*x275
            x277 = 3.9384002057613168724e-6*x77
            x278 = x255*x53
            x279 = 3.1021992361481566157e-11*x82
            x280 = x179*x279
            x281 = 4.6532988542222349235e-11*x252
            x282 = x21*x248
            x283 = 3.9384002057613168724e-6*x49
            x284 = x279*x96
            x285 = x263*x272
            x286 = 3.1021992361481566157e-11*x285
            x287 = x275*x285
            x288 = x260*x3
            x289 = 4.6532988542222349235e-11*x262
            x290 = 6.2043984722963132314e-11*x82
            x291 = x271*x290
            x292 = x164*x179
            x293 = x278*x290
            x294 = x256*x96
            x295 = x255*x57
            x296 = 9.3065977084444698471e-11*x82
            x297 = x292*x296
            x298 = 7.8768004115226337449e-6*x82
            x299 = x255*x63
            x300 = 9.3065977084444698471e-11*x256
            x301 = x278*x300
            x302 = x19*x282
            x303 = 7.8768004115226337449e-6*x53
            x304 = x294*x296
            x305 = x288*x93
            x306 = x271*x300
            x307 = 1.3959896562666704771e-10*x256
            x308 = x307*x86
            x309 = x307*x95
            x310 = 0.000011815200617283950617*x302
            return jnp.asarray([x0*x32, x32*x33, x33*x35, x0*x35, -x28*x39, x29*x43, -x26*x47, x27*x51, -x25*x55, x57*x59, x61*x62, x59*x63, -x19*x55, x21*x51, -x20*x47, x23*x43, -x22*x39, -x67*x68, x67*x69, -x71*x73, x76*x78, -x81*x83, x86*x88, x90*x91, x88*x95, -x87*x97, x87*x99, -x100*x81, x101*x76, -x102*x71, -x104*x38, x105*x42, -x106*x46, x107*x50, -x108*x19, x109*x63, x34*x60*x62, x109*x57, -x108*x25, x106*x27*x50, -x105*x26*x46, x104*x29*x42, -x110*x15*x34*x38, -x102*x112, x101*x114, -x100*x116, x117*x99, -x117*x97, x118*x95, x0*x89*x91, x118*x86, -x116*x83, x114*x78, -x112*x73, x119*x69, -x119*x68, x121*x123, x123*x124, x125*x127, x127*x128, -x132*x29, x135*x26, -x138*x27, x141*x25, -x143*x57, -x144*x145, -x143*x63, x141*x19, -x138*x21, x135*x20, -x132*x23, -x147*x64, x125*x148, -x151*x74, x153*x154, -x156*x86, -x157*x158, -x156*x95, x155*x159, -x151*x161, x148*x153*x4, -x147*x163, -x131*x166, x134*x167, -x137*x168, x169*x19, -x170*x63, -x145*x171*x70, -x170*x57, x169*x25, -x137*x167*x27, x134*x166*x26, -x126*x131*x173, -x163*x174, x148*x176*x79, -x161*x177, x159*x176, -x178*x95, -x158*x28*x90, -x178*x86, x121*x154*x179, -x177*x74, x128*x148, -x174*x64, x180*x182, x182*x183, x183*x185, x180*x185, -x186*x188, x191*x27, -x194*x25, x196*x57, x144*x197, x196*x63, -x19*x194, x191*x21, -x188*x198, -x165*x200, x203*x74, -x183*x205, x207*x86, x208*x209, x207*x95, -x206*x210, x161*x203, -x206*x211, -x198*x213, x190*x215, -x19*x216, x217*x63, x197*x218*x74, x217*x57, -x216*x25, x190*x212*x219, -x186*x213, -x211*x221, x161*x222, -x210*x221, x223*x95, x157*x209*x29, x223*x86, -x180*x205, x222*x74, -x173*x200, x186*x225, x198*x225, x228*x229, x229*x232, -x233*x235, x238*x25, -x240*x57, -x241*x61*x70, -x240*x63, x19*x238, -x235*x242, -x244*x74, x226*x246, -x247*x86, -x248*x249, -x247*x95, x228*x250, -x161*x244, -x252*x254, x255*x258, -x259*x63, -x241*x260*x79, -x259*x57, x258*x261, -x254*x262, -x263*x264, x232*x250, -x265*x95, -x208*x249*x26, -x265*x86, x230*x246, -x264*x266, x267*x269, x251*x269, x251*x270, x267*x270, -x271*x274, x276*x57, x171*x277*x74, x276*x63, -x274*x278, -x251*x280, x281*x86, x282*x283, x281*x95, -x252*x284, -x278*x286, x287*x63, x277*x288*x84, x287*x57, -x271*x286, -x262*x284, x289*x95, x248*x27*x283, x289*x86, -x267*x280, x291*x292, x292*x293, x293*x294, x291*x294, -x295*x297, -x218*x298*x79, -x297*x299, -x301*x86, -x302*x303, -x301*x95, -x299*x304, -x298*x305*x9, -x295*x304, -x306*x95, -x25*x282*x303, -x306*x86, x295*x308, x299*x308, x299*x309, x295*x309, 0.000011815200617283950617*x260*x86, x310*x63, 0.000011815200617283950617*x305*x92, x310*x57, x60*x89])

        case 256:
          def shape_functions(xi):
            x0 = 15.0*xi[1]
            x1 = x0 + 13.0
            x2 = xi[0] - 1.0
            x3 = x1*x2
            x4 = 3.0*xi[0]
            x5 = x4 + 1.0
            x6 = 3.0*xi[1]
            x7 = x6 + 1.0
            x8 = 5.0*xi[0]
            x9 = x8 + 1.0
            x10 = 5.0*xi[1]
            x11 = x10 + 1.0
            x12 = 15.0*xi[0]
            x13 = x12 + 1.0
            x14 = x0 + 1.0
            x15 = x8 + 3.0
            x16 = x10 + 3.0
            x17 = x12 + 7.0
            x18 = x0 + 7.0
            x19 = x12 + 11.0
            x20 = x0 + 11.0
            x21 = x12 + 13.0
            x22 = xi[1] - 1.0
            x23 = x4 - 1.0
            x24 = x6 - 1.0
            x25 = x8 - 1.0
            x26 = x10 - 1.0
            x27 = x12 - 1.0
            x28 = x0 - 1.0
            x29 = x8 - 3.0
            x30 = x10 - 3.0
            x31 = x12 - 7.0
            x32 = x0 - 7.0
            x33 = x12 - 11.0
            x34 = x0 - 11.0
            x35 = x12 - 13.0
            x36 = x0 - 13.0
            x37 = x11*x13*x14*x15*x16*x17*x18*x19*x20*x21*x22*x23*x24*x25*x26*x27*x28*x29*x30*x31*x32*x33*x34*x35*x36*x5*x7*x9
            x38 = 5.0249700898396030198e-25*x37
            x39 = xi[0] + 1.0
            x40 = x1*x39
            x41 = xi[1] + 1.0
            x42 = x21*x41
            x43 = x40*x42
            x44 = x11*x13*x14*x15*x16*x17*x18*x19*x20*x23*x24*x25*x26*x27*x28*x29*x30*x31*x32*x33*x34*x35*x36*x5*x7*x9
            x45 = 5.0249700898396030198e-25*x44
            x46 = x3*x42
            x47 = 1.1306182702139106794e-22*x39
            x48 = x22*x3
            x49 = x47*x48
            x50 = x11*x13*x14*x15*x17*x18*x20*x23*x24*x25*x26*x27*x28*x29*x30*x31*x32*x33*x34*x35*x36*x5*x7*x9
            x51 = x48*x50
            x52 = x16*x39
            x53 = 7.9143278914973747561e-22*x52
            x54 = x21*x53
            x55 = x11*x13*x14*x17*x18*x23*x24*x25*x26*x27*x28*x29*x30*x31*x32*x33*x34*x35*x36*x5*x7*x9
            x56 = x48*x55
            x57 = 1.143180695438509687e-21*x19
            x58 = x20*x52
            x59 = x57*x58
            x60 = x21*x59
            x61 = x30*x48
            x62 = x21*x61
            x63 = x11*x13*x14*x18*x23*x24*x25*x26*x27*x28*x29*x31*x32*x33*x34*x35*x36*x5*x7*x9
            x64 = x15*x19
            x65 = x58*x64
            x66 = x63*x65
            x67 = 1.0288626258946587183e-20*x66
            x68 = x11*x13*x14*x17*x18*x23*x24*x25*x26*x27*x28*x31*x32*x33*x34*x35*x36*x7*x9
            x69 = x29*x65
            x70 = x62*x69
            x71 = 4.5269955539364983605e-21*x70
            x72 = x11*x13*x14*x17*x18*x23*x24*x25*x26*x27*x28*x31*x32*x33*x34*x5*x7
            x73 = x35*x36
            x74 = x70*x73
            x75 = 1.2574987649823606557e-20*x74
            x76 = x11*x14*x17*x18*x23*x24*x25*x26*x27*x28*x31*x32*x5*x7*x9
            x77 = x33*x34
            x78 = x74*x77
            x79 = 4.8503523792176768148e-20*x78
            x80 = x11*x13*x14*x23*x24*x25*x26*x28*x31*x32*x5*x7*x9
            x81 = x17*x18
            x82 = x80*x81
            x83 = x11*x13*x14*x23*x24*x26*x28*x32*x5*x7*x81*x9
            x84 = x27*x77
            x85 = x31*x84
            x86 = x83*x85
            x87 = x11*x14*x24*x25*x26*x28*x32*x5*x81*x85*x9
            x88 = x13*x73
            x89 = x7*x88
            x90 = x87*x89
            x91 = 1.0288626258946587183e-20*x25
            x92 = x27*x83
            x93 = x5*x68
            x94 = x15*x93
            x95 = x61*x88
            x96 = x34*x76
            x97 = x29*x64
            x98 = x20*x97
            x99 = x96*x98
            x100 = x16*x98
            x101 = x30*x72
            x102 = x101*x9
            x103 = x102*x36
            x104 = x100*x103
            x105 = x41*x47
            x106 = x22*x43
            x107 = x106*x16
            x108 = 7.9143278914973747561e-22*x55*x64
            x109 = x50*x57
            x110 = x100*x106
            x111 = x30*x73
            x112 = x111*x84
            x113 = x17*x80
            x114 = x112*x113
            x115 = 1.0288626258946587183e-20*x114
            x116 = x30*x88
            x117 = x110*x116
            x118 = x23*x87
            x119 = 4.5269955539364983605e-21*x118
            x120 = x14*x23*x24*x26*x28*x32*x5*x81*x9
            x121 = x120*x85
            x122 = x25*x30
            x123 = x122*x89
            x124 = x110*x123
            x125 = 1.2574987649823606557e-20*x124
            x126 = x28*x85
            x127 = x124*x126
            x128 = x11*x23*x24*x26*x32*x5*x81*x9
            x129 = 4.8503523792176768148e-20*x128
            x130 = x14*x85
            x131 = x129*x130
            x132 = x11*x23*x24*x32*x5*x9
            x133 = x14*x81
            x134 = x126*x133
            x135 = x132*x134
            x136 = x11*x26*x5*x9
            x137 = x23*x32
            x138 = x136*x137
            x139 = 4.5269955539364983605e-21*x138
            x140 = x30*x89
            x141 = x134*x24
            x142 = x136*x23
            x143 = x141*x142
            x144 = x140*x143
            x145 = x144*x91
            x146 = x17*x63
            x147 = x146*x15
            x148 = x147*x20*x57
            x149 = x33*x76
            x150 = 7.9143278914973747561e-22*x149
            x151 = x102*x35
            x152 = 1.1306182702139106794e-22*x151
            x153 = x46*x53
            x154 = x116*x99
            x155 = x30*x46
            x156 = x46*x69
            x157 = x112*x83
            x158 = x69*x90
            x159 = 4.5269955539364983605e-21*x155
            x160 = x111*x156
            x161 = x77*x82
            x162 = 4.8503523792176768148e-20*x160
            x163 = x76*x77
            x164 = x101*x73
            x165 = x68*x69
            x166 = x46*x55
            x167 = x22*x46
            x168 = x100*x167
            x169 = x116*x168
            x170 = x16*x167
            x171 = x123*x168
            x172 = 1.2574987649823606557e-20*x171
            x173 = x2*x41
            x174 = 2.5438911079812990288e-20*x22
            x175 = x2*x42
            x176 = x175*x69
            x177 = x72*x9
            x178 = x35*x69
            x179 = x41*x61
            x180 = x175*x22
            x181 = 1.7807237755869093201e-19*x52
            x182 = 2.5721565647366467957e-19*x180
            x183 = x19*x58
            x184 = x180*x30
            x185 = 1.0185739996357121311e-18*x184
            x186 = x176*x22
            x187 = 2.8293722212103114753e-18*x186
            x188 = 1.0913292853239772833e-17*x163
            x189 = x111*x186
            x190 = 1.0913292853239772833e-17*x161
            x191 = x111*x86
            x192 = 2.3149409082629821162e-18*x25
            x193 = x65*x93
            x194 = 1.7807237755869093201e-19*x96
            x195 = x103*x167
            x196 = 2.5721565647366467957e-19*x39
            x197 = x22*x36
            x198 = 2.3149409082629821162e-18*x197
            x199 = x155*x69
            x200 = x113*x84
            x201 = 1.0185739996357121311e-18*x118
            x202 = x13*x197
            x203 = x199*x202
            x204 = 2.8293722212103114753e-18*x121
            x205 = x122*x156
            x206 = x205*x7
            x207 = x202*x206
            x208 = 1.0913292853239772833e-17*x128
            x209 = x207*x208
            x210 = 2.8293722212103114753e-18*x135
            x211 = x134*x138
            x212 = 1.0185739996357121311e-18*x211
            x213 = x13*x206
            x214 = 1.7807237755869093201e-19*x149
            x215 = x155*x22
            x216 = x178*x215
            x217 = x13*x216
            x218 = x151*x167
            x219 = 2.5721565647366467957e-19*x218
            x220 = x22*x35
            x221 = 2.3149409082629821162e-18*x84
            x222 = x11*x137*x26
            x223 = x141*x9
            x224 = x222*x223
            x225 = x18*x80
            x226 = x183*x29
            x227 = x15*x29*x58
            x228 = x41*x69*x95
            x229 = x179*x69
            x230 = x229*x89
            x231 = x230*x25
            x232 = x208*x231
            x233 = 1.2465066429108365241e-18*x52
            x234 = x166*x22
            x235 = x116*x167
            x236 = x156*x22
            x237 = x116*x236
            x238 = x149*x235
            x239 = 1.800509595315652757e-18*x52
            x240 = x19*x234
            x241 = x215*x52
            x242 = x63*x64
            x243 = 7.1300179974499849178e-18*x241*x97
            x244 = 1.9805605548472180327e-17*x167
            x245 = x52*x97
            x246 = x244*x245
            x247 = x111*x163
            x248 = x167*x245
            x249 = 7.6393049972678409834e-17*x248
            x250 = x111*x161
            x251 = 1.6204586357840874813e-17*x25
            x252 = x215*x64*x93
            x253 = 1.800509595315652757e-18*x167
            x254 = x253*x39
            x255 = x27*x34
            x256 = 1.6204586357840874813e-17*x255
            x257 = x160*x22
            x258 = x113*x257
            x259 = x255*x31
            x260 = x120*x259
            x261 = x237*x25
            x262 = x11*x261
            x263 = x123*x236
            x264 = 1.9805605548472180327e-17*x263
            x265 = 7.6393049972678409834e-17*x128
            x266 = x259*x28
            x267 = x263*x266
            x268 = x14*x263
            x269 = x133*x264
            x270 = 7.1300179974499849178e-18*x133
            x271 = x142*x24
            x272 = x263*x28
            x273 = x133*x272
            x274 = x273*x31
            x275 = x271*x274
            x276 = 1.800509595315652757e-18*x238
            x277 = x257*x33
            x278 = x277*x92
            x279 = x136*x32
            x280 = x24*x279
            x281 = x272*x31
            x282 = x27*x33
            x283 = x270*x281*x282
            x284 = x222*x24
            x285 = x284*x5
            x286 = x284*x9
            x287 = x123*x227
            x288 = x167*x287
            x289 = x244*x287
            x290 = x265*x288
            x291 = x118*x235
            x292 = x114*x167
            x293 = x20*x39
            x294 = 2.6007360821226095379e-18*x293
            x295 = 2.6007360821226095379e-18*x167
            x296 = 1.029891488520553377e-17*x68
            x297 = x39*x98
            x298 = x215*x297
            x299 = x167*x297
            x300 = 2.8608096903348704917e-17*x299
            x301 = 1.1034551662720214754e-16*x299
            x302 = 1.029891488520553377e-17*x90
            x303 = 2.3406624739103485841e-17*x25
            x304 = 2.3406624739103485841e-17*x65
            x305 = 1.029891488520553377e-17*x65
            x306 = x123*x167
            x307 = x306*x65
            x308 = 2.8608096903348704917e-17*x307
            x309 = 1.1034551662720214754e-16*x128
            x310 = x307*x309
            x311 = x211*x306
            x312 = x143*x306
            x313 = x236*x73
            x314 = 2.8608096903348704917e-17*x313
            x315 = 1.1034551662720214754e-16*x313
            x316 = 2.3406624739103485841e-17*x226
            x317 = 1.029891488520553377e-17*x226
            x318 = x226*x306
            x319 = 2.8608096903348704917e-17*x318
            x320 = x309*x318
            x321 = 2.1065962265193137257e-16*x84
            x322 = x17*x268
            x323 = x132*x26
            x324 = x126*x18
            x325 = x268*x324
            x326 = x126*x322
            x327 = 9.2690233966849803931e-17*x326
            x328 = 9.9310964964481932784e-16*x323
            x329 = x25*x257*x7
            x330 = x126*x14*x17
            x331 = 9.9310964964481932784e-16*x77
            x332 = 2.5747287213013834425e-16*x236
            x333 = x120*x84
            x334 = 2.5747287213013834425e-16*x263
            x335 = 9.9310964964481932784e-16*x128*x84
            x336 = x273*x84
            x337 = 2.5747287213013834425e-16*x132
            x338 = 9.2690233966849803931e-17*x138
            x339 = 9.2690233966849803931e-17*x263
            x340 = x11*x23
            x341 = x26*x5
            x342 = x141*x341
            x343 = x223*x26
            x344 = x263*x328
            x345 = x134*x263
            x346 = 4.078370294541391373e-17*x345
            x347 = 1.1328806373726087147e-16*x5
            x348 = x222*x347
            x349 = 4.3696824584372050425e-16*x77
            x350 = x120*x31
            x351 = 1.1328806373726087147e-16*x11
            x352 = x223*x263
            x353 = x32*x352
            x354 = x263*x81
            x355 = 4.3696824584372050425e-16*x354
            x356 = x280*x355
            x357 = x140*x236
            x358 = x286*x355
            x359 = x137*x263
            x360 = 3.1468906593683575409e-16*x359
            x361 = 3.1468906593683575409e-16*x357
            x362 = 1.2138006828992236229e-15*x329
            x363 = 1.2138006828992236229e-15*x77
            x364 = 1.2138006828992236229e-15*x126
            x365 = x128*x357
            x366 = 1.2138006828992236229e-15*x130
            x367 = x285*x354
            x368 = 4.6818026340398625455e-15*x128
            x369 = x329*x368
            x370 = x368*x77
            return jnp.asarray([x3*x38, -x38*x40, x43*x45, -x45*x46, -x44*x49, x51*x54, -x56*x60, x62*x67, -x68*x71, x72*x75, -x76*x79, x79*x82, -x75*x86, x71*x90, -x78*x91*x92, x60*x61*x94, -x54*x95*x99, x104*x21*x49, x105*x37, -x107*x108, x106*x109, -x110*x115, x117*x119, -x121*x125, x127*x129, -x124*x131, x125*x135, -x127*x133*x139, x110*x145, -x107*x148, x117*x150, -x110*x152, -x104*x46*x47, x153*x154, -x155*x59*x94, x156*x157*x91, -x158*x159, 1.2574987649823606557e-20*x160*x86, -x161*x162, x162*x163, -1.2574987649823606557e-20*x156*x164, x159*x165, -x155*x67, x166*x59, -x153*x50, x105*x3*x44, x152*x168, -x150*x169, x148*x170, -x145*x168, x134*x139*x171, -x135*x172, x131*x171, -x126*x129*x171, x121*x172, -x119*x169, x115*x168, -x109*x167, x108*x170, -1.1306182702139106794e-22*x173*x37, x173*x174*x39*x44, -x103*x174*x176, x102*x156*x174, -2.5438911079812990288e-20*x177*x178*x179, -x180*x181*x50, x182*x183*x55, -2.3149409082629821162e-18*x184*x66, x165*x185, -x164*x187, x188*x189, -x189*x190, x187*x191, -x158*x185, x157*x186*x192, -x182*x193*x30, x116*x186*x194, x181*x195*x97, -x195*x196*x98, x198*x199*x200, -x201*x203, x204*x207, -x126*x209, x130*x209, -x207*x210, x207*x212, -x143*x198*x213, 2.5721565647366467957e-19*x156*x177*x197, -x203*x214, -x194*x217, x219*x65, -x205*x220*x221*x83, 1.0185739996357121311e-18*x217*x7*x87, -2.8293722212103114753e-18*x216*x86, x190*x216, -x188*x216, 2.8293722212103114753e-18*x101*x156*x220, -1.0185739996357121311e-18*x213*x220*x224, x216*x221*x225, -x219*x226, 1.7807237755869093201e-19*x218*x227, x214*x228, -2.5721565647366467957e-19*x17*x41*x48*x66, x143*x192*x230, -x212*x231, x210*x231, -x130*x232, x126*x232, -x204*x231, x201*x228, -2.3149409082629821162e-18*x200*x229*x73, x19*x196*x41*x51, -x181*x41*x56*x64, x15*x233*x234, -x233*x235*x96*x97, 1.2465066429108365241e-18*x237*x76, -1.2465066429108365241e-18*x227*x238, -x239*x240, 1.6204586357840874813e-17*x241*x242, -x243*x68, x164*x246, -x247*x249, x249*x250, -x191*x246, x243*x90, -x157*x248*x251, x239*x252, x154*x254, -x256*x258, 7.1300179974499849178e-18*x260*x262, -x260*x264, x265*x267, -x259*x265*x268, x132*x266*x269, -x138*x267*x270, x256*x275, -1.800509595315652757e-18*x236*x88*x96, -x276*x65, x251*x278, -x280*x283, 1.9805605548472180327e-17*x278*x31, -7.6393049972678409834e-17*x277*x82, 7.6393049972678409834e-17*x149*x257, -x269*x28*x282*x285*x31, x283*x286, -1.6204586357840874813e-17*x225*x27*x277, x226*x276, x147*x253*x58, -1.6204586357840874813e-17*x143*x288, 7.1300179974499849178e-18*x211*x288, -x135*x289, x130*x290, -x126*x290, x121*x289, -7.1300179974499849178e-18*x227*x291, 1.6204586357840874813e-17*x227*x292, -x254*x50, x240*x294, -x252*x294, x193*x295, -x146*x183*x295, -2.3406624739103485841e-17*x215*x242*x293, x296*x298, -x164*x300, x247*x301, -x250*x301, x191*x300, -x298*x302, x157*x299*x303, x292*x304, -x291*x305, x121*x308, -x126*x310, x130*x310, -x135*x308, x305*x311, -x304*x312, -x303*x313*x83*x84, x236*x302, -x314*x86, x161*x315, -x163*x315, x314*x72, -x236*x296, 2.3406624739103485841e-17*x167*x66, x312*x316, -x311*x317, x135*x319, -x130*x320, x126*x320, -x121*x319, x291*x317, -x292*x316, 2.1065962265193137257e-16*x112*x236*x80, -x28*x321*x322*x323, x271*x273*x321, -2.1065962265193137257e-16*x271*x325, -x286*x327, 2.5747287213013834425e-16*x285*x326, -x328*x329*x330, x258*x331, -x140*x323*x330*x332, x280*x327, 9.2690233966849803931e-17*x262*x333, -x333*x334, x272*x335, -x268*x335, x336*x337, -x336*x338, -x136*x141*x339, x144*x332, -x275*x331, 9.9310964964481932784e-16*x143*x329, -x334*x340*x342, x339*x340*x343, x325*x338, -x325*x337, x130*x18*x344, -x324*x344, 2.5747287213013834425e-16*x137*x24*x325*x341*x9, -9.2690233966849803931e-17*x14*x261*x323*x324, 4.078370294541391373e-17*x224*x261, -4.078370294541391373e-17*x237*x87, x279*x346, -x222*x346*x9, -x141*x261*x348, 4.3696824584372050425e-16*x118*x257, -x262*x349*x350, x121*x237*x351, 1.1328806373726087147e-16*x341*x353, -x126*x356, x130*x356, -x11*x347*x353, -1.1328806373726087147e-16*x211*x357, x138*x274*x349, -4.3696824584372050425e-16*x211*x329, x345*x348, x137*x351*x352, -x130*x358, x126*x358, -1.1328806373726087147e-16*x343*x359, x342*x360, -x121*x361, x135*x361, -x11*x141*x360*x5, -x121*x362, x263*x350*x363, x364*x365, -x365*x366, -x132*x274*x363, x135*x362, x366*x367, -x364*x367, x126*x369, -x281*x370, x268*x31*x370, -x130*x369])

        case 289:
          def shape_functions(xi):
            x0 = xi[0] - 1.0
            x1 = xi[1] - 1.0
            x2 = 2.0*xi[1]
            x3 = x2 + 1.0
            x4 = 4.0*xi[1]
            x5 = x4 + 1.0
            x6 = 8.0*xi[1]
            x7 = x6 + 1.0
            x8 = x4 + 3.0
            x9 = x6 + 3.0
            x10 = x6 + 5.0
            x11 = x6 + 7.0
            x12 = x2 - 1.0
            x13 = x4 - 1.0
            x14 = x6 - 1.0
            x15 = x4 - 3.0
            x16 = x6 - 3.0
            x17 = x6 - 5.0
            x18 = x6 - 7.0
            x19 = x10*x11*x12*x13*x14*x15*x16*x17*x18*x3*x5*x7*x8*x9*xi[1]
            x20 = x1*x19
            x21 = 2.0*xi[0]
            x22 = x21 + 1.0
            x23 = 4.0*xi[0]
            x24 = x23 + 1.0
            x25 = 8.0*xi[0]
            x26 = x25 + 1.0
            x27 = x23 + 3.0
            x28 = x25 + 3.0
            x29 = x25 + 5.0
            x30 = x25 + 7.0
            x31 = x21 - 1.0
            x32 = x23 - 1.0
            x33 = x25 - 1.0
            x34 = x23 - 3.0
            x35 = x25 - 3.0
            x36 = x25 - 5.0
            x37 = x25 - 7.0
            x38 = x22*x24*x26*x27*x28*x29*x30*x31*x32*x33*x34*x35*x36*x37*xi[0]
            x39 = 6.1319794541210246319e-19*x38
            x40 = x20*x39
            x41 = xi[0] + 1.0
            x42 = xi[1] + 1.0
            x43 = x19*x39*x42
            x44 = x20*x41
            x45 = x0*x22*x24*x26*x27*x28*x29*x31*x32*x33*x34*x35*x36*xi[0]
            x46 = 7.8489337012749115288e-17*x45
            x47 = x44*x46
            x48 = x30*x44
            x49 = x0*x22*x24*x26*x28*x29*x31*x32*x33*x35*x36*x37*xi[0]
            x50 = 2.9433501379780918233e-16*x49
            x51 = x48*x50
            x52 = x27*x48
            x53 = x0*x22*x24*x26*x28*x31*x32*x33*x34*x35*x37*xi[0]
            x54 = 2.7471267954462190351e-15*x53
            x55 = x52*x54
            x56 = x29*x52
            x57 = x0*x24*x26*x28*x32*x33*x34*x35*x36*x37*xi[0]
            x58 = 2.232040521300052966e-15*x57
            x59 = x56*x58
            x60 = x22*x56
            x61 = x0*x24*x26*x31*x32*x33*x34*x36*x37*xi[0]
            x62 = 2.1427589004480508474e-14*x61
            x63 = x60*x62
            x64 = x28*x60
            x65 = x0*x26*x31*x33*x34*x35*x36*x37*xi[0]
            x66 = 1.9641956587440466101e-14*x65
            x67 = x64*x66
            x68 = x0*x31*x32*x34*x35*x36*x37*xi[0]
            x69 = x33*x68
            x70 = 5.6119875964115617431e-14*x24
            x71 = x64*x70
            x72 = 173140.53095490047871*xi[0]**16 - 551885.44241874527589*xi[0]**14 + 694168.40804232804233*xi[0]**12 - 441984.42698916603679*xi[0]**10 + 152107.76187452758881*xi[0]**8 - 28012.603597883597884*xi[0]**6 + 2562.5271453766691862*xi[0]**4 - 97.755011337868480726*xi[0]**2 + 1.0
            x73 = x1*x72
            x74 = 7.8306956613834920713e-10*x19
            x75 = x26*x68
            x76 = x10*x12*x13*x14*x15*x16*x17*x18*x3*x42*x5*x7*x9*xi[1]
            x77 = x1*x38
            x78 = x41*x77
            x79 = x76*x78
            x80 = 7.8489337012749115288e-17*x8
            x81 = 2.9433501379780918233e-16*x11
            x82 = x11*x12*x13*x14*x15*x16*x17*x3*x42*x5*x7*x9*xi[1]
            x83 = x78*x82
            x84 = x18*x8
            x85 = 2.7471267954462190351e-15*x84
            x86 = x12*x13*x14*x16*x17*x42*x5*x7*x84*x9*xi[1]
            x87 = x10*x78
            x88 = x86*x87
            x89 = x11*x15
            x90 = 2.232040521300052966e-15*x89
            x91 = x12*x13*x14*x16*x42*x5*x7*x89*xi[1]
            x92 = x3*x87
            x93 = x91*x92
            x94 = x17*x84
            x95 = 2.1427589004480508474e-14*x94
            x96 = x13*x14*x16*x42*x7*x94*xi[1]
            x97 = x9*x92
            x98 = x96*x97
            x99 = x12*x89
            x100 = 1.9641956587440466101e-14*x99
            x101 = x13*x14*x42*x99*xi[1]
            x102 = x16*x94
            x103 = x101*x102
            x104 = x5*x97
            x105 = 5.6119875964115617431e-14*x104
            x106 = 173140.53095490047871*xi[1]**16 - 551885.44241874527589*xi[1]**14 + 694168.40804232804233*xi[1]**12 - 441984.42698916603679*xi[1]**10 + 152107.76187452758881*xi[1]**8 - 28012.603597883597884*xi[1]**6 + 2562.5271453766691862*xi[1]**4 - 97.755011337868480726*xi[1]**2 + 1.0
            x107 = x106*x41
            x108 = 7.8306956613834920713e-10*x38
            x109 = x13*x99
            x110 = x102*x42*xi[1]
            x111 = x110*x7
            x112 = x109*x111
            x113 = x111*x14
            x114 = x100*x113
            x115 = x101*x7
            x116 = x115*x95
            x117 = x5*x90
            x118 = x85*x9
            x119 = x3*x81
            x120 = x10*x80
            x121 = x30*x41
            x122 = x121*x19*x42
            x123 = x122*x27
            x124 = x123*x29
            x125 = x124*x22
            x126 = x125*x28
            x127 = x126*x66
            x128 = x126*x70
            x129 = x37*x41
            x130 = x0*x77
            x131 = x130*x82
            x132 = x10*x130
            x133 = x132*x86
            x134 = x132*x3
            x135 = x134*x91
            x136 = x134*x9
            x137 = x136*x96
            x138 = x136*x5
            x139 = 5.6119875964115617431e-14*x138
            x140 = x130*x76
            x141 = x1*x45
            x142 = x129*x141
            x143 = x76*x8
            x144 = 1.0046635137631886757e-14*x143
            x145 = x121*x141
            x146 = x145*x82
            x147 = x10*x8
            x148 = 1.0046635137631886757e-14*x147
            x149 = x142*x82
            x150 = x1*x121
            x151 = x143*x150
            x152 = 3.7674881766119575338e-14*x49
            x153 = x151*x152
            x154 = x151*x27
            x155 = 3.5163222981711603649e-13*x53
            x156 = x154*x155
            x157 = x154*x29
            x158 = 2.8570118672640677965e-13*x57
            x159 = x157*x158
            x160 = x157*x22
            x161 = 2.7427313925735050846e-12*x61
            x162 = x160*x161
            x163 = x160*x28
            x164 = 2.5141704431923796609e-12*x65
            x165 = x163*x164
            x166 = 7.1833441234067990312e-12*x24
            x167 = x163*x166
            x168 = x73*x76
            x169 = 1.0023290446570869851e-7*x8
            x170 = 3.7674881766119575338e-14*x11
            x171 = x145*x170
            x172 = 3.5163222981711603649e-13*x84
            x173 = x10*x145
            x174 = 2.8570118672640677965e-13*x89
            x175 = x173*x174
            x176 = x173*x3
            x177 = x176*x91
            x178 = 2.7427313925735050846e-12*x94
            x179 = x176*x9
            x180 = 2.5141704431923796609e-12*x99
            x181 = x179*x180
            x182 = x179*x5
            x183 = 7.1833441234067990312e-12*x182
            x184 = x107*x30
            x185 = 1.0023290446570869851e-7*x45
            x186 = x113*x5
            x187 = x115*x178
            x188 = x3*x9
            x189 = x188*x5
            x190 = x189*x96
            x191 = x10*x3
            x192 = x191*x86
            x193 = x150*x27
            x194 = x193*x82
            x195 = x147*x194
            x196 = x195*x29
            x197 = x196*x22
            x198 = x197*x28
            x199 = x164*x198
            x200 = x166*x198
            x201 = x10*x73
            x202 = x150*x34
            x203 = x202*x82
            x204 = x142*x170
            x205 = x10*x142
            x206 = x188*x205
            x207 = x174*x205
            x208 = x189*x205
            x209 = x180*x206
            x210 = 7.1833441234067990312e-12*x208
            x211 = x191*x91
            x212 = x202*x49
            x213 = x11*x76
            x214 = 1.4128080662294840752e-13*x213
            x215 = x193*x49
            x216 = x11*x192
            x217 = 1.4128080662294840752e-13*x216
            x218 = x36*x53
            x219 = x193*x213
            x220 = 1.3186208618141851368e-12*x219
            x221 = x219*x29
            x222 = 1.0713794502240254237e-12*x57
            x223 = x221*x222
            x224 = x22*x221
            x225 = 1.0285242722150644067e-11*x61
            x226 = x224*x225
            x227 = x224*x28
            x228 = 9.4281391619714237284e-12*x65
            x229 = x227*x228
            x230 = 2.6937540462775496367e-11*x24
            x231 = x227*x230
            x232 = 3.7587339174640761942e-7*x11
            x233 = x29*x53
            x234 = 1.3186208618141851368e-12*x84
            x235 = x234*x49
            x236 = x10*x215
            x237 = 1.0713794502240254237e-12*x89
            x238 = x236*x237
            x239 = 1.0285242722150644067e-11*x94
            x240 = x211*x239
            x241 = x188*x236
            x242 = 9.4281391619714237284e-12*x99
            x243 = x241*x242
            x244 = x189*x236
            x245 = 2.6937540462775496367e-11*x244
            x246 = x184*x27
            x247 = 3.7587339174640761942e-7*x49
            x248 = x115*x239
            x249 = x234*x91
            x250 = x193*x216
            x251 = 1.3186208618141851368e-12*x250
            x252 = x22*x29
            x253 = x250*x252
            x254 = x253*x28
            x255 = x228*x254
            x256 = x230*x254
            x257 = x201*x3
            x258 = x29*x31
            x259 = x10*x212
            x260 = x188*x259
            x261 = x237*x259
            x262 = x189*x259
            x263 = x242*x260
            x264 = 2.6937540462775496367e-11*x262
            x265 = 1.2307128043599061277e-11*x84
            x266 = x194*x265
            x267 = x193*x233
            x268 = x10*x267
            x269 = x188*x268
            x270 = x265*x91
            x271 = x193*x218
            x272 = x10*x188
            x273 = x271*x272
            x274 = x258*x57
            x275 = x194*x84
            x276 = 9.9995415354242372877e-12*x275
            x277 = x252*x275
            x278 = 9.5995598740072677962e-11*x61
            x279 = x277*x278
            x280 = x277*x28
            x281 = 8.7995965511733288132e-11*x65
            x282 = x280*x281
            x283 = 2.5141704431923796609e-10*x24
            x284 = x280*x283
            x285 = 3.508151656299804448e-6*x84
            x286 = x252*x57
            x287 = 9.9995415354242372877e-12*x89
            x288 = x268*x287
            x289 = 9.5995598740072677962e-11*x94
            x290 = x211*x289
            x291 = 8.7995965511733288132e-11*x99
            x292 = x269*x291
            x293 = x189*x268
            x294 = 2.5141704431923796609e-10*x293
            x295 = x246*x29
            x296 = 3.508151656299804448e-6*x53
            x297 = x115*x289
            x298 = x193*x286
            x299 = x272*x298
            x300 = x84*x91
            x301 = 9.9995415354242372877e-12*x300
            x302 = x193*x272
            x303 = x252*x28
            x304 = x302*x303
            x305 = x300*x304
            x306 = x281*x305
            x307 = x283*x305
            x308 = x257*x9
            x309 = x252*x35
            x310 = x302*x309
            x311 = x274*x302
            x312 = x10*x271
            x313 = x287*x312
            x314 = x189*x312
            x315 = x273*x291
            x316 = 2.5141704431923796609e-10*x314
            x317 = x193*x274
            x318 = 8.1246274975321927963e-12*x10*x89
            x319 = x318*x86
            x320 = x190*x318
            x321 = x193*x61
            x322 = x309*x321
            x323 = 7.7996423976309050844e-11*x10
            x324 = x86*x89
            x325 = x323*x324
            x326 = x32*x65
            x327 = x10*x303
            x328 = x193*x327
            x329 = x324*x328
            x330 = 7.1496721978283296607e-11*x329
            x331 = 2.0427634850938084745e-10*x24
            x332 = x329*x331
            x333 = 2.850373220743591114e-6*x89
            x334 = x24*x65
            x335 = x303*x321
            x336 = x298*x94
            x337 = 7.7996423976309050844e-11*x211
            x338 = 7.1496721978283296607e-11*x99
            x339 = x299*x338
            x340 = x10*x189
            x341 = 2.0427634850938084745e-10*x340
            x342 = x298*x341
            x343 = x22*x295
            x344 = 2.850373220743591114e-6*x57
            x345 = x115*x189
            x346 = x323*x345
            x347 = x190*x89
            x348 = x323*x347
            x349 = x328*x347
            x350 = 7.1496721978283296607e-11*x349
            x351 = x331*x349
            x352 = x308*x5
            x353 = x317*x94
            x354 = x311*x338
            x355 = x317*x341
            x356 = 7.4876567017256688811e-10*x94
            x357 = x211*x356
            x358 = x321*x327
            x359 = x322*x340
            x360 = 6.8636853099151964743e-10*x94
            x361 = x326*x360
            x362 = x193*x211*x303
            x363 = x24*x69
            x364 = 1.9610529456900561355e-9*x94
            x365 = x362*x364
            x366 = 0.000027363582919138474694*x94
            x367 = x24*x75
            x368 = x334*x360
            x369 = 6.8636853099151964743e-10*x61
            x370 = x304*x99
            x371 = x370*x96
            x372 = 1.9610529456900561355e-9*x103
            x373 = x189*x358
            x374 = x28*x343
            x375 = 0.000027363582919138474694*x61
            x376 = 1.9610529456900561355e-9*x112
            x377 = x186*x369
            x378 = x328*x345
            x379 = x364*x378
            x380 = x352*x7
            x381 = x310*x99
            x382 = 6.2917115340889301014e-10*x371
            x383 = x186*x370
            x384 = 6.2917115340889301014e-10*x383
            x385 = 1.7976318668825514576e-9*x371
            x386 = 0.000025083284342543601803*x99
            x387 = x189*x328
            x388 = 1.7976318668825514576e-9*x387
            x389 = x334*x388
            x390 = x24*x374
            x391 = 0.000025083284342543601803*x65
            x392 = 1.7976318668825514576e-9*x383
            x393 = x110*x380
            x394 = x326*x388
            x395 = 5.1360910482358613073e-9*x387
            x396 = x103*x395
            x397 = x112*x395
            x398 = 0.000071666526692981719437*x390
            return jnp.asarray([x0*x40, x40*x41, x41*x43, x0*x43, -x37*x47, x34*x51, -x36*x55, x31*x59, -x35*x63, x32*x67, -x69*x71, x73*x74, -x71*x75, x24*x67, -x28*x63, x22*x59, -x29*x55, x27*x51, -x30*x47, -x79*x80, x79*x81, -x83*x85, x88*x90, -x93*x95, x100*x98, -x103*x105, x107*x108, -x105*x112, x104*x114, -x104*x116, x117*x98, -x118*x93, x119*x88, -x120*x83, -x122*x46, x123*x50, -x124*x54, x125*x58, -x126*x62, x127*x24, -x128*x75, x42*x72*x74, -x128*x69, x127*x32, -x125*x35*x62, x124*x31*x58, -x123*x36*x54, x122*x34*x50, -x129*x19*x42*x46, -x120*x131, x119*x133, -x118*x135, x117*x137, -x116*x138, x114*x138, -x112*x139, x0*x106*x108, -x103*x139, x100*x137, -x135*x95, x133*x90, -x131*x85, x140*x81, -x140*x80, x142*x144, x144*x145, x146*x148, x148*x149, -x153*x34, x156*x36, -x159*x31, x162*x35, -x165*x32, x167*x69, -x168*x169, x167*x75, -x165*x24, x162*x28, -x159*x22, x156*x29, -x153*x27, -x171*x76, x146*x172, -x175*x86, x177*x178, -x181*x96, x103*x183, -x184*x185, x112*x183, -x181*x186, x182*x187, -x175*x190, x172*x177*x9, -x171*x192, -x152*x195, x155*x196, -x158*x197, x161*x198, -x199*x24, x200*x75, -x169*x201*x82, x200*x69, -x199*x32, x161*x197*x35, -x158*x196*x31, x155*x195*x36, -x147*x152*x203, -x192*x204, x172*x206*x91, -x190*x207, x187*x208, -x186*x209, x112*x210, -x107*x185*x37, x103*x210, -x209*x96, x142*x178*x211, -x207*x86, x149*x172, -x204*x76, x212*x214, x214*x215, x215*x217, x212*x217, -x218*x220, x223*x31, -x226*x35, x229*x32, -x231*x69, x168*x232, -x231*x75, x229*x24, -x226*x28, x22*x223, -x220*x233, -x194*x235, x238*x86, -x215*x240, x243*x96, -x103*x245, x246*x247, -x112*x245, x186*x243, -x244*x248, x190*x238, -x241*x249, -x233*x251, x222*x253, -x225*x254, x24*x255, -x256*x75, x232*x257*x86, -x256*x69, x255*x32, -x225*x253*x35, x222*x250*x258, -x218*x251, -x249*x260, x190*x261, -x248*x262, x186*x263, -x112*x264, x184*x247*x34, -x103*x264, x263*x96, -x212*x240, x261*x86, -x203*x235, x218*x266, x233*x266, x269*x270, x270*x273, -x274*x276, x279*x35, -x282*x32, x284*x69, -x285*x73*x82, x284*x75, -x24*x282, x279*x28, -x276*x286, -x288*x86, x267*x290, -x292*x96, x103*x294, -x295*x296, x112*x294, -x186*x292, x293*x297, -x190*x288, -x299*x301, x278*x305, -x24*x306, x307*x75, -x285*x308*x91, x307*x69, -x306*x32, x278*x300*x310, -x301*x311, -x190*x313, x297*x314, -x186*x315, x112*x316, -x246*x296*x36, x103*x316, -x315*x96, x271*x290, -x313*x86, x317*x319, x298*x319, x298*x320, x317*x320, -x322*x325, x326*x330, -x332*x69, x201*x333*x86, -x332*x75, x330*x334, -x325*x335, -x336*x337, x339*x96, -x103*x342, x343*x344, -x112*x342, x186*x339, -x336*x346, -x335*x348, x334*x350, -x351*x75, x333*x352*x96, -x351*x69, x326*x350, -x322*x348, -x346*x353, x186*x354, -x112*x355, x295*x31*x344, -x103*x355, x354*x96, -x337*x353, x322*x357, x335*x357, x345*x356*x358, x115*x356*x359, -x361*x362, x363*x365, -x257*x366*x91, x365*x367, -x362*x368, -x369*x371, x372*x373, -x374*x375, x373*x376, -x370*x377, -x368*x378, x367*x379, -x101*x366*x380, x363*x379, -x361*x378, -x377*x381, x359*x376, -x343*x35*x375, x359*x372, -x369*x381*x96, x326*x382, x334*x382, x334*x384, x326*x384, -x363*x385, x308*x386*x96, -x367*x385, -x103*x389, x390*x391, -x112*x389, -x367*x392, x14*x386*x393, -x363*x392, -x112*x394, x32*x374*x391, -x103*x394, x363*x396, x367*x396, x367*x397, x363*x397, -0.000071666526692981719437*x103*x352, -x398*x75, -0.000071666526692981719437*x109*x393, -x398*x69, x106*x72])

        case 324:
          def shape_functions(xi):
            x0 = 17.0*xi[1]
            x1 = x0 + 15.0
            x2 = xi[0] - 1.0
            x3 = x1*x2
            x4 = 17.0*xi[0]
            x5 = x4 + 1.0
            x6 = x0 + 1.0
            x7 = x4 + 3.0
            x8 = x0 + 3.0
            x9 = x4 + 5.0
            x10 = x0 + 5.0
            x11 = x4 + 7.0
            x12 = x0 + 7.0
            x13 = x4 + 9.0
            x14 = x0 + 9.0
            x15 = x4 + 11.0
            x16 = x0 + 11.0
            x17 = x4 + 13.0
            x18 = x0 + 13.0
            x19 = x4 + 15.0
            x20 = xi[1] - 1.0
            x21 = x4 - 1.0
            x22 = x0 - 1.0
            x23 = x4 - 3.0
            x24 = x0 - 3.0
            x25 = x4 - 5.0
            x26 = x0 - 5.0
            x27 = x4 - 7.0
            x28 = x0 - 7.0
            x29 = x4 - 9.0
            x30 = x0 - 9.0
            x31 = x4 - 11.0
            x32 = x0 - 11.0
            x33 = x4 - 13.0
            x34 = x0 - 13.0
            x35 = x4 - 15.0
            x36 = x0 - 15.0
            x37 = x10*x11*x12*x13*x14*x15*x16*x17*x18*x19*x20*x21*x22*x23*x24*x25*x26*x27*x28*x29*x30*x31*x32*x33*x34*x35*x36*x5*x6*x7*x8*x9
            x38 = 1.3296610891588984942e-37*x37
            x39 = xi[0] + 1.0
            x40 = x1*x39
            x41 = xi[1] + 1.0
            x42 = x19*x41
            x43 = x40*x42
            x44 = x10*x11*x12*x13*x14*x15*x16*x17*x18*x21*x22*x23*x24*x25*x26*x27*x28*x29*x30*x31*x32*x33*x34*x35*x36*x5*x6*x7*x8*x9
            x45 = 1.3296610891588984942e-37*x44
            x46 = x3*x42
            x47 = 3.8427205476692166482e-35*x39
            x48 = x20*x3
            x49 = x47*x48
            x50 = x10*x11*x12*x13*x14*x15*x18*x21*x22*x23*x24*x25*x26*x27*x28*x29*x30*x31*x32*x33*x34*x35*x36*x5*x6*x7*x8*x9
            x51 = x48*x50
            x52 = x16*x39
            x53 = 3.0741764381353733186e-34*x52
            x54 = x19*x53
            x55 = x10*x11*x12*x13*x14*x21*x22*x23*x24*x25*x26*x27*x28*x29*x30*x31*x32*x33*x34*x35*x36*x5*x6*x7*x8*x9
            x56 = x48*x55
            x57 = 1.5370882190676866593e-33*x17
            x58 = x18*x52
            x59 = x57*x58
            x60 = x19*x59
            x61 = x32*x48
            x62 = x19*x61
            x63 = x10*x11*x12*x14*x21*x22*x23*x24*x25*x26*x27*x28*x29*x30*x31*x33*x34*x35*x36*x5*x6*x7*x8*x9
            x64 = x15*x17
            x65 = x58*x64
            x66 = x63*x65
            x67 = 5.3798087667369033075e-33*x66
            x68 = x10*x12*x13*x14*x21*x22*x23*x24*x25*x26*x27*x28*x29*x30*x33*x34*x35*x36*x5*x6*x7*x8*x9
            x69 = x31*x65
            x70 = x62*x69
            x71 = 1.39875027935159486e-32*x70
            x72 = x10*x11*x12*x13*x14*x21*x22*x23*x24*x25*x26*x27*x28*x29*x30*x33*x34*x5*x6*x7*x8
            x73 = x35*x36
            x74 = x70*x73
            x75 = 2.7975005587031897199e-32*x74
            x76 = x10*x11*x12*x13*x14*x21*x22*x23*x24*x25*x26*x27*x28*x29*x30*x5*x6*x8*x9
            x77 = x33*x34
            x78 = x74*x77
            x79 = 4.3960723065335838456e-32*x78
            x80 = x10*x11*x12*x21*x22*x23*x24*x25*x26*x27*x28*x29*x30*x6*x7*x8*x9
            x81 = x13*x14
            x82 = x78*x81
            x83 = 5.495090383166979807e-32*x82
            x84 = x10*x11*x12*x22*x23*x24*x25*x26*x27*x28*x5*x6*x7*x8*x9
            x85 = x29*x30
            x86 = x84*x85
            x87 = x10*x12*x22*x24*x25*x26*x27*x28*x5*x6*x7*x8*x85*x9
            x88 = x21*x81
            x89 = x11*x88
            x90 = x87*x89
            x91 = x10*x22*x23*x24*x26*x27*x28*x6*x7*x8*x85*x89*x9
            x92 = x5*x77
            x93 = x12*x92
            x94 = x91*x93
            x95 = x22*x23*x24*x25*x26*x28*x6*x8*x85*x89*x9*x93
            x96 = x7*x73
            x97 = x10*x96
            x98 = x95*x97
            x99 = 5.3798087667369033075e-33*x84
            x100 = x30*x99
            x101 = x11*x68
            x102 = x101*x15
            x103 = x61*x96
            x104 = x34*x76
            x105 = x31*x64
            x106 = x105*x18
            x107 = x104*x106
            x108 = x106*x16
            x109 = x32*x72
            x110 = x109*x9
            x111 = x110*x36
            x112 = x108*x111
            x113 = x41*x47
            x114 = x20*x43
            x115 = x114*x16
            x116 = 3.0741764381353733186e-34*x55*x64
            x117 = x50*x57
            x118 = x108*x114
            x119 = x32*x73
            x120 = x119*x80
            x121 = x13*x92
            x122 = x120*x121
            x123 = 5.3798087667369033075e-33*x122
            x124 = x118*x25
            x125 = 1.39875027935159486e-32*x124
            x126 = x119*x91
            x127 = x126*x92
            x128 = x118*x96
            x129 = x27*x32
            x130 = 2.7975005587031897199e-32*x129
            x131 = x130*x95
            x132 = x22*x23*x24*x26*x28*x6*x85*x9*x93
            x133 = x89*x97
            x134 = x129*x133
            x135 = x124*x134
            x136 = 4.3960723065335838456e-32*x135
            x137 = x22*x23*x24*x26*x28*x85*x9
            x138 = x8*x93
            x139 = 5.495090383166979807e-32*x138
            x140 = x135*x139
            x141 = x23*x24*x26*x28*x9
            x142 = x6*x85
            x143 = x141*x142
            x144 = x142*x23*x28*x9
            x145 = x138*x22
            x146 = x145*x26
            x147 = x144*x146
            x148 = x133*x24
            x149 = x144*x145
            x150 = x148*x149
            x151 = x130*x150
            x152 = x134*x24
            x153 = x142*x146*x23*x9
            x154 = x152*x153
            x155 = x119*x77
            x156 = x29*x88
            x157 = x155*x156
            x158 = x157*x99
            x159 = x13*x63
            x160 = x15*x159
            x161 = x160*x18*x57
            x162 = x32*x76
            x163 = x162*x33
            x164 = 3.0741764381353733186e-34*x163
            x165 = x110*x35
            x166 = 3.8427205476692166482e-35*x165
            x167 = x32*x46
            x168 = x167*x96
            x169 = x107*x168
            x170 = x46*x69
            x171 = x155*x170
            x172 = x171*x88
            x173 = x69*x98
            x174 = 1.39875027935159486e-32*x167
            x175 = 2.7975005587031897199e-32*x170
            x176 = x119*x94
            x177 = 4.3960723065335838456e-32*x171
            x178 = 5.495090383166979807e-32*x81
            x179 = x171*x86
            x180 = x120*x77
            x181 = x170*x180
            x182 = x109*x73
            x183 = x68*x69
            x184 = x46*x55
            x185 = x46*x50
            x186 = x20*x46
            x187 = x108*x186
            x188 = x187*x96
            x189 = x16*x186
            x190 = x187*x25
            x191 = 1.39875027935159486e-32*x190
            x192 = x134*x190
            x193 = 4.3960723065335838456e-32*x192
            x194 = x139*x192
            x195 = x2*x41
            x196 = 1.1105462382764036113e-32*x20
            x197 = x2*x42
            x198 = x197*x69
            x199 = x72*x9
            x200 = x35*x69
            x201 = x41*x61
            x202 = x197*x20
            x203 = 8.8843699062112288907e-32*x52
            x204 = 4.4421849531056144454e-31*x202
            x205 = x17*x58
            x206 = x202*x32
            x207 = 4.0423883073261091453e-30*x206
            x208 = x198*x20
            x209 = 8.0847766146522182906e-30*x208
            x210 = x155*x208
            x211 = 1.2704648965882057314e-29*x210
            x212 = 1.5880811207352571642e-29*x81
            x213 = x212*x86
            x214 = 1.5547647335869650559e-30*x84
            x215 = x30*x88
            x216 = x214*x215
            x217 = x101*x65
            x218 = 8.8843699062112288907e-32*x104
            x219 = x111*x186
            x220 = 4.4421849531056144454e-31*x39
            x221 = x20*x36
            x222 = x221*x69
            x223 = x167*x80
            x224 = 1.5547647335869650559e-30*x223
            x225 = x167*x222
            x226 = 4.0423883073261091453e-30*x25
            x227 = x226*x91*x92
            x228 = x7*x95
            x229 = x170*x221
            x230 = x129*x229
            x231 = 8.0847766146522182906e-30*x230
            x232 = x132*x89
            x233 = 1.2704648965882057314e-29*x25
            x234 = x10*x7
            x235 = x230*x234
            x236 = x233*x235
            x237 = x137*x138
            x238 = 1.5880811207352571642e-29*x237
            x239 = x25*x89
            x240 = x235*x239
            x241 = x138*x143
            x242 = 1.5880811207352571642e-29*x241
            x243 = x149*x24
            x244 = x239*x243
            x245 = x153*x226
            x246 = x156*x214*x77
            x247 = x20*x200
            x248 = x167*x247
            x249 = x165*x186
            x250 = 4.4421849531056144454e-31*x249
            x251 = x248*x77
            x252 = x170*x20
            x253 = x252*x35
            x254 = x23*x87
            x255 = x254*x88
            x256 = x14*x92
            x257 = x205*x31
            x258 = x15*x31*x58
            x259 = x33*x69
            x260 = x103*x41
            x261 = x201*x69
            x262 = x261*x73
            x263 = x261*x27
            x264 = x25*x263
            x265 = x133*x147
            x266 = x233*x263
            x267 = x133*x264
            x268 = x132*x133
            x269 = x69*x95
            x270 = x20*x52
            x271 = 7.1074959249689831126e-31*x270
            x272 = 7.1074959249689831126e-31*x96
            x273 = x163*x186
            x274 = 3.5537479624844915563e-30*x270
            x275 = x17*x184
            x276 = x167*x270
            x277 = x63*x64
            x278 = 3.2339106458608873162e-29*x105*x276
            x279 = 6.4678212917217746324e-29*x186
            x280 = x105*x52
            x281 = x279*x280
            x282 = 1.0163719172705645851e-28*x76
            x283 = x186*x280
            x284 = x155*x283
            x285 = 1.2704648965882057314e-28*x81
            x286 = x285*x86
            x287 = 1.0163719172705645851e-28*x90
            x288 = 1.2438117868695720447e-29*x84
            x289 = x215*x288
            x290 = x101*x167*x64
            x291 = x20*x39
            x292 = 3.5537479624844915563e-30*x291
            x293 = x252*x34
            x294 = x293*x5
            x295 = 1.2438117868695720447e-29*x120
            x296 = x25*x294
            x297 = 3.2339106458608873162e-29*x296
            x298 = x12*x8
            x299 = x137*x298
            x300 = x129*x96
            x301 = x239*x6
            x302 = x300*x301
            x303 = x299*x302
            x304 = x134*x296
            x305 = 1.0163719172705645851e-28*x304
            x306 = x137*x6
            x307 = x12*x306
            x308 = 1.2704648965882057314e-28*x304
            x309 = x143*x298
            x310 = x22*x298
            x311 = x144*x310
            x312 = x26*x311
            x313 = 6.4678212917217746324e-29*x152
            x314 = x152*x9
            x315 = x142*x23
            x316 = x26*x310*x315
            x317 = x314*x316
            x318 = 3.5537479624844915563e-30*x96
            x319 = x273*x318
            x320 = x252*x33
            x321 = x119*x320
            x322 = x25*x5
            x323 = x167*x20
            x324 = x133*x323
            x325 = x324*x6
            x326 = x320*x5
            x327 = x186*x258
            x328 = x25*x327
            x329 = 3.2339106458608873162e-29*x328
            x330 = x258*x279
            x331 = x152*x25
            x332 = x149*x331
            x333 = x134*x328
            x334 = 1.0163719172705645851e-28*x333
            x335 = 1.2704648965882057314e-28*x333
            x336 = x300*x95
            x337 = x18*x291
            x338 = 1.7768739812422457781e-29*x337
            x339 = 1.7768739812422457781e-29*x186
            x340 = 1.6169553229304436581e-28*x68
            x341 = x106*x167*x291
            x342 = 3.2339106458608873162e-28*x186
            x343 = x106*x39
            x344 = x342*x343
            x345 = 5.0818595863528229255e-28*x76
            x346 = x186*x343
            x347 = x155*x346
            x348 = 6.3523244829410286569e-28*x81
            x349 = x348*x86
            x350 = 5.0818595863528229255e-28*x90
            x351 = 1.6169553229304436581e-28*x98
            x352 = 6.2190589343478602235e-29*x84
            x353 = x215*x352
            x354 = x186*x65
            x355 = 6.2190589343478602235e-29*x122
            x356 = x25*x354
            x357 = 1.6169553229304436581e-28*x356
            x358 = x342*x65
            x359 = x134*x356
            x360 = 5.0818595863528229255e-28*x359
            x361 = 6.3523244829410286569e-28*x359
            x362 = x157*x352
            x363 = x252*x73
            x364 = x363*x77
            x365 = 3.2339106458608873162e-28*x363
            x366 = x186*x257
            x367 = x25*x366
            x368 = 1.6169553229304436581e-28*x367
            x369 = x257*x342
            x370 = x134*x367
            x371 = 5.0818595863528229255e-28*x370
            x372 = 6.3523244829410286569e-28*x370
            x373 = x252*x92
            x374 = 2.1766706270217510782e-28*x20*x84
            x375 = x13*x21
            x376 = x171*x375
            x377 = x171*x29
            x378 = x14*x21
            x379 = x20*x376
            x380 = x129*x97
            x381 = x24*x380
            x382 = x28*x315
            x383 = x146*x25
            x384 = x382*x383
            x385 = 1.1318687260513105607e-27*x252
            x386 = x11*x375
            x387 = x385*x386
            x388 = x25*x252
            x389 = 1.7786508552234880239e-27*x388
            x390 = x132*x8
            x391 = x386*x390
            x392 = x119*x27
            x393 = x10*x392
            x394 = 2.2233135690293600299e-27*x20
            x395 = x13*x394
            x396 = 5.6593436302565528034e-28*x25
            x397 = x396*x69
            x398 = x373*x8
            x399 = x22*x398
            x400 = x141*x30
            x401 = x134*x400
            x402 = x401*x6
            x403 = x145*x385
            x404 = x22*x93
            x405 = 2.2233135690293600299e-27*x388
            x406 = x134*x6
            x407 = x23*x28
            x408 = x30*x407
            x409 = 1.7786508552234880239e-27*x252
            x410 = x383*x409
            x411 = x314*x6
            x412 = x252*x383
            x413 = x141*x29
            x414 = x145*x413
            x415 = x406*x413
            x416 = x301*x393
            x417 = x146*x331
            x418 = x407*x417
            x419 = 5.6593436302565528034e-28*x388
            x420 = x380*x419
            x421 = x11*x378
            x422 = x153*x381
            x423 = x380*x421
            x424 = x25*x385
            x425 = x389*x423
            x426 = x405*x423
            x427 = x300*x390
            x428 = x306*x8
            x429 = 1.4714293438667037289e-27*x25
            x430 = x380*x88
            x431 = x306*x398
            x432 = x429*x69
            x433 = x388*x422
            x434 = x26*x399
            x435 = x331*x382
            x436 = 5.7806152794763360777e-27*x388
            x437 = x436*x77
            x438 = 5.7806152794763360777e-27*x25
            x439 = x11*x81
            x440 = x380*x439
            x441 = 4.6244922235810688622e-27*x25
            x442 = x142*x28
            x443 = x314*x442
            x444 = 2.9428586877334074578e-27*x252
            x445 = x323*x69
            x446 = x441*x445
            x447 = x324*x438*x69
            x448 = x314*x412
            x449 = x252*x393
            x450 = x239*x24
            x451 = x388*x430
            x452 = 4.6244922235810688622e-27*x451
            x453 = x430*x436
            x454 = x388*x427
            x455 = 5.8857173754668149155e-27*x252
            x456 = x300*x455
            x457 = x146*x450
            x458 = 9.2489844471621377244e-27*x252
            x459 = 1.1561230558952672155e-26*x252
            x460 = x459*x77
            x461 = x442*x9
            x462 = x134*x458
            x463 = x134*x459
            x464 = x388*x443
            x465 = x388*x440
            x466 = x26*x435
            x467 = x26*x464
            x468 = x239*x449
            x469 = x134*x388*x77
            x470 = 1.8167648021211341959e-26*x469
            x471 = 1.8167648021211341959e-26*x465
            x472 = 1.8167648021211341959e-26*x468
            x473 = 2.2709560026514177448e-26*x469
            x474 = 2.2709560026514177448e-26*x465
            return jnp.asarray([x3*x38, -x38*x40, x43*x45, -x45*x46, -x44*x49, x51*x54, -x56*x60, x62*x67, -x68*x71, x72*x75, -x76*x79, x80*x83, -x83*x86, x79*x90, -x75*x94, x71*x98, -x100*x21*x82, x102*x60*x61, -x103*x107*x54, x112*x19*x49, x113*x37, -x115*x116, x114*x117, -x118*x123, x125*x127, -x128*x131, x132*x136, -x137*x140, x140*x143, -x136*x147, x124*x151, -x125*x154, x118*x158, -x115*x161, x128*x164, -x118*x166, -x112*x46*x47, x169*x53, -x102*x167*x59, x100*x172, -x173*x174, x175*x176, -x177*x90, x178*x179, -x178*x181, x177*x76, -x175*x182, x174*x183, -x167*x67, x184*x59, -x185*x53, x113*x3*x44, x166*x187, -x164*x188, x161*x189, -x158*x187, x154*x191, -x151*x190, x147*x193, -x143*x194, x137*x194, -x132*x193, x131*x188, -x127*x191, x123*x187, -x117*x186, x116*x189, -3.8427205476692166482e-35*x195*x37, x195*x196*x39*x44, -x111*x196*x198, x110*x170*x196, -1.1105462382764036113e-32*x199*x200*x201, -x202*x203*x50, x204*x205*x55, -1.5547647335869650559e-30*x206*x66, x183*x207, -x182*x209, x211*x76, -x180*x208*x212, x210*x213, -x211*x90, x176*x209, -x173*x207, x210*x216, -x204*x217*x32, x208*x218*x32*x96, x105*x203*x219, -x106*x219*x220, x121*x222*x224, -x225*x227, x228*x231, -x232*x236, x238*x240, -x240*x242, x147*x236*x89, -x231*x234*x244, x235*x24*x245*x89, -x225*x246, 4.4421849531056144454e-31*x199*x229, -8.8843699062112288907e-32*x163*x229*x7, -x218*x248*x7, x250*x65, -x216*x251, 4.0423883073261091453e-30*x10*x228*x248, -8.0847766146522182906e-30*x248*x94, 1.2704648965882057314e-29*x251*x90, -x213*x251, x212*x223*x247*x77, -1.2704648965882057314e-29*x162*x253*x77, 8.0847766146522182906e-30*x109*x253, -4.0423883073261091453e-30*x251*x255, x224*x247*x256, -x250*x257, 8.8843699062112288907e-32*x249*x258, 8.8843699062112288907e-32*x259*x260*x76, -4.4421849531056144454e-31*x13*x41*x48*x66, x246*x262, -x148*x245*x263, 8.0847766146522182906e-30*x150*x264, -x265*x266, x242*x267, -x238*x267, x266*x268, -8.0847766146522182906e-30*x260*x269*x27, x227*x262, -1.5547647335869650559e-30*x121*x262*x80, x17*x220*x41*x51, -x203*x41*x56*x64, x15*x184*x271, -x104*x105*x168*x271, x162*x252*x272, -x258*x272*x273, -x274*x275, 1.2438117868695720447e-29*x276*x277, -x278*x68, x182*x281, -x282*x284, x180*x283*x285, -x284*x286, x284*x287, -x176*x281, x278*x98, -x284*x289, x274*x290, x169*x292, -x13*x294*x295, x126*x297, -6.4678212917217746324e-29*x294*x303, x305*x307, -x299*x308, x308*x309, -x305*x312, x296*x311*x313, -x297*x317, x119*x156*x288*x293, -x104*x252*x318, -x319*x65, x289*x321, -3.2339106458608873162e-29*x259*x299*x322*x325, 6.4678212917217746324e-29*x12*x126*x326, -x287*x321, x286*x321, -x120*x285*x320, x282*x321, -x28*x313*x316*x320*x322, 3.2339106458608873162e-29*x255*x321, -x14*x295*x326, x257*x319, 3.5537479624844915563e-30*x160*x186*x58, -x157*x288*x327, x154*x329, -x330*x332, x147*x334, -x241*x335, x237*x335, -x132*x334, x330*x336, -x127*x329, 1.2438117868695720447e-29*x122*x327, -x185*x292, x275*x338, -x290*x338, x217*x339, -x159*x205*x339, -6.2190589343478602235e-29*x167*x277*x337, x340*x341, -x182*x344, x345*x347, -x180*x346*x348, x347*x349, -x347*x350, x176*x344, -x341*x351, x347*x353, x354*x355, -x127*x357, x336*x358, -x132*x360, x237*x361, -x241*x361, x147*x360, -x332*x358, x154*x357, -x354*x362, -x353*x364, x252*x351, -x365*x94, x350*x364, -x349*x364, x348*x364*x80, -x345*x364, x365*x72, -x252*x340, 6.2190589343478602235e-29*x186*x66, x362*x366, -x154*x368, x332*x369, -x147*x371, x241*x372, -x237*x372, x132*x371, -x336*x369, x127*x368, -x355*x366, 2.1766706270217510782e-28*x120*x373, -x30*x374*x376, x172*x374, -x374*x377*x378, -5.6593436302565528034e-28*x254*x379, x381*x384*x387, -x389*x391*x393, x181*x395, -x179*x395, 1.7786508552234880239e-27*x11*x379*x87, -x380*x387*x390, x323*x391*x397*x97, x396*x399*x402, -x302*x400*x403, x389*x402*x404, -x145*x401*x405, x138*x402*x405, -x406*x408*x410*x9, x25*x403*x408*x411, -5.6593436302565528034e-28*x23*x30*x411*x412, -x325*x397*x414, x403*x415, -x28*x29*x410*x411, x377*x394*x81*x84, -x310*x405*x415*x77, x409*x414*x416, -x29*x385*x418*x6, x141*x145*x156*x420*x6, x419*x421*x422, -x243*x423*x424, x147*x425, -x241*x426, x237*x426, -x132*x425, x421*x424*x427, -x11*x21*x256*x420*x428, x429*x430*x431, -x324*x428*x432*x92, x148*x153*x323*x432, -1.4714293438667037289e-27*x433*x88, -2.9428586877334074578e-27*x434*x435, 4.6244922235810688622e-27*x137*x398*x416, -x126*x437, x431*x438*x440, -x434*x441*x443, x127*x444, 2.9428586877334074578e-27*x168*x20*x269, -x268*x446, x237*x447, -x241*x447, x265*x446, -2.9428586877334074578e-27*x150*x25*x445, -x154*x444, 4.6244922235810688622e-27*x142*x448, -5.7806152794763360777e-27*x433*x439, x317*x437, -4.6244922235810688622e-27*x153*x449*x450, x315*x417*x444, 2.9428586877334074578e-27*x243*x451, -x147*x452, x241*x453, -x237*x453, x132*x452, -2.9428586877334074578e-27*x454*x88, x382*x456*x457, -x232*x456*x8, x149*x152*x455, -x145*x435*x455, -x392*x458*x95, x303*x460, -1.1561230558952672155e-26*x439*x454, x300*x457*x458*x461, x132*x462, -x237*x463, x241*x463, -x147*x462, -9.2489844471621377244e-27*x145*x464, 1.1561230558952672155e-26*x243*x465, -x311*x331*x460, 9.2489844471621377244e-27*x244*x449, x384*x462, -x138*x459*x466, x418*x459*x85, -x404*x458*x466, 1.4534118416969073567e-26*x232*x388*x393, -1.4534118416969073567e-26*x404*x467, 1.4534118416969073567e-26*x134*x412*x461, -1.4534118416969073567e-26*x147*x468, -x307*x470, x132*x471, 1.8167648021211341959e-26*x28*x448*x85, -1.8167648021211341959e-26*x138*x467, -x147*x471, x312*x470, x241*x472, -x237*x472, x299*x473, -x237*x474, x241*x474, -x309*x473])

        case 361:
          def shape_functions(xi):
            x0 = xi[0] - 1.0
            x1 = xi[1] - 1.0
            x2 = 3.0*xi[1]
            x3 = x2 + 1.0
            x4 = 9.0*xi[1]
            x5 = x4 + 1.0
            x6 = x2 + 2.0
            x7 = x4 + 2.0
            x8 = x4 + 4.0
            x9 = x4 + 8.0
            x10 = x4 + 5.0
            x11 = x4 + 7.0
            x12 = x2 - 1.0
            x13 = x4 - 1.0
            x14 = x2 - 2.0
            x15 = x4 - 2.0
            x16 = x4 - 4.0
            x17 = x4 - 8.0
            x18 = x4 - 5.0
            x19 = x4 - 7.0
            x20 = x10*x11*x12*x13*x14*x15*x16*x17*x18*x19*x3*x5*x6*x7*x8*x9*xi[1]
            x21 = x1*x20
            x22 = 3.0*xi[0]
            x23 = x22 + 1.0
            x24 = 9.0*xi[0]
            x25 = x24 + 1.0
            x26 = x22 + 2.0
            x27 = x24 + 2.0
            x28 = x24 + 4.0
            x29 = x24 + 8.0
            x30 = x24 + 5.0
            x31 = x24 + 7.0
            x32 = x22 - 1.0
            x33 = x24 - 1.0
            x34 = x22 - 2.0
            x35 = x24 - 2.0
            x36 = x24 - 4.0
            x37 = x24 - 8.0
            x38 = x24 - 5.0
            x39 = x24 - 7.0
            x40 = x23*x25*x26*x27*x28*x29*x30*x31*x32*x33*x34*x35*x36*x37*x38*x39*xi[0]
            x41 = 1.0501661969785547778e-24*x40
            x42 = x21*x41
            x43 = xi[0] + 1.0
            x44 = xi[1] + 1.0
            x45 = x20*x41*x44
            x46 = x21*x43
            x47 = x0*x23*x25*x26*x27*x28*x30*x31*x32*x33*x34*x35*x36*x38*x39*xi[0]
            x48 = 1.70126923910525874e-22*x47
            x49 = x46*x48
            x50 = x29*x46
            x51 = x0*x23*x25*x26*x27*x28*x30*x32*x33*x34*x35*x36*x37*x38*xi[0]
            x52 = 1.446078853239469929e-21*x51
            x53 = x50*x52
            x54 = x31*x50
            x55 = x0*x23*x25*x27*x28*x30*x32*x33*x35*x36*x37*x38*x39*xi[0]
            x56 = 2.570806850203502096e-21*x55
            x57 = x54*x56
            x58 = x26*x54
            x59 = x0*x23*x25*x27*x28*x32*x33*x34*x35*x36*x37*x39*xi[0]
            x60 = 2.892157706478939858e-20*x59
            x61 = x58*x60
            x62 = x30*x58
            x63 = x0*x23*x25*x27*x32*x33*x34*x35*x37*x38*x39*xi[0]
            x64 = 8.0980415781410316025e-20*x63
            x65 = x62*x64
            x66 = x28*x62
            x67 = x0*x25*x27*x33*x34*x35*x36*x37*x38*x39*xi[0]
            x68 = 5.8485855842129672685e-20*x67
            x69 = x66*x68
            x70 = x23*x66
            x71 = x0*x25*x32*x33*x34*x36*x37*x38*x39*xi[0]
            x72 = 3.0078440147380974524e-19*x71
            x73 = x70*x72
            x74 = x0*x32*x34*x35*x36*x37*x38*x39*xi[0]
            x75 = x33*x74
            x76 = 4.135785520264883997e-19*x27
            x77 = x70*x76
            x78 = -1139827.4301937679369*xi[0]**18 + 4010503.9210521464445*xi[0]**16 - 5723632.7564645448023*xi[0]**14 + 4288221.5882976921237*xi[0]**12 - 1825541.0625608358578*xi[0]**10 + 447065.31380067163584*xi[0]**8 - 60894.246929607780612*xi[0]**6 + 4228.3941844706632653*xi[0]**4 - 124.72118622448979592*xi[0]**2 + 1.0
            x79 = x1*x78
            x80 = 1.0247761692089423182e-12*x20
            x81 = x25*x74
            x82 = x10*x12*x13*x14*x15*x16*x17*x18*x19*x3*x44*x5*x6*x7*x8*xi[1]
            x83 = x1*x40
            x84 = x43*x83
            x85 = x82*x84
            x86 = 1.70126923910525874e-22*x11
            x87 = 1.446078853239469929e-21*x9
            x88 = x10*x12*x13*x14*x15*x16*x18*x19*x3*x44*x5*x7*x8*x9*xi[1]
            x89 = x84*x88
            x90 = x11*x17
            x91 = 2.570806850203502096e-21*x90
            x92 = x12*x13*x14*x15*x16*x18*x3*x44*x5*x7*x8*x90*xi[1]
            x93 = x6*x84
            x94 = x92*x93
            x95 = x19*x9
            x96 = 2.892157706478939858e-20*x95
            x97 = x12*x13*x15*x16*x18*x3*x44*x5*x7*x95*xi[1]
            x98 = x10*x93
            x99 = x97*x98
            x100 = x14*x90
            x101 = 8.0980415781410316025e-20*x100
            x102 = x100*x12*x13*x15*x16*x44*x5*x7*xi[1]
            x103 = x8*x98
            x104 = x102*x103
            x105 = x18*x95
            x106 = 5.8485855842129672685e-20*x105
            x107 = x105*x12*x13*x15*x44*x5*xi[1]
            x108 = x103*x3
            x109 = x107*x108
            x110 = x100*x16
            x111 = 3.0078440147380974524e-19*x110
            x112 = x110*x13*x15*x44*xi[1]
            x113 = x105*x12
            x114 = x112*x113
            x115 = x108*x7
            x116 = 4.135785520264883997e-19*x115
            x117 = -1139827.4301937679369*xi[1]**18 + 4010503.9210521464445*xi[1]**16 - 5723632.7564645448023*xi[1]**14 + 4288221.5882976921237*xi[1]**12 - 1825541.0625608358578*xi[1]**10 + 447065.31380067163584*xi[1]**8 - 60894.246929607780612*xi[1]**6 + 4228.3941844706632653*xi[1]**4 - 124.72118622448979592*xi[1]**2 + 1.0
            x118 = x117*x43
            x119 = 1.0247761692089423182e-12*x40
            x120 = x110*x15
            x121 = x113*x44*xi[1]
            x122 = x121*x5
            x123 = x120*x122
            x124 = x122*x13
            x125 = x111*x124
            x126 = x112*x5
            x127 = x106*x126
            x128 = x101*x7
            x129 = x3*x96
            x130 = x8*x91
            x131 = x10*x87
            x132 = x6*x86
            x133 = x29*x43
            x134 = x133*x20*x44
            x135 = x134*x31
            x136 = x135*x26
            x137 = x136*x30
            x138 = x137*x28
            x139 = x138*x23
            x140 = x139*x72
            x141 = x139*x76
            x142 = x37*x43
            x143 = x0*x83
            x144 = x143*x88
            x145 = x143*x6
            x146 = x145*x92
            x147 = x10*x145
            x148 = x147*x97
            x149 = x147*x8
            x150 = x102*x149
            x151 = x149*x3
            x152 = x107*x151
            x153 = x151*x7
            x154 = 4.135785520264883997e-19*x153
            x155 = x143*x82
            x156 = x1*x47
            x157 = x142*x156
            x158 = x11*x82
            x159 = 2.7560561673505191588e-20*x158
            x160 = x133*x156
            x161 = x160*x88
            x162 = x11*x6
            x163 = 2.7560561673505191588e-20*x162
            x164 = x157*x88
            x165 = x1*x133
            x166 = x158*x165
            x167 = 2.342647742247941285e-19*x51
            x168 = x166*x167
            x169 = x166*x31
            x170 = 4.1647070973296733956e-19*x55
            x171 = x169*x170
            x172 = x169*x26
            x173 = 4.68529548449588257e-18*x59
            x174 = x172*x173
            x175 = x172*x30
            x176 = 1.3118827356588471196e-17*x63
            x177 = x175*x176
            x178 = x175*x28
            x179 = 9.4747086464250069749e-18*x67
            x180 = x178*x179
            x181 = x178*x23
            x182 = 4.8727073038757178728e-17*x71
            x183 = x181*x182
            x184 = 6.6999725428291120751e-17*x27
            x185 = x181*x184
            x186 = x79*x82
            x187 = 1.6601373941184865555e-10*x11
            x188 = 2.342647742247941285e-19*x9
            x189 = x160*x188
            x190 = 4.1647070973296733956e-19*x90
            x191 = x160*x6
            x192 = 4.68529548449588257e-18*x95
            x193 = x191*x192
            x194 = x10*x191
            x195 = x194*x97
            x196 = 1.3118827356588471196e-17*x100
            x197 = x194*x8
            x198 = 9.4747086464250069749e-18*x105
            x199 = x197*x198
            x200 = x197*x3
            x201 = x107*x200
            x202 = 4.8727073038757178728e-17*x110
            x203 = x200*x7
            x204 = 6.6999725428291120751e-17*x203
            x205 = x118*x29
            x206 = 1.6601373941184865555e-10*x47
            x207 = x124*x202
            x208 = x3*x7
            x209 = x126*x208
            x210 = x10*x8
            x211 = x210*x3
            x212 = x102*x211
            x213 = x10*x6
            x214 = x213*x92
            x215 = x165*x31
            x216 = x215*x88
            x217 = x162*x216
            x218 = x217*x26
            x219 = x218*x30
            x220 = x219*x28
            x221 = x220*x23
            x222 = x182*x221
            x223 = x184*x221
            x224 = x6*x79
            x225 = x165*x39
            x226 = x225*x88
            x227 = x157*x188
            x228 = x157*x6
            x229 = x210*x228
            x230 = x192*x228
            x231 = x208*x229
            x232 = x198*x229
            x233 = 6.6999725428291120751e-17*x231
            x234 = x107*x211
            x235 = x213*x97
            x236 = x225*x51
            x237 = x82*x9
            x238 = 1.9912505809107500923e-18*x237
            x239 = x215*x51
            x240 = x214*x9
            x241 = 1.9912505809107500923e-18*x240
            x242 = x34*x55
            x243 = x215*x237
            x244 = 3.5400010327302223862e-18*x243
            x245 = x243*x26
            x246 = 3.9825011618215001845e-17*x59
            x247 = x245*x246
            x248 = x245*x30
            x249 = 1.1151003253100200517e-16*x63
            x250 = x248*x249
            x251 = x248*x28
            x252 = 8.0535023494612559287e-17*x67
            x253 = x251*x252
            x254 = x23*x251
            x255 = 4.1418012082943601919e-16*x71
            x256 = x254*x255
            x257 = 5.6949766614047452638e-16*x27
            x258 = x254*x257
            x259 = 1.4111167850007135721e-9*x9
            x260 = x26*x55
            x261 = 3.5400010327302223862e-18*x90
            x262 = x261*x51
            x263 = x239*x6
            x264 = 3.9825011618215001845e-17*x95
            x265 = x263*x264
            x266 = 1.1151003253100200517e-16*x100
            x267 = x235*x266
            x268 = x210*x263
            x269 = 8.0535023494612559287e-17*x105
            x270 = x268*x269
            x271 = 4.1418012082943601919e-16*x110
            x272 = x234*x271
            x273 = x208*x268
            x274 = 5.6949766614047452638e-16*x273
            x275 = x205*x31
            x276 = 1.4111167850007135721e-9*x51
            x277 = x124*x271
            x278 = x107*x266
            x279 = x261*x97
            x280 = x215*x240
            x281 = 3.5400010327302223862e-18*x280
            x282 = x26*x30
            x283 = x280*x282
            x284 = x28*x283
            x285 = x23*x284
            x286 = x255*x285
            x287 = x257*x285
            x288 = x10*x224
            x289 = x26*x38
            x290 = x236*x6
            x291 = x210*x290
            x292 = x264*x290
            x293 = x208*x291
            x294 = x269*x291
            x295 = 5.6949766614047452638e-16*x293
            x296 = 6.2933351692981731311e-18*x90
            x297 = x216*x296
            x298 = x215*x260
            x299 = x298*x6
            x300 = x210*x299
            x301 = x296*x97
            x302 = x215*x242
            x303 = x210*x6
            x304 = x302*x303
            x305 = x289*x59
            x306 = x216*x90
            x307 = 7.0800020654604447725e-17*x306
            x308 = x282*x306
            x309 = 1.9824005783289245363e-16*x63
            x310 = x308*x309
            x311 = x28*x308
            x312 = 1.4317337510153343873e-16*x67
            x313 = x311*x312
            x314 = x23*x311
            x315 = 7.3632021480788625634e-16*x71
            x316 = x314*x315
            x317 = 1.0124402953608436025e-15*x27
            x318 = x314*x317
            x319 = 2.5086520622234907949e-9*x90
            x320 = x282*x59
            x321 = 7.0800020654604447725e-17*x95
            x322 = x299*x321
            x323 = 1.9824005783289245363e-16*x100
            x324 = x235*x323
            x325 = 1.4317337510153343873e-16*x105
            x326 = x300*x325
            x327 = 7.3632021480788625634e-16*x110
            x328 = x234*x327
            x329 = x208*x300
            x330 = 1.0124402953608436025e-15*x329
            x331 = x26*x275
            x332 = 2.5086520622234907949e-9*x55
            x333 = x124*x327
            x334 = x107*x323
            x335 = x215*x320
            x336 = x303*x335
            x337 = x90*x97
            x338 = 7.0800020654604447725e-17*x337
            x339 = x215*x303
            x340 = x28*x282
            x341 = x339*x340
            x342 = x337*x341
            x343 = x23*x342
            x344 = x315*x343
            x345 = x317*x343
            x346 = x288*x8
            x347 = x282*x36
            x348 = x339*x347
            x349 = x305*x339
            x350 = x302*x6
            x351 = x321*x350
            x352 = x208*x304
            x353 = x304*x325
            x354 = 1.0124402953608436025e-15*x352
            x355 = x215*x305
            x356 = 7.965002323643000369e-16*x6*x95
            x357 = x356*x92
            x358 = x212*x356
            x359 = x215*x63
            x360 = x347*x359
            x361 = x6*x95
            x362 = x361*x92
            x363 = 2.2302006506200401033e-15*x362
            x364 = x32*x67
            x365 = x215*x340
            x366 = x362*x365
            x367 = 1.6107004698922511857e-15*x366
            x368 = x23*x366
            x369 = 8.2836024165887203838e-15*x71
            x370 = x368*x369
            x371 = 1.1389953322809490528e-14*x27
            x372 = x368*x371
            x373 = 2.8222335700014271443e-8*x95
            x374 = x23*x67
            x375 = x340*x359
            x376 = 2.2302006506200401033e-15*x100
            x377 = x235*x376
            x378 = 1.6107004698922511857e-15*x105
            x379 = x336*x378
            x380 = 8.2836024165887203838e-15*x110
            x381 = x234*x6
            x382 = x380*x381
            x383 = x208*x336
            x384 = 1.1389953322809490528e-14*x383
            x385 = x30*x331
            x386 = 2.8222335700014271443e-8*x59
            x387 = x124*x380
            x388 = x107*x376
            x389 = x212*x361
            x390 = 2.2302006506200401033e-15*x389
            x391 = x365*x389
            x392 = 1.6107004698922511857e-15*x391
            x393 = x23*x391
            x394 = x369*x393
            x395 = x371*x393
            x396 = x3*x346
            x397 = x208*x349
            x398 = x349*x378
            x399 = 1.1389953322809490528e-14*x397
            x400 = 6.2445618217361122893e-15*x100
            x401 = x235*x400
            x402 = x341*x63
            x403 = x208*x402
            x404 = x107*x400
            x405 = x348*x63
            x406 = x208*x405
            x407 = x364*x365
            x408 = 4.5099613156983033201e-15*x100
            x409 = x235*x408
            x410 = x100*x23
            x411 = x235*x365*x410
            x412 = x35*x71
            x413 = 2.3194086766448417075e-14*x412
            x414 = x27*x411
            x415 = 3.1891869303866573478e-14*x414
            x416 = 7.902253996003996004e-8*x100
            x417 = 2.3194086766448417075e-14*x71
            x418 = x365*x374
            x419 = 4.5099613156983033201e-15*x105
            x420 = x402*x419
            x421 = 2.3194086766448417075e-14*x110
            x422 = x381*x421
            x423 = 3.1891869303866573478e-14*x403
            x424 = x28*x385
            x425 = 7.902253996003996004e-8*x63
            x426 = x124*x421
            x427 = x341*x374
            x428 = x107*x208
            x429 = x408*x428
            x430 = x27*x341
            x431 = x410*x428
            x432 = x430*x431
            x433 = 3.1891869303866573478e-14*x432
            x434 = x396*x7
            x435 = x341*x364
            x436 = x405*x419
            x437 = 3.1891869303866573478e-14*x406
            x438 = 3.2571942835598857312e-15*x105
            x439 = x102*x438
            x440 = x209*x438
            x441 = x341*x412
            x442 = x105*x23
            x443 = x102*x442
            x444 = 1.6751284886879412332e-14*x443
            x445 = 2.3033016719459191956e-14*x430
            x446 = x443*x445
            x447 = 5.7071834415584415584e-8*x105
            x448 = x430*x71
            x449 = 1.6751284886879412332e-14*x110
            x450 = x381*x449
            x451 = x208*x427
            x452 = 2.3033016719459191956e-14*x451
            x453 = x23*x424
            x454 = 5.7071834415584415584e-8*x67
            x455 = x124*x449
            x456 = x209*x442
            x457 = 1.6751284886879412332e-14*x456
            x458 = x445*x456
            x459 = x434*x5
            x460 = x208*x435
            x461 = 2.3033016719459191956e-14*x460
            x462 = x110*x23
            x463 = 8.6149465132522691991e-14*x462
            x464 = x365*x381
            x465 = x463*x464
            x466 = x208*x448
            x467 = x124*x463
            x468 = x208*x441
            x469 = 1.1845551455721870149e-13*x462
            x470 = x469*x75
            x471 = x27*x464
            x472 = 2.9351229128014842301e-7*x110
            x473 = x469*x81
            x474 = 1.1845551455721870149e-13*x23
            x475 = x466*x474
            x476 = x27*x453
            x477 = 2.9351229128014842301e-7*x71
            x478 = x208*x430
            x479 = x124*x478
            x480 = x121*x459
            x481 = x468*x474
            x482 = 1.6287633251617571455e-13*x23*x478
            x483 = x114*x482
            x484 = x123*x482
            x485 = 4.0357940051020408163e-7*x476
            return jnp.asarray([x0*x42, x42*x43, x43*x45, x0*x45, -x37*x49, x39*x53, -x34*x57, x38*x61, -x36*x65, x32*x69, -x35*x73, x75*x77, x79*x80, x77*x81, -x27*x73, x23*x69, -x28*x65, x30*x61, -x26*x57, x31*x53, -x29*x49, -x85*x86, x85*x87, -x89*x91, x94*x96, -x101*x99, x104*x106, -x109*x111, x114*x116, x118*x119, x116*x123, -x115*x125, x115*x127, -x109*x128, x104*x129, -x130*x99, x131*x94, -x132*x89, -x134*x48, x135*x52, -x136*x56, x137*x60, -x138*x64, x139*x68, -x140*x27, x141*x81, x44*x78*x80, x141*x75, -x140*x35, x138*x32*x68, -x137*x36*x64, x136*x38*x60, -x135*x34*x56, x134*x39*x52, -x142*x20*x44*x48, -x132*x144, x131*x146, -x130*x148, x129*x150, -x128*x152, x127*x153, -x125*x153, x123*x154, x0*x117*x119, x114*x154, -x111*x152, x106*x150, -x101*x148, x146*x96, -x144*x91, x155*x87, -x155*x86, x157*x159, x159*x160, x161*x163, x163*x164, -x168*x39, x171*x34, -x174*x38, x177*x36, -x180*x32, x183*x35, -x185*x75, -x186*x187, -x185*x81, x183*x27, -x180*x23, x177*x28, -x174*x30, x171*x26, -x168*x31, -x189*x82, x161*x190, -x193*x92, x195*x196, -x102*x199, x201*x202, -x114*x204, -x205*x206, -x123*x204, x203*x207, -x199*x209, x196*x201*x7, -x193*x212, x190*x195*x8, -x189*x214, -x167*x217, x170*x218, -x173*x219, x176*x220, -x179*x221, x222*x27, -x223*x81, -x187*x224*x88, -x223*x75, x222*x35, -x179*x220*x32, x176*x219*x36, -x173*x218*x38, x170*x217*x34, -x162*x167*x226, -x214*x227, x190*x229*x97, -x212*x230, x107*x196*x231, -x209*x232, x207*x231, -x123*x233, -x118*x206*x37, -x114*x233, x202*x228*x234, -x102*x232, x157*x196*x235, -x230*x92, x164*x190, -x227*x82, x236*x238, x238*x239, x239*x241, x236*x241, -x242*x244, x247*x38, -x250*x36, x253*x32, -x256*x35, x258*x75, x186*x259, x258*x81, -x256*x27, x23*x253, -x250*x28, x247*x30, -x244*x260, -x216*x262, x265*x92, -x239*x267, x102*x270, -x263*x272, x114*x274, x275*x276, x123*x274, -x273*x277, x209*x270, -x273*x278, x212*x265, -x268*x279, -x260*x281, x246*x283, -x249*x284, x252*x285, -x27*x286, x287*x81, x259*x288*x92, x287*x75, -x286*x35, x252*x284*x32, -x249*x283*x36, x246*x280*x289, -x242*x281, -x279*x291, x212*x292, -x278*x293, x209*x294, -x277*x293, x123*x295, x205*x276*x39, x114*x295, -x272*x290, x102*x294, -x236*x267, x292*x92, -x226*x262, x242*x297, x260*x297, x300*x301, x301*x304, -x305*x307, x310*x36, -x313*x32, x316*x35, -x318*x75, -x319*x79*x88, -x318*x81, x27*x316, -x23*x313, x28*x310, -x307*x320, -x322*x92, x298*x324, -x102*x326, x299*x328, -x114*x330, -x331*x332, -x123*x330, x329*x333, -x209*x326, x329*x334, -x212*x322, -x336*x338, x309*x342, -x312*x343, x27*x344, -x345*x81, -x319*x346*x97, -x345*x75, x344*x35, -x312*x32*x342, x309*x337*x348, -x338*x349, -x212*x351, x334*x352, -x209*x353, x333*x352, -x123*x354, -x275*x332*x34, -x114*x354, x328*x350, -x102*x353, x302*x324, -x351*x92, x355*x357, x335*x357, x335*x358, x355*x358, -x360*x363, x364*x367, -x35*x370, x372*x75, x224*x373*x92, x372*x81, -x27*x370, x367*x374, -x363*x375, -x335*x377, x102*x379, -x335*x382, x114*x384, x385*x386, x123*x384, -x383*x387, x209*x379, -x383*x388, -x375*x390, x374*x392, -x27*x394, x395*x81, x102*x373*x396, x395*x75, -x35*x394, x364*x392, -x360*x390, -x388*x397, x209*x398, -x387*x397, x123*x399, x331*x38*x386, x114*x399, -x355*x382, x102*x398, -x355*x377, x360*x401, x375*x401, x403*x404, x404*x406, -x407*x409, x411*x413, -x415*x75, -x288*x416*x97, -x415*x81, x414*x417, -x409*x418, -x102*x420, x375*x422, -x114*x423, -x424*x425, -x123*x423, x403*x426, -x209*x420, -x427*x429, x417*x432, -x433*x81, -x107*x416*x434, -x433*x75, x341*x413*x431, -x429*x435, -x209*x436, x406*x426, -x123*x437, -x36*x385*x425, -x114*x437, x360*x422, -x102*x436, x435*x439, x427*x439, x427*x440, x435*x440, -x441*x444, x446*x75, x102*x346*x447, x446*x81, -x444*x448, -x418*x450, x114*x452, x453*x454, x123*x452, -x451*x455, -x448*x457, x458*x81, x112*x447*x459, x458*x75, -x441*x457, -x455*x460, x123*x461, x32*x424*x454, x114*x461, -x407*x450, x412*x465, x27*x465*x71, x466*x467, x467*x468, -x470*x471, -x107*x396*x472, -x471*x473, -x114*x475, -x476*x477, -x123*x475, -x473*x479, -x13*x472*x480, -x470*x479, -x123*x481, -x35*x453*x477, -x114*x481, x483*x75, x483*x81, x484*x81, x484*x75, 4.0357940051020408163e-7*x114*x434, x485*x81, 4.0357940051020408163e-7*x120*x480, x485*x75, x117*x78])

        case 400:
          def shape_functions(xi):
            x0 = 19.0*xi[1]
            x1 = x0 + 17.0
            x2 = xi[0] - 1.0
            x3 = x1*x2
            x4 = 19.0*xi[0]
            x5 = x4 + 1.0
            x6 = x0 + 1.0
            x7 = x4 + 3.0
            x8 = x0 + 3.0
            x9 = x4 + 5.0
            x10 = x0 + 5.0
            x11 = x4 + 7.0
            x12 = x0 + 7.0
            x13 = x4 + 9.0
            x14 = x0 + 9.0
            x15 = x4 + 11.0
            x16 = x0 + 11.0
            x17 = x4 + 13.0
            x18 = x0 + 13.0
            x19 = x4 + 15.0
            x20 = x0 + 15.0
            x21 = x4 + 17.0
            x22 = xi[1] - 1.0
            x23 = x4 - 1.0
            x24 = x0 - 1.0
            x25 = x4 - 3.0
            x26 = x0 - 3.0
            x27 = x4 - 5.0
            x28 = x0 - 5.0
            x29 = x4 - 7.0
            x30 = x0 - 7.0
            x31 = x4 - 9.0
            x32 = x0 - 9.0
            x33 = x4 - 11.0
            x34 = x0 - 11.0
            x35 = x4 - 13.0
            x36 = x0 - 13.0
            x37 = x4 - 15.0
            x38 = x0 - 15.0
            x39 = x4 - 17.0
            x40 = x0 - 17.0
            x41 = x10*x11*x12*x13*x14*x15*x16*x17*x18*x19*x20*x21*x22*x23*x24*x25*x26*x27*x28*x29*x30*x31*x32*x33*x34*x35*x36*x37*x38*x39*x40*x5*x6*x7*x8*x9
            x42 = 8.8751995036557687094e-44*x41
            x43 = xi[0] + 1.0
            x44 = x1*x43
            x45 = xi[1] + 1.0
            x46 = x21*x45
            x47 = x44*x46
            x48 = x10*x11*x12*x13*x14*x15*x16*x17*x18*x19*x20*x23*x24*x25*x26*x27*x28*x29*x30*x31*x32*x33*x34*x35*x36*x37*x38*x39*x40*x5*x6*x7*x8*x9
            x49 = 8.8751995036557687094e-44*x48
            x50 = x3*x46
            x51 = 3.2039470208197325041e-41*x43
            x52 = x22*x3
            x53 = x51*x52
            x54 = x10*x11*x12*x13*x14*x15*x16*x17*x20*x23*x24*x25*x26*x27*x28*x29*x30*x31*x32*x33*x34*x35*x36*x37*x38*x39*x40*x5*x6*x7*x8*x9
            x55 = x52*x54
            x56 = x18*x43
            x57 = 2.8835523187377592537e-40*x56
            x58 = x21*x57
            x59 = x10*x11*x12*x13*x14*x15*x16*x23*x24*x25*x26*x27*x28*x29*x30*x31*x32*x33*x34*x35*x36*x37*x38*x39*x40*x5*x6*x7*x8*x9
            x60 = x52*x59
            x61 = 1.6340129806180635771e-39*x19
            x62 = x20*x56
            x63 = x61*x62
            x64 = x21*x63
            x65 = x36*x52
            x66 = x21*x65
            x67 = x10*x11*x12*x13*x14*x16*x23*x24*x25*x26*x27*x28*x29*x30*x31*x32*x33*x34*x35*x37*x38*x39*x40*x5*x6*x7*x8*x9
            x68 = x17*x19
            x69 = x62*x68
            x70 = x67*x69
            x71 = 6.5360519224722543084e-39*x70
            x72 = x10*x11*x12*x14*x15*x16*x23*x24*x25*x26*x27*x28*x29*x30*x31*x32*x33*x34*x37*x38*x39*x40*x5*x6*x7*x8*x9
            x73 = x35*x69
            x74 = x66*x73
            x75 = 1.9608155767416762925e-38*x74
            x76 = x10*x12*x13*x14*x15*x16*x23*x24*x25*x26*x27*x28*x29*x30*x31*x32*x33*x34*x37*x38*x5*x6*x7*x8*x9
            x77 = x39*x40
            x78 = x74*x77
            x79 = 4.5752363457305780158e-38*x78
            x80 = x10*x11*x12*x13*x14*x15*x16*x23*x24*x25*x26*x27*x28*x29*x30*x31*x32*x33*x34*x5*x6*x7*x8
            x81 = x37*x38
            x82 = x78*x81
            x83 = 8.4968674992139306009e-38*x82
            x84 = x10*x11*x12*x13*x14*x23*x24*x25*x26*x27*x28*x29*x30*x31*x32*x33*x34*x5*x6*x8*x9
            x85 = x15*x16
            x86 = x82*x85
            x87 = 1.2745301248820895901e-37*x86
            x88 = x10*x11*x12*x13*x14*x23*x24*x25*x26*x27*x28*x29*x30*x31*x32*x6*x7*x8*x9
            x89 = x86*x88
            x90 = x33*x34
            x91 = 1.5577590415225539435e-37*x90
            x92 = x10*x11*x12*x24*x25*x26*x27*x28*x29*x30*x31*x32*x5*x6*x7*x8*x9
            x93 = x13*x14
            x94 = x91*x93
            x95 = x92*x94
            x96 = x10*x11*x12*x24*x26*x27*x28*x29*x30*x31*x32*x5*x7*x8*x9
            x97 = x23*x90
            x98 = x6*x93
            x99 = x97*x98
            x100 = x96*x99
            x101 = x25*x85
            x102 = x10*x101*x11*x12*x26*x28*x29*x30*x31*x32*x5*x7*x8*x9
            x103 = x24*x99
            x104 = x102*x103
            x105 = x10*x101*x103*x11*x26*x27*x28*x30*x31*x32*x5*x8*x9
            x106 = x7*x81
            x107 = x106*x12
            x108 = x105*x107
            x109 = x101*x103*x107*x11*x26*x27*x28*x29*x30*x32*x5*x8
            x110 = x77*x9
            x111 = x10*x110
            x112 = x109*x111
            x113 = 6.5360519224722543084e-39*x5
            x114 = x113*x34
            x115 = x13*x72
            x116 = x115*x17
            x117 = x110*x65
            x118 = x38*x80
            x119 = x35*x68
            x120 = x119*x20
            x121 = x118*x120
            x122 = x120*x18
            x123 = x36*x76
            x124 = x11*x123
            x125 = x124*x40
            x126 = x122*x125
            x127 = x45*x51
            x128 = x22*x47
            x129 = x128*x18
            x130 = 2.8835523187377592537e-40*x59*x68
            x131 = x54*x61
            x132 = x122*x128
            x133 = x36*x77
            x134 = x133*x84
            x135 = x106*x15
            x136 = x134*x135
            x137 = 6.5360519224722543084e-39*x136
            x138 = x132*x133
            x139 = x13*x97
            x140 = x81*x85
            x141 = x140*x92
            x142 = x139*x141
            x143 = 1.9608155767416762925e-38*x142
            x144 = 4.5752363457305780158e-38*x29
            x145 = x105*x106
            x146 = x144*x145
            x147 = x110*x36
            x148 = x132*x147
            x149 = x109*x31
            x150 = 8.4968674992139306009e-38*x149
            x151 = x101*x103*x11*x26*x27*x28*x30*x31*x32*x5
            x152 = x107*x151
            x153 = x29*x36
            x154 = x111*x153
            x155 = x132*x154
            x156 = 1.2745301248820895901e-37*x155
            x157 = x101*x96
            x158 = x23*x81
            x159 = x158*x94
            x160 = x138*x159
            x161 = x102*x27
            x162 = x161*x6
            x163 = x101*x103*x11*x27*x30*x31*x32*x5*x8
            x164 = x107*x28
            x165 = x163*x164
            x166 = x155*x26
            x167 = x107*x163
            x168 = 8.4968674992139306009e-38*x167
            x169 = x36*x5
            x170 = x111*x26
            x171 = x101*x103*x11*x164*x27*x31*x8
            x172 = x171*x32
            x173 = x170*x172
            x174 = x169*x173
            x175 = x144*x174
            x176 = x30*x5
            x177 = x171*x176
            x178 = 1.9608155767416762925e-38*x177
            x179 = x140*x88
            x180 = x179*x33
            x181 = x113*x180
            x182 = x15*x67
            x183 = x17*x182
            x184 = x183*x20*x61
            x185 = x37*x80
            x186 = 2.8835523187377592537e-40*x185
            x187 = x124*x39
            x188 = 3.2039470208197325041e-41*x187
            x189 = x50*x57
            x190 = x121*x147
            x191 = x36*x50
            x192 = x50*x73
            x193 = x133*x192
            x194 = x179*x193
            x195 = x112*x73
            x196 = 1.9608155767416762925e-38*x191
            x197 = x104*x81
            x198 = 8.4968674992139306009e-38*x193
            x199 = x140*x193
            x200 = x134*x140
            x201 = x80*x81
            x202 = x123*x77
            x203 = x72*x73
            x204 = x50*x59
            x205 = x22*x50
            x206 = x122*x205
            x207 = x147*x206
            x208 = x18*x205
            x209 = x133*x206
            x210 = x154*x206
            x211 = x210*x26
            x212 = 1.2745301248820895901e-37*x210
            x213 = x159*x209
            x214 = x2*x45
            x215 = 1.156624874515923434e-38*x22
            x216 = x2*x46
            x217 = x216*x73
            x218 = x11*x76
            x219 = x39*x73
            x220 = x45*x65
            x221 = x216*x22
            x222 = 1.0409623870643310906e-37*x56
            x223 = 5.8987868600312095133e-37*x221
            x224 = x19*x62
            x225 = x221*x36
            x226 = 7.078544232037451416e-36*x225
            x227 = x217*x22
            x228 = 1.6516603208087386637e-35*x227
            x229 = x133*x227
            x230 = 3.0673691672162289469e-35*x201
            x231 = x179*x229
            x232 = 5.623510139896419736e-35*x90
            x233 = x141*x93
            x234 = x232*x233
            x235 = 4.6010537508243434204e-35*x140
            x236 = x100*x235
            x237 = 3.0673691672162289469e-35*x197
            x238 = x108*x133
            x239 = 2.3595147440124838053e-36*x34
            x240 = x115*x69
            x241 = 1.0409623870643310906e-37*x118
            x242 = x125*x205
            x243 = 5.8987868600312095133e-37*x43
            x244 = x22*x40
            x245 = x191*x84
            x246 = 2.3595147440124838053e-36*x73
            x247 = x135*x246
            x248 = 7.078544232037451416e-36*x244
            x249 = x191*x73
            x250 = 1.6516603208087386637e-35*x145
            x251 = x153*x192
            x252 = x244*x251
            x253 = x244*x249
            x254 = 3.0673691672162289469e-35*x9
            x255 = 4.6010537508243434204e-35*x152
            x256 = x10*x252
            x257 = x256*x9
            x258 = 5.623510139896419736e-35*x81
            x259 = x253*x258
            x260 = x157*x93
            x261 = x260*x97
            x262 = x161*x99
            x263 = 4.6010537508243434204e-35*x165
            x264 = x167*x26
            x265 = 1.6516603208087386637e-35*x5
            x266 = x172*x26
            x267 = x10*x9
            x268 = x177*x26
            x269 = x192*x244
            x270 = 1.0409623870643310906e-37*x185
            x271 = x219*x22
            x272 = x191*x271
            x273 = x187*x205
            x274 = 5.8987868600312095133e-37*x273
            x275 = x192*x22
            x276 = x275*x39
            x277 = 7.078544232037451416e-36*x272
            x278 = x245*x271
            x279 = x14*x97
            x280 = x106*x16
            x281 = x224*x35
            x282 = x17*x35*x62
            x283 = x117*x45*x73
            x284 = x220*x77
            x285 = x180*x5
            x286 = x29*x73
            x287 = x220*x286
            x288 = x170*x287
            x289 = x111*x287
            x290 = x284*x73
            x291 = x258*x290
            x292 = 9.3686614835789798152e-37*x56
            x293 = x204*x22
            x294 = x147*x205
            x295 = x147*x275
            x296 = x185*x294
            x297 = 5.308908174028088562e-36*x56
            x298 = x19*x293
            x299 = 2.1235632696112354248e-35*x56
            x300 = x191*x22
            x301 = x300*x68
            x302 = x301*x67
            x303 = x300*x72
            x304 = x119*x56
            x305 = 6.3706898088337062743e-35*x304
            x306 = x205*x304
            x307 = 1.4864942887278647973e-34*x306
            x308 = x133*x306
            x309 = 2.7606322504946060522e-34*x308
            x310 = 5.0611591259067777624e-34*x90
            x311 = x308*x310
            x312 = 4.1409483757419090783e-34*x100
            x313 = x112*x300
            x314 = x34*x5
            x315 = x133*x205
            x316 = x179*x315
            x317 = x314*x316
            x318 = x115*x301
            x319 = 5.308908174028088562e-36*x205
            x320 = x319*x43
            x321 = x38*x7
            x322 = x134*x275
            x323 = x15*x322
            x324 = x193*x22
            x325 = x324*x38
            x326 = x325*x85
            x327 = x139*x92
            x328 = x29*x321
            x329 = 1.4864942887278647973e-34*x105*x324
            x330 = x12*x151
            x331 = x295*x8
            x332 = x330*x331
            x333 = x154*x275
            x334 = x321*x333
            x335 = 4.1409483757419090783e-34*x334
            x336 = 5.0611591259067777624e-34*x325
            x337 = x12*x163
            x338 = x28*x337
            x339 = x26*x334
            x340 = 1.4864942887278647973e-34*x5
            x341 = x103*x27*x31*x32*x8
            x342 = x101*x11
            x343 = x28*x342
            x344 = x12*x343
            x345 = x339*x344
            x346 = x176*x27*x31
            x347 = x103*x8
            x348 = 6.3706898088337062743e-35*x347
            x349 = 2.1235632696112354248e-35*x88
            x350 = x33*x5
            x351 = 5.308908174028088562e-36*x296
            x352 = x324*x37
            x353 = x352*x85
            x354 = x37*x7
            x355 = x176*x26
            x356 = x32*x333
            x357 = x355*x356
            x358 = x27*x357
            x359 = x12*x354
            x360 = x310*x353
            x361 = x92*x93
            x362 = x101*x333
            x363 = x341*x355
            x364 = x28*x363
            x365 = x362*x364
            x366 = x279*x92
            x367 = x282*x315
            x368 = x205*x282
            x369 = x154*x368
            x370 = 4.1409483757419090783e-34*x369
            x371 = x262*x81
            x372 = 5.0611591259067777624e-34*x367
            x373 = x261*x81
            x374 = x149*x294
            x375 = x145*x29
            x376 = x20*x43
            x377 = 3.0083812986159168518e-35*x376
            x378 = 3.0083812986159168518e-35*x205
            x379 = x120*x43
            x380 = 3.6100575583391002221e-34*x379
            x381 = x205*x379
            x382 = 8.423467636124567185e-34*x381
            x383 = x315*x379
            x384 = 1.5643582752802767629e-33*x201
            x385 = 2.8679901713471740654e-33*x90
            x386 = x233*x385
            x387 = 2.3465374129204151444e-33*x140
            x388 = x100*x387
            x389 = 1.5643582752802767629e-33*x197
            x390 = x205*x69
            x391 = 1.2033525194463667407e-34*x136
            x392 = x315*x69
            x393 = 3.6100575583391002221e-34*x142
            x394 = 8.423467636124567185e-34*x375
            x395 = 1.5643582752802767629e-33*x374
            x396 = x154*x390
            x397 = 2.3465374129204151444e-33*x396
            x398 = 2.8679901713471740654e-33*x392
            x399 = 1.5643582752802767629e-33*x264
            x400 = x266*x5
            x401 = 8.423467636124567185e-34*x400
            x402 = 3.6100575583391002221e-34*x268
            x403 = 1.2033525194463667407e-34*x285
            x404 = x275*x77
            x405 = x179*x404
            x406 = 3.6100575583391002221e-34*x275
            x407 = 8.423467636124567185e-34*x404
            x408 = x281*x315
            x409 = x205*x281
            x410 = x154*x409
            x411 = 2.3465374129204151444e-33*x410
            x412 = 2.8679901713471740654e-33*x408
            x413 = x15*x324
            x414 = x413*x81
            x415 = 4.8134100777854669628e-34*x88
            x416 = x324*x81
            x417 = x16*x416
            x418 = x333*x363
            x419 = 3.369387054449826874e-33*x164
            x420 = x15*x25
            x421 = x419*x420
            x422 = x10*x29
            x423 = x11*x164
            x424 = 6.2574331011211070517e-33*x423
            x425 = x25*x424
            x426 = 1.1471960685388696261e-32*x414*x90
            x427 = x347*x420
            x428 = x31*x357
            x429 = x11*x341
            x430 = x169*x170*x275*x30
            x431 = 1.4440230233356400889e-33*x358
            x432 = x23*x34
            x433 = 1.4440230233356400889e-33*x432
            x434 = x141*x324
            x435 = x13*x434
            x436 = x106*x343
            x437 = x24*x98
            x438 = x437*x8
            x439 = x26*x346
            x440 = x356*x439
            x441 = x432*x440
            x442 = x438*x441
            x443 = 6.2574331011211070517e-33*x342
            x444 = x164*x439
            x445 = x443*x444
            x446 = x29*x32*x331
            x447 = x437*x446
            x448 = x164*x342
            x449 = 9.3861496516816605776e-33*x448
            x450 = x158*x324
            x451 = 1.1471960685388696261e-32*x34*x450
            x452 = x161*x98
            x453 = x356*x438
            x454 = x346*x453
            x455 = x23*x419
            x456 = x26*x342
            x457 = x27*x31
            x458 = x453*x457
            x459 = x342*x438
            x460 = x333*x444
            x461 = x459*x460
            x462 = x23*x33
            x463 = x438*x462
            x464 = x33*x455
            x465 = x32*x430*x457
            x466 = x199*x22*x96
            x467 = 1.1471960685388696261e-32*x324
            x468 = x438*x440
            x469 = x324*x422
            x470 = x32*x469
            x471 = x16*x25
            x472 = x333*x347
            x473 = x423*x439
            x474 = x472*x473
            x475 = x333*x471
            x476 = x26*x5
            x477 = x107*x11*x418
            x478 = 9.3861496516816605776e-33*x423
            x479 = x176*x341
            x480 = x440*x471
            x481 = x8*x99
            x482 = x423*x481
            x483 = x93*x97
            x484 = x103*x446
            x485 = x11*x333*x364
            x486 = 4.3320690700069202666e-33*x448
            x487 = x139*x6
            x488 = x24*x8
            x489 = x487*x488
            x490 = x355*x472
            x491 = x279*x6
            x492 = x342*x488
            x493 = x491*x492
            x494 = x440*x489
            x495 = x101*x164
            x496 = 1.8772299303363321155e-32*x342
            x497 = x444*x470
            x498 = x344*x81
            x499 = 2.8158448955044981733e-32*x498
            x500 = x448*x489
            x501 = x26*x30
            x502 = 3.4415882056166088784e-32*x501
            x503 = x356*x457
            x504 = 3.4415882056166088784e-32*x90
            x505 = x102*x416
            x506 = x24*x505
            x507 = x347*x358
            x508 = 1.0108161163349480622e-32*x436
            x509 = x358*x448
            x510 = 3.4415882056166088784e-32*x509
            x511 = x483*x488
            x512 = 2.8158448955044981733e-32*x448
            x513 = x356*x512
            x514 = x27*x347
            x515 = x448*x476
            x516 = 1.0108161163349480622e-32*x515
            x517 = x31*x448
            x518 = x488*x491
            x519 = x107*x440
            x520 = 3.4415882056166088784e-32*x279*x416
            x521 = x24*x491
            x522 = x341*x476
            x523 = x164*x362
            x524 = 4.3802031707847749362e-32*x469
            x525 = 8.030372479772087383e-32*x436
            x526 = x333*x341
            x527 = 6.5703047561771624043e-32*x85
            x528 = 4.3802031707847749362e-32*x347
            x529 = x22*x249
            x530 = 6.5703047561771624043e-32*x111*x529
            x531 = 8.030372479772087383e-32*x511
            x532 = x448*x465
            x533 = 8.030372479772087383e-32*x481
            x534 = x333*x522
            x535 = x440*x495
            x536 = 1.2201994547186158751e-31*x81
            x537 = 1.491354889100530514e-31*x30
            x538 = 1.491354889100530514e-31*x90
            x539 = 1.2201994547186158751e-31*x85
            x540 = x103*x448
            x541 = 1.491354889100530514e-31*x483
            x542 = 1.2201994547186158751e-31*x469
            x543 = 1.8302991820779238126e-31*x333
            x544 = x543*x81
            x545 = x423*x85
            x546 = x501*x503
            x547 = 2.237032333650795771e-31*x440
            x548 = x448*x90
            x549 = x498*x547
            x550 = 2.7341506300176392756e-31*x448*x546
            x551 = 2.7341506300176392756e-31*x416*x90
            return jnp.asarray([x3*x42, -x42*x44, x47*x49, -x49*x50, -x48*x53, x55*x58, -x60*x64, x66*x71, -x72*x75, x76*x79, -x80*x83, x84*x87, -x89*x91, x86*x95, -x100*x87, x104*x83, -x108*x79, x112*x75, -x114*x89, x116*x64*x65, -x117*x121*x58, x126*x21*x53, x127*x41, -x129*x130, x128*x131, -x132*x137, x138*x143, -x138*x146, x148*x150, -x152*x156, x157*x160, -x160*x162, x156*x165, -x166*x168, x132*x175, -x166*x178, x138*x181, -x129*x184, x148*x186, -x132*x188, -x126*x50*x51, x189*x190, -x116*x191*x63, x114*x194, -x195*x196, 4.5752363457305780158e-38*x108*x193, -x197*x198, 1.2745301248820895901e-37*x100*x199, -x199*x95, x194*x91, -1.2745301248820895901e-37*x192*x200, x198*x201, -4.5752363457305780158e-38*x192*x202, x196*x203, -x191*x71, x204*x63, -x189*x54, x127*x3*x48, x188*x206, -x186*x207, x184*x208, -x181*x209, x178*x211, -x175*x206, x168*x211, -x165*x212, x162*x213, -x157*x213, x152*x212, -x150*x207, x146*x209, -x143*x209, x137*x206, -x131*x205, x130*x208, -3.2039470208197325041e-41*x214*x41, x214*x215*x43*x48, -x125*x215*x217, x124*x192*x215, -1.156624874515923434e-38*x218*x219*x220, -x221*x222*x54, x223*x224*x59, -2.3595147440124838053e-36*x225*x70, x203*x226, -x202*x228, x229*x230, -4.6010537508243434204e-35*x200*x227, x231*x232, -x229*x234, x229*x236, -x229*x237, x228*x238, -x195*x226, x231*x239*x5, -x223*x240*x36, x147*x227*x241, x119*x222*x242, -x120*x242*x243, x244*x245*x247, -x142*x248*x249, x250*x252, -x149*x253*x254, x255*x257, -x259*x261, x259*x262, -x257*x263, x254*x256*x264, -x257*x265*x266, x248*x251*x267*x268, -2.3595147440124838053e-36*x169*x180*x269, 5.8987868600312095133e-37*x218*x269, -x253*x270*x9, -x241*x272*x9, x274*x69, -x169*x179*x239*x276, x109*x267*x277, -1.6516603208087386637e-35*x108*x272, x237*x272, -x236*x272, x234*x272, -x179*x232*x272, x235*x278, -x230*x272, 1.6516603208087386637e-35*x123*x276, -x141*x277*x279, 2.3595147440124838053e-36*x278*x280, -x274*x281, 1.0409623870643310906e-37*x273*x282, x270*x283, -5.8987868600312095133e-37*x15*x45*x52*x70, x246*x284*x285, -7.078544232037451416e-36*x177*x288, x173*x265*x287, -3.0673691672162289469e-35*x167*x288, x263*x289, -x262*x291, x261*x291, -x255*x289, 3.0673691672162289469e-35*x149*x283, -x250*x284*x286, 7.078544232037451416e-36*x142*x290, -x247*x284*x84, x19*x243*x45*x55, -x222*x45*x60*x68, x17*x292*x293, -x118*x119*x292*x294, 9.3686614835789798152e-37*x295*x80, -9.3686614835789798152e-37*x282*x296, -x297*x298, x299*x302, -x303*x305, x202*x307, -x201*x309, 4.1409483757419090783e-34*x200*x306, -x179*x311, x233*x311, -x140*x308*x312, x197*x309, -x238*x307, x305*x313, -x119*x299*x317, x297*x318, x190*x320, -2.1235632696112354248e-35*x321*x323, 6.3706898088337062743e-35*x326*x327, -x328*x329, 2.7606322504946060522e-34*x328*x332, -x330*x335, x261*x336, -x262*x336, x335*x338, -2.7606322504946060522e-34*x337*x339, x340*x341*x345, -x345*x346*x348, x326*x349*x350, -5.308908174028088562e-36*x110*x118*x275, -x351*x69, x314*x349*x353, -x344*x348*x354*x358, x329*x359, -2.7606322504946060522e-34*x104*x352, x312*x353, -x360*x361, x360*x88, -4.1409483757419090783e-34*x322*x37*x85, 2.7606322504946060522e-34*x185*x324, -1.4864942887278647973e-34*x359*x365, 6.3706898088337062743e-35*x353*x366, -2.1235632696112354248e-35*x16*x322*x354, x281*x351, x183*x319*x62, -2.1235632696112354248e-35*x285*x367, 6.3706898088337062743e-35*x268*x369, -x266*x340*x369, 2.7606322504946060522e-34*x264*x369, -x165*x370, x371*x372, -x372*x373, x152*x370, -2.7606322504946060522e-34*x282*x374, 1.4864942887278647973e-34*x367*x375, -6.3706898088337062743e-35*x142*x367, 2.1235632696112354248e-35*x136*x368, -x320*x54, x298*x377, -x318*x377, x240*x378, -x182*x224*x378, -1.2033525194463667407e-34*x302*x376, x303*x380, -x202*x382, x383*x384, -2.3465374129204151444e-33*x200*x381, x316*x379*x385, -x383*x386, x383*x388, -x383*x389, x238*x382, -x313*x380, 1.2033525194463667407e-34*x317*x379, x390*x391, -x392*x393, x392*x394, -x395*x69, x152*x397, -x373*x398, x371*x398, -x165*x397, x396*x399, -x396*x401, x396*x402, -x392*x403, -1.2033525194463667407e-34*x314*x405, x112*x406, -x108*x407, x389*x404, -x388*x404, x386*x404, -x385*x405, x387*x404*x84, -x384*x404, x407*x76, -x406*x72, 1.2033525194463667407e-34*x205*x70, x403*x408, -x402*x410, x401*x410, -x399*x410, x165*x411, -x371*x412, x373*x412, -x152*x411, x281*x395, -x394*x408, x393*x408, -x391*x409, 4.8134100777854669628e-34*x106*x322, -x314*x414*x415, 4.8134100777854669628e-34*x194*x22*x5, -x350*x415*x417, -1.4440230233356400889e-33*x366*x414, x418*x421, -x363*x413*x422*x425, 9.3861496516816605776e-33*x323*x81, -x426*x88, x361*x426, -9.3861496516816605776e-33*x100*x414, x424*x427*x428, -x421*x429*x430, x423*x427*x431, x433*x435, -3.369387054449826874e-33*x436*x442, x432*x445*x447, -x437*x441*x449, x260*x451, -x451*x452, x432*x449*x454, -x107*x442*x443, x314*x455*x456*x458, -x433*x461, -x431*x448*x463, x459*x464*x465, -6.2574331011211070517e-33*x102*x33*x437*x450, 9.3861496516816605776e-33*x462*x466*x98, -x233*x33*x467, x180*x467, -9.3861496516816605776e-33*x158*x33*x344*x468, x445*x463*x470, -x101*x464*x468, 1.4440230233356400889e-33*x14*x434*x462, 1.4440230233356400889e-33*x471*x474, -x419*x429*x475*x476, 6.2574331011211070517e-33*x471*x477, -x475*x478*x479, 1.1471960685388696261e-32*x480*x482, -1.1471960685388696261e-32*x25*x417*x483*x96, x103*x478*x480, -x16*x425*x439*x484, 3.369387054449826874e-33*x25*x280*x485, -1.4440230233356400889e-33*x327*x417, 4.3320690700069202666e-33*x434*x97, -x358*x486*x489, x27*x486*x490, -4.3320690700069202666e-33*x460*x493, -1.0108161163349480622e-32*x494*x495, x489*x496*x497, -x494*x499, x500*x502*x503, -x435*x504, 2.8158448955044981733e-32*x466*x487, -1.8772299303363321155e-32*x487*x506, 1.0108161163349480622e-32*x465*x500, x507*x508, -1.8772299303363321155e-32*x109*x295, 2.8158448955044981733e-32*x103*x509, -x510*x511, x481*x510, -x176*x513*x514, x107*x496*x507, -x356*x514*x516, -1.0108161163349480622e-32*x171*x430, 1.8772299303363321155e-32*x490*x517, -2.8158448955044981733e-32*x474*x85, x461*x504, -x171*x333*x502, x439*x472*x499, -1.8772299303363321155e-32*x268*x469, 1.0108161163349480622e-32*x347*x362*x444, x503*x516*x518, -1.8772299303363321155e-32*x493*x519, x346*x513*x518, -x162*x520, x157*x520, -x440*x512*x521, x444*x446*x496*x521, -x440*x508*x518, 2.3585709381148788118e-32*x106*x365, -2.3585709381148788118e-32*x145*x324, 2.3585709381148788118e-32*x174*x275, -2.3585709381148788118e-32*x522*x523, -x106*x151*x524*x8, 6.5703047561771624043e-32*x105*x29*x416, -x501*x525*x526, x468*x525*x90, -x106*x485*x527, x428*x436*x528, 4.3802031707847749362e-32*x152*x331, -x152*x530, x531*x532, -x532*x533, x165*x530, -4.3802031707847749362e-32*x167*x170*x529, -x31*x356*x515*x528, x423*x527*x534, -8.030372479772087383e-32*x458*x515*x90, 8.030372479772087383e-32*x266*x333, -6.5703047561771624043e-32*x498*x534, x400*x524, 4.3802031707847749362e-32*x107*x362*x363, -6.5703047561771624043e-32*x479*x523, x533*x535, -x531*x535, 6.5703047561771624043e-32*x103*x535, -4.3802031707847749362e-32*x101*x444*x484, 8.1346630314574391672e-32*x149*x324, -8.1346630314574391672e-32*x355*x484*x517, 8.1346630314574391672e-32*x107*x342*x347*x428, -8.1346630314574391672e-32*x264*x469, -x29*x332*x536, x266*x29*x295*x537, -x342*x444*x447*x538, x473*x484*x539, 1.2201994547186158751e-31*x428*x540, -x506*x541, 1.491354889100530514e-31*x505*x99, -1.2201994547186158751e-31*x176*x347*x356*x517, -x477*x539, x459*x519*x538, -x107*x456*x526*x537, x26*x333*x337*x536, x165*x542, -1.491354889100530514e-31*x342*x481*x497, x492*x497*x541, -x152*x542, x330*x544, -1.8302991820779238126e-31*x103*x440*x545, x479*x543*x545, -x338*x544, -2.237032333650795771e-31*x540*x546, x437*x547*x548, 2.237032333650795771e-31*x466*x483, -x482*x547*x85, -2.237032333650795771e-31*x454*x548, 2.237032333650795771e-31*x172*x30*x333, x481*x549, -x511*x549, x511*x550, -x260*x551, x452*x551, -x481*x550])

        case 441:
          def shape_functions(xi):
            x0 = xi[0] - 1.0
            x1 = xi[1] - 1.0
            x2 = 2.0*xi[1]
            x3 = x2 + 1.0
            x4 = 5.0*xi[1]
            x5 = x4 + 1.0
            x6 = 10.0*xi[1]
            x7 = x6 + 1.0
            x8 = x4 + 2.0
            x9 = x4 + 4.0
            x10 = x4 + 3.0
            x11 = x6 + 3.0
            x12 = x6 + 7.0
            x13 = x6 + 9.0
            x14 = x2 - 1.0
            x15 = x4 - 1.0
            x16 = x6 - 1.0
            x17 = x4 - 2.0
            x18 = x4 - 4.0
            x19 = x4 - 3.0
            x20 = x6 - 3.0
            x21 = x6 - 7.0
            x22 = x6 - 9.0
            x23 = x10*x11*x12*x13*x14*x15*x16*x17*x18*x19*x20*x21*x22*x3*x5*x7*x8*x9*xi[1]
            x24 = x1*x23
            x25 = 2.0*xi[0]
            x26 = x25 + 1.0
            x27 = 5.0*xi[0]
            x28 = x27 + 1.0
            x29 = 10.0*xi[0]
            x30 = x29 + 1.0
            x31 = x27 + 2.0
            x32 = x27 + 4.0
            x33 = x27 + 3.0
            x34 = x29 + 3.0
            x35 = x29 + 7.0
            x36 = x29 + 9.0
            x37 = x25 - 1.0
            x38 = x27 - 1.0
            x39 = x29 - 1.0
            x40 = x27 - 2.0
            x41 = x27 - 4.0
            x42 = x27 - 3.0
            x43 = x29 - 3.0
            x44 = x29 - 7.0
            x45 = x29 - 9.0
            x46 = x26*x28*x30*x31*x32*x33*x34*x35*x36*x37*x38*x39*x40*x41*x42*x43*x44*x45*xi[0]
            x47 = 6.9200736110635268372e-26*x46
            x48 = x24*x47
            x49 = xi[0] + 1.0
            x50 = xi[1] + 1.0
            x51 = x23*x47*x50
            x52 = x24*x49
            x53 = x0*x26*x28*x30*x31*x32*x33*x34*x35*x37*x38*x39*x40*x41*x42*x43*x44*xi[0]
            x54 = 1.3840147222127053674e-23*x53
            x55 = x52*x54
            x56 = x36*x52
            x57 = x0*x26*x28*x30*x31*x33*x34*x35*x37*x38*x39*x40*x42*x43*x44*x45*xi[0]
            x58 = 6.5740699305103504953e-23*x57
            x59 = x56*x58
            x60 = x32*x56
            x61 = x0*x26*x28*x30*x31*x33*x34*x37*x38*x39*x40*x41*x42*x43*x45*xi[0]
            x62 = 7.8888839166124205944e-22*x61
            x63 = x60*x62
            x64 = x35*x60
            x65 = x0*x26*x28*x30*x31*x34*x37*x38*x39*x40*x41*x43*x44*x45*xi[0]
            x66 = 1.6763878322801393763e-21*x65
            x67 = x64*x66
            x68 = x33*x64
            x69 = x0*x28*x30*x31*x34*x38*x39*x40*x41*x42*x43*x44*x45*xi[0]
            x70 = 2.1457764253185784017e-21*x69
            x71 = x68*x70
            x72 = x26*x68
            x73 = x0*x28*x30*x34*x37*x38*x39*x41*x42*x43*x44*x45*xi[0]
            x74 = 1.341110265824111501e-20*x73
            x75 = x72*x74
            x76 = x31*x72
            x77 = x0*x28*x30*x37*x38*x39*x40*x41*x42*x44*x45*xi[0]
            x78 = 5.3644410632964460042e-20*x77
            x79 = x76*x78
            x80 = x34*x76
            x81 = x0*x30*x37*x39*x40*x41*x42*x43*x44*x45*xi[0]
            x82 = 4.3586083639283623784e-20*x81
            x83 = x80*x82
            x84 = x0*x37*x38*x40*x41*x42*x43*x44*x45*xi[0]
            x85 = x39*x84
            x86 = 1.1622955637142299676e-19*x28
            x87 = x80*x86
            x88 = 7594058.4281266233059*xi[0]**20 - 29237124.948287499728*xi[0]**18 + 46662451.417466849566*xi[0]**16 - 40202717.496749499983*xi[0]**14 + 20418933.234909475907*xi[0]**12 - 6274158.9818744284408*xi[0]**10 + 1153141.6151619398331*xi[0]**8 - 121028.00916570045942*xi[0]**6 + 6598.7171853566529492*xi[0]**4 - 154.97677311665406904*xi[0]**2 + 1.0
            x89 = x1*x88
            x90 = 2.6306032789197855094e-13*x23
            x91 = x30*x84
            x92 = x10*x11*x12*x14*x15*x16*x17*x18*x19*x20*x21*x22*x3*x5*x50*x7*x8*xi[1]
            x93 = x1*x46
            x94 = x49*x93
            x95 = x92*x94
            x96 = 1.3840147222127053674e-23*x9
            x97 = 6.5740699305103504953e-23*x13
            x98 = x10*x11*x13*x14*x15*x16*x17*x18*x19*x20*x21*x3*x5*x50*x7*x8*xi[1]
            x99 = x94*x98
            x100 = x22*x9
            x101 = 7.8888839166124205944e-22*x100
            x102 = x100*x11*x14*x15*x16*x17*x19*x20*x21*x3*x5*x50*x7*x8*xi[1]
            x103 = x12*x94
            x104 = x102*x103
            x105 = x13*x18
            x106 = 1.6763878322801393763e-21*x105
            x107 = x105*x11*x14*x15*x16*x17*x19*x20*x5*x50*x7*x8*xi[1]
            x108 = x10*x103
            x109 = x107*x108
            x110 = x100*x21
            x111 = 2.1457764253185784017e-21*x110
            x112 = x11*x110*x14*x15*x16*x17*x20*x5*x50*x7*xi[1]
            x113 = x108*x3
            x114 = x112*x113
            x115 = x105*x19
            x116 = 1.341110265824111501e-20*x115
            x117 = x115*x15*x16*x17*x20*x5*x50*x7*xi[1]
            x118 = x113*x8
            x119 = x117*x118
            x120 = x110*x14
            x121 = 5.3644410632964460042e-20*x120
            x122 = x120*x15*x16*x20*x50*x7*xi[1]
            x123 = x11*x118
            x124 = x122*x123
            x125 = x115*x17
            x126 = 4.3586083639283623784e-20*x125
            x127 = x125*x15*x16*x50*xi[1]
            x128 = x120*x20
            x129 = x127*x128
            x130 = x123*x5
            x131 = 1.1622955637142299676e-19*x130
            x132 = 7594058.4281266233059*xi[1]**20 - 29237124.948287499728*xi[1]**18 + 46662451.417466849566*xi[1]**16 - 40202717.496749499983*xi[1]**14 + 20418933.234909475907*xi[1]**12 - 6274158.9818744284408*xi[1]**10 + 1153141.6151619398331*xi[1]**8 - 121028.00916570045942*xi[1]**6 + 6598.7171853566529492*xi[1]**4 - 154.97677311665406904*xi[1]**2 + 1.0
            x133 = x132*x49
            x134 = 2.6306032789197855094e-13*x46
            x135 = x125*x15
            x136 = x128*x50*xi[1]
            x137 = x136*x7
            x138 = x135*x137
            x139 = x137*x16
            x140 = x126*x139
            x141 = x127*x7
            x142 = x121*x141
            x143 = x116*x5
            x144 = x11*x111
            x145 = x106*x8
            x146 = x101*x3
            x147 = x10*x97
            x148 = x12*x96
            x149 = x36*x49
            x150 = x149*x23*x50
            x151 = x150*x32
            x152 = x151*x35
            x153 = x152*x33
            x154 = x153*x26
            x155 = x154*x31
            x156 = x155*x34
            x157 = x156*x82
            x158 = x156*x86
            x159 = x45*x49
            x160 = x0*x93
            x161 = x160*x98
            x162 = x12*x160
            x163 = x102*x162
            x164 = x10*x162
            x165 = x107*x164
            x166 = x164*x3
            x167 = x112*x166
            x168 = x166*x8
            x169 = x117*x168
            x170 = x11*x168
            x171 = x122*x170
            x172 = x170*x5
            x173 = 1.1622955637142299676e-19*x172
            x174 = x160*x92
            x175 = x1*x53
            x176 = x159*x175
            x177 = x9*x92
            x178 = 2.7680294444254107349e-21*x177
            x179 = x149*x175
            x180 = x179*x98
            x181 = x12*x9
            x182 = 2.7680294444254107349e-21*x181
            x183 = x176*x98
            x184 = x1*x149
            x185 = x177*x184
            x186 = 1.3148139861020700991e-20*x57
            x187 = x185*x186
            x188 = x185*x32
            x189 = 1.5777767833224841189e-19*x61
            x190 = x188*x189
            x191 = x188*x35
            x192 = 3.3527756645602787526e-19*x65
            x193 = x191*x192
            x194 = x191*x33
            x195 = 4.2915528506371568033e-19*x69
            x196 = x194*x195
            x197 = x194*x26
            x198 = 2.6822205316482230021e-18*x73
            x199 = x197*x198
            x200 = x197*x31
            x201 = 1.0728882126592892008e-17*x77
            x202 = x200*x201
            x203 = x200*x34
            x204 = 8.7172167278567247568e-18*x81
            x205 = x203*x204
            x206 = 2.3245911274284599351e-17*x28
            x207 = x203*x206
            x208 = x89*x92
            x209 = 5.2612065578395710189e-11*x9
            x210 = 1.3148139861020700991e-20*x13
            x211 = x179*x210
            x212 = 1.5777767833224841189e-19*x100
            x213 = x12*x179
            x214 = 3.3527756645602787526e-19*x105
            x215 = x213*x214
            x216 = x10*x213
            x217 = x107*x216
            x218 = 4.2915528506371568033e-19*x110
            x219 = x216*x3
            x220 = 2.6822205316482230021e-18*x115
            x221 = x219*x220
            x222 = x219*x8
            x223 = x117*x222
            x224 = 1.0728882126592892008e-17*x120
            x225 = x11*x222
            x226 = 8.7172167278567247568e-18*x125
            x227 = x225*x226
            x228 = x225*x5
            x229 = 2.3245911274284599351e-17*x228
            x230 = x133*x36
            x231 = 5.2612065578395710189e-11*x53
            x232 = x139*x5
            x233 = x141*x224
            x234 = x11*x8
            x235 = x234*x5
            x236 = x122*x235
            x237 = x10*x3
            x238 = x237*x8
            x239 = x112*x238
            x240 = x10*x12
            x241 = x102*x240
            x242 = x184*x32
            x243 = x242*x98
            x244 = x181*x243
            x245 = x244*x35
            x246 = x245*x33
            x247 = x246*x26
            x248 = x247*x31
            x249 = x248*x34
            x250 = x204*x249
            x251 = x206*x249
            x252 = x12*x89
            x253 = x184*x41
            x254 = x253*x98
            x255 = x176*x210
            x256 = x12*x176
            x257 = x237*x256
            x258 = x214*x256
            x259 = x234*x257
            x260 = x220*x257
            x261 = x235*x257
            x262 = x226*x259
            x263 = 2.3245911274284599351e-17*x261
            x264 = x117*x238
            x265 = x107*x240
            x266 = x253*x57
            x267 = x13*x92
            x268 = 6.2453664339848329705e-20*x267
            x269 = x242*x57
            x270 = x13*x241
            x271 = 6.2453664339848329705e-20*x270
            x272 = x44*x61
            x273 = x242*x267
            x274 = 7.4944397207817995646e-19*x273
            x275 = x273*x35
            x276 = 1.5925684406661324075e-18*x65
            x277 = x275*x276
            x278 = x275*x33
            x279 = 2.0384876040526494816e-18*x69
            x280 = x278*x279
            x281 = x26*x278
            x282 = 1.274054752532905926e-17*x73
            x283 = x281*x282
            x284 = x281*x31
            x285 = 5.096219010131623704e-17*x77
            x286 = x284*x285
            x287 = x284*x34
            x288 = 4.1406779457319442595e-17*x81
            x289 = x287*x288
            x290 = 1.1041807855285184692e-16*x28
            x291 = x287*x290
            x292 = 2.499073114973796234e-10*x13
            x293 = x35*x61
            x294 = 7.4944397207817995646e-19*x100
            x295 = x294*x57
            x296 = x12*x269
            x297 = 1.5925684406661324075e-18*x105
            x298 = x296*x297
            x299 = 2.0384876040526494816e-18*x110
            x300 = x265*x299
            x301 = x237*x296
            x302 = 1.274054752532905926e-17*x115
            x303 = x301*x302
            x304 = 5.096219010131623704e-17*x120
            x305 = x264*x304
            x306 = x234*x301
            x307 = 4.1406779457319442595e-17*x125
            x308 = x306*x307
            x309 = x235*x301
            x310 = 1.1041807855285184692e-16*x309
            x311 = x230*x32
            x312 = 2.499073114973796234e-10*x57
            x313 = x141*x304
            x314 = x117*x299
            x315 = x107*x294
            x316 = x242*x270
            x317 = 7.4944397207817995646e-19*x316
            x318 = x33*x35
            x319 = x316*x318
            x320 = x26*x319
            x321 = x31*x320
            x322 = x321*x34
            x323 = x288*x322
            x324 = x290*x322
            x325 = x10*x252
            x326 = x35*x42
            x327 = x12*x266
            x328 = x237*x327
            x329 = x297*x327
            x330 = x234*x328
            x331 = x302*x328
            x332 = x235*x328
            x333 = x307*x330
            x334 = 1.1041807855285184692e-16*x332
            x335 = 8.9933276649381594776e-18*x100
            x336 = x243*x335
            x337 = x242*x293
            x338 = x12*x337
            x339 = x237*x338
            x340 = x107*x335
            x341 = x242*x272
            x342 = x12*x237
            x343 = x341*x342
            x344 = x326*x65
            x345 = x100*x243
            x346 = 1.911082128799358889e-17*x345
            x347 = x318*x345
            x348 = 2.4461851248631793779e-17*x69
            x349 = x347*x348
            x350 = x26*x347
            x351 = 1.5288657030394871112e-16*x73
            x352 = x350*x351
            x353 = x31*x350
            x354 = 6.1154628121579484447e-16*x77
            x355 = x353*x354
            x356 = x34*x353
            x357 = 4.9688135348783331114e-16*x81
            x358 = x356*x357
            x359 = 1.325016942634222163e-15*x28
            x360 = x356*x359
            x361 = 2.9988877379685554807e-9*x100
            x362 = x318*x65
            x363 = 1.911082128799358889e-17*x105
            x364 = x338*x363
            x365 = 2.4461851248631793779e-17*x110
            x366 = x265*x365
            x367 = 1.5288657030394871112e-16*x115
            x368 = x339*x367
            x369 = 6.1154628121579484447e-16*x120
            x370 = x264*x369
            x371 = x234*x339
            x372 = 4.9688135348783331114e-16*x125
            x373 = x371*x372
            x374 = x235*x339
            x375 = 1.325016942634222163e-15*x374
            x376 = x311*x35
            x377 = 2.9988877379685554807e-9*x61
            x378 = x141*x369
            x379 = x117*x365
            x380 = x242*x362
            x381 = x342*x380
            x382 = x100*x107
            x383 = 1.911082128799358889e-17*x382
            x384 = x242*x342
            x385 = x26*x318
            x386 = x384*x385
            x387 = x382*x386
            x388 = x31*x387
            x389 = x34*x388
            x390 = x357*x389
            x391 = x359*x389
            x392 = x3*x325
            x393 = x318*x37
            x394 = x384*x393
            x395 = x344*x384
            x396 = x12*x341
            x397 = x363*x396
            x398 = x234*x343
            x399 = x343*x367
            x400 = x235*x343
            x401 = x372*x398
            x402 = 1.325016942634222163e-15*x400
            x403 = x242*x344
            x404 = 4.0610495236986376391e-17*x105*x12
            x405 = x102*x404
            x406 = x239*x404
            x407 = x242*x69
            x408 = x393*x407
            x409 = x105*x12
            x410 = x102*x409
            x411 = 5.198143390334256178e-17*x410
            x412 = x40*x73
            x413 = x242*x385
            x414 = x410*x413
            x415 = 3.2488396189589101113e-16*x414
            x416 = x31*x414
            x417 = 1.2995358475835640445e-15*x77
            x418 = x416*x417
            x419 = x34*x416
            x420 = 1.0558728761616457862e-15*x81
            x421 = x419*x420
            x422 = 2.8156610030977220964e-15*x28
            x423 = x419*x422
            x424 = 6.3726364431831803966e-9*x105
            x425 = x31*x73
            x426 = x385*x407
            x427 = 5.198143390334256178e-17*x110
            x428 = x265*x427
            x429 = 3.2488396189589101113e-16*x115
            x430 = x381*x429
            x431 = 1.2995358475835640445e-15*x120
            x432 = x12*x264
            x433 = x431*x432
            x434 = x234*x381
            x435 = 1.0558728761616457862e-15*x125
            x436 = x434*x435
            x437 = x235*x381
            x438 = 2.8156610030977220964e-15*x437
            x439 = x33*x376
            x440 = 6.3726364431831803966e-9*x65
            x441 = x141*x431
            x442 = x117*x427
            x443 = x239*x409
            x444 = 5.198143390334256178e-17*x443
            x445 = x413*x443
            x446 = 3.2488396189589101113e-16*x445
            x447 = x31*x34
            x448 = x445*x447
            x449 = x420*x448
            x450 = x422*x448
            x451 = x392*x8
            x452 = x31*x43
            x453 = x234*x395
            x454 = x395*x429
            x455 = x235*x395
            x456 = x435*x453
            x457 = 2.8156610030977220964e-15*x455
            x458 = 6.6536235396278479079e-17*x110
            x459 = x265*x458
            x460 = x386*x69
            x461 = x234*x460
            x462 = x117*x458
            x463 = x394*x69
            x464 = x234*x463
            x465 = x412*x413
            x466 = 4.1585147122674049424e-16*x110
            x467 = x265*x466
            x468 = x110*x265*x413
            x469 = x452*x77
            x470 = 1.663405884906961977e-15*x469
            x471 = x447*x468
            x472 = 1.3515172814869066063e-15*x81
            x473 = x471*x472
            x474 = 3.6040460839650842834e-15*x28
            x475 = x471*x474
            x476 = 8.1569746472744709076e-9*x110
            x477 = 1.663405884906961977e-15*x77
            x478 = x413*x425
            x479 = 4.1585147122674049424e-16*x115
            x480 = x460*x479
            x481 = 1.663405884906961977e-15*x120
            x482 = x432*x481
            x483 = 1.3515172814869066063e-15*x125
            x484 = x461*x483
            x485 = x235*x460
            x486 = 3.6040460839650842834e-15*x485
            x487 = x26*x439
            x488 = 8.1569746472744709076e-9*x69
            x489 = x141*x481
            x490 = x386*x425
            x491 = x117*x234
            x492 = x466*x491
            x493 = x386*x447
            x494 = x110*x491
            x495 = x493*x494
            x496 = x472*x495
            x497 = x474*x495
            x498 = x11*x451
            x499 = x386*x412
            x500 = x463*x479
            x501 = x235*x463
            x502 = x464*x483
            x503 = 3.6040460839650842834e-15*x501
            x504 = 2.599071695167128089e-15*x115
            x505 = x112*x504
            x506 = x236*x504
            x507 = x386*x469
            x508 = x112*x115
            x509 = 1.0396286780668512356e-14*x508
            x510 = x38*x81
            x511 = x493*x508
            x512 = 8.4469830092931662893e-15*x511
            x513 = 2.2525288024781776771e-14*x28
            x514 = x511*x513
            x515 = 5.0981091545465443173e-8*x115
            x516 = x28*x81
            x517 = x493*x77
            x518 = 1.0396286780668512356e-14*x120
            x519 = x432*x518
            x520 = x125*x234
            x521 = 8.4469830092931662893e-15*x520
            x522 = x490*x521
            x523 = x235*x490
            x524 = 2.2525288024781776771e-14*x523
            x525 = x31*x487
            x526 = 5.0981091545465443173e-8*x73
            x527 = x141*x518
            x528 = x115*x236
            x529 = 1.0396286780668512356e-14*x528
            x530 = x493*x528
            x531 = 8.4469830092931662893e-15*x530
            x532 = x513*x530
            x533 = x498*x5
            x534 = x235*x499
            x535 = x499*x521
            x536 = 2.2525288024781776771e-14*x534
            x537 = 4.1585147122674049424e-14*x120
            x538 = x413*x432
            x539 = x537*x538
            x540 = x235*x517
            x541 = x141*x537
            x542 = x235*x507
            x543 = x447*x538
            x544 = 3.3787932037172665157e-14*x120
            x545 = x510*x544
            x546 = x28*x85
            x547 = 9.0101152099127107086e-14*x120
            x548 = x543*x547
            x549 = 2.0392436618186177269e-7*x120
            x550 = x28*x91
            x551 = x516*x544
            x552 = 3.3787932037172665157e-14*x520
            x553 = x517*x552
            x554 = 9.0101152099127107086e-14*x540
            x555 = x34*x525
            x556 = 2.0392436618186177269e-7*x77
            x557 = x235*x493
            x558 = x141*x557
            x559 = x547*x558
            x560 = x533*x7
            x561 = x507*x552
            x562 = 9.0101152099127107086e-14*x542
            x563 = x493*x520
            x564 = 2.745269478020279044e-14*x563
            x565 = x122*x564
            x566 = x232*x564
            x567 = 7.3207186080540774507e-14*x563
            x568 = x122*x567
            x569 = 1.6568854752276269031e-7*x125
            x570 = 7.3207186080540774507e-14*x557
            x571 = x516*x570
            x572 = x28*x555
            x573 = 1.6568854752276269031e-7*x81
            x574 = x232*x567
            x575 = x136*x560
            x576 = x510*x570
            x577 = 1.9521916288144206535e-13*x557
            x578 = x129*x577
            x579 = x138*x577
            x580 = 4.4183612672736717416e-7*x572
            return jnp.asarray([x0*x48, x48*x49, x49*x51, x0*x51, -x45*x55, x41*x59, -x44*x63, x42*x67, -x37*x71, x40*x75, -x43*x79, x38*x83, -x85*x87, x89*x90, -x87*x91, x28*x83, -x34*x79, x31*x75, -x26*x71, x33*x67, -x35*x63, x32*x59, -x36*x55, -x95*x96, x95*x97, -x101*x99, x104*x106, -x109*x111, x114*x116, -x119*x121, x124*x126, -x129*x131, x133*x134, -x131*x138, x130*x140, -x130*x142, x124*x143, -x119*x144, x114*x145, -x109*x146, x104*x147, -x148*x99, -x150*x54, x151*x58, -x152*x62, x153*x66, -x154*x70, x155*x74, -x156*x78, x157*x28, -x158*x91, x50*x88*x90, -x158*x85, x157*x38, -x155*x43*x78, x154*x40*x74, -x153*x37*x70, x152*x42*x66, -x151*x44*x62, x150*x41*x58, -x159*x23*x50*x54, -x148*x161, x147*x163, -x146*x165, x145*x167, -x144*x169, x143*x171, -x142*x172, x140*x172, -x138*x173, x0*x132*x134, -x129*x173, x126*x171, -x121*x169, x116*x167, -x111*x165, x106*x163, -x101*x161, x174*x97, -x174*x96, x176*x178, x178*x179, x180*x182, x182*x183, -x187*x41, x190*x44, -x193*x42, x196*x37, -x199*x40, x202*x43, -x205*x38, x207*x85, -x208*x209, x207*x91, -x205*x28, x202*x34, -x199*x31, x196*x26, -x193*x33, x190*x35, -x187*x32, -x211*x92, x180*x212, -x102*x215, x217*x218, -x112*x221, x223*x224, -x122*x227, x129*x229, -x230*x231, x138*x229, -x227*x232, x228*x233, -x221*x236, x11*x218*x223, -x215*x239, x212*x217*x3, -x211*x241, -x186*x244, x189*x245, -x192*x246, x195*x247, -x198*x248, x201*x249, -x250*x28, x251*x91, -x209*x252*x98, x251*x85, -x250*x38, x201*x248*x43, -x198*x247*x40, x195*x246*x37, -x192*x245*x42, x189*x244*x44, -x181*x186*x254, -x241*x255, x107*x212*x257, -x239*x258, x117*x218*x259, -x236*x260, x233*x261, -x232*x262, x138*x263, -x133*x231*x45, x129*x263, -x122*x262, x224*x256*x264, -x112*x260, x176*x218*x265, -x102*x258, x183*x212, -x255*x92, x266*x268, x268*x269, x269*x271, x266*x271, -x272*x274, x277*x42, -x280*x37, x283*x40, -x286*x43, x289*x38, -x291*x85, x208*x292, -x291*x91, x28*x289, -x286*x34, x283*x31, -x26*x280, x277*x33, -x274*x293, -x243*x295, x102*x298, -x269*x300, x112*x303, -x296*x305, x122*x308, -x129*x310, x311*x312, -x138*x310, x232*x308, -x309*x313, x236*x303, -x306*x314, x239*x298, -x301*x315, -x293*x317, x276*x319, -x279*x320, x282*x321, -x285*x322, x28*x323, -x324*x91, x102*x292*x325, -x324*x85, x323*x38, -x285*x321*x43, x282*x320*x40, -x279*x319*x37, x276*x316*x326, -x272*x317, -x315*x328, x239*x329, -x314*x330, x236*x331, -x313*x332, x232*x333, -x138*x334, x230*x312*x41, -x129*x334, x122*x333, -x305*x327, x112*x331, -x266*x300, x102*x329, -x254*x295, x272*x336, x293*x336, x339*x340, x340*x343, -x344*x346, x349*x37, -x352*x40, x355*x43, -x358*x38, x360*x85, -x361*x89*x98, x360*x91, -x28*x358, x34*x355, -x31*x352, x26*x349, -x346*x362, -x102*x364, x337*x366, -x112*x368, x338*x370, -x122*x373, x129*x375, -x376*x377, x138*x375, -x232*x373, x374*x378, -x236*x368, x371*x379, -x239*x364, -x381*x383, x348*x387, -x351*x388, x354*x389, -x28*x390, x391*x91, -x107*x361*x392, x391*x85, -x38*x390, x354*x388*x43, -x351*x387*x40, x348*x382*x394, -x383*x395, -x239*x397, x379*x398, -x236*x399, x378*x400, -x232*x401, x138*x402, -x311*x377*x44, x129*x402, -x122*x401, x370*x396, -x112*x399, x341*x366, -x102*x397, x403*x405, x380*x405, x380*x406, x403*x406, -x408*x411, x412*x415, -x418*x43, x38*x421, -x423*x85, x102*x252*x424, -x423*x91, x28*x421, -x34*x418, x415*x425, -x411*x426, -x380*x428, x112*x430, -x380*x433, x122*x436, -x129*x438, x439*x440, -x138*x438, x232*x436, -x437*x441, x236*x430, -x434*x442, -x426*x444, x425*x446, -x417*x448, x28*x449, -x450*x91, x112*x424*x451, -x450*x85, x38*x449, -x417*x445*x452, x412*x446, -x408*x444, -x442*x453, x236*x454, -x441*x455, x232*x456, -x138*x457, x376*x42*x440, -x129*x457, x122*x456, -x403*x433, x112*x454, -x403*x428, x408*x459, x426*x459, x461*x462, x462*x464, -x465*x467, x468*x470, -x38*x473, x475*x85, -x107*x325*x476, x475*x91, -x28*x473, x471*x477, -x467*x478, -x112*x480, x426*x482, -x122*x484, x129*x486, -x487*x488, x138*x486, -x232*x484, x485*x489, -x236*x480, -x490*x492, x477*x495, -x28*x496, x497*x91, -x117*x476*x498, x497*x85, -x38*x496, x386*x470*x494, -x492*x499, -x236*x500, x489*x501, -x232*x502, x138*x503, -x37*x439*x488, x129*x503, -x122*x502, x408*x482, -x112*x500, x499*x505, x490*x505, x490*x506, x499*x506, -x507*x509, x510*x512, -x514*x85, x112*x392*x515, -x514*x91, x512*x516, -x509*x517, -x478*x519, x122*x522, -x129*x524, x525*x526, -x138*x524, x232*x522, -x523*x527, -x517*x529, x516*x531, -x532*x91, x122*x515*x533, -x532*x85, x510*x531, -x507*x529, -x527*x534, x232*x535, -x138*x536, x40*x487*x526, -x129*x536, x122*x535, -x465*x519, x469*x539, x447*x539*x77, x540*x541, x541*x542, -x543*x545, x546*x548, -x117*x451*x549, x548*x550, -x543*x551, -x122*x553, x129*x554, -x555*x556, x138*x554, -x232*x553, -x551*x558, x550*x559, -x127*x549*x560, x546*x559, -x545*x558, -x232*x561, x138*x562, -x43*x525*x556, x129*x562, -x122*x561, x510*x565, x516*x565, x516*x566, x510*x566, -x546*x568, x122*x498*x569, -x550*x568, -x129*x571, x572*x573, -x138*x571, -x550*x574, x16*x569*x575, -x546*x574, -x138*x576, x38*x555*x573, -x129*x576, x546*x578, x550*x578, x550*x579, x546*x579, -4.4183612672736717416e-7*x129*x533, -x580*x91, -4.4183612672736717416e-7*x135*x575, -x580*x85, x132*x88])
        case _:
          assert False, "Order of shape functions not implemented or number of nodes not adequat"
    case 3:
      match n_nodes:
        case 8: 
          def shape_functions(xi):
            x0 = xi[0] - 1.0
            x1 = xi[1] - 1.0
            x2 = 0.125*xi[2] - 0.125
            x3 = x1*x2
            x4 = xi[0] + 1.0
            x5 = xi[1] + 1.0
            x6 = x2*x5
            x7 = 0.125*xi[2] + 0.125
            x8 = x1*x7
            x9 = x5*x7
            return jnp.asarray([-x0*x3, x3*x4, -x4*x6, x0*x6, x0*x8, -x4*x8, x4*x9, -x0*x9])
        case 27:
          def shape_functions(xi):
            x0 = xi[0] - 1.0
            x1 = xi[1] - 1.0
            x2 = xi[0]*xi[1]
            x3 = x1*x2
            x4 = xi[2]*(xi[2] - 1.0)
            x5 = 0.125*x4
            x6 = x3*x5
            x7 = xi[0] + 1.0
            x8 = xi[1] + 1.0
            x9 = x2*x8
            x10 = x5*x9
            x11 = xi[2]*(xi[2] + 1.0)
            x12 = 0.125*x11
            x13 = x12*x3
            x14 = x12*x9
            x15 = xi[0]**2 - 1.0
            x16 = x15*xi[1]
            x17 = x1*x16
            x18 = 0.25*x4
            x19 = xi[1]**2 - 1.0
            x20 = x19*xi[0]
            x21 = x18*x20
            x22 = x16*x8
            x23 = 0.25*x0
            x24 = xi[2]**2 - 1.0
            x25 = x24*x3
            x26 = 0.25*x7
            x27 = x24*x9
            x28 = 0.25*x11
            x29 = x11*x20
            x30 = x15*x19
            x31 = 0.5*x30
            x32 = 0.5*x24
            x33 = x20*x32
            return jnp.asarray([x0*x6, x6*x7, x10*x7, x0*x10, x0*x13, x13*x7, x14*x7, x0*x14, -x17*x18, -x21*x7, -x18*x22, -x0*x21, -x23*x25, -x25*x26, -x26*x27, -x23*x27, -x17*x28, -x26*x29, -x22*x28, -x23*x29, x31*x4, x17*x32, x33*x7, x22*x32, x0*x33, x11*x31, -x24*x30])
        case _:
          assert False, "Order of shape functions not implemented or number of nodes not adequat"
    case _:
      assert False, "Dimensionality not implemented."
  
  if overwrite_diff:
    # Overwrite derivative to be with respect to initial configuration instead of reference configuration
    @jax.custom_jvp
    def ansatz(xi, fI, xI):
      return shape_functions(xi) @ fI
    
    @ansatz.defjvp
    def f_jvp(primals, tangents):
      xi, fI, xI = primals
      x_dot, fI_dot, _ = tangents
      primal_out = ansatz(xi, fI, xI)

      # Isoparametric mapping
      initial_coor = lambda xi: shape_functions(xi) @ xI

      fun = lambda xi: shape_functions(xi) @ fI
      df_dxi = jax.jacfwd(fun)(xi)
      dX_dxi = jax.jacfwd(initial_coor)(xi)
      tangent_out = df_dxi @ jnp.linalg.solve(dX_dxi, x_dot)

      # Add tangent with respect to fI
      if fI_dot is not None:
          tangent_out += shape_functions(xi) @ fI_dot

      return primal_out, tangent_out

    return ansatz(x, fI, xI)
  else:
    return shape_functions(x) @ fI

### Pre-computing shape functions

@jit_with_docstring(static_argnames=['static_settings', 'set', 'num_diff'])
def precompute_shape_functions(dofs, settings, static_settings, set, num_diff):
  """
  Precompute shape functions and their derivatives for all integration points.

  Parameters:
    dofs (jnp.ndarray): The degrees of freedom.
    settings (dict): Dictionary containing various settings for the computation.
        Keywords used:
        - 'connectivity': Connectivity information for the integration points or elements.
        - 'integration coordinates': Coordinates of the integration points.
    static_settings (dict): Dictionary containing static settings for the solution space.
    set (int): The index of the current set of settings being used.
    num_diff (int): The number of derivatives to compute (0, 1, or 2).

  Returns:
    tuple
        A tuple containing the precomputed shape functions and their derivatives.
  """
  neighbor_list = jnp.asarray(static_settings['connectivity'][set])
  local_dofs = dofs[neighbor_list]
  x_int = settings['integration coordinates'][set]
  int_point_numbers = jnp.arange(0, x_int.shape[0], 1)

  # Computing shape functions and derivatives
  shp_i = _shape_funs(x_int, int_point_numbers, local_dofs, settings, static_settings, set)
  if num_diff == 0:
    return (shp_i)

  dshp_i = _shape_funs_dx(x_int, int_point_numbers, local_dofs, settings, static_settings, set)
  if num_diff == 1:
    return (shp_i, dshp_i)

  ddshp_i = _shape_funs_dxx(x_int, int_point_numbers, local_dofs, settings, static_settings, set)
  if num_diff == 2:
    return (shp_i, dshp_i, ddshp_i)

  assert False, 'Number of differentiations not implemented!'

### Helper functions

def _polynomial_basis(x, order):
  """
  Generate a polynomial basis of a given order and dimensionality.

  Parameters:
    x (jnp.ndarray): The input coordinates.
    order (int): The order of the polynomial basis.

  Returns:
    jnp.ndarray: The polynomial basis evaluated at the input coordinates.

  Notes:
    - Supports up to 4 dimensions and polynomial orders up to 10 for 1D and 3 for 2D, 3D and 4D.
  """
  n_dim = x.shape[0]
  match n_dim:
    case 1:
      match order:
        case 0:
          return jnp.asarray([1.])
        case 1:
          return jnp.asarray([1., x])
        case 2:
          return jnp.asarray([1., x, x**2])
        case 3:
          return jnp.asarray([1., x, x**2, x**3])
        case 4:
          return jnp.asarray([1., x, x**2, x**3, x**4])
        case 5:
          return jnp.asarray([1., x, x**2, x**3, x**4, x**5])
        case 6:
          return jnp.asarray([1., x, x**2, x**3, x**4, x**5, x**6])
        case 7:
          return jnp.asarray([1., x, x**2, x**3, x**4, x**5, x**6, x**7])
        case 8:
          return jnp.asarray([1., x, x**2, x**3, x**4, x**5, x**6, x**7, x**8])
        case 9:
          return jnp.asarray([1., x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9])
        case 10:
          return jnp.asarray([1., x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10])
        case _:
          sys.ext("Polynomial basis not implemented for this order!")
    case 2:
      match order:
        case 0:
          return jnp.asarray([1.])
        case 1:
          return jnp.asarray([1., x[0], x[1]])
        case 2:
          return jnp.asarray([1., x[0], x[1], x[0]**2, x[0]*x[1], x[1]**2])
        case 3:
          return jnp.asarray([1., x[0], x[1], x[0]**2, x[0]*x[1], x[1]**2, x[0]**3, x[0]**2*x[1], x[0]*x[1]**2, x[1]**3])
        case _:
          sys.ext("Polynomial basis not implemented for this order!")
    case 3:
      match order:
        case 0:
          return jnp.asarray([1.])
        case 1:
          return jnp.asarray([1., x[0], x[1], x[2]])
        case 2:
          return jnp.asarray([1., x[0], x[1], x[2], x[0]**2, x[1]**2, x[2]**2, x[0]*x[1], x[0]*x[2], x[1]*x[2]])
        case 3:
          return jnp.asarray([1., x[0], x[1], x[2], x[0]**2, x[1]**2, x[2]**2, x[0]*x[1], x[0]*x[2], x[1]*x[2], x[0]**3, x[1]**3, x[2]**3, x[0]**2 * x[1], x[0]**2 * x[2], x[1]**2 * x[0], x[1]**2 * x[2], x[2]**2 * x[0], x[2]**2 * x[1], x[0]*x[1]*x[2]])
        case _:
          sys.ext("Polynomial basis not implemented for this order!")
    case 4:
      match order:
        case 0:
          return jnp.asarray([1.])
        case 1:
          return jnp.asarray([1., x[0], x[1], x[2], x[3]])
        case 2:
          return jnp.asarray([1., x[0], x[1], x[2], x[3], x[0]**2, x[1]**2, x[2]**2, x[3]**2, x[0]*x[1], x[0]*x[2], x[0]*x[3], x[1]*x[2], x[1]*x[3], x[2]*x[3]])
        case 3:
          return jnp.asarray([1., x[0], x[1], x[2], x[3], x[0]**2, x[1]**2, x[2]**2, x[3]**2, x[0]*x[1], x[0]*x[2], x[0]*x[3], x[1]*x[2], x[1]*x[3], x[2]*x[3], x[0]**3, x[1]**3, x[2]**3, x[3]**3, x[0]**2 * x[1], x[0]**2 * x[2], x[0]**2 * x[3], x[1]**2 * x[0], x[1]**2 * x[2], x[1]**2 * x[3], x[2]**2 * x[0], x[2]**2 * x[1], x[2]**2 * x[3], x[3]**2 * x[0], x[3]**2 * x[1], x[3]**2 * x[1], x[0]*x[1]*x[2], x[0]*x[1]*x[3], x[0]*x[2]*x[3], x[1]*x[2]*x[3]])
        case _:
          sys.exit("Polynomial basis not implemented for this order!")
    case _:
      sys.exit("Polynomial basis not implemented for this dimensionality!")

def _compute_poly_basis_length(n_dim, order):
  """
  Compute the number of coefficients for a polynomial basis given its dimensionality and order.

  Parameters:
    n_dim (int): The number of dimensions of the polynomial basis.
    order (int): The order of the polynomial basis.

  Returns:
    int: The number of coefficients in the polynomial basis.
  """
  match n_dim:
    case 1:
      return order + 1
    case 2:
      return np.sum([i+1 for i in range(order+1)])
    case 3:
      return np.sum([np.sum([j+1 for j in range(i+1)]) for i in range(order+1)])
    case 4:
      return np.sum([np.sum([np.sum([k+1 for k in range(j+1)]) for j in range(i+1)]) for i in range(order+1)])
    case _:
      sys.exit("Polynomial basis not implemented for this dimensionality!")

@jit_with_docstring(static_argnames=['static_settings', 'set'])
def _shape_fun(x_i, i, local_dofs, settings, static_settings, set):
  return solution_space(x_i, i, local_dofs, settings, static_settings, set)
_shape_funs = jax.jit(jax.vmap(_shape_fun, (0,0,0,None,None,None), 0), static_argnames=['static_settings', 'set'])

@jit_with_docstring(static_argnames=['static_settings', 'set'])
def _shape_fun_dx(x_i, i, local_dofs, settings, static_settings, set):
  return jax.jacfwd(solution_space)(x_i, i, local_dofs, settings, static_settings, set)
_shape_funs_dx = jax.jit(jax.vmap(_shape_fun_dx, (0,0,0,None,None,None), 0), static_argnames=['static_settings', 'set'])

@jit_with_docstring(static_argnames=['static_settings', 'set'])
def _shape_fun_dxx(x_i, i, local_dofs, settings, static_settings, set):
  return jax.jacfwd(jax.jacfwd(solution_space))(x_i, i, local_dofs, settings, static_settings, set)
_shape_funs_dxx = jax.jit(jax.vmap(_shape_fun_dxx, (0,0,0,None,None,None), 0), static_argnames=['static_settings', 'set'])

