# models.py
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
This module contains premade jax-transformable partial differential equation (PDE) formulations, weak forms, user-defined elements, and time integration procedures.

The models-functions return callables with appropriate arguments for the use required in the other modules of AutoPDEx. These functions can then be passed e.g. to the solver via static_settings['model'].

For instance, a typical weak form function has the following arguments:
  - x (jnp.ndarray): Spatial coordinates at the integration point.
  - ansatz (function): Ansatz function representing the field variable (e.g., displacement or temperature).
  - test_ansatz (function): Test ansatz function representing the virtual displacement or virtual temperature.
  - settings (dict): Settings for the computation.
  - static_settings (dict): Static settings for the computation.
  - int_point_number (int): Integration point number.
  - set: Number of domain.

Note: Some of the models work for degrees of freedoms (dofs) as a jnp.ndarray
and some work for dicts of jnp.ndarrays. It is specified in each docstring.
"""

from inspect import signature

import jax
import jax.numpy as jnp

from autopdex import solution_structures, utility


### Linear equations
def transport_equation(c):
  """
    Transport equation: du/dt + c * du/dx = 0

    Args:
      c (float): Transport coefficient.

    Returns:
      pde_fun (function): Function to compute the PDE residual.

    Notes:
      - This models works for DOFs as a jnp.ndarray.
    """

  def pde_fun(x, ansatz, settings, static_settings, int_point_number, set):
    u = ansatz
    u_derivative = jax.jacrev(u)(x)
    u_x = u_derivative[0]
    u_t = u_derivative[1]
    return u_t + c * u_x

  return pde_fun


def poisson(coefficient_fun=lambda x, settings: 1.0, source_fun=None):
  """
    Poisson equation in n dimensions: coefficient * Laplace(Theta) + source_term = 0

    Args:
      coefficient_fun (function): Function to compute the coefficient
      source_fun (function (dependend on the position jnp.ndarray x), optional): Function to compute the source term, defaults to None.

    Returns:
      pde_fun (function): Function to compute the PDE residual.

    Notes:
      - This models works for DOFs as a jnp.ndarray.
    """

  def pde_fun(x, ansatz, settings, static_settings, int_point_number, set):
    hessian = jax.jacfwd(jax.jacrev(ansatz))(x)

    # Coefficient
    coefficient = coefficient_fun(x, settings)

    if source_fun == None:
      pde = coefficient * jnp.trace(hessian)
    else:
      pde = coefficient * jnp.trace(hessian) + source_fun(x)

    return pde

  return pde_fun


def poisson_weak(coefficient_fun=lambda x, settings: 1.0, source_fun=None):
  """
    Poisson equation in n dimensions as weak form.

    Args:
      coefficient_fun (function): Function to compute the coefficient of the laplacian, defaults to 1.
      source_fun (function (dependend on the position jnp.ndarray x), optional): Function to compute the source term, defaults to None.

    Returns:
      pde_fun (function): Function to compute the weak form of the PDE.

    Notes:
      - This models works for DOFs as a jnp.ndarray.
    """

  def pde_fun(
      x, ansatz, test_ansatz, settings, static_settings, int_point_number, set
  ):
    # Virtual temperature change and gradient
    var_theta = test_ansatz(x)
    var_theta_grad = jax.jacrev(test_ansatz)(x)

    # Temperature gradient
    theta_grad = jax.jacrev(ansatz)(x)

    # Coefficient
    coefficient = coefficient_fun(x, settings)

    if source_fun == None:
      pde = -coefficient * jnp.dot(theta_grad, var_theta_grad)
    else:
      pde = (
          -coefficient * jnp.dot(theta_grad, var_theta_grad)
          + source_fun(x) * var_theta
      )

    return pde

  return pde_fun


def poisson_fos(spacing, coefficient_fun=lambda x, settings: 1.0, source_fun=None):
  """
    Poisson equation as set of first order models augmented by curl(v)=0

    First field of solution space has to be scalar function; remaining fields form the gradient, i.e.
    theta = lambda x: ansatz(x)[0]
    grad_theta = lambda x: ansatz(x)[1:]
    n_dim = 2 or 3

    Args:
      spacing (float): Spacing for the weights.
      coefficient_fun (function): Function to compute the coefficient of the laplacian, defaults to 1.
      source_fun (function, optional): Function to compute the source term, defaults to None.

    Returns:
      pde_fun (function): Function to compute the first order PDE system residuals.

    Notes:
      - This models works for DOFs as a jnp.ndarray.
    """

  def pde_fun(x, ansatz, settings, static_settings, int_point_number, set):
    n_dim = x.shape[0]

    # Coefficient
    coefficient = coefficient_fun(x, settings)

    # Primary fields
    theta = lambda t: ansatz(t)[0]
    grad_theta = lambda t: ansatz(t)[1:]  # v
    grad_v = jax.jacfwd(grad_theta)(x)

    # Poisson equation
    if source_fun == None:
      pde_1 = coefficient * jnp.asarray([jnp.trace(grad_v)])
    else:
      pde_1 = coefficient * jnp.asarray([jnp.trace(grad_v) + source_fun(x)])

    # Substituted derivatives
    pde_2 = grad_theta(x) - jax.jacrev(theta)(x)

    # Additional constraint in order to make functional fully H1 coercive
    if n_dim == 2:
      curl_v = jnp.asarray([grad_v[1, 0] - grad_v[0, 1]])
    elif n_dim == 3:
      curl_v = jnp.asarray(
          [
              grad_v[2, 1] - grad_v[1, 2],
              grad_v[0, 2] - grad_v[2, 0],
              grad_v[1, 0] - grad_v[0, 1],
          ]
      )
    else:
      assert False, "This dimensionality is not supported by laplacian_fos"
    pde_3 = curl_v

    w1 = 1
    w2 = 1 / spacing
    w3 = 1 / spacing
    return jnp.concatenate([w1 * pde_1, w2 * pde_2, w3 * pde_3]).flatten()

  return pde_fun


def heat_equation(diffusivity_fun):
  """
    Space-time heat equation with thermal diffusivity alpha.

    Last dimension is the time dimension, remaining dimensions are spatial dimensions

    Args:
      diffusivity_fun (function): Function to compute the thermal diffusivity.

    Returns:
      heat_fun (function): Function to compute the PDE residual.

    Notes:
      - This models works for DOFs as a jnp.ndarray.
    """

  def heat_fun(x, ansatz, settings, static_settings, int_point_number, set):
    n_dim = x.shape[0]
    alpha = diffusivity_fun(x)

    rate = jax.jacfwd(ansatz)(x)[-1]  # du / dt
    spatial_derivatives = jax.jacfwd(jax.jacrev(ansatz))(x).diagonal()[: n_dim - 1]
    laplacian = spatial_derivatives.sum()  # Laplace(u)

    pde = rate - alpha * laplacian
    return pde

  return heat_fun


def heat_equation_fos(diffusivity_fun):
  """Space-time heat equation with thermal diffusivity alpha as a first order system with curl-augmentation for more than 2 spatial dimensions

    - supports n_dim = 2 to n_dim = 3, i.e. 1 to 3 spatial dimensions plus one time dimension
    - number of fiels: n_dim (1 temperature field, n_dim-1 spatial gradients)
    - last dimension is the time dimension, the remaining ones are spacial dimensions

    e.g. n_dim = 2:
    - first dimension is x-dimension, second dimension is time dimension
    - first field is temperature, second field is derivative of temperature with respect to x

    Args:
      diffusivity_fun (function): Function to compute the thermal diffusivity.

    Returns:
      heat_fun (function): Function to compute the PDE residual.

    Notes:
      - This models works for DOFs as a jnp.ndarray.
    """

  def heat_fun(x, ansatz, settings, static_settings, int_point_number, set):
    n_dim = x.shape[0]
    alpha = diffusivity_fun(x)

    temperature = lambda t: ansatz(t)[0]  # u
    derivative_field = lambda t: ansatz(t)[1:]  # v
    derivatives = jax.jacrev(temperature)(x)  # [du/dx, du/dt] (flat)

    # Heat equation with substituted spatial derivatives v
    rate = derivatives[-1]  # du / dt
    grad_v = jax.jacfwd(derivative_field)(x)[:, : n_dim - 1]
    laplacian = grad_v.diagonal().sum()  # div(v)
    pde_1 = jnp.asarray([rate - alpha * laplacian])

    # Substituted spatial derivatives: v - du / dx
    pde_2 = derivative_field(x) - derivatives[: n_dim - 1]

    # Augmentation for stabilization
    if n_dim == 2:
      curl_v = jnp.asarray([0.0])
    elif n_dim == 3:
      curl_v = jnp.asarray([grad_v[1, 0] - grad_v[0, 1]])
    elif n_dim == 4:
      curl_v = jnp.asarray(
          [
              grad_v[2, 1] - grad_v[1, 2],
              grad_v[0, 2] - grad_v[2, 0],
              grad_v[1, 0] - grad_v[0, 1],
          ]
      )
    else:
      assert False, "This dimensionality is not supported by heat_equation_fos"
    pde_3 = curl_v

    return jnp.concatenate([pde_1, pde_2, pde_3]).flatten()

  return heat_fun


def d_alembert(wave_number_fun):
  """
    Space-time d'Alembert operator 1 temporal [-1] and (n-1) spatial [:n-1] dimension with wave number c.

    Args:
      wave_number_fun (function): Function to compute the wave number.

    Returns:
      d_alembert_fun (function): Function to compute the PDE residual.

    Notes:
      - This models works for DOFs as a jnp.ndarray.
    """

  def d_alembert_fun(x, ansatz, settings, static_settings, int_point_number, set):
    c = wave_number_fun(x)

    hessian = jax.jacfwd(jax.jacrev(ansatz))(x)
    hessian_diagonal = jnp.diagonal()
    coefficients = -jnp.ones_like(hessian_diagonal)
    coefficients = coefficients.at[-1].set(1 / c**2)

    pde = jnp.dot(hessian_diagonal, coefficients)
    return pde

  return d_alembert_fun


def d_alembert_fos(wave_number_fun, spacing):
  """
    Constructs a first-order system for the space-time d'Alembert operator with wave speed c.

    This function returns a PDE function representing the d'Alembert operator for one temporal dimension
    and (n-1) spatial dimensions, with augmented conditions to ensure a fully H1 coercive functional.

    Args:
      wave_number_fun (function): A function that computes the wave number (inverse of wave speed) given spatial coordinates.
      spacing (float): The spatial spacing parameter used to scale the augmented conditions.

    Returns:
      function: A function that evaluates the first-order system of PDEs for the d'Alembert operator, which includes:
        - The primary d'Alembert operator.
        - Substituted derivative fields.
        - Additional constraints for stabilization (curl conditions).

    Notes:
    - The last dimension of the input coordinates x is considered the time dimension; the remaining dimensions are spatial dimensions.
    - This models works for DOFs as a jnp.ndarray.
    """

  def d_alembert_fun(x, ansatz, settings, static_settings, int_point_number, set):
    # Initialization
    n_dim = x.shape[0]
    c = wave_number_fun(x)
    field = lambda t: ansatz(t)[0]
    derivative_field = lambda t: ansatz(t)[1:]

    # First pde: D' Alembert operator
    hessian = jax.jacfwd(derivative_field)(x)
    hessian_diagonal = hessian.diagonal()
    coefficients = -jnp.ones_like(hessian_diagonal)
    coefficients = coefficients.at[0].set(
        1 / c
    )  # The other c is already in the other pde
    pde_1 = jnp.asarray([jnp.dot(hessian_diagonal, coefficients)])

    # Second pde: Gradient field
    scaled_grad = jax.jacrev(field)(x)
    scaled_grad = scaled_grad.at[-1].multiply(1 / c)
    pde_2 = derivative_field(x) - scaled_grad

    # Additional constraint in order to make functional fully H1 coercive
    # grad_v = derivative_field(x)
    grad_v = jax.jacfwd(derivative_field)(x)
    if n_dim == 2:
      curl_v = jnp.asarray([0.0])
    elif n_dim == 3:
      curl_v = jnp.asarray([grad_v[1, 0] - grad_v[0, 1]])
    elif n_dim == 4:
      curl_v = jnp.asarray(
          [
              grad_v[2, 1] - grad_v[1, 2],
              grad_v[0, 2] - grad_v[2, 0],
              grad_v[1, 0] - grad_v[0, 1],
          ]
      )
    else:
      assert False, "This dimensionality is not supported by laplacian_fos"
    pde_3 = curl_v

    w1 = 1
    w2 = 1 / spacing
    w3 = 1 / spacing
    return jnp.concatenate([w1 * pde_1, w2 * pde_2, w3 * pde_3]).flatten()

  return d_alembert_fun


def linear_elasticity(youngs_mod_fun, poisson_ratio_fun, mode, volume_load_fun=None):
  """
    Constructs the strong form of linear elasticity in Voigt notation, displacement based.

    This function supports 2D plain strain, 2D plain stress, and 3D elasticity problems.

    Args:
    youngs_mod_fun (function): Function to compute Young's modulus given spatial coordinates (and optional settings).
    poisson_ratio_fun (function): Function to compute Poisson's ratio given spatial coordinates (and optional settings).
    mode (str): Specifies the mode of the elasticity problem. It can be 'plain strain', 'plain stress', or '3d'.
    volume_load_fun (function, optional): Function to compute volume load given spatial coordinates. Default is None.

    Returns:
    function: A function that evaluates the strong form PDE for linear elasticity.

    Notes:
      - This models works for DOFs as a jnp.ndarray.
    """
  if mode == "plain strain" or mode == "plain stress":
    n_dim = 2
  elif mode == "3d":
    n_dim = 3
  else:
    assert False, "'mode' for linear elasticity not properly set."

  def lin_el_fun(x, ansatz, settings, static_settings, int_point_number, set):
    def get_stress(xf):
      # Kinematics
      displacement = ansatz
      displ_grad = jax.jacfwd(ansatz)(xf)
      strain = (1 / 2) * (displ_grad + displ_grad.transpose())
      if n_dim == 2:
        strain_voigt = jnp.asarray(
            [strain[0, 0], strain[1, 1], 2 * strain[0, 1]]
        )
      else:
        strain_voigt = jnp.asarray(
            [
                strain[0, 0],
                strain[1, 1],
                strain[2, 2],
                2 * strain[0, 1],
                2 * strain[0, 2],
                2 * strain[1, 2],
            ]
        )

      # Constiutive model
      if len(signature(poisson_ratio_fun).parameters) == 1:
        nu = poisson_ratio_fun(x)
      else:
        nu = poisson_ratio_fun(x, settings)
      if len(signature(youngs_mod_fun).parameters) == 1:
        Em = youngs_mod_fun(x)
      else:
        Em = youngs_mod_fun(x, settings)

      match mode:
        case "plain strain":
          mu = Em / (2 * (1 + nu))
          c1 = 1 - 2 * nu
          c2 = 1 - nu
          coeff = 2 * mu / c1
          material_tangent = coeff * jnp.asarray(
              [[c2, nu, 0], [nu, c2, 0], [0, 0, c1]]
          )

        case "plain stress":
          coeff = Em / (1 - nu**2)
          material_tangent = coeff * jnp.asarray(
              [[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]]
          )

        case "3d":
          coeff = Em / (1 + nu)
          c1 = 1 - 2 * nu
          c2 = (1 - nu) / c1
          c3 = nu / c1
          c4 = 1 / 2
          material_tangent = coeff * jnp.asarray(
              [
                  [c2, c3, c3, 0, 0, 0],
                  [c3, c2, c3, 0, 0, 0],
                  [c3, c3, c2, 0, 0, 0],
                  [0, 0, 0, c4, 0, 0],
                  [0, 0, 0, 0, c4, 0],
                  [0, 0, 0, 0, 0, c4],
              ]
          )

      return material_tangent @ strain_voigt

    # stress_voigt = get_stress(x)

    # Internal forces
    grad_stress = jax.jacfwd(get_stress)(x)
    if n_dim == 2:
      div_stress = jnp.asarray(
          [
              grad_stress[0, 0] + grad_stress[2, 1],
              grad_stress[2, 0] + grad_stress[1, 1],
          ]
      )
    else:
      div_stress = jnp.asarray(
          [
              grad_stress[0, 0] + grad_stress[3, 1] + grad_stress[4, 2],
              grad_stress[3, 0] + grad_stress[1, 1] + grad_stress[5, 2],
              grad_stress[4, 0] + grad_stress[5, 1] + grad_stress[2, 2],
          ]
      )

    # Balance equation
    if volume_load_fun == None:
      return div_stress
    else:
      return div_stress + volume_load_fun(x)

  return lin_el_fun


def linear_elasticity_weak(
    youngs_mod_fun, poisson_ratio_fun, mode, volume_load_fun=None
):
  """
    Constructs the weak form of linear elasticity in Voigt notation, displacement based.

    This function supports 2D plain strain, 2D plain stress, and 3D elasticity problems.

    Args:
    youngs_mod_fun (function): Function to compute Young's modulus given spatial coordinates (and optional settings).
    poisson_ratio_fun (function): Function to compute Poisson's ratio given spatial coordinates (and optional settings).
    mode (str): Specifies the mode of the elasticity problem. It can be 'plain strain', 'plain stress', or '3d'.
    volume_load_fun (function, optional): Function to compute volume load given spatial coordinates. Default is None.

    Returns:
    function: A function that evaluates the weak form PDE for linear elasticity.

    Notes:
      - This models works for DOFs as a jnp.ndarray.
    """
  if mode == "plain strain" or mode == "plain stress":
    n_dim = 2
  elif mode == "3d":
    n_dim = 3
  else:
    assert False, "'mode' for linear elasticity not properly set."

  def lin_el_fun(
      x, ansatz, test_ansatz, settings, static_settings, int_point_number, set
  ):
    def get_stress(xf):
      # Kinematics
      displacement = ansatz
      displ_grad = jax.jacfwd(ansatz)(xf)
      strain = (1 / 2) * (displ_grad + displ_grad.transpose())
      if n_dim == 2:
        strain_voigt = jnp.asarray(
            [strain[0, 0], strain[1, 1], 2 * strain[0, 1]]
        )
      else:
        strain_voigt = jnp.asarray(
            [
                strain[0, 0],
                strain[1, 1],
                strain[2, 2],
                2 * strain[0, 1],
                2 * strain[0, 2],
                2 * strain[1, 2],
            ]
        )

      # Constiutive model
      if len(signature(poisson_ratio_fun).parameters) == 1:
        nu = poisson_ratio_fun(x)
      else:
        nu = poisson_ratio_fun(x, settings)
      if len(signature(youngs_mod_fun).parameters) == 1:
        Em = youngs_mod_fun(x)
      else:
        Em = youngs_mod_fun(x, settings)
      match mode:
        case "plain strain":
          mu = Em / (2 * (1 + nu))
          c1 = 1 - 2 * nu
          c2 = 1 - nu
          coeff = 2 * mu / c1
          material_tangent = coeff * jnp.asarray(
              [[c2, nu, 0], [nu, c2, 0], [0, 0, c1]]
          )

        case "plain stress":
          coeff = Em / (1 - nu**2)
          material_tangent = coeff * jnp.asarray(
              [[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]]
          )

        case "3d":
          coeff = Em / (1 + nu)
          c1 = 1 - 2 * nu
          c2 = (1 - nu) / c1
          c3 = nu / c1
          c4 = 1 / 2
          material_tangent = coeff * jnp.asarray(
              [
                  [c2, c3, c3, 0, 0, 0],
                  [c3, c2, c3, 0, 0, 0],
                  [c3, c3, c2, 0, 0, 0],
                  [0, 0, 0, c4, 0, 0],
                  [0, 0, 0, 0, c4, 0],
                  [0, 0, 0, 0, 0, c4],
              ]
          )

      return material_tangent @ strain_voigt

    stress_voigt = get_stress(x)

    # Test function
    virtual_disp = test_ansatz(x)
    virtual_disp_grad = jax.jacfwd(test_ansatz)(x)
    virtual_strain = (1 / 2) * (virtual_disp_grad + virtual_disp_grad.transpose())
    if n_dim == 2:
      virtual_strain_voigt = jnp.asarray(
          [virtual_strain[0, 0], virtual_strain[1, 1], 2 * virtual_strain[0, 1]]
      )
    else:
      virtual_strain_voigt = jnp.asarray(
          [
              virtual_strain[0, 0],
              virtual_strain[1, 1],
              virtual_strain[2, 2],
              2 * virtual_strain[0, 1],
              2 * virtual_strain[0, 2],
              2 * virtual_strain[1, 2],
          ]
      )

    # Balance equation
    if volume_load_fun == None:
      return jnp.dot(stress_voigt, virtual_strain_voigt)
    else:
      return jnp.dot(stress_voigt, virtual_strain_voigt) - jnp.dot(
          volume_load_fun(x), virtual_disp
      )

  return lin_el_fun


def linear_elasticity_fos(
    youngs_mod_fun, poisson_ratio_fun, mode, spacing, volume_load_fun=None
):
  """
    Constructs the first order system (FOS) of linear elasticity in Voigt notation, displacement based.

    This function supports 2D plain strain and 3D elasticity problems. The formulation is based on the first order
    system described in https://doi.org/10.1137/S0036142902418357 with additional asymmetry penalization.

    Args:
      youngs_mod_fun (function): Function to compute Young's modulus given spatial coordinates (and optional settings).
      poisson_ratio_fun (function): Function to compute Poisson's ratio given spatial coordinates (and optional settings).
      mode (str): Specifies the mode of the elasticity problem. It can be 'plain strain' or '3d'.
      spacing (float): The spatial spacing parameter used to scale the augmented conditions.
      volume_load_fun (function, optional): Function to compute volume load given spatial coordinates. Default is None.

    Returns:
      function: A function that evaluates the first order system of PDEs for linear elasticity.

    The returned function computes the following:
      - The kinematic strain based on the displacement gradient.
      - The constitutive relation using the specified Young's modulus and Poisson's ratio.
      - The internal force balance equation, optionally including volume loads.
      - Weak enforcement of symmetry of the Cauchy stress tensor.

    Notes:
      - In 2D, the number of fields is 2 (displacement) + 4 (stress) = 6.
      - In 3D, the number of fields is 3 (displacement) + 9 (stress) = 12.
      - This function only supports 'plain strain' and '3d' modes.
      - This models works for DOFs as a jnp.ndarray.
    """
  if mode == "plain strain":
    n_dim = 2
  elif mode == "3d":
    n_dim = 3
  else:
    assert False, "'mode' for linear elasticity not properly set."

  def lin_el_fun(x, ansatz, settings, static_settings, int_point_number, set):
    # Primary fields
    displacement = lambda t: ansatz(t)[:n_dim]
    stress = lambda t: ansatz(t)[n_dim:].reshape((n_dim, n_dim))

    # Kinematics
    displ_grad = jax.jacfwd(displacement)(x)
    strain = (1 / 2) * (displ_grad + displ_grad.transpose())

    # Constiutive model
    if len(signature(poisson_ratio_fun).parameters) == 1:
      nu = poisson_ratio_fun(x)
    else:
      nu = poisson_ratio_fun(x, settings)
    if len(signature(youngs_mod_fun).parameters) == 1:
      Em = youngs_mod_fun(x)
    else:
      Em = youngs_mod_fun(x, settings)
    mu = Em / (2 * (1 + nu))  # Shear modulus
    lam = Em * nu / ((1 + nu) * (1 - 2 * nu))  # First Lame constant

    # Complience tensor times stress
    sigma = stress(x)
    A_sigma = (1 / (2 * mu)) * (
        sigma
        - (lam / (n_dim * lam + 2 * mu)) * jnp.trace(sigma) * jnp.identity(n_dim)
    )

    # Second pde: constitutive model
    pde_2 = (A_sigma - strain).flatten()

    # Internal forces
    grad_stress = jax.jacfwd(stress)(x)
    if n_dim == 2:
      div_stress = jnp.asarray(
          [
              grad_stress[0, 0, 0] + grad_stress[0, 1, 1],
              grad_stress[1, 0, 0] + grad_stress[1, 1, 1],
          ]
      )
    else:
      div_stress = jnp.asarray(
          [
              grad_stress[0, 0, 0] + grad_stress[0, 1, 1] + grad_stress[0, 2, 2],
              grad_stress[1, 0, 0] + grad_stress[1, 1, 1] + grad_stress[1, 2, 2],
              grad_stress[2, 0, 0] + grad_stress[2, 1, 1] + grad_stress[2, 2, 2],
          ]
      )

    # First pde: Balance equation
    if volume_load_fun == None:
      pde_1 = div_stress
    else:
      pde_1 = div_stress + volume_load_fun(x)

    # Weak enforcement of symmetry of Cauchy stress
    pde_3 = ((1 / 2) * (stress(x) - stress(x).transpose())).flatten()

    # Return first order system
    w_1 = 1.0
    w_2 = 1.0
    w_3 = 10.0 / mu
    return jnp.concatenate([w_1 * pde_1, w_2 * pde_2, w_3 * pde_3]).flatten()

  # Return function for evaluating first order system with arguments x and ansatz
  return lin_el_fun


def neumann_weak(neumann_fun):
  """
    Constructs the weak form of Neumann boundary conditions for the virtual work of surface tractions or heat inflow.

    This function can be used to impose Neumann boundary conditions in the weak form of problems such as linear
    momentum balance or heat conduction.

    Args:
      neumann_fun (callable): A function that computes the Neumann boundary condition (e.g., surface tractions or heat inflow)
          given spatial coordinates. This function can optionally take the settings as an additional argument.

    Returns:
      function
          A function that evaluates the weak form of the Neumann boundary condition.

    The returned function computes the virtual work of the Neumann boundary condition, which is the dot product of
    the virtual displacement (or temperature) and the Neumann boundary condition (e.g., surface tractions or heat inflow).

    Notes:
      - This models works for DOFs as a jnp.ndarray.
    """

  def fun(x, ansatz, test_ansatz, settings, static_settings, int_point_number, set):
    # Test function
    virtual_disp = test_ansatz(x)

    # Tractions
    if len(signature(neumann_fun).parameters) == 1:
      tractions = neumann_fun(x)
    else:
      tractions = neumann_fun(x, settings)

    # Virtual work
    return -jnp.dot(virtual_disp, tractions)

  return fun


### Nonlinear equations
def burgers_equation_inviscid():
  """
    Constructs the inviscid Burgers' equation: du/dt + u * du/dx = 0.

    Args:
      None

    Returns:
      function: A function that evaluates the inviscid Burgers' equation.

    Notes:
      - first dimension: x
      - second dimension: t
      - This models works for DOFs as a jnp.ndarray.
    """

  def pde_fun(x, ansatz, settings, static_settings, int_point_number, set):
    u = ansatz(x)
    u_x = jax.jacrev(ansatz)(x)[0]
    u_t = jax.jacrev(ansatz)(x)[1]
    return u_t + u * u_x

  return pde_fun


def hyperelastic_steady_state_fos(
    strain_energy_fun,
    youngs_mod_fun,
    poisson_ratio_fun,
    mode,
    spacing,
    volume_load_fun=None,
):
  """
    Constructs a first-order PDE system for given strain energy function

    This function is based on equation (28) in https://doi.org/10.1002/nme.5951 and returns a function
    representing the first-order PDE system for steady-state hyperelasticity.

    Args:
      strain_energy_fun (function): Function that returns the strain energy given the deformation gradient and Lame parameters.
      youngs_mod_fun (function): Function to compute Young's modulus given spatial coordinates.
      poisson_ratio_fun (function): Function to compute Poisson's ratio given spatial coordinates.
      mode (str): Specifies the mode of the elasticity problem. It can be 'plain strain' or '3d'.
      spacing (float): The spatial spacing parameter used to scale the augmented conditions.
      volume_load_fun (function, optional): Function to compute volume load given spatial coordinates. Default is None.

    Returns:
      function: A function that evaluates the first-order PDE system for hyperelasticity.

    Notes:
      - The volume load function, if provided, must be volume-specific.
      - This models works for DOFs as a jnp.ndarray.
    """

  def pde_fun(x, ansatz, settings, static_settings, int_point_number, set):
    # Initialization
    n_dim = x.shape[0]
    if len(signature(poisson_ratio_fun).parameters) == 1:
      nu = poisson_ratio_fun(x)
    else:
      nu = poisson_ratio_fun(x, settings)
    if len(signature(youngs_mod_fun).parameters) == 1:
      Em = youngs_mod_fun(x)
    else:
      Em = youngs_mod_fun(x, settings)

    # Lame parameters
    lam = Em * nu / ((1 + nu) * (1 - 2 * nu))
    mu = Em / (2 * (1 + nu))

    # Ansatz
    u = lambda t: ansatz(t)[:n_dim]  # Displacement
    P = lambda t: ansatz(t)[n_dim:].reshape(
        (n_dim, n_dim)
    )  # 1. Piola-Kirchoff stress

    # Linear momentum balance
    Grad_P = jax.jacfwd(P)(x)
    Div_P = jnp.trace(Grad_P, axis1=1, axis2=2)

    # First pde: Balance equation
    if volume_load_fun == None:
      pde_1 = Div_P
    else:
      b = volume_load_fun(x)
      pde_1 = Div_P + b

    if mode == "plain strain":
      # Deformation gradient
      H = jax.jacfwd(u)(x)
      F = jnp.asarray(
          [
              jnp.asarray([H[0, 0], H[0, 1], 0.0]),
              jnp.asarray([H[1, 0], H[1, 1], 0.0]),
              jnp.asarray([0.0, 0.0, 0.0]),
          ]
      ) + jnp.identity(3)

      # First Piola-Kirchhoff stress
      dpsi_dF = jax.jacrev(neo_hooke)(F, (lam, mu))
      dpsi_dF_FT = (dpsi_dF @ jnp.transpose(F))[:n_dim, :n_dim]
      PFT = P(x)[:n_dim, :n_dim] @ jnp.transpose(F[:n_dim, :n_dim])

    elif mode == "3d":
      # Deformation gradient
      F = jax.jacfwd(u)(x) + jnp.identity(3)

      # First Piola-Kirchhoff stress
      dpsi_dF = jax.jacrev(strain_energy_fun)(F, (lam, mu))
      dpsi_dF_FT = dpsi_dF @ jnp.transpose(F)
      PFT = P(x) @ F.transpose()

    else:
      assert (
          False
      ), "Hyperelastic model supports only 'plain strain' and '3d' modes."

    # Constitutive law
    pde_2 = ((1 / 2) * (PFT + jnp.transpose(PFT)) - dpsi_dF_FT).flatten()

    # Enforce symmetry of P \cdot F^T
    PFT_asym = (1 / 2) * (PFT - PFT.transpose())
    pde_3 = PFT_asym.flatten()

    # Weightings
    w_1 = 1.0
    w_2 = 1.0 / mu
    w_3 = 10.0 / mu
    return jnp.concatenate([w_1 * pde_1, w_2 * pde_2, w_3 * pde_3])

  return pde_fun


def hyperelastic_steady_state_weak(
    strain_energy_fun, youngs_mod_fun, poisson_ratio_fun, mode, volume_load_fun=None
):
  """
    Constructs the weak form of hyperelasticity for given strain energy function

    This function returns a function representing the weak form PDE system for steady-state hyperelasticity.

    Args:
      strain_energy_fun (function): Function that returns the strain energy given the deformation gradient and Lame parameters.
      youngs_mod_fun (function): Function to compute Young's modulus given spatial coordinates.
      poisson_ratio_fun (function): Function to compute Poisson's ratio given spatial coordinates.
      mode (str): Specifies the mode of the elasticity problem. It can be 'plain strain' or '3d'.
      volume_load_fun (function, optional): Function to compute volume load given spatial coordinates. Default is None.

    Returns:
      function: A function that evaluates the weak form PDE system for hyperelasticity.

    Notes:
      - The volume load function, if provided, must be volume-specific.
      - This models works for DOFs as a jnp.ndarray.
    """

  def pde_fun(
      x, ansatz, test_ansatz, settings, static_settings, int_point_number, set
  ):
    # Initialization
    n_dim = x.shape[0]
    if len(signature(poisson_ratio_fun).parameters) == 1:
      nu = poisson_ratio_fun(x)
    else:
      nu = poisson_ratio_fun(x, settings)
    if len(signature(youngs_mod_fun).parameters) == 1:
      Em = youngs_mod_fun(x)
    else:
      Em = youngs_mod_fun(x, settings)

    # Lame parameters
    lam = Em * nu / ((1 + nu) * (1 - 2 * nu))
    mu = Em / (2 * (1 + nu))

    # (Virtual) displacement
    u = ansatz
    virt_u = test_ansatz

    # Deformation gradient
    if mode == "plain strain":
      H = jax.jacfwd(u)(x)
      F = jnp.asarray(
          [[H[0, 0], H[0, 1], 0.0], [H[1, 0], H[1, 1], 0.0], [0.0, 0.0, 0.0]]
      ) + jnp.identity(3)

      virt_H = jax.jacfwd(virt_u)(x)
      virt_F = jnp.asarray(
          [
              [virt_H[0, 0], virt_H[0, 1], 0.0],
              [virt_H[1, 0], virt_H[1, 1], 0.0],
              [0.0, 0.0, 0.0],
          ]
      )
    elif mode == "3d":
      F = jax.jacfwd(u)(x) + jnp.identity(3)
      virt_F = jax.jacfwd(virt_u)(x)
    else:
      assert (
          False
      ), "Hyperelastic model supports only 'plain strain' and '3d' modes."

    # First Piola-Kirchhoff stress
    P = jax.jacrev(strain_energy_fun)(F, (lam, mu))

    # Virtual work of internal forces
    PI_int = jnp.einsum("ij, ij -> ", P, virt_F)
    # PI_int = jnp.trace(P @ virt_F.transpose())

    if volume_load_fun == None:
      return PI_int
    else:
      # Add virtual work of volume forces
      b = volume_load_fun(x)
      PI_ext = jnp.dot(b, virt_u)
      return PI_int - PI_ext

  return pde_fun


def navier_stokes_incompressible_steady(
    dynamic_viscosity, density=1.0, incompressible_weighting=1.0, volume_load_fun=None
):
  """
    Constructs the steady-state incompressible Navier-Stokes equations.

    Args:
      dynamic_viscosity (float): Dynamic viscosity of the fluid.
      density (float, optional): Density of the fluid. Default is 1.0.
      incompressible_weighting (float, optional): Weighting factor for the incompressibility constraint. Default is 1.0.
      volume_load_fun (function, optional): Function to compute volume load given spatial coordinates. Default is None.

    Returns:
      function: A function that evaluates the steady-state incompressible Navier-Stokes equations.

    Notes:
      - The first equation is the constraint of a divergence-free flow field.
      - The remaining equations are the linear momentum balance.
      - The first field in the ansatz function is the pressure field, and the remaining fields are the velocity components.
      - The density and viscosity are assumed to be constant.
      - The volume load function, if provided, must be volume-specific.
      - This models works for DOFs as a jnp.ndarray.
    """

  def pde_fun(x, ansatz, settings, static_settings, int_point_number, set):
    pressure = lambda t: ansatz(t)[0]  # p
    velocity = lambda t: ansatz(t)[1:]  # v

    grad_p = jax.jacfwd(pressure)(x)
    grad_v = jax.jacfwd(velocity)(x)
    laplace_v = jnp.trace(jax.jacfwd(jax.jacfwd(velocity))(x), axis1=1, axis2=2)

    # Incompressible flow: div(v) = 0
    pde_1 = jnp.asarray([jnp.trace(grad_v)])

    # Linear momentum balance
    if volume_load_fun == None:
      pde_2 = (
          grad_v @ velocity(x)
          + (1 / density) * grad_p
          - (dynamic_viscosity / density) * laplace_v * jnp.ones_like(velocity(x))
      )
    else:
      pde_2 = (
          grad_v @ velocity(x)
          + (1 / density) * grad_p
          - (dynamic_viscosity / density) * laplace_v * jnp.ones_like(velocity(x))
          - volume_load_fun(x)
      )

    return jnp.concatenate([incompressible_weighting * pde_1, pde_2]).flatten()

  return pde_fun


def navier_stokes_incompressible(
    dynamic_viscosity, density=1.0, incompressible_weighting=1.0, volume_load_fun=None
):
  """
    Constructs the transient incompressible Navier-Stokes equations.

    Args:
      dynamic_viscosity (float): Dynamic viscosity of the fluid.
      density (float, optional): Density of the fluid. Default is 1.0.
      incompressible_weighting (float, optional): Weighting factor for the incompressibility constraint. Default is 1.0.
      volume_load_fun (function, optional): Function to compute volume load given spatial coordinates. Default is None.

    Returns:
      function: A function that evaluates the transient incompressible Navier-Stokes equations.

    Notes:
      - The first equation is the constraint of a divergence-free flow field.
      - The remaining equations are the linear momentum balance.
      - The first field in the ansatz function is the pressure field, and the remaining fields are the velocity components.
      - The density and viscosity are assumed to be constant.
      - The volume load function, if provided, must be volume-specific.
      - This models works for DOFs as a jnp.ndarray.
    """

  def pde_fun(x, ansatz, settings, static_settings, int_point_number, set):
    # Number of spatial dimensions
    n_sdim = x.shape[0] - 1

    pressure = lambda t: ansatz(t)[0]  # p
    velocity = lambda t: ansatz(t)[1:]  # v

    grad_p = jax.jacfwd(pressure)(x)[:n_sdim]
    grad_v = jax.jacfwd(velocity)(x)[:, :n_sdim]
    dvdt = jax.jacfwd(velocity)(x)[:, n_sdim]
    laplace_v = jnp.trace(
        jax.jacfwd(jax.jacfwd(velocity))(x)[:, :, :n_sdim], axis1=1, axis2=2
    )

    # Incompressible flow: div(v) = 0
    pde_1 = jnp.asarray([jnp.trace(grad_v)])

    # Linear momentum balance
    if volume_load_fun == None:
      pde_2 = (
          dvdt
          + grad_v @ velocity(x)
          + (1 / density) * grad_p
          - (dynamic_viscosity / density) * laplace_v * jnp.ones_like(velocity(x))
      )
    else:
      pde_2 = (
          dvdt
          + grad_v @ velocity(x)
          + (1 / density) * grad_p
          - (dynamic_viscosity / density) * laplace_v * jnp.ones_like(velocity(x))
          - volume_load_fun(x)
      )

    return jnp.concatenate([incompressible_weighting * pde_1, pde_2]).flatten()

  return pde_fun


### Strain energy functions
def neo_hooke(F, param):
  """
    Computes the strain energy for a neo-Hookean material model of Ciarlet type given the deformation gradient.

    Args:
      F (jnp.ndarray): Deformation gradient tensor.
      param (tuple): Lame parameters (lambda, mu).

    Returns:
      float: Strain energy value for the given deformation gradient.

    Notes:
      - The strain energy function is based on the Ciarlet type neo-Hookean model.
      - This model is suitable for large deformation analysis in hyperelastic materials.
    """
  (lam, mu) = param

  # Right Cauchy-Green tensor
  tr_C = jnp.einsum("ij, ij", F, F)
  J = utility.matrix_det(F)
  ln_J = jnp.log(J)

  # Neo-Hookean strain energy function of Ciarlet type
  psi = (mu / 2) * (tr_C - 3 - 2 * ln_J) + (lam / 4) * (J**2 - 1 - 2 * ln_J)
  return psi


def isochoric_neo_hooke(F, mu):
  """
    Computes the strain energy for an isochoric neo-Hookean material model.

    Args:
      F (jnp.ndarray): Deformation gradient tensor.
      mu (float): Shear modulus.

    Returns:
      float: Strain energy value for the given deformation gradient.
    """
  # Right Cauchy-Green tensor
  tr_C = jnp.einsum("ij, ij", F, F)

  # Strain energy function
  return (mu / 2) * (tr_C - 3)


def linear_elastic_strain_energy(F, param):
  """
    Computes the strain energy for a linear elastic material for small deformations.

    Args:
      F (jnp.ndarray): Deformation gradient tensor.
      param (tuple): Lame parameters (lambda, mu).

    Returns:
      float: Strain energy value for the given deformation gradient.
    """
  (lam, mu) = param

  # Right Cauchy-Green tensor
  disp_grad = F - jnp.identity(F.shape[0])
  strain = (disp_grad + disp_grad.T) / 2

  return (lam / 2) * (jnp.trace(strain)) ** 2 + mu * jnp.trace(strain @ strain.T)


### User potentials/elements
def mixed_reference_domain_potential(
    integrand_fun, ansatz_fun, ref_int_coor, ref_int_weights, mapping_key
):
  """
    Constructs a multi-field 'user potential' for integration of a potential in the reference configuration of finite elements. Works for DOFs as a dict of jnp.ndarrays.

    Args:
        integrand_fun (callable): Function that evaluates the integrand given the integration point, trial ansatz, settings, static settings, and element number.
        ansatz_fun (dict of callables): Functions that constructs the ansatz functions for each field. Keywords have to match with those defined in the DOFs dictionary.
        ref_int_coor (jnp.ndarray): Reference integration coordinates for the isoparametric element.
        ref_int_weights (jnp.ndarray): Reference integration weights for the isoparametric element.
        mapping_key (string): Which solution space to use for the isoparametric mapping.

    Returns:
        function
            A function that evaluates the integrand for the isoparametric element.

    The returned function has the following parameters:
        - fI (dict of jnp.ndarray): Element nodal values.
        - xI (dict of jnp.ndarray): Element nodal coordinates.
        - elem_number (int): Element number.
        - settings (dict): Settings for the computation.
        - static_settings (dict): Static settings for the computation.

    Exemplary integrand function:
        --> def integrand_fun(x_int, ansatz_fun, settings, static_settings, elem_number, set):
        -->     # Definition of custom functional
        -->     x = ansatz_fun['physical coor'](x_int)      # Physical coordinates via isoparametric mapping
        -->     phi_fun = ansatz_fun['phi']                 # Field approximation function with key words as defined in the DOFs dictionary
        -->     phi = phi_fun(x_int)
        -->     dphi_dx = jax.jacrev(phi_fun)(x_int)
        --> 
        -->     x_1 = x
        -->     x_2 = x - jnp.array([1., 0.5])
        -->     source_term = 20 * (jnp.sin(10 * x_1 @ x_1) - jnp.cos(10 * x_2 @ x_2))
        -->     return (1/2) * dphi_dx @ dphi_dx - source_term * phi                    # Returns a scalar potential that is to be integrated in reference configuration

    Notes:
      This models works for DOFs as a dict of jnp.ndarrays.
    """
  field_keys = list(ansatz_fun.keys())

  def user_potential(fI, xI, elem_number, settings, static_settings, set):
    n_dim = xI[field_keys[0]].shape[-1]

    # Define solution spaces
    trial_ansatz = {
        key: (
            lambda x, key=key: ansatz_fun[key](
                x, xI[key], fI[key], settings, True, n_dim
            )
        )
        for key in field_keys
    }

    # Isoparametric mapping
    mapping = lambda x: ansatz_fun[mapping_key](
        x, xI[mapping_key], xI[mapping_key], settings, False, n_dim
    )
    trial_ansatz["physical coor"] = mapping

    # One Gauß point contribution
    def functional(x_int, w_int):
      # Evaluate functional
      discrete_functional_fun = integrand_fun(
          x_int, trial_ansatz, settings, static_settings, elem_number, set
      )

      # Use ansatz for isoparametric mapping and compute determinant of Jacobian
      jacobian = jax.jacfwd(mapping)(x_int)
      det_jacobian = utility.matrix_det(jacobian)

      # Functional contribution
      return discrete_functional_fun * w_int * det_jacobian

    # Sum over Gauß points using vmap (faster but needs more memory)
    return jax.vmap(functional, (0, 0), 0)(ref_int_coor, ref_int_weights).sum()

    # # Sum over Gauß points with jax.lax.map
    # return jax.lax.map(lambda i: functional(ref_int_coor[i], ref_int_weights[i]), jnp.arange(ref_int_coor.shape[0])).sum()

  return user_potential


def mixed_reference_domain_residual(
    integrand_fun, ansatz_fun, ref_int_coor, ref_int_weights, mapping_key
):
  """
    Constructs a multi-field 'user residual' for integration of a weak form in the reference configuration of finite elements. Works for DOFs as a dict of jnp.ndarrays.

    Args:
      integrand_fun (callable): Function that evaluates the integrand given the integration point, trial ansatz, settings, static settings, and element number.
      ansatz_fun (dict of callables): Functions that constructs the ansatz functions for each field. Keywords have to match with those defined in the DOFs dictionary.
      ref_int_coor (jnp.ndarray): Reference integration coordinates for the isoparametric element.
      ref_int_weights (jnp.ndarray): Reference integration weights for the isoparametric element.
      mapping_key (string): Which solution space to use for the isoparametric mapping.

    Returns:
      function
          A function that evaluates the integrand for the isoparametric element.

    The returned function has the following parameters:
      - fI (dict of jnp.ndarray): Element nodal values.
      - xI (dict of jnp.ndarray): Element nodal coordinates.
      - elem_number (int): Element number.
      - settings (dict): Settings for the computation.
      - static_settings (dict): Static settings for the computation.

    Notes:
      This models works for DOFs as a dict of jnp.ndarrays.
    """
  field_keys = list(ansatz_fun.keys())

  def user_residual(fI, xI, elem_number, settings, static_settings, set):
    n_dim = xI[field_keys[0]].shape[-1]

    def integrated_weak_form(test_fI):

      # Define solution spaces
      trial_ansatz = {
          key: (
              lambda x, key=key: ansatz_fun[key](
                  x, xI[key], fI[key], settings, True, n_dim
              )
          )
          for key in field_keys
      }
      test_ansatz = {
          key: (
              lambda x, key=key: ansatz_fun[key](
                  x, xI[key], test_fI[key], settings, True, n_dim
              )
          )
          for key in field_keys
      }

      # Isoparametric mapping
      mapping = lambda x: ansatz_fun[mapping_key](
          x, xI[mapping_key], xI[mapping_key], settings, False, n_dim
      )
      trial_ansatz["physical coor"] = mapping

      # One Gauß point contribution
      def functional(x_int, w_int):
        weak_form = integrand_fun(
            x_int,
            trial_ansatz,
            test_ansatz,
            settings,
            static_settings,
            elem_number,
            set,
        )

        # Use ansatz for isoparametric mapping and compute determinant of Jacobian
        jacobian = jax.jacfwd(mapping)(x_int)
        det_jacobian = utility.matrix_det(jacobian)

        # Weak form contribution
        return weak_form * w_int * det_jacobian

      # Sum over Gauß points
      return jax.vmap(functional, (0, 0), 0)(ref_int_coor, ref_int_weights).sum(
          axis=0
      )

    residual = jax.jacrev(integrated_weak_form)(fI)
    return residual

  return user_residual


def mixed_reference_surface_potential(
    integrand_fun, ansatz_fun, ref_int_coor, ref_int_weights, mapping_key
):
  """
    Constructs a multi-field 'user potential' for integration of a potential in the reference configuration of surface elements. Works for DOFs as a dict of jnp.ndarrays.

    Args:
      integrand_fun (callable): Function that evaluates the integrand given the integration point, trial ansatz, settings, static settings, and element number.
      ansatz_fun (dict of callables): Functions that constructs the ansatz functions for each field. Keywords have to match with those defined in the DOFs dictionary.
      ref_int_coor (jnp.ndarray): Reference integration coordinates for the isoparametric element.
      ref_int_weights (jnp.ndarray): Reference integration weights for the isoparametric element.
      mapping_key (string): Which solution space to use for the isoparametric mapping.

    Returns:
      function
          A function that evaluates the integrand for the isoparametric element.

    The returned function has the following parameters:
      - fI (dict of jnp.ndarray): Element nodal values.
      - xI (dict of jnp.ndarray): Element nodal coordinates.
      - elem_number (int): Element number.
      - settings (dict): Settings for the computation.
      - static_settings (dict): Static settings for the computation.

    Notes:
      This models works for DOFs as a dict of jnp.ndarrays.
    """
  field_keys = list(ansatz_fun.keys())

  def user_potential(fI, xI, elem_number, settings, static_settings, set):
    n_dim = xI[field_keys[0]].shape[-1]

    # Define solution spaces
    trial_ansatz = {
        key: (
            lambda x, key=key: ansatz_fun[key](
                x, xI[key], fI[key], settings, True, n_dim
            )
        )
        for key in field_keys
    }

    # Isoparametric mapping
    mapping = lambda x: ansatz_fun[mapping_key](
        x, xI[mapping_key], xI[mapping_key], settings, False, n_dim - 1
    )  # Shape functions in n_dim-1 dimensions
    trial_ansatz["physical coor"] = mapping

    # One Gauß point contribution
    def functional(x_int, w_int):
      integrand = integrand_fun(
          x_int, trial_ansatz, settings, static_settings, elem_number, set
      )

      # Use ansatz for isoparametric mapping and compute determinant of Jacobian
      jacobian = jax.jacfwd(mapping)(x_int)
      match n_dim:
        case 2:
          G_1 = jnp.concatenate([jacobian, jnp.asarray([0])])
          E_3 = jnp.asarray([0, 0, 1])
          scaling = jnp.linalg.norm(jnp.cross(G_1, E_3))
        case 3:
          G_1 = jacobian[:, 0]
          G_2 = jacobian[:, 1]
          scaling = jnp.linalg.norm(jnp.cross(G_1, G_2))
        case _:
          raise ValueError(
              "Surface element supports currently only lines in 2d and 2d-surfaces in 3d."
          )

      # Functional contribution
      return integrand * w_int * scaling

    # Sum over Gauß points
    potential = jax.vmap(functional, (0, 0), 0)(ref_int_coor, ref_int_weights).sum(
        axis=0
    )

    return potential

  return user_potential


def mixed_reference_surface_residual(
    integrand_fun, ansatz_fun, ref_int_coor, ref_int_weights, mapping_key
):
  """
    Constructs a multi-field 'user residual' for integration of a weak form in the reference configuration of surface elements. Works for DOFs as a dict of jnp.ndarrays.

    Args:
      integrand_fun (callable): Function that evaluates the integrand given the integration point, trial ansatz, settings, static settings, and element number.
      ansatz_fun (dict of callables): Functions that constructs the ansatz functions for each field. Keywords have to match with those defined in the DOFs dictionary.
      ref_int_coor (jnp.ndarray): Reference integration coordinates for the isoparametric element.
      ref_int_weights (jnp.ndarray): Reference integration weights for the isoparametric element.
      mapping_key (string): Which solution space to use for the isoparametric mapping.

    Returns:
      function
          A function that evaluates the integrand for the isoparametric element.

    The returned function has the following parameters:
      - fI (dict of jnp.ndarray): Element nodal values.
      - xI (dict of jnp.ndarray): Element nodal coordinates.
      - elem_number (int): Element number.
      - settings (dict): Settings for the computation.
      - static_settings (dict): Static settings for the computation.

    Notes:
      This models works for DOFs as a dict of jnp.ndarrays.
    """
  field_keys = list(ansatz_fun.keys())

  def user_residual(fI, xI, elem_number, settings, static_settings, set):
    n_dim = xI[field_keys[0]].shape[-1]

    def integrated_weak_form(test_fI):

      # Define solution spaces
      trial_ansatz = {
          key: (
              lambda x, key=key: ansatz_fun[key](
                  x, xI[key], fI[key], settings, True, n_dim
              )
          )
          for key in field_keys
      }
      test_ansatz = {
          key: (
              lambda x, key=key: ansatz_fun[key](
                  x, xI[key], test_fI[key], settings, True, n_dim
              )
          )
          for key in field_keys
      }

      # Isoparametric mapping
      mapping = lambda x: ansatz_fun[mapping_key](
          x, xI[mapping_key], xI[mapping_key], settings, False, n_dim - 1
      )  # Shape functions in n_dim-1 dimensions
      trial_ansatz["physical coor"] = mapping

      # One Gauß point contribution
      def functional(x_int, w_int):
        weak_form = integrand_fun(
            x_int,
            trial_ansatz,
            test_ansatz,
            settings,
            static_settings,
            elem_number,
            set,
        )

        # Use ansatz for isoparametric mapping and compute determinant of Jacobian
        jacobian = jax.jacfwd(mapping)(x_int)
        match n_dim:
          case 2:
            G_1 = jnp.concatenate([jacobian, jnp.asarray([0])])
            E_3 = jnp.asarray([0, 0, 1])
            scaling = jnp.linalg.norm(jnp.cross(G_1, E_3))
          case 3:
            G_1 = jacobian[:, 0]
            G_2 = jacobian[:, 1]
            scaling = jnp.linalg.norm(jnp.cross(G_1, G_2))
          case _:
            raise ValueError(
                "Surface element supports currently only lines in 2d and 2d-surfaces in 3d."
            )

        # Weak form contribution
        return weak_form * w_int * scaling

      # Sum over Gauß points
      return jax.vmap(functional, (0, 0), 0)(ref_int_coor, ref_int_weights).sum(
          axis=0
      )

    residual = jax.jacrev(integrated_weak_form)(fI)
    return residual

  return user_residual


def isoparametric_domain_integrate_potential(
    integrand_fun, ansatz_fun, ref_int_coor, ref_int_weights, initial_config=True
):
  """
    Constructs a local integrand fun (user potential) for integration of functions in the reference configuration of isoparametric elements. Works for DOFs as a jnp.ndarray.

    Args:
      integrand_fun (callable): Function that evaluates the integrand given the integration point, trial ansatz, settings, static settings, and element number.
      ansatz_fun (callable): Function that constructs the ansatz functions.
      ref_int_coor (jnp.ndarray): Reference integration coordinates for the isoparametric element.
      ref_int_weights (jnp.ndarray): Reference integration weights for the isoparametric element.
      initial_config (bool, optional): If True, use the initial configuration for isoparametric mapping. Default is True.

    Returns:
      function
          A function that evaluates the integrand for the isoparametric element.

    The returned function has the following parameters:
      - fI (jnp.ndarray): Element nodal values.
      - xI (jnp.ndarray): Element nodal coordinates.
      - elem_number (int): Element number.
      - settings (dict): Settings for the computation.
      - static_settings (dict): Static settings for the computation.

    Notes:
      - This models works for DOFs as a jnp.ndarray.
    """

  def user_potential(fI, xI, elem_number, settings, static_settings, set):
    n_dofs = fI.flatten().shape[0]
    n_dim = xI.shape[-1]

    # One Gauß point contribution
    def functional(x_int, w_int):

      def integrated_weak_form(trial_fI):
        # Define trial and test function ansatz
        trial_ansatz = lambda x: ansatz_fun(
            x, xI, trial_fI, settings, True, n_dim
        )

        # Evaluate functional
        discrete_functional_fun = integrand_fun(
            x_int, trial_ansatz, settings, static_settings, elem_number, set
        )

        # Use ansatz for isoparametric mapping and compute determinant of Jacobian
        if initial_config:
          mapping = lambda x: ansatz_fun(x, xI, xI, settings, False, n_dim)
        else:
          # Take first n_dim fields as displacements
          uI = fI[:, : xI.shape[-1]]
          mapping = lambda x: ansatz_fun(
              x, xI + uI, xI + uI, settings, overwrite_diff=False
          )

        jacobian = jax.jacfwd(mapping)(x_int)
        det_jacobian = utility.matrix_det(jacobian)

        return w_int * det_jacobian * discrete_functional_fun

      # Residual
      return integrated_weak_form(fI)

    # Sum over Gauß points
    functional_contrib = jax.vmap(functional, (0, 0), 0)(
        ref_int_coor, ref_int_weights
    )
    return functional_contrib.sum(axis=0)

  return user_potential


def isoparametric_domain_element_galerkin(
    weak_form_fun, ansatz_fun, ref_int_coor, ref_int_weights, initial_config=True
):
  """
    Constructs an isoparametric domain element for Galerkin methods using a given weak form function and DOFs as jnp.ndarray.

    This function returns a user element function that evaluates the residual and tangent matrix for an isoparametric
    element based on the provided weak form and ansatz functions.
    The function can e.g. be used in combination with the weak_form_fun hyperelastic_steady_state_weak.

    Args:
      weak_form_fun (callable): Function that evaluates the weak form given the integration point, trial ansatz,
          test ansatz, settings, static settings, and element number.
      ansatz_fun (callable): Function that constructs the ansatz (trial and test) functions.
      ref_int_coor (jnp.ndarray): Reference integration coordinates for the isoparametric element.
      ref_int_weights (jnp.ndarray): Reference integration weights for the isoparametric element.
      initial_config (bool, optional): If True, use the initial configuration for isoparametric mapping. Default is True.

    Returns:
      function
          A function that evaluates the residual and tangent matrix for the isoparametric element.

    Notes:
      - The returned function can be used to evaluate the residual and tangent matrix for the element.
      - The mode parameter in the returned function can be 'residual' or 'tangent'.

    The returned function has the following parameters:
      - fI (jnp.ndarray): Element nodal values.
      - xI (jnp.ndarray): Element nodal coordinates.
      - elem_number (int): Element number.
      - settings (dict): Settings for the computation.
      - static_settings (dict): Static settings for the computation.
      - mode (str): Specifies whether to compute the 'residual' or 'tangent'.

    Notes:
      - This models works for DOFs as a jnp.ndarray.
    """

  def user_element(fI, xI, elem_number, settings, static_settings, mode, set):
    n_dofs = fI.flatten().shape[0]
    n_dim = xI.shape[-1]

    # One Gauß point contribution
    def residual_and_tangent(x_int, w_int):

      def integrated_weak_form(test_fI, trial_fI):
        # Define trial and test function ansatz
        trial_ansatz = lambda x: ansatz_fun(
            x, xI, trial_fI, settings, True, n_dim
        )
        test_ansatz = lambda x: ansatz_fun(
            x, xI, test_fI, settings, True, n_dim
        )

        # Evaluate weak form
        discrete_weak_form = weak_form_fun(
            x_int,
            trial_ansatz,
            test_ansatz,
            settings,
            static_settings,
            elem_number,
            set,
        )

        # Use ansatz for isoparametric mapping and compute determinant of Jacobian
        if initial_config:
          mapping = lambda x: ansatz_fun(x, xI, xI, settings, False, n_dim)
        else:
          # Take first n_dim fields as displacements
          uI = fI[:, : xI.shape[-1]]
          mapping = lambda x: ansatz_fun(
              x, xI + uI, xI + uI, settings, overwrite_diff=False
          )

        jacobian = jax.jacfwd(mapping)(x_int)
        det_jacobian = utility.matrix_det(jacobian)

        return w_int * det_jacobian * discrete_weak_form

      # Residual
      residual_fun = jax.jacrev(integrated_weak_form, argnums=0)
      residual = residual_fun(fI, fI).flatten()

      # Tangent
      tangent_fun = jax.jacfwd(residual_fun, argnums=1)
      tangent = tangent_fun(fI, fI).reshape(n_dofs, n_dofs)

      return (residual, tangent)

    # Sum over Gauß points
    (residual_contrib, tangent_contrib) = jax.vmap(
        residual_and_tangent, (0, 0), (0, 0)
    )(ref_int_coor, ref_int_weights)
    residual = residual_contrib.sum(axis=0)
    tangent = tangent_contrib.sum(axis=0)

    if mode == "residual":
      return residual
    elif mode == "tangent":
      return tangent
    else:
      raise ValueError("User elem mode has to be 'residual' or 'tangent'.")

  return user_element


def isoparametric_surface_element_galerkin(
    weak_form_fun,
    ansatz_fun,
    ref_int_coor,
    ref_int_weights,
    tangent_contributions,
    initial_config=True,
):
  """
    Constructs an isoparametric surface element for Galerkin methods using a given weak form function and DOFs as jnp.ndarray.

    This function returns a user element function that evaluates the residual and tangent matrix for an isoparametric
    surface element based on the provided weak form and ansatz functions. The function can, for example, be used in
    combination with the weak_form_fun neumann_weak.

    Args:
      weak_form_fun (callable): Function that evaluates the weak form given the integration point, trial ansatz,
          test ansatz, settings, static settings, and element number.
      ansatz_fun (callable): Function that constructs the ansatz (trial and test) functions.
      ref_int_coor (jnp.ndarray): Reference integration coordinates for the isoparametric element.
      ref_int_weights (jnp.ndarray): Reference integration weights for the isoparametric element.
      tangent_contributions (bool): If True, compute the tangent contributions; otherwise, return zero tangent.
      initial_config (bool, optional): If True, use the initial configuration for isoparametric mapping. Default is True.

    Returns:
      callable:
        A function that evaluates the residual and tangent matrix for the isoparametric surface element.

    Notes:
      - Currently, the weak form function must not contain derivatives wrt. the primary fields.
      - The returned function can be used to evaluate the residual and tangent matrix for the element.
      - The mode parameter in the returned function can be 'residual' or 'tangent'.

    The returned function has the following parameters:
      - fI (jnp.ndarray): Element nodal values.
      - xI (jnp.ndarray): Element nodal coordinates.
      - elem_number (int): Element number.
      - settings (dict): Settings for the computation.
      - static_settings (dict): Static settings for the computation.
      - mode (str): Specifies whether to compute the 'residual' or 'tangent'.

    Notes:
      - This models works for DOFs as a jnp.ndarray.
    """

  def user_element(fI, xI, elem_number, settings, static_settings, mode, set):
    n_dofs = fI.flatten().shape[0]
    n_dim = xI.shape[-1]

    # One Gauß point contribution
    def residual_and_tangent(x_int, w_int):

      def integrated_weak_form(test_fI, trial_fI):
        # Define trial and test function ansatz
        trial_ansatz = lambda x: ansatz_fun(
            x, xI, trial_fI, settings, False, n_dim - 1
        )
        test_ansatz = lambda x: ansatz_fun(
            x, xI, test_fI, settings, False, n_dim - 1
        )

        # Evaluate weak form
        discrete_weak_form = weak_form_fun(
            x_int,
            trial_ansatz,
            test_ansatz,
            settings,
            static_settings,
            elem_number,
            set,
        )

        # Compute the Jacobian
        if initial_config:
          mapping = lambda x: ansatz_fun(
              x, xI, xI, settings, False, n_dim - 1
          )
        else:
          # Take first n_dim fields as displacements
          uI = fI[:, : xI.shape[-1]]
          mapping = lambda x: ansatz_fun(
              x, xI + uI, xI + uI, settings, overwrite_diff=False
          )
        jacobian = jax.jacrev(mapping)(x_int)

        match n_dim:
          case 2:
            G_1 = jnp.concatenate([jacobian, jnp.asarray([0])])
            E_3 = jnp.asarray([0, 0, 1])
            scaling = jnp.linalg.norm(jnp.cross(G_1, E_3))
          case 3:
            G_1 = jacobian[:, 0]
            G_2 = jacobian[:, 1]
            scaling = jnp.linalg.norm(jnp.cross(G_1, G_2))
          case _:
            raise ValueError(
                "Surface element supports currently only lines in 2d and 2d-surfaces in 3d."
            )
        return w_int * scaling * discrete_weak_form

      # Residual
      residual_fun = jax.jacrev(integrated_weak_form, argnums=0)
      residual = residual_fun(fI, fI).flatten()

      # Tangent
      if tangent_contributions:
        tangent_fun = jax.jacfwd(residual_fun, argnums=1)
        tangent = tangent_fun(fI, fI).reshape(n_dofs, n_dofs)
      else:
        tangent = jnp.zeros((n_dofs, n_dofs), dtype=jnp.float64)

      return (residual, tangent)

    # Sum over Gauß points
    (residual_contrib, tangent_contrib) = jax.vmap(
        residual_and_tangent, (0, 0), (0, 0)
    )(ref_int_coor, ref_int_weights)
    residual = residual_contrib.sum(axis=0)
    tangent = tangent_contrib.sum(axis=0)

    if mode == "residual":
      return residual
    elif mode == "tangent":
      return tangent
    else:
      raise ValueError("User elem mode has to be 'residual' or 'tangent'.")

  return user_element

## Functions for coupling with DAE solver

def mixed_reference_domain_residual_time(integrand_fun, ansatz_fun, ref_int_coor, ref_int_weights, mapping_key):
  """
  Constructs a multi-field 'user residual' for time-dependent weak forms.

  Args:
    integrand_fun (callable): Function evaluating the integrand given:
        x_int, trial_ansatz (callable with args = (x, t)), test_ansatz 
        (callable with args = (x,)), settings, static_settings, elem_number, set.
    ansatz_fun (dict of callables): Each function must have signature:
        f(x, t, xI, fI, settings, is_trial, n_dim)
    ref_int_coor (jnp.ndarray): Reference integration coordinates.
    ref_int_weights (jnp.ndarray): Integration weights.
    mapping_key (str): Key for the isoparametric mapping.

  Returns:
    A function with signature:
        user_residual(fI, xI, elem_number, settings, static_settings, set)
    that computes the element residual.
  """
  keys = ansatz_fun.keys()

  def user_residual(fI, xI, elem_number, settings, static_settings, set):
    n_dim = xI[next(iter(keys))].shape[-1]

    def integrated_weak_form(test_fI):

      trial_ansatz = {}
      test_ansatz = {}
      for key in keys:
        trial_ansatz[key] = lambda x, t, key=key: ansatz_fun[key](x, xI[key], fI(t)[key], settings, True, n_dim)
        test_ansatz[key] = lambda x, key=key: ansatz_fun[key](x, xI[key], test_fI[key], settings, True, n_dim)

      # Isoparametric mapping
      @jax.jit
      def mapping(x):
        return ansatz_fun[mapping_key](x, xI[mapping_key], xI[mapping_key], settings, False, n_dim)
      trial_ansatz['physical coor'] = mapping

      def functional(x_int, w_int, int_num):

        num_args = len(signature(integrand_fun).parameters)
        match num_args:
          case 7:
            weak_form = integrand_fun(x_int, trial_ansatz, test_ansatz, settings, static_settings, elem_number, set)
          case 8:
            weak_form = integrand_fun(x_int, trial_ansatz, test_ansatz, settings, static_settings, elem_number, int_num, set)

        # Calculate the jacobian and its determinant.
        jacobian = jax.jacfwd(mapping)(x_int)
        det_jacobian = utility.matrix_det(jacobian)
        return weak_form * w_int * det_jacobian

      # Numerical integration
      int_point_numbers = jnp.arange(ref_int_coor.shape[0])
      return jax.vmap(functional, (0, 0, 0), 0)(ref_int_coor, ref_int_weights, int_point_numbers).sum(axis=0)

    # Get residual by taking the derivative of the weak form wrt the test functions nodal values.
    test_fI = utility.dict_zeros_like(fI(0.))
    residual = jax.jacrev(integrated_weak_form)(test_fI)
    return residual

  return user_residual

def mixed_reference_domain_int_var_updates(local_int_var_updates_fun, ansatz_fun, ref_int_coor, ref_int_weights, mapping_key):
  """
  Similar as mixed_reference_domain_residual_time, but generates a function that doesn't compute the residual, but the internal variables based on the local_int_var_updates_fun.
  """
  keys = ansatz_fun.keys()

  def int_var_fun(fI, xI, elem_number, settings, static_settings, set):
    n_dim = xI[next(iter(keys))].shape[-1]

    trial_ansatz = {}
    for key in keys:
      trial_ansatz[key] = lambda x, t, key=key: ansatz_fun[key](x, xI[key], fI(t)[key], settings, True, n_dim)

    # Isoparametric mapping
    @jax.jit
    def mapping(x):
      return ansatz_fun[mapping_key](x, xI[mapping_key], xI[mapping_key], settings, False, n_dim)
    trial_ansatz['physical coor'] = mapping

    def local_fun(x_int, int_num):
      return local_int_var_updates_fun(x_int, trial_ansatz, settings, static_settings, elem_number, int_num, set)

    int_point_numbers = jnp.arange(ref_int_coor.shape[0])
    return jax.vmap(local_fun, (0, 0), 0)(ref_int_coor, int_point_numbers)

  return int_var_fun


### Time integration procedures
def forward_backward_euler_weak(inertia_coeff_fun):
  """
    Constructs a weak form time discretization function using the Forward or Backward Euler method.

    Deprecated: This function is deprecated and may be removed in future versions. The new time integration procedures
    are now implemented in the dae module.

    For an examplary use, see the examples with explicit and implicit time integration.

    Args:
      inertia_coeff_fun (callable): Function to compute the inertia coefficient given spatial coordinates and settings.

    Returns:
      callable:
        A function that evaluates the weak form time discretization for the PDE.

    The returned function has the following parameters:
      - x (jnp.ndarray): Spatial coordinates at the integration point.
      - ansatz (callable): Ansatz function representing the primary field at time step n+1.
      - test_ansatz (callable): Test ansatz function representing the test function.
      - settings (dict): Settings for the computation, including:
        - 'dofs n': Degrees of freedom at the previous time step.
        - 'connectivity': Connectivity information for the elements.
        - 'time increment': Time increment delta_t.
      - static_settings (dict): Static settings for the computation.
      - int_point_number (int): Integration point number.
      - set: Number of domain

    Notes:
      - The primary field is evaluated at both time steps n and n+1.
      - The time derivative is computed using the Forward or Backward Euler method, depending on the use case.
      - The inertia coefficient is used to scale the time derivative term.
      - This models works for DOFs as a jnp.ndarray.
    """

  def time_discretization_fun(
      x, ansatz, test_ansatz, settings, static_settings, int_point_number, set
  ):
    # Test function
    test_function = test_ansatz(x)

    # Warning if it was defined in static_settings
    assert "connectivity" not in static_settings, \
        "'connectivity' has been moved to 'settings' in order to reduce compile time. \
            Further, you should not transform it to a tuple of tuples anymore."

    # Primary field evaluated at n and n+1
    dofs_n = settings["dofs n"]
    neighbor_list = settings["connectivity"][set][int_point_number]
    local_dofs_n = dofs_n[neighbor_list]
    ansatz_n = lambda x: solution_structures.solution_structure(
        x, int_point_number, local_dofs_n, settings, static_settings, set
    )
    field_n = ansatz_n(x)
    field = ansatz(x)

    # Forward/Backward Euler
    delta_t = settings["time increment"]
    time_derivative = (field - field_n) / delta_t

    # E.g. heat capacity or whatever is multiplied with first order time derivative
    inertia_coeff = inertia_coeff_fun(x, settings)
    return -inertia_coeff * jnp.dot(test_function, time_derivative)

  return time_discretization_fun

def central_differences(density_fun, damping):
  """
    Constructs weak form contributions for acceleration (and optional damping) term based on central differences.

    Deprecated: This function is deprecated and may be removed in future versions. The new time integration procedures
    are now implemented in the dae module.

    Args:
      density_fun (callable): Function to compute the density coefficient given spatial coordinates and settings.
      damping (bool): Whether to turn on damping. Damping requires settings['damping coefficient'] to be set.

    Returns:
      callable:
        A function that evaluates the weak form contributions.

    The returned function has the following parameters:
      - x (jnp.ndarray): Spatial coordinates at the integration point.
      - ansatz (callable): Ansatz function representing the primary field at time step n+1.
      - test_ansatz (callable): Test ansatz function representing the test function.
      - settings (dict): Settings for the computation, including:
        - 'dofs n': Degrees of freedom at the previous time step.
        - 'connectivity': Connectivity information for the elements.
        - 'time increment': Time increment delta_t.
      - static_settings (dict): Static settings for the computation.
      - int_point_number (int): Integration point number.
      - set: Number of domain

    Notes:
      - Requires settings['dofs n], settings['dofs n-1'] and settings['time increment'].
      - This models works for DOFs as a jnp.ndarray.
    """

  def weak_form_fun(x, ansatz, test_ansatz, settings, static_settings, int_point_number, set):

    # Test function
    test_function = test_ansatz(x)

    # Warning if it was defined in static_settings
    assert "connectivity" not in static_settings, \
        "'connectivity' has been moved to 'settings' in order to reduce compile time. \
            Further, you should not transform it to a tuple of tuples anymore."

    # Primary field evaluated at n-1, n and n+1
    dofs_n = settings["dofs n"]
    dofs_n_min_1 = settings["dofs n-1"]
    neighbor_list = settings["connectivity"][set][int_point_number]
    ansatz_n = lambda x: solution_structures.solution_structure(x, int_point_number, dofs_n[neighbor_list], settings,
                                                                static_settings, set)
    ansatz_n_min_1 = lambda x: solution_structures.solution_structure(
        x,
        int_point_number,
        dofs_n_min_1[neighbor_list],
        settings,
        static_settings,
        set,
    )
    u_n = ansatz_n(x)
    u_n_min_1 = ansatz_n_min_1(x)
    u_n_plus_1 = ansatz(x)

    # Central differences
    delta_t = settings["time increment"]
    a_n = (u_n_plus_1 - 2 * u_n - u_n_min_1) / (delta_t**2)
    v_n = (u_n_plus_1 - u_n_min_1) / (2 * delta_t)

    # E.g. heat capacity or whatever is multiplied with first order time derivative
    density = density_fun(x, settings)
    if damping:
      damping_coeff = settings["damping coefficient"]
      return density * jnp.dot(test_function, a_n + damping_coeff * v_n)
    else:
      return density * jnp.dot(test_function, a_n)

  return weak_form_fun

### Convenience functions for modelling

@utility.jit_with_docstring(inline=True)
def aug_lag_potential(lag, constr, eps):
  """
  Augmented Lagrangian potential function for an inequality constraint (constr >= 0).

  This function constructs a C1 continuous augmented Lagrangian potential that can be added 
  to a potential function to enforce the inequality constraint. It is analogous to adding a term 
  'lambda * g' when imposing an equality constraint (g = 0). For maximization problems, the potential 
  should be subtracted.

  Args:
    lag: Scalar Lagrange multiplier.
    constr: Scalar evaluated constraint.
    eps: Scalar numerical parameter controlling the conditioning.

  Returns:
    The value of the augmented Lagrangian potential.
  """

  aug_lag = lag + constr * eps
  condlist = [aug_lag <= 0, aug_lag > 0]
  funclist = [constr * (aug_lag - eps / 2 * constr), -(aug_lag - eps * constr)**2 / (2 * eps)]
  return jnp.piecewise(aug_lag, condlist, funclist)

@utility.jit_with_docstring(inline=True)
def kelvin_mandel_extract(a):
  """
  Converts a 6-component Kelvin-Mandel representation vector into a 3x3 symmetric matrix.

  In the Kelvin-Mandel notation, the off-diagonal components are scaled by 1/sqrt(2) to preserve 
  norm equivalence between the tensor and its vector representation. This function extracts the 
  full symmetric matrix from its compact vector representation.

  Args:
    a: 1-dimensional array-like with 6 elements corresponding to the Kelvin-Mandel representation 
      of a 3x3 symmetric tensor. The expected ordering is: [a11, a22, a33, a12, a13, a23]

  Returns:
    A 3x3 jnp.ndarray representing the symmetric tensor in matrix form.
  """
  one_sqrt_2 = 1 / jnp.sqrt(2.)
  A12 = one_sqrt_2 * a[3]
  A13 = one_sqrt_2 * a[4]
  A23 = one_sqrt_2 * a[5]
  return jnp.asarray([a[0], A12, A13, A12, a[1], A23, A13, A23, a[2]]).reshape((3, 3))
