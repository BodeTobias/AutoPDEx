# variational_schemes.py
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
This module contains functions for computing residuals and functionals at integration points using different variational schemes.

Usually, the variational_schemes module does not have to be called by the user, but it is invoked by the assembly module in the 
'sparse' and 'dense' modes and evaluates integration point contributions of the functional to be integrated or the residual to be assembled. 
A distinction is made between the least squares variational method for solving PDEs or 
approximating functions and the Galerkin method in its weak and strong (without integration by parts) forms.
"""

import jax
import jax.numpy as jnp

from autopdex import spaces
from autopdex import solution_structures as ss
from autopdex.utility import jit_with_docstring


@jit_with_docstring(static_argnames=["static_settings", "set"])
def functional_at_int_point(
    x_i, w_i, int_point_number, local_dofs, settings, static_settings, set
):
    """
    Computes the functional at an integration point for 'least square pde loss' or 'least square function approximation'.

    Args:
      x_i (jnp.ndarray): Coordinates of the integration point.
      w_i (float): Integration weight.
      int_point_number (int): Integration point number.
      local_dofs (jnp.ndarray): Local degrees of freedom.
      settings (dict): Settings for the computation.
      static_settings (dict): Static settings for the computation.
      set (str): Identifier for the current set.

    Returns:
      float: Computed functional value.
    """
    algorithm = static_settings["variational scheme"][set]
    match algorithm:
        case "least square function approximation":
            return least_square_function_approximation(
                x_i, w_i, int_point_number, local_dofs, settings, static_settings, set
            )
        case "least square pde loss":
            return least_square_pde_loss(
                x_i, w_i, int_point_number, local_dofs, settings, static_settings, set
            )
        case _:
            assert (
                False
            ), "variational scheme is not defined or not available with this settings."


@jit_with_docstring(static_argnames=["static_settings", "set"])
def direct_residual_at_int_point(
    x_i, w_i, int_point_number, local_dofs, settings, static_settings, set
):
    """
    Computes the direct residual at an integration point for 'strong form galerkin'.

    Args:
      x_i (jnp.ndarray): Coordinates of the integration point.
      w_i (float): Integration weight.
      int_point_number (int): Integration point number.
      local_dofs (jnp.ndarray): Local degrees of freedom.
      settings (dict): Settings for the computation.
      static_settings (dict): Static settings for the computation.
      set (str): Identifier for the current set.

    Returns:
      float: Computed residual value.
    """
    algorithm = static_settings["variational scheme"][set]
    match algorithm:
        case "strong form galerkin":
            return strong_form_galerkin(
                x_i, w_i, int_point_number, local_dofs, settings, static_settings, set
            )
        case _:
            assert (
                False
            ), "variational scheme is not defined or not available with this settings."


@jit_with_docstring(static_argnames=["static_settings", "set"])
def residual_from_deriv_at_int_point(
    x_i,
    w_i,
    int_point_number,
    local_dofs,
    local_virt_dofs,
    settings,
    static_settings,
    set,
):
    """
    Computes the residual from derivatives at an integration point for 'weak form galerkin'.

    Args:
      x_i (jnp.ndarray): Coordinates of the integration point.
      w_i (float): Integration weight.
      int_point_number (int): Integration point number.
      local_dofs (jnp.ndarray): Local degrees of freedom.
      local_virt_dofs (jnp.ndarray): Local virtual degrees of freedom.
      settings (dict): Settings for the computation.
      static_settings (dict): Static settings for the computation.
      set (str): Identifier for the current set.

    Returns:
      float: Computed residual value.
    """
    algorithm = static_settings["variational scheme"][set]
    match algorithm:
        case "weak form galerkin":
            return weak_form_galerkin(
                x_i,
                w_i,
                int_point_number,
                local_dofs,
                local_virt_dofs,
                settings,
                static_settings,
                set,
            )
        case _:
            assert (
                False
            ), "variational scheme is not defined or not available with this settings."


@jit_with_docstring(static_argnames=["static_settings", "set"])
def least_square_pde_loss(
    x_i, w_i, int_point_number, local_dofs, settings, static_settings, set
):
    """
    Computes the least square PDE loss at an integration point.

    Takes the set of pdes from static_settings['model']

    Args:
      x_i (jnp.ndarray): Coordinates of the integration point.
      w_i (float): Integration weight.
      int_point_number (int): Integration point number.
      local_dofs (jnp.ndarray): Local degrees of freedom.
      settings (dict): Settings for the computation.
      static_settings (dict): Static settings for the computation.
      set (str): Identifier for the current set.

    Returns:
      float: Computed least square PDE loss.
    """

    def approximate_solution(x):
        ansatz = ss.solution_structure(
            x, int_point_number, local_dofs, settings, static_settings, set
        )

        if ansatz.shape[0] == 1:
            return ansatz[0]
        else:
            return ansatz

    if static_settings["shape function mode"] == "compiled":
        x_i = 0 * x_i

    pde = static_settings["model"][set](
        x_i, approximate_solution, settings, static_settings, int_point_number, set
    )

    return w_i * jnp.dot(pde, pde) / 2


@jit_with_docstring(static_argnames=["static_settings", "set"])
def weak_form_galerkin(
    x_i,
    w_i,
    int_point_number,
    local_dofs,
    local_virt_dofs,
    settings,
    static_settings,
    set,
):
    """
    Computes the weak form Galerkin residual at an integration point.

    Takes the set of pdes from static_settings['model']

    Args:
      x_i (jnp.ndarray): Coordinates of the integration point.
      w_i (float): Integration weight.
      int_point_number (int): Integration point number.
      local_dofs (jnp.ndarray): Local degrees of freedom.
      local_virt_dofs (jnp.ndarray): Local virtual degrees of freedom.
      settings (dict): Settings for the computation.
      static_settings (dict): Static settings for the computation.
      set (str): Identifier for the current set.

    Returns:
      float: Computed weak form Galerkin residual.
    """
    if static_settings["shape function mode"] == "compiled":
        x_i = 0 * x_i

    def weak_form(virtual_dofs):
        def test_function(x):
            if local_dofs.shape[1] == 1:
                return ss.solution_structure(
                    x, int_point_number, virtual_dofs, settings, static_settings, set
                )[0]
            else:
                return ss.solution_structure(
                    x, int_point_number, virtual_dofs, settings, static_settings, set
                )

        def trial_function(x):
            if local_dofs.shape[1] == 1:
                return ss.solution_structure(
                    x, int_point_number, local_dofs, settings, static_settings, set
                )[0]
            else:
                return ss.solution_structure(
                    x, int_point_number, local_dofs, settings, static_settings, set
                )

        # Evaluate weak form and take derivative with respect to local_virtual_dofs
        weak_pde = static_settings["model"][set](
            x_i,
            trial_function,
            test_function,
            settings,
            static_settings,
            int_point_number,
            set,
        )
        return weak_pde

    # Test performance for jacrev and jacfwd
    residual = w_i * jax.jacrev(weak_form)(local_virt_dofs)
    return residual


@jit_with_docstring(static_argnames=["static_settings", "set"])
def strong_form_galerkin(
    x_i, w_i, int_point_number, local_dofs, settings, static_settings, set
):
    """
    Computes the strong form Galerkin residual at an integration point.

    Takes the set of pdes from static_settings['model']

    Args:
        x_i (jnp.ndarray): Coordinates of the integration point.
      w_i (float): Integration weight.
      int_point_number (int): Integration point number.
      local_dofs (jnp.ndarray): Local degrees of freedom.
      settings (dict): Settings for the computation.
      static_settings (dict): Static settings for the computation.
      set (str): Identifier for the current set.

    Returns:
      float: Computed strong form Galerkin residual.
    """

    def approximate_solution(x):
        if local_dofs.shape[1] == 1:
            return ss.solution_structure(
                x, int_point_number, local_dofs, settings, static_settings, set
            )[0]
        else:
            return ss.solution_structure(
                x, int_point_number, local_dofs, settings, static_settings, set
            )

    shp_fun_mode = static_settings["shape function mode"]
    if shp_fun_mode == "compiled":
        x_i = 0 * x_i
        test_functions = settings["compiled shape functions"][set][0][int_point_number]
    elif shp_fun_mode == "direct":
        test_functions = jax.jacrev(spaces.solution_space, argnums=2)(
            x_i, int_point_number, local_dofs, settings, static_settings, set
        )[0, :, 0]
    else:
        assert (
            False
        ), "Shape function mode neither specified as 'direct' nor 'compiled'!"

    pde = static_settings["model"][set](
        x_i, approximate_solution, settings, static_settings, int_point_number, set
    )

    if local_dofs.shape[1] == 1:
        return w_i * test_functions * pde
    else:
        return w_i * jnp.outer(test_functions, pde)


@jit_with_docstring(static_argnames=["static_settings", "set"])
def least_square_function_approximation(
    x_i, w_i, int_point_number, local_dofs, settings, static_settings, set
):
    """
    Computes the least square function approximation at an integration point.

    Takes the set of functions to fit from static_settings['analytic solution'].

    Args:
      x_i (jnp.ndarray): Coordinates of the integration point.
      w_i (float): Integration weight.
      int_point_number (int): Integration point number.
      local_dofs (jnp.ndarray): Local degrees of freedom.
      settings (dict): Settings for the computation.
      static_settings (dict): Static settings for the computation.
      set (str): Identifier for the current set.

    Returns:
      float: Computed least square function approximation.
    """

    def approximate_solution(x):
        return ss.solution_structure(
            x, int_point_number, local_dofs, settings, static_settings, set
        )

    analytic_solution = static_settings["analytic solution"][set]

    # L2 error
    error = analytic_solution(x_i) - approximate_solution(x_i)
    return w_i * jnp.dot(error, error) / 2
