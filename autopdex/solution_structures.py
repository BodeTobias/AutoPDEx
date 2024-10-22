# solution_structures.py
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
In solving boundary or initial value problems, one generally seeks a function
that satisfies both certain boundary conditions and differential equations.
In solution structures, the boundary conditions (or at least part of them)
are already incorporated into the solution space. This provides greater flexibility
in the choice of variational methods and, depending on the approach, allows for
exact satisfaction of the boundary conditions.

This module provides functions to compute the solution structure at a given
integration point using different modes, as well as to precompile and evaluate discrete
solution structures by evaluating and storing shape functions, boundary conditions,
and projection operators. Precompilation can improve the efficiency,
but one has to consider how many discrete derivatives need to be evaluated for the given solution structure and pde.

The idea of the RFM (Rvachev-function method) solution structures is basically to multiply the solution space :math:`\\phi`
with a smooth function :math:`\\omega` that vanishes on the boundary and to add subsequently a function
:math:`b` that satisfies the boundary conditions: :math:`\\tilde{\\phi} = \\omega \\phi + b`.
The initial approximation :math:`\\phi` can e.g. be a finite element ansatz, but also some approach in which boundary conditions
can not simply nodally imposed, such as smooth meshfree shape functions or a neural network.
The global weighting function :math:`\\omega` can be constructed for example based on Rvachev functions 
that can be used as smooth distance functions to certain primitive elements. The boundary function :math:`b` can be set up
via transfinite interpolation in case there are different conditions on different segments of the boundary. 
The above procedure works well e.g. for a Laplace equation with Dirichlet conditions, but special techniques have to be used for 
more complex problems, see e.g. `Rvachev and Sheiko (1995) <https://doi.org/10.1115/1.3005099>`_. These kind of solution structures 
may be implemented as a user-specific solution structure 'user' (see the source code for examples). 

Here, a general technique is implemented with the solution structure type 'first order set' which can handle 
arbitrary order PDEs with multiple coupled boundary conditions. 
However, the prerequisite is that the problem is reformulated into a set of first-order problems and that a suitable variational 
method is used, such as a norm-equivalent first-order least square formulation 
(see, e.g., `Gunzburger and Bochev (2009) <https://link.springer.com/book/10.1007/b13382>`_). 
This means that not only, for example, the temperature field :math:`\\theta` serves as a primary variable, but also its derivatives, such as
:math:`\\theta_{,X}` and :math:`\\theta_{,t}`. If we gather all the primary fields into the vector :math:`\\boldsymbol{\\phi}`, 
boundary conditions of the form :math:`\\boldsymbol{a}_i \\cdot \\boldsymbol{\\phi} = b_i` can be imposed by the following approach:
:math:`\\tilde{\\boldsymbol{\\phi}} = \\boldsymbol{P} \\cdot \\boldsymbol{\\phi} + \\tilde{\\boldsymbol{b}}`
with the projector field :math:`\\boldsymbol{P} = \\boldsymbol{I} - \\sum_i (1 - \\omega_i) \\boldsymbol{a}_i \\otimes \\boldsymbol{a}_i`,
where :math:`\\boldsymbol{I}` is the identity matrix, :math:`\\omega_i` is a boundary segment selector as a smooth distance funcion and the 
set of :math:`\\boldsymbol{a}_i` have to be orthogonal to each other at the boundary segments with each having a length of 1. 
The boundary function can be set up based on transfinite interpolations based on 
:math:`\\tilde{\\boldsymbol{b}} = \\sum_i b_i \\boldsymbol{a}_i`. An example can be found e.g. in 
`space_time_fos_dirichlet_neumann_robin.py <https://github.com/BodeTobias/AutoPDEx/tree/main/examples/heat_conduction/space_time_fos_dirichlet_neumann_robin.py>`_
"""

import jax
import jax.numpy as jnp

from autopdex import spaces
from autopdex.utility import jit_with_docstring

## Helper functions
@jax.custom_jvp
def _compiled_function_1(x, f, df):
    return f


@_compiled_function_1.defjvp
def _compiled_function_1_jvp(primals, tangents):
    x, f, df = primals
    x_dot, _, _ = tangents
    primal_out = _compiled_function_1(x, f, df)
    tangent_out = df @ x_dot
    return primal_out, tangent_out


compiled_function_1 = jax.jit(jax.vmap(_compiled_function_1, (None, 0, 0), 0))


@jax.custom_jvp
def _compiled_function_2(x, f, df, ddf):
    return f


@_compiled_function_2.defjvp
def _compiled_function_2_jvp(primals, tangents):
    x, f, df, ddf = primals
    x_dot, _, _, _ = tangents
    primal_out = _compiled_function_2(x, f, df, ddf)
    tangent_out = df @ x_dot + (ddf @ x) @ x_dot
    return primal_out, tangent_out


compiled_function_2 = jax.jit(jax.vmap(_compiled_function_2, (None, 0, 0, 0), 0))


### Rvachev function solution structures and precompilation utilities
@jit_with_docstring(static_argnames=["static_settings", "set"])
def solution_structure(x, int_point_number, local_dofs, settings, static_settings, set):
    """
    Computes the solution structure at a given integration point using various predefined schemes.

    Args:
      x (jnp.ndarray): Coordinates of the integration point.
      int_point_number (int): Integration point number.
      local_dofs (jnp.ndarray): Local degrees of freedom.
      settings (dict): Settings for the computation.
      static_settings (dict): Static settings for the computation.
      set (str): Identifier for the current set.

    Returns:
      array: Computed solution structure.

    Notes:
      This function supports different solution structures, including 'dirichlet', 'first order set',
      'second order set', 'nodal imposition', and 'off'. Depending on the shape function mode ('direct'
      or 'compiled'), it evaluates shape functions and computes the solution structure accordingly.

      For 'dirichlet', 'first order set', and 'second order set' structures, boundary conditions and
      projections are applied to the space. The function uses either direct evaluation or precomputed
      shape functions and their derivatives based on the shape function mode.
    """
    if isinstance(local_dofs, dict):
        raise TypeError("solution_structure does currently not support DOFs as dicts.")

    try:
        structure_type = static_settings["solution structure"][set]
    except KeyError:
        print("The solution structure was not set. Continuing with 'off'... ")
        structure_type = "off"

    try:
        shp_fun_mode = static_settings["shape function mode"]
    except KeyError:
        shp_fun_mode = None

    if structure_type == "dirichlet":  ###Todo: obsolet?
        # Direct evaluation of shape functions
        if shp_fun_mode == "direct":
            space = spaces.solution_space(
                x, int_point_number, local_dofs, settings, static_settings, set
            )
            b_condition = static_settings["boundary conditions"][set](x)
            psdf_d = static_settings["psdf"][set](x)
            return psdf_d * space + b_condition

        # Evaluation with precomputed shape functions -> shape function mode
        elif shp_fun_mode == "compiled":
            (f, dx, ddx) = settings["compiled shape functions"][set]
            (bc, dbc, ddbc) = settings["compiled bc"][set]
            (psdf, dpsdf, ddpsdf) = settings["compiled projection"][set]
            shps = compiled_function_2(
                x, f[int_point_number], dx[int_point_number], ddx[int_point_number]
            )
            space = shps @ local_dofs
            bc_x = _compiled_function_2(
                x, bc[int_point_number], dbc[int_point_number], ddbc[int_point_number]
            )
            psdf_x = _compiled_function_2(
                x,
                psdf[int_point_number],
                dpsdf[int_point_number],
                ddpsdf[int_point_number],
            )
            return psdf_x * space + bc_x
        else:
            assert (
                False
            ), "Shape function mode neither specified as 'direct' nor 'compiled'!"

    elif structure_type == "first order set" or structure_type == "second order set":
        # alpha has to be normalized at boundaries
        # consider only orthogonal alphas
        # Direct evaluation of shape functions
        if shp_fun_mode == "direct":
            space = spaces.solution_space(
                x, int_point_number, local_dofs, settings, static_settings, set
            )
            phi_0 = static_settings["boundary conditions"][set](x)
            alpha = static_settings["boundary coefficients"][set](x)
            omega = static_settings["psdf"][set](x)
            n_fields = static_settings["number of fields"][set]

            # Boundary function
            phi = phi_0 @ alpha

            # Projections deletes the influence of space where omega_i are zero in the direction of alpha_i
            def project_i(omega_i, alpha_i):
                return (1 - omega_i) * jnp.outer(alpha_i, alpha_i)

            projection = jnp.identity(n_fields) - jax.vmap(project_i, (0, 0), 0)(
                omega, alpha
            ).sum(axis=0)

            structure = projection @ space + phi
            return structure

        # Evaluation with precomputed shape functions -> shape function mode
        elif shp_fun_mode == "compiled":
            if structure_type == "first order set":
                (f, dx) = settings["compiled shape functions"][set]
                (bc, dbc) = settings["compiled bc"][set]
                (proj, dproj) = settings["compiled projection"][set]

                shps = compiled_function_1(x, f[int_point_number], dx[int_point_number])
                bc_x = _compiled_function_1(
                    x, bc[int_point_number], dbc[int_point_number]
                )
                proj_x = _compiled_function_1(
                    x, proj[int_point_number], dproj[int_point_number]
                )
            elif structure_type == "second order set":
                (f, dx, ddx) = settings["compiled shape functions"][set]
                (bc, dbc, ddbc) = settings["compiled bc"][set]
                (proj, dproj, ddproj) = settings["compiled projection"][set]

                shps = compiled_function_2(
                    x, f[int_point_number], dx[int_point_number], ddx[int_point_number]
                )
                bc_x = _compiled_function_2(
                    x,
                    bc[int_point_number],
                    dbc[int_point_number],
                    ddbc[int_point_number],
                )
                proj_x = _compiled_function_2(
                    x,
                    proj[int_point_number],
                    dproj[int_point_number],
                    ddproj[int_point_number],
                )

            space = shps @ local_dofs
            structure = proj_x @ space + bc_x
            return structure
        else:
            assert (
                False
            ), "Shape function mode neither specified as 'direct' nor 'compiled'!"

    elif structure_type == "nodal imposition" or structure_type == "off":
        # Just forward the space
        if shp_fun_mode == "compiled":
            (f, dx) = settings["compiled shape functions"][set]
            shps = compiled_function_1(x, f[int_point_number], dx[int_point_number])
            space = shps @ local_dofs
            return space
        else:
            return spaces.solution_space(
                x, int_point_number, local_dofs, settings, static_settings, set
            )

    elif structure_type == "user":
        return static_settings["user solution structure"][set](
            x, int_point_number, local_dofs, settings, static_settings, set
        )
    else:
        assert False, "Solution structure type not available."


@jit_with_docstring(static_argnames=["static_settings"])
def precompile(dofs0, settings, static_settings, max_diff_order=1):
    """
    Manually precompiles solution structures by evaluating and storing shape functions, boundary conditions,
    and projection operators for different integration points and domains.

    Args:
      dofs0 (jnp.ndarray): Initial degrees of freedom.
      settings (dict): Settings for the computation.
      static_settings (dict): Static settings for the computation.
      max_diff_order (int, optional): Maximum order of derivatives to precompute. Default is 1.

    Returns:
      dict: Updated settings with precomputed shape functions, boundary conditions, and projections.

    Notes:
      This function loops over all sets of integration points or domains and compiles the necessary
      components for different solution structures. The supported solution structures include:

      - 'first order set': Compiles shape functions, boundary conditions, and projection operators with first derivatives.
      - 'second order set': Compiles shape functions, boundary conditions, and projection operators with second derivatives.
      - 'dirichlet': Compiles shape functions, boundary conditions, and projection operators with second derivatives.
      - 'nodal imposition' or 'off': Compiles shape functions with derivatives up to max_diff_order.

      The precomputed components are stored in the settings dictionary under 'compiled shape functions',
      'compiled bc', and 'compiled projection'.

      Warning: The compiled shape function mode does currently not support implicit differentiation,
      since only the derivatives with respect to the coordinates (and not other quantities in the settings) are set.
    """
    if isinstance(dofs0, dict):
        raise TypeError("precompile does currently not support DOFs as dicts.")

    # Loop over all sets of integration points/ domains
    num_sets = len(static_settings["assembling mode"])
    compiled_shps = ()
    compiled_bcs = ()
    compiled_proj = ()
    for set in range(num_sets):

        # Compilation of different terms in solution structures
        mode = static_settings["solution structure"][set]

        if mode == "first order set":
            b_cond = static_settings["boundary conditions"][set]
            psdf = static_settings["psdf"][set]
            x_int = settings["integration coordinates"][set]
            b_coeff = static_settings["boundary coefficients"][set]

            compiled_shps += (
                spaces.precompute_shape_functions(
                    dofs0, settings, static_settings, set, num_diff=1
                ),
            )
            compiled_bcs += (precompute_coupled_boundary(b_cond, b_coeff, x_int),)
            compiled_proj += (precompute_projection(psdf, b_coeff, x_int),)

        elif mode == "nodal imposition" or mode == "off":
            compiled_shps += (
                spaces.precompute_shape_functions(
                    dofs0, settings, static_settings, set, num_diff=max_diff_order
                ),
            )
            compiled_bcs += (None,)
            compiled_proj += (None,)

        elif mode == "second order set":
            b_cond = static_settings["boundary conditions"][set]
            psdf = static_settings["psdf"][set]
            x_int = settings["integration coordinates"][set]
            b_coeff = static_settings["boundary coefficients"][set]

            compiled_shps += (
                spaces.precompute_shape_functions(
                    dofs0, settings, static_settings, set, num_diff=2
                ),
            )
            compiled_bcs += (
                precompute_coupled_boundary(b_cond, b_coeff, x_int, num_diff=2),
            )
            compiled_proj += (precompute_projection(psdf, b_coeff, x_int, num_diff=2),)

        elif mode == "dirichlet":
            b_cond = static_settings["boundary conditions"][set]
            psdf = static_settings["psdf"][set]
            x_int = settings["integration coordinates"][set]

            compiled_shps += (
                spaces.precompute_shape_functions(
                    dofs0, settings, static_settings, set, num_diff=2
                ),
            )
            compiled_bcs += (
                precompute_function_and_derivatives(b_cond, x_int, num_diff=2),
            )
            compiled_proj += (
                precompute_function_and_derivatives(psdf, x_int, num_diff=2),
            )

        else:
            assert (
                False
            ), "There is no default compilation defined for the specified solution structure!"

    settings["compiled shape functions"] = compiled_shps
    settings["compiled bc"] = compiled_bcs
    settings["compiled projection"] = compiled_proj

    jax.debug.print("Compilation of solution structure finished.")
    return settings


@jit_with_docstring(static_argnames=["fun", "num_diff"])
def precompute_function_and_derivatives(fun, x_int, num_diff=1):
    """
    Precomputes a function and its derivatives up to a specified order at given integration points.

    Args:
      fun (callable): The function to precompute.
      x_int (jnp.ndarray): Integration points.
      num_diff (int, optional): Number of derivatives to compute (0, 1, or 2). Default is 1.

    Returns:
      tuple: Precomputed function values and derivatives.
        - If num_diff == 0: (fun_i,)
        - If num_diff == 1: (fun_i, dfun_i)
        - If num_diff == 2: (fun_i, dfun_i, ddfun_i)
    """
    fun_i = jax.jit(jax.vmap(fun, (0), 0))(x_int)
    if num_diff == 0:
        return fun_i

    dfun_i = jax.jit(jax.vmap(jax.jacrev(fun), (0), 0))(x_int)
    if num_diff == 1:
        return (fun_i, dfun_i)

    ddfun_i = jax.jit(jax.vmap(jax.jacfwd(jax.jacrev(fun)), (0), 0))(x_int)
    if num_diff == 2:
        return (fun_i, dfun_i, ddfun_i)

    assert False, "Number of differentiations not implemented!"


@jit_with_docstring(static_argnames=["phi_0_fun", "alpha_fun", "num_diff"])
def precompute_coupled_boundary(phi_0_fun, alpha_fun, x_int, num_diff=1):
    """
    Precomputes coupled boundary conditions and their derivatives up to a specified order at given integration points.

    Args:
      phi_0_fun (callable): Boundary condition function.
      alpha_fun (callable): Coefficient function.
      x_int (jnp.ndarray): Integration points.
      num_diff (int, optional): Number of derivatives to compute (0, 1, or 2). Default is 1.

    Returns:
      tuple: Precomputed boundary condition values and derivatives.
        - If num_diff == 0: (fun_i,)
        - If num_diff == 1: (fun_i, dfun_i)
        - If num_diff == 2: (fun_i, dfun_i, ddfun_i)
    """

    def phi(x):
        phi_0 = phi_0_fun(x)
        alpha = alpha_fun(x)
        return phi_0 @ alpha

    fun = phi
    fun_i = jax.jit(jax.vmap(fun, (0), 0))(x_int)
    if num_diff == 0:
        return fun_i

    dfun_i = jax.jit(jax.vmap(jax.jacrev(fun), (0), 0))(x_int)
    if num_diff == 1:
        return (fun_i, dfun_i)

    ddfun_i = jax.jit(jax.vmap(jax.jacfwd(jax.jacrev(fun)), (0), 0))(x_int)
    if num_diff == 2:
        return (fun_i, dfun_i, ddfun_i)


@jit_with_docstring(static_argnames=["psdf_fun", "alpha_fun", "num_diff"])
def precompute_projection(psdf_fun, alpha_fun, x_int, num_diff=1):
    """
    Precomputes the projection operator and its derivatives up to a specified order at given integration points.

    Args:
      psdf_fun (callable): Projection function.
      alpha_fun (callable): Coefficient function.
      x_int (jnp.ndarray): Integration points.
      num_diff (int, optional): Number of derivatives to compute (0, 1, or 2). Default is 1.

    Returns:
      tuple: Precomputed projection values and derivatives.
        - If num_diff == 0: (fun_i,)
        - If num_diff == 1: (fun_i, dfun_i)
        - If num_diff == 2: (fun_i, dfun_i, ddfun_i)
    """

    def projection(x):
        alpha = alpha_fun(x)
        omega = psdf_fun(x)
        n_fields = alpha.shape[-1]

        def project_i(omega_i, alpha_i):
            return (1 - omega_i) * jnp.outer(alpha_i, alpha_i)

        return jnp.identity(n_fields) - jax.vmap(project_i, (0, 0), 0)(
            omega, alpha
        ).sum(axis=0)

    fun = projection
    fun_i = jax.vmap(fun, (0), 0)(x_int)
    if num_diff == 0:
        return fun_i

    dfun_i = jax.jit(jax.vmap(jax.jacrev(fun), (0), 0))(x_int)
    if num_diff == 1:
        return (fun_i, dfun_i)

    ddfun_i = jax.jit(jax.vmap(jax.jacfwd(jax.jacrev(fun)), (0), 0))(x_int)
    if num_diff == 2:
        return (fun_i, dfun_i, ddfun_i)

