# solver.py
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
This module is the central module of the analysis phase.
Based on the given entries in settings and static_settings, the functions solver and adaptive_load_stepping can be used to find the roots of the global residual vector. 
Depending on the settings, linear equation solvers, the Newton-Raphson method, or nonlinear minimizers are utilized. 
The residual vectors and (in the case of external solvers) the tangent matrix are automatically assembled according to the chosen settings. 
The solver module uses the assembler, which in turn calls the variational_scheme, the solution_structure, and the spaces modules. 
Additionally, automatic implicit differentiation in forward or reverse mode via the implicit_diff module is provided for the adaptive_load_stepping function.
For solving the linear equation systems, wrappers for different backends on CPU and GPU are available, including Pardiso and PETSc.
"""

import time
import sys

import jax
import jaxopt
import jax.numpy as jnp
from jax.experimental import sparse
from jax import lax
import numpy as np
import scipy as scp
from flax.core import FrozenDict

from autopdex import assembler, implicit_diff, utility



### Solvers as specified by the static_settings and settings
@utility.jit_with_docstring(static_argnames=["static_settings", "**kwargs"])
def solver(dofs, settings, static_settings, **kwargs):
    """
    General solver function to solve a given problem based on provided settings.

    This function chooses and runs the appropriate solver type (e.g., minimization, linear, Newton) based on the provided
    `static_settings` and returns the solution and any additional information.

    Args:
        dofs (jnp.ndarray or dict): Degrees of freedom or initial guess for the solution.
        settings (dict): Dictionary containing various settings and parameters required for assembling the problem.
        static_settings (dict): Dictionary containing static settings such as solver type, verbose level, and variational schemes.
        **kwargs (dict): Additional keyword arguments passed to the specific solver functions.

    Returns:
        jnp.ndarray: The solution obtained from the selected solver.
        Any: Additional information from the solver, such as number of iterations or convergence status.

    Solver Types:
      - 'minimize' : Uses nonlinear minimization solvers (e.g., LBFGS, BFGS, etc.).
      - 'linear' : Solves linear systems using specified backend (e.g., JAX, PETSc, PARDISO, PyAMG, Scipy).
      - 'diagonal linear' : Solves linear systems assuming a diagonal tangent matrix.
      - 'newton' : Uses the Newton method for solving nonlinear systems.
      - 'damped newton' : Uses a damped Newton method for solving nonlinear systems.

    For the different solvers, the function conducts the following functions in which more documentation is provided:
      - 'minimize' : solver.solve_nonlinear_minimization
      - 'linear' : solver.solve_linear
      - 'diagonal linear' : solver.solve_diagonal_linear
      - 'newton' : solver.solve_newton
      - 'damped newton' : solver.solve_damped_newton

    Notes:
      - If all domains are using 'least square pde loss' variational scheme and verbosity level is >=1, it prints the L2 error before and after optimization.
    """

    # Give global error estimator with unfitted dofs if all domains are 'least square pde loss'
    verbose = static_settings["verbose"]
    try:
        if (
            all(
                [
                    scheme == "least square pde loss"
                    for scheme in static_settings["variational scheme"]
                ]
            )
            and verbose >= 1
        ):
            jax.debug.print(
                "L2 error unoptimized: {x}",
                x=assembler.integrate_functional(dofs, settings, static_settings),
            )
    except KeyError:
        pass

    # Choose type of solver and call it
    solver_type = static_settings["solver type"]
    match solver_type:
        case "minimize":
            sol = solve_nonlinear_minimization(
                dofs, settings, static_settings, **kwargs
            )
            infos = None
        case "linear":
            sol = solve_linear(dofs, settings, static_settings, **kwargs)
            infos = None
        case "diagonal linear":
            sol = solve_diagonal_linear(dofs, settings, static_settings, **kwargs)
            infos = None
        # case 'diagonal newton':
        #   # ToDo
        #   sol, infos = solve_diagonal_newton(dofs, settings, static_settings, **kwargs)
        case "newton":
            sol, infos = solve_newton(dofs, settings, static_settings, **kwargs)
        case "damped newton":
            sol, infos = solve_damped_newton(dofs, settings, static_settings, **kwargs)
        case _:
            assert False, "Solver type not implemented!"

    # Give global error estimator with fitted dofs if all domains are 'least square pde loss'
    try:
        if (
            all(
                [
                    scheme == "least square pde loss"
                    for scheme in static_settings["variational scheme"]
                ]
            )
            and verbose >= 1
        ):
            jax.debug.print(
                "L2 error optimized: {x}",
                x=assembler.integrate_functional(sol, settings, static_settings),
            )
    except KeyError:
        pass
    return sol, infos

@utility.jit_with_docstring(
    static_argnames=[
        "static_settings",
        "multiplier_settings",
        "path_dependent",
        "implicit_diff_mode",
        "max_load_steps",
        "max_multiplier",
        "min_increment",
        "max_increment",
        "init_increment",
        "target_num_newton_iter",
        "newton_tol",
        "**kwargs",
    ]
    )
def adaptive_load_stepping(
    dofs,
    settings,
    static_settings,
    multiplier_settings=lambda settings, multiplier: (
        settings.update({"load multiplier": multiplier}),
        settings,
    )[1],
    path_dependent=True,
    implicit_diff_mode=None,
    max_multiplier=1.0,
    min_increment=0.01,
    max_increment=1.0,
    init_increment=0.2,
    max_load_steps=1000,
    target_num_newton_iter=7,
    newton_tol=1e-10,
    **kwargs,
    ):
    """
    Performs adaptive load stepping to solve a nonlinear system of equations.

    This function iteratively adjusts the load increment to ensure convergence
    using a Newton-Raphson solver. The increment size is adaptively controlled
    based on the convergence behavior of the solver. Works currently only with
    solver types 'newton' and 'damped newton'.

    Args:
      dofs (jnp.ndarray or dict): Initial degrees of freedom.
      settings (dict): Dictionary of problem settings.
      static_settings (dict): Dictionary of static settings that do not change during load steps.
      multiplier_settings (callable): Function to update settings based on the current load multiplier.
      path_dependent (bool): Specifies wether problem is path-dependent (experimental) or not (has an influence on the implicit differentiation).
      implicit_diff_mode (string): Can be either \'reverse\', \'forward\' or None. In case of \'reverse\', only reverse mode differentiation is supported (jacrev), in case of \'forward\', only forward mode differentiation is supported (jacfwd).
      max_multiplier (float): Maximum value for the load multiplier.
      min_increment (float): Minimum allowable increment size.
      max_increment (float): Maximum allowable increment size.
      init_increment (float): Initial increment size.
      max_load_steps (int): Maximal number of load steps. Only used in case implicit_diff_mod is not None
      target_num_newton_iter (int): Target number of Newton iterations for each load step.
      newton_tol (float, optional): Tolerance for Newton solver convergence. Default is 1e-10.
      **kwargs: Additional keyword arguments for the solver.

    Returns:
        - jnp.ndarray: Solution degrees of freedom after load stepping.
    """
    verbose = static_settings["verbose"]

    if implicit_diff_mode is not None:
        # Set-up the decorator for implicit differentiation
        residual_fun = lambda dofs, settings: assembler.assemble_residual(dofs, settings, static_settings)
        tangent_fun = lambda dofs, settings: assembler.assemble_tangent(dofs, settings, static_settings)

        try:
            dirichlet_dofs = settings["dirichlet dofs"]
        except KeyError:
            # Warning if it was defined in static_settings
            assert "dirichlet dofs" not in static_settings, \
                "'dirichlet dofs' has been moved to 'settings' in order to reduce compile time. \
                Further, you should not transform it to a tuple of tuples anymore."
            pass
        free_dofs = None
        free_dofs_flat = None
        if dirichlet_dofs is not None:
            free_dofs = utility.mask_op(dirichlet_dofs, utility.dict_ones_like(dirichlet_dofs), mode="apply", ufunc=lambda x: ~x)
            free_dofs_flat = utility.dict_flatten(free_dofs)
        is_constrained = True if free_dofs is not None else False

        solver_backend = static_settings["solver backend"]
        solver_subtype = static_settings["solver"]
        try:
            sensitivity_solver_backend = static_settings["sensitivity solver backend"]
            sensitivity_solver_subtype = static_settings["sensitivity solver"]
        except KeyError:
            sensitivity_solver_backend = solver_backend
            sensitivity_solver_subtype = solver_subtype

        match sensitivity_solver_backend:
            case "petsc":
                n_fields = static_settings["number of fields"]
                pc_type = static_settings["type of preconditioner"]
                lin_solve_fun = lambda mat, rhs, free_dofs_flat: linear_solve_petsc(
                    mat,
                    rhs,
                    n_fields,
                    sensitivity_solver_subtype,
                    pc_type,
                    verbose,
                    free_dofs_flat,
                    **kwargs,
                )
            case "pardiso":
                lin_solve_fun = lambda mat, rhs, free_dofs_flat: linear_solve_pardiso(
                    mat,
                    rhs,
                    sensitivity_solver_subtype,
                    verbose,
                    free_dofs_flat,
                    **kwargs,
                )
            case "pyamg":
                pc_type = static_settings["type of preconditioner"]
                lin_solve_fun = lambda mat, rhs, free_dofs_flat: linear_solve_pyamg(
                    mat,
                    rhs,
                    sensitivity_solver_subtype,
                    pc_type,
                    verbose,
                    free_dofs_flat,
                    **kwargs,
                )
            case "scipy":
                lin_solve_fun = lambda mat, rhs, free_dofs_flat: linear_solve_scipy(
                    mat,
                    rhs,
                    sensitivity_solver_subtype,
                    verbose,
                    free_dofs_flat,
                    **kwargs,
                )
            case _:
                raise ValueError(
                    "Specified sensitivity solver backend not available. Choose 'pardiso', 'petsc', 'pyamg' or 'scipy'."
                )

        def lin_solve_callback_fun(mat, rhs, free_dofs_flat):
            rhs_flat = utility.dict_flatten(rhs)
            sol = jax.pure_callback(lin_solve_fun, jnp.zeros(rhs_flat.shape, rhs_flat.dtype), mat, rhs_flat, free_dofs_flat, vmap_method='sequential')
            return utility.reshape_as(sol, rhs)

        # def lin_solve_callback_fun(mat, rhs, free_dofs_flat):
        #     rhs_flat = utility.dict_flatten(rhs)
        #     # linear_solver = lambda mat, rhs: jax.pure_callback(lin_solve_fun, jnp.zeros(rhs_flat.shape, rhs_flat.dtype), mat, rhs_flat, None, vmap_method='sequential')
        #     linear_solver = lambda mat, rhs: jax.pure_callback(lin_solve_fun, jnp.zeros(rhs_flat.shape, rhs_flat.dtype), mat, rhs_flat, free_dofs_flat, vmap_method='sequential')
        #     solve = lambda matvec, x: linear_solver(mat, x)
        #     transpose_solve = lambda vecmat, x: linear_solver(mat.T, x)
        #     matvec = lambda x: utility.reshape_as(mat @ utility.dict_flatten(x), x)
        #     sol = jax.lax.custom_linear_solve(matvec, rhs_flat, solve, transpose_solve)
        #     return utility.reshape_as(sol, rhs)

    # Set up functions for adaptive load stepping loop
    def continue_check(carry):
        _, multiplier, increment, _, _, _ = carry
        _continue_1 = jnp.logical_and(
            multiplier < max_multiplier, increment > min_increment
        )
        jax.lax.cond(
            jnp.logical_and(
                multiplier < max_multiplier - min_increment, increment < min_increment
            ),
            lambda: jax.debug.print(
                "Adaptive load stepping could not converge; increment size below min_increment!"
            ),
            lambda: None,
        )
        return _continue_1

    def step(carry):
        dofs0, multiplier, increment, load_step, settings, _ = carry

        # Update multiplier
        multiplier += increment

        # Update boundary conditions
        if verbose > -1:
            if verbose > 0:
                jax.debug.print("")
            jax.debug.print("Multiplier: {x}", x=multiplier)
        settings = multiplier_settings(settings, multiplier)

        # Call newton solver
        if (
            path_dependent and implicit_diff_mode is not None
        ):  # Add implicit differentiation for each load step

            @implicit_diff.custom_root(
                residual_fun,
                tangent_fun,
                lin_solve_callback_fun,
                is_constrained,
                True,
                implicit_diff_mode,
            )
            def diffable_solve(dofs0, settings):
                dofs, (needed_steps, res_norm_free_dofs, divergence) = solver(
                    dofs0, settings, static_settings, newton_tol=newton_tol, **kwargs
                )
                return dofs, (needed_steps.astype(float), res_norm_free_dofs, divergence.astype(float))

            dofs, infos = diffable_solve(dofs0, settings)
            needed_steps, res_norm_free_dofs, divergence = infos
            divergence = divergence.astype(bool)


        else:  # Add implicit diff wrapper on the adaptive load stepping level
            dofs, infos = solver(
                dofs0, settings, static_settings, newton_tol=newton_tol, **kwargs
            )
            needed_steps, res_norm_free_dofs, divergence = infos

        # Adaptive incrementation
        multiplier = jnp.where(divergence, multiplier - increment, multiplier)
        increment = jnp.where(
            divergence,
            0.5 * increment,
            (1 + 0.5 * (target_num_newton_iter - needed_steps) / target_num_newton_iter)
            * increment,
        )
        increment = jnp.where(increment > max_increment, max_increment, increment)

        if isinstance(dofs, dict):
            dofs = {
                key: jnp.where(divergence, dofs0[key], dofs[key])
                for key in dofs0.keys()
            }
        else:
            dofs = jnp.where(divergence, dofs0, dofs)

        # Limit to max_multiplier
        increment = jnp.where(
            multiplier + increment > max_multiplier,
            max_multiplier - multiplier,
            increment,
        )
        return (dofs, multiplier, increment, 1.0 * load_step, settings, res_norm_free_dofs)

    # Use implicit diff wrappers to make it differentiable
    if implicit_diff_mode is not None:
        if not path_dependent:  # Conservative problem

            # Set Dirichlet conditions for derivative w.r.t. them
            settings = multiplier_settings(settings, max_multiplier)

            @implicit_diff.custom_root(
                residual_fun,
                tangent_fun,
                lin_solve_callback_fun,
                is_constrained,
                True,
                implicit_diff_mode,
            )
            def diffable_adaptive_load_stepping(dofs, settings):
                (dofs, multiplier, increment, load_step, settings, res_norm_free_dofs) = \
                    jax.lax.while_loop(
                        continue_check, step, (dofs, 0.0, init_increment, 0, settings, 0.0)
                        )
                return dofs, (multiplier, increment, load_step, res_norm_free_dofs)

            dofs, (multiplier, increment, load_step, res_norm_free_dofs) = \
                diffable_adaptive_load_stepping(dofs, settings)
            return dofs, (multiplier, increment, load_step, settings, res_norm_free_dofs)

        else:  # Pathdependent problem; uses fori_loop with static limits for supporting reverse mode differentiation

            # Set Dirichlet conditions for derivative w.r.t. them
            settings = multiplier_settings(settings, max_multiplier)

            def body_fn(i, carry):
                def continue_check(carry):
                    _, multiplier, increment, _, _, divergence, stop = carry
                    _continue_1 = jnp.logical_and(
                        multiplier < max_multiplier, increment > min_increment
                    )
                    jax.lax.cond(
                        jnp.logical_and(
                            jnp.logical_and(
                                multiplier < max_multiplier, increment < min_increment
                            ),
                            jnp.logical_not(stop),
                        ),
                        lambda: jax.debug.print(
                            "Adaptive load stepping could not converge; increment size below min_increment!"
                        ),
                        lambda: None,
                    )
                    return _continue_1

                def step_extended(carry):
                    (dofs, multiplier, increment, load_step, settings, res_norm_free_dofs, stop) = carry
                    args = (dofs, multiplier, increment, load_step, settings, res_norm_free_dofs)
                    return step(args) + (False,)

                def finish(carry):
                    (dofs, multiplier, increment, load_step, settings, res_norm_free_dofs, stop) = carry
                    return (dofs, multiplier, increment, load_step, settings, res_norm_free_dofs, True)

                carry = jax.lax.cond(
                    continue_check(carry),
                    lambda x: step_extended(x),
                    lambda x: finish(x),
                    carry,
                )

                # ToDo: Verify accuracy of derivatives with finite differences.
                return carry

            init_state = (dofs, 0.0, init_increment, 0., settings, 0.0, False)
            return jax.lax.fori_loop(0, max_load_steps, body_fn, init_state)

    else:  # No definition of implicit derivatives
        return jax.lax.while_loop(
            continue_check, step, (dofs, 0.0, init_increment, 0., settings, 0.0)
        )

### Minimizers
@utility.jit_with_docstring(static_argnames=["static_settings", "**kwargs"])
def solve_nonlinear_minimization(dofs, settings, static_settings, **kwargs):
    """
    Solves a nonlinear minimization problem using specified optimization methods.

    This function wraps nonlinear minimization solvers provided by `jaxopt` to minimize a functional
    and solve the given problem.

    Args:
      dofs (jnp.ndarray or dict): Degrees of freedom or initial guess for the solution.
      settings (dict): Dictionary containing various settings and parameters required for assembling the problem.
      static_settings (dict): Dictionary containing static settings such as solver type, verbose level, and variational schemes.
      **kwargs (dict): Additional keyword arguments passed to the specific solver functions.

    Returns:
      jnp.ndarray: The optimized solution obtained from the selected solver.

    Solver Types
      - 'gradient descent' : Uses gradient descent for optimization.
      - 'lbfgs' : Uses Limited-memory Broyden-Fletcher-Goldfarb-Shanno (LBFGS) algorithm for optimization.
      - 'bfgs' : Uses Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm for optimization.
      - 'nonlinear cg' : Uses nonlinear conjugate gradient method for optimization.
      - 'gauss newton' : Uses Gauss-Newton method for optimization.
      - 'levenberg marquart' : Uses Levenberg-Marquardt algorithm for optimization.
      - Default : If solver name is not set or not available, uses 'lbfgs' as the default solver.

    Notes:
      - This function should just be called, if the variational scheme involves the definition of a functional that is to be minimized,
        e.g. 'least square pde loss'. The modes 'gauss newton' and 'levenberg marquart' are an exeption, since they utilize the residual.
      - The function conducts the assembler.integrate_functional and assembler.assemble_residual functions in order
        to set up suitable optimization functions or residual functions, depending on what the solver needs.
      - The current implementation does not support nodal imposition of DOFs.
    """
    nodal_imposition = "nodal imposition" in static_settings["solution structure"]
    assert (
        not nodal_imposition
    ), "solver type 'minimize' does currently not support nodal imposition of DOFs."
    # ToDo: impose boundary conditions and freeze dirichlet dofs

    def functional(params):
        return assembler.integrate_functional(params, settings, static_settings)

    def residual_function(params):
        return assembler.assemble_residual(params, settings, static_settings)

    # Select minimizer
    solver_name = static_settings["solver"]
    match solver_name:
        case "gradient descent":
            solver = jaxopt.GradientDescent(functional, **kwargs)
            sol = solver.run(dofs).params
        case "lbfgs":
            solver = jaxopt.LBFGS(functional, **kwargs)
            sol = solver.run(dofs).params
        case "bfgs":
            solver = jaxopt.BFGS(functional, **kwargs)
            sol = solver.run(dofs).params
        case "nonlinear cg":
            solver = jaxopt.NonlinearCG(functional, **kwargs)
            sol = solver.run(dofs).params
        case "gauss newton":
            solver = jaxopt.GaussNewton(residual_function, **kwargs)
            sol = solver.run(dofs).params
        case "levenberg marquart":
            assert not isinstance(
                dofs, dict
            ), "solver 'levenberg marquart' does currently not support dicts as dofs"

            def residual_function_flat(params):
                return assembler.assemble_residual(
                    jnp.reshape(params, dofs.shape), settings, static_settings
                ).flatten()

            solver = jaxopt.LevenbergMarquardt(residual_function_flat, **kwargs)
            sol = jnp.reshape(solver.run(dofs.flatten()).params, dofs.shape)
        case _:
            solver = jaxopt.LBFGS(functional, **kwargs)
            sol = solver.run(dofs).params
            print(
                "Solver name not set or not available in combination with this solver type. Using static_settings['solver name'] = 'lbfgs' as default."
            )
    return sol

### Root finders
@utility.jit_with_docstring(static_argnames=["static_settings", "**kwargs"])
def solve_linear(dofs, settings, static_settings, **kwargs):
    """
    Solves a linear system using the specified backend and solver settings.

    This function determines the appropriate linear solver based on the provided settings and forwards the
    call to the selected solver function. It supports both JAX matrix-free solvers and external solvers
    like PETSc, PARDISO, PyAMG, and Scipy using jax.pure_callback.

    Args:
      dofs (jnp.ndarray or dict): Degrees of freedom or initial guess for the solution.
      settings (dict): Dictionary containing various settings and parameters required for assembling the problem.
      static_settings (dict): Dictionary containing static settings such as solver backend, type of solver, and preconditioner.
      **kwargs (dict): Additional keyword arguments passed to the specific solver functions.

    Returns:
      jnp.ndarray: The solution obtained from the selected linear solver.

    Solver Backends:
      - 'jax' : Uses JAX's matrix-free solvers.
      - 'petsc' : Uses PETSc for solving linear systems.
      - 'pardiso' : Uses PARDISO for solving linear systems.
      - 'pyamg' : Uses PyAMG for solving linear systems.
      - 'scipy' : Uses Scipy's sparse solvers for solving linear systems.

    Notes:
      - If `nodal imposition` is detected in the `static_settings`, the function imposes Dirichlet boundary conditions
        and adjusts the degrees of freedom accordingly.
      - The function assembles the residual and tangent matrix before solving the system in case an external solver is used.
      - If the tangent matrix is dense and an external solver is used, it is converted to a sparse format.
      - The function uses JAX's `pure_callback` to call external solvers and handle the solution.
    """
    solver_backend = static_settings["solver backend"]

    ### Jax matrix-free solver
    if solver_backend == "jax":
        return linear_solve_jax(dofs, settings, static_settings, **kwargs)

    ### External linear solver

    nodal_imposition = "nodal imposition" in static_settings["solution structure"]
    # Impose nodal dofs
    if nodal_imposition:
        dirichlet_conditions = settings["dirichlet conditions"]

        if isinstance(settings["dirichlet dofs"], (dict, FrozenDict)):
            dirichlet_dofs_dict_flat = {
                key: jnp.asarray(val).flatten()
                for key, val in settings["dirichlet dofs"].items()
            }
            dirichlet_dofs_flat = jnp.concatenate(
                list(dirichlet_dofs_dict_flat.values())
            )
            dofs = utility.mask_op(dofs, dirichlet_dofs_dict_flat, dirichlet_conditions)
        else:
            dirichlet_dofs_flat = jnp.asarray(
                settings["dirichlet dofs"]
            ).flatten()
            dofs = utility.mask_op(dofs, dirichlet_dofs_flat, dirichlet_conditions)

        free_dofs_flat = jnp.invert(dirichlet_dofs_flat)

    # Assembling
    verbose = static_settings["verbose"]
    solver = static_settings["solver"]

    rhs = assembler.assemble_residual(dofs, settings, static_settings)
    rhs = -utility.dict_flatten(rhs)
    mat = assembler.assemble_tangent(dofs, settings, static_settings)

    # If dense matrix and external solver, convert to sparse
    if (
        solver_backend in ("petsc", "pardiso", "pyamg", "scipy")
        and type(mat) == jnp.ndarray
    ):
        mat = sparse.bcoo_fromdense()

    match solver_backend:
        case "petsc":
            n_fields = static_settings["number of fields"]
            pc_type = static_settings["type of preconditioner"]
            solve_fun = lambda a, b, c: linear_solve_petsc(
                a, b, n_fields, solver, pc_type, verbose, free_dofs=c, **kwargs
            )
        case "pardiso":
            solve_fun = lambda a, b, c: linear_solve_pardiso(
                a, b, solver, verbose, free_dofs=c
            )
        case "pyamg":
            pc_type = static_settings["type of preconditioner"]
            solve_fun = lambda a, b, c: linear_solve_pyamg(
                a, b, solver, pc_type, verbose, free_dofs=c, **kwargs
            )
        case "scipy":
            solve_fun = lambda a, b, c: linear_solve_scipy(
                a, b, solver, verbose, free_dofs=c
            )

    # Prepare callback
    result_shape_dtype = jax.ShapeDtypeStruct(shape=rhs.shape, dtype=rhs.dtype)

    # Compose solution dofs
    if nodal_imposition:
        sol = jax.pure_callback(solve_fun, result_shape_dtype, mat, rhs, free_dofs_flat, vmap_method='sequential')
        if isinstance(settings["dirichlet dofs"], (dict, FrozenDict)):
            free_dofs_dict = {
                key: jnp.invert(jnp.asarray(val))
                for key, val in settings["dirichlet dofs"].items()
            }
            sol = utility.reshape_as(sol, dofs)
            return utility.mask_op(dofs, free_dofs_dict, sol)
        else:
            return utility.mask_op(dofs, free_dofs_flat, sol)
    else:
        sol = jax.pure_callback(solve_fun, result_shape_dtype, mat, rhs, None, vmap_method='sequential')
        return utility.reshape_as(sol, dofs)

@utility.jit_with_docstring(static_argnames=["static_settings", "**kwargs"])
def solve_diagonal_linear(dofs, settings, static_settings, **kwargs):
    """
    Solves a linear system assuming the tangent matrix is diagonal.

    This function solves the linear system by leveraging the assumption that the tangent matrix is diagonal,
    which simplifies the solution process. It supports nodal imposition of Dirichlet boundary conditions and
    handles the assembly of the residual and diagonal tangent matrix.

    If the tangent matrix is not diagonal, it will produce a wrong diagonal of the tangent!

    Args:
      dofs (jnp.ndarray or dict): Degrees of freedom or initial guess for the solution.
      settings (dict): Dictionary containing various settings and parameters required for assembling the problem.
      static_settings (dict): Dictionary containing static settings such as solution structure and solver backend.
      **kwargs (dict): Additional keyword arguments passed to the solver function.

    Returns:
      sol (jnp.ndarray): The solution obtained by solving the linear system assuming a diagonal tangent matrix.

    Notes:
      - If `nodal imposition` is detected in the `static_settings`, the function imposes Dirichlet boundary conditions
        and adjusts the degrees of freedom accordingly.
      - The function assembles the residual and diagonal tangent matrix before solving the system.
      - The solution process involves element-wise division of the residual by the diagonal elements of the tangent matrix.
    """
    nodal_imposition = "nodal imposition" in static_settings["solution structure"]

    # Impose nodal dofs
    if nodal_imposition:
        dirichlet_conditions = utility.dict_flatten(settings["dirichlet conditions"])

        if isinstance(settings["dirichlet dofs"], (dict, FrozenDict)):
            dirichlet_dofs_dict_flat = {
                key: jnp.asarray(val).flatten()
                for key, val in settings["dirichlet dofs"].items()
            }
            dirichlet_dofs_flat = jnp.concatenate(
                list(dirichlet_dofs_dict_flat.vlaues())
            )
            dofs = utility.mask_op(
                dofs, dirichlet_dofs_dict_flat, dirichlet_conditions
            )
        else:
            dirichlet_dofs_flat = jnp.asarray(
                settings["dirichlet dofs"]
            ).flatten()
            dofs = utility.mask_op(dofs, dirichlet_dofs_flat, dirichlet_conditions)

        free_dofs_flat = jnp.invert(dirichlet_dofs_flat)

    # Assembling
    rhs = -assembler.assemble_residual(dofs, settings, static_settings)
    diag_mat = assembler.assemble_tangent(dofs, settings, static_settings).data

    # Delete rows
    rhs = utility.dict_flatten(rhs)
    if nodal_imposition:
        # Delete rows
        rhs = rhs[free_dofs_flat]
        diag_mat = diag_mat[free_dofs_flat]

    # Solve while assuming a diagonal tangent
    sol = jnp.multiply((1 / diag_mat), rhs)

    # Compose solution dofs
    if nodal_imposition:
        if isinstance(settings["dirichlet dofs"], (dict, FrozenDict)):
            free_dofs_dict = {
                key: jnp.invert(jnp.asarray(val))
                for key, val in settings["dirichlet dofs"].items()
            }
            sol = utility.reshape_as(sol, dofs)
            return utility.mask_op(dofs, free_dofs_dict, sol)
        else:
            return utility.mask_op(dofs, free_dofs_flat, sol)
    else:
        return utility.reshape_as(sol, dofs)

@utility.jit_with_docstring(static_argnames=["static_settings", "**kwargs"])
def solve_newton(
    dofs, settings, static_settings, newton_tol=1e-8, maxiter=30, **kwargs
    ):
    """
    Solves a nonlinear system using the Newton-Raphson method.

    This function is a wrapper for the damped Newton method with a damping coefficient of 1.0,
    effectively performing standard Newton-Raphson iterations.

    Args:
      dofs (jnp.ndarray or dict): Degrees of freedom or initial guess for the solution.
      settings (dict): Dictionary containing various settings and parameters required for assembling the problem.
      static_settings (dict): Dictionary containing static settings such as solution structure and solver backend.
      newton_tol (float, optional): Tolerance for the Newton method convergence criterion. Default is 1e-8.
      maxiter (int, optional): Maximum number of iterations for the Newton method. Default is 30.
      **kwargs (dict): Additional keyword arguments passed to the solver function.

    Returns:
      tuple: A tuple containing the following elements:
        - sol (jnp.ndarray): The solution obtained by solving the nonlinear system using the Newton method.
        - infos (tuple): Additional information about the solution process, including:
            - num_iterations (int): The number of iterations performed.
            - residual_norm (float): The norm of the residual at the solution.
            - diverged (bool): Flag indicating whether the method diverged.
    """
    return solve_damped_newton(
        dofs, settings, static_settings, newton_tol, 1.0, maxiter
    )

@utility.jit_with_docstring(static_argnames=["static_settings", "**kwargs"])
def solve_damped_newton(
    dofs,
    settings,
    static_settings,
    newton_tol=1e-8,
    damping_coefficient=0.8,
    maxiter=30,
    **kwargs,
    ):
    """
    Solves a nonlinear system using the damped Newton method.

    This function performs damped Newton iterations to solve a nonlinear system,
    with support for nodal imposition of Dirichlet boundary conditions specified as a boolean tuple-tree in static_settings['dirichlet dofs'].

    In this function the information needed for solver.damped_newton is prepared and the function is then called.

    Args:
      dofs (jnp.ndarray or dict): Degrees of freedom or initial guess for the solution.
      settings (dict): Dictionary containing various settings and parameters required for assembling the problem.
      static_settings (dict): Dictionary containing static settings such as solution structure and solver backend.
      newton_tol (float, optional): Tolerance for the Newton method convergence criterion. Default is 1e-8.
      damping_coefficient (float, optional): Damping coefficient for the Newton updates. Default is 0.8.
      maxiter (int, optional): Maximum number of iterations for the Newton method. Default is 30.
      **kwargs (dict): Additional keyword arguments passed to the solver function.

    Returns:
      tuple: A tuple containing the following elements:
        - sol (jnp.ndarray): The solution obtained by solving the nonlinear system using the damped Newton method.
        - infos (tuple): Additional information about the solution process, including:
            - num_iterations (int): The number of iterations performed.
            - residual_norm (float): The norm of the residual at the solution.
            - diverged (bool): Flag indicating whether the method diverged.
    """
    residual_fun = lambda x_i: assembler.assemble_residual(
        x_i, settings, static_settings
    )
    lin_solve_fun = lambda x_i: solve_linear(x_i, settings, static_settings, **kwargs)

    nodal_imposition = "nodal imposition" in static_settings["solution structure"]
    if nodal_imposition:
        # Impose Dirichlet boundaries. Dirichlet dofs has to be concrete, therefore it is passed in static_settings as tuple of tuples
        if isinstance(settings["dirichlet dofs"], (dict, FrozenDict)):
            free_dofs_flat = {
                key: jnp.invert(jnp.asarray(val).flatten())
                for key, val in settings["dirichlet dofs"].items()
            }
        else:
            free_dofs_flat = jnp.invert(
                jnp.asarray(settings["dirichlet dofs"]).flatten()
            )
    else:
        free_dofs_flat = jnp.ones(utility.dict_flatten(dofs).shape, dtype=jnp.bool_)

    verbose = static_settings["verbose"]
    return damped_newton(
        dofs,
        residual_fun,
        lin_solve_fun,
        free_dofs_flat,
        newton_tol,
        maxiter,
        damping_coefficient,
        verbose=verbose,
    )

def damped_newton(
    dofs_0,
    residual_fun,
    lin_solve_fun,
    free_dofs,
    newton_tol,
    maxiter,
    damping_coefficient,
    verbose=1,
    ):
    """
    Performs damped Newton iterations to solve a nonlinear system.

    This function implements the damped Newton method for solving a nonlinear system. It updates
    the solution iteratively based on the residual and chosen linear solver for the Newton step,
    with a damping coefficient to control the step size.

    Args:
      dofs_0 (jnp.ndarray or dict): Initial guess for the degrees of freedom.
      residual_fun (callable): Function to compute the residual of the system.
      lin_solve_fun (callable): Function to solve the linearized system for the Newton step.
      free_dofs (jnp.ndarray or None): Boolean array indicating the free degrees of freedom. 
            If None, all degrees of freedom are free.      
      newton_tol (float): Tolerance for the Newton method convergence criterion.
      maxiter (int): Maximum number of iterations for the Newton method.
      damping_coefficient (float): Damping coefficient for the Newton updates.

    Returns:
      tuple: A tuple containing the following elements:
        - sol (jnp.ndarray): The solution obtained by solving the nonlinear system using the damped Newton method.
        - infos (tuple): Additional information about the solution process, including:
            - num_iterations (int): The number of iterations performed.
            - residual_norm (float): The norm of the residual at the solution.
            - diverged (bool): Flag indicating whether the method diverged.
    """
    def step(carry):
        dofs_i, itt, _, res_norm_old, _ = carry

        # Update formula of newton scheme
        delta_x_i = lin_solve_fun(dofs_i)

        # If free_dofs is None, apply update to all dofs, otherwise apply masking
        if free_dofs is not None:
            # Apply damping to the Newton step on free dofs
            delta_x_i = utility.mask_op(
                delta_x_i, free_dofs, mode="apply", ufunc=lambda x: damping_coefficient * x
            )
            # Update free dofs with the damped step
            dofs_i = utility.mask_op(dofs_i, free_dofs, delta_x_i, "add")

            # Set boundary conditions for dirichlet_dofs
            if isinstance(free_dofs, dict):
                dirichlet_dofs = {key: jnp.invert(val) for (key, val) in free_dofs.items()}
            else:
                dirichlet_dofs = jnp.invert(free_dofs)
            dofs_i = utility.mask_op(dofs_i, dirichlet_dofs, delta_x_i, "set")
        else:
            # If free_dofs is None, apply the update to all dofs directly
            dofs_i += damping_coefficient * delta_x_i

        # Compute residual for next step as convergence test
        residual = residual_fun(dofs_i)
        if free_dofs is not None:
            residual_flat = utility.dict_flatten(utility.mask_select(residual, free_dofs))
        else:
            residual_flat = utility.dict_flatten(residual)  # Use full residual if no mask
        res_norm = jnp.linalg.norm(residual_flat)
        not_stop = jnp.where(res_norm > newton_tol, True, False)

        def report():
            if verbose > 0:
                jax.debug.print(
                    "Residual after Newton iteration {x}: {y}", x=itt + 1, y=res_norm
                )
                if verbose > 1:
                    jax.debug.print("")

            # Check for divergence
            divergence = jnp.where(
                jnp.logical_and(res_norm / res_norm_old > 10, itt > 1), True, False
            )
            # If nan or inf in residual, set divergence
            divergence = jnp.where(
                jnp.any(jnp.logical_or(jnp.isnan(residual_flat), jnp.isinf(residual_flat))),
                True,
                divergence,
            )
            return jnp.invert(divergence), divergence

        def stop_newton():
            jax.debug.print("Warning: Newton scheme could not converge!")
            return False, True

        next_step, divergence = jax.lax.cond(itt < maxiter, report, stop_newton)
        return (
            dofs_i,
            itt + 1,
            jnp.logical_and(not_stop, next_step),
            res_norm,
            divergence,
        )

    def convergence_check(carry):
        _, _, not_stop, _, _ = carry
        return not_stop

    # Start Newton iteration loop
    sol, load_steps, _, res_norm, divergence = lax.while_loop(
        convergence_check, step, (dofs_0, 0, True, 0.0, False)
    )

    return (sol, (load_steps, res_norm, divergence))

### Linear solvers for different backends
@utility.jit_with_docstring(static_argnames=["static_settings", "**kwargs"])
def linear_solve_jax(dofs, settings, static_settings, **kwargs):
    """
    Solves a linear system of equations using JAX's matrix free solvers.

    This function performs a linear solve using different itterative solvers,
    optionally imposing Dirichlet boundary conditions and preconditioning and
    using different matrix or Hessian vector product (HVP) methods.

    Args:
      dofs (jnp.ndarray): Initial degrees of freedom.
      settings (dict): Dictionary of problem settings.
      static_settings (dict): Dictionary of static settings that do not change during iterations.
      **kwargs: Additional keyword arguments for the solver.

    Returns:
      jnp.ndarray: Solution degrees of freedom after solving the linear system.

    Hessian Vector Product (HVP) Types:
      - 'fwdrev': Forward and reverse mode differentiation.
      - 'revrev': Reverse mode differentiation (only for symmetric matrices).
      - 'assemble': Assembles the tangent matrix explicitly (not supported with nodal imposition, then call e.g. PetSc).
      - 'linearize': Uses JAX's linearize function.
      - Default: Uses 'fwdrev' if not specified or if an unsupported type is provided.

    Solver Types:
      - 'cg': Conjugate Gradient.
      - 'normal cg': Normal Conjugate Gradient.
      - 'gmres': Generalized Minimal Residual Method.
      - 'bicgstab': BiConjugate Gradient Stabilized.
      - 'lu': LU Decomposition.
      - 'cholesky': Cholesky Decomposition (converts tangent to dense mode; not supported with nodal imposition).
      - 'qr': QR Decomposition (not supported with nodal imposition).
      - 'jacobi': Jacobi Method.
      - Default: Uses 'bicgstab' if not specified or if an unsupported type is provided.

    Notes:
      - Dirichlet boundary conditions are imposed if 'nodal imposition' is specified as the solution structure.
      - The itterative solver can be preconditioned with 'jacobi'.
      - When using 'assemble' HVP type, the function will explicitly assemble the tangent matrix.
    """
    assert not isinstance(
        dofs, dict
    ), "dofs as dict are currently not implemented in linear_solve_jax."

    solver = static_settings["solver"]
    hvp_type = static_settings["hvp type"]
    nodal_imposition = "nodal imposition" in static_settings["solution structure"]

    if nodal_imposition:
        # Impose Dirichlet boundaries. Dirichlet dofs has to be concrete, therefore it is passed in static_settings as tuple of tuples
        dirichlet_dofs_flat = jnp.asarray(settings["dirichlet dofs"]).flatten()
        dirichlet_conditions = settings["dirichlet conditions"].flatten()
        free_dofs_flat = jnp.invert(dirichlet_dofs_flat)

        # Impose nodal dofs
        flat_dofs = dofs.flatten()
        idx = jnp.arange(flat_dofs.shape[0])[dirichlet_dofs_flat]
        flat_dofs = flat_dofs.at[idx].set(dirichlet_conditions[idx])
        dofs = flat_dofs.reshape(dofs.shape)
        free_dofs = dofs.flatten()[free_dofs_flat]

        def residual_fun(x):
            flat_dofs = dofs.flatten()
            idx = jnp.arange(flat_dofs.shape[0])[free_dofs_flat]
            flat_dofs = flat_dofs.at[idx].set(x)
            return assembler.assemble_residual(
                flat_dofs.reshape(dofs.shape), settings, static_settings
            ).flatten()[free_dofs_flat]

        def diag_assemble_fun(x):
            flat_dofs = dofs.flatten()
            idx = jnp.arange(flat_dofs.shape[0])[free_dofs_flat]
            flat_dofs = flat_dofs.at[idx].set(x)
            return assembler.assemble_tangent_diagonal(
                flat_dofs.reshape(dofs.shape), settings, static_settings
            ).flatten()[free_dofs_flat]

        rhs = -residual_fun(dofs.flatten()[free_dofs_flat])

    else:
        free_dofs = dofs
        residual_fun = lambda x: assembler.assemble_residual(
            x, settings, static_settings
        )
        mat_assemble_fun = lambda x: assembler.assemble_tangent(
            x, settings, static_settings
        )
        diag_assemble_fun = lambda x: assembler.assemble_tangent_diagonal(
            x, settings, static_settings
        )
        rhs = -residual_fun(dofs)

    # Type of Hessian vector product
    match hvp_type:
        case "fwdrev":

            def hvp_fwdrev(v):
                return jax.jvp(residual_fun, (free_dofs,), (v,))[1]

            hvp = hvp_fwdrev
        case "revrev":  # only works for symmetric matrices

            def hvp_revrev(v):
                return jax.jacrev(lambda x: jnp.vdot(residual_fun(x), v))(free_dofs)

            hvp = hvp_revrev
        case "assemble":
            assert (
                not nodal_imposition
            ), "hvp type 'assemble' is currently not supported for nodal imposition of DOFs."

            tangent = mat_assemble_fun(free_dofs)

            def hvp_assembled(v):
                mapped = tangent @ v.flatten()
                return jnp.reshape(mapped, v.shape)

            hvp = hvp_assembled
        case "linearize":

            def hvp_linearized(v):
                (_, linearized) = jax.linearize(residual_fun, free_dofs)
                return linearized(v)

            hvp = hvp_linearized
        case _:
            hvp = hvp_fwdrev
            print(
                "Type of hessian vector product has not been set or is not available. Using static_settings['matrix-free'] = 'revrev' as default."
            )

    # Preconditioning for itterative solvers
    if (
        solver == "cg"
        or solver == "bicgstab"
        or solver == "normal cg"
        or solver == "gmres"
    ):
        try:
            precond_type = static_settings["type of preconditioner"]
            match precond_type:
                case "jacobi":
                    # Inversion of diagonal part of tangent matrix as preconditioner
                    M = 1 / diag_assemble_fun(free_dofs)

                    def preconditioner(v):
                        preconditioned = jnp.multiply(M, v.flatten())
                        return jnp.reshape(preconditioned, free_dofs.shape)

                case _:

                    def preconditioner(v):
                        return v

                    print(
                        "Wrong preconditioner keyword. Continue without preconditioner."
                    )
        except KeyError:
            preconditioner = None
            pass

    # Select linear solver (itterative or direct)
    match solver:
        case "cg":
            (sol, _) = jax.scipy.sparse.linalg.cg(hvp, rhs, M=preconditioner, **kwargs)
        case "normal cg":
            sol = jaxopt.linear_solve.solve_normal_cg(hvp, rhs, **kwargs)
        case "gmres":
            (sol, _) = jax.scipy.sparse.linalg.gmres(
                hvp, rhs, M=preconditioner, **kwargs
            )
        case "bicgstab":
            (sol, _) = jax.scipy.sparse.linalg.bicgstab(
                hvp, rhs, M=preconditioner, **kwargs
            )
        case "lu":
            sol = jaxopt.linear_solve.solve_lu(hvp, rhs)
        case "cholesky":
            assert (
                not nodal_imposition
            ), "solver 'cholesky' is currently not supported for nodal imposition of DOFs."
            chol, lower = jax.scipy.linalg.cho_factor(mat_assemble_fun(dofs).todense())
            sol = jnp.reshape(
                jax.scipy.linalg.cho_solve((chol, lower), rhs.flatten()), dofs.shape
            )
        case "qr":
            assert (
                not nodal_imposition
            ), "solver 'qr' is currently not supported for nodal imposition of DOFs."
            bcoo_tangent = mat_assemble_fun(dofs)
            bcsr_tangent = sparse.BCSR.from_bcoo(bcoo_tangent).sum_duplicates(
                nse=bcoo_tangent.nse
            )
            sol = sparse.linalg.spsolve(
                bcsr_tangent.data,
                bcsr_tangent.indices,
                bcsr_tangent.indptr,
                rhs.flatten(),
            ).reshape(dofs.shape)
        case "jacobi":
            diag = diag_assemble_fun(free_dofs)
            sol = jacobi_method(hvp, diag, free_dofs, rhs, **kwargs)
        case _:
            (sol, _) = jax.scipy.sparse.linalg.bicgstab(
                hvp, rhs, M=preconditioner, **kwargs
            )
            print(
                "Solver name not set or not available in combination with this solver type. Using static_settings['solver'] = 'bicgstab' as default."
            )

    if static_settings["verbose"] >= 1:
        residual = rhs - hvp(sol)
        jax.debug.print(
            "The relative residual is: {value}",
            value=jnp.linalg.norm(residual) / jnp.linalg.norm(rhs),
        )

    # Compose solution dofs
    if nodal_imposition:
        flat_dofs = dofs.flatten()
        flat_dofs = flat_dofs.at[idx].set(dirichlet_conditions[idx])
        idx = jnp.arange(flat_dofs.shape[0])[free_dofs_flat]
        flat_dofs = flat_dofs.at[idx].set(sol)
        dofs = flat_dofs.reshape(dofs.shape)
        return dofs
    else:
        return sol

def scipy_assembling(tangent_with_duplicates, verbose, free_dofs):
    """
    Convert a JAX BCOO matrix to a SciPy CSR matrix while summing duplicates.

    This function converts a JAX BCOO matrix, which may contain duplicate entries,
    into a SciPy CSR matrix. It optionally deletes rows and columns corresponding
    to Dirichlet degrees of freedom.

    Args:
      tangent_with_duplicates (jax.experimental.sparse.BCOO): The input JAX BCOO matrix.
      verbose (int): Verbosity level. If >= 2, timing information is printed.
      free_dofs (array or None): Boolean array indicating which degrees of freedom are free.
                                  If not None, rows and columns corresponding to Dirichlet DOFs
                                  are removed from the matrix.

    Returns:
      scipy.sparse.csr_matrix: The converted and possibly reduced CSR matrix.
    """
    if verbose >= 2:
        start = time.time()

    data = tangent_with_duplicates.data
    indices = tangent_with_duplicates.indices
    rows = indices[:, 0]
    cols = indices[:, 1]

    # This is currently done on CPU and seems to be one of the computational bottlenecks when using GPU
    tangent_coo = scp.sparse.coo_matrix(
        (data, (rows, cols)), shape=tangent_with_duplicates.shape
    )

    tangent_csr = scp.sparse.csr_matrix(tangent_coo)

    # Row deletion for Dirichlet-DOFs
    if free_dofs is not None:
        # Deleting rows and columns
        tangent_csr = tangent_csr[:, free_dofs]
        tangent_csr = tangent_csr[free_dofs]

    if verbose >= 2:
        print("Time for summing duplicates: ", time.time() - start)

    return tangent_csr

def linear_solve_petsc(mat, rhs, n_fields, solver, pc_type, verbose, free_dofs, tol=1e-8, **kwargs):
    """
    Solve a linear system using the PETSc solver (requires PETSc and petsc4py to be installed).

    This function solves a linear system using PETSc, with options for different
    solvers and preconditioners. The input matrix is first converted to a SciPy
    CSR matrix, and rows and columns corresponding to Dirichlet DOFs are optionally
    removed.

    Args:
      mat (jax.experimental.sparse.BCOO): The input matrix in JAX BCOO format.
      rhs (jnp.ndarray): The (reduced/free) right-hand side vector.
      n_fields (int): The number of fields.
      solver (str): The type of solver to use.
      pc_type (str): The type of preconditioner to use.
      verbose (int): Verbosity level. If >= 1, timing and solver information is printed.
      free_dofs (array or None): Boolean array indicating which degrees of freedom are free.
      tol (float): The relative tolerance for the solver.
      **kwargs: Additional keyword arguments for the solver.

    Returns:
      jax.numpy.array: The solution vector.

    Notes:
      - The solver settings can also be set from the command line. See PETSc and petsc4py documentation.
    """
    try:
        import petsc4py

        petsc4py.init(sys.argv)
        from petsc4py import PETSc
    except ModuleNotFoundError:
        print("Linear solver requires petsc and petsc4py")

    if free_dofs is not None:
        reduced_rhs = rhs[free_dofs]
        n_dofs = reduced_rhs.shape[0]
    else:
        reduced_rhs = rhs
        n_dofs = rhs.shape[0]

    # Transform matrix to csr format and sum duplicates
    tangent_csr = scipy_assembling(mat, verbose, free_dofs)
    if verbose >= 2:
        start = time.time()

    # Load to petsc
    mat_petsc = PETSc.Mat().createAIJ(
        size=tangent_csr.shape,
        csr=(
            tangent_csr.indptr.astype(PETSc.IntType),
            tangent_csr.indices.astype(PETSc.IntType),
            tangent_csr.data,
        ),
    )
    mat_petsc.setFromOptions()
    mat_petsc.setBlockSize(n_fields)

    if verbose >= 2:
        print("To PETSc transformation time: ", time.time() - start)
        print("Matrix infos: ")
        print()
        print(mat_petsc.getInfo())
        start = time.time()

    # Initialization of right hand side and solution vector
    b = PETSc.Vec().createSeq(n_dofs)
    b.setFromOptions()
    b.setArray(np.array(reduced_rhs))
    x = PETSc.Vec().createSeq(n_dofs)
    x.setFromOptions()

    # Solver settings
    rtol = tol
    ksp = PETSc.KSP().create()
    ksp.setTolerances(rtol=rtol, **kwargs)
    ksp.setOperators(mat_petsc)
    ksp.setType(solver)
    ksp.setFromOptions()
    ksp.setConvergenceHistory()
    ksp.getPC().setType(pc_type)
    ksp.getPC().setFromOptions()

    # Monitoring
    if verbose >= 2:
        print("Iteration   Residual")

        def monitor(ksp, its, rnorm):
            print("%5d      %20.15g" % (its, rnorm))

        ksp.setMonitor(monitor)

    # Solving
    ksp.solve(b, x)

    # Fill solution vector with computed values
    if free_dofs is not None:
        sol = jnp.zeros(rhs.shape)
        sol = sol.at[free_dofs].set(x.getArray())
    else:
        sol = jnp.array(x.getArray())

    if verbose >= 2:
        residual = mat_petsc * x - b
        print("Itterative linear solver time: ", time.time() - start)
        print("Number of iterations: ", ksp.getIterationNumber())
        print("Type: ", ksp.getType())
        print("Tolerances: ", ksp.getTolerances())
        print(f"The relative residual is: {residual.norm() / (b.norm() + 1e-12)}.")
    return sol

def linear_solve_pardiso(mat, rhs, solver, verbose, free_dofs):
    """
    Solve a linear system using the PARDISO solver (requires Intel MKL and pypardiso('lu') or sparse_dot_mkl('qr') to be installed).

    This function solves a linear system using PARDISO, with options for different
    solver types. The input matrix is first converted to a SciPy CSR matrix, and
    rows and columns corresponding to Dirichlet DOFs are optionally removed.

    Args:
      mat (jax.experimental.sparse.BCOO): The input matrix in JAX BCOO format.
      rhs (jnp.ndarray): The (reduced/free) right-hand side vector.
      solver (str): The type of solver to use ('lu' or 'qr').
      verbose (int): Verbosity level. If >= 2, timing information is printed.
      free_dofs (array or None): Boolean array indicating which degrees of freedom are free.

    Returns:
      jax.numpy.array: The solution vector.
    """
    # Transform matrix to csr format and sum duplicates
    tangent_csr = scipy_assembling(mat, verbose, free_dofs)
    if verbose >= 2:
        start = time.time()

    # Prepare right hand side
    if free_dofs is not None:
        b = np.asarray(rhs[free_dofs])
    else:
        b = np.asarray(rhs)

    if solver == "lu":
        try:
            import pypardiso
        except ModuleNotFoundError:
            print("Linear solver requires the installation of pypardiso.")

        # ToDo: make use of symmetries and other settings available#, msglvl=verbose, iparm=iparm
        # pypardiso_solver = pypardiso.PyPardisoSolver(mtype=11) # spd: 2
        # x = pypardiso.spsolve(tangent_csr, b, solver=pypardiso_solver)
        x = pypardiso.spsolve(tangent_csr, b)
    elif solver == "qr":
        try:
            import sparse_dot_mkl
        except ModuleNotFoundError:
            print("Linear solver requires the installation of sparse_dot_mkl.")

        x = sparse_dot_mkl.sparse_qr_solve_mkl(tangent_csr, b)
    else:
        assert False, "Type of solver not supported. Choose 'lu' or 'qr'"

    # Fill solution vector with computed values
    if free_dofs is not None:
        sol = jnp.zeros(rhs.shape)
        sol = sol.at[free_dofs].set(x)
    else:
        sol = jnp.array(x)

    if verbose >= 2:
        residual = b - tangent_csr * x
        print(
            f"The relative residual after linear solve is: {np.linalg.norm(residual) / (np.linalg.norm(b) + 1e-12)}."
        )
        print("Linear solver time: ", time.time() - start)
    return sol

def linear_solve_pyamg(mat, rhs, solver, pc_type, verbose, free_dofs, **kwargs):
    """
    Solve a linear system using the PyAMG solver (requires pyamg to be installed).

    This function solves a linear system using PyAMG, an algebraic multi-grid solver with options
    for different solvers and preconditioners. The input matrix is first converted to a SciPy
    CSR matrix, and rows and columns corresponding to Dirichlet DOFs are optionally
    removed.

    Args:
      mat (jax.experimental.sparse.BCOO): The input matrix in JAX BCOO format.
      rhs (jnp.ndarray): The (reduced/free) right-hand side vector.
      solver (str): The type of solver to use ('cg', 'bcgs', or 'gmres').
      pc_type (str): The type of preconditioner to use ('ruge stuben' or 'smoothed aggregation').
      verbose (int): Verbosity level. If >= 1, timing and solver information is printed.
      free_dofs (array or None): Boolean array indicating which degrees of freedom are free.
      **kwargs: Additional keyword arguments for the solver.

    Returns:
      jax.numpy.array: The solution vector.
    """
    # Transform matrix to csr format and sum duplicates
    pyamg_tangent = scipy_assembling(mat, verbose, free_dofs)

    if verbose >= 2:
        start = time.time()

    # Set up solver
    try:
        import pyamg
    except ModuleNotFoundError:
        print("Linear solver requires the installation of pyamg.")

    if pc_type == "ruge stuben":
        ml = pyamg.ruge_stuben_solver(A=pyamg_tangent)
    elif pc_type == "smoothed aggregation":
        ml = pyamg.smoothed_aggregation_solver(A=pyamg_tangent)
    elif pc_type == "root node":
        ml = pyamg.rootnode_solver(A=pyamg_tangent)
    elif pc_type == "pairwise":
        ml = pyamg.pairwise_solver(A=pyamg_tangent)
    else:
        assert (
            False
        ), "Type of preconditioner not supported. Choose 'ruge stuben' or 'smoothed aggregation', 'root node' or 'pairwise'"

    if verbose >= 2:
        print(ml)
        print("Time for setting up multigrid preconditioner: ", time.time() - start)
        start = time.time()

    # Prepare right hand side
    if free_dofs is not None:
        b = np.asarray(rhs[free_dofs])
    else:
        b = np.asarray(rhs)

    residuals = []

    # Solving
    if solver == None:
        x = ml.solve(b, tol=1e-8, residuals=residuals, cycle='W')
    elif solver == "cg":
        x = ml.solve(b, accel=scp.sparse.linalg.cg, tol=1e-8, residuals=residuals, cycle='W')
    elif solver == "bcgs":
        x = ml.solve(b, accel=scp.sparse.linalg.bicgstab, tol=1e-8, residuals=residuals, cycle='W')
    elif solver == "gmres":
        x = ml.solve(b, accel=scp.sparse.linalg.gmres, tol=1e-8, residuals=residuals, cycle='W')
    else:
        assert False, "Type of solver not supported. Choose None, 'cg', 'bcgs' or 'gmres'"

    # Fill solution vector with computed values
    if free_dofs is not None:
        sol = jnp.zeros(rhs.shape)
        sol = sol.at[free_dofs].set(x)
    else:
        sol = jnp.array(x)

    if verbose >= 2:
        residual = b - pyamg_tangent * x
        print(
            f"The relative residual is: {np.linalg.norm(residual) / (np.linalg.norm(b) + 1e-12)}."
        )
        print("Itterative linear solver time: ", time.time() - start)

    if verbose >= 3:
        import matplotlib.pyplot as plt
        plt.semilogy(residuals/residuals[0], 'o-')
        plt.xlabel('iterations')
        plt.ylabel('normalized residual')
        plt.show()

    return sol

def linear_solve_scipy(mat, rhs, solver, verbose, free_dofs):
    """
    Solves a linear system using a specified SciPy solver.

    Args:
      mat (bcoo): JAX BCOO matrix representing the system's tangent matrix.
      rhs (jnp.ndarray): The (reduced/free) right-hand side vector.
      solver (str): Type of solver to use. Options are 'lapack' or 'umfpack'.
      verbose (int): Verbosity level for logging.
      free_dofs (jnp.ndarray): Boolean array indicating free degrees of freedom for Dirichlet boundary conditions.

    Returns:
      sol (jnp.ndarray): Solution vector to the linear system.
    """
    # Transform matrix to csr format and sum duplicates
    tangent_csr = scipy_assembling(mat, verbose, free_dofs)

    if verbose >= 2:
        start = time.time()

    # Prepare right hand side
    if free_dofs is not None:
        b = rhs[free_dofs]
    else:
        b = rhs

    if solver == "lapack":
        x = scp.sparse.linalg.spsolve(tangent_csr, b)
    elif solver == "umfpack":
        x = scp.sparse.linalg.spsolve(tangent_csr, b, use_umfpack=True)
    else:
        assert False, "Type of solver not supported. Choose 'lapack' or 'umfpack'"

    # Fill solution vector with computed values
    if free_dofs is not None:
        sol = jnp.zeros(rhs.shape)
        sol = sol.at[free_dofs].set(x)
    else:
        sol = jnp.array(x)

    if verbose >= 2:
        residual = b - tangent_csr * x
        print(
            f"The relative residual is: {np.linalg.norm(residual) / (np.linalg.norm(b) + 1e-12)}."
        )
        print("Direct solver time: ", time.time() - start)
    return sol

### Iterative solvers/smoothers
def jacobi_method(hvp_fun, diag, x_0, rhs, tol=1e-6, atol=1e-6, maxiter=1000):
    """
    Solve Ax = b using Jacobi iterations (experimental).

    Args:
      hvp_fun (function): Hessian vector product function.
      diag (jnp.ndarray): Diagonal of the Hessian matrix.
      x_0 (jnp.ndarray): Initial guess for the solution.
      rhs (jnp.ndarray): Right-hand side vector of the linear system.
      tol (float): Relative tolerance for convergence.
      atol (float): Absolute tolerance for convergence.
      maxiter (int): Maximum number of iterations.

    Returns:
      array: Solution vector x.
    """
    assert not isinstance(
        x_0, dict
    ), "dofs as dict are currently not implemented in jacobi_method."

    # Initialization
    inverse_diag = 1 / diag
    scaled_rhs = jnp.multiply(inverse_diag, rhs)
    rhs_squared = jnp.vdot(rhs, rhs)

    # Itterations
    def body_fun(value):
        x_k, k = value
        dx_k1 = scaled_rhs - jnp.multiply(inverse_diag, hvp_fun(x_k))
        x_k1 = x_k + dx_k1
        return (x_k1, k + 1)

    def cond_fun(value):
        x_k, k = value
        r_k = rhs - hvp_fun(x_k)
        r_k_squared = jnp.vdot(r_k, r_k)
        return k < maxiter  # and r_k_squared > jnp.max(tol**2 * rhs_squared, atol**2)

    x_final, *_ = lax.while_loop(cond_fun, body_fun, (x_0, 0))

    return x_final

def damped_jacobi_relaxation(hvp_fun, diag, x_0, rhs, damping_factor=0.3333333, **kwargs):
    """
    Damped Jacobi smoother (experimental).

    Args:
      hvp_fun (callable): Hessian vector product function.
      diag (jnp.ndarray): Diagonal of the Hessian matrix.
      x_0 (jnp.ndarray): Initial guess for the solution.
      rhs (jnp.ndarray): Right-hand side vector of the linear system.
      damping_factor (float): Damping factor for the iterations (<=0.5 guarantees a good smoother).
      **kwargs (dict): Additional keyword arguments for customization.

    Returns:
      array: Solution vector x.
    """
    assert not isinstance(
        x_0, dict
    ), "dofs as dict are currently not implemented in damped_jacobi_relaxation."

    # Initialization
    inverse_diag = 1 / diag
    scaled_rhs = jnp.multiply(inverse_diag, rhs)

    # Itterations
    def body_fun(x_k, idx):
        dx_k1 = scaled_rhs - jnp.multiply(inverse_diag, hvp_fun(x_k))
        x_k1 = x_k + damping_factor * dx_k1
        return x_k1, None

    iterations = 20
    x_final, *_ = lax.scan(body_fun, init=x_0, xs=None, length=iterations)
    return x_final
