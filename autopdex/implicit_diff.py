# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
These functions are a modification and extension of the functions in jaxopt._src.implicit_diff for
external solve functions with a jax.experimental.sparse.BCOO matrix as an argument instead of a matvec function.
The root_vjp and root_jvp functions were modified in a way that external solvers 
can be used via a pure_callback and constraints can be taken into account.
With the wrapper custom_root, root solvers can be made differentiable both in forward or reverse mode of arbitrary order.
Mixing the differentiation mode is currently not possible.
"""

import inspect
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple

import jax
import jax.numpy as jnp

from autopdex import utility


def tree_scalar_mul(scalar, tree_x):
    """Compute scalar * tree_x."""
    return jax.tree_util.tree_map(lambda x: scalar * x, tree_x)

def _extract_kwargs(kwarg_keys, flat_args):
    n = len(flat_args) - len(kwarg_keys)
    args, kwarg_vals = flat_args[:n], flat_args[n:]
    kwargs = dict(zip(kwarg_keys, kwarg_vals))
    return args, kwargs

def _signature_bind(signature, *args, **kwargs):
    ba = signature.bind(*args, **kwargs)
    ba.apply_defaults()
    return ba.args, ba.kwargs

def _signature_bind_and_match(signature, *args, **kwargs):
    # We want to bind *args and **kwargs based on the provided
    # signature, but also to associate the resulting positional
    # arguments back. To achieve this, we lift arguments to a triple:
    #
    #   (was_kwarg, ref, value)
    #
    # where ref is an index position (int) if the original argument was
    # from *args and a dictionary key if the original argument was from
    # **kwargs. After binding to the inspected signature, we use the
    # tags to associate the resolved positional arguments back to their
    # arg and kwarg source.

    args = [(False, i, v) for i, v in enumerate(args)]
    kwargs = {k: (True, k, v) for (k, v) in kwargs.items()}
    ba = signature.bind(*args, **kwargs)

    mapping = [(was_kwarg, ref) for was_kwarg, ref, _ in ba.args]

    def map_back(out_args):
        src_args = [None] * len(args)
        src_kwargs = {}
        for (was_kwarg, ref), out_arg in zip(mapping, out_args):
            if was_kwarg:
                src_kwargs[ref] = out_arg
            else:
                src_args[ref] = out_arg
        return src_args, src_kwargs

    out_args = tuple(v for _, _, v in ba.args)
    out_kwargs = {k: v for k, (_, _, v) in ba.kwargs.items()}
    return out_args, out_kwargs, map_back

def _jvp_args(residual_fun, sol, args, tangents):
    """JVP in the second argument of residual_fun."""
    # We close over the solution.
    fun = lambda *y: residual_fun(sol, *y)
    return jax.jvp(fun, args, tangents)[1]

def _root_vjp(
    residual_fun: Callable,
    mat_fun: Callable,
    sol: Any,
    args: Tuple,
    cotangent: Any,
    solve_fun: Callable,
    free_dofs: Any,
) -> Any:
    """Vector-Jacobian product of a root.

    The invariant is ``residual_fun(sol, *args) == 0``.

    Args:
      residual_fun: the optimality function to use.
      sol: solution / root (pytree).
      mat_fun: a function that has to compute the sparse tangent matrix with sol and args as arguments.
      args: tuple containing the arguments with respect to which we wish to
        differentiate ``sol`` against.
      cotangent: vector to left-multiply the Jacobian with
        (pytree, same structure as ``sol``).
      solve_fun: a linear solver of the form ``x = solve_fun(mat, b)``,
        where ``mat`` is as jax.experimental.sparse.BCOO matrix.
    Returns:
      tuple of the same length as ``len(args)`` containing the vjps w.r.t.
      each argument. Each ``vjps[i]`` has the same pytree structure as
      ``args[i]``.
    """

    free_dofs_flat = None
    if free_dofs is not None:
        assert "dirichlet conditions" in args[0], "'dirichlet conditions' \
        have to be defined in a dict as the second argument of the root solver function."
        dirichlet_dofs_flat = utility.dict_flatten(args[0]["dirichlet dofs"])
        free_dofs_flat = ~dirichlet_dofs_flat

    @jax.custom_vjp
    def linear_solver_fun_vjp(A, b, free_dofs_flat):
        return solve_fun(A, b, free_dofs_flat)  # Here can be an external callback

    def linear_solver_fun_vjp_fwd(A, b, free_dofs_flat):
        result = linear_solver_fun_vjp(A, b, free_dofs_flat)
        return result, (A, b, free_dofs_flat, result)

    def linear_solver_fun_vjp_bwd(res, g):
        A, b, free_dofs_flat, result = res

        # Ensure g matches the size of the expected gradient
        if free_dofs is not None:

            # Sparse outer product
            result_dot = linear_solver_fun_vjp(A.T, g, free_dofs_flat)

            size = A.shape[0]
            empty = jnp.zeros((size,), dtype=float)
            result_tmp = utility.mask_op(empty, free_dofs_flat, result, "set")
            result_dot_tmp = utility.mask_op(empty, free_dofs_flat, result_dot, "set")

            # # Use the sparsity structure of A to build Fx_dot
            indices = jnp.asarray(A.indices, dtype=int)
            data = -result_dot_tmp[indices[:, 0]] * result_tmp[indices[:, 1]]

            # Construct Fx_dot with the same sparsity pattern
            Fx_dot = jax.experimental.sparse.BCOO((data, indices), shape=A.shape)
            Fy_dot = result_dot
        else:
            # Sparse outer product
            result_dot = linear_solver_fun_vjp(A.T, g, None)

            # Use the sparsity structure of A to build Fx_dot
            indices = jnp.asarray(A.indices, dtype=int)
            data = -result_dot[indices[:, 0]] * result[indices[:, 1]]

            # Construct Fx_dot with the same sparsity pattern
            Fx_dot = jax.experimental.sparse.BCOO((data, indices), shape=A.shape)
            Fy_dot = result_dot

        # Correctly return the shapes that match the input arguments
        return (Fx_dot, Fy_dot, None)

    linear_solver_fun_vjp.defvjp(linear_solver_fun_vjp_fwd, linear_solver_fun_vjp_bwd)
    diffable_solve_fun = linear_solver_fun_vjp

    mat = mat_fun(sol, *args)

    # The solution of A^T u = v, where
    # A = jacobian(residual_fun, argnums=0)
    # v = -cotangent.
    v = tree_scalar_mul(-1, cotangent)

    if free_dofs is not None:
        dirichlet_dofs = utility.reshape_as(dirichlet_dofs_flat, sol)

        v_flat = utility.dict_flatten(v)
        u_f = diffable_solve_fun(mat.T, v_flat, free_dofs_flat)
        u_flat = utility.mask_op(
            utility.dict_flatten(utility.dict_zeros_like(v)), free_dofs_flat, u_f, "set"
        )

        def fun_args(*args):
            def residual_fun_tmp(sol, *args):
                dirichlet_values = args[0]["dirichlet conditions"]
                sol_with_bc = utility.mask_op(
                    sol, dirichlet_dofs, dirichlet_values, "set"
                )
                return residual_fun(sol_with_bc, *args)

            return residual_fun_tmp(sol, *args)

        u = utility.reshape_as(u_flat, v)
    else:

        def fun_args(*args):
            return residual_fun(sol, *args)

        v_flat = utility.dict_flatten(v)
        u = utility.reshape_as(diffable_solve_fun(mat.T, v_flat, None), v)

    _, vjp_fun_args = jax.vjp(fun_args, *args)
    args_vjp = vjp_fun_args(u)

    if free_dofs is not None:
        updated_args0 = args_vjp[0]
        tmpl = updated_args0["dirichlet conditions"]
        updated_args0["dirichlet conditions"] = utility.mask_op(
            updated_args0["dirichlet conditions"], utility.reshape_as(dirichlet_dofs, tmpl), utility.reshape_as(cotangent, tmpl), "add"
        )
        args_vjp = (updated_args0,) + args_vjp[1:]

    return args_vjp

def _root_jvp(
    residual_fun: Callable,
    mat_fun: Callable,
    sol: Any,
    args: Tuple,
    tangents: Tuple,
    solve_fun: Callable,
    free_dofs: Any,
    # dirichlet_dofs: Any,
) -> Any:
    """
    Jacobian-vector product of a root.

    The invariant is ``residual_fun(sol, *args) == 0``.

    Args:
      residual_fun: the optimality function to use.
      mat_fun: a function that has to compute the sparse tangent matrix with sol and args as arguments.
      sol: solution / root (pytree).
      args: tuple containing the arguments with respect to which to differentiate.
      tangents: a tuple of the same size as ``len(args)``. Each ``tangents[i]``
        has the same pytree structure as ``args[i]``.
      solve_fun: a linear solver of the form ``x = solve_fun(mat, b)``,
        where ``mat`` is as jax.experimental.sparse.BCOO matrix.
    Returns:
      a pytree with the same structure as ``sol``.
    """
    free_dofs_flat = None
    if free_dofs is not None:
        assert "dirichlet conditions" in args[0], "'dirichlet conditions' \
        have to be defined in a dict as the second argument of the root solver function."
        dirichlet_dofs_flat = utility.dict_flatten(args[0]["dirichlet dofs"])
        free_dofs_flat = ~dirichlet_dofs_flat

    # Compute tangent matrix
    A = mat_fun(sol, *args)
    mat_shape = A.shape

    # Forward differentiable sparse linear solver
    # TODO: register as primitive in order to allow mixed jacfwd/jacrev
    @jax.custom_jvp
    def linear_solver_fun_jvp(data, indices, b, free_dofs_flat):
        A = jax.experimental.sparse.BCOO((data, indices), shape=mat_shape)
        return solve_fun(A, b, free_dofs_flat)

    @linear_solver_fun_jvp.defjvp
    def linear_solver_fun_jvp_rule(primals, tangents):
        data, indices, b, free_dofs_flat = primals
        data_dot, _, b_dot, _ = tangents

        # Compute the primal result using the linear solver function
        primal_result = linear_solver_fun_jvp(data, indices, b, free_dofs_flat)

        # ToDo: is it somehow possible without A_dot via jvps?
        A_dot = jax.experimental.sparse.BCOO((data_dot, indices), shape=mat_shape)

        # Handle the tangent calculation
        if free_dofs is not None:
            primal_result_tmp = utility.mask_op(
                jnp.zeros((mat_shape[0],), dtype=float),
                free_dofs_flat,
                primal_result,
            )
            rhs = b_dot - (A_dot @ primal_result_tmp)
            result_dot = linear_solver_fun_jvp(data, indices, rhs, free_dofs_flat)
        else:
            result_dot = linear_solver_fun_jvp(
                data, indices, b_dot - A_dot @ primal_result, None
            )

        return primal_result, result_dot

    # Assign the jvp-enabled solver function
    solve_func = linear_solver_fun_jvp

    if free_dofs is not None:
        dirichlet_dofs = utility.reshape_as(dirichlet_dofs_flat, sol)

        # Explicit imposition of DOFs in order to be able to take derivatives w.r.t. nodally imposed DOFs
        def residual_fun_tmp(sol, *args):
            sol_with_bc = utility.mask_op(
                sol, dirichlet_dofs, args[0]["dirichlet conditions"], "set"
            )
            return residual_fun(sol_with_bc, *args)

        Bv = _jvp_args(residual_fun_tmp, sol, args, tangents)
        Bv_free = utility.dict_flatten(Bv)
        Jv_free = solve_func(A.data, A.indices, -Bv_free, free_dofs_flat)

        empty_flat = utility.dict_flatten(utility.dict_zeros_like(sol))
        Jv = utility.reshape_as(
            utility.mask_op(empty_flat, free_dofs_flat, Jv_free, "set"), sol
        )

        Jv = utility.mask_op(
            Jv, dirichlet_dofs, tangents[0]["dirichlet conditions"], "set"
        )

    else:
        Bv = _jvp_args(residual_fun, sol, args, tangents)
        Jv = utility.reshape_as(
            solve_func(A.data, A.indices, -utility.dict_flatten(Bv), None), Bv
        )

    return Jv

def _custom_root(
    solver_fun,
    residual_fun,
    mat_fun,
    free_dofs,
    solve,
    has_aux,
    mode="reverse",
    reference_signature=None,
):
    # When caling through `jax.custom_vjp`, jax attempts to resolve all
    # arguments passed by keyword to positions (this is in order to
    # match against a `nondiff_argnums` parameter that we do not use
    # here). It does so by resolving them according to the custom_jvp'ed
    # function's signature. It disallows functions defined with a
    # catch-all `**kwargs` expression, since their signature cannot
    # always resolve all keyword arguments to positions.
    #
    # We can loosen the constraint on the signature of `solver_fun` so
    # long as we resolve keywords to positions ourselves. We can do so
    # just in time, by flattening the `kwargs` dict (respecting its
    # iteration order) and supplying `custom_vjp` with a
    # positional-argument-only function. We then explicitly coordinate
    # flattening and unflattening around the `custom_vjp` boundary.
    #
    # Once we make it past the `custom_vjp` boundary, we do some more
    # work to align arguments with the reference signature (which is, by
    # default, the signature of `residual_fun`).

    solver_fun_signature = inspect.signature(solver_fun)

    if reference_signature is None:
        reference_signature = inspect.signature(residual_fun)

    elif not isinstance(reference_signature, inspect.Signature):
        # If is a CompositeLinearFunction, accesses subfun.
        # Otherwise, assumes a Callable.
        fun = getattr(reference_signature, "subfun", reference_signature)
        reference_signature = inspect.signature(fun)

    def make_custom_solver_fun(solver_fun, kwarg_keys):

        def solver_fun_flat_tmp(*flat_args):
            args, kwargs = _extract_kwargs(kwarg_keys, flat_args)
            return solver_fun(*args, **kwargs)

        if mode == "reverse" or mode == "backward":
            solver_fun_flat = jax.custom_vjp(solver_fun_flat_tmp)
        elif mode == "forward":
            solver_fun_flat = jax.custom_jvp(solver_fun_flat_tmp)
        else:
            raise ValueError("Mode must be either 'forward' or 'reverse'.")

        # Forward-mode differentiation (JVP)
        def solver_fun_jvp(primals, tangents):
            args, kwargs = _extract_kwargs(kwarg_keys, primals)
            tangent_args, tangent_kwargs = _extract_kwargs(kwarg_keys, tangents)

            # Compute the primal solution using the root solver function
            primal_sol = solver_fun_flat(*args, **kwargs)

            # Handle has_aux case
            if has_aux:
                sol = primal_sol[0]
                aux_data = primal_sol[1:]

                # TODO: allow integer and boolean auxilary data
            else:
                sol = primal_sol

            # Compute JVP using root_jvp
            jvp_sol = _root_jvp(
                residual_fun=residual_fun,
                mat_fun=mat_fun,
                sol=sol,
                args=args[1:],  # Exclude the initial params from args
                tangents=tangent_args[1:],  # Exclude the initial params from tangents
                solve_fun=solve,
                free_dofs=free_dofs,
            )

            if has_aux:
                # Return primal and tangent for both solution and auxiliary data
                aux_tangent = jax.tree_util.tree_map(jnp.zeros_like, aux_data)
                return (sol,) + aux_data, (jvp_sol,) + aux_tangent
            else:
                return primal_sol, jvp_sol

        # Reverse-mode differentiation (VJP)
        def solver_fun_fwd(*flat_args):
            res = solver_fun_flat(*flat_args)
            return res, (res, flat_args)

        def solver_fun_bwd(tup, cotangent):
            res, flat_args = tup
            args, kwargs = _extract_kwargs(kwarg_keys, flat_args)

            # solver_fun can return auxiliary data if has_aux = True.
            if has_aux:
                cotangent = cotangent[0]
                sol = res[0]
            else:
                sol = res

            ba_args, ba_kwargs, map_back = _signature_bind_and_match(
                reference_signature, *args, **kwargs
            )
            if ba_kwargs:
                raise TypeError(
                    "keyword arguments to solver_fun could not be resolved to "
                    "positional arguments based on the signature "
                    f"{reference_signature}. This can happen under custom_root if "
                    "residual_fun takes catch-all **kwargs, or under "
                    "custom_fixed_point if fixed_point_fun takes catch-all **kwargs, "
                    "both of which are currently unsupported."
                )

            # Compute VJPs w.r.t. args.
            vjps = _root_vjp(
                residual_fun=residual_fun,
                mat_fun=mat_fun,
                sol=sol,
                args=ba_args[1:],
                cotangent=cotangent,
                solve_fun=solve,
                free_dofs=free_dofs,
            )
            # Prepend None as the vjp for init_params.
            vjps = (None,) + vjps

            arg_vjps, kws_vjps = map_back(vjps)
            ordered_vjps = tuple(arg_vjps) + tuple(kws_vjps[k] for k in kwargs.keys())
            return ordered_vjps

        if mode == "reverse" or mode == "backward":
            solver_fun_flat.defvjp(solver_fun_fwd, solver_fun_bwd)
        else:
            solver_fun_flat.defjvp(solver_fun_jvp)
        return solver_fun_flat

    def wrapped_solver_fun(*args, **kwargs):
        args, kwargs = _signature_bind(solver_fun_signature, *args, **kwargs)
        keys, vals = list(kwargs.keys()), list(kwargs.values())
        return make_custom_solver_fun(solver_fun, keys)(*args, *vals)

    return wrapped_solver_fun

def custom_root(
    residual_fun: Callable,
    mat_fun: Callable,
    solve: Callable,
    free_dofs = None,
    has_aux: bool = False,
    mode="reverse",
    reference_signature: Optional[Callable] = None,
):
    """Decorator for adding implicit differentiation to a root solver.

    Args:
      residual_fun: A callable the returns the possibly nonlinear residual of which to find the root of,
        ``residual_fun(dofs, *args)``.
        The invariant is ``residual_fun(sol, *args) == 0`` at the solution / root ``sol``.
      mat_fun: A callable that returns the sparse tangent matrix as a jax.experimental.BCOO with dofs and
        args as arguments. Can also be a pure callback.
      solve: A linear solver of the form ``solve(mat[jax.experimental.BCOO], b[jnp.ndarray])``.
      free_dofs: For constraining certain degrees of freedom. In case free_dofs is not None, the second argument of the solver
        has to be a dictionary having the keys 'dirichlet dofs' and 'dirichlet conditions'. The first one is a
        dictionary of jnp.ndarrays with the same structure as dofs, where the entries are boolean masks indicating
        the dofs that are constrained. The second one is a dictionary of jnp.ndarrays with the same structure as dofs,
        where the entries are the values of the constrained dofs (see source code of _root_jvp and _root_vjp
        for details or solver.adaptive_load_stepping for exemplary use).
      has_aux: whether the decorated root solver function returns auxiliary data.
      mode: The differentiation mode ('forward' or 'reverse'/'backward').
      reference_signature: optional function signature
        (i.e. arguments and keyword arguments), with which the
        solver and optimality functions are expected to agree. Defaults
        to ``residual_fun``. It is required that solver and optimality
        functions share the same input signature, but both might be
        defined in such a way that the signature correspondence is
        ambiguous (e.g. if both accept catch-all ``**kwargs``). To
        satisfy custom_root's requirement, any function with an
        unambiguous signature can be provided here.

    Returns:
      The decorated root solver function that is equipped a with custom vjp or jvp rule.

    Example:
      See e.g. the implementation of autopdex.solver.adaptive_load_stepping.
    """

    def wrapper(solver_fun):
        return _custom_root(
            solver_fun,
            residual_fun,
            mat_fun,
            free_dofs,
            solve,
            has_aux,
            mode,
            reference_signature,
        )

    return wrapper
