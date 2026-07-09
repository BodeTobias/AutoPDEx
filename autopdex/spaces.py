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

# TODO: refactoring

from __future__ import annotations
from typing import Any, Callable, Dict
import sys
from abc import ABC, abstractmethod
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np

from autopdex.utility import jit_with_docstring, lin_solve


## Helper functions
def _polynomial_basis(x, order):
    """
    Generate a polynomial basis of a given order and dimensionality.

    Args:
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
                    return jnp.asarray([1.0])
                case 1:
                    return jnp.asarray([1.0, x])
                case 2:
                    return jnp.asarray([1.0, x, x**2])
                case 3:
                    return jnp.asarray([1.0, x, x**2, x**3])
                case 4:
                    return jnp.asarray([1.0, x, x**2, x**3, x**4])
                case 5:
                    return jnp.asarray([1.0, x, x**2, x**3, x**4, x**5])
                case 6:
                    return jnp.asarray([1.0, x, x**2, x**3, x**4, x**5, x**6])
                case 7:
                    return jnp.asarray([1.0, x, x**2, x**3, x**4, x**5, x**6, x**7])
                case 8:
                    return jnp.asarray(
                        [1.0, x, x**2, x**3, x**4, x**5, x**6, x**7, x**8]
                    )
                case 9:
                    return jnp.asarray(
                        [1.0, x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9]
                    )
                case 10:
                    return jnp.asarray(
                        [1.0, x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10]
                    )
                case _:
                    sys.ext("Polynomial basis not implemented for this order!")
        case 2:
            match order:
                case 0:
                    return jnp.asarray([1.0])
                case 1:
                    return jnp.asarray([1.0, x[0], x[1]])
                case 2:
                    return jnp.asarray(
                        [1.0, x[0], x[1], x[0] ** 2, x[0] * x[1], x[1] ** 2]
                    )
                case 3:
                    return jnp.asarray(
                        [
                            1.0,
                            x[0],
                            x[1],
                            x[0] ** 2,
                            x[0] * x[1],
                            x[1] ** 2,
                            x[0] ** 3,
                            x[0] ** 2 * x[1],
                            x[0] * x[1] ** 2,
                            x[1] ** 3,
                        ]
                    )
                case _:
                    sys.ext("Polynomial basis not implemented for this order!")
        case 3:
            match order:
                case 0:
                    return jnp.asarray([1.0])
                case 1:
                    return jnp.asarray([1.0, x[0], x[1], x[2]])
                case 2:
                    return jnp.asarray(
                        [
                            1.0,
                            x[0],
                            x[1],
                            x[2],
                            x[0] ** 2,
                            x[1] ** 2,
                            x[2] ** 2,
                            x[0] * x[1],
                            x[0] * x[2],
                            x[1] * x[2],
                        ]
                    )
                case 3:
                    return jnp.asarray(
                        [
                            1.0,
                            x[0],
                            x[1],
                            x[2],
                            x[0] ** 2,
                            x[1] ** 2,
                            x[2] ** 2,
                            x[0] * x[1],
                            x[0] * x[2],
                            x[1] * x[2],
                            x[0] ** 3,
                            x[1] ** 3,
                            x[2] ** 3,
                            x[0] ** 2 * x[1],
                            x[0] ** 2 * x[2],
                            x[1] ** 2 * x[0],
                            x[1] ** 2 * x[2],
                            x[2] ** 2 * x[0],
                            x[2] ** 2 * x[1],
                            x[0] * x[1] * x[2],
                        ]
                    )
                case _:
                    sys.ext("Polynomial basis not implemented for this order!")
        case 4:
            match order:
                case 0:
                    return jnp.asarray([1.0])
                case 1:
                    return jnp.asarray([1.0, x[0], x[1], x[2], x[3]])
                case 2:
                    return jnp.asarray(
                        [
                            1.0,
                            x[0],
                            x[1],
                            x[2],
                            x[3],
                            x[0] ** 2,
                            x[1] ** 2,
                            x[2] ** 2,
                            x[3] ** 2,
                            x[0] * x[1],
                            x[0] * x[2],
                            x[0] * x[3],
                            x[1] * x[2],
                            x[1] * x[3],
                            x[2] * x[3],
                        ]
                    )
                case 3:
                    return jnp.asarray(
                        [
                            1.0,
                            x[0],
                            x[1],
                            x[2],
                            x[3],
                            x[0] ** 2,
                            x[1] ** 2,
                            x[2] ** 2,
                            x[3] ** 2,
                            x[0] * x[1],
                            x[0] * x[2],
                            x[0] * x[3],
                            x[1] * x[2],
                            x[1] * x[3],
                            x[2] * x[3],
                            x[0] ** 3,
                            x[1] ** 3,
                            x[2] ** 3,
                            x[3] ** 3,
                            x[0] ** 2 * x[1],
                            x[0] ** 2 * x[2],
                            x[0] ** 2 * x[3],
                            x[1] ** 2 * x[0],
                            x[1] ** 2 * x[2],
                            x[1] ** 2 * x[3],
                            x[2] ** 2 * x[0],
                            x[2] ** 2 * x[1],
                            x[2] ** 2 * x[3],
                            x[3] ** 2 * x[0],
                            x[3] ** 2 * x[1],
                            x[3] ** 2 * x[1],
                            x[0] * x[1] * x[2],
                            x[0] * x[1] * x[3],
                            x[0] * x[2] * x[3],
                            x[1] * x[2] * x[3],
                        ]
                    )
                case _:
                    sys.exit("Polynomial basis not implemented for this order!")
        case _:
            sys.exit("Polynomial basis not implemented for this dimensionality!")

def _compute_poly_basis_length(n_dim, order):
    """
    Compute the number of coefficients for a polynomial basis given its dimensionality and order.

    Args:
      n_dim (int): The number of dimensions of the polynomial basis.
      order (int): The order of the polynomial basis.

    Returns:
      int: The number of coefficients in the polynomial basis.
    """
    match n_dim:
        case 1:
            return order + 1
        case 2:
            return np.sum([i + 1 for i in range(order + 1)])
        case 3:
            return np.sum(
                [np.sum([j + 1 for j in range(i + 1)]) for i in range(order + 1)]
            )
        case 4:
            return np.sum(
                [
                    np.sum(
                        [np.sum([k + 1 for k in range(j + 1)]) for j in range(i + 1)]
                    )
                    for i in range(order + 1)
                ]
            )
        case _:
            sys.exit("Polynomial basis not implemented for this dimensionality!")

@jit_with_docstring(static_argnames=["static_settings", "set"])
def _shape_fun(x_i, i, local_dofs, settings, static_settings, set):
    return solution_space(x_i, i, local_dofs, settings, static_settings, set)

_shape_funs = jax.jit(
    jax.vmap(_shape_fun, (0, 0, 0, None, None, None), 0),
    static_argnames=["static_settings", "set"],
)

@jit_with_docstring(static_argnames=["static_settings", "set"])
def _shape_fun_dx(x_i, i, local_dofs, settings, static_settings, set):
    return jax.jacfwd(solution_space)(
        x_i, i, local_dofs, settings, static_settings, set
    )

_shape_funs_dx = jax.jit(
    jax.vmap(_shape_fun_dx, (0, 0, 0, None, None, None), 0),
    static_argnames=["static_settings", "set"],
)

@jit_with_docstring(static_argnames=["static_settings", "set"])
def _shape_fun_dxx(x_i, i, local_dofs, settings, static_settings, set):
    return jax.jacfwd(jax.jacfwd(solution_space))(
        x_i, i, local_dofs, settings, static_settings, set
    )

_shape_funs_dxx = jax.jit(
    jax.vmap(_shape_fun_dxx, (0, 0, 0, None, None, None), 0),
    static_argnames=["static_settings", "set"],
)

### Cached shape function definitions
from functools import lru_cache
@lru_cache(maxsize=None)
def get_jitted_shape_functions(shp_fun_key):
    return jax.jit(get_shape_functions(*shp_fun_key))

def get_shape_functions(name, n_dim, n_nodes):
    # print("Generating shape functions for " + name + " with " + str(n_nodes) + " nodes in " + str(n_dim) + " dimensions...")
    match name:
        case "line_quad_brick":
            match n_dim:
                case 1:
                    """The following shape functions were generated using the code below:

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
                                x0 = 0.5 * xi
                                return jnp.stack([0.5 - x0, x0 + 0.5])

                        case 3:

                            def shape_functions(xi):
                                x0 = 0.5 * xi
                                return jnp.stack(
                                    [x0 * (xi - 1.0), x0 * (xi + 1.0), 1.0 - xi**2]
                                )

                        case 4:

                            def shape_functions(xi):
                                x0 = 3.0 * xi
                                x1 = x0 + 1.0
                                x2 = xi + 1.0
                                x3 = x2 * (x0 - 1.0)
                                x4 = 0.5625 * xi - 0.5625
                                return jnp.stack(
                                    [
                                        -0.5625 * xi**3 + 0.5625 * xi**2 + 0.0625 * xi - 0.0625,
                                        0.0625 * x1 * x3,
                                        x3 * x4,
                                        -x1 * x2 * x4,
                                    ]
                                )

                        case 5:

                            def shape_functions(xi):
                                x0 = xi - 1.0
                                x1 = 2.0 * xi
                                x2 = x1 - 1.0
                                x3 = x0 * x2 * xi
                                x4 = x1 + 1.0
                                x5 = 0.16666666666666666667 * x4
                                x6 = xi + 1.0
                                x7 = x6 * xi
                                return jnp.stack(
                                    [
                                        x3 * x5,
                                        x2 * x5 * x7,
                                        -1.3333333333333333333 * x3 * x6,
                                        4.0 * xi**4 - 5.0 * xi**2 + 1.0,
                                        -1.3333333333333333333 * x0 * x4 * x7,
                                    ]
                                )

                        case 6:

                            def shape_functions(xi):
                                x0 = 5.0 * xi
                                x1 = x0 + 1.0
                                x2 = xi - 1.0
                                x3 = x0 - 1.0
                                x4 = x0 - 3.0
                                x5 = x1 * x2 * x3 * x4
                                x6 = x0 + 3.0
                                x7 = 0.0013020833333333333333 * x6
                                x8 = xi + 1.0
                                x9 = x3 * x4 * x8
                                x10 = 0.032552083333333333333 * x8
                                x11 = x2 * x6
                                x12 = 0.065104166666666666667 * x11
                                return jnp.stack(
                                    [
                                        -x5 * x7,
                                        x1 * x7 * x9,
                                        x10 * x5,
                                        -x12 * x9,
                                        x1 * x12 * x4 * x8,
                                        -x1 * x10 * x11 * x3,
                                    ]
                                )

                        case 7:

                            def shape_functions(xi):
                                x0 = 3.0 * xi
                                x1 = x0 + 1.0
                                x2 = xi - 1.0
                                x3 = x0 - 1.0
                                x4 = x0 - 2.0
                                x5 = x1 * x2 * x3 * x4 * xi
                                x6 = x0 + 2.0
                                x7 = 0.0125 * x6
                                x8 = xi + 1.0
                                x9 = x3 * x4 * x8 * xi
                                x10 = 0.225 * x8
                                x11 = x2 * x6
                                x12 = 0.5625 * x11
                                x13 = x1 * xi
                                return jnp.stack(
                                    [
                                        x5 * x7,
                                        x1 * x7 * x9,
                                        -x10 * x5,
                                        x12 * x9,
                                        -20.25 * xi**6 + 31.5 * xi**4 - 12.25 * xi**2 + 1.0,
                                        x12 * x13 * x4 * x8,
                                        -x10 * x11 * x13 * x3,
                                    ]
                                )

                        case 8:

                            def shape_functions(xi):
                                x0 = 7.0 * xi
                                x1 = x0 + 1.0
                                x2 = x0 + 3.0
                                x3 = xi - 1.0
                                x4 = x0 - 1.0
                                x5 = x0 - 3.0
                                x6 = x0 - 5.0
                                x7 = x1 * x2 * x3 * x4 * x5 * x6
                                x8 = x0 + 5.0
                                x9 = 0.000010850694444444444444 * x8
                                x10 = xi + 1.0
                                x11 = x1 * x10 * x4 * x5 * x6
                                x12 = 0.00053168402777777777778 * x10
                                x13 = x3 * x8
                                x14 = 0.0015950520833333333333 * x13
                                x15 = x10 * x2 * x4 * x6
                                x16 = x13 * x5
                                x17 = 0.0026584201388888888889 * x16
                                x18 = x1 * x2
                                return jnp.stack(
                                    [
                                        -x7 * x9,
                                        x11 * x2 * x9,
                                        x12 * x7,
                                        -x11 * x14,
                                        x15 * x17,
                                        -x10 * x17 * x18 * x6,
                                        x1 * x14 * x15,
                                        -x12 * x16 * x18 * x4,
                                    ]
                                )

                        case 9:

                            def shape_functions(xi):
                                x0 = 2.0 * xi
                                x1 = x0 + 1.0
                                x2 = 4.0 * xi
                                x3 = x2 + 1.0
                                x4 = xi - 1.0
                                x5 = x0 - 1.0
                                x6 = x2 - 1.0
                                x7 = x2 - 3.0
                                x8 = x1 * x3 * x4 * x5 * x6 * x7 * xi
                                x9 = x2 + 3.0
                                x10 = 0.0015873015873015873016 * x9
                                x11 = xi + 1.0
                                x12 = x11 * x3 * x5 * x6 * x7 * xi
                                x13 = 0.050793650793650793651 * x11
                                x14 = x4 * x9
                                x15 = 0.088888888888888888889 * x14
                                x16 = x1 * x11 * x6 * x7 * xi
                                x17 = x14 * x5
                                x18 = 0.35555555555555555556 * x17
                                x19 = x1 * x3 * xi
                                return jnp.stack(
                                    [
                                        x10 * x8,
                                        x1 * x10 * x12,
                                        -x13 * x8,
                                        x12 * x15,
                                        -x16 * x18,
                                        113.77777777777777778 * xi**8
                                        - 213.33333333333333333 * xi**6
                                        + 121.33333333333333333 * xi**4
                                        - 22.777777777777777778 * xi**2
                                        + 1.0,
                                        -x11 * x18 * x19 * x7,
                                        x15 * x16 * x3,
                                        -x13 * x17 * x19 * x6,
                                    ]
                                )

                        case 10:

                            def shape_functions(xi):
                                x0 = 3.0 * xi
                                x1 = x0 + 1.0
                                x2 = 9.0 * xi
                                x3 = x2 + 1.0
                                x4 = x2 + 5.0
                                x5 = xi - 1.0
                                x6 = x0 - 1.0
                                x7 = x2 - 1.0
                                x8 = x2 - 5.0
                                x9 = x2 - 7.0
                                x10 = x1 * x3 * x4 * x5 * x6 * x7 * x8 * x9
                                x11 = x2 + 7.0
                                x12 = 4.3596540178571428571e-7 * x11
                                x13 = xi + 1.0
                                x14 = x1 * x13 * x3 * x6 * x7 * x8 * x9
                                x15 = 0.000035313197544642857143 * x13
                                x16 = x11 * x5
                                x17 = 0.00014125279017857142857 * x16
                                x18 = x13 * x3 * x4 * x6 * x7 * x9
                                x19 = x16 * x8
                                x20 = 0.00010986328125 * x19
                                x21 = x1 * x13 * x4 * x7 * x9
                                x22 = x19 * x6
                                x23 = 0.000494384765625 * x22
                                x24 = x1 * x3 * x4
                                return jnp.stack(
                                    [
                                        -x10 * x12,
                                        x12 * x14 * x4,
                                        x10 * x15,
                                        -x14 * x17,
                                        x18 * x20,
                                        -x21 * x23,
                                        x13 * x23 * x24 * x9,
                                        -x20 * x21 * x3,
                                        x1 * x17 * x18,
                                        -x15 * x22 * x24 * x7,
                                    ]
                                )

                        case 11:

                            def shape_functions(xi):
                                x0 = 5.0 * xi
                                x1 = x0 + 1.0
                                x2 = x0 + 2.0
                                x3 = x0 + 3.0
                                x4 = xi - 1.0
                                x5 = x0 - 1.0
                                x6 = x0 - 2.0
                                x7 = x0 - 4.0
                                x8 = x0 - 3.0
                                x9 = x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * xi
                                x10 = x0 + 4.0
                                x11 = 6.8893298059964726631e-6 * x10
                                x12 = xi + 1.0
                                x13 = x1 * x12 * x2 * x5 * x6 * x7 * x8 * xi
                                x14 = 0.00034446649029982363316 * x12
                                x15 = x10 * x4
                                x16 = 0.0015500992063492063492 * x15
                                x17 = x1 * x12 * x3 * x5 * x6 * x7 * xi
                                x18 = x15 * x8
                                x19 = 0.0041335978835978835979 * x18
                                x20 = x12 * x2 * x3 * x5 * x7 * xi
                                x21 = x18 * x6
                                x22 = 0.0072337962962962962963 * x21
                                x23 = x1 * x2 * x3 * xi
                                return jnp.stack(
                                    [
                                        x11 * x9,
                                        x11 * x13 * x3,
                                        -x14 * x9,
                                        x13 * x16,
                                        -x17 * x19,
                                        x20 * x22,
                                        -678.16840277777777778 * xi**10
                                        + 1491.9704861111111111 * xi**8
                                        - 1110.0260416666666667 * xi**6
                                        + 331.81423611111111111 * xi**4
                                        - 36.590277777777777778 * xi**2
                                        + 1.0,
                                        x12 * x22 * x23 * x7,
                                        -x1 * x19 * x20,
                                        x16 * x17 * x2,
                                        -x14 * x21 * x23 * x5,
                                    ]
                                )

                        case _:
                            assert (
                                False
                            ), "Order of shape functions not implemented or number of nodes not adequat"
                case 2:
                    """The following shape functions were generated using the code below:

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
                                x0 = 0.5 * xi[0]
                                x1 = 0.5 - x0
                                x2 = 0.5 * xi[1]
                                x3 = 0.5 - x2
                                x4 = x0 + 0.5
                                x5 = x2 + 0.5
                                return jnp.stack([x1 * x3, x3 * x4, x4 * x5, x1 * x5])

                        case 9:

                            def shape_functions(xi):
                                x0 = xi[0] * (xi[0] - 1.0)
                                x1 = xi[1] * (xi[1] - 1.0)
                                x2 = 0.25 * x1
                                x3 = xi[0] * (xi[0] + 1.0)
                                x4 = xi[1] * (xi[1] + 1.0)
                                x5 = 0.25 * x4
                                x6 = 1.0 - xi[0] ** 2
                                x7 = 0.5 * x6
                                x8 = 1.0 - xi[1] ** 2
                                x9 = 0.5 * x8
                                return jnp.stack(
                                    [
                                        x0 * x2,
                                        x2 * x3,
                                        x3 * x5,
                                        x0 * x5,
                                        x1 * x7,
                                        x3 * x9,
                                        x4 * x7,
                                        x0 * x9,
                                        x6 * x8,
                                    ]
                                )

                        case 16:

                            def shape_functions(xi):
                                x0 = (
                                    -0.5625 * xi[0] ** 3
                                    + 0.5625 * xi[0] ** 2
                                    + 0.0625 * xi[0]
                                    - 0.0625
                                )
                                x1 = (
                                    -0.5625 * xi[1] ** 3
                                    + 0.5625 * xi[1] ** 2
                                    + 0.0625 * xi[1]
                                    - 0.0625
                                )
                                x2 = xi[0] + 1.0
                                x3 = 3.0 * xi[0]
                                x4 = x2 * (x3 + 1.0)
                                x5 = x3 - 1.0
                                x6 = x1 * x5
                                x7 = xi[1] + 1.0
                                x8 = 3.0 * xi[1]
                                x9 = x7 * (x8 - 1.0)
                                x10 = x5 * x9
                                x11 = x8 + 1.0
                                x12 = x11 * x4
                                x13 = x0 * x11
                                x14 = xi[0] - 1.0
                                x15 = 0.5625 * x14
                                x16 = xi[1] - 1.0
                                x17 = 0.03515625 * x16
                                x18 = x12 * x7
                                x19 = 0.03515625 * x14
                                x20 = x10 * x2
                                x21 = 0.5625 * x16
                                x22 = 0.31640625 * x14 * x16
                                return jnp.stack(
                                    [
                                        x0 * x1,
                                        0.0625 * x4 * x6,
                                        0.00390625 * x10 * x12,
                                        0.0625 * x13 * x9,
                                        x15 * x2 * x6,
                                        -x1 * x15 * x4,
                                        x10 * x17 * x4,
                                        -x17 * x18 * x5,
                                        -x12 * x19 * x9,
                                        x11 * x19 * x20,
                                        -x13 * x21 * x7,
                                        x0 * x21 * x9,
                                        x20 * x22,
                                        -x22 * x4 * x9,
                                        x18 * x22,
                                        -x11 * x2 * x22 * x5 * x7,
                                    ]
                                )

                        case 25:

                            def shape_functions(xi):
                                x0 = xi[0] - 1.0
                                x1 = xi[1] - 1.0
                                x2 = 2.0 * xi[1]
                                x3 = x2 + 1.0
                                x4 = x2 - 1.0
                                x5 = x3 * x4 * xi[1]
                                x6 = x1 * x5
                                x7 = 2.0 * xi[0]
                                x8 = x7 + 1.0
                                x9 = x7 - 1.0
                                x10 = x8 * x9 * xi[0]
                                x11 = 0.027777777777777777778 * x10
                                x12 = x11 * x6
                                x13 = xi[0] + 1.0
                                x14 = xi[1] + 1.0
                                x15 = x11 * x14 * x5
                                x16 = x0 * xi[0]
                                x17 = x16 * x9
                                x18 = 0.22222222222222222222 * x13
                                x19 = x18 * x6
                                x20 = 4.0 * xi[0] ** 4 - 5.0 * xi[0] ** 2 + 1.0
                                x21 = x1 * x20
                                x22 = 0.16666666666666666667 * x5
                                x23 = x16 * x8
                                x24 = x14 * xi[1]
                                x25 = x24 * x4
                                x26 = x1 * x10
                                x27 = x18 * x26
                                x28 = 4.0 * xi[1] ** 4 - 5.0 * xi[1] ** 2 + 1.0
                                x29 = x13 * x28
                                x30 = 0.16666666666666666667 * x10
                                x31 = x24 * x3
                                x32 = x14 * x18 * x5
                                x33 = 0.22222222222222222222 * x0 * x26
                                x34 = 1.7777777777777777778 * x1 * x13
                                x35 = x25 * x34
                                x36 = x31 * x34
                                x37 = 1.3333333333333333333 * x21
                                x38 = 1.3333333333333333333 * x29
                                return jnp.stack(
                                    [
                                        x0 * x12,
                                        x12 * x13,
                                        x13 * x15,
                                        x0 * x15,
                                        -x17 * x19,
                                        x21 * x22,
                                        -x19 * x23,
                                        -x25 * x27,
                                        x29 * x30,
                                        -x27 * x31,
                                        -x23 * x32,
                                        x14 * x20 * x22,
                                        -x17 * x32,
                                        -x31 * x33,
                                        x0 * x28 * x30,
                                        -x25 * x33,
                                        x17 * x35,
                                        x23 * x35,
                                        x23 * x36,
                                        x17 * x36,
                                        -x25 * x37,
                                        -x23 * x38,
                                        -x31 * x37,
                                        -x17 * x38,
                                        x20 * x28,
                                    ]
                                )

                        case 36:

                            def shape_functions(xi):
                                x0 = 5.0 * xi[1]
                                x1 = x0 + 3.0
                                x2 = xi[0] - 1.0
                                x3 = x1 * x2
                                x4 = 5.0 * xi[0]
                                x5 = x4 + 1.0
                                x6 = x0 + 1.0
                                x7 = x4 + 3.0
                                x8 = xi[1] - 1.0
                                x9 = x4 - 1.0
                                x10 = x0 - 1.0
                                x11 = x4 - 3.0
                                x12 = x0 - 3.0
                                x13 = x10 * x11 * x12 * x5 * x6 * x7 * x8 * x9
                                x14 = 1.6954210069444444444e-6 * x13
                                x15 = xi[0] + 1.0
                                x16 = x1 * x15
                                x17 = xi[1] + 1.0
                                x18 = x17 * x7
                                x19 = x16 * x18
                                x20 = x10 * x11 * x12 * x5 * x6 * x9
                                x21 = 1.6954210069444444444e-6 * x20
                                x22 = x18 * x3
                                x23 = 0.000042385525173611111111 * x15
                                x24 = x3 * x8
                                x25 = x23 * x24
                                x26 = x10 * x11 * x6 * x9
                                x27 = x24 * x26
                                x28 = x12 * x15
                                x29 = 0.000084771050347222222222 * x28
                                x30 = x29 * x7
                                x31 = x10 * x5
                                x32 = x11 * x6
                                x33 = x31 * x32
                                x34 = x12 * x9
                                x35 = x31 * x34
                                x36 = x35 * x6
                                x37 = x17 * x23
                                x38 = x19 * x8
                                x39 = 0.000084771050347222222222 * x38
                                x40 = x11 * x35
                                x41 = x32 * x5
                                x42 = x34 * x41
                                x43 = 0.000042385525173611111111 * x26 * x5
                                x44 = x22 * x29
                                x45 = x22 * x8
                                x46 = 0.000084771050347222222222 * x45
                                x47 = x17 * x2
                                x48 = 0.0010596381293402777778 * x8
                                x49 = x28 * x9
                                x50 = x31 * x6
                                x51 = x18 * x2
                                x52 = 0.0010596381293402777778 * x15
                                x53 = 0.0021192762586805555556 * x26
                                x54 = x28 * x51 * x8
                                x55 = 0.0021192762586805555556 * x33
                                x56 = 0.0021192762586805555556 * x49
                                x57 = x31 * x45
                                x58 = x15 * x45
                                x59 = x17 * x24 * x56
                                x60 = 0.0042385525173611111111 * x11
                                x61 = x45 * x49
                                return jnp.stack(
                                    [
                                        x14 * x3,
                                        -x14 * x16,
                                        x19 * x21,
                                        -x21 * x22,
                                        -x20 * x25,
                                        x27 * x30,
                                        -x24 * x30 * x33,
                                        x25 * x36 * x7,
                                        x13 * x37,
                                        -x39 * x40,
                                        x39 * x42,
                                        -x38 * x43,
                                        -x22 * x23 * x36,
                                        x33 * x44,
                                        -x26 * x44,
                                        x20 * x3 * x37,
                                        x43 * x45,
                                        -x42 * x46,
                                        x40 * x46,
                                        -0.000042385525173611111111 * x13 * x47,
                                        x15 * x20 * x47 * x48,
                                        -x48 * x49 * x50 * x51,
                                        x45 * x50 * x52 * x9,
                                        -x17 * x27 * x5 * x52,
                                        -x53 * x54,
                                        x54 * x55,
                                        x56 * x57,
                                        -x45 * x5 * x56 * x6,
                                        -x55 * x58,
                                        x53 * x58,
                                        x41 * x59,
                                        -x11 * x31 * x59,
                                        x10 * x60 * x61,
                                        -x28 * x57 * x60,
                                        0.0042385525173611111111 * x28 * x41 * x45,
                                        -0.0042385525173611111111 * x32 * x61,
                                    ]
                                )

                        case 49:

                            def shape_functions(xi):
                                x0 = xi[0] - 1.0
                                x1 = xi[1] - 1.0
                                x2 = 3.0 * xi[1]
                                x3 = x2 + 1.0
                                x4 = x2 + 2.0
                                x5 = x2 - 1.0
                                x6 = x2 - 2.0
                                x7 = x3 * x4 * x5 * x6 * xi[1]
                                x8 = x1 * x7
                                x9 = 3.0 * xi[0]
                                x10 = x9 + 1.0
                                x11 = x9 + 2.0
                                x12 = x9 - 1.0
                                x13 = x9 - 2.0
                                x14 = x10 * x11 * x12 * x13 * xi[0]
                                x15 = 0.00015625 * x14
                                x16 = x15 * x8
                                x17 = xi[0] + 1.0
                                x18 = xi[1] + 1.0
                                x19 = x15 * x18 * x7
                                x20 = x17 * x8
                                x21 = x0 * x10 * x12 * xi[0]
                                x22 = 0.0028125 * x21
                                x23 = x20 * x22
                                x24 = x0 * x13 * xi[0]
                                x25 = x12 * x24
                                x26 = 0.00703125 * x11
                                x27 = x20 * x26
                                x28 = (
                                    -20.25 * xi[0] ** 6
                                    + 31.5 * xi[0] ** 4
                                    - 12.25 * xi[0] ** 2
                                    + 1.0
                                )
                                x29 = x1 * x28
                                x30 = 0.0125 * x7
                                x31 = x10 * x24
                                x32 = x18 * x5 * x6 * xi[1]
                                x33 = x1 * x14
                                x34 = x17 * x33
                                x35 = x32 * x34
                                x36 = 0.0028125 * x3
                                x37 = 0.00703125 * x4
                                x38 = (
                                    -20.25 * xi[1] ** 6
                                    + 31.5 * xi[1] ** 4
                                    - 12.25 * xi[1] ** 2
                                    + 1.0
                                )
                                x39 = x17 * x38
                                x40 = 0.0125 * x14
                                x41 = x18 * x4 * xi[1]
                                x42 = x34 * x41
                                x43 = x3 * x6
                                x44 = 0.00703125 * x43
                                x45 = x36 * x5
                                x46 = x11 * x17
                                x47 = x18 * x7
                                x48 = x22 * x47
                                x49 = x17 * x26 * x47
                                x50 = x13 * x17
                                x51 = x0 * x33
                                x52 = x41 * x51
                                x53 = x32 * x51
                                x54 = x1 * x21
                                x55 = x50 * x54
                                x56 = x3 * x32
                                x57 = 0.050625 * x56
                                x58 = x46 * x54
                                x59 = x41 * x58
                                x60 = x3 * x5
                                x61 = 0.050625 * x60
                                x62 = x41 * x55
                                x63 = x1 * x46
                                x64 = 0.1265625 * x63
                                x65 = x56 * x64
                                x66 = x29 * x32
                                x67 = 0.225 * x3
                                x68 = x32 * x4
                                x69 = 0.1265625 * x68
                                x70 = x11 * x39
                                x71 = 0.225 * x21
                                x72 = 0.1265625 * x43
                                x73 = x31 * x41
                                x74 = x60 * x64
                                x75 = x29 * x41
                                x76 = x25 * x41
                                x77 = 0.31640625 * x63
                                x78 = x68 * x77
                                x79 = x43 * x77
                                x80 = 0.5625 * x70
                                return jnp.stack(
                                    [
                                        x0 * x16,
                                        x16 * x17,
                                        x17 * x19,
                                        x0 * x19,
                                        -x13 * x23,
                                        x25 * x27,
                                        x29 * x30,
                                        x27 * x31,
                                        -x11 * x23,
                                        -x35 * x36,
                                        x35 * x37,
                                        x39 * x40,
                                        x42 * x44,
                                        -x42 * x45,
                                        -x46 * x48,
                                        x31 * x49,
                                        x18 * x28 * x30,
                                        x25 * x49,
                                        -x48 * x50,
                                        -x45 * x52,
                                        x44 * x52,
                                        x0 * x38 * x40,
                                        x37 * x53,
                                        -x36 * x53,
                                        x55 * x57,
                                        x57 * x58,
                                        x59 * x61,
                                        x61 * x62,
                                        -x25 * x65,
                                        -x66 * x67,
                                        -x31 * x65,
                                        -x58 * x69,
                                        -x70 * x71,
                                        -x59 * x72,
                                        -x73 * x74,
                                        -x5 * x67 * x75,
                                        -x74 * x76,
                                        -x62 * x72,
                                        -x13 * x39 * x71,
                                        -x55 * x69,
                                        x25 * x78,
                                        x31 * x78,
                                        x73 * x79,
                                        x76 * x79,
                                        0.5625 * x4 * x66,
                                        x31 * x80,
                                        0.5625 * x43 * x75,
                                        x25 * x80,
                                        x28 * x38,
                                    ]
                                )

                        case 64:

                            def shape_functions(xi):
                                x0 = 7.0 * xi[1]
                                x1 = x0 + 5.0
                                x2 = xi[0] - 1.0
                                x3 = x1 * x2
                                x4 = 7.0 * xi[0]
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
                                x17 = (
                                    x10
                                    * x11
                                    * x12
                                    * x13
                                    * x14
                                    * x15
                                    * x16
                                    * x5
                                    * x6
                                    * x7
                                    * x8
                                    * x9
                                )
                                x18 = 1.1773756992669753086e-10 * x17
                                x19 = xi[0] + 1.0
                                x20 = x1 * x19
                                x21 = xi[1] + 1.0
                                x22 = x21 * x9
                                x23 = x20 * x22
                                x24 = x11 * x12 * x13 * x14 * x15 * x16 * x5 * x6 * x7 * x8
                                x25 = 1.1773756992669753086e-10 * x24
                                x26 = x22 * x3
                                x27 = 5.7691409264081790123e-9 * x19
                                x28 = x10 * x3
                                x29 = x27 * x28
                                x30 = x11 * x12 * x13 * x14 * x15 * x16 * x5 * x8
                                x31 = x28 * x30
                                x32 = x19 * x6
                                x33 = 1.7307422779224537037e-8 * x32
                                x34 = x33 * x9
                                x35 = x11 * x12 * x13 * x14 * x15 * x16
                                x36 = x28 * x35
                                x37 = 2.8845704632040895062e-8 * x7
                                x38 = x32 * x8
                                x39 = x37 * x38
                                x40 = x39 * x9
                                x41 = x12 * x13 * x14 * x5
                                x42 = x15 * x16
                                x43 = x28 * x42
                                x44 = x11 * x8
                                x45 = x14 * x44
                                x46 = x5 * x7
                                x47 = x12 * x46
                                x48 = x43 * x47
                                x49 = x44 * x6
                                x50 = x41 * x7
                                x51 = x16 * x50
                                x52 = x49 * x51
                                x53 = x21 * x27
                                x54 = x10 * x23
                                x55 = x54 * x6
                                x56 = 1.7307422779224537037e-8 * x35 * x46
                                x57 = x30 * x37
                                x58 = x13 * x42
                                x59 = x5 * x58
                                x60 = x37 * x45 * x59
                                x61 = x49 * x54
                                x62 = x47 * x58
                                x63 = 1.7307422779224537037e-8 * x62
                                x64 = x15 * x50
                                x65 = 5.7691409264081790123e-9 * x64
                                x66 = x26 * x42
                                x67 = x47 * x66
                                x68 = x45 * x67
                                x69 = x26 * x35
                                x70 = x26 * x30
                                x71 = x10 * x26
                                x72 = x49 * x71
                                x73 = x6 * x71
                                x74 = x2 * x21
                                x75 = 2.826879053940007716e-7 * x10
                                x76 = x11 * x38
                                x77 = x2 * x22
                                x78 = x76 * x77
                                x79 = 2.826879053940007716e-7 * x76
                                x80 = x10 * x77
                                x81 = 8.4806371618200231481e-7 * x32
                                x82 = 1.413439526970003858e-6 * x38
                                x83 = x80 * x82
                                x84 = 8.4806371618200231481e-7 * x47
                                x85 = x10 * x14
                                x86 = x51 * x71
                                x87 = 1.413439526970003858e-6 * x19
                                x88 = x13 * x16
                                x89 = x71 * x76
                                x90 = x14 * x89
                                x91 = 1.413439526970003858e-6 * x90
                                x92 = x12 * x7
                                x93 = x13 * x21 * x76
                                x94 = x14 * x46
                                x95 = 2.5441911485460069444e-6 * x32
                                x96 = x10 * x69
                                x97 = 4.2403185809100115741e-6 * x32
                                x98 = x7 * x96
                                x99 = x10 * x50 * x66
                                x100 = 4.2403185809100115741e-6 * x10 * x19
                                x101 = x38 * x71
                                x102 = 7.0671976348500192901e-6 * x19 * x8
                                x103 = 7.0671976348500192901e-6 * x58
                                return jnp.stack(
                                    [
                                        x18 * x3,
                                        -x18 * x20,
                                        x23 * x25,
                                        -x25 * x26,
                                        -x24 * x29,
                                        x31 * x34,
                                        -x36 * x40,
                                        x40 * x41 * x43,
                                        -x34 * x45 * x48,
                                        x29 * x52 * x9,
                                        x17 * x53,
                                        -x55 * x56,
                                        x54 * x57,
                                        -x55 * x60,
                                        x61 * x63,
                                        -x61 * x65,
                                        -x26 * x27 * x52,
                                        x33 * x68,
                                        -x39 * x41 * x66,
                                        x39 * x69,
                                        -x33 * x70,
                                        x24 * x3 * x53,
                                        x65 * x72,
                                        -x63 * x72,
                                        x60 * x73,
                                        -x57 * x71,
                                        x56 * x73,
                                        -5.7691409264081790123e-9 * x17 * x74,
                                        x19 * x24 * x74 * x75,
                                        -x51 * x75 * x78,
                                        x50 * x71 * x79,
                                        -x21 * x28 * x64 * x79,
                                        -x30 * x80 * x81,
                                        x35 * x7 * x83,
                                        -x42 * x50 * x83,
                                        x42 * x78 * x84 * x85,
                                        x11 * x81 * x86,
                                        -x44 * x86 * x87,
                                        x46 * x88 * x91,
                                        -x84 * x88 * x89,
                                        -x15 * x84 * x90,
                                        x64 * x71 * x82,
                                        -x13 * x15 * x91 * x92,
                                        8.4806371618200231481e-7 * x15 * x41 * x89,
                                        8.4806371618200231481e-7 * x48 * x93,
                                        -1.413439526970003858e-6 * x43 * x93 * x94,
                                        x21 * x31 * x7 * x87,
                                        -x21 * x36 * x46 * x81,
                                        x5 * x95 * x96,
                                        -x11 * x67 * x85 * x95,
                                        2.5441911485460069444e-6 * x10 * x67 * x76,
                                        -2.5441911485460069444e-6 * x12 * x59 * x89,
                                        -x97 * x98,
                                        x97 * x99,
                                        x100 * x68,
                                        -4.2403185809100115741e-6 * x46 * x66 * x76 * x85,
                                        -4.2403185809100115741e-6 * x101 * x62,
                                        4.2403185809100115741e-6 * x58 * x89 * x92,
                                        4.2403185809100115741e-6 * x59 * x90,
                                        -x100 * x70,
                                        x102 * x98,
                                        -x102 * x99,
                                        x101 * x103 * x94,
                                        -x103 * x7 * x90,
                                    ]
                                )

                        case 81:

                            def shape_functions(xi):
                                x0 = xi[0] - 1.0
                                x1 = xi[1] - 1.0
                                x2 = 2.0 * xi[1]
                                x3 = x2 + 1.0
                                x4 = 4.0 * xi[1]
                                x5 = x4 + 1.0
                                x6 = x4 + 3.0
                                x7 = x2 - 1.0
                                x8 = x4 - 1.0
                                x9 = x4 - 3.0
                                x10 = x3 * x5 * x6 * x7 * x8 * x9 * xi[1]
                                x11 = x1 * x10
                                x12 = 2.0 * xi[0]
                                x13 = x12 + 1.0
                                x14 = 4.0 * xi[0]
                                x15 = x14 + 1.0
                                x16 = x14 + 3.0
                                x17 = x12 - 1.0
                                x18 = x14 - 1.0
                                x19 = x14 - 3.0
                                x20 = x13 * x15 * x16 * x17 * x18 * x19 * xi[0]
                                x21 = 2.5195263290501385739e-6 * x20
                                x22 = x11 * x21
                                x23 = xi[0] + 1.0
                                x24 = xi[1] + 1.0
                                x25 = x10 * x21 * x24
                                x26 = x11 * x23
                                x27 = x0 * x13 * x15 * x17 * x18 * xi[0]
                                x28 = 0.000080624842529604434366 * x27
                                x29 = x26 * x28
                                x30 = x16 * x26
                                x31 = x0 * x15 * x18 * x19 * xi[0]
                                x32 = 0.00014109347442680776014 * x31
                                x33 = x30 * x32
                                x34 = x0 * x17 * x19 * xi[0]
                                x35 = x18 * x34
                                x36 = 0.00056437389770723104056 * x13
                                x37 = x30 * x36
                                x38 = (
                                    113.77777777777777778 * xi[0] ** 8
                                    - 213.33333333333333333 * xi[0] ** 6
                                    + 121.33333333333333333 * xi[0] ** 4
                                    - 22.777777777777777778 * xi[0] ** 2
                                    + 1.0
                                )
                                x39 = x1 * x38
                                x40 = 0.0015873015873015873016 * x10
                                x41 = x15 * x34
                                x42 = x24 * x5 * x7 * x8 * x9 * xi[1]
                                x43 = x1 * x20
                                x44 = x23 * x43
                                x45 = x42 * x44
                                x46 = 0.000080624842529604434366 * x3
                                x47 = 0.00014109347442680776014 * x6
                                x48 = x24 * x6 * x7 * x8 * xi[1]
                                x49 = x3 * x9
                                x50 = x48 * x49
                                x51 = 0.00056437389770723104056 * x44
                                x52 = (
                                    113.77777777777777778 * xi[1] ** 8
                                    - 213.33333333333333333 * xi[1] ** 6
                                    + 121.33333333333333333 * xi[1] ** 4
                                    - 22.777777777777777778 * xi[1] ** 2
                                    + 1.0
                                )
                                x53 = x23 * x52
                                x54 = 0.0015873015873015873016 * x20
                                x55 = x6 * x7
                                x56 = x24 * x49 * xi[1]
                                x57 = x5 * x56
                                x58 = x55 * x57
                                x59 = x57 * x8
                                x60 = x47 * x59
                                x61 = x48 * x5
                                x62 = x46 * x61
                                x63 = x16 * x23
                                x64 = x10 * x24 * x63
                                x65 = x32 * x64
                                x66 = x36 * x64
                                x67 = x19 * x23
                                x68 = x0 * x43
                                x69 = 0.00056437389770723104056 * x68
                                x70 = x42 * x68
                                x71 = x1 * x27
                                x72 = x67 * x71
                                x73 = x3 * x42
                                x74 = 0.0025799949609473418997 * x73
                                x75 = x63 * x71
                                x76 = x3 * x61
                                x77 = 0.0025799949609473418997 * x76
                                x78 = x1 * x63
                                x79 = x73 * x78
                                x80 = 0.0045149911816578483245 * x31
                                x81 = x79 * x80
                                x82 = 0.018059964726631393298 * x13
                                x83 = x79 * x82
                                x84 = x39 * x42
                                x85 = 0.050793650793650793651 * x3
                                x86 = 0.0045149911816578483245 * x6
                                x87 = x75 * x86
                                x88 = 0.018059964726631393298 * x75
                                x89 = x16 * x53
                                x90 = 0.050793650793650793651 * x27
                                x91 = x76 * x78
                                x92 = x80 * x91
                                x93 = x82 * x91
                                x94 = x39 * x5
                                x95 = x72 * x86
                                x96 = 0.018059964726631393298 * x72
                                x97 = x31 * x78
                                x98 = x17 * x97
                                x99 = x42 * x6
                                x100 = 0.0079012345679012345679 * x99
                                x101 = x13 * x97
                                x102 = x59 * x6
                                x103 = 0.0079012345679012345679 * x102
                                x104 = x13 * x78
                                x105 = 0.031604938271604938272 * x104
                                x106 = x105 * x99
                                x107 = 0.088888888888888888889 * x6
                                x108 = 0.031604938271604938272 * x101
                                x109 = x13 * x89
                                x110 = 0.088888888888888888889 * x31
                                x111 = x102 * x105
                                x112 = x56 * x94
                                x113 = 0.031604938271604938272 * x98
                                x114 = 0.12641975308641975309 * x104
                                x115 = x114 * x50
                                x116 = x114 * x58
                                x117 = 0.35555555555555555556 * x109
                                return jnp.stack(
                                    [
                                        x0 * x22,
                                        x22 * x23,
                                        x23 * x25,
                                        x0 * x25,
                                        -x19 * x29,
                                        x17 * x33,
                                        -x35 * x37,
                                        x39 * x40,
                                        -x37 * x41,
                                        x13 * x33,
                                        -x16 * x29,
                                        -x45 * x46,
                                        x45 * x47,
                                        -x50 * x51,
                                        x53 * x54,
                                        -x51 * x58,
                                        x44 * x60,
                                        -x44 * x62,
                                        -x28 * x64,
                                        x13 * x65,
                                        -x41 * x66,
                                        x24 * x38 * x40,
                                        -x35 * x66,
                                        x17 * x65,
                                        -x10 * x24 * x28 * x67,
                                        -x62 * x68,
                                        x60 * x68,
                                        -x58 * x69,
                                        x0 * x52 * x54,
                                        -x50 * x69,
                                        x47 * x70,
                                        -x46 * x70,
                                        x72 * x74,
                                        x74 * x75,
                                        x75 * x77,
                                        x72 * x77,
                                        -x17 * x81,
                                        x35 * x83,
                                        -x84 * x85,
                                        x41 * x83,
                                        -x13 * x81,
                                        -x42 * x87,
                                        x50 * x88,
                                        -x89 * x90,
                                        x58 * x88,
                                        -x59 * x87,
                                        -x13 * x92,
                                        x41 * x93,
                                        -x48 * x85 * x94,
                                        x35 * x93,
                                        -x17 * x92,
                                        -x59 * x95,
                                        x58 * x96,
                                        -x19 * x53 * x90,
                                        x50 * x96,
                                        -x42 * x95,
                                        x100 * x98,
                                        x100 * x101,
                                        x101 * x103,
                                        x103 * x98,
                                        -x106 * x35,
                                        x107 * x84,
                                        -x106 * x41,
                                        -x108 * x50,
                                        x109 * x110,
                                        -x108 * x58,
                                        -x111 * x41,
                                        x107 * x112 * x8,
                                        -x111 * x35,
                                        -x113 * x58,
                                        x110 * x17 * x89,
                                        -x113 * x50,
                                        x115 * x35,
                                        x115 * x41,
                                        x116 * x41,
                                        x116 * x35,
                                        -0.35555555555555555556 * x39 * x50,
                                        -x117 * x41,
                                        -0.35555555555555555556 * x112 * x55,
                                        -x117 * x35,
                                        x38 * x52,
                                    ]
                                )

                        case 100:

                            def shape_functions(xi):
                                x0 = 9.0 * xi[1]
                                x1 = x0 + 7.0
                                x2 = xi[0] - 1.0
                                x3 = x1 * x2
                                x4 = 3.0 * xi[0]
                                x5 = x4 + 1.0
                                x6 = 3.0 * xi[1]
                                x7 = x6 + 1.0
                                x8 = 9.0 * xi[0]
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
                                x23 = (
                                    x10
                                    * x11
                                    * x12
                                    * x13
                                    * x14
                                    * x15
                                    * x16
                                    * x17
                                    * x18
                                    * x19
                                    * x20
                                    * x21
                                    * x22
                                    * x5
                                    * x7
                                    * x9
                                )
                                x24 = 1.900658315541792889e-13 * x23
                                x25 = xi[0] + 1.0
                                x26 = x1 * x25
                                x27 = xi[1] + 1.0
                                x28 = x13 * x27
                                x29 = x26 * x28
                                x30 = (
                                    x10
                                    * x11
                                    * x12
                                    * x15
                                    * x16
                                    * x17
                                    * x18
                                    * x19
                                    * x20
                                    * x21
                                    * x22
                                    * x5
                                    * x7
                                    * x9
                                )
                                x31 = 1.900658315541792889e-13 * x30
                                x32 = x28 * x3
                                x33 = 1.5395332355888522401e-11 * x25
                                x34 = x14 * x3
                                x35 = x33 * x34
                                x36 = (
                                    x10
                                    * x12
                                    * x15
                                    * x16
                                    * x17
                                    * x18
                                    * x19
                                    * x20
                                    * x21
                                    * x22
                                    * x5
                                    * x9
                                )
                                x37 = x34 * x36
                                x38 = x25 * x7
                                x39 = 6.1581329423554089605e-11 * x38
                                x40 = x13 * x39
                                x41 = x10 * x15 * x16 * x17 * x18 * x19 * x20 * x21 * x22 * x9
                                x42 = x34 * x41
                                x43 = 4.7896589551653180804e-11 * x11
                                x44 = x12 * x38
                                x45 = x43 * x44
                                x46 = x13 * x45
                                x47 = x10 * x15 * x17 * x18 * x19 * x20 * x21 * x22
                                x48 = x11 * x5
                                x49 = x44 * x48
                                x50 = x47 * x49
                                x51 = 2.1553465298243931362e-10 * x50
                                x52 = x16 * x34
                                x53 = x13 * x52
                                x54 = x10 * x18 * x19 * x20 * x21 * x22 * x9
                                x55 = x15 * x49
                                x56 = x54 * x55
                                x57 = 2.1553465298243931362e-10 * x56
                                x58 = x17 * x52
                                x59 = x5 * x54
                                x60 = x10 * x18 * x20
                                x61 = x21 * x60
                                x62 = x58 * x61
                                x63 = x12 * x9
                                x64 = x15 * x48
                                x65 = x22 * x64
                                x66 = x63 * x65
                                x67 = x19 * x60
                                x68 = x16 * x17
                                x69 = x66 * x68
                                x70 = x69 * x7
                                x71 = x67 * x70
                                x72 = x27 * x33
                                x73 = x14 * x29
                                x74 = x7 * x73
                                x75 = 6.1581329423554089605e-11 * x41 * x48
                                x76 = x36 * x43
                                x77 = x18 * x21
                                x78 = x70 * x77
                                x79 = x73 * x78
                                x80 = x19 * x20
                                x81 = 2.1553465298243931362e-10 * x80
                                x82 = x10 * x21
                                x83 = x70 * x81 * x82
                                x84 = x63 * x74
                                x85 = x47 * x5
                                x86 = x43 * x85
                                x87 = x10 * x19
                                x88 = 6.1581329423554089605e-11 * x87
                                x89 = x61 * x68
                                x90 = x19 * x89
                                x91 = 1.5395332355888522401e-11 * x64 * x90
                                x92 = x32 * x39
                                x93 = x61 * x69
                                x94 = x32 * x45
                                x95 = x16 * x32
                                x96 = x14 * x32
                                x97 = x7 * x96
                                x98 = x63 * x97
                                x99 = x78 * x96
                                x100 = x2 * x27
                                x101 = 1.2470219208269703145e-9 * x14
                                x102 = x2 * x28
                                x103 = x102 * x68
                                x104 = x55 * x9
                                x105 = x104 * x22
                                x106 = x105 * x67
                                x107 = 1.2470219208269703145e-9 * x104
                                x108 = x67 * x96
                                x109 = x108 * x68
                                x110 = x102 * x14
                                x111 = 4.988087683307881258e-9 * x38
                                x112 = 3.8796237536839076451e-9 * x11
                                x113 = x112 * x44
                                x114 = 1.7458306891577584403e-8 * x110 * x16
                                x115 = 3.8796237536839076451e-9 * x49
                                x116 = x14 * x54
                                x117 = 4.988087683307881258e-9 * x105
                                x118 = x65 * x9
                                x119 = x18 * x68
                                x120 = 1.7458306891577584403e-8 * x96
                                x121 = x105 * x80
                                x122 = x120 * x121
                                x123 = x10 * x68
                                x124 = x17 * x96
                                x125 = x117 * x87
                                x126 = 4.988087683307881258e-9 * x96
                                x127 = x9 * x90
                                x128 = x127 * x96
                                x129 = x14 * x95
                                x130 = x15 * x44
                                x131 = x130 * x5
                                x132 = x27 * x58
                                x133 = x132 * x77
                                x134 = 1.7458306891577584403e-8 * x121
                                x135 = 1.9952350733231525032e-8 * x96
                                x136 = x135 * x38
                                x137 = x135 * x77
                                x138 = x22 * x68
                                x139 = x138 * x87
                                x140 = x139 * x9
                                x141 = 1.551849501473563058e-8 * x96
                                x142 = x141 * x38
                                x143 = x11 * x41
                                x144 = 6.9833227566310337612e-8 * x38
                                x145 = x129 * x47 * x48
                                x146 = x116 * x64 * x95
                                x147 = x48 * x54 * x68
                                x148 = x141 * x25
                                x149 = x105 * x77
                                x150 = 6.9833227566310337612e-8 * x96
                                x151 = x150 * x20 * x68
                                x152 = x49 * x77
                                x153 = x140 * x141
                                x154 = x150 * x77
                                x155 = x11 * x130
                                x156 = x155 * x77
                                x157 = x44 * x9
                                x158 = x138 * x80
                                x159 = x158 * x9
                                x160 = x131 * x159
                                x161 = 1.2069940567016601562e-8 * x96
                                x162 = x12 * x25
                                x163 = x161 * x162
                                x164 = 5.4314732551574707031e-8 * x162
                                x165 = 5.4314732551574707031e-8 * x96
                                x166 = x159 * x165
                                x167 = x166 * x82
                                x168 = 2.4441629648208618164e-7 * x77
                                x169 = x158 * x55 * x96
                                x170 = x121 * x129
                                x171 = 2.4441629648208618164e-7 * x82
                                return jnp.stack(
                                    [
                                        x24 * x3,
                                        -x24 * x26,
                                        x29 * x31,
                                        -x31 * x32,
                                        -x30 * x35,
                                        x37 * x40,
                                        -x42 * x46,
                                        x51 * x53,
                                        -x53 * x57,
                                        x46 * x58 * x59,
                                        -x40 * x62 * x66,
                                        x13 * x35 * x71,
                                        x23 * x72,
                                        -x74 * x75,
                                        x73 * x76,
                                        -x79 * x81,
                                        x73 * x83,
                                        -x84 * x86,
                                        x79 * x88,
                                        -x84 * x91,
                                        -x32 * x33 * x71,
                                        x92 * x93,
                                        -x59 * x68 * x94,
                                        x57 * x95,
                                        -x51 * x95,
                                        x41 * x94,
                                        -x36 * x92,
                                        x3 * x30 * x72,
                                        x91 * x98,
                                        -x88 * x99,
                                        x86 * x98,
                                        -x83 * x96,
                                        x81 * x99,
                                        -x76 * x96,
                                        x75 * x97,
                                        -1.5395332355888522401e-11 * x100 * x23,
                                        x100 * x101 * x25 * x30,
                                        -x101 * x103 * x106,
                                        x107 * x109,
                                        -x107 * x19 * x27 * x62,
                                        -x110 * x111 * x36,
                                        x110 * x113 * x41,
                                        -x114 * x50,
                                        x114 * x56,
                                        -x103 * x115 * x116,
                                        x110 * x117 * x89,
                                        x109 * x111 * x118,
                                        -3.8796237536839076451e-9 * x108 * x25 * x69,
                                        x119 * x122,
                                        -x122 * x123,
                                        3.8796237536839076451e-9 * x106 * x124,
                                        -x119 * x125 * x96,
                                        -x104 * x126 * x89,
                                        x115 * x128,
                                        -1.7458306891577584403e-8 * x104 * x129 * x19 * x61,
                                        x120 * x55 * x90,
                                        -x113 * x128 * x15,
                                        x126 * x127 * x131,
                                        x125 * x133,
                                        -3.8796237536839076451e-9 * x27 * x34 * x50 * x9,
                                        x132 * x134 * x82,
                                        -x133 * x134,
                                        x112 * x25 * x27 * x37,
                                        -x111 * x27 * x42 * x48,
                                        x136 * x41 * x5,
                                        -x118 * x136 * x89,
                                        x105 * x123 * x137,
                                        -x131 * x137 * x140,
                                        -x142 * x143,
                                        x144 * x145,
                                        -x144 * x146,
                                        x142 * x147,
                                        x148 * x93,
                                        -x149 * x151,
                                        x105 * x151 * x82,
                                        -1.551849501473563058e-8 * x105 * x124 * x61,
                                        -x152 * x153,
                                        6.9833227566310337612e-8 * x129 * x149 * x87,
                                        -x139 * x154 * x55,
                                        x153 * x156,
                                        x141 * x157 * x85,
                                        -x150 * x160 * x82,
                                        x154 * x160,
                                        -x148 * x36,
                                        x143 * x163,
                                        -x147 * x163,
                                        1.2069940567016601562e-8 * x124 * x49 * x54,
                                        -x11 * x157 * x161 * x47,
                                        -x145 * x164,
                                        x146 * x164,
                                        x152 * x166,
                                        -x167 * x49,
                                        -x165 * x56,
                                        x165 * x50,
                                        x155 * x167,
                                        -x156 * x166,
                                        x168 * x169,
                                        -x168 * x170,
                                        x170 * x171,
                                        -x169 * x171,
                                    ]
                                )

                        case 121:

                            def shape_functions(xi):
                                x0 = xi[0] - 1.0
                                x1 = xi[1] - 1.0
                                x2 = 5.0 * xi[1]
                                x3 = x2 + 1.0
                                x4 = x2 + 2.0
                                x5 = x2 + 4.0
                                x6 = x2 + 3.0
                                x7 = x2 - 1.0
                                x8 = x2 - 2.0
                                x9 = x2 - 4.0
                                x10 = x2 - 3.0
                                x11 = x10 * x3 * x4 * x5 * x6 * x7 * x8 * x9 * xi[1]
                                x12 = x1 * x11
                                x13 = 5.0 * xi[0]
                                x14 = x13 + 1.0
                                x15 = x13 + 2.0
                                x16 = x13 + 4.0
                                x17 = x13 + 3.0
                                x18 = x13 - 1.0
                                x19 = x13 - 2.0
                                x20 = x13 - 4.0
                                x21 = x13 - 3.0
                                x22 = x14 * x15 * x16 * x17 * x18 * x19 * x20 * x21 * xi[0]
                                x23 = 4.7462865175791395662e-11 * x22
                                x24 = x12 * x23
                                x25 = xi[0] + 1.0
                                x26 = xi[1] + 1.0
                                x27 = x11 * x23 * x26
                                x28 = x12 * x25
                                x29 = x0 * x14 * x15 * x17 * x18 * x19 * x21 * xi[0]
                                x30 = 2.3731432587895697831e-9 * x29
                                x31 = x28 * x30
                                x32 = x16 * x28
                                x33 = x0 * x14 * x15 * x18 * x19 * x20 * xi[0]
                                x34 = 1.0679144664553064024e-8 * x33
                                x35 = x32 * x34
                                x36 = x17 * x32
                                x37 = x0 * x14 * x18 * x20 * x21 * xi[0]
                                x38 = 2.8477719105474837397e-8 * x37
                                x39 = x36 * x38
                                x40 = x0 * x19 * x20 * x21 * xi[0]
                                x41 = x18 * x40
                                x42 = 4.9836008434580965445e-8 * x15
                                x43 = x36 * x42
                                x44 = (
                                    -678.16840277777777778 * xi[0] ** 10
                                    + 1491.9704861111111111 * xi[0] ** 8
                                    - 1110.0260416666666667 * xi[0] ** 6
                                    + 331.81423611111111111 * xi[0] ** 4
                                    - 36.590277777777777778 * xi[0] ** 2
                                    + 1.0
                                )
                                x45 = x1 * x44
                                x46 = 6.8893298059964726631e-6 * x11
                                x47 = x14 * x40
                                x48 = x10 * x26 * x3 * x4 * x7 * x8 * x9 * xi[1]
                                x49 = x1 * x22
                                x50 = x25 * x49
                                x51 = x48 * x50
                                x52 = 2.3731432587895697831e-9 * x6
                                x53 = 1.0679144664553064024e-8 * x5
                                x54 = x10 * x26 * x3 * x5 * x7 * x8 * xi[1]
                                x55 = x50 * x54
                                x56 = x6 * x9
                                x57 = 2.8477719105474837397e-8 * x56
                                x58 = x26 * x56 * x7 * x8 * xi[1]
                                x59 = x10 * x5
                                x60 = x58 * x59
                                x61 = x4 * x50
                                x62 = 4.9836008434580965445e-8 * x61
                                x63 = (
                                    -678.16840277777777778 * xi[1] ** 10
                                    + 1491.9704861111111111 * xi[1] ** 8
                                    - 1110.0260416666666667 * xi[1] ** 6
                                    + 331.81423611111111111 * xi[1] ** 4
                                    - 36.590277777777777778 * xi[1] ** 2
                                    + 1.0
                                )
                                x64 = x25 * x63
                                x65 = 6.8893298059964726631e-6 * x22
                                x66 = x56 * x8
                                x67 = x26 * x59 * xi[1]
                                x68 = x3 * x67
                                x69 = x66 * x68
                                x70 = x68 * x7
                                x71 = x57 * x70
                                x72 = x3 * x58
                                x73 = x53 * x72
                                x74 = x4 * x52
                                x75 = x16 * x25
                                x76 = x11 * x26 * x75
                                x77 = x17 * x76
                                x78 = x38 * x77
                                x79 = x42 * x77
                                x80 = x20 * x25
                                x81 = x0 * x49
                                x82 = x54 * x81
                                x83 = x4 * x81
                                x84 = 4.9836008434580965445e-8 * x83
                                x85 = x48 * x81
                                x86 = x1 * x29
                                x87 = x80 * x86
                                x88 = x48 * x6
                                x89 = 1.1865716293947848916e-7 * x88
                                x90 = x75 * x86
                                x91 = x54 * x90
                                x92 = x4 * x6
                                x93 = 1.1865716293947848916e-7 * x92
                                x94 = x54 * x87
                                x95 = x1 * x75
                                x96 = x88 * x95
                                x97 = 5.339572332276532012e-7 * x33
                                x98 = x96 * x97
                                x99 = x17 * x96
                                x100 = 1.4238859552737418699e-6 * x37
                                x101 = x100 * x99
                                x102 = 2.4918004217290482723e-6 * x15
                                x103 = x102 * x99
                                x104 = x45 * x48
                                x105 = 0.00034446649029982363316 * x6
                                x106 = 5.339572332276532012e-7 * x5
                                x107 = x106 * x90
                                x108 = 1.4238859552737418699e-6 * x56
                                x109 = x4 * x90
                                x110 = 2.4918004217290482723e-6 * x109
                                x111 = x16 * x64
                                x112 = 0.00034446649029982363316 * x29
                                x113 = x108 * x70
                                x114 = x4 * x72
                                x115 = x17 * x95
                                x116 = x115 * x54
                                x117 = x116 * x92
                                x118 = x100 * x117
                                x119 = x102 * x117
                                x120 = x4 * x45
                                x121 = x21 * x95
                                x122 = x121 * x54
                                x123 = x106 * x87
                                x124 = x4 * x87
                                x125 = 2.4918004217290482723e-6 * x124
                                x126 = x121 * x33
                                x127 = x48 * x5
                                x128 = 2.4028075495244394054e-6 * x127
                                x129 = x115 * x33
                                x130 = x114 * x5
                                x131 = 2.4028075495244394054e-6 * x130
                                x132 = x19 * x37
                                x133 = x115 * x127
                                x134 = 6.4074867987318384144e-6 * x133
                                x135 = 0.000011213101897780717225 * x15
                                x136 = x133 * x135
                                x137 = 0.0015500992063492063492 * x5
                                x138 = x15 * x37
                                x139 = 6.4074867987318384144e-6 * x56
                                x140 = x139 * x33
                                x141 = x129 * x4
                                x142 = 0.000011213101897780717225 * x141
                                x143 = x111 * x17
                                x144 = 0.0015500992063492063492 * x33
                                x145 = x139 * x70
                                x146 = x115 * x130
                                x147 = 6.4074867987318384144e-6 * x146
                                x148 = x135 * x146
                                x149 = x120 * x3
                                x150 = x126 * x4
                                x151 = 0.000011213101897780717225 * x150
                                x152 = 0.000017086631463284902438 * x56
                                x153 = x116 * x152
                                x154 = x115 * x4
                                x155 = x138 * x154
                                x156 = x152 * x70
                                x157 = x132 * x154
                                x158 = x15 * x41
                                x159 = 0.000029901605060748579267 * x56
                                x160 = x116 * x159
                                x161 = 0.0041335978835978835979 * x56
                                x162 = x15 * x47
                                x163 = 0.000029901605060748579267 * x155
                                x164 = x143 * x15
                                x165 = 0.0041335978835978835979 * x37
                                x166 = x154 * x162
                                x167 = x159 * x70
                                x168 = x149 * x67
                                x169 = x154 * x158
                                x170 = 0.000029901605060748579267 * x157
                                x171 = 0.000052327808856310013717 * x60
                                x172 = 0.000052327808856310013717 * x69
                                x173 = 0.0072337962962962962963 * x164
                                return jnp.stack(
                                    [
                                        x0 * x24,
                                        x24 * x25,
                                        x25 * x27,
                                        x0 * x27,
                                        -x20 * x31,
                                        x21 * x35,
                                        -x19 * x39,
                                        x41 * x43,
                                        x45 * x46,
                                        x43 * x47,
                                        -x15 * x39,
                                        x17 * x35,
                                        -x16 * x31,
                                        -x51 * x52,
                                        x51 * x53,
                                        -x55 * x57,
                                        x60 * x62,
                                        x64 * x65,
                                        x62 * x69,
                                        -x61 * x71,
                                        x61 * x73,
                                        -x55 * x74,
                                        -x30 * x76,
                                        x34 * x77,
                                        -x15 * x78,
                                        x47 * x79,
                                        x26 * x44 * x46,
                                        x41 * x79,
                                        -x19 * x78,
                                        x21 * x34 * x76,
                                        -x11 * x26 * x30 * x80,
                                        -x74 * x82,
                                        x73 * x83,
                                        -x71 * x83,
                                        x69 * x84,
                                        x0 * x63 * x65,
                                        x60 * x84,
                                        -x57 * x82,
                                        x53 * x85,
                                        -x52 * x85,
                                        x87 * x89,
                                        x89 * x90,
                                        x91 * x93,
                                        x93 * x94,
                                        -x21 * x98,
                                        x101 * x19,
                                        -x103 * x41,
                                        -x104 * x105,
                                        -x103 * x47,
                                        x101 * x15,
                                        -x17 * x98,
                                        -x107 * x48,
                                        x108 * x91,
                                        -x110 * x60,
                                        -x111 * x112,
                                        -x110 * x69,
                                        x109 * x113,
                                        -x107 * x114,
                                        -x117 * x97,
                                        x118 * x15,
                                        -x119 * x47,
                                        -x105 * x120 * x54,
                                        -x119 * x41,
                                        x118 * x19,
                                        -x122 * x92 * x97,
                                        -x114 * x123,
                                        x113 * x124,
                                        -x125 * x69,
                                        -x112 * x20 * x64,
                                        -x125 * x60,
                                        x108 * x94,
                                        -x123 * x48,
                                        x126 * x128,
                                        x128 * x129,
                                        x129 * x131,
                                        x126 * x131,
                                        -x132 * x134,
                                        x136 * x41,
                                        x104 * x137,
                                        x136 * x47,
                                        -x134 * x138,
                                        -x116 * x140,
                                        x142 * x60,
                                        x143 * x144,
                                        x142 * x69,
                                        -x141 * x145,
                                        -x138 * x147,
                                        x148 * x47,
                                        x137 * x149 * x58,
                                        x148 * x41,
                                        -x132 * x147,
                                        -x145 * x150,
                                        x151 * x69,
                                        x111 * x144 * x21,
                                        x151 * x60,
                                        -x122 * x140,
                                        x132 * x153,
                                        x138 * x153,
                                        x155 * x156,
                                        x156 * x157,
                                        -x158 * x160,
                                        -x161 * x45 * x54,
                                        -x160 * x162,
                                        -x163 * x60,
                                        -x164 * x165,
                                        -x163 * x69,
                                        -x166 * x167,
                                        -x161 * x168 * x7,
                                        -x167 * x169,
                                        -x170 * x69,
                                        -x143 * x165 * x19,
                                        -x170 * x60,
                                        x169 * x171,
                                        x166 * x171,
                                        x166 * x172,
                                        x169 * x172,
                                        0.0072337962962962962963 * x120 * x60,
                                        x173 * x47,
                                        0.0072337962962962962963 * x168 * x66,
                                        x173 * x41,
                                        x44 * x63,
                                    ]
                                )

                        case _:
                            assert (
                                False
                            ), "Order of shape functions not implemented or number of nodes not adequat"
                case 3:
                    match n_nodes:
                        case 8:

                            def shape_functions(xi):
                                x0 = xi[0] - 1.0
                                x1 = xi[1] - 1.0
                                x2 = 0.125 * xi[2] - 0.125
                                x3 = x1 * x2
                                x4 = xi[0] + 1.0
                                x5 = xi[1] + 1.0
                                x6 = x2 * x5
                                x7 = 0.125 * xi[2] + 0.125
                                x8 = x1 * x7
                                x9 = x5 * x7
                                return jnp.stack(
                                    [
                                        -x0 * x3,
                                        x3 * x4,
                                        -x4 * x6,
                                        x0 * x6,
                                        x0 * x8,
                                        -x4 * x8,
                                        x4 * x9,
                                        -x0 * x9,
                                    ]
                                )

                        case 27:

                            def shape_functions(xi):
                                x0 = xi[0] - 1.0
                                x1 = xi[1] - 1.0
                                x2 = xi[0] * xi[1]
                                x3 = x1 * x2
                                x4 = xi[2] * (xi[2] - 1.0)
                                x5 = 0.125 * x4
                                x6 = x3 * x5
                                x7 = xi[0] + 1.0
                                x8 = xi[1] + 1.0
                                x9 = x2 * x8
                                x10 = x5 * x9
                                x11 = xi[2] * (xi[2] + 1.0)
                                x12 = 0.125 * x11
                                x13 = x12 * x3
                                x14 = x12 * x9
                                x15 = xi[0] ** 2 - 1.0
                                x16 = x15 * xi[1]
                                x17 = x1 * x16
                                x18 = 0.25 * x4
                                x19 = xi[1] ** 2 - 1.0
                                x20 = x19 * xi[0]
                                x21 = x18 * x20
                                x22 = x16 * x8
                                x23 = 0.25 * x0
                                x24 = xi[2] ** 2 - 1.0
                                x25 = x24 * x3
                                x26 = 0.25 * x7
                                x27 = x24 * x9
                                x28 = 0.25 * x11
                                x29 = x11 * x20
                                x30 = x15 * x19
                                x31 = 0.5 * x30
                                x32 = 0.5 * x24
                                x33 = x20 * x32
                                return jnp.stack(
                                    [
                                    x0 * x6,
                                    x6 * x7,
                                    x10 * x7,
                                    x0 * x10,
                                    x0 * x13,
                                    x13 * x7,
                                    x14 * x7,
                                    x0 * x14,
                                    -x17 * x18,
                                    -x21 * x7,
                                    -x18 * x22,
                                    -x0 * x21,
                                    -x17 * x28,
                                    -x26 * x29,
                                    -x22 * x28,
                                    -x23 * x29,
                                    -x23 * x25,
                                    -x25 * x26,
                                    -x26 * x27,
                                    -x23 * x27,
                                    x0 * x33,
                                    x33 * x7,
                                    x17 * x32,
                                    x22 * x32,
                                    x31 * x4,
                                    x11 * x31,
                                    -x24 * x30,

                                    ]
                                )

                        case _:
                            assert (
                                False
                            ), "Order of shape functions not implemented or number of nodes not adequat"
                case _:
                    assert False, "Dimensionality not implemented."
        case "line_tri_tet":
            match n_dim:
                case 1:
                    """The following shape functions were generated using the code below:

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
                                x0 = 0.5 * xi
                                return jnp.stack([0.5 - x0, x0 + 0.5])

                        case 3:

                            def shape_functions(xi):
                                x0 = 0.5 * xi
                                return jnp.stack(
                                    [x0 * (xi - 1.0), x0 * (xi + 1.0), 1.0 - xi**2]
                                )

                        case 4:

                            def shape_functions(xi):
                                x0 = 3.0 * xi
                                x1 = x0 + 1.0
                                x2 = xi + 1.0
                                x3 = x2 * (x0 - 1.0)
                                x4 = 0.5625 * xi - 0.5625
                                return jnp.stack(
                                    [
                                        -0.5625 * xi**3 + 0.5625 * xi**2 + 0.0625 * xi - 0.0625,
                                        0.0625 * x1 * x3,
                                        x3 * x4,
                                        -x1 * x2 * x4,
                                    ]
                                )

                        case 5:

                            def shape_functions(xi):
                                x0 = xi - 1.0
                                x1 = 2.0 * xi
                                x2 = x1 - 1.0
                                x3 = x0 * x2 * xi
                                x4 = x1 + 1.0
                                x5 = 0.16666666666666666667 * x4
                                x6 = xi + 1.0
                                x7 = x6 * xi
                                return jnp.stack(
                                    [
                                        x3 * x5,
                                        x2 * x5 * x7,
                                        -1.3333333333333333333 * x3 * x6,
                                        4.0 * xi**4 - 5.0 * xi**2 + 1.0,
                                        -1.3333333333333333333 * x0 * x4 * x7,
                                    ]
                                )

                        case 6:

                            def shape_functions(xi):
                                x0 = 5.0 * xi
                                x1 = x0 + 1.0
                                x2 = xi - 1.0
                                x3 = x0 - 1.0
                                x4 = x0 - 3.0
                                x5 = x1 * x2 * x3 * x4
                                x6 = x0 + 3.0
                                x7 = 0.0013020833333333333333 * x6
                                x8 = xi + 1.0
                                x9 = x3 * x4 * x8
                                x10 = 0.032552083333333333333 * x8
                                x11 = x2 * x6
                                x12 = 0.065104166666666666667 * x11
                                return jnp.stack(
                                    [
                                        -x5 * x7,
                                        x1 * x7 * x9,
                                        x10 * x5,
                                        -x12 * x9,
                                        x1 * x12 * x4 * x8,
                                        -x1 * x10 * x11 * x3,
                                    ]
                                )

                        case 7:

                            def shape_functions(xi):
                                x0 = 3.0 * xi
                                x1 = x0 + 1.0
                                x2 = xi - 1.0
                                x3 = x0 - 1.0
                                x4 = x0 - 2.0
                                x5 = x1 * x2 * x3 * x4 * xi
                                x6 = x0 + 2.0
                                x7 = 0.0125 * x6
                                x8 = xi + 1.0
                                x9 = x3 * x4 * x8 * xi
                                x10 = 0.225 * x8
                                x11 = x2 * x6
                                x12 = 0.5625 * x11
                                x13 = x1 * xi
                                return jnp.stack(
                                    [
                                        x5 * x7,
                                        x1 * x7 * x9,
                                        -x10 * x5,
                                        x12 * x9,
                                        -20.25 * xi**6 + 31.5 * xi**4 - 12.25 * xi**2 + 1.0,
                                        x12 * x13 * x4 * x8,
                                        -x10 * x11 * x13 * x3,
                                    ]
                                )

                        case 8:

                            def shape_functions(xi):
                                x0 = 7.0 * xi
                                x1 = x0 + 1.0
                                x2 = x0 + 3.0
                                x3 = xi - 1.0
                                x4 = x0 - 1.0
                                x5 = x0 - 3.0
                                x6 = x0 - 5.0
                                x7 = x1 * x2 * x3 * x4 * x5 * x6
                                x8 = x0 + 5.0
                                x9 = 0.000010850694444444444444 * x8
                                x10 = xi + 1.0
                                x11 = x1 * x10 * x4 * x5 * x6
                                x12 = 0.00053168402777777777778 * x10
                                x13 = x3 * x8
                                x14 = 0.0015950520833333333333 * x13
                                x15 = x10 * x2 * x4 * x6
                                x16 = x13 * x5
                                x17 = 0.0026584201388888888889 * x16
                                x18 = x1 * x2
                                return jnp.stack(
                                    [
                                        -x7 * x9,
                                        x11 * x2 * x9,
                                        x12 * x7,
                                        -x11 * x14,
                                        x15 * x17,
                                        -x10 * x17 * x18 * x6,
                                        x1 * x14 * x15,
                                        -x12 * x16 * x18 * x4,
                                    ]
                                )

                        case 9:

                            def shape_functions(xi):
                                x0 = 2.0 * xi
                                x1 = x0 + 1.0
                                x2 = 4.0 * xi
                                x3 = x2 + 1.0
                                x4 = xi - 1.0
                                x5 = x0 - 1.0
                                x6 = x2 - 1.0
                                x7 = x2 - 3.0
                                x8 = x1 * x3 * x4 * x5 * x6 * x7 * xi
                                x9 = x2 + 3.0
                                x10 = 0.0015873015873015873016 * x9
                                x11 = xi + 1.0
                                x12 = x11 * x3 * x5 * x6 * x7 * xi
                                x13 = 0.050793650793650793651 * x11
                                x14 = x4 * x9
                                x15 = 0.088888888888888888889 * x14
                                x16 = x1 * x11 * x6 * x7 * xi
                                x17 = x14 * x5
                                x18 = 0.35555555555555555556 * x17
                                x19 = x1 * x3 * xi
                                return jnp.stack(
                                    [
                                        x10 * x8,
                                        x1 * x10 * x12,
                                        -x13 * x8,
                                        x12 * x15,
                                        -x16 * x18,
                                        113.77777777777777778 * xi**8
                                        - 213.33333333333333333 * xi**6
                                        + 121.33333333333333333 * xi**4
                                        - 22.777777777777777778 * xi**2
                                        + 1.0,
                                        -x11 * x18 * x19 * x7,
                                        x15 * x16 * x3,
                                        -x13 * x17 * x19 * x6,
                                    ]
                                )

                        case 10:

                            def shape_functions(xi):
                                x0 = 3.0 * xi
                                x1 = x0 + 1.0
                                x2 = 9.0 * xi
                                x3 = x2 + 1.0
                                x4 = x2 + 5.0
                                x5 = xi - 1.0
                                x6 = x0 - 1.0
                                x7 = x2 - 1.0
                                x8 = x2 - 5.0
                                x9 = x2 - 7.0
                                x10 = x1 * x3 * x4 * x5 * x6 * x7 * x8 * x9
                                x11 = x2 + 7.0
                                x12 = 4.3596540178571428571e-7 * x11
                                x13 = xi + 1.0
                                x14 = x1 * x13 * x3 * x6 * x7 * x8 * x9
                                x15 = 0.000035313197544642857143 * x13
                                x16 = x11 * x5
                                x17 = 0.00014125279017857142857 * x16
                                x18 = x13 * x3 * x4 * x6 * x7 * x9
                                x19 = x16 * x8
                                x20 = 0.00010986328125 * x19
                                x21 = x1 * x13 * x4 * x7 * x9
                                x22 = x19 * x6
                                x23 = 0.000494384765625 * x22
                                x24 = x1 * x3 * x4
                                return jnp.stack(
                                    [
                                        -x10 * x12,
                                        x12 * x14 * x4,
                                        x10 * x15,
                                        -x14 * x17,
                                        x18 * x20,
                                        -x21 * x23,
                                        x13 * x23 * x24 * x9,
                                        -x20 * x21 * x3,
                                        x1 * x17 * x18,
                                        -x15 * x22 * x24 * x7,
                                    ]
                                )

                        case 11:

                            def shape_functions(xi):
                                x0 = 5.0 * xi
                                x1 = x0 + 1.0
                                x2 = x0 + 2.0
                                x3 = x0 + 3.0
                                x4 = xi - 1.0
                                x5 = x0 - 1.0
                                x6 = x0 - 2.0
                                x7 = x0 - 4.0
                                x8 = x0 - 3.0
                                x9 = x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * xi
                                x10 = x0 + 4.0
                                x11 = 6.8893298059964726631e-6 * x10
                                x12 = xi + 1.0
                                x13 = x1 * x12 * x2 * x5 * x6 * x7 * x8 * xi
                                x14 = 0.00034446649029982363316 * x12
                                x15 = x10 * x4
                                x16 = 0.0015500992063492063492 * x15
                                x17 = x1 * x12 * x3 * x5 * x6 * x7 * xi
                                x18 = x15 * x8
                                x19 = 0.0041335978835978835979 * x18
                                x20 = x12 * x2 * x3 * x5 * x7 * xi
                                x21 = x18 * x6
                                x22 = 0.0072337962962962962963 * x21
                                x23 = x1 * x2 * x3 * xi
                                return jnp.stack(
                                    [
                                        x11 * x9,
                                        x11 * x13 * x3,
                                        -x14 * x9,
                                        x13 * x16,
                                        -x17 * x19,
                                        x20 * x22,
                                        -678.16840277777777778 * xi**10
                                        + 1491.9704861111111111 * xi**8
                                        - 1110.0260416666666667 * xi**6
                                        + 331.81423611111111111 * xi**4
                                        - 36.590277777777777778 * xi**2
                                        + 1.0,
                                        x12 * x22 * x23 * x7,
                                        -x1 * x19 * x20,
                                        x16 * x17 * x2,
                                        -x14 * x21 * x23 * x5,
                                    ]
                                )

                        case 12:

                            def shape_functions(xi):
                                x0 = 11.0 * xi
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
                                x11 = x1 * x10 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9
                                x12 = x0 + 9.0
                                x13 = 1.345572227733686067e-10 * x12
                                x14 = xi + 1.0
                                x15 = x1 * x10 * x14 * x2 * x3 * x6 * x7 * x8 * x9
                                x16 = 1.6281423955577601411e-8 * x14
                                x17 = x12 * x5
                                x18 = 8.1407119777888007055e-8 * x17
                                x19 = x1 * x10 * x14 * x2 * x4 * x6 * x7 * x8
                                x20 = x17 * x9
                                x21 = 2.4422135933366402116e-7 * x20
                                x22 = x1 * x10 * x14 * x3 * x4 * x6 * x7
                                x23 = x20 * x8
                                x24 = 4.8844271866732804233e-7 * x23
                                x25 = x10 * x14 * x2 * x3 * x4 * x6
                                x26 = x23 * x7
                                x27 = 6.8381980613425925926e-7 * x26
                                x28 = x1 * x2 * x3 * x4
                                return jnp.stack(
                                    [
                                        -x11 * x13,
                                        x13 * x15 * x4,
                                        x11 * x16,
                                        -x15 * x18,
                                        x19 * x21,
                                        -x22 * x24,
                                        x25 * x27,
                                        -x10 * x14 * x27 * x28,
                                        x1 * x24 * x25,
                                        -x2 * x21 * x22,
                                        x18 * x19 * x3,
                                        -x16 * x26 * x28 * x6,
                                    ]
                                )

                        case 13:

                            def shape_functions(xi):
                                x0 = 2.0 * xi
                                x1 = x0 + 1.0
                                x2 = 3.0 * xi
                                x3 = x2 + 1.0
                                x4 = 6.0 * xi
                                x5 = x4 + 1.0
                                x6 = x2 + 2.0
                                x7 = xi - 1.0
                                x8 = x0 - 1.0
                                x9 = x2 - 1.0
                                x10 = x4 - 1.0
                                x11 = x2 - 2.0
                                x12 = x4 - 5.0
                                x13 = x1 * x10 * x11 * x12 * x3 * x5 * x6 * x7 * x8 * x9 * xi
                                x14 = x4 + 5.0
                                x15 = 0.000010822510822510822511 * x14
                                x16 = xi + 1.0
                                x17 = x1 * x10 * x11 * x12 * x16 * x3 * x5 * x8 * x9 * xi
                                x18 = 0.00077922077922077922078 * x16
                                x19 = x14 * x7
                                x20 = 0.0021428571428571428571 * x19
                                x21 = x10 * x12 * x16 * x3 * x5 * x6 * x8 * x9 * xi
                                x22 = x11 * x19
                                x23 = 0.0047619047619047619048 * x22
                                x24 = x1 * x10 * x12 * x16 * x5 * x6 * x9 * xi
                                x25 = x22 * x8
                                x26 = 0.016071428571428571429 * x25
                                x27 = x1 * x10 * x12 * x16 * x3 * x6 * xi
                                x28 = x25 * x9
                                x29 = 0.051428571428571428571 * x28
                                x30 = x1 * x3 * x5 * x6 * xi
                                return jnp.stack(
                                    [
                                        x13 * x15,
                                        x15 * x17 * x6,
                                        -x13 * x18,
                                        x17 * x20,
                                        -x21 * x23,
                                        x24 * x26,
                                        -x27 * x29,
                                        4199.04 * xi**12
                                        - 10614.24 * xi**10
                                        + 9729.72 * xi**8
                                        - 4002.57 * xi**6
                                        + 740.74 * xi**4
                                        - 53.69 * xi**2
                                        + 1.0,
                                        -x12 * x16 * x29 * x30,
                                        x26 * x27 * x5,
                                        -x23 * x24 * x3,
                                        x1 * x20 * x21,
                                        -x10 * x18 * x28 * x30,
                                    ]
                                )

                        case 14:

                            def shape_functions(xi):
                                x0 = 13.0 * xi
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
                                x13 = (
                                    x1 * x10 * x11 * x12 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9
                                )
                                x14 = x0 + 11.0
                                x15 = 2.5484322494956175512e-13 * x14
                                x16 = xi + 1.0
                                x17 = x1 * x10 * x11 * x12 * x16 * x2 * x3 * x4 * x7 * x8 * x9
                                x18 = 4.3068505016475936615e-11 * x16
                                x19 = x14 * x6
                                x20 = 2.5841103009885561969e-10 * x19
                                x21 = x1 * x10 * x12 * x16 * x2 * x3 * x5 * x7 * x8 * x9
                                x22 = x11 * x19
                                x23 = 9.4750711036247060553e-10 * x22
                                x24 = x1 * x12 * x16 * x2 * x4 * x5 * x7 * x8 * x9
                                x25 = x10 * x22
                                x26 = 2.3687677759061765138e-9 * x25
                                x27 = x1 * x12 * x16 * x3 * x4 * x5 * x7 * x8
                                x28 = x25 * x9
                                x29 = 4.2637819966311177249e-9 * x28
                                x30 = x12 * x16 * x2 * x3 * x4 * x5 * x7
                                x31 = x28 * x8
                                x32 = 5.6850426621748236332e-9 * x31
                                x33 = x1 * x2 * x3 * x4 * x5
                                return jnp.stack(
                                    [
                                        -x13 * x15,
                                        x15 * x17 * x5,
                                        x13 * x18,
                                        -x17 * x20,
                                        x21 * x23,
                                        -x24 * x26,
                                        x27 * x29,
                                        -x30 * x32,
                                        x12 * x16 * x32 * x33,
                                        -x1 * x29 * x30,
                                        x2 * x26 * x27,
                                        -x23 * x24 * x3,
                                        x20 * x21 * x4,
                                        -x18 * x31 * x33 * x7,
                                    ]
                                )

                        case 15:

                            def shape_functions(xi):
                                x0 = 7.0 * xi
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
                                x13 = (
                                    x1
                                    * x10
                                    * x11
                                    * x12
                                    * x2
                                    * x3
                                    * x4
                                    * x5
                                    * x6
                                    * x7
                                    * x8
                                    * x9
                                    * xi
                                )
                                x14 = x0 + 6.0
                                x15 = 5.6206653428875651098e-10 * x14
                                x16 = xi + 1.0
                                x17 = (
                                    x1
                                    * x10
                                    * x11
                                    * x12
                                    * x16
                                    * x2
                                    * x3
                                    * x4
                                    * x7
                                    * x8
                                    * x9
                                    * xi
                                )
                                x18 = 5.5082520360298138076e-8 * x16
                                x19 = x14 * x6
                                x20 = 3.5803638234193789749e-7 * x19
                                x21 = x1 * x10 * x11 * x16 * x2 * x4 * x5 * x7 * x8 * x9 * xi
                                x22 = x12 * x19
                                x23 = 1.43214552936775159e-6 * x22
                                x24 = x1 * x10 * x11 * x16 * x2 * x3 * x5 * x7 * x8 * xi
                                x25 = x22 * x9
                                x26 = 3.9384002057613168724e-6 * x25
                                x27 = x1 * x11 * x16 * x3 * x4 * x5 * x7 * x8 * xi
                                x28 = x10 * x25
                                x29 = 7.8768004115226337449e-6 * x28
                                x30 = x11 * x16 * x2 * x3 * x4 * x5 * x7 * xi
                                x31 = x28 * x8
                                x32 = 0.000011815200617283950617 * x31
                                x33 = x1 * x2 * x3 * x4 * x5 * xi
                                return jnp.stack(
                                    [
                                        x13 * x15,
                                        x15 * x17 * x5,
                                        -x13 * x18,
                                        x17 * x20,
                                        -x21 * x23,
                                        x24 * x26,
                                        -x27 * x29,
                                        x30 * x32,
                                        -26700.013890817901235 * xi**14
                                        + 76285.753973765432099 * xi**12
                                        - 82980.21809799382716 * xi**10
                                        + 43487.464081790123457 * xi**8
                                        - 11465.29836612654321 * xi**6
                                        + 1445.3903549382716049 * xi**4
                                        - 74.078055555555555556 * xi**2
                                        + 1.0,
                                        x11 * x16 * x32 * x33,
                                        -x1 * x29 * x30,
                                        x2 * x26 * x27,
                                        -x23 * x24 * x4,
                                        x20 * x21 * x3,
                                        -x18 * x31 * x33 * x7,
                                    ]
                                )

                        case 16:

                            def shape_functions(xi):
                                x0 = 3.0 * xi
                                x1 = x0 + 1.0
                                x2 = 5.0 * xi
                                x3 = x2 + 1.0
                                x4 = 15.0 * xi
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
                                x17 = (
                                    x1
                                    * x10
                                    * x11
                                    * x12
                                    * x13
                                    * x14
                                    * x15
                                    * x16
                                    * x3
                                    * x5
                                    * x6
                                    * x7
                                    * x8
                                    * x9
                                )
                                x18 = x4 + 13.0
                                x19 = 7.0887023423470131059e-13 * x18
                                x20 = xi + 1.0
                                x21 = (
                                    x1
                                    * x10
                                    * x11
                                    * x12
                                    * x13
                                    * x14
                                    * x15
                                    * x16
                                    * x20
                                    * x3
                                    * x5
                                    * x6
                                    * x7
                                )
                                x22 = 1.5949580270280779488e-10 * x20
                                x23 = x18 * x9
                                x24 = 1.1164706189196545642e-9 * x23
                                x25 = (
                                    x1
                                    * x10
                                    * x11
                                    * x12
                                    * x13
                                    * x14
                                    * x16
                                    * x20
                                    * x3
                                    * x5
                                    * x7
                                    * x8
                                )
                                x26 = x15 * x23
                                x27 = 1.6126797828839454816e-9 * x26
                                x28 = x1 * x10 * x11 * x12 * x14 * x16 * x20 * x3 * x5 * x6 * x8
                                x29 = x13 * x26
                                x30 = 1.4514118045955509334e-8 * x29
                                x31 = x10 * x11 * x12 * x16 * x20 * x3 * x5 * x6 * x7 * x8
                                x32 = x14 * x29
                                x33 = 6.3862119402204241071e-9 * x32
                                x34 = x1 * x11 * x12 * x16 * x20 * x5 * x6 * x7 * x8
                                x35 = x10 * x32
                                x36 = 1.7739477611723400298e-8 * x35
                                x37 = x1 * x12 * x16 * x20 * x3 * x6 * x7 * x8
                                x38 = x11 * x35
                                x39 = 6.8423699359504544005e-8 * x38
                                x40 = x1 * x3 * x5 * x6 * x7 * x8
                                return jnp.stack(
                                    [
                                        -x17 * x19,
                                        x19 * x21 * x8,
                                        x17 * x22,
                                        -x21 * x24,
                                        x25 * x27,
                                        -x28 * x30,
                                        x31 * x33,
                                        -x34 * x36,
                                        x37 * x39,
                                        -x16 * x20 * x39 * x40,
                                        x36 * x37 * x5,
                                        -x3 * x33 * x34,
                                        x1 * x30 * x31,
                                        -x27 * x28 * x7,
                                        x24 * x25 * x6,
                                        -x12 * x22 * x38 * x40,
                                    ]
                                )

                        case 17:

                            def shape_functions(xi):
                                x0 = 2.0 * xi
                                x1 = x0 + 1.0
                                x2 = 4.0 * xi
                                x3 = x2 + 1.0
                                x4 = 8.0 * xi
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
                                x17 = (
                                    x1
                                    * x10
                                    * x11
                                    * x12
                                    * x13
                                    * x14
                                    * x15
                                    * x16
                                    * x3
                                    * x5
                                    * x6
                                    * x7
                                    * x8
                                    * x9
                                    * xi
                                )
                                x18 = x4 + 7.0
                                x19 = 7.8306956613834920713e-10 * x18
                                x20 = xi + 1.0
                                x21 = (
                                    x1
                                    * x10
                                    * x11
                                    * x12
                                    * x13
                                    * x14
                                    * x15
                                    * x16
                                    * x20
                                    * x3
                                    * x5
                                    * x7
                                    * x8
                                    * xi
                                )
                                x22 = 1.0023290446570869851e-7 * x20
                                x23 = x18 * x9
                                x24 = 3.7587339174640761942e-7 * x23
                                x25 = (
                                    x1
                                    * x10
                                    * x11
                                    * x12
                                    * x14
                                    * x15
                                    * x16
                                    * x20
                                    * x3
                                    * x5
                                    * x6
                                    * x7
                                    * xi
                                )
                                x26 = x13 * x23
                                x27 = 3.508151656299804448e-6 * x26
                                x28 = (
                                    x10
                                    * x11
                                    * x12
                                    * x14
                                    * x16
                                    * x20
                                    * x3
                                    * x5
                                    * x6
                                    * x7
                                    * x8
                                    * xi
                                )
                                x29 = x15 * x26
                                x30 = 2.850373220743591114e-6 * x29
                                x31 = x1 * x11 * x12 * x14 * x16 * x20 * x3 * x5 * x6 * x8 * xi
                                x32 = x10 * x29
                                x33 = 0.000027363582919138474694 * x32
                                x34 = x1 * x11 * x12 * x16 * x20 * x5 * x6 * x7 * x8 * xi
                                x35 = x14 * x32
                                x36 = 0.000025083284342543601803 * x35
                                x37 = x1 * x12 * x16 * x20 * x3 * x6 * x7 * x8 * xi
                                x38 = x11 * x35
                                x39 = 0.000071666526692981719437 * x38
                                x40 = x1 * x3 * x5 * x6 * x7 * x8 * xi
                                return jnp.stack(
                                    [
                                        x17 * x19,
                                        x19 * x21 * x6,
                                        -x17 * x22,
                                        x21 * x24,
                                        -x25 * x27,
                                        x28 * x30,
                                        -x31 * x33,
                                        x34 * x36,
                                        -x37 * x39,
                                        173140.53095490047871 * xi**16
                                        - 551885.44241874527589 * xi**14
                                        + 694168.40804232804233 * xi**12
                                        - 441984.42698916603679 * xi**10
                                        + 152107.76187452758881 * xi**8
                                        - 28012.603597883597884 * xi**6
                                        + 2562.5271453766691862 * xi**4
                                        - 97.755011337868480726 * xi**2
                                        + 1.0,
                                        -x16 * x20 * x39 * x40,
                                        x36 * x37 * x5,
                                        -x3 * x33 * x34,
                                        x30 * x31 * x7,
                                        -x1 * x27 * x28,
                                        x24 * x25 * x8,
                                        -x12 * x22 * x38 * x40,
                                    ]
                                )

                        case 18:

                            def shape_functions(xi):
                                x0 = 17.0 * xi
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
                                x17 = (
                                    x1
                                    * x10
                                    * x11
                                    * x12
                                    * x13
                                    * x14
                                    * x15
                                    * x16
                                    * x2
                                    * x3
                                    * x4
                                    * x5
                                    * x6
                                    * x7
                                    * x8
                                    * x9
                                )
                                x18 = x0 + 15.0
                                x19 = 3.6464518221949655895e-19 * x18
                                x20 = xi + 1.0
                                x21 = (
                                    x1
                                    * x10
                                    * x11
                                    * x12
                                    * x13
                                    * x14
                                    * x15
                                    * x16
                                    * x2
                                    * x20
                                    * x3
                                    * x4
                                    * x5
                                    * x6
                                    * x9
                                )
                                x22 = 1.0538245766143450554e-16 * x20
                                x23 = x18 * x8
                                x24 = 8.4305966129147604429e-16 * x23
                                x25 = (
                                    x1
                                    * x10
                                    * x11
                                    * x12
                                    * x13
                                    * x14
                                    * x16
                                    * x2
                                    * x20
                                    * x3
                                    * x4
                                    * x5
                                    * x7
                                    * x9
                                )
                                x26 = x15 * x23
                                x27 = 4.2152983064573802214e-15 * x26
                                x28 = (
                                    x1
                                    * x10
                                    * x11
                                    * x12
                                    * x13
                                    * x16
                                    * x2
                                    * x20
                                    * x3
                                    * x4
                                    * x6
                                    * x7
                                    * x9
                                )
                                x29 = x14 * x26
                                x30 = 1.4753544072600830775e-14 * x29
                                x31 = (
                                    x1
                                    * x10
                                    * x11
                                    * x12
                                    * x16
                                    * x2
                                    * x20
                                    * x3
                                    * x5
                                    * x6
                                    * x7
                                    * x9
                                )
                                x32 = x13 * x29
                                x33 = 3.8359214588762160015e-14 * x32
                                x34 = x1 * x10 * x11 * x16 * x2 * x20 * x4 * x5 * x6 * x7 * x9
                                x35 = x12 * x32
                                x36 = 7.671842917752432003e-14 * x35
                                x37 = x1 * x10 * x16 * x20 * x3 * x4 * x5 * x6 * x7 * x9
                                x38 = x11 * x35
                                x39 = 1.2055753156468107433e-13 * x38
                                x40 = x16 * x2 * x20 * x3 * x4 * x5 * x6 * x7 * x9
                                x41 = x10 * x38
                                x42 = 1.5069691445585134292e-13 * x41
                                x43 = x1 * x2 * x3 * x4 * x5 * x6 * x7
                                return jnp.stack(
                                    [
                                        -x17 * x19,
                                        x19 * x21 * x7,
                                        x17 * x22,
                                        -x21 * x24,
                                        x25 * x27,
                                        -x28 * x30,
                                        x31 * x33,
                                        -x34 * x36,
                                        x37 * x39,
                                        -x40 * x42,
                                        x16 * x20 * x42 * x43,
                                        -x1 * x39 * x40,
                                        x2 * x36 * x37,
                                        -x3 * x33 * x34,
                                        x30 * x31 * x4,
                                        -x27 * x28 * x5,
                                        x24 * x25 * x6,
                                        -x22 * x41 * x43 * x9,
                                    ]
                                )

                        case 19:

                            def shape_functions(xi):
                                x0 = 3.0 * xi
                                x1 = x0 + 1.0
                                x2 = 9.0 * xi
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
                                x18 = (
                                    x1
                                    * x10
                                    * x11
                                    * x12
                                    * x13
                                    * x14
                                    * x15
                                    * x16
                                    * x17
                                    * x3
                                    * x4
                                    * x5
                                    * x6
                                    * x7
                                    * x8
                                    * x9
                                    * xi
                                )
                                x19 = x2 + 8.0
                                x20 = 1.0247761692089423182e-12 * x19
                                x21 = xi + 1.0
                                x22 = (
                                    x1
                                    * x10
                                    * x11
                                    * x12
                                    * x13
                                    * x14
                                    * x15
                                    * x16
                                    * x17
                                    * x21
                                    * x3
                                    * x4
                                    * x5
                                    * x6
                                    * x7
                                    * xi
                                )
                                x23 = 1.6601373941184865555e-10 * x21
                                x24 = x19 * x9
                                x25 = 1.4111167850007135721e-9 * x24
                                x26 = (
                                    x1
                                    * x10
                                    * x11
                                    * x12
                                    * x13
                                    * x14
                                    * x15
                                    * x16
                                    * x21
                                    * x3
                                    * x5
                                    * x6
                                    * x7
                                    * x8
                                    * xi
                                )
                                x27 = x17 * x24
                                x28 = 2.5086520622234907949e-9 * x27
                                x29 = (
                                    x1
                                    * x10
                                    * x11
                                    * x13
                                    * x14
                                    * x15
                                    * x16
                                    * x21
                                    * x3
                                    * x4
                                    * x5
                                    * x6
                                    * x8
                                    * xi
                                )
                                x30 = x12 * x27
                                x31 = 2.8222335700014271443e-8 * x30
                                x32 = (
                                    x1
                                    * x10
                                    * x11
                                    * x13
                                    * x14
                                    * x15
                                    * x21
                                    * x3
                                    * x4
                                    * x5
                                    * x7
                                    * x8
                                    * xi
                                )
                                x33 = x16 * x30
                                x34 = 7.902253996003996004e-8 * x33
                                x35 = (
                                    x10
                                    * x11
                                    * x13
                                    * x15
                                    * x21
                                    * x3
                                    * x4
                                    * x5
                                    * x6
                                    * x7
                                    * x8
                                    * xi
                                )
                                x36 = x14 * x33
                                x37 = 5.7071834415584415584e-8 * x36
                                x38 = x1 * x11 * x13 * x15 * x21 * x3 * x4 * x6 * x7 * x8 * xi
                                x39 = x10 * x36
                                x40 = 2.9351229128014842301e-7 * x39
                                x41 = x1 * x11 * x15 * x21 * x4 * x5 * x6 * x7 * x8 * xi
                                x42 = x13 * x39
                                x43 = 4.0357940051020408163e-7 * x42
                                x44 = x1 * x3 * x4 * x5 * x6 * x7 * x8 * xi
                                return jnp.stack(
                                    [
                                        x18 * x20,
                                        x20 * x22 * x8,
                                        -x18 * x23,
                                        x22 * x25,
                                        -x26 * x28,
                                        x29 * x31,
                                        -x32 * x34,
                                        x35 * x37,
                                        -x38 * x40,
                                        x41 * x43,
                                        -1139827.4301937679369 * xi**18
                                        + 4010503.9210521464445 * xi**16
                                        - 5723632.7564645448023 * xi**14
                                        + 4288221.5882976921237 * xi**12
                                        - 1825541.0625608358578 * xi**10
                                        + 447065.31380067163584 * xi**8
                                        - 60894.246929607780612 * xi**6
                                        + 4228.3941844706632653 * xi**4
                                        - 124.72118622448979592 * xi**2
                                        + 1.0,
                                        x15 * x21 * x43 * x44,
                                        -x3 * x40 * x41,
                                        x37 * x38 * x5,
                                        -x1 * x34 * x35,
                                        x31 * x32 * x6,
                                        -x28 * x29 * x7,
                                        x25 * x26 * x4,
                                        -x11 * x23 * x42 * x44,
                                    ]
                                )

                        case 20:

                            def shape_functions(xi):
                                x0 = 19.0 * xi
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
                                x19 = (
                                    x1
                                    * x10
                                    * x11
                                    * x12
                                    * x13
                                    * x14
                                    * x15
                                    * x16
                                    * x17
                                    * x18
                                    * x2
                                    * x3
                                    * x4
                                    * x5
                                    * x6
                                    * x7
                                    * x8
                                    * x9
                                )
                                x20 = x0 + 17.0
                                x21 = 2.9791273057148411679e-22 * x20
                                x22 = xi + 1.0
                                x23 = (
                                    x1
                                    * x10
                                    * x11
                                    * x12
                                    * x13
                                    * x14
                                    * x15
                                    * x16
                                    * x17
                                    * x18
                                    * x2
                                    * x22
                                    * x3
                                    * x4
                                    * x5
                                    * x6
                                    * x7
                                )
                                x24 = 1.0754649573630576616e-19 * x22
                                x25 = x20 * x9
                                x26 = 9.6791846162675189544e-19 * x25
                                x27 = (
                                    x1
                                    * x10
                                    * x11
                                    * x12
                                    * x13
                                    * x14
                                    * x15
                                    * x16
                                    * x18
                                    * x2
                                    * x22
                                    * x3
                                    * x4
                                    * x5
                                    * x6
                                    * x8
                                )
                                x28 = x17 * x25
                                x29 = 5.4848712825515940742e-18 * x28
                                x30 = (
                                    x1
                                    * x10
                                    * x11
                                    * x12
                                    * x13
                                    * x14
                                    * x15
                                    * x18
                                    * x2
                                    * x22
                                    * x3
                                    * x4
                                    * x5
                                    * x7
                                    * x8
                                )
                                x31 = x16 * x28
                                x32 = 2.1939485130206376297e-17 * x31
                                x33 = (
                                    x1
                                    * x10
                                    * x11
                                    * x12
                                    * x13
                                    * x14
                                    * x18
                                    * x2
                                    * x22
                                    * x3
                                    * x4
                                    * x6
                                    * x7
                                    * x8
                                )
                                x34 = x15 * x31
                                x35 = 6.581845539061912889e-17 * x34
                                x36 = (
                                    x1
                                    * x10
                                    * x11
                                    * x12
                                    * x13
                                    * x18
                                    * x2
                                    * x22
                                    * x3
                                    * x5
                                    * x6
                                    * x7
                                    * x8
                                )
                                x37 = x14 * x34
                                x38 = 1.5357639591144463408e-16 * x37
                                x39 = (
                                    x1
                                    * x10
                                    * x11
                                    * x12
                                    * x18
                                    * x2
                                    * x22
                                    * x4
                                    * x5
                                    * x6
                                    * x7
                                    * x8
                                )
                                x40 = x13 * x37
                                x41 = 2.8521330669268289186e-16 * x40
                                x42 = x1 * x10 * x11 * x18 * x22 * x3 * x4 * x5 * x6 * x7 * x8
                                x43 = x12 * x40
                                x44 = 4.2781996003902433779e-16 * x43
                                x45 = x10 * x18 * x2 * x22 * x3 * x4 * x5 * x6 * x7 * x8
                                x46 = x11 * x43
                                x47 = 5.2289106226991863507e-16 * x46
                                x48 = x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8
                                return jnp.stack(
                                    [
                                        -x19 * x21,
                                        x21 * x23 * x8,
                                        x19 * x24,
                                        -x23 * x26,
                                        x27 * x29,
                                        -x30 * x32,
                                        x33 * x35,
                                        -x36 * x38,
                                        x39 * x41,
                                        -x42 * x44,
                                        x45 * x47,
                                        -x18 * x22 * x47 * x48,
                                        x1 * x44 * x45,
                                        -x2 * x41 * x42,
                                        x3 * x38 * x39,
                                        -x35 * x36 * x4,
                                        x32 * x33 * x5,
                                        -x29 * x30 * x6,
                                        x26 * x27 * x7,
                                        -x10 * x24 * x46 * x48,
                                    ]
                                )

                        case 21:

                            def shape_functions(xi):
                                x0 = 2.0 * xi
                                x1 = x0 + 1.0
                                x2 = 5.0 * xi
                                x3 = x2 + 1.0
                                x4 = 10.0 * xi
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
                                x21 = (
                                    x1
                                    * x10
                                    * x11
                                    * x12
                                    * x13
                                    * x14
                                    * x15
                                    * x16
                                    * x17
                                    * x18
                                    * x19
                                    * x20
                                    * x3
                                    * x5
                                    * x6
                                    * x7
                                    * x8
                                    * x9
                                    * xi
                                )
                                x22 = x4 + 9.0
                                x23 = 2.6306032789197855094e-13 * x22
                                x24 = xi + 1.0
                                x25 = (
                                    x1
                                    * x10
                                    * x12
                                    * x13
                                    * x14
                                    * x15
                                    * x16
                                    * x17
                                    * x18
                                    * x19
                                    * x20
                                    * x24
                                    * x3
                                    * x5
                                    * x6
                                    * x8
                                    * x9
                                    * xi
                                )
                                x26 = 5.2612065578395710189e-11 * x24
                                x27 = x11 * x22
                                x28 = 2.499073114973796234e-10 * x27
                                x29 = (
                                    x1
                                    * x12
                                    * x13
                                    * x14
                                    * x15
                                    * x17
                                    * x18
                                    * x19
                                    * x20
                                    * x24
                                    * x3
                                    * x5
                                    * x6
                                    * x7
                                    * x8
                                    * x9
                                    * xi
                                )
                                x30 = x16 * x27
                                x31 = 2.9988877379685554807e-9 * x30
                                x32 = (
                                    x1
                                    * x10
                                    * x12
                                    * x13
                                    * x14
                                    * x15
                                    * x17
                                    * x18
                                    * x20
                                    * x24
                                    * x3
                                    * x5
                                    * x6
                                    * x7
                                    * x9
                                    * xi
                                )
                                x33 = x19 * x30
                                x34 = 6.3726364431831803966e-9 * x33
                                x35 = (
                                    x10
                                    * x12
                                    * x13
                                    * x14
                                    * x15
                                    * x18
                                    * x20
                                    * x24
                                    * x3
                                    * x5
                                    * x6
                                    * x7
                                    * x8
                                    * x9
                                    * xi
                                )
                                x36 = x17 * x33
                                x37 = 8.1569746472744709076e-9 * x36
                                x38 = (
                                    x1
                                    * x10
                                    * x13
                                    * x14
                                    * x15
                                    * x18
                                    * x20
                                    * x24
                                    * x3
                                    * x5
                                    * x7
                                    * x8
                                    * x9
                                    * xi
                                )
                                x39 = x12 * x36
                                x40 = 5.0981091545465443173e-8 * x39
                                x41 = (
                                    x1
                                    * x10
                                    * x13
                                    * x14
                                    * x18
                                    * x20
                                    * x24
                                    * x3
                                    * x5
                                    * x6
                                    * x7
                                    * x8
                                    * xi
                                )
                                x42 = x15 * x39
                                x43 = 2.0392436618186177269e-7 * x42
                                x44 = (
                                    x1
                                    * x10
                                    * x13
                                    * x14
                                    * x20
                                    * x24
                                    * x5
                                    * x6
                                    * x7
                                    * x8
                                    * x9
                                    * xi
                                )
                                x45 = x18 * x42
                                x46 = 1.6568854752276269031e-7 * x45
                                x47 = x1 * x10 * x14 * x20 * x24 * x3 * x6 * x7 * x8 * x9 * xi
                                x48 = x13 * x45
                                x49 = 4.4183612672736717416e-7 * x48
                                x50 = x1 * x10 * x3 * x5 * x6 * x7 * x8 * x9 * xi
                                return jnp.stack(
                                    [
                                        x21 * x23,
                                        x23 * x25 * x7,
                                        -x21 * x26,
                                        x25 * x28,
                                        -x29 * x31,
                                        x32 * x34,
                                        -x35 * x37,
                                        x38 * x40,
                                        -x41 * x43,
                                        x44 * x46,
                                        -x47 * x49,
                                        7594058.4281266233059 * xi**20
                                        - 29237124.948287499728 * xi**18
                                        + 46662451.417466849566 * xi**16
                                        - 40202717.496749499983 * xi**14
                                        + 20418933.234909475907 * xi**12
                                        - 6274158.9818744284408 * xi**10
                                        + 1153141.6151619398331 * xi**8
                                        - 121028.00916570045942 * xi**6
                                        + 6598.7171853566529492 * xi**4
                                        - 154.97677311665406904 * xi**2
                                        + 1.0,
                                        -x20 * x24 * x49 * x50,
                                        x46 * x47 * x5,
                                        -x3 * x43 * x44,
                                        x40 * x41 * x9,
                                        -x37 * x38 * x6,
                                        x1 * x34 * x35,
                                        -x31 * x32 * x8,
                                        x10 * x28 * x29,
                                        -x14 * x26 * x48 * x50,
                                    ]
                                )

                        case _:
                            assert (
                                False
                            ), "Order of shape functions not implemented or number of nodes not adequat"
                case 2:
                    match n_nodes:
                        case 3:
                            def shape_functions(xi):
                                return jnp.stack([-xi[0] - xi[1] + 1, xi[0], xi[1]])

                        case 6:
                            def shape_functions(xi):
                                x0 = 4*xi[0]
                                x1 = x0*xi[1]
                                x2 = -xi[0] - xi[1] + 1
                                return jnp.stack([x1 + 2*xi[0]**2 - 3*xi[0] + 2*xi[1]**2 - 3*xi[1] + 1, xi[0]*(2*xi[0] - 1), xi[1]*(2*xi[1] - 1), x0*x2, x1, 4*x2*xi[1]])

                        case 10:
                            def shape_functions(xi):
                                x0 = xi[0]**2
                                x1 = xi[1]**2
                                x2 = xi[0]*xi[1]
                                x3 = 3*x0
                                x4 = 3*x1
                                x5 = 6*x2 + x3 + x4 - 5*xi[0] - 5*xi[1] + 2
                                x6 = 9*xi[0]/2
                                x7 = 3*xi[0]
                                x8 = x7*xi[1] + 1
                                x9 = 9*x2/2
                                x10 = 9*xi[1]/2
                                return jnp.stack([-27*x0*xi[1]/2 + 9*x0 - 27*x1*xi[0]/2 + 9*x1 - 9*xi[0]**3/2 + 18*xi[0]*xi[1] - 11*xi[0]/2 - 9*xi[1]**3/2 - 11*xi[1]/2 + 1, xi[0]*(9*x0 - 9*xi[0] + 2)/2, xi[1]*(9*x1 - 9*xi[1] + 2)/2, x5*x6, x6*(-x3 - x8 + 4*xi[0] + xi[1]), x9*(x7 - 1), x9*(3*xi[1] - 1), x10*(-x4 - x8 + xi[0] + 4*xi[1]), x10*x5, 27*x2*(-xi[0] - xi[1] + 1)])

                        case 15:
                            def shape_functions(xi):
                                x0 = xi[0]*xi[1]
                                x1 = xi[0]**2
                                x2 = xi[0]**3
                                x3 = xi[1]**2
                                x4 = xi[1]**3
                                x5 = x3*xi[0]
                                x6 = x1*xi[1]
                                x7 = 8*x2
                                x8 = 8*x4
                                x9 = -36*x0 - 3
                                x10 = 18*x1 + 18*x3 - 24*x5 - 24*x6 - x7 - x8 - x9 - 13*xi[0] - 13*xi[1]
                                x11 = 16*xi[0]/3
                                x12 = 7*xi[1]
                                x13 = 32*x1
                                x14 = 4*x3
                                x15 = 4*xi[0]
                                x16 = 7*xi[0]
                                x17 = 8*x1
                                x18 = -6*xi[0]*xi[1] - 1
                                x19 = 16*x0/3
                                x20 = 4*xi[1]
                                x21 = x15*xi[1]
                                x22 = 8*x3
                                x23 = 16*xi[1]/3
                                x24 = 32*x3
                                x25 = 4*x1
                                x26 = 32*x0
                                x27 = x21 + 1
                                return jnp.stack([140*x0/3 + 64*x1*x3 + 70*x1/3 + 128*x2*xi[1]/3 - 80*x2/3 + 70*x3/3 + 128*x4*xi[0]/3 - 80*x4/3 - 80*x5 - 80*x6 + 32*xi[0]**4/3 - 25*xi[0]/3 + 32*xi[1]**4/3 - 25*xi[1]/3 + 1, xi[0]*(-48*x1 + 32*x2 + 22*xi[0] - 3)/3, xi[1]*(-48*x3 + 32*x4 + 22*xi[1] - 3)/3, x10*x11, x15*(x12 + x13*xi[1] - x13 - x14 + 16*x2 + 16*x5 + x9 + 19*xi[0]), x11*(14*x1 - x16 - x17*xi[1] - x18 - x7 - xi[1]), x19*(x17 - 6*xi[0] + 1), x21*(16*x0 - x15 - x20 + 1), x19*(x22 - 6*xi[1] + 1), x23*(-x12 - x18 - x22*xi[0] + 14*x3 - x8 - xi[0]), x20*(x16 + x24*xi[0] - x24 - x25 + 16*x4 + 16*x6 + x9 + 19*xi[1]), x10*x23, x26*(8*x0 - x12 + x14 - x16 + x25 + 3), x26*(-x25 - x27 + 5*xi[0] + xi[1]), x26*(-x14 - x27 + xi[0] + 5*xi[1])])

                        case 21:
                            def shape_functions(xi):
                                x0 = xi[0]**2
                                x1 = xi[0]**3
                                x2 = xi[0]**4
                                x3 = xi[1]**2
                                x4 = xi[1]**3
                                x5 = xi[1]**4
                                x6 = x3*xi[0]
                                x7 = x0*xi[1]
                                x8 = 250*xi[0]
                                x9 = 250*xi[1]
                                x10 = xi[0]*xi[1]
                                x11 = 125*x2
                                x12 = 125*x5
                                x13 = x4*xi[0]
                                x14 = x1*xi[1]
                                x15 = x0*x3
                                x16 = 355*x0 - 350*x1 + 710*x10 + x11 + x12 + 500*x13 + 500*x14 + 750*x15 + 355*x3 - 350*x4 - 1050*x6 - 1050*x7 - 154*xi[0] - 154*xi[1] + 24
                                x17 = 25*xi[0]/24
                                x18 = 125*x4
                                x19 = x18*xi[0]
                                x20 = -375*x6
                                x21 = 47*xi[1]
                                x22 = 25*x4
                                x23 = 60*x3
                                x24 = 355*x10 + 375*x15 + 12
                                x25 = 25*xi[0]/12
                                x26 = -375*x7
                                x27 = 75*x6
                                x28 = 155*x10 + 125*x15 + 8
                                x29 = 6*xi[1]
                                x30 = 125*x1
                                x31 = x30*xi[1]
                                x32 = 55*xi[0]
                                x33 = x32*xi[1]
                                x34 = x33 + 6
                                x35 = 25*x10/24
                                x36 = 25*x0
                                x37 = -75*x10 - 2
                                x38 = 25*x10/12
                                x39 = 25*x3
                                x40 = 6*xi[0]
                                x41 = 25*xi[1]/24
                                x42 = 75*x7
                                x43 = 25*xi[1]/12
                                x44 = 47*xi[0]
                                x45 = 25*x1
                                x46 = 60*x0
                                x47 = 125*x10/6
                                x48 = x36*xi[1]
                                x49 = -15*xi[0]*xi[1] - 2
                                x50 = x39*xi[0]
                                x51 = 50*x0
                                x52 = -5*x3 + x50
                                x53 = -x33 - 4
                                x54 = 125*x10/4
                                x55 = -5*x0 + x48
                                x56 = 50*x3
                                return jnp.stack([1875*x0*x3/4 - 3125*x0*x4/12 + 375*x0/8 - 3125*x1*x3/12 + 625*x1*xi[1]/2 - 2125*x1/24 - 3125*x2*xi[1]/24 + 625*x2/8 + 375*x3/8 + 625*x4*xi[0]/2 - 2125*x4/24 - 3125*x5*xi[0]/24 + 625*x5/8 - 2125*x6/8 - 2125*x7/8 - 625*xi[0]**5/24 + 375*xi[0]*xi[1]/4 - 137*xi[0]/12 - 625*xi[1]**5/24 - 137*xi[1]/12 + 1, xi[0]*(875*x0 - 1250*x1 + 625*x2 - x8 + 24)/24, xi[1]*(875*x3 - 1250*x4 + 625*x5 - x9 + 24)/24, x16*x17, x25*(675*x0*xi[1] - 295*x0 + 325*x1 - x11 - 375*x14 - x19 - x20 + x21 + x22 - x23 - x24 + 107*xi[0]), x25*(245*x0 + x1*x9 - 300*x1 + x11 + x26 - x27 + x28 + 10*x3 - 78*xi[0] - 18*xi[1]), x17*(150*x0*xi[1] - 205*x0 + 275*x1 - x11 + x29 - x31 - x34 + 61*xi[0]), x35*(-150*x0 + x30 + x32 - 6), x38*(-x36 + x37 + 125*x7 + 15*xi[0] + 10*xi[1]), x38*(x37 - x39 + 125*x6 + 10*xi[0] + 15*xi[1]), x35*(x18 - 150*x3 + 55*xi[1] - 6), x41*(-x12 - x19 + 150*x3*xi[0] - 205*x3 - x34 + 275*x4 + x40 + 61*xi[1]), x43*(10*x0 + x12 + x20 + x28 + 245*x3 + x4*x8 - 300*x4 - x42 - 18*xi[0] - 78*xi[1]), x43*(-x12 - 375*x13 - x24 - x26 + 675*x3*xi[0] - 295*x3 - x31 + 325*x4 + x44 + x45 - x46 + 107*xi[1]), x16*x41, x47*(-x21 - x22 + x23 - x27 - x42 - x44 - x45 + x46 + 120*xi[0]*xi[1] + 12), x47*(40*x0 - x45 - x48 - x49 - 17*xi[0] - 2*xi[1]), x47*(-x22 + 40*x3 - x49 - x50 - 2*xi[0] - 17*xi[1]), x54*(x45 + x51*xi[1] - x51 + x52 + x53 + 29*xi[0] + 9*xi[1]), x54*(-x29 - x40 - x52 - x55 + 35*xi[0]*xi[1] + 1), x54*(x22 + x53 + x55 + x56*xi[0] - x56 + 9*xi[0] + 29*xi[1])])

                        case 28:
                            def shape_functions(xi):
                                x0 = xi[0]*xi[1]
                                x1 = xi[0]**2
                                x2 = xi[0]**3
                                x3 = xi[0]**4
                                x4 = xi[0]**5
                                x5 = xi[1]**2
                                x6 = xi[1]**3
                                x7 = xi[1]**4
                                x8 = xi[1]**5
                                x9 = x5*xi[0]
                                x10 = x6*xi[0]
                                x11 = x7*xi[0]
                                x12 = x1*xi[1]
                                x13 = x2*xi[1]
                                x14 = x3*xi[1]
                                x15 = x1*x5
                                x16 = x1*x6
                                x17 = 972*x7
                                x18 = x2*x5
                                x19 = 972*x3
                                x20 = 108*x4
                                x21 = 108*x8
                                x22 = -580*x0 - 2160*x15 - 10
                                x23 = 290*x1 - 540*x11 - 1395*x12 - 540*x14 - 1080*x16 - 1080*x18 + 1440*x2*xi[1] - 465*x2 - x20 - x21 - x22 + 360*x3 + 290*x5 + 1440*x6*xi[0] - 465*x6 + 360*x7 - 1395*x9 - 87*xi[0] - 87*xi[1]
                                x24 = 18*xi[0]/5
                                x25 = 216*x4
                                x26 = 119*x5
                                x27 = 36*x7
                                x28 = 57*xi[1]
                                x29 = 108*x6
                                x30 = 9*xi[0]/2
                                x31 = 45*x5
                                x32 = 18*x6
                                x33 = 37*xi[1]
                                x34 = -1296*x1*x5 - 423*xi[0]*xi[1] - 10
                                x35 = 6*x5
                                x36 = 11*xi[1]
                                x37 = 66*xi[0]
                                x38 = 216*x15
                                x39 = -133*x0 - x38 - 5
                                x40 = 27*xi[0]
                                x41 = 105*x1
                                x42 = 108*x3
                                x43 = -25*xi[0]*xi[1] - 2
                                x44 = 18*x0/5
                                x45 = 11*xi[0]
                                x46 = -x45
                                x47 = x37*xi[1] + 1
                                x48 = 36*x2
                                x49 = 216*x13
                                x50 = -x48 + x49
                                x51 = 9*x0/2
                                x52 = 324*x15
                                x53 = -x36
                                x54 = 36*x6
                                x55 = 216*x10
                                x56 = -x54 + x55
                                x57 = 105*x5
                                x58 = 108*x7
                                x59 = 18*xi[1]/5
                                x60 = 6*x1
                                x61 = 66*xi[1]
                                x62 = 216*x8
                                x63 = 9*xi[1]/2
                                x64 = 45*x1
                                x65 = 18*x2
                                x66 = 37*xi[0]
                                x67 = 119*x1
                                x68 = 36*x3
                                x69 = 57*xi[0]
                                x70 = 108*x2
                                x71 = -324*x12
                                x72 = 10 - 324*x9
                                x73 = 54*x0
                                x74 = x36*xi[0] + 1
                                x75 = x29*xi[0] - x32
                                x76 = 312*x0 + x52
                                x77 = 36*x0
                                x78 = x35 - 54*x9
                                x79 = 108*x15
                                x80 = 111*x0 + x79 + 5
                                x81 = -x65 + x70*xi[1]
                                x82 = 69*x0 + x79 + 1
                                x83 = -54*x12 + x60
                                return jnp.stack([812*x0/5 + x1*x17 + 406*x1/5 + 1260*x10 - 1134*x11 - 1323*x12/2 + 1260*x13 - 1134*x14 + 1890*x15 - 2268*x16 - 2268*x18 + x19*x5 + 1296*x2*x6 - 441*x2/2 + 315*x3 + 1944*x4*xi[1]/5 - 1134*x4/5 + 406*x5/5 - 441*x6/2 + 315*x7 + 1944*x8*xi[0]/5 - 1134*x8/5 - 1323*x9/2 + 324*xi[0]**6/5 - 147*xi[0]/10 + 324*xi[1]**6/5 - 147*xi[1]/10 + 1, xi[0]*(-675*x1 + 1530*x2 - 1620*x3 + 648*x4 + 137*xi[0] - 10)/10, xi[1]*(-675*x5 + 1530*x6 - 1620*x7 + 648*x8 + 137*xi[1] - 10)/10, x23*x24, x30*(-461*x1 - 792*x10 + 216*x11 + 1752*x12 - 2088*x13 + 864*x14 + 864*x16 + 1296*x18 + 822*x2 + x22 + x25 - x26 - x27 + x28 + x29 - 684*x3 + 1038*x9 + 117*xi[0]), 4*xi[0]*(558*x1 - 1530*x12 - 324*x16 - 972*x18 - x19*xi[1] + 2106*x2*xi[1] - 1089*x2 + 972*x3 + x31 - x32 - x33 - x34 - 324*x4 + 162*x6*xi[0] - 459*x9 - 127*xi[0]), x30*(-307*x1 + 528*x12 - 828*x13 + 432*x14 + 216*x18 + 642*x2 + x25 - 612*x3 - x35 + x36 + x37*x5 + x37 + x39), x24*(130*x1 + 180*x2*xi[1] - 285*x2 - x20 + 288*x3 - x40 - x41*xi[1] - x42*xi[1] - x43 - 2*xi[1]), x44*(-180*x2 + x41 + x42 - 25*xi[0] + 2), x51*(36*x1 - 216*x12 + x46 + x47 + x50 - 6*xi[1]), 4*x0*(81*x0 + 18*x1 - 162*x12 + 18*x5 + x52 - 162*x9 - 9*xi[0] - 9*xi[1] + 1), x51*(x47 + 36*x5 + x53 + x56 - 216*x9 - 6*xi[0]), x44*(x57 + x58 - 180*x6 - 25*xi[1] + 2), x59*(-x21 - x43 + 130*x5 - x57*xi[0] - x58*xi[0] + 180*x6*xi[0] - 285*x6 + 288*x7 - 2*xi[0] - 27*xi[1]), x63*(x1*x61 - 828*x10 + 432*x11 + 216*x16 + x39 + x45 - 307*x5 + 642*x6 - x60 + x61 + x62 - 612*x7 + 528*x9), 4*xi[1]*(-459*x12 - 972*x16 - x17*xi[0] - 324*x18 + 162*x2*xi[1] - x34 + 558*x5 + 2106*x6*xi[0] - 1089*x6 + x64 - x65 - x66 + 972*x7 - 324*x8 - 1530*x9 - 127*xi[1]), x63*(-2088*x10 + 864*x11 + 1038*x12 - 792*x13 + 216*x14 + 1296*x16 + 864*x18 + x22 - 461*x5 + 822*x6 + x62 - x67 - x68 + x69 - 684*x7 + x70 + 1752*x9 + 117*xi[1]), x23*x59, x73*(238*x0 + 144*x10 + 144*x13 + x26 + x27 - x28 - x29 + x38 + x67 + x68 - x69 - x70 + x71 + x72), x73*(36*x1*xi[1] - 47*x1 + 72*x2 - x48*xi[1] - x68 - x74 + 12*xi[0] + xi[1]), x73*(-x27 + 36*x5*xi[0] - 47*x5 - x54*xi[0] + 72*x6 - x74 + xi[0] + 12*xi[1]), x77*(594*x1*xi[1] - 267*x1 - 324*x13 + 288*x2 - x31 + x33 - x42 - x72 - x75 - x76 + 97*xi[0]), x77*(195*x1 - 306*x12 - 252*x2 + x42 + x49 + x53 + x78 + x80 - 56*xi[0]), x77*(180*x1*xi[1] - 27*x1 - x78 - x81 - x82 + 10*xi[0] + 7*xi[1]), x77*(180*x5*xi[0] - 27*x5 - x75 - x82 - x83 + 7*xi[0] + 10*xi[1]), x77*(x46 + 195*x5 + x55 + x58 - 252*x6 + x80 + x83 - 306*x9 - 56*xi[1]), x77*(-324*x10 + 594*x5*xi[0] - 267*x5 - x58 + 288*x6 - x64 + x66 - x71 - x76 - x81 + 97*xi[1] - 10), x40*xi[1]*(324*x0 + 72*x1 - 504*x12 + 432*x15 + 72*x5 + x50 + x56 - 504*x9 - 41*xi[0] - 41*xi[1] + 5)])

                        case 36:
                            def shape_functions(xi):
                                x0 = xi[0]**2
                                x1 = xi[0]**3
                                x2 = xi[0]**4
                                x3 = xi[0]**5
                                x4 = xi[0]**6
                                x5 = xi[1]**2
                                x6 = xi[1]**3
                                x7 = xi[1]**4
                                x8 = xi[1]**5
                                x9 = xi[1]**6
                                x10 = x5*xi[0]
                                x11 = x7*xi[0]
                                x12 = x0*xi[1]
                                x13 = x2*xi[1]
                                x14 = x0*x6
                                x15 = x1*x5
                                x16 = xi[0]*xi[1]
                                x17 = 16807*x4
                                x18 = 16807*x9
                                x19 = x6*xi[0]
                                x20 = x8*xi[0]
                                x21 = x1*xi[1]
                                x22 = x3*xi[1]
                                x23 = x0*x5
                                x24 = x0*x7
                                x25 = x1*x6
                                x26 = x2*x5
                                x27 = 35728*x0 - 81585*x1 - 244755*x10 - 324135*x11 - 244755*x12 - 324135*x13 - 648270*x14 - 648270*x15 + 71456*x16 + x17 + x18 + 404740*x19 + 101185*x2 + 100842*x20 + 404740*x21 + 100842*x22 + 607110*x23 + 252105*x24 + 336140*x25 + 252105*x26 - 64827*x3 + 35728*x5 - 81585*x6 + 101185*x7 - 64827*x8 - 8028*xi[0] - 8028*xi[1] + 720
                                x28 = 49*xi[0]/720
                                x29 = 16807*x8
                                x30 = x29*xi[0]
                                x31 = 12005*x6
                                x32 = 2754*xi[1]
                                x33 = 2401*x8
                                x34 = 8225*x5
                                x35 = 8575*x7
                                x36 = 35728*x16 + 303555*x23 + 168070*x25 + 360
                                x37 = 49*xi[0]/240
                                x38 = 16807*x24
                                x39 = 7203*x7*xi[0]
                                x40 = 18410*x16 + 133427*x23 + 67228*x25 + 240
                                x41 = 49*xi[0]/144
                                x42 = 9751*x16 + 48363*x23 + 16807*x25 + 180
                                x43 = 24010*x15
                                x44 = 16807*x26
                                x45 = 4886*x16 + 12005*x23 + 144
                                x46 = 16807*x3
                                x47 = x46*xi[1]
                                x48 = 29155*x1
                                x49 = 1918*xi[0]
                                x50 = x49*xi[1] + 120
                                x51 = 49*x16/720
                                x52 = 2401*x2
                                x53 = 1715*x0
                                x54 = -24010*x21
                                x55 = 12005*x12
                                x56 = -2450*x16 - 24
                                x57 = 49*x16/240
                                x58 = -14406*x23
                                x59 = -1617*x16 + x58 - 12
                                x60 = 49*x16/144
                                x61 = 2401*x7
                                x62 = 1715*x5
                                x63 = -24010*x19
                                x64 = 12005*x10
                                x65 = 29155*x6
                                x66 = 49*xi[1]/720
                                x67 = 24010*x14
                                x68 = 49*xi[1]/240
                                x69 = 49*xi[1]/144
                                x70 = 7203*x2*xi[1]
                                x71 = 12005*x1
                                x72 = 2754*xi[0]
                                x73 = 2401*x3
                                x74 = 8225*x0
                                x75 = 8575*x2
                                x76 = 343*x16/120
                                x77 = x52*xi[1]
                                x78 = -350*xi[0]*xi[1] - 24
                                x79 = x61*xi[0]
                                x80 = -343*x7 + x79
                                x81 = -6972*x16 - 24696*x23 - 120
                                x82 = 343*x16/48
                                x83 = 7203*x15
                                x84 = 2401*x14
                                x85 = -1029*x19 + 98*x6 + x84
                                x86 = -9261*x0*x5 - 2751*xi[0]*xi[1] - 60
                                x87 = 343*x16/36
                                x88 = 2401*x15
                                x89 = 539*x10 - 42*x5 + x88
                                x90 = -2058*x23
                                x91 = -1085*x16 + x90 - 36
                                x92 = -343*x2 + x77
                                x93 = x90 - 658*xi[0]*xi[1] - 6
                                x94 = 98*x1 - 1029*x21 + x88
                                x95 = -42*x0 + 539*x12 + x84
                                x96 = 7203*x14
                                x97 = 343*x16/24
                                x98 = -1365*x16 - 6860*x23 - 12
                                return jnp.stack([16807*x0*x5/3 + 117649*x0*x7/12 - 823543*x0*x8/240 + 22981*x0/180 + 117649*x1*x6/9 - 823543*x1*x7/144 + 33614*x1*xi[1]/9 - 331681*x1/720 - 331681*x10/240 - 386561*x11/72 - 331681*x12/240 - 386561*x13/72 - 386561*x14/36 - 386561*x15/36 + 117649*x2*x5/12 - 823543*x2*x6/144 + 16807*x2/18 - 823543*x3*x5/240 + 117649*x3*xi[1]/30 - 386561*x3/360 - 823543*x4*xi[1]/720 + 117649*x4/180 + 22981*x5/180 + 33614*x6*xi[0]/9 - 331681*x6/720 + 16807*x7/18 + 117649*x8*xi[0]/30 - 386561*x8/360 - 823543*x9*xi[0]/720 + 117649*x9/180 - 117649*xi[0]**7/720 + 22981*xi[0]*xi[1]/90 - 363*xi[0]/20 - 117649*xi[1]**7/720 - 363*xi[1]/20 + 1, xi[0]*(79576*x0 - 252105*x1 + 420175*x2 - 352947*x3 + 117649*x4 - 12348*xi[0] + 720)/720, xi[1]*(79576*x5 - 252105*x6 + 420175*x7 - 352947*x8 + 117649*x9 - 12348*xi[1] + 720)/720, x27*x28, x37*(264110*x0*x6 + 151165*x0*xi[1] - 27503*x0 + 384160*x1*x5 + 69580*x1 - x17 - 118335*x19 + 252105*x2*xi[1] - 92610*x2 - 286405*x21 - 84035*x22 - 84035*x24 - 168070*x26 + 62426*x3 - x30 + x31 + x32 + x33 - x34 - x35 - x36 + 93590*x5*xi[0] + 72030*x7*xi[0] + 5274*xi[0]), x41*(21784*x0 - 59731*x1 - 32781*x10 - 90356*x12 - 187278*x13 - 81634*x14 - 201684*x15 + x17 + 25382*x19 + 84721*x2 + 193452*x21 + 67228*x22 + 100842*x26 - 60025*x3 + x38 - x39 + x40 + 2506*x5 - 2156*x6 + 686*x7 - 3796*xi[0] - 1276*xi[1]), x41*(14406*x0*x6 + 51744*x0*xi[1] - 17815*x0 + 86436*x1*x5 + 51744*x1 - x17 - 3773*x19 + 129654*x2*xi[1] - 77518*x2 - 122108*x21 - 50421*x22 - 50421*x26 + 57624*x3 - x42 + 10584*x5*xi[0] - 756*x5 + 294*x6 + 2952*xi[0] + 642*xi[1]), x37*(15008*x0 - 45325*x1 - 2450*x10 - 27195*x12 - 79233*x13 + x17 + 71001*x2 + 68600*x21 + 33614*x22 - 55223*x3 - x43 + x44 + x45 + 168*x5 - 2412*xi[0] - 312*xi[1]), x28*(11025*x0*xi[1] - 12943*x0 + 40180*x1 - x17 + 36015*x2*xi[1] - 65170*x2 + 52822*x3 - x47 - x48*xi[1] - x50 + 2038*xi[0] + 120*xi[1]), x51*(-11025*x0 - 36015*x2 + x46 + x48 + x49 - 120), x57*(3430*x1 + 16807*x13 - x52 - x53 + x54 + x55 + x56 + 350*xi[0] + 168*xi[1]), x60*(-588*x0 + 686*x1 + 3773*x10 + 6174*x12 + 16807*x15 - 7203*x21 - 294*x5 + x59 + 154*xi[0] + 126*xi[1]), x60*(-294*x0 + 6174*x10 + 3773*x12 + 16807*x14 - 7203*x19 - 588*x5 + x59 + 686*x6 + 126*xi[0] + 154*xi[1]), x57*(16807*x11 + x56 + 3430*x6 - x61 - x62 + x63 + x64 + 168*xi[0] + 350*xi[1]), x51*(x29 - 11025*x5 + x65 - 36015*x7 + 1918*xi[1] - 120), x66*(-x18 - x30 + 11025*x5*xi[0] - 12943*x5 - x50 + 40180*x6 - x65*xi[0] + 36015*x7*xi[0] - 65170*x7 + 52822*x8 + 120*xi[0] + 2038*xi[1]), x68*(168*x0 - 27195*x10 - 79233*x11 - 2450*x12 + x18 + 68600*x19 + 33614*x20 + x38 + x45 + 15008*x5 - 45325*x6 - x67 + 71001*x7 - 55223*x8 - 312*xi[0] - 2412*xi[1]), x69*(86436*x0*x6 + 10584*x0*xi[1] - 756*x0 + 14406*x1*x5 + 294*x1 - x18 - 122108*x19 - 50421*x20 - 3773*x21 - 50421*x24 - x42 + 51744*x5*xi[0] - 17815*x5 + 51744*x6 + 129654*x7*xi[0] - 77518*x7 + 57624*x8 + 642*xi[0] + 2952*xi[1]), x69*(2506*x0 - 2156*x1 - 90356*x10 - 187278*x11 - 32781*x12 - 201684*x14 - 81634*x15 + x18 + 193452*x19 + 686*x2 + 67228*x20 + 25382*x21 + 100842*x24 + x40 + x44 + 21784*x5 - 59731*x6 + 84721*x7 - x70 - 60025*x8 - 1276*xi[0] - 3796*xi[1]), x68*(384160*x0*x6 + 93590*x0*xi[1] + 264110*x1*x5 - x18 - 286405*x19 + 72030*x2*xi[1] - 84035*x20 - 118335*x21 - 168070*x24 - 84035*x26 - x36 - x47 + 151165*x5*xi[0] - 27503*x5 + 69580*x6 + 252105*x7*xi[0] - 92610*x7 + x71 + x72 + x73 - x74 - x75 + 62426*x8 + 5274*xi[1]), x27*x66, x76*(51450*x0*x5 + 34300*x1*xi[1] - 36015*x10 - 12005*x11 - 36015*x12 - 12005*x13 - x31 - x32 - x33 + x34 + x35 - x43 + 34300*x6*xi[0] - x67 - x71 - x72 - x73 + x74 + x75 + 16450*xi[0]*xi[1] + 360), x76*(2065*x0 + 3430*x1*xi[1] - 5145*x1 + 5831*x2 - x53*xi[1] - x73 - x77 - x78 - 374*xi[0] - 24*xi[1]), x76*(-x33 + 2065*x5 + 3430*x6*xi[0] - 5145*x6 - x62*xi[0] + 5831*x7 - x78 - x79 - 24*xi[0] - 374*xi[1]), x82*(-5719*x0 + 9849*x1 + 20776*x12 + 9604*x13 + 9604*x14 + 14406*x15 - 8918*x19 - 7889*x2 - 1253*x5 + x54 + 1078*x6 + x64 + x73 + x80 + x81 + 1478*xi[0] + 638*xi[1]), x87*(3969*x0 + 15435*x1*xi[1] - 7987*x1 - 2940*x10 - 10829*x12 + 7203*x2 + 252*x5 - x70 - x73 - x83 - x85 - x86 - 844*xi[0] - 214*xi[1]), x82*(-2807*x0 + 6419*x1 + 4900*x12 + 4802*x13 - 6517*x2 - 8575*x21 + x73 + x89 + x91 + 540*xi[0] + 78*xi[1]), x82*(371*x0 + 4802*x1*xi[1] - 637*x1 - 2891*x12 - x89 - x92 - x93 - 83*xi[0] - 48*xi[1]), x87*(4459*x0*x5 + 140*x0 - 1568*x10 - 1568*x12 + 140*x5 - x85 - x94 + 525*xi[0]*xi[1] - 46*xi[0] - 46*xi[1] + 4), x82*(-2891*x10 + 371*x5 + 4802*x6*xi[0] - 637*x6 - x80 - x93 - x95 - 48*xi[0] - 83*xi[1]), x82*(4900*x10 + 4802*x11 - 8575*x19 + x33 - 2807*x5 + 6419*x6 - 6517*x7 + x91 + x95 + 78*xi[0] + 540*xi[1]), x87*(252*x0 - 10829*x10 - 2940*x12 - x33 - x39 + 3969*x5 + 15435*x6*xi[0] - 7987*x6 + 7203*x7 - x86 - x94 - x96 - 214*xi[0] - 844*xi[1]), x82*(-1253*x0 + 1078*x1 + 20776*x10 + 9604*x11 + 14406*x14 + 9604*x15 - 8918*x21 + x33 - 5719*x5 + x55 + 9849*x6 + x63 - 7889*x7 + x81 + x92 + 638*xi[0] + 1478*xi[1]), x97*(875*x0 + 7546*x1*xi[1] - 931*x1 - 8036*x10 - 8036*x12 + 875*x5 - x58 + 7546*x6*xi[0] - 931*x6 - x80 - x83 - x92 - x96 + 3220*xi[0]*xi[1] - 317*xi[0] - 317*xi[1] + 30), x97*(-581*x0 + 784*x1 + 2254*x10 + 4998*x12 + 4802*x15 - 6174*x21 - 196*x5 + x85 + x92 + x98 + 152*xi[0] + 110*xi[1]), x97*(-196*x0 + 4998*x10 + 2254*x12 + 4802*x14 - 6174*x19 - 581*x5 + 784*x6 + x80 + x94 + x98 + 110*xi[0] + 152*xi[1])])

                        case 45:
                            def shape_functions(xi):
                                x0 = xi[0]*xi[1]
                                x1 = xi[0]**2
                                x2 = xi[0]**3
                                x3 = xi[0]**4
                                x4 = xi[0]**5
                                x5 = xi[0]**6
                                x6 = xi[0]**7
                                x7 = xi[1]**2
                                x8 = xi[1]**3
                                x9 = xi[1]**4
                                x10 = xi[1]**5
                                x11 = xi[1]**6
                                x12 = xi[1]**7
                                x13 = x7*xi[0]
                                x14 = x8*xi[0]
                                x15 = x9*xi[0]
                                x16 = x10*xi[0]
                                x17 = x11*xi[0]
                                x18 = x1*xi[1]
                                x19 = x2*xi[1]
                                x20 = x3*xi[1]
                                x21 = x4*xi[1]
                                x22 = x5*xi[1]
                                x23 = x1*x7
                                x24 = x1*x8
                                x25 = x1*x9
                                x26 = x1*x10
                                x27 = x2*x7
                                x28 = x2*x8
                                x29 = x2*x9
                                x30 = 65536*x29
                                x31 = x3*x7
                                x32 = x3*x8
                                x33 = 65536*x32
                                x34 = x4*x7
                                x35 = 16384*x6
                                x36 = 16384*x12
                                x37 = -48860*x0 - 772800*x23 - 1433600*x28 - 315
                                x38 = 1075200*x1*x9 + 24430*x1 + 430080*x10*xi[0] - 130816*x10 + 71680*x11 - 221088*x13 - 654080*x15 - 114688*x17 - 221088*x18 + 515200*x2*xi[1] - 73696*x2 - 654080*x20 - 114688*x22 - 1308160*x24 - 344064*x26 - 1308160*x27 - 573440*x29 + 1075200*x3*x7 + 128800*x3 - 573440*x32 - 344064*x34 - x35 - x36 - x37 + 430080*x4*xi[1] - 130816*x4 + 71680*x5 + 24430*x7 + 515200*x8*xi[0] - 73696*x8 + 128800*x9 - 4329*xi[0] - 4329*xi[1]
                                x39 = 64*xi[0]/315
                                x40 = 32768*x6
                                x41 = 28480*x9
                                x42 = 12154*x7
                                x43 = 4096*x11
                                x44 = 3069*xi[1]
                                x45 = 16896*x10
                                x46 = 25080*x8
                                x47 = 16*xi[0]/45
                                x48 = 2070*x7
                                x49 = 1920*x9
                                x50 = 512*x10
                                x51 = 743*xi[1]
                                x52 = 2840*x8
                                x53 = -180000*x1*x7 - 307200*x2*x8 - 13056*xi[0]*xi[1] - 105
                                x54 = 64*xi[0]/45
                                x55 = 262144*x5
                                x56 = 49152*x25
                                x57 = -29476*x0 - 307200*x23 - 409600*x28 - 315
                                x58 = 252*x7
                                x59 = 96*x8
                                x60 = 219*xi[1]
                                x61 = -28320*x1*x7 - 20480*x2*x8 - 4154*xi[0]*xi[1] - 63
                                x62 = 61440*x31
                                x63 = -4350*x0 - 14400*x23 - 105
                                x64 = 45*xi[1]
                                x65 = 6496*x1
                                x66 = 16384*x5
                                x67 = 44800*x3
                                x68 = -882*xi[0]*xi[1] - 45
                                x69 = 64*x0/315
                                x70 = 5440*x2
                                x71 = 4096*x4
                                x72 = 274*xi[0]
                                x73 = -61440*x20
                                x74 = 32768*x21
                                x75 = 2192*x0 + 15
                                x76 = 16*x0/45
                                x77 = 600*x0 + 8960*x23 + 3
                                x78 = 64*x0/45
                                x79 = 5440*x8
                                x80 = 4096*x10
                                x81 = -61440*x15
                                x82 = 32768*x16
                                x83 = 6496*x7
                                x84 = 44800*x9
                                x85 = 16384*x11
                                x86 = 45*xi[0]
                                x87 = 64*xi[1]/315
                                x88 = 32768*x12
                                x89 = 61440*x25
                                x90 = 16*xi[1]/45
                                x91 = 252*x1
                                x92 = 96*x2
                                x93 = 219*xi[0]
                                x94 = 64*xi[1]/45
                                x95 = 262144*x11
                                x96 = 49152*x31
                                x97 = 2070*x1
                                x98 = 1920*x3
                                x99 = 512*x4
                                x100 = 743*xi[0]
                                x101 = 2840*x2
                                x102 = 28480*x3
                                x103 = 12154*x1
                                x104 = 4096*x5
                                x105 = 3069*xi[0]
                                x106 = 16896*x4
                                x107 = 25080*x2
                                x108 = 24576*x16
                                x109 = 24576*x21
                                x110 = 256*x0/45
                                x111 = x71*xi[1]
                                x112 = x72*xi[1] + 15
                                x113 = x80*xi[0]
                                x114 = -17920*x15
                                x115 = x113 - x50
                                x116 = 10084*x0 + 79680*x23 + 40960*x28 + 105
                                x117 = 256*x0/15
                                x118 = 8192*x5
                                x119 = 3072*x9
                                x120 = -x119*xi[0] + 8192*x25 + 256*x9
                                x121 = 8404*x0 + 63616*x23 + 32768*x28 + 105
                                x122 = 128*x0/9
                                x123 = 24576*x31
                                x124 = -6144*x1*x8 + 1408*x14 - x59
                                x125 = 8192*x28
                                x126 = 3716*x0 + x125 + 20352*x23 + 63
                                x127 = -17920*x20
                                x128 = -400*x13 - 5120*x27 + 4096*x31 + 24*x7
                                x129 = 8192*x21 + 21
                                x130 = 2240*x23
                                x131 = 798*x0 + x130
                                x132 = x111 - x99
                                x133 = 474*x0 + x130 + 3
                                x134 = 3072*x3
                                x135 = -x134*xi[1] + 256*x3 + 8192*x31
                                x136 = 608*x0 + x125 + 9856*x23 + 3
                                x137 = 1408*x19 - 6144*x2*x7 - x92
                                x138 = 24*x1 - 400*x18 - 5120*x24 + 4096*x25
                                x139 = 8192*x16 + 21
                                x140 = 8192*x11
                                x141 = 24576*x25
                                x142 = 16384*x21 - 2048*x4
                                x143 = -2048*x10 + 16384*x16
                                x144 = 32*x0/3
                                x145 = 16384*x28
                                x146 = 3532*x0 + x145 + 30208*x23 + 21
                                x147 = 3644*x0 + 39424*x23 + 24576*x28
                                x148 = 256*x0/9
                                return jnp.stack([118124*x0/315 + 524288*x1*x11/45 + 59062*x1/315 + 1048576*x10*x2/45 - 18432*x10/5 + 53248*x11/15 + 1048576*x12*xi[0]/315 - 65536*x12/35 - 12816*x13/5 + 136832*x14/15 - 18432*x15 + 106496*x16/5 - 65536*x17/5 - 12816*x18/5 + 136832*x19/15 - 4272*x2/5 - 18432*x20 + 106496*x21/5 - 65536*x22/5 + 68416*x23/5 - 36864*x24 + 53248*x25 - 196608*x26/5 - 36864*x27 + 212992*x28/3 + 262144*x3*x9/9 + 34208*x3/15 - x30 + 53248*x31 - x33 - 196608*x34/5 + 1048576*x4*x8/45 - 18432*x4/5 + 524288*x5*x7/45 + 53248*x5/15 + 1048576*x6*xi[1]/315 - 65536*x6/35 + 59062*x7/315 - 4272*x8/5 + 34208*x9/15 + 131072*xi[0]**8/315 - 761*xi[0]/35 + 131072*xi[1]**8/315 - 761*xi[1]/35 + 1, xi[0]*(-52528*x1 + 216608*x2 - 501760*x3 + 659456*x4 - 458752*x5 + 131072*x6 + 6534*xi[0] - 315)/315, xi[1]*(659456*x10 - 458752*x11 + 131072*x12 - 52528*x7 + 216608*x8 - 501760*x9 + 6534*xi[1] - 315)/315, x38*x39, x47*(-36706*x1 + 172472*x13 - 314560*x14 + 312320*x15 - 159744*x16 + 32768*x17 + 269704*x18 - 715840*x19 + 122312*x2 + 995840*x20 - 700416*x21 + 196608*x22 + 1080320*x24 - 737280*x25 + 196608*x26 + 1536000*x27 + 491520*x29 - 229120*x3 - 1413120*x31 + 655360*x32 + 491520*x34 + x37 + 244736*x4 + x40 - x41 - x42 - x43 + x44 + x45 + x46 - 139264*x5 + 5589*xi[0]), x54*(92160*x1*x9 + 14346*x1 + 6144*x10*xi[0] - 33360*x13 - 25600*x15 - 81976*x18 + 242400*x2*xi[1] - 51456*x2 - 367360*x20 - 81920*x22 - 188160*x24 - 16384*x26 - 416000*x27 - 81920*x29 + 430080*x3*x7 + 102240*x3 - 163840*x32 - 163840*x34 - x35 + 276480*x4*xi[1] - 114432*x4 + x48 + x49 + 67584*x5 - x50 - x51 - x52 - x53 + 41760*x8*xi[0] - 2003*xi[0]), 4*xi[0]*(-46624*x1 + 51664*x13 - 39680*x14 + 11264*x15 + 198176*x18 - 634880*x19 + 175888*x2 + 1038336*x20 - 835584*x21 + 204800*x24 + 803840*x27 - 366592*x3 + x30 - 933888*x31 + 262144*x32 + 393216*x34 + 428032*x4 + x55*xi[1] - x55 - x56 + x57 + 65536*x6 - 3012*x7 + 2496*x8 - 768*x9 + 6219*xi[0] + 1599*xi[1])/9, x54*(9782*x1 - 4488*x13 - 29128*x18 + 98560*x2*xi[1] - 38176*x2 - 171776*x20 - 49152*x22 - 8960*x24 - 80640*x27 + 104448*x3*x7 + 82592*x3 - 16384*x32 - 49152*x34 - x35 + 147456*x4*xi[1] - 100096*x4 + 63488*x5 + x58 - x59 - x60 - x61 + 1600*x8*xi[0] - 1269*xi[0]), x47*(-16830*x1 + 2192*x13 + 31384*x18 - 110400*x19 + 67272*x2 + 202240*x20 - 184320*x21 + 65536*x22 + 43520*x27 - 149760*x3 + 32768*x34 + 187392*x4 + x40 - 122880*x5 - x62 + x63 - 120*x7 + 2143*xi[0] + 225*xi[1]), x39*(7378*x1 + 23520*x2*xi[1] - 30016*x2 + 68320*x3 - x35 + 43008*x4*xi[1] - 87808*x4 + 59392*x5 - x64 - x65*xi[1] - x66*xi[1] - x67*xi[1] - x68 - 927*xi[0]), x69*(-23520*x2 - 43008*x4 + x65 + x66 + x67 - 882*xi[0] + 45), x76*(1800*x1 - 14400*x18 + 43520*x19 + 7680*x3 - x70 - x71 - x72 + x73 + x74 + x75 - 120*xi[1]), x78*(280*x1 - 1600*x13 - 3360*x18 + 7680*x19 - 640*x2 - 6144*x20 - 20480*x27 + 512*x3 + 16384*x31 + 96*x7 + x77 - 50*xi[0] - 36*xi[1]), 4*x0*(1936*x0 + 576*x1 - 8448*x13 + 11264*x14 - 8448*x18 + 11264*x19 - 768*x2 + 36864*x23 - 49152*x24 - 49152*x27 + 65536*x28 + 576*x7 - 768*x8 - 132*xi[0] - 132*xi[1] + 9)/9, x78*(96*x1 - 3360*x13 + 7680*x14 - 6144*x15 - 1600*x18 - 20480*x24 + 16384*x25 + 280*x7 + x77 - 640*x8 + 512*x9 - 36*xi[0] - 50*xi[1]), x76*(-14400*x13 + 43520*x14 + 1800*x7 + x75 - x79 - x80 + x81 + x82 + 7680*x9 - 120*xi[0] - 274*xi[1]), x69*(-43008*x10 - 23520*x8 + x83 + x84 + x85 - 882*xi[1] + 45), x87*(43008*x10*xi[0] - 87808*x10 + 59392*x11 - x36 - x68 + 7378*x7 + 23520*x8*xi[0] - 30016*x8 - x83*xi[0] - x84*xi[0] - x85*xi[0] - x86 + 68320*x9 - 927*xi[1]), x90*(-120*x1 + 187392*x10 - 122880*x11 + 31384*x13 - 110400*x14 + 202240*x15 - 184320*x16 + 65536*x17 + 2192*x18 + 43520*x24 + 32768*x26 + x63 - 16830*x7 + 67272*x8 + x88 - x89 - 149760*x9 + 225*xi[0] + 2143*xi[1]), x94*(104448*x1*x9 + 147456*x10*xi[0] - 100096*x10 + 63488*x11 - 29128*x13 - 171776*x15 - 49152*x17 - 4488*x18 + 1600*x2*xi[1] - 80640*x24 - 49152*x26 - 8960*x27 - 16384*x29 - x36 - x61 + 9782*x7 + 98560*x8*xi[0] - 38176*x8 + 82592*x9 + x91 - x92 - x93 - 1269*xi[1]), 4*xi[1]*(-3012*x1 + 428032*x10 + 65536*x12 + 198176*x13 - 634880*x14 + 1038336*x15 - 835584*x16 + 51664*x18 - 39680*x19 + 2496*x2 + 11264*x20 + 803840*x24 - 933888*x25 + 393216*x26 + 204800*x27 + 262144*x29 - 768*x3 + x33 + x57 - 46624*x7 + 175888*x8 - 366592*x9 + x95*xi[0] - x95 - x96 + 1599*xi[0] + 6219*xi[1])/9, x94*(430080*x1*x9 + 276480*x10*xi[0] - 114432*x10 - x100 - x101 + 67584*x11 - 81976*x13 - 367360*x15 - 81920*x17 - 33360*x18 + 41760*x2*xi[1] - 25600*x20 - 416000*x24 - 163840*x26 - 188160*x27 - 163840*x29 + 92160*x3*x7 - 81920*x32 - 16384*x34 - x36 + 6144*x4*xi[1] - x53 + 14346*x7 + 242400*x8*xi[0] - 51456*x8 + 102240*x9 + x97 + x98 - x99 - 2003*xi[1]), x90*(244736*x10 - x102 - x103 - x104 + x105 + x106 + x107 - 139264*x11 + 269704*x13 - 715840*x14 + 995840*x15 - 700416*x16 + 196608*x17 + 172472*x18 - 314560*x19 + 312320*x20 - 159744*x21 + 32768*x22 + 1536000*x24 - 1413120*x25 + 491520*x26 + 1080320*x27 + 655360*x29 - 737280*x31 + 491520*x32 + 196608*x34 + x37 - 36706*x7 + 122312*x8 + x88 - 229120*x9 + 5589*xi[1]), x38*x87, x110*(24308*x0 + x102 + x103 + x104 - x105 - x106 - x107 + x108 + x109 - 75240*x13 + 113920*x14 - 84480*x15 - 75240*x18 + 113920*x19 - 84480*x20 + 170880*x23 - 168960*x24 - 168960*x27 + 81920*x28 + x41 + x42 + x43 - x44 - x45 - x46 + x62 + x89 + 315), x110*(1800*x1*xi[1] - 2074*x1 - x104 - x111 - x112 + 7240*x2 + 7680*x3*xi[1] - 13120*x3 + 11776*x4 - x70*xi[1] + 289*xi[0] + 15*xi[1]), x110*(11776*x10 - x112 - x113 - x43 + 1800*x7*xi[0] - 2074*x7 - x79*xi[0] + 7240*x8 + 7680*x9*xi[0] - 13120*x9 + 15*xi[0] + 289*xi[1]), x117*(66560*x1*x8 + 41640*x1*xi[1] - 8014*x1 - x104 - x114 - x115 - x116 - 30400*x14 - 75840*x19 + 97280*x2*x7 + 19400*x2 - 20480*x21 - 20480*x25 + 64000*x3*xi[1] - 24640*x3 - 40960*x31 + 15872*x4 - x48 - x49 + x51 + x52 + 25080*x7*xi[0] + 1583*xi[0]), x122*(10760*x1 + x118 + x120 + x121 - 14544*x13 + 11008*x14 - 43648*x18 + 95232*x19 - 29936*x2 - 92160*x20 - 38912*x24 - 98304*x27 + 42368*x3 - 29696*x4 + 1004*x7 + x74 - 832*x8 + x96 - 1793*xi[0] - 533*xi[1]), x122*(21696*x1*xi[1] - 7496*x1 - x109 - x118 - x123 - x124 - x126 - 55168*x19 + 39936*x2*x7 + 23184*x2 - 36224*x3 + 27648*x4 - x58 + x60 + 3984*x7*xi[0] - x73 + 1143*xi[0]), x117*(2734*x1 + x104 + x127 + x128 + x129 + x131 - 5000*x18 + 14080*x19 - 9080*x2 + 15424*x3 - 12800*x4 - x64 - 395*xi[0]), x117*(2920*x1*xi[1] - 330*x1 - x128 - x132 - x133 - 8000*x19 + 920*x2 + 9728*x3*xi[1] - 1152*x3 + 53*xi[0] + 27*xi[1]), x122*(3024*x1*xi[1] - 236*x1 - x124 - x135 - x136 - 5632*x19 + 17408*x2*x7 + 448*x2 + 2032*x7*xi[0] - 132*x7 + 47*xi[0] + 39*xi[1]), x122*(17408*x1*x8 + 2032*x1*xi[1] - 132*x1 - x120 - x136 - x137 - 5632*x14 + 3024*x7*xi[0] - 236*x7 + 448*x8 + 39*xi[0] + 47*xi[1]), x117*(-x115 - x133 - x138 - 8000*x14 + 2920*x7*xi[0] - 330*x7 + 920*x8 + 9728*x9*xi[0] - 1152*x9 + 27*xi[0] + 53*xi[1]), x117*(-12800*x10 + x114 - 5000*x13 + x131 + x138 + x139 + 14080*x14 + x43 + 2734*x7 - 9080*x8 - x86 + 15424*x9 - 395*xi[1]), x122*(39936*x1*x8 + 3984*x1*xi[1] + 27648*x10 - x108 - x126 - x137 - 55168*x14 - x140 - x141 + 21696*x7*xi[0] - 7496*x7 + 23184*x8 - x81 - 36224*x9 - x91 + x93 + 1143*xi[1]), x122*(1004*x1 - 29696*x10 + x121 - 43648*x13 + x135 + 95232*x14 + x140 - 92160*x15 - 14544*x18 + 11008*x19 - 832*x2 - 98304*x24 - 38912*x27 + x56 + 10760*x7 - 29936*x8 + x82 + 42368*x9 - 533*xi[0] - 1793*xi[1]), x117*(97280*x1*x8 + 25080*x1*xi[1] + 15872*x10 + x100 + x101 - x116 - x127 - x132 - 75840*x14 - 20480*x16 - 30400*x19 + 66560*x2*x7 - 40960*x25 - 20480*x31 - x43 + 41640*x7*xi[0] - 8014*x7 + 19400*x8 + 64000*x9*xi[0] - 24640*x9 - x97 - x98 + 1583*xi[1]), x144*(17256*x0 + 5268*x1 - 60704*x13 + 91904*x14 + x142 + x143 - 63488*x15 - 60704*x18 + 91904*x19 - 8864*x2 - 63488*x20 + 169984*x23 - 180224*x24 + 65536*x25 - 180224*x27 + 98304*x28 + 6912*x3 + 65536*x31 + 5268*x7 - 8864*x8 + 6912*x9 - 1373*xi[0] - 1373*xi[1] + 105), x144*(2028*x1 - 6016*x13 + 2816*x14 + x142 + x146 - 19808*x18 + 47104*x19 - 5024*x2 - 47104*x20 - 12288*x24 - 57344*x27 + 5376*x3 + 32768*x31 + 384*x7 - 192*x8 - 353*xi[0] - 213*xi[1]), x144*(384*x1 - 19808*x13 + 47104*x14 + x143 + x146 - 47104*x15 - 6016*x18 + 2816*x19 - 192*x2 - 57344*x24 + 32768*x25 - 12288*x27 + 2028*x7 - 5024*x8 + 5376*x9 - 213*xi[0] - 353*xi[1]), x148*(31744*x1*x8 + 17504*x1*xi[1] - 1632*x1 - x120 - x123 - x129 - x134 - 9216*x14 - x147 - 33536*x19 + 55296*x2*x7 + 3376*x2 + 27648*x3*xi[1] + 1024*x4 + 9456*x7*xi[0] - 668*x7 + 704*x8 + 325*xi[0] + 241*xi[1]), x148*(1384*x0 + 412*x1 + x120 - 5616*x13 + x135 + 7424*x14 + x145 - 5616*x18 + 7424*x19 - 576*x2 + 21504*x23 - 24576*x24 - 24576*x27 + 412*x7 - 576*x8 - 99*xi[0] - 99*xi[1] + 7), x148*(55296*x1*x8 + 9456*x1*xi[1] - 668*x1 + 1024*x10 - x119 - x135 - x139 - 33536*x14 - x141 - x147 - 9216*x19 + 31744*x2*x7 + 704*x2 + 17504*x7*xi[0] - 1632*x7 + 3376*x8 + 27648*x9*xi[0] + 241*xi[0] + 325*xi[1])])

                        case 55:
                            def shape_functions(xi):
                                x0 = xi[0]**2
                                x1 = xi[0]**3
                                x2 = xi[0]**4
                                x3 = xi[0]**5
                                x4 = xi[0]**6
                                x5 = xi[0]**7
                                x6 = xi[0]**8
                                x7 = xi[1]**2
                                x8 = xi[1]**3
                                x9 = xi[1]**4
                                x10 = xi[1]**5
                                x11 = xi[1]**6
                                x12 = xi[1]**7
                                x13 = xi[1]**8
                                x14 = x7*xi[0]
                                x15 = x9*xi[0]
                                x16 = x11*xi[0]
                                x17 = x0*xi[1]
                                x18 = x2*xi[1]
                                x19 = x4*xi[1]
                                x20 = x0*x8
                                x21 = x0*x10
                                x22 = x1*x7
                                x23 = x1*x9
                                x24 = x2*x8
                                x25 = x3*x7
                                x26 = 29760696*x3
                                x27 = 29760696*x10
                                x28 = xi[0]*xi[1]
                                x29 = 531441*x6
                                x30 = 531441*x13
                                x31 = x8*xi[0]
                                x32 = x10*xi[0]
                                x33 = x12*xi[0]
                                x34 = x1*xi[1]
                                x35 = x3*xi[1]
                                x36 = x5*xi[1]
                                x37 = x0*x7
                                x38 = x0*x9
                                x39 = x0*x11
                                x40 = x1*x8
                                x41 = x2*x7
                                x42 = x2*x9
                                x43 = x4*x7
                                x44 = 509004*x0 + x1*x27 - 1932084*x1 - 6286896*x10 + 5419386*x11 - 2598156*x12 - 5796252*x14 - 31434480*x15 - 18187092*x16 - 5796252*x17 - 31434480*x18 - 18187092*x19 + 4426569*x2 - 62868960*x20 - 54561276*x21 - 62868960*x22 - 90935460*x23 - 90935460*x24 - 54561276*x25 + x26*x8 + 1018008*x28 + x29 - 6286896*x3 + x30 + 17706276*x31 + 32516316*x32 + 4251528*x33 + 17706276*x34 + 32516316*x35 + 4251528*x36 + 26559414*x37 + 81290790*x38 + 14880348*x39 + 5419386*x4 + 108387720*x40 + 81290790*x41 + 37200870*x42 + 14880348*x43 - 2598156*x5 + 509004*x7 - 1932084*x8 + 4426569*x9 - 73744*xi[0] - 73744*xi[1] + 4480
                                x45 = 81*xi[0]/4480
                                x46 = 531441*x12
                                x47 = x46*xi[0]
                                x48 = 540918*x10
                                x49 = 363321*x8
                                x50 = 59049*x12
                                x51 = 26792*xi[1]
                                x52 = 133938*x7
                                x53 = 275562*x11
                                x54 = 578340*x9
                                x55 = x1*x10
                                x56 = x3*x8
                                x57 = 509004*x28 + 13279707*x37 + 54193860*x40 + 18600435*x42 + 2240
                                x58 = 81*xi[0]/1120
                                x59 = 1594323*x6
                                x60 = 531441*x16
                                x61 = 836832*x28 + 19308537*x37 + 73023930*x40 + 23914845*x42 + 4480
                                x62 = 9*xi[0]/160
                                x63 = 26190*x8
                                x64 = 7516*xi[1]
                                x65 = 4374*x10
                                x66 = 17010*x9
                                x67 = 19950*x7
                                x68 = 72171*x10
                                x69 = 163914*x28 + 3014415*x37 + 9415035*x40 + 2657205*x42 + 1120
                                x70 = 81*xi[0]/320
                                x71 = 98580*x28 + 1325889*x37 + 2886840*x40 + 531441*x42 + 896
                                x72 = 174282*x28 + 1511946*x37 + 1673055*x40 + 2240
                                x73 = 1240029*x25
                                x74 = 31428*x28 + 131544*x37 + 640
                                x75 = 531441*x5
                                x76 = x75*xi[1]
                                x77 = 548289*x1
                                x78 = 2112642*x3
                                x79 = 13068*xi[0]
                                x80 = x79*xi[1] + 560
                                x81 = 81*x28/4480
                                x82 = 127575*x2
                                x83 = 59049*x4
                                x84 = 14616*x0
                                x85 = 531441*x19
                                x86 = -15876*x28 - 80
                                x87 = 81*x28/1120
                                x88 = -22194*x28 - 492075*x37 - 80
                                x89 = 9*x28/160
                                x90 = -1296*x7
                                x91 = 531441*x24
                                x92 = -4950*x28 - 153090*x37 - 590490*x40 - 16
                                x93 = 81*x28/320
                                x94 = -1296*x0
                                x95 = 531441*x23
                                x96 = 127575*x9
                                x97 = 59049*x11
                                x98 = 14616*x7
                                x99 = 548289*x8
                                x100 = 2112642*x10
                                x101 = 81*xi[1]/4480
                                x102 = 1240029*x21
                                x103 = 81*xi[1]/1120
                                x104 = 1594323*x13
                                x105 = 9*xi[1]/160
                                x106 = 81*xi[1]/320
                                x107 = 26190*x1
                                x108 = 7516*xi[0]
                                x109 = 4374*x3
                                x110 = 17010*x2
                                x111 = 19950*x0
                                x112 = 72171*x3
                                x113 = 540918*x3
                                x114 = 363321*x1
                                x115 = 59049*x5
                                x116 = 26792*xi[0]
                                x117 = 133938*x0
                                x118 = 275562*x4
                                x119 = 578340*x2
                                x120 = 1089963*x14 - 4133430*x38 - 2240
                                x121 = 1089963*x17 - 4133430*x41
                                x122 = 729*x28/560
                                x123 = -1764*xi[0]*xi[1] - 80
                                x124 = 177147*x5
                                x125 = -19683*x11 + 177147*x16
                                x126 = -328092*x28 - 4749435*x37 - 8070030*x40
                                x127 = 243*x28/160
                                x128 = 1771470*x24
                                x129 = 177147*x21
                                x130 = 59049*x10
                                x131 = x129 - x130*xi[0] + x65
                                x132 = -1970730*x0*x7 - 3346110*x1*x8 - 141366*xi[0]*xi[1] - 1120
                                x133 = 243*x28/80
                                x134 = 354294*x25
                                x135 = 236196*x4
                                x136 = -39366*x0*x9 + 8019*x15 + 59049*x23 - 486*x9
                                x137 = -22170*x28 - 245916*x37 - 354294*x40 - 224
                                x138 = 729*x28/64
                                x139 = 531441*x25
                                x140 = -196830*x40
                                x141 = 177147*x24
                                x142 = x141 + 76545*x20 - 12150*x31 + 648*x8
                                x143 = x140 + x142
                                x144 = -240570*x0*x7 - 32106*xi[0]*xi[1] - 448
                                x145 = 177147*x25
                                x146 = 7398*x14 + x145 + 185895*x22 - 295245*x41 - 360*x7
                                x147 = -54675*x37
                                x148 = x147 - 14694*x28 - 320
                                x149 = 177147*x19 - 19683*x4
                                x150 = x147 - 8580*xi[0]*xi[1] - 40
                                x151 = 59049*x3
                                x152 = x109 + x145 - x151*xi[1]
                                x153 = -114210*x0*x7 + x140 - 4566*xi[0]*xi[1] - 16
                                x154 = 8019*x18 - 39366*x2*x7 - 486*x2 + 59049*x24
                                x155 = 177147*x23
                                x156 = 648*x1 + x155 + 76545*x22 - 12150*x34
                                x157 = -360*x0 + x129 + 7398*x17 + 185895*x20 - 295245*x38
                                x158 = 177147*x12
                                x159 = 531441*x21
                                x160 = x140 + x156
                                x161 = 354294*x21
                                x162 = 236196*x11
                                x163 = 1771470*x23
                                x164 = 729*x28/160
                                x165 = -14718*x28 - 177390*x37 - 64
                                x166 = -2125764*x41 - 224
                                x167 = -55986*x28 - 1083051*x37 - 2204496*x40
                                x168 = 243*x28/32
                                x169 = 24057*x15 + x155 - 118098*x38 - 1458*x9
                                x170 = -454167*x0*x7 - 846369*x1*x8 - 27237*xi[0]*xi[1] - 112
                                x171 = -9240*x28 - 245187*x37 - 629856*x40 - 32
                                x172 = x141 + 24057*x18 - 1458*x2 - 118098*x41
                                x173 = -2125764*x38
                                return jnp.stack([4782969*x0*x11/32 - 43046721*x0*x12/1120 + 1869885*x0*x7/64 + 13286025*x0*x9/64 + 58635*x0/224 + 4782969*x1*x10/16 - 14348907*x1*x11/160 + 4428675*x1*x8/16 + 623295*x1*xi[1]/32 - 40707*x1/28 - 43046721*x10*x2/320 + 2657205*x10*xi[0]/32 - 6589431*x10/640 + 885735*x11/64 + 4782969*x12*xi[0]/112 - 5137263*x12/448 - 43046721*x13*xi[0]/4480 + 4782969*x13/896 - 122121*x14/28 - 6589431*x15/128 - 5137263*x16/64 - 122121*x17/28 - 6589431*x18/128 - 5137263*x19/64 + 13286025*x2*x7/64 + 23914845*x2*x9/64 + 623295*x2/128 - 6589431*x20/64 - 15411789*x21/64 - 6589431*x22/64 - 25686315*x23/64 - 25686315*x24/64 - 15411789*x25/64 + 4782969*x3*x8/16 - 43046721*x3*x9/320 + 2657205*x3*xi[1]/32 - 6589431*x3/640 + 4782969*x4*x7/32 - 14348907*x4*x8/160 + 885735*x4/64 - 43046721*x5*x7/1120 + 4782969*x5*xi[1]/112 - 5137263*x5/448 - 43046721*x6*xi[1]/4480 + 4782969*x6/896 + 58635*x7/224 + 623295*x8*xi[0]/32 - 40707*x8/28 + 623295*x9/128 - 4782969*xi[0]**9/4480 + 58635*xi[0]*xi[1]/112 - 7129*xi[0]/280 - 4782969*xi[1]**9/4480 - 7129*xi[1]/280 + 1, xi[0]*(1063116*x0 - 5450004*x1 + 16365321*x2 - x26 + 32240754*x4 - 19131876*x5 + 4782969*x6 - 109584*xi[0] + 4480)/4480, xi[1]*(32240754*x11 - 19131876*x12 + 4782969*x13 - x27 + 1063116*x7 - 5450004*x8 + 16365321*x9 - 109584*xi[1] + 4480)/4480, x44*x45, x58*(16120377*x0*x10 + 26229420*x0*x8 + 3500847*x0*xi[1] - 375066*x0 + 36639540*x1*x7 + 39267585*x1*x9 + 1568763*x1 + 2893401*x11*xi[0] + 51667875*x2*x8 + 23524830*x2*xi[1] - 3848229*x2 - x29 + 38440899*x3*x7 + 5745978*x3 - 5583249*x31 - 6521634*x32 - 12123027*x34 - 25994682*x35 - 3720087*x36 - 28474740*x38 - 3720087*x39 + 15293691*x4*xi[1] - 5143824*x4 - 52816050*x41 - 11160261*x43 - x47 + x48 + x49 + 2539107*x5 + x50 + x51 - x52 - x53 - x54 - 11160261*x55 - 18600435*x56 - x57 + 2295405*x7*xi[0] + 7909650*x9*xi[0] + 46952*xi[0]), x62*(870828*x0 - 3900636*x1 - 170586*x10 + 39366*x11 - 2843424*x14 - 4953555*x15 - 6459750*x17 - 51799095*x18 - 37732311*x19 + 10113417*x2 - 29622915*x20 - 10097379*x21 - 60853275*x22 - 42515280*x23 - 79716150*x24 - 77058945*x25 - 15785766*x3 + 5053185*x31 + 2539107*x32 + 24672519*x34 + 60958251*x35 + 9565938*x36 + 24406920*x38 + 1594323*x39 + 14644152*x4 + 97430850*x41 + 23914845*x43 - 7440174*x5 + 9565938*x55 + 31886460*x56 + x59 - x60 + x61 + 147444*x7 - 284310*x8 + 303750*x9 - 100624*xi[0] - 40144*xi[1]), x70*(354294*x0*x10 + 3287790*x0*x8 + 1345716*x0*xi[1] - 234684*x0 + 10515825*x1*x7 + 3838185*x1*x9 + 1100241*x1 + 11809800*x2*x8 + 12356550*x2*xi[1] - 2978289*x2 - x29 + 15943230*x3*x7 + 4830354*x3 - 500175*x31 - 5509539*x34 - 15418350*x35 - 2657205*x36 - 1738665*x38 + 10038330*x4*xi[1] - 4632066*x4 - 18534825*x41 - 5314410*x43 + 2421009*x5 - 531441*x55 - 5314410*x56 + x63 + x64 + x65 - x66 - x67 - x68*xi[0] - x69 + 407745*x7*xi[0] + 302535*x9*xi[0] + 25996*xi[0]), x70*(196380*x0 - 949140*x1 - 170190*x14 - 36450*x15 - 840690*x17 - 8529300*x18 - 7676370*x19 + 2654613*x2 - 911250*x20 - 4957200*x22 - 590490*x23 - 4133430*x24 - 8857350*x25 + x29 - 4446900*x3 + 129276*x31 + 3608226*x34 + 11219310*x35 + 2125764*x36 + 229635*x38 + 4395870*x4 + 9480645*x41 + 3188646*x43 - 2361960*x5 + 2125764*x56 + 8040*x7 + x71 - 6480*x8 + 1944*x9 - 21200*xi[0] - 4400*xi[1]), x62*(492075*x0*x8 + 1525149*x0*xi[1] - 505842*x0 + 5937705*x1*x7 + 2497797*x1 + 2657205*x2*x8 + 16664940*x2*xi[1] - 7160967*x2 + 12223143*x3*x7 + 12321558*x3 - 66582*x31 - 6769251*x34 - 22950378*x35 - 4782969*x36 + 16474671*x4*xi[1] - 12518388*x4 - 12105045*x41 - 4782969*x43 + 6908733*x5 - 1594323*x56 - x59 + 187272*x7*xi[0] - 8640*x7 - x72 + 3240*x8 + 53672*xi[0] + 7640*xi[1]), x58*(147636*x0 - 740628*x1 - 15876*x14 - 280224*x17 - 3240405*x18 - 3483891*x19 + 2164239*x2 - 535815*x22 + x29 - 3806838*x3 + 1275183*x34 + 4638627*x35 + 1062882*x36 + 3962844*x4 + 1148175*x41 + 531441*x43 - 2243862*x5 + 720*x7 - x73 + x74 - 15472*xi[0] - 1360*xi[1]), x45*(118188*x0*xi[1] - 131256*x0 + 666477*x1 + 1428840*x2*xi[1] - 1977129*x2 - x29 + 3541482*x3 + 1653372*x4*xi[1] - 3766014*x4 + 2184813*x5 - x76 - x77*xi[1] - x78*xi[1] - x80 + 13628*xi[0] + 560*xi[1]), x81*(-118188*x0 - 1428840*x2 - 1653372*x4 + x75 + x77 + x78 + x79 - 560), x87*(59535*x1 + 131544*x17 + 1148175*x18 + 137781*x3 - 535815*x34 - 1240029*x35 - x82 - x83 - x84 + x85 + x86 + 1764*xi[0] + 720*xi[1]), x89*(-12150*x0 + 41310*x1 + 66582*x14 + 164025*x17 + 885735*x18 - 65610*x2 + 1673055*x22 + 1594323*x25 + 39366*x3 - 557685*x34 - 531441*x35 - 2657205*x41 - 3240*x7 + x88 + 1644*xi[0] + 1080*xi[1]), x93*(-1890*x0 + 4860*x1 + 24300*x14 + 31185*x17 + 72171*x18 - 4374*x2 + 229635*x20 + 393660*x22 - 36450*x31 - 80190*x34 - 354294*x41 + 1944*x8 + x90 + x91 + x92 + 300*xi[0] + 264*xi[1]), x93*(1944*x1 + 31185*x14 + 72171*x15 + 24300*x17 + 393660*x20 + 229635*x22 - 80190*x31 - 36450*x34 - 354294*x38 - 1890*x7 + 4860*x8 - 4374*x9 + x92 + x94 + x95 + 264*xi[0] + 300*xi[1]), x89*(-3240*x0 + 39366*x10 + 164025*x14 + 885735*x15 + 66582*x17 + 1673055*x20 + 1594323*x21 - 557685*x31 - 531441*x32 - 2657205*x38 - 12150*x7 + 41310*x8 + x88 - 65610*x9 + 1080*xi[0] + 1644*xi[1]), x87*(137781*x10 + 131544*x14 + 1148175*x15 - 535815*x31 - 1240029*x32 + x60 + 59535*x8 + x86 - x96 - x97 - x98 + 720*xi[0] + 1764*xi[1]), x81*(x100 - 1653372*x11 + x46 - 118188*x7 - 1428840*x9 + x99 + 13068*xi[1] - 560), x101*(3541482*x10 - x100*xi[0] + 1653372*x11*xi[0] - 3766014*x11 + 2184813*x12 - x30 - x47 + 118188*x7*xi[0] - 131256*x7 + 666477*x8 - x80 + 1428840*x9*xi[0] - 1977129*x9 - x99*xi[0] + 560*xi[0] + 13628*xi[1]), x103*(720*x0 - 3806838*x10 - x102 + 3962844*x11 - 2243862*x12 - 280224*x14 - 3240405*x15 - 3483891*x16 - 15876*x17 - 535815*x20 + x30 + 1275183*x31 + 4638627*x32 + 1062882*x33 + 1148175*x38 + 531441*x39 + 147636*x7 + x74 - 740628*x8 + 2164239*x9 - 1360*xi[0] - 15472*xi[1]), x105*(12223143*x0*x10 + 5937705*x0*x8 + 187272*x0*xi[1] - 8640*x0 + 492075*x1*x7 + 2657205*x1*x9 + 3240*x1 + 12321558*x10 - x104 + 16474671*x11*xi[0] - 12518388*x11 + 6908733*x12 - 6769251*x31 - 22950378*x32 - 4782969*x33 - 66582*x34 - 12105045*x38 - 4782969*x39 - 1594323*x55 + 1525149*x7*xi[0] - 505842*x7 - x72 + 2497797*x8 + 16664940*x9*xi[0] - 7160967*x9 + 7640*xi[0] + 53672*xi[1]), x106*(8040*x0 - 6480*x1 - 4446900*x10 + 4395870*x11 - 2361960*x12 - 840690*x14 - 8529300*x15 - 7676370*x16 - 170190*x17 - 36450*x18 + 1944*x2 - 4957200*x20 - 8857350*x21 - 911250*x22 - 4133430*x23 - 590490*x24 + x30 + 3608226*x31 + 11219310*x32 + 2125764*x33 + 129276*x34 + 9480645*x38 + 3188646*x39 + 229635*x41 + 2125764*x55 + 196380*x7 + x71 - 949140*x8 + 2654613*x9 - 4400*xi[0] - 21200*xi[1]), x106*(15943230*x0*x10 + 10515825*x0*x8 + 407745*x0*xi[1] + 3287790*x1*x7 + 11809800*x1*x9 + 4830354*x10 + x107 + x108 + x109 + 10038330*x11*xi[0] - 4632066*x11 - x110 - x111 - x112*xi[1] + 2421009*x12 + 3838185*x2*x8 + 302535*x2*xi[1] + 354294*x3*x7 - x30 - 5509539*x31 - 15418350*x32 - 2657205*x33 - 500175*x34 - 18534825*x38 - 5314410*x39 - 1738665*x41 - 5314410*x55 - 531441*x56 - x69 + 1345716*x7*xi[0] - 234684*x7 + 1100241*x8 + 12356550*x9*xi[0] - 2978289*x9 + 25996*xi[1]), x105*(147444*x0 - 284310*x1 - 15785766*x10 + x104 + 14644152*x11 - 7440174*x12 - 6459750*x14 - 51799095*x15 - 37732311*x16 - 2843424*x17 - 4953555*x18 + 303750*x2 - 60853275*x20 - 77058945*x21 - 29622915*x22 - 79716150*x23 - 42515280*x24 - 10097379*x25 - 170586*x3 + 24672519*x31 + 60958251*x32 + 9565938*x33 + 5053185*x34 + 2539107*x35 + 97430850*x38 + 23914845*x39 + 39366*x4 + 24406920*x41 + 1594323*x43 + 31886460*x55 + 9565938*x56 + x61 + 870828*x7 - 3900636*x8 - x85 + 10113417*x9 - 40144*xi[0] - 100624*xi[1]), x103*(38440899*x0*x10 + 36639540*x0*x8 + 2295405*x0*xi[1] + 26229420*x1*x7 + 51667875*x1*x9 + 5745978*x10 + 15293691*x11*xi[0] - 5143824*x11 + x113 + x114 + x115 + x116 - x117 - x118 - x119 + 2539107*x12 + 39267585*x2*x8 + 7909650*x2*xi[1] + 16120377*x3*x7 - x30 - 12123027*x31 - 25994682*x32 - 3720087*x33 - 5583249*x34 - 6521634*x35 - 52816050*x38 - 11160261*x39 + 2893401*x4*xi[1] - 28474740*x41 - 3720087*x43 - 18600435*x55 - 11160261*x56 - x57 + 3500847*x7*xi[0] - 375066*x7 - x76 + 1568763*x8 + 23524830*x9*xi[0] - 3848229*x9 + 46952*xi[1]), x101*x44, x122*(3470040*x0*x7 + 5511240*x1*x8 + 2313360*x1*xi[1] + 1653372*x10*xi[0] - x102 - x113 - x114 - x115 - x116 + x117 + x118 + x119 - x120 - x121 - 2704590*x15 - 413343*x16 - 2704590*x18 - 413343*x19 - 5409180*x20 - 5409180*x22 - 2066715*x23 - 2066715*x24 + 1653372*x3*xi[1] - x48 - x49 - x50 - x51 + x52 + x53 + x54 - x73 + 2313360*x8*xi[0] + 267876*xi[0]*xi[1]), x122*(16380*x0 + 59535*x1*xi[1] - 74151*x1 - x115 - x123 + 187110*x2 + 137781*x3*xi[1] - 265356*x3 + 196830*x4 - x82*xi[1] - x83*xi[1] - x84*xi[1] - 1844*xi[0] - 80*xi[1]), x122*(137781*x10*xi[0] - 265356*x10 + 196830*x11 - x123 - x50 + 16380*x7 + 59535*x8*xi[0] - 74151*x8 + 187110*x9 - x96*xi[0] - x97*xi[0] - x98*xi[0] - 80*xi[0] - 1844*xi[1]), x127*(-254370*x0 + 805653*x1 + 85293*x10 + x120 + x124 + x125 + x126 + 1793340*x15 + 1753461*x17 + 5893965*x18 + 1062882*x19 - 1431270*x2 + 6320430*x20 + 1062882*x21 + 9054180*x22 + 2657205*x23 + 3542940*x24 + 2657205*x25 + 1452168*x3 - 1886895*x31 - 885735*x32 - 4445685*x34 - 3956283*x35 - 787320*x4 - 7971615*x41 - 73722*x7 + 142155*x8 - 151875*x9 + 40232*xi[0] + 20072*xi[1]), x133*(984150*x0*x9 + 166776*x0 + 2744685*x1*xi[1] - 599913*x1 - x124 - x128 - x131 - x132 - 347895*x14 - 251505*x15 - 921618*x17 - 4122495*x18 - 885735*x19 + 4723920*x2*x7 + 1178550*x2 - 2022975*x20 - 4603635*x22 - 885735*x23 - 1771470*x25 + 3050865*x3*xi[1] - 1294704*x3 + 747954*x4 - x63 - x64 + x66 + x67 + 421605*x8*xi[0] - 22636*xi[0]), x138*(-38304*x0 + 151101*x1 + x115 + x134 + x135*xi[1] - x135 + x136 + x137 + 38025*x14 + 160290*x17 + 914166*x18 - 323676*x2 + 163296*x20 + 685989*x22 + 236196*x24 + 383454*x3 - 28674*x31 - 541404*x34 - 747954*x35 - 826686*x41 - 2010*x7 + 1620*x8 + 4796*xi[0] + 1100*xi[1]), x133*(83298*x0 + 908820*x1*xi[1] - 349623*x1 - x124 - x139 - 34344*x14 - x143 - x144 - 247239*x17 - 1697112*x18 + 1062882*x2*x7 + 802872*x2 - 754515*x22 + 1535274*x3*xi[1] - 1019142*x3 + 669222*x4 + 1728*x7 - x85 - 9928*xi[0] - 1528*xi[1]), x127*(-62934*x0 + 275913*x1 + x124 + x146 + x148 + 118071*x17 + 929475*x18 + 354294*x19 - 668250*x2 + 901044*x3 - 460485*x34 - 925101*x35 - 629856*x4 + 7256*xi[0] + 680*xi[1]), x127*(6897*x0 + 261225*x1*xi[1] - 26730*x1 - x146 - x149 - x150 - 68148*x17 - 513945*x18 + 53460*x2 + 492075*x3*xi[1] - 52488*x3 - 862*xi[0] - 400*xi[1]), x133*(2190*x0 + 95985*x1*xi[1] - 6750*x1 - 16848*x14 - x142 - x152 - x153 - 31455*x17 - 129033*x18 + 433026*x2*x7 + 9234*x2 - 338985*x22 + 864*x7 - 316*xi[0] - 232*xi[1]), x138*(36936*x0*x7 + 390*x0 + 137781*x1*x8 + 13851*x1*xi[1] - 810*x1 - x136 - 6759*x14 - x154 - 6759*x17 - 73629*x20 - 73629*x22 + 390*x7 + 13851*x8*xi[0] - 810*x8 + 1221*xi[0]*xi[1] - 70*xi[0] - 70*xi[1] + 4), x133*(433026*x0*x9 + 864*x0 - x131 - 31455*x14 - 129033*x15 - x153 - x156 - 16848*x17 - 338985*x20 + 2190*x7 + 95985*x8*xi[0] - 6750*x8 + 9234*x9 - 232*xi[0] - 316*xi[1]), x127*(492075*x10*xi[0] - 52488*x10 - x125 - 68148*x14 - 513945*x15 - x150 - x157 + 6897*x7 + 261225*x8*xi[0] - 26730*x8 + 53460*x9 - 400*xi[0] - 862*xi[1]), x127*(901044*x10 - 629856*x11 + 118071*x14 + x148 + 929475*x15 + x157 + x158 + 354294*x16 - 460485*x31 - 925101*x32 - 62934*x7 + 275913*x8 - 668250*x9 + 680*xi[0] + 7256*xi[1]), x133*(1062882*x0*x9 + 1728*x0 + 1535274*x10*xi[0] - 1019142*x10 + 669222*x11 - 247239*x14 - x144 - 1697112*x15 - x158 - x159 - x160 - 34344*x17 - 754515*x20 - x60 + 83298*x7 + 908820*x8*xi[0] - 349623*x8 + 802872*x9 - 1528*xi[0] - 9928*xi[1]), x138*(-2010*x0 + 1620*x1 + 383454*x10 + x137 + 160290*x14 + 914166*x15 + x154 + x161 + x162*xi[0] - x162 + 38025*x17 + 685989*x20 + 163296*x22 + 236196*x23 - 541404*x31 - 747954*x32 - 28674*x34 - 826686*x38 + x50 - 38304*x7 + 151101*x8 - 323676*x9 + 1100*xi[0] + 4796*xi[1]), x133*(4723920*x0*x9 + 421605*x1*xi[1] + 3050865*x10*xi[0] - 1294704*x10 - x107 - x108 + 747954*x11 + x110 + x111 - x132 - 921618*x14 - 4122495*x15 - x152 - x158 - 885735*x16 - x163 - 347895*x17 - 251505*x18 + 984150*x2*x7 - 4603635*x20 - 1771470*x21 - 2022975*x22 - 885735*x24 + 166776*x7 + 2744685*x8*xi[0] - 599913*x8 + 1178550*x9 - 22636*xi[1]), x127*(-73722*x0 + 142155*x1 + 1452168*x10 - 787320*x11 + x121 + x126 + 1753461*x14 + x149 + 5893965*x15 + x158 + 1062882*x16 + 1793340*x18 - 151875*x2 + 9054180*x20 + 2657205*x21 + 6320430*x22 + 3542940*x23 + 2657205*x24 + 1062882*x25 + 85293*x3 - 4445685*x31 - 3956283*x32 - 1886895*x34 - 885735*x35 - 7971615*x38 - 254370*x7 + 805653*x8 - 1431270*x9 + 20072*xi[0] + 40232*xi[1] - 2240), x164*(2374110*x0*x7 + 3050865*x0*x9 + 43797*x0 + 4527090*x1*x8 + 1313415*x1*xi[1] - 102870*x1 + 807003*x10*xi[0] - 78732*x10 - x125 - x128 - 613008*x14 - x149 - 1454355*x15 - x163 - 613008*x17 - 1454355*x18 + 3050865*x2*x7 + 126360*x2 - 3969405*x20 - 885735*x21 - 3969405*x22 - 885735*x25 + 807003*x3*xi[1] - 78732*x3 + 43797*x7 + 1313415*x8*xi[0] - 102870*x8 + 126360*x9 + 132954*xi[0]*xi[1] - 8798*xi[0] - 8798*xi[1] + 560), x164*(-10182*x0 + 36855*x1 + x134 + 25596*x14 + x143 + x149 + x151 + x165 + 110403*x17 + 686718*x18 - 67311*x2 + 546750*x22 - 390015*x34 - 570807*x35 - 747954*x41 + x90 + 1336*xi[0] + 712*xi[1]), x164*(x125 + x130 + 110403*x14 + 686718*x15 + x160 + x161 + x165 + 25596*x17 + 546750*x20 - 390015*x31 - 570807*x32 - 747954*x38 - 10182*x7 + 36855*x8 - 67311*x9 + x94 + 712*xi[0] + 1336*xi[1]), x168*(-25932*x0 + 73305*x1 + x112 + x131 + 201825*x14 + x149 + 220887*x15 + x166 + x167 + 337068*x17 + 1156923*x18 - 103761*x2 + 1441233*x20 + 2309472*x22 + 708588*x23 + 1062882*x24 + 708588*x25 - 311769*x31 - 890109*x34 - 728271*x35 - 846369*x38 - 11910*x7 + 19710*x8 - 15066*x9 + 4124*xi[0] + 3116*xi[1]), x168*(15807*x0 + 590976*x1*xi[1] - 51840*x1 - x139 - 72117*x14 - x149 - x169 - 187947*x17 - x170 - 901044*x18 + 1358127*x2*x7 + 84078*x2 - 400221*x20 - 1211598*x22 + 649539*x3*xi[1] - 65610*x3 + 3870*x7 + 71199*x8*xi[0] - 4050*x8 - x91 - 2230*xi[0] - 1390*xi[1]), x168*(-3750*x0 + 9990*x1 + 40851*x14 + x152 + x169 + 57321*x17 + x171 + 159651*x18 - 11178*x2 + 310554*x20 + 599238*x22 + 354294*x24 - 56376*x31 - 148959*x34 - 570807*x41 - 2250*x7 + 3240*x8 + 596*xi[0] + 500*xi[1]), x168*(-2250*x0 + 3240*x1 + x131 + 57321*x14 + 159651*x15 + 40851*x17 + x171 + x172 + 599238*x20 + 310554*x22 + 354294*x23 - 148959*x31 - 56376*x34 - 570807*x38 - 3750*x7 + 9990*x8 - 11178*x9 + 500*xi[0] + 596*xi[1]), x168*(1358127*x0*x9 + 3870*x0 + 71199*x1*xi[1] - 4050*x1 + 649539*x10*xi[0] - 65610*x10 - x125 - 187947*x14 - 901044*x15 - x159 - 72117*x17 - x170 - x172 - 1211598*x20 - 400221*x22 + 15807*x7 + 590976*x8*xi[0] - 51840*x8 + 84078*x9 - x95 - 1390*xi[0] - 2230*xi[1]), x168*(-11910*x0 + 19710*x1 + x125 + 337068*x14 + 1156923*x15 + x152 + x167 + 201825*x17 + x173 + 220887*x18 - 15066*x2 + 2309472*x20 + 708588*x21 + 1441233*x22 + 1062882*x23 + 708588*x24 - 890109*x31 - 728271*x32 - 311769*x34 - 846369*x41 + x68 - 25932*x7 + 73305*x8 - 103761*x9 + 3116*xi[0] + 4124*xi[1] - 224), 27*x28*(1614006*x0*x7 + 20250*x0 + 3897234*x1*x8 + 662661*x1*xi[1] - 42930*x1 + 177147*x10*xi[0] - 13122*x10 - x139 - 328617*x14 - 570807*x15 - x159 - x166 - 328617*x17 - x173 - 570807*x18 + 39366*x2 - 2899962*x20 - 2899962*x22 - 1594323*x23 - 1594323*x24 + 177147*x3*xi[1] - 13122*x3 + 20250*x7 + 662661*x8*xi[0] - 42930*x8 + 39366*x9 + 63180*xi[0]*xi[1] - 3788*xi[0] - 3788*xi[1])/8])

                        case 66:
                            def shape_functions(xi):
                                x0 = xi[0]*xi[1]
                                x1 = xi[0]**2
                                x2 = xi[0]**3
                                x3 = xi[0]**4
                                x4 = xi[0]**5
                                x5 = xi[0]**6
                                x6 = xi[0]**7
                                x7 = xi[0]**8
                                x8 = xi[0]**9
                                x9 = xi[1]**2
                                x10 = xi[1]**3
                                x11 = xi[1]**4
                                x12 = xi[1]**5
                                x13 = xi[1]**6
                                x14 = xi[1]**7
                                x15 = xi[1]**8
                                x16 = xi[1]**9
                                x17 = x9*xi[0]
                                x18 = x10*xi[0]
                                x19 = x11*xi[0]
                                x20 = x12*xi[0]
                                x21 = x13*xi[0]
                                x22 = x14*xi[0]
                                x23 = x15*xi[0]
                                x24 = x1*xi[1]
                                x25 = x2*xi[1]
                                x26 = x3*xi[1]
                                x27 = x4*xi[1]
                                x28 = x5*xi[1]
                                x29 = x6*xi[1]
                                x30 = x7*xi[1]
                                x31 = x1*x9
                                x32 = x1*x10
                                x33 = x1*x11
                                x34 = x1*x12
                                x35 = x1*x13
                                x36 = x1*x14
                                x37 = x2*x9
                                x38 = x10*x2
                                x39 = x11*x2
                                x40 = x12*x2
                                x41 = x13*x2
                                x42 = x3*x9
                                x43 = x10*x3
                                x44 = x11*x3
                                x45 = x12*x3
                                x46 = x4*x9
                                x47 = x10*x4
                                x48 = x11*x4
                                x49 = x5*x9
                                x50 = x10*x5
                                x51 = x6*x9
                                x52 = 1250000*x8
                                x53 = 1250000*x16
                                x54 = -1438434*x0 - 57087450*x31 - 422100000*x38 - 472500000*x44 - 4536
                                x55 = 316575000*x1*x11 + 189000000*x1*x13 + 719217*x1 + 378000000*x10*x4 + 38058300*x10*xi[0] - 3319705*x10 + 9514575*x11 + 378000000*x12*x2 + 126630000*x12*xi[0] - 17611125*x12 + 21105000*x13 + 54000000*x14*xi[0] - 15825000*x14 + 6750000*x15 - 9959115*x17 - 88055625*x19 + 38058300*x2*xi[1] - 3319705*x2 - 110775000*x21 - 11250000*x23 - 9959115*x24 - 88055625*x26 - 110775000*x28 + 316575000*x3*x9 + 9514575*x3 - 11250000*x30 - 176111250*x32 - 332325000*x34 - 45000000*x36 - 176111250*x37 - 553875000*x39 + 126630000*x4*xi[1] - 17611125*x4 - 105000000*x41 - 553875000*x43 - 157500000*x45 - 332325000*x46 - 157500000*x48 + 189000000*x5*x9 + 21105000*x5 - 105000000*x50 - 45000000*x51 - x52 - x53 - x54 + 54000000*x6*xi[1] - 15825000*x6 + 6750000*x7 + 719217*x9 - 87498*xi[0] - 87498*xi[1]
                                x56 = 25*xi[0]/1134
                                x57 = 2500000*x8
                                x58 = 2905000*x13
                                x59 = 2794225*x11
                                x60 = 395127*x9
                                x61 = 250000*x15
                                x62 = 64818*xi[1]
                                x63 = 1300000*x14
                                x64 = 1344070*x10
                                x65 = 3640000*x12
                                x66 = 25*xi[0]/504
                                x67 = 287875*x11
                                x68 = 122500*x13
                                x69 = 76489*x9
                                x70 = 16566*xi[1]
                                x71 = 25000*x14
                                x72 = 193060*x10
                                x73 = 253750*x12
                                x74 = 1250000*x14
                                x75 = -14239400*x1*x9 - 98087500*x10*x2 - 105000000*x11*x3 - 401468*xi[0]*xi[1] - 1512
                                x76 = 50*xi[0]/189
                                x77 = -485187*x0 - 14174475*x31 - 83212500*x38 - 78750000*x44 - 2268
                                x78 = 25*xi[0]/108
                                x79 = 15000*x12
                                x80 = 6250000*x40
                                x81 = -17250625*x1*x9 - 77812500*x10*x2 - 56250000*x11*x3 - 760125*xi[0]*xi[1] - 4536
                                x82 = 12930*x9
                                x83 = 3000*x11
                                x84 = 7242*xi[1]
                                x85 = 10200*x10
                                x86 = 3750000*x44
                                x87 = -191219*x0 - 3141075*x31 - 9475000*x38 - x86 - 1512
                                x88 = -614250*x1*x9 - 918750*x10*x2 - 58221*xi[0]*xi[1] - 648
                                x89 = 7000000*x49
                                x90 = -64593*x0 - 328300*x31 - 1134
                                x91 = 147655*x1
                                x92 = 1250000*x7
                                x93 = 2806125*x3
                                x94 = 6825000*x5
                                x95 = -13698*xi[0]*xi[1] - 504
                                x96 = 25*x0/1134
                                x97 = 805000*x4
                                x98 = 169225*x2
                                x99 = 3267*xi[0]
                                x100 = 32670*x0 + 126
                                x101 = 250000*x6
                                x102 = -x101 + 2500000*x29
                                x103 = 25*x0/504
                                x104 = 2187500*x42
                                x105 = 6615*x0 + 203000*x31 + 18
                                x106 = -375000*x28 + 1250000*x49 + 25000*x5
                                x107 = 50*x0/189
                                x108 = 22500*x3
                                x109 = 7535*x0 + 337500*x31 + 2125000*x38 + 18
                                x110 = 15000*x4
                                x111 = 2500000*x47
                                x112 = -x110 + x111 + 275000*x27 - 1500000*x46
                                x113 = 25*x0/108
                                x114 = 15000*x3
                                x115 = 15000*x11
                                x116 = 2187500*x33
                                x117 = 6250000*x44
                                x118 = 22500*x11
                                x119 = 2500000*x40
                                x120 = x119 + 275000*x20 - 1500000*x34 - x79
                                x121 = 25000*x13 - 375000*x21 + 1250000*x35
                                x122 = 805000*x12
                                x123 = 169225*x10
                                x124 = 250000*x14
                                x125 = -x124 + 2500000*x22
                                x126 = 147655*x9
                                x127 = 2806125*x11
                                x128 = 6825000*x13
                                x129 = 1250000*x15
                                x130 = 25*xi[1]/1134
                                x131 = 2500000*x16
                                x132 = 7000000*x35
                                x133 = 25*xi[1]/504
                                x134 = 50*xi[1]/189
                                x135 = 12930*x1
                                x136 = 3000*x3
                                x137 = 7242*xi[0]
                                x138 = 10200*x2
                                x139 = 25*xi[1]/108
                                x140 = 6250000*x47
                                x141 = 287875*x3
                                x142 = 122500*x5
                                x143 = 76489*x1
                                x144 = 16566*xi[0]
                                x145 = 25000*x6
                                x146 = 193060*x2
                                x147 = 253750*x4
                                x148 = 1250000*x6
                                x149 = 2905000*x5
                                x150 = 2794225*x3
                                x151 = 395127*x1
                                x152 = 250000*x7
                                x153 = 64818*xi[0]
                                x154 = 1300000*x6
                                x155 = 1344070*x2
                                x156 = 3640000*x4
                                x157 = 125*x0/126
                                x158 = x101*xi[1]
                                x159 = x99*xi[1] + 126
                                x160 = x124*xi[0]
                                x161 = x160 - x71
                                x162 = 318638*x0 + 7519050*x31 + 27825000*x38 + 8750000*x44 + 1512
                                x163 = 250*x0/63
                                x164 = 3750000*x49
                                x165 = 5000000*x47
                                x166 = 250000*x35
                                x167 = 5000*x13 + x166 - 75000*x21
                                x168 = 140419*x0 + 3202975*x31 + 11825000*x38 + x86 + 756
                                x169 = 250*x0/27
                                x170 = 12500000*x47
                                x171 = -750000*x1*x12 - 7500*x12 + 137500*x20 + 1250000*x40
                                x172 = 343455*x0 + x117 + 6549375*x31 + 21312500*x38 + 2268
                                x173 = 25*x0/9
                                x174 = -9250000*x43
                                x175 = 437500*x33
                                x176 = 1250000*x44
                                x177 = -62500*x19 - 1250000*x39 + x83
                                x178 = x175 + x176 + x177
                                x179 = 176735*x0 + 2541125*x31 + 6000000*x38 + 1512
                                x180 = -56250*x1*x10 - 375000*x10*x3 - 300*x10 + 6850*x18 + 250000*x47
                                x181 = 212500*x38
                                x182 = 18197*x0 + x181 + 172425*x31 + 216
                                x183 = 437500*x42
                                x184 = 1487500*x4
                                x185 = 500000*x29
                                x186 = 250000*x49
                                x187 = -4410*x17 + x186 - 183750*x37 - 525000*x46 + 180*x9
                                x188 = 40600*x31
                                x189 = 8739*x0 + x188 + 162
                                x190 = x183 + 18
                                x191 = -x145 + x158
                                x192 = 5031*x0 + x188
                                x193 = x186 - 75000*x28 + 5000*x5
                                x194 = 2282*x0 + x181 + 79975*x31 + 6
                                x195 = x175 + 18
                                x196 = 7580*x0 + x176 + 348125*x31 + 2437500*x38
                                x197 = 137500*x27 - 750000*x4*x9 - 7500*x4 + 1250000*x47
                                x198 = x136 - 62500*x26 - 1250000*x43
                                x199 = -375000*x11*x2 - 56250*x2*x9 - 300*x2 + 6850*x25 + 250000*x40
                                x200 = 180*x1 + x166 - 4410*x24 - 183750*x32 - 525000*x34
                                x201 = 1487500*x12
                                x202 = 500000*x22
                                x203 = -9250000*x39
                                x204 = 5000000*x40
                                x205 = x176 + x183 + x198
                                x206 = 12500000*x40
                                x207 = 3750000*x35
                                x208 = x185 - 50000*x6
                                x209 = -50000*x14 + x202
                                x210 = 10000000*x44 + 756
                                x211 = 125*x0/18
                                x212 = 1000000*x5
                                x213 = 16193*x0 + 252400*x31 + 425000*x38 + 54
                                x214 = 1000000*x13
                                x215 = x148*xi[1] - 125000*x6
                                x216 = 260425*x0 + 8066125*x31 + 35437500*x38 + 12500000*x44 + 756
                                x217 = 50*x0/9
                                x218 = 254125*x0 + x210 + 7080500*x31 + 29175000*x38
                                x219 = 125*x0/36
                                x220 = 3750000*x47
                                x221 = 69005*x0 + 1542125*x31 + 4812500*x38 + 216
                                x222 = 21080*x0 + 811125*x31 + 3625000*x38 + 54
                                x223 = 3750000*x40
                                x224 = -125000*x14 + x74*xi[0]
                                x225 = 42610*x0 + 1715750*x31 + 9200000*x38 + x86 + 108
                                return jnp.stack([177133*x0/252 + 7812500*x1*x15/63 + 177133*x1/504 + 62500000*x10*x6/189 - 10511875*x10/4536 + 15625000*x11*x5/27 + 42711625*x11/4536 + 6250000*x12*x4/9 - 5369375*x12/216 + 15625000*x13*x3/27 + 4695625*x13/108 + 62500000*x14*x2/189 - 9453125*x14/189 + 6875000*x15/189 + 15625000*x16*xi[0]/567 - 8593750*x16/567 - 10511875*x17/1512 + 42711625*x18/1134 - 26846875*x19/216 - 10511875*x2/4536 + 4695625*x20/18 - 9453125*x21/27 + 55000000*x22/189 - 8593750*x23/63 - 10511875*x24/1512 + 42711625*x25/1134 - 26846875*x26/216 + 4695625*x27/18 - 9453125*x28/27 + 55000000*x29/189 + 42711625*x3/4536 - 8593750*x30/63 + 42711625*x31/756 - 26846875*x32/108 + 23478125*x33/36 - 9453125*x34/9 + 27500000*x35/27 - 34375000*x36/63 - 26846875*x37/108 + 23478125*x38/27 - 47265625*x39/27 - 5369375*x4/216 + 55000000*x40/27 - 34375000*x41/27 + 23478125*x42/36 - 47265625*x43/27 + 68750000*x44/27 - 17187500*x45/9 - 9453125*x46/9 + 55000000*x47/27 - 17187500*x48/9 + 27500000*x49/27 + 4695625*x5/108 - 34375000*x50/27 - 34375000*x51/63 - 9453125*x6/189 + 7812500*x7*x9/63 + 6875000*x7/189 + 15625000*x8*xi[1]/567 - 8593750*x8/567 + 177133*x9/504 + 1562500*xi[0]**10/567 - 7381*xi[0]/252 + 1562500*xi[1]**10/567 - 7381*xi[1]/252 + 1, xi[0]*(-1465875*x1 + 9046000*x2 - 33665625*x3 + 79091250*x4 - 118125000*x5 + 108750000*x6 - 56250000*x7 + 12500000*x8 + 128322*xi[0] - 4536)/4536, xi[1]*(9046000*x10 - 33665625*x11 + 79091250*x12 - 118125000*x13 + 108750000*x14 - 56250000*x15 + 12500000*x16 - 1465875*x9 + 128322*xi[1] - 4536)/4536, x55*x56, x66*(-1043307*x1 + 7983480*x17 - 24617600*x18 + 46142250*x19 + 5295340*x2 - 53830000*x20 + 38150000*x21 - 15000000*x22 + 2500000*x23 + 11934750*x24 - 51499000*x25 + 129969000*x26 - 199430000*x27 + 183400000*x28 - 93000000*x29 - 16234925*x3 + 20000000*x30 + 148169000*x32 - 225575000*x33 + 201600000*x34 - 98000000*x35 + 20000000*x36 + 204053500*x37 + 481250000*x39 + 31582250*x4 - 287000000*x40 + 70000000*x41 - 407575000*x42 + 626500000*x43 + 140000000*x45 + 463050000*x46 - 469000000*x47 + 175000000*x48 - 280000000*x49 - 39305000*x5 + 140000000*x50 + 70000000*x51 + x54 + x57 - x58 - x59 + 30350000*x6 - x60 - x61 + x62 + x63 + x64 + x65 - 13250000*x7 + 110178*xi[0]), x76*(35262500*x1*x11 + 8750000*x1*x13 - x1*x74 + 400579*x1 + 135625000*x10*x4 + 4047400*x10*xi[0] + 44625000*x12*x2 + 4541250*x12*xi[0] + 375000*x14*xi[0] - 1726515*x17 - 5586875*x19 + 17488100*x2*xi[1] - 2168695*x2 - 2012500*x21 - 3702150*x24 - 47500250*x26 - 74637500*x28 + 126262500*x3*x9 + 7008225*x3 - 8750000*x30 - 29463000*x32 - 24237500*x34 - 57405250*x37 - 91875000*x39 + 77341250*x4*xi[1] - 14224875*x4 - 8750000*x41 - 164500000*x43 - 26250000*x45 - 154962500*x46 - 43750000*x48 + 99750000*x5*x9 + 18322500*x5 - 43750000*x50 - 26250000*x51 - x52 + 39375000*x6*xi[1] - 14550000*x6 + x67 + x68 + x69 + 6500000*x7 - x70 - x71 - x72 - x73 - x75 - 39246*xi[0]), x78*(-645201*x1 + 122625*x10 - 125250*x11 + 67500*x12 - 15000*x13 + 1592285*x17 - 2749125*x18 + 2633750*x19 + 3642935*x2 - 1327500*x20 + 275000*x21 + 4734595*x24 - 23799075*x25 + 68572000*x26 - 117690000*x27 + 118900000*x28 - 65250000*x29 - 12248475*x3 + 15000000*x30 + 22122500*x32 - 18937500*x33 + 8400000*x34 - 1500000*x35 + 62371000*x37 + 58750000*x39 + 25757250*x4 - 20250000*x40 + 2500000*x41 - 149062500*x42 + 156500000*x43 + 15000000*x45 + 196875000*x46 - 142500000*x47 + 37500000*x48 - 135000000*x49 - 34215000*x5 + 50000000*x50 + 37500000*x51 + x57 + 27900000*x6 - 12750000*x7 + x77 - 66786*x9 + 60759*xi[0] + 19179*xi[1]), xi[0]*(10312500*x1*x11 + 1346625*x1 + 162500000*x10*x4 + 2224375*x10*xi[0] - 95250*x10 + 60000*x11 + 312500*x12*xi[0] - 1848250*x17 - 1325000*x19 + 40270625*x2*xi[1] - 7818625*x2 - 7681625*x24 - 121449375*x26 - 230000000*x28 + 205937500*x3*x9 + 27074375*x3 - 19040625*x32 - 2187500*x34 - 80571875*x37 - 35937500*x39 + 218125000*x4*xi[1] - 58608125*x4 - 161562500*x43 - 6250000*x45 - 290937500*x46 - 31250000*x48 + 212500000*x5*x9 + 80000000*x5 - 62500000*x50 - 62500000*x51 + 131250000*x6*xi[1] - 66875000*x6 - 31250000*x7*xi[1] + 31250000*x7 - x79 - 6250000*x8 + x80 - x81 + 75000*x9 - 123786*xi[0] - 29286*xi[1])/9, x78*(-461789*x1 + 325835*x17 - 244900*x18 + 68500*x19 + 2734310*x2 + 1978945*x24 - 10689200*x25 + 33381500*x26 - 62285000*x27 + 68300000*x28 - 40500000*x29 - 9680025*x3 + 10000000*x30 + 2186500*x32 - 562500*x33 + 15307250*x37 + 2125000*x39 + 21452250*x4 - 41212500*x42 + 21250000*x43 + 61775000*x46 - 23500000*x47 + 2500000*x48 - 48000000*x49 - 29985000*x5 + 10000000*x50 + 15000000*x51 + x57 + 25650000*x6 - 12250000*x7 - x82 - x83 + x84 + x85 + x87 + 41766*xi[0]), x76*(201951*x1 + 2625000*x10*x4 + 22050*x10*xi[0] - 900*x10 - 62235*x17 + 3385725*x2*xi[1] - 1213195*x2 - 613030*x24 - 10864000*x26 - 23762500*x28 + 8662500*x3*x9 + 4368525*x3 - 3750000*x30 - 203000*x32 - 3089625*x37 + 20921250*x4*xi[1] - 9867375*x4 - 2187500*x43 - 13650000*x46 + 11250000*x5*x9 + 14077500*x5 - 1250000*x50 - 3750000*x51 - x52 + 14625000*x6*xi[1] - 12300000*x6 + 6000000*x7 - x88 + 2430*x9 - 18054*xi[0] - 2178*xi[1]), x66*(-358803*x1 + 32670*x17 + 2179465*x2 + 689110*x24 - 3871875*x25 + 12694500*x26 - 25095000*x27 + 29400000*x28 - 18750000*x29 - 7953575*x3 + 5000000*x30 + 1692250*x37 + 18247250*x4 - 4900000*x42 + 8050000*x46 - 26495000*x5 + 2500000*x51 + x57 + 23600000*x6 - 11750000*x7 - x89 - 1260*x9 + x90 + 31797*xi[0] + 2394*xi[1]), x56*(161353*x1 + 841050*x2*xi[1] - 988705*x2 + 3647175*x3 + 5670000*x4*xi[1] - 8476125*x4 + 12495000*x5 - x52 + 4500000*x6*xi[1] - 11325000*x6 + 5750000*x7 - x91*xi[1] - x92*xi[1] - x93*xi[1] - x94*xi[1] - x95 - 14202*xi[0] - 504*xi[1]), x96*(-841050*x2 - 5670000*x4 - 4500000*x6 + x91 + x92 + x93 + x94 - 13698*xi[0] + 504), x103*(32830*x1 + x100 + x102 - 328300*x24 + 1692250*x25 - 4900000*x26 + 8050000*x27 - 7000000*x28 + 490000*x3 + 700000*x5 - x97 - x98 - x99 - 1260*xi[1]), x107*(4060*x1 + x104 + x105 + x106 - 22050*x17 - 18375*x2 - 60900*x24 + 275625*x25 - 656250*x26 + 787500*x27 + 43750*x3 - 918750*x37 - 52500*x4 - 2625000*x46 + 900*x9 - 441*xi[0] - 270*xi[1]), x113*(3375*x1 - 3000*x10 + x108 + x109 + x112 - 41100*x17 + 68500*x18 - 12750*x2 - 61875*x24 + 233750*x25 - 412500*x26 - 562500*x32 - 1275000*x37 + 2250000*x42 - 3750000*x43 + 1800*x9 - 411*xi[0] - 330*xi[1]), x0*(15625*x0 + 5250*x1 - 15000*x10 + x104 + x114 + x115 + x116 + x117 - 109375*x17 + 312500*x18 - 312500*x19 - 15000*x2 - 109375*x24 + 312500*x25 - 312500*x26 + 765625*x31 - 2187500*x32 - 2187500*x37 + 6250000*x38 - 6250000*x39 - 6250000*x43 + 5250*x9 - 750*xi[0] - 750*xi[1] + 36)/9, x113*(1800*x1 - 12750*x10 + x109 + x118 + x120 - 61875*x17 + 233750*x18 - 412500*x19 - 3000*x2 - 41100*x24 + 68500*x25 - 1275000*x32 + 2250000*x33 - 562500*x37 - 3750000*x39 + 3375*x9 - 330*xi[0] - 411*xi[1]), x107*(900*x1 - 18375*x10 + x105 + 43750*x11 + x116 - 52500*x12 + x121 - 60900*x17 + 275625*x18 - 656250*x19 + 787500*x20 - 22050*x24 - 918750*x32 - 2625000*x34 + 4060*x9 - 270*xi[0] - 441*xi[1]), x103*(x100 + 490000*x11 - x122 - x123 + x125 + 700000*x13 - 328300*x17 + 1692250*x18 - 4900000*x19 + 8050000*x20 - 7000000*x21 + 32830*x9 - 1260*xi[0] - 3267*xi[1]), x96*(-841050*x10 - 5670000*x12 + x126 + x127 + x128 + x129 - 4500000*x14 - 13698*xi[1] + 504), x130*(841050*x10*xi[0] - 988705*x10 + 3647175*x11 + 5670000*x12*xi[0] - 8476125*x12 - x126*xi[0] - x127*xi[0] - x128*xi[0] - x129*xi[0] + 12495000*x13 + 4500000*x14*xi[0] - 11325000*x14 + 5750000*x15 - x53 + 161353*x9 - x95 - 504*xi[0] - 14202*xi[1]), x133*(-1260*x1 + 2179465*x10 - 7953575*x11 + 18247250*x12 - 26495000*x13 + x131 - x132 + 23600000*x14 - 11750000*x15 + 689110*x17 - 3871875*x18 + 12694500*x19 - 25095000*x20 + 29400000*x21 - 18750000*x22 + 5000000*x23 + 32670*x24 + 1692250*x32 - 4900000*x33 + 8050000*x34 + 2500000*x36 - 358803*x9 + x90 + 2394*xi[0] + 31797*xi[1]), x134*(8662500*x1*x11 + 11250000*x1*x13 + 2430*x1 + 3385725*x10*xi[0] - 1213195*x10 + 4368525*x11 + 2625000*x12*x2 + 20921250*x12*xi[0] - 9867375*x12 + 14077500*x13 + 14625000*x14*xi[0] - 12300000*x14 + 6000000*x15 - 613030*x17 - 10864000*x19 + 22050*x2*xi[1] - 900*x2 - 23762500*x21 - 3750000*x23 - 62235*x24 - 3089625*x32 - 13650000*x34 - 3750000*x36 - 203000*x37 - 2187500*x39 - 1250000*x41 - x53 - x88 + 201951*x9 - 2178*xi[0] - 18054*xi[1]), x139*(2734310*x10 - 9680025*x11 + 21452250*x12 - 29985000*x13 + x131 - x135 - x136 + x137 + x138 + 25650000*x14 - 12250000*x15 + 1978945*x17 - 10689200*x18 + 33381500*x19 - 62285000*x20 + 68300000*x21 - 40500000*x22 + 10000000*x23 + 325835*x24 - 244900*x25 + 68500*x26 + 15307250*x32 - 41212500*x33 + 61775000*x34 - 48000000*x35 + 15000000*x36 + 2186500*x37 + 21250000*x39 - 23500000*x40 + 10000000*x41 - 562500*x42 + 2125000*x43 + 2500000*x45 + x87 - 461789*x9 + 41766*xi[1]), xi[1]*(205937500*x1*x11 + 212500000*x1*x13 + 75000*x1 + 40270625*x10*xi[0] - 7818625*x10 + 27074375*x11 - x110 + 162500000*x12*x2 + 218125000*x12*xi[0] - 58608125*x12 + 80000000*x13 + 131250000*x14*xi[0] - 66875000*x14 + x140 - 31250000*x15*xi[0] + 31250000*x15 - 6250000*x16 - 7681625*x17 - 121449375*x19 + 2224375*x2*xi[1] - 95250*x2 - 230000000*x21 - 1848250*x24 - 1325000*x26 + 10312500*x3*x9 + 60000*x3 - 80571875*x32 - 290937500*x34 - 62500000*x36 - 19040625*x37 - 161562500*x39 + 312500*x4*xi[1] - 62500000*x41 - 35937500*x43 - 31250000*x45 - 2187500*x46 - 6250000*x48 - x81 + 1346625*x9 - 29286*xi[0] - 123786*xi[1])/9, x139*(-66786*x1 + 3642935*x10 - 12248475*x11 + 25757250*x12 - 34215000*x13 + x131 + 27900000*x14 - 12750000*x15 + 4734595*x17 - 23799075*x18 + 68572000*x19 + 122625*x2 - 117690000*x20 + 118900000*x21 - 65250000*x22 + 15000000*x23 + 1592285*x24 - 2749125*x25 + 2633750*x26 - 1327500*x27 + 275000*x28 - 125250*x3 + 62371000*x32 - 149062500*x33 + 196875000*x34 - 135000000*x35 + 37500000*x36 + 22122500*x37 + 156500000*x39 + 67500*x4 - 142500000*x40 + 50000000*x41 - 18937500*x42 + 58750000*x43 + 37500000*x45 + 8400000*x46 - 20250000*x47 + 15000000*x48 - 1500000*x49 - 15000*x5 + 2500000*x50 + x77 - 645201*x9 + 19179*xi[0] + 60759*xi[1]), x134*(126262500*x1*x11 + 99750000*x1*x13 + 44625000*x10*x4 + 17488100*x10*xi[0] - 2168695*x10 + 7008225*x11 + 135625000*x12*x2 + 77341250*x12*xi[0] - 14224875*x12 + 18322500*x13 + 39375000*x14*xi[0] - 14550000*x14 + x141 + x142 + x143 - x144 - x145 - x146 - x147 - x148*x9 + 6500000*x15 - 3702150*x17 - 47500250*x19 + 4047400*x2*xi[1] - 74637500*x21 - 8750000*x23 - 1726515*x24 - 5586875*x26 - 2012500*x28 + 35262500*x3*x9 - 57405250*x32 - 154962500*x34 - 26250000*x36 - 29463000*x37 - 164500000*x39 + 4541250*x4*xi[1] - 43750000*x41 - 91875000*x43 - 43750000*x45 - 24237500*x46 - 26250000*x48 + 8750000*x5*x9 - 8750000*x50 - x53 + 375000*x6*xi[1] - x75 + 400579*x9 - 39246*xi[1]), x133*(5295340*x10 - 16234925*x11 + 31582250*x12 - 39305000*x13 + x131 + 30350000*x14 - x149 - 13250000*x15 - x150 - x151 - x152 + x153 + x154 + x155 + x156 + 11934750*x17 - 51499000*x18 + 129969000*x19 - 199430000*x20 + 183400000*x21 - 93000000*x22 + 20000000*x23 + 7983480*x24 - 24617600*x25 + 46142250*x26 - 53830000*x27 + 38150000*x28 - 15000000*x29 + 2500000*x30 + 204053500*x32 - 407575000*x33 + 463050000*x34 - 280000000*x35 + 70000000*x36 + 148169000*x37 + 626500000*x39 - 469000000*x40 + 140000000*x41 - 225575000*x42 + 481250000*x43 + 175000000*x45 + 201600000*x46 - 287000000*x47 + 140000000*x48 - 98000000*x49 + 70000000*x50 + 20000000*x51 + x54 - 1043307*x9 + 110178*xi[1]), x130*x55, x157*(790254*x0 + x132 + x149 + x150 + x151 + x152 - x153 - x154 - x155 - x156 - 4032210*x17 + 11176900*x18 - 18200000*x19 + 17430000*x20 - 9100000*x21 + 2000000*x22 - 4032210*x24 + 11176900*x25 - 18200000*x26 + 17430000*x27 - 9100000*x28 + 2000000*x29 + 16765350*x31 - 36400000*x32 + 43575000*x33 - 27300000*x34 - 36400000*x37 + 58100000*x38 - 45500000*x39 + 14000000*x40 + 43575000*x42 - 45500000*x43 + 17500000*x44 - 27300000*x46 + 14000000*x47 + x58 + x59 + x60 + x61 - x62 - x63 - x64 - x65 + x89 + 4536), x157*(32830*x1*xi[1] - 36097*x1 - x152 - x158 - x159 + 202055*x2 + 490000*x3*xi[1] - 659225*x3 + 1295000*x4 + 700000*x5*xi[1] - 1505000*x5 + 950000*x6 - x97*xi[1] - x98*xi[1] + 3393*xi[0] + 126*xi[1]), x157*(202055*x10 + 490000*x11*xi[0] - 659225*x11 + 1295000*x12 - x122*xi[0] - x123*xi[0] + 700000*x13*xi[0] - 1505000*x13 + 950000*x14 - x159 - x160 - x61 + 32830*x9*xi[0] - 36097*x9 + 126*xi[0] + 3393*xi[1]), x163*(14052500*x1*x10 + 7875000*x1*x12 + 2108960*x1*xi[1] - 242149*x1 + 25375000*x10*x3 + 19250000*x11*x2 + 4147500*x11*xi[0] + 1400000*x13*xi[0] - x152 - x161 - x162 - 3082100*x18 + 19810000*x2*x9 + 957950*x2 - 3272500*x20 - 6943300*x25 - 13422500*x27 - 1750000*x29 + 12783750*x3*xi[1] - 2218475*x3 - 14525000*x33 - 1750000*x35 + 18900000*x4*x9 + 3132500*x4 - 5250000*x40 - 27212500*x42 - 8750000*x47 - 5250000*x49 + 7525000*x5*xi[1] - 2660000*x5 + 1250000*x6 - x67 - x68 - x69 + x70 + x72 + x73 + 1344070*x9*xi[0] + 31686*xi[0]), x169*(155957*x1 - 40875*x10 + 41750*x11 - 22500*x12 + x152 + x164 + x165 + x167 + x168 - 456555*x17 + 780125*x18 - 738750*x19 - 694455*x2 + 367500*x20 - 1110135*x24 + 4232575*x25 - 8748750*x26 + 10067500*x27 - 6075000*x28 + 1500000*x29 + 1767975*x3 - 4773750*x32 + 3850000*x33 - 1575000*x34 - 10113750*x37 - 6750000*x39 - 2692500*x4 + 1500000*x40 + 15975000*x42 - 12750000*x43 - 12375000*x46 + 2430000*x5 - 1200000*x6 + 22262*x9 - 17733*xi[0] - 6393*xi[1]), x173*(7037500*x1*x10 + 2982175*x1*xi[1] - 532755*x1 + 27500000*x10*x3 + 47625*x10 + 8750000*x11*x2 + 587500*x11*xi[0] - 30000*x11 - x170 - x171 - x172 - 993125*x18 + 23912500*x2*x9 + 2577425*x2 - 12679875*x25 - 36500000*x27 - 6250000*x29 + 29025000*x3*xi[1] - 7093625*x3 - 3687500*x33 + 37500000*x4*x9 + 11570000*x4 - 43187500*x42 - 12500000*x49 + 23750000*x5*xi[1] - 11075000*x5 + 5750000*x6 + 830375*x9*xi[0] - 37500*x9 - x92 + 56223*xi[0] + 14643*xi[1]), x173*(384305*x1 + x165 - 299975*x17 + x174 + x178 + x179 + 224500*x18 - 1965700*x2 - 1625475*x24 + 7438250*x25 - 18505000*x26 + 25275000*x27 - 17750000*x28 + 5000000*x29 + 5748625*x3 - 1737500*x32 - 10225000*x37 - 9955000*x4 + 20762500*x42 - 20250000*x46 + 7500000*x49 + 10075000*x5 - 5500000*x6 + x82 - x84 - x85 + x92 - 38742*xi[0]), x169*(174015*x1*xi[1] - 57887*x1 - x152 - x180 - x182 + 742500*x2*x9 + 307920*x2 - 838550*x25 - 3267500*x27 - 750000*x29 + 2223750*x3*xi[1] - 942975*x3 + 1800000*x4*x9 + 1717500*x4 - 1650000*x42 - 750000*x49 + 2475000*x5*xi[1] - 1830000*x5 + 1050000*x6 + 19395*x9*xi[0] - 810*x9 + 5658*xi[0] + 726*xi[1]), x163*(45099*x1 + x152 + x183 - x184 + x185 + x187 + x189 - 246925*x2 - 85960*x24 + 430325*x25 - 1198750*x26 + 1872500*x27 - 1525000*x28 + 783475*x3 + 1660000*x5 - 1000000*x6 - 4311*xi[0] - 342*xi[1]), x163*(49070*x1*xi[1] - 4501*x1 - x187 - x190 - x191 - x192 + 22435*x2 - 242725*x25 - 1015000*x27 + 665000*x3*xi[1] - 62125*x3 + 96250*x4 + 800000*x5*xi[1] - 77500*x5 + 459*xi[0] + 198*xi[1]), x169*(20055*x1*xi[1] - 1262*x1 - x180 - x193 - x194 + 332500*x2*x9 + 5375*x2 - 84875*x25 - 192500*x27 + 183750*x3*xi[1] - 11750*x3 + 700000*x4*x9 + 12500*x4 - 700000*x42 + 9205*x9*xi[0] - 390*x9 + 143*xi[0] + 96*xi[1]), x173*(762500*x1*x10 + 57625*x1*xi[1] - 3000*x1 + 3250000*x10*x3 + 4800*x10 - x114 - x177 - 103000*x18 - x195 - x196 - x197 + 1150000*x2*x9 + 10125*x2 - 193125*x25 + 282500*x3*xi[1] - 1637500*x42 + 46175*x9*xi[0] - 2130*x9 + 393*xi[0] + 348*xi[1]), x173*(1150000*x1*x10 + 46175*x1*xi[1] - 2130*x1 + 10125*x10 + 3250000*x11*x2 + 282500*x11*xi[0] - x115 - x171 - 193125*x18 - x190 - x196 - x198 + 762500*x2*x9 + 4800*x2 - 103000*x25 - 1637500*x33 + 57625*x9*xi[0] - 3000*x9 + 348*xi[0] + 393*xi[1]), x169*(332500*x1*x10 + 700000*x1*x12 + 9205*x1*xi[1] - 390*x1 + 5375*x10 + 183750*x11*xi[0] - 11750*x11 + 12500*x12 - x167 - 84875*x18 - x194 - x199 - 192500*x20 - 700000*x33 + 20055*x9*xi[0] - 1262*x9 + 96*xi[0] + 143*xi[1]), x163*(22435*x10 + 665000*x11*xi[0] - 62125*x11 + 96250*x12 + 800000*x13*xi[0] - 77500*x13 - x161 - 242725*x18 - x192 - x195 - 1015000*x20 - x200 + 49070*x9*xi[0] - 4501*x9 + 198*xi[0] + 459*xi[1]), x163*(-246925*x10 + 783475*x11 + 1660000*x13 - 1000000*x14 - 85960*x17 + x175 + 430325*x18 + x189 - 1198750*x19 + 1872500*x20 + x200 - x201 + x202 - 1525000*x21 + x61 + 45099*x9 - 342*xi[0] - 4311*xi[1]), x169*(742500*x1*x10 + 1800000*x1*x12 + 19395*x1*xi[1] - 810*x1 + 307920*x10 + 2223750*x11*xi[0] - 942975*x11 + 1717500*x12 + 2475000*x13*xi[0] - 1830000*x13 + 1050000*x14 - 838550*x18 - x182 - x199 - 3267500*x20 - 750000*x22 - 1650000*x33 - 750000*x35 - x61 + 174015*x9*xi[0] - 57887*x9 + 726*xi[0] + 5658*xi[1]), x173*(-1965700*x10 + 5748625*x11 - 9955000*x12 + x129 + 10075000*x13 + x135 - x137 - x138 - 5500000*x14 - 1625475*x17 + x179 + 7438250*x18 - 18505000*x19 + 25275000*x20 + x203 + x204 + x205 - 17750000*x21 + 5000000*x22 - 299975*x24 + 224500*x25 - 10225000*x32 + 20762500*x33 - 20250000*x34 + 7500000*x35 - 1737500*x37 + 384305*x9 - 38742*xi[1]), x173*(23912500*x1*x10 + 37500000*x1*x12 + 830375*x1*xi[1] - 37500*x1 + 8750000*x10*x3 + 2577425*x10 + 27500000*x11*x2 + 29025000*x11*xi[0] - 7093625*x11 + 11570000*x12 - x129 + 23750000*x13*xi[0] - 11075000*x13 + 5750000*x14 - x172 - 12679875*x18 - x197 + 7037500*x2*x9 + 47625*x2 - 36500000*x20 - x206 - 6250000*x22 - 993125*x25 + 587500*x3*xi[1] - 30000*x3 - 43187500*x33 - 12500000*x35 - 3687500*x42 + 2982175*x9*xi[0] - 532755*x9 + 14643*xi[0] + 56223*xi[1]), x169*(22262*x1 - 694455*x10 + 1767975*x11 - 2692500*x12 + 2430000*x13 - 1200000*x14 + x168 - 1110135*x17 + 4232575*x18 - 8748750*x19 + x193 - 40875*x2 + 10067500*x20 + x204 + x207 - 6075000*x21 + 1500000*x22 - 456555*x24 + 780125*x25 - 738750*x26 + 367500*x27 + 41750*x3 - 10113750*x32 + 15975000*x33 - 12375000*x34 - 4773750*x37 - 12750000*x39 - 22500*x4 + 3850000*x42 - 6750000*x43 - 1575000*x46 + 1500000*x47 + x61 + 155957*x9 - 6393*xi[0] - 17733*xi[1]), x163*(19810000*x1*x10 + 18900000*x1*x12 + 1344070*x1*xi[1] + 19250000*x10*x3 + 957950*x10 + 25375000*x11*x2 + 12783750*x11*xi[0] - 2218475*x11 + 3132500*x12 + 7525000*x13*xi[0] - 2660000*x13 + 1250000*x14 - x141 - x142 - x143 + x144 + x146 + x147 - x162 - 6943300*x18 - x191 + 14052500*x2*x9 - 13422500*x20 - 1750000*x22 - 3082100*x25 - 3272500*x27 + 4147500*x3*xi[1] - 27212500*x33 - 5250000*x35 + 7875000*x4*x9 - 8750000*x40 - 14525000*x42 - 5250000*x47 - 1750000*x49 + 1400000*x5*xi[1] - x61 + 2108960*x9*xi[0] - 242149*x9 + 31686*xi[1]), x211*(247984*x0 + 86192*x1 - 263495*x10 + 450500*x11 - 440000*x12 + 230000*x13 - 1429785*x17 + 4028200*x18 - 6287500*x19 - 263495*x2 + 5555000*x20 + x208 + x209 - 2600000*x21 + x210 - 1429785*x24 + 4028200*x25 - 6287500*x26 + 5555000*x27 - 2600000*x28 + 450500*x3 + 7155400*x31 - 16662500*x32 + 20150000*x33 - 12300000*x34 + 3000000*x35 - 16662500*x37 + 29650000*x38 - 24250000*x39 - 440000*x4 + 7500000*x40 + 20150000*x42 - 24250000*x43 - 12300000*x46 + 7500000*x47 + 3000000*x49 + 230000*x5 + 86192*x9 - 13953*xi[0] - 13953*xi[1]), x211*(12788*x1 - 600*x10 - 28600*x17 + 13700*x18 - 60995*x2 + x208 + x212*x9 + x213 - 151995*x24 + 713200*x25 - 1822500*x26 + 2545000*x27 - 1800000*x28 + 159500*x3 - 112500*x32 - 1075000*x37 - 230000*x4 + 2350000*x42 - 750000*x43 - 2500000*x46 + 500000*x47 + 170000*x5 + 1200*x9 - 1347*xi[0] - 654*xi[1]), x211*(x1*x214 + 1200*x1 - 60995*x10 + 159500*x11 - 230000*x12 + 170000*x13 - 151995*x17 + 713200*x18 - 1822500*x19 - 600*x2 + 2545000*x20 + x209 - 1800000*x21 + x213 - 28600*x24 + 13700*x25 - 1075000*x32 + 2350000*x33 - 2500000*x34 - 112500*x37 - 750000*x39 + 500000*x40 + 12788*x9 - 654*xi[0] - 1347*xi[1]), x217*(15412500*x1*x10 + 7000000*x1*x12 + 1901825*x1*xi[1] - 123515*x1 + 35000000*x10*x3 + 140875*x10 + 24375000*x11*x2 + 2943750*x11*xi[0] - 168750*x11 + 102500*x12 - x121 - x170 - 2629375*x18 + 23350000*x2*x9 + 447425*x2 - 1662500*x20 - x215 - x216 - 6478625*x25 - 11650000*x27 + 11725000*x3*xi[1] - 873125*x3 - 14875000*x33 + 23125000*x4*x9 + 946250*x4 - 33312500*x42 - 6250000*x49 + 6000000*x5*xi[1] - 537500*x5 - x80 + 1217275*x9*xi[0] - 61310*x9 + 16221*xi[0] + 12441*xi[1]), x219*(148450*x1 - 69750*x10 + x102 + 52500*x11 + x120 - 926975*x17 + 1437750*x18 - 1022500*x19 - 611725*x2 + x212 + x218 - 2098950*x24 + 8300250*x25 - 17245000*x26 + 19300000*x27 - 11000000*x28 + 1345000*x3 - 9980000*x32 + 6350000*x33 - 24602500*x37 - 14750000*x39 - 1615000*x4 + 41400000*x42 - 35500000*x43 - 33000000*x46 + 15000000*x47 + 10000000*x49 + 42675*x9 - 17481*xi[0] - 11181*xi[1]), x217*(1412500*x1*x10 + 614875*x1*xi[1] - 47435*x1 + 7250000*x10*x3 + 8400*x10 - x164 - x178 - 184000*x18 + 6043750*x2*x9 + 213050*x2 - x215 - x220 - x221 - 2688000*x25 - 7887500*x27 + 6277500*x3*xi[1] - 516875*x3 + 10875000*x4*x9 + 683750*x4 - 11800000*x42 + 5000000*x5*xi[1] - 462500*x5 + 185225*x9*xi[0] - 8070*x9 + 5226*xi[0] + 2886*xi[1]), x217*(10310*x1 - 6600*x10 + x106 + x111 - 100625*x17 + x178 + 143500*x18 - 40375*x2 + x222 - 173775*x24 + 670625*x25 - 1278750*x26 + 1137500*x27 + 78750*x3 - 1087500*x32 - 3018750*x37 - 72500*x4 + 5425000*x42 - 5250000*x43 - 4375000*x46 + 4470*x9 - 1239*xi[0] - 924*xi[1]), x219*(33675*x0 + 11325*x1 - 32250*x10 + 37500*x11 + x112 + x120 - 228025*x17 + x174 + 637750*x18 - 717500*x19 - 32250*x2 + x203 - 228025*x24 + 637750*x25 - 717500*x26 + 37500*x3 + 1515500*x31 - 4092500*x32 + 4300000*x33 - 4092500*x37 + 10300000*x38 + 4300000*x42 + 5000000*x44 + 11325*x9 - 1656*xi[0] - 1656*xi[1] + 81), x217*(4470*x1 - 40375*x10 + 78750*x11 + x119 - 72500*x12 + x121 - 173775*x17 + 670625*x18 - 1278750*x19 - 6600*x2 + 1137500*x20 + x205 + x222 - 100625*x24 + 143500*x25 - 3018750*x32 + 5425000*x33 - 4375000*x34 - 1087500*x37 - 5250000*x39 + 10310*x9 - 924*xi[0] - 1239*xi[1]), x217*(6043750*x1*x10 + 10875000*x1*x12 + 185225*x1*xi[1] - 8070*x1 + 213050*x10 + 7250000*x11*x2 + 6277500*x11*xi[0] - 516875*x11 + 683750*x12 + 5000000*x13*xi[0] - 462500*x13 - 2688000*x18 + 1412500*x2*x9 + 8400*x2 - 7887500*x20 - x205 - x207 - x221 - x223 - x224 - 184000*x25 - 11800000*x33 + 614875*x9*xi[0] - 47435*x9 + 2886*xi[0] + 5226*xi[1]), x219*(42675*x1 - 611725*x10 + 1345000*x11 + x112 - 1615000*x12 + x125 - 2098950*x17 + 8300250*x18 - 17245000*x19 - 69750*x2 + 19300000*x20 - 11000000*x21 + x214 + x218 - 926975*x24 + 1437750*x25 - 1022500*x26 + 52500*x3 - 24602500*x32 + 41400000*x33 - 33000000*x34 + 10000000*x35 - 9980000*x37 - 35500000*x39 + 15000000*x40 + 6350000*x42 - 14750000*x43 + 148450*x9 - 11181*xi[0] - 17481*xi[1]), x217*(23350000*x1*x10 + 23125000*x1*x12 + 1217275*x1*xi[1] - 61310*x1 + 24375000*x10*x3 + 447425*x10 - x106 + 35000000*x11*x2 + 11725000*x11*xi[0] - 873125*x11 + 946250*x12 + 6000000*x13*xi[0] - 537500*x13 - x140 - 6478625*x18 + 15412500*x2*x9 + 140875*x2 - 11650000*x20 - x206 - x216 - x224 - 2629375*x25 - 1662500*x27 + 2943750*x3*xi[1] - 168750*x3 - 33312500*x33 - 6250000*x35 + 7000000*x4*x9 + 102500*x4 - 14875000*x42 + 1901825*x9*xi[0] - 123515*x9 + 12441*xi[0] + 16221*xi[1]), x169*(97220*x0 + 32860*x1 - 94375*x10 + x106 + 133750*x11 - 92500*x12 + x121 + x165 - 623000*x17 + 1709625*x18 + x184*xi[1] - 2291250*x19 - 94375*x2 + x201*xi[0] + x204 - 623000*x24 + 1709625*x25 - 2291250*x26 + 133750*x3 + 3781750*x31 - 9428750*x32 + 11112500*x33 - 6125000*x34 - 9428750*x37 + 19300000*x38 - 16500000*x39 - 92500*x4 + 11112500*x42 - 16500000*x43 + 7500000*x44 - 6125000*x46 + 32860*x9 - 4987*xi[0] - 4987*xi[1] + 252), x169*(3326250*x1*x10 + 319500*x1*xi[1] - 17860*x1 + 10125000*x10*x3 + 24375*x10 - x106 + 6000000*x11*x2 + 435000*x11*xi[0] - x118 - x171 - 494125*x18 + 5452500*x2*x9 + 61875*x2 - x220 - x225 - 1077875*x25 - 1312500*x27 + 1736250*x3*xi[1] - 103750*x3 - 2662500*x33 + 5250000*x4*x9 + 82500*x4 - 7962500*x42 + 237000*x9*xi[0] - 11250*x9 + 2343*xi[0] + 1983*xi[1]), x169*(5452500*x1*x10 + 5250000*x1*x12 + 237000*x1*xi[1] - 11250*x1 + 6000000*x10*x3 + 61875*x10 - x108 + 10125000*x11*x2 + 1736250*x11*xi[0] - 103750*x11 + 82500*x12 - x121 - 1077875*x18 - x197 + 3326250*x2*x9 + 24375*x2 - 1312500*x20 - x223 - x225 - 494125*x25 + 435000*x3*xi[1] - 7962500*x33 - 2662500*x42 + 319500*x9*xi[0] - 17860*x9 + 1983*xi[0] + 2343*xi[1])])
                        case _:
                            assert (
                                False
                            ), "Order of shape functions not implemented or number of nodes not adequat"
                case 3:
                    match n_nodes:
                        case 4:
                            def shape_functions(xi):
                        
                                return jnp.stack([-xi[0] - xi[1] - xi[2] + 1, xi[0], xi[1], xi[2]])

                        case 10:
                            def shape_functions(xi):
                                x0 = 4*xi[0]
                                x1 = x0*xi[1]
                                x2 = x0*xi[2]
                                x3 = 4*xi[1]
                                x4 = x3*xi[2]
                                x5 = -xi[0] - xi[1] - xi[2] + 1
                                return jnp.stack([x1 + x2 + x4 + 2*xi[0]**2 - 3*xi[0] + 2*xi[1]**2 - 3*xi[1] + 2*xi[2]**2 - 3*xi[2] + 1, xi[0]*(2*xi[0] - 1), xi[1]*(2*xi[1] - 1), xi[2]*(2*xi[2] - 1), x0*x5, x1, x3*x5, 4*x5*xi[2], x2, x4])

                        case 20:
                            def shape_functions(xi):
                                x0 = xi[0]**2
                                x1 = xi[1]**2
                                x2 = xi[2]**2
                                x3 = xi[1]*xi[2]
                                x4 = 27*xi[0]
                                x5 = x3*x4
                                x6 = 27*xi[0]/2
                                x7 = 27*xi[1]/2
                                x8 = 27*xi[2]/2
                                x9 = 6*xi[0]
                                x10 = 3*x0
                                x11 = 3*x1
                                x12 = 3*x2
                                x13 = x10 + x11 + x12 + 6*x3 + x9*xi[1] + x9*xi[2] - 5*xi[0] - 5*xi[1] - 5*xi[2] + 2
                                x14 = 9*xi[0]/2
                                x15 = 3*xi[0]
                                x16 = x15*xi[1] - xi[2] + 1
                                x17 = x15*xi[2] - xi[1]
                                x18 = x15 - 1
                                x19 = x14*xi[1]
                                x20 = 3*xi[1]
                                x21 = x20 - 1
                                x22 = x20*xi[2] - xi[0]
                                x23 = 9*xi[1]/2
                                x24 = 9*xi[2]/2
                                x25 = x14*xi[2]
                                x26 = 3*xi[2] - 1
                                x27 = 9*x3/2
                                x28 = -xi[0] - xi[1] - xi[2] + 1
                                x29 = x28*x4
                                return jnp.stack([-x0*x7 - x0*x8 + 9*x0 - x1*x6 - x1*x8 + 9*x1 - x2*x6 - x2*x7 + 9*x2 - x5 - 9*xi[0]**3/2 + 18*xi[0]*xi[1] + 18*xi[0]*xi[2] - 11*xi[0]/2 - 9*xi[1]**3/2 + 18*xi[1]*xi[2] - 11*xi[1]/2 - 9*xi[2]**3/2 - 11*xi[2]/2 + 1, xi[0]*(9*x0 - 9*xi[0] + 2)/2, xi[1]*(9*x1 - 9*xi[1] + 2)/2, xi[2]*(9*x2 - 9*xi[2] + 2)/2, x13*x14, x14*(-x10 - x16 - x17 + 4*xi[0]), x18*x19, x19*x21, x23*(-x11 - x16 - x22 + 4*xi[1]), x13*x23, x13*x24, x24*(-x12 - x17 - x22 + 4*xi[2] - 1), x18*x25, x25*x26, x21*x27, x26*x27, x29*xi[1], x29*xi[2], 27*x28*x3, x5])

                        case 35:
                            def shape_functions(xi):
                                x0 = 140*xi[0]/3
                                x1 = xi[1]*xi[2]
                                x2 = xi[0]**2
                                x3 = xi[0]**3
                                x4 = xi[1]**2
                                x5 = xi[1]**3
                                x6 = xi[2]**2
                                x7 = xi[2]**3
                                x8 = x1*xi[0]
                                x9 = 80*xi[0]
                                x10 = 128*xi[0]/3
                                x11 = 80*xi[1]
                                x12 = 80*xi[2]
                                x13 = 128*xi[1]/3
                                x14 = 128*xi[2]/3
                                x15 = 128*xi[0]
                                x16 = 64*x2
                                x17 = 8*x3
                                x18 = 8*x5
                                x19 = 8*x7
                                x20 = -36*x1
                                x21 = 24*xi[0]
                                x22 = 24*xi[1]
                                x23 = 24*xi[2]
                                x24 = 36*xi[0]
                                x25 = -x24*xi[1]
                                x26 = -x24*xi[2]
                                x27 = x25 + x26 - 3
                                x28 = -x17 - x18 - x19 - x2*x22 - x2*x23 + 18*x2 - x20 - x21*x4 - x21*x6 - x22*x6 - x23*x4 - x27 + 18*x4 + 18*x6 - 48*x8 - 13*xi[0] - 13*xi[1] - 13*xi[2]
                                x29 = 16*xi[0]/3
                                x30 = 32*x2
                                x31 = 8*x1
                                x32 = 16*xi[0]
                                x33 = 7*xi[2]
                                x34 = 4*x6
                                x35 = 32*x8
                                x36 = x33 - x34 + x35
                                x37 = 7*xi[1]
                                x38 = 4*x4
                                x39 = x37 - x38
                                x40 = 4*xi[0]
                                x41 = 7*xi[0]
                                x42 = -6*xi[0]*xi[1]
                                x43 = -6*xi[0]*xi[2]
                                x44 = 8*x2
                                x45 = xi[1] + xi[2] - 1
                                x46 = x44 - 6*xi[0] + 1
                                x47 = x29*xi[1]
                                x48 = 4*xi[1]
                                x49 = -x48
                                x50 = 1 - x40
                                x51 = x40*xi[1]
                                x52 = 8*x4
                                x53 = x52 - 6*xi[1] + 1
                                x54 = xi[0] - 6*xi[1]*xi[2] - 1
                                x55 = 16*xi[1]/3
                                x56 = 32*x4
                                x57 = 8*xi[0]
                                x58 = x57*xi[2]
                                x59 = 16*xi[1]
                                x60 = 4*x2
                                x61 = x20 + x41 - x60 - 3
                                x62 = 16*xi[2]/3
                                x63 = 32*x6
                                x64 = x57*xi[1]
                                x65 = 16*xi[2]
                                x66 = 4*xi[2]
                                x67 = 8*x6
                                x68 = x29*xi[2]
                                x69 = -x66
                                x70 = x40*xi[2]
                                x71 = x67 - 6*xi[2] + 1
                                x72 = 16*x1/3
                                x73 = x48*xi[2]
                                x74 = x31 - x33 + x34 - x37 + x38 - x41 + x58 + x60 + x64 + 3
                                x75 = 32*xi[0]
                                x76 = x75*xi[1]
                                x77 = x51 - xi[2] + 1
                                x78 = x70 - xi[1]
                                x79 = -x60 - x77 - x78 + 5*xi[0]
                                x80 = x73 - xi[0]
                                x81 = -x38 - x77 - x80 + 5*xi[1]
                                x82 = x75*xi[2]
                                x83 = -x34 - x78 - x80 + 5*xi[2] - 1
                                x84 = 32*x1
                                return jnp.stack([x0*xi[1] + x0*xi[2] + 128*x1*x2 + 140*x1/3 + x10*x5 + x10*x7 - x11*x2 - x11*x6 - x12*x2 - x12*x4 + x13*x3 + x13*x7 + x14*x3 + x14*x5 + x15*x4*xi[2] + x15*x6*xi[1] + x16*x4 + x16*x6 + 70*x2/3 - 80*x3/3 + 64*x4*x6 - x4*x9 + 70*x4/3 - 80*x5/3 - x6*x9 + 70*x6/3 - 80*x7/3 - 160*x8 + 32*xi[0]**4/3 - 25*xi[0]/3 + 32*xi[1]**4/3 - 25*xi[1]/3 + 32*xi[2]**4/3 - 25*xi[2]/3 + 1, xi[0]*(-48*x2 + 32*x3 + 22*xi[0] - 3)/3, xi[1]*(-48*x4 + 32*x5 + 22*xi[1] - 3)/3, xi[2]*(-48*x6 + 32*x7 + 22*xi[2] - 3)/3, x28*x29, x40*(x27 + 16*x3 + x30*xi[1] + x30*xi[2] - x30 - x31 + x32*x4 + x32*x6 + x36 + x39 + 19*xi[0]), x29*(-x17 + 14*x2 - x41 - x42 - x43 - x44*xi[1] - x44*xi[2] - x45), x46*x47, x51*(x32*xi[1] + x49 + x50), x47*x53, x55*(-x18 - x37 + 14*x4 - x42 - x52*xi[0] - x52*xi[2] - x54 - xi[2]), x48*(x2*x59 + x25 + x36 + 16*x5 + x56*xi[0] + x56*xi[2] - x56 - x58 + x59*x6 + x61 + 19*xi[1]), x28*x55, x28*x62, x66*(x2*x65 + x26 + x35 + x39 + x4*x65 + x61 + x63*xi[0] + x63*xi[1] - x63 - x64 + 16*x7 + 19*xi[2]), x62*(-x19 - x33 - x43 - x54 + 14*x6 - x67*xi[0] - x67*xi[1] - xi[1]), x46*x68, x70*(x32*xi[2] + x50 + x69), x68*x71, x53*x72, x73*(16*x1 + x49 + x69 + 1), x71*x72, x74*x76, x76*x79, x76*x81, x74*x82, x79*x82, x82*x83, x74*x84, x83*x84, x81*x84, x35*(x40 - 1), x35*(x48 - 1), x35*(x66 - 1), 256*x8*(-x45 - xi[0])])

                        case 56:
                            def shape_functions(xi):
                                x0 = xi[0]**2
                                x1 = xi[0]**3
                                x2 = xi[0]**4
                                x3 = xi[1]**2
                                x4 = xi[1]**3
                                x5 = xi[1]**4
                                x6 = xi[2]**2
                                x7 = xi[2]**3
                                x8 = xi[2]**4
                                x9 = xi[1]*xi[2]
                                x10 = x9*xi[0]
                                x11 = 2125*xi[0]/8
                                x12 = 3125*xi[0]/24
                                x13 = 2125*xi[1]/8
                                x14 = 2125*xi[2]/8
                                x15 = 3125*xi[1]/24
                                x16 = 3125*xi[2]/24
                                x17 = xi[0]*xi[1]
                                x18 = xi[0]*xi[2]
                                x19 = 3125*x0/12
                                x20 = 3125*x1/12
                                x21 = x3*x6
                                x22 = x6*xi[1]
                                x23 = x0*xi[2]
                                x24 = 250*xi[0]
                                x25 = 250*xi[1]
                                x26 = 250*xi[2]
                                x27 = 710*xi[0]
                                x28 = 125*x2
                                x29 = 125*x5
                                x30 = 125*x8
                                x31 = 1050*xi[0]
                                x32 = 500*xi[0]
                                x33 = 1050*xi[1]
                                x34 = 1050*xi[2]
                                x35 = 500*xi[1]
                                x36 = 500*xi[2]
                                x37 = x17*x6
                                x38 = x3*xi[0]
                                x39 = x38*xi[2]
                                x40 = 750*x0
                                x41 = -x0*x33 - x0*x34 + 1500*x0*x9 + 355*x0 + x1*x35 + x1*x36 - 350*x1 - 2100*x10 + 750*x21 + x27*xi[1] + x27*xi[2] + x28 + x29 - x3*x31 - x3*x34 + x3*x40 + 355*x3 + x30 - x31*x6 + x32*x4 + x32*x7 - x33*x6 + x35*x7 + x36*x4 + 1500*x37 + 1500*x39 - 350*x4 + x40*x6 + 355*x6 - 350*x7 + 710*x9 - 154*xi[0] - 154*xi[1] - 154*xi[2] + 24
                                x42 = 25*xi[0]/24
                                x43 = 125*x4
                                x44 = x43*xi[0]
                                x45 = 125*x7
                                x46 = x45*xi[0]
                                x47 = 120*x9
                                x48 = 375*x1
                                x49 = 375*xi[0]
                                x50 = x3*x49
                                x51 = 75*x6
                                x52 = x51*xi[1]
                                x53 = -x50 - x52
                                x54 = x49*x6
                                x55 = 75*x3
                                x56 = x55*xi[2]
                                x57 = -x54 - x56
                                x58 = 47*xi[2]
                                x59 = 25*x7
                                x60 = 60*x6
                                x61 = 355*xi[0]
                                x62 = 375*x0
                                x63 = -750*xi[0]*xi[1]*xi[2]
                                x64 = x3*x62 + x54*xi[1] - x58 - x59 + x60 + x61*xi[1] + x63 + 12
                                x65 = 47*xi[1]
                                x66 = 25*x4
                                x67 = 60*x3
                                x68 = x50*xi[2] + x6*x62 + x61*xi[2] - x65 - x66 + x67
                                x69 = 25*xi[0]/12
                                x70 = x51*xi[0]
                                x71 = -x62*xi[1] - x70
                                x72 = x55*xi[0]
                                x73 = -x62*xi[2] - x72
                                x74 = 125*x0
                                x75 = 155*xi[0]
                                x76 = 150*x10
                                x77 = -x76
                                x78 = x3*x74 + 10*x6 + x75*xi[1] + x77 - 18*xi[2] + 8
                                x79 = 10*x3 + x6*x74 + x75*xi[2] - 18*xi[1]
                                x80 = 125*x1
                                x81 = x80*xi[1]
                                x82 = x80*xi[2]
                                x83 = 6*xi[2]
                                x84 = -x83
                                x85 = 55*xi[0]
                                x86 = x85*xi[1]
                                x87 = x84 + x86 + 6
                                x88 = 6*xi[1]
                                x89 = -x88
                                x90 = x85*xi[2]
                                x91 = x89 + x90
                                x92 = -150*x0 + x80 + x85 - 6
                                x93 = x42*xi[1]
                                x94 = 10*xi[1]
                                x95 = 75*xi[1]
                                x96 = -x95*xi[0] - 2
                                x97 = 25*x0
                                x98 = 15*xi[0]
                                x99 = -x97 + x98
                                x100 = x69*xi[1]
                                x101 = 10*xi[0]
                                x102 = 25*x3
                                x103 = 15*xi[1]
                                x104 = -x102 + x103
                                x105 = 55*xi[1]
                                x106 = x105 - 150*x3 + x43 - 6
                                x107 = x43*xi[2]
                                x108 = 6*xi[0]
                                x109 = -x108
                                x110 = x105*xi[2]
                                x111 = x109 + x110
                                x112 = 25*xi[1]/24
                                x113 = x3*xi[2]
                                x114 = 375*xi[2]
                                x115 = x0*x95
                                x116 = -x114*x3 - x115
                                x117 = 10*x0 + 125*x21 + 155*x9 - 18*xi[0]
                                x118 = 25*xi[1]/12
                                x119 = x45*xi[1]
                                x120 = 120*x18
                                x121 = 375*xi[1]
                                x122 = 75*x23
                                x123 = -x121*x6 - x122
                                x124 = 47*xi[0]
                                x125 = 25*x1
                                x126 = 60*x0
                                x127 = -x124 - x125 + x126 + 375*x21 + x62*x9 + 355*x9
                                x128 = 25*xi[2]/24
                                x129 = 120*x17
                                x130 = 25*xi[2]/12
                                x131 = x42*xi[2]
                                x132 = 10*xi[2]
                                x133 = -75*x18 - 2
                                x134 = x69*xi[2]
                                x135 = 125*x6
                                x136 = 25*x6
                                x137 = 15*xi[2]
                                x138 = -x136 + x137
                                x139 = x45 - 150*x6 + 55*xi[2] - 6
                                x140 = 25*x9/24
                                x141 = -75*x9 - 2
                                x142 = 25*x9/12
                                x143 = -x115 + x120 - x122 - x124 - x125 + x126 + x129 + x47 - x52 - x56 - x58 - x59 + x60 - x65 - x66 + x67 - x70 - x72 - x76 + 12
                                x144 = 125*x17/6
                                x145 = x97*xi[1]
                                x146 = x97*xi[2]
                                x147 = -15*xi[0]*xi[1] + 2*xi[2] - 2
                                x148 = -15*xi[0]*xi[2] + 2*xi[1]
                                x149 = 40*x0 - x125 - x145 - x146 - x147 - x148 - 17*xi[0]
                                x150 = x102*xi[0]
                                x151 = x102*xi[2]
                                x152 = 2*xi[0] - 15*xi[1]*xi[2]
                                x153 = -x147 - x150 - x151 - x152 + 40*x3 - x66 - 17*xi[1]
                                x154 = 50*x0
                                x155 = x94*xi[2]
                                x156 = 5*x3
                                x157 = -x156
                                x158 = x150 + x157
                                x159 = x136*xi[0]
                                x160 = 5*x6
                                x161 = -x160
                                x162 = x159 + x161
                                x163 = 9*xi[2]
                                x164 = 50*x10
                                x165 = x163 + x164 - x86 - 4
                                x166 = 9*xi[1]
                                x167 = x166 - x90
                                x168 = x125 + x154*xi[1] + x154*xi[2] - x154 - x155 + x158 + x162 + x165 + x167 + 29*xi[0]
                                x169 = 125*x17/4
                                x170 = 5*x0
                                x171 = -x170
                                x172 = x145 + x171
                                x173 = 5*xi[1]
                                x174 = x173*xi[2]
                                x175 = 25*x9
                                x176 = x175*xi[0]
                                x177 = x108 - x174 + x176 - 1
                                x178 = 5*xi[0]
                                x179 = x178*xi[2]
                                x180 = -x179 + x88
                                x181 = 50*x3
                                x182 = x101*xi[2]
                                x183 = x136*xi[1]
                                x184 = x161 + x183
                                x185 = 9*xi[0]
                                x186 = -x110 + x185
                                x187 = x165 + x172 + x181*xi[0] + x181*xi[2] - x181 - x182 + x184 + x186 + x66 + 29*xi[1]
                                x188 = 125*x18/6
                                x189 = -x148 - x152 - x159 - x183 - x59 + 40*x6 - 17*xi[2] + 2
                                x190 = 125*x18/4
                                x191 = x146 + x171
                                x192 = x178*xi[1]
                                x193 = -x192 + x83
                                x194 = 50*x6
                                x195 = x94*xi[0]
                                x196 = x151 + x157
                                x197 = x164 + x167 + x186 + x191 + x194*xi[0] + x194*xi[1] - x194 - x195 + x196 + x59 + 29*xi[2] - 4
                                x198 = 125*x9/6
                                x199 = 125*x9/4
                                x200 = 125*x10/6
                                x201 = -x178
                                x202 = 1 - x173
                                x203 = 125*x10/4
                                x204 = -5*xi[2]
                                x205 = 625*x9*xi[0]/2
                                x206 = x174 - xi[0] + 1
                                x207 = x192 - xi[2]
                                x208 = x179 - xi[1]
                                return jnp.stack([-x0*x13 - x0*x14 - 3125*x0*x22/4 + 1875*x0*x3/4 + 1875*x0*x6/4 + 1875*x0*xi[1]*xi[2]/2 + 375*x0/8 - 3125*x1*x9/6 + 625*x1*xi[1]/2 + 625*x1*xi[2]/2 - 2125*x1/24 - 2125*x10/4 - x11*x3 - x11*x6 - x12*x5 - x12*x8 - x13*x6 - x14*x3 - x15*x2 - x15*x8 - x16*x2 - x16*x5 - 3125*x17*x7/6 - 3125*x18*x4/6 - x19*x4 - x19*x7 + 625*x2/8 - x20*x3 - x20*x6 - 3125*x21*xi[0]/4 - 3125*x23*x3/4 + 1875*x3*x6/4 - 3125*x3*x7/12 + 1875*x3*xi[0]*xi[2]/2 + 375*x3/8 - 3125*x4*x6/12 + 625*x4*xi[0]/2 + 625*x4*xi[2]/2 - 2125*x4/24 + 625*x5/8 + 1875*x6*xi[0]*xi[1]/2 + 375*x6/8 + 625*x7*xi[0]/2 + 625*x7*xi[1]/2 - 2125*x7/24 + 625*x8/8 - 625*xi[0]**5/24 + 375*xi[0]*xi[1]/4 + 375*xi[0]*xi[2]/4 - 137*xi[0]/12 - 625*xi[1]**5/24 + 375*xi[1]*xi[2]/4 - 137*xi[1]/12 - 625*xi[2]**5/24 - 137*xi[2]/12 + 1, xi[0]*(875*x0 - 1250*x1 + 625*x2 - x24 + 24)/24, xi[1]*(-x25 + 875*x3 - 1250*x4 + 625*x5 + 24)/24, xi[2]*(-x26 + 875*x6 - 1250*x7 + 625*x8 + 24)/24, x41*x42, x69*(675*x0*xi[1] + 675*x0*xi[2] - 295*x0 + 325*x1 - x28 - x40*x9 - x44 - x46 - x47 - x48*xi[1] - x48*xi[2] - x53 - x57 - x64 - x68 + 107*xi[0]), x69*(245*x0 + x1*x25 + x1*x26 - 300*x1 + x23*x25 + x28 + x71 + x73 + x78 + x79 + 20*x9 - 78*xi[0]), x42*(150*x0*xi[1] + 150*x0*xi[2] - 205*x0 + 275*x1 - x28 - x81 - x82 - x87 - x91 + 61*xi[0]), x92*x93, x100*(x74*xi[1] + x94 + x96 + x99), x100*(x101 + x104 + 125*x38 + x96), x106*x93, x112*(-x107 - x111 - x29 + 150*x3*xi[0] + 150*x3*xi[2] - 205*x3 + 275*x4 - x44 - x87 + 61*xi[1]), x118*(x113*x24 + x116 + x117 + 20*x18 + x24*x4 + x26*x4 + x29 + 245*x3 - 300*x4 + x53 + x78 - 78*xi[1]), x118*(-x114*x4 - x119 - x120 - x123 - x127 - x29 + 675*x3*xi[0] + 675*x3*xi[2] - 295*x3 - 750*x39 - x4*x49 + 325*x4 - x64 - x71 - x81 + 107*xi[1]), x112*x41, x128*x41, x130*(-x107 - x116 - x121*x7 - x127 - x129 - x30 - 750*x37 - x49*x7 + 675*x6*xi[0] + 675*x6*xi[1] - 295*x6 - x63 - x68 + 325*x7 - x73 - x82 + 107*xi[2] - 12), x130*(x117 + x123 + 20*x17 + x22*x24 + x24*x7 + x25*x7 + x30 + x57 + 245*x6 - 300*x7 + x77 + x79 - 78*xi[2] + 8), x128*(-x111 - x119 - x30 - x46 + 150*x6*xi[0] + 150*x6*xi[1] - 205*x6 + 275*x7 - x91 + 61*xi[2] - 6), x131*x92, x134*(x132 + x133 + x74*xi[2] + x99), x134*(x101 + x133 + x135*xi[0] + x138), x131*x139, x106*x140, x142*(x104 + 125*x113 + x132 + x141), x142*(x135*xi[1] + x138 + x141 + x94), x139*x140, x143*x144, x144*x149, x144*x153, x168*x169, x169*(-x158 - x172 - x177 - x180 + 35*xi[0]*xi[1] - xi[2]), x169*x187, x143*x188, x149*x188, x188*x189, x168*x190, x190*(-x162 - x177 - x191 - x193 + 35*xi[0]*xi[2] - xi[1]), x190*x197, x143*x198, x189*x198, x153*x198, x197*x199, x199*(-x176 - x180 - x184 - x193 - x196 - xi[0] + 35*xi[1]*xi[2] + 1), x187*x199, x200*(x97 - x98 + 2), x200*(x102 - x103 + 2), x200*(x136 - x137 + 2), x203*(25*x17 + x201 + x202), x203*(x175 + x202 + x204), x203*(25*x18 + x201 + x204 + 1), x205*(x155 + x156 + x160 - x163 - x166 + x170 + x182 - x185 + x195 + 4), x205*(-x156 - x206 - x207 - x89), x205*(-x160 - x206 - x208 - x84), x205*(-x109 - x170 - x207 - x208 - 1)])

                case _:
                    assert False, "Dimensionality not implemented."
        case _:
            assert False, "Shape functions not implemented for type {name}."
    return shape_functions

### Spaces defined in the reference configuration, for assembling modes user potential/residual/element
def fem_iso_line_quad_brick(x, xI, fI, settings, overwrite_diff, n_dim):
    """
    Compute isoparametric finite element shape functions for line, quadrilateral, and brick elements.

    Args:
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
      - Warning: Only first-order spatial derivatives are supported in the custom JVP implementation with overwritten derivatives.
      - Warning: The derivatives with respect to xI are set to zero.
    """

    n_nodes = xI.shape[0]
    key = ("line_quad_brick", n_dim, n_nodes)
    shape_functions = get_jitted_shape_functions(key)

    if overwrite_diff:
        # Overwrite derivative to be with respect to initial configuration instead of reference configuration
        @jax.custom_jvp
        def ansatz(xi, fI, xI):
            return jnp.einsum('i, i...-> ...', shape_functions(xi), fI)

        @ansatz.defjvp
        def f_jvp(primals, tangents):
            xi, fI, xI = primals
            x_dot, fI_dot, _ = tangents

            # Isoparametric mapping
            initial_coor = lambda xi: jnp.einsum('i, i...-> ...', shape_functions(xi), xI)
            J = jax.jacfwd(initial_coor)(xi)

            fun = lambda xi: jnp.einsum('i, i...-> ...', shape_functions(xi), fI)
            primal_out = fun(xi)
            df_dxi = jax.jacfwd(fun)(xi)

            if J.ndim == 1:
                G = jnp.sum(J * J)
                rhs = jnp.sum(J * x_dot)
                xi_dot = rhs / G
            elif J.shape[0] == J.shape[1]:
                xi_dot = lin_solve(J, x_dot)
            else:
                _, R = jnp.linalg.qr(J, mode="reduced")
                xi_dot = lin_solve(R, x_dot)

            tangent_out = df_dxi * xi_dot if jnp.ndim(xi_dot) == 0 else jnp.einsum('...i, i -> ...', df_dxi, xi_dot)
               
            # primal_out, tangent_out = jax.jvp(fun, (xi,), (lin_solve(dX_dxi, x_dot),))

            # Add tangent with respect to fI
            if fI_dot is not None:
                tangent_out += jnp.einsum('i, i...-> ...', shape_functions(xi), fI_dot)

            return primal_out, tangent_out

        return ansatz(x, fI, xI)
    else:
        return jnp.einsum('i, i...-> ...', shape_functions(x), fI)

def fem_iso_line_tri_tet(x, xI, fI, settings, overwrite_diff, n_dim):
    """
    Compute isoparametric finite element shape functions for line, triangular, and tetrahedral elements.

    Args:
      x (jnp.ndarray): The position of the evaluation point.
      xI (jnp.ndarray): The positions of neighboring nodes.
      fI (jnp.ndarray): The data at neighboring nodes.
      settings (dict): Dictionary containing various settings (not directly used in this function but passed for compatibility).
      overwrite_diff (bool): If True, overwrites the derivative to be with respect to the initial configuration instead of the reference configuration.
      n_dim (int): The dimensionality of the elements (1 for line, 2 for triangle, 3 for tetrahedron).

    Returns:
      float:
        The computed finite element approximation (sum_i shape_fun_i nodal_values_i)

    Notes:
      - This function currently supports line elements up to order 20, triangles up to order 10 and tetrahedrons up to order 5.
      - Though the input `x` is a reference coordinate, its derivative is replaced with respect to the initial configuration when `overwrite_diff` is True.
      - Warning: Only first-order spatial derivatives are supported in the custom JVP implementation with overwritten derivatives.
      - Warning: The derivatives with respect to xI are set to zero.
    """
    n_nodes = xI.shape[0]
    key = ("line_tri_tet", n_dim, n_nodes)
    shape_functions = get_jitted_shape_functions(key)

    if overwrite_diff:
        # Overwrite derivative to be with respect to initial configuration instead of reference configuration
        @jax.custom_jvp
        def ansatz(xi, fI, xI):
            return jnp.einsum('i, i...-> ...', shape_functions(xi), fI)

        @ansatz.defjvp
        def f_jvp(primals, tangents):
            xi, fI, xI = primals
            x_dot, fI_dot, _ = tangents

            # Isoparametric mapping
            initial_coor = lambda xi: jnp.einsum('i, i...-> ...', shape_functions(xi), xI)
            J = jax.jacfwd(initial_coor)(xi)

            fun = lambda xi: jnp.einsum('i, i...-> ...', shape_functions(xi), fI)
            primal_out = fun(xi)
            df_dxi = jax.jacfwd(fun)(xi)

            if J.ndim == 1:
                G = jnp.sum(J * J)
                rhs = jnp.sum(J * x_dot)
                xi_dot = rhs / G
            elif J.shape[0] == J.shape[1]:
                xi_dot = lin_solve(J, x_dot)
            else:
                _, R = jnp.linalg.qr(J, mode="reduced")
                xi_dot = lin_solve(R, x_dot)

            tangent_out = df_dxi * xi_dot if jnp.ndim(xi_dot) == 0 else jnp.einsum('...i, i -> ...', df_dxi, xi_dot)
               
            # primal_out, tangent_out = jax.jvp(fun, (xi,), (lin_solve(dX_dxi, x_dot),))

            # Add tangent with respect to fI
            if fI_dot is not None:
                tangent_out += jnp.einsum('i, i...-> ...', shape_functions(xi), fI_dot)

            return primal_out, tangent_out

        return ansatz(x, fI, xI)
    else:
        return jnp.einsum('i, i...-> ...', shape_functions(x), fI)

def fem_line_quad_brick(x, xI, fI, settings, overwrite_diff, n_dim):
    """
    Compute finite element shape functions for line, quadrilateral, and brick elements.

    Args:
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
      - This function currently supports line elements up to order 10, quadrilateral elements up to order 10, and brick elements up to order 2.
      - Though the input `x` is a reference coordinate, its derivative is replaced with respect to the initial configuration when `overwrite_diff` is True.
      - Warning: Only first-order spatial derivatives are supported in the custom JVP implementation with overwritten derivatives.
      - Warning: The derivatives with respect to xI are set to zero.
    """

    n_mapping_nodes = xI.shape[0]
    n_ansatz_nodes = fI.shape[0]

    mapping_shape_key = ("line_quad_brick", n_dim, n_mapping_nodes)
    ansatz_shape_key = ("line_quad_brick", n_dim, n_ansatz_nodes)
    mapping_shp_fun = get_jitted_shape_functions(mapping_shape_key)
    ansatz_shp_fun = get_jitted_shape_functions(ansatz_shape_key)

    if overwrite_diff:
        # Overwrite derivative to be with respect to initial configuration instead of reference configuration
        @jax.custom_jvp
        def ansatz(xi, fI, xI):
            return jnp.einsum('i, i...-> ...', ansatz_shp_fun(xi), fI)

        @ansatz.defjvp
        def f_jvp(primals, tangents):
            xi, fI, xI = primals
            x_dot, fI_dot, _ = tangents

            # Isoparametric mapping
            initial_coor = lambda xi: jnp.einsum('i, i...-> ...', mapping_shp_fun(xi), xI)
            J = jax.jacfwd(initial_coor)(xi)

            fun = lambda xi: jnp.einsum('i, i...-> ...', ansatz_shp_fun(xi), fI)
            primal_out = fun(xi)
            df_dxi = jax.jacfwd(fun)(xi)

            if J.ndim == 1:
                G = jnp.sum(J * J)
                rhs = jnp.sum(J * x_dot)
                xi_dot = rhs / G
            elif J.shape[0] == J.shape[1]:
                xi_dot = lin_solve(J, x_dot)
            else:
                _, R = jnp.linalg.qr(J, mode="reduced")
                xi_dot = lin_solve(R, x_dot)

            tangent_out = df_dxi * xi_dot if jnp.ndim(xi_dot) == 0 else jnp.einsum('...i, i -> ...', df_dxi, xi_dot)
               
            # Add tangent with respect to fI
            if fI_dot is not None:
                tangent_out += jnp.einsum('i, i...-> ...', ansatz_shp_fun(xi), fI_dot)

            return primal_out, tangent_out

        return ansatz(x, fI, xI)
    else:
        return jnp.einsum('i, i...-> ...', ansatz_shp_fun(x), fI)

def fem_line_tri_tet(x, xI, fI, settings, overwrite_diff, n_dim):
    """
    Compute finite element shape functions for line, triangular, and tetrahedral elements.

    Args:
      x (jnp.ndarray): The position of the evaluation point.
      xI (jnp.ndarray): The positions of neighboring nodes.
      fI (jnp.ndarray): The data at neighboring nodes.
      settings (dict): Dictionary containing various settings (not directly used in this function but passed for compatibility).
      overwrite_diff (bool): If True, overwrites the derivative to be with respect to the initial configuration instead of the reference configuration.
      n_dim (int): The dimensionality of the elements (1 for line, 2 for triangle, 3 for tetrahedron).

    Returns:
      float:
        The computed finite element approximation (sum_i shape_fun_i nodal_values_i)

    Notes:
      - This function currently supports line elements up to order 10, triangles up to order 10 and tetrahedrons up to order 5.
      - Though the input `x` is a reference coordinate, its derivative is replaced with respect to the initial configuration when `overwrite_diff` is True.
      - Warning: Only first-order spatial derivatives are supported in the custom JVP implementation with overwritten derivatives.
      - Warning: The derivatives with respect to xI are set to zero.
    """

    n_mapping_nodes = xI.shape[0]
    n_ansatz_nodes = fI.shape[0]

    mapping_shape_key = ("line_tri_tet", n_dim, n_mapping_nodes)
    ansatz_shape_key = ("line_tri_tet", n_dim, n_ansatz_nodes)
    mapping_shp_fun = get_jitted_shape_functions(mapping_shape_key)
    ansatz_shp_fun = get_jitted_shape_functions(ansatz_shape_key)

    if overwrite_diff:
        # Overwrite derivative to be with respect to initial configuration instead of reference configuration
        @jax.custom_jvp
        def ansatz(xi, fI, xI):
            return jnp.einsum('i, i...-> ...', ansatz_shp_fun(xi), fI)

        @ansatz.defjvp
        def f_jvp(primals, tangents):
            xi, fI, xI = primals
            x_dot, fI_dot, _ = tangents

            # Isoparametric mapping
            initial_coor = lambda xi: jnp.einsum('i, i...-> ...', mapping_shp_fun(xi), xI)
            J = jax.jacfwd(initial_coor)(xi)

            fun = lambda xi: jnp.einsum('i, i...-> ...', ansatz_shp_fun(xi), fI)
            primal_out = fun(xi)
            df_dxi = jax.jacfwd(fun)(xi)

            if J.ndim == 1:
                G = jnp.sum(J * J)
                rhs = jnp.sum(J * x_dot)
                xi_dot = rhs / G
            elif J.shape[0] == J.shape[1]:
                xi_dot = lin_solve(J, x_dot)
            else:
                _, R = jnp.linalg.qr(J, mode="reduced")
                xi_dot = lin_solve(R, x_dot)

            tangent_out = df_dxi * xi_dot if jnp.ndim(xi_dot) == 0 else jnp.einsum('...i, i -> ...', df_dxi, xi_dot)
               
            # Add tangent with respect to fI
            if fI_dot is not None:
                tangent_out += jnp.einsum('i, i...-> ...', ansatz_shp_fun(xi), fI_dot)

            return primal_out, tangent_out

        return ansatz(x, fI, xI)
    else:
        return jnp.einsum('i, i...-> ...', ansatz_shp_fun(x), fI)

def fem_constant(x, xI, fI, settings, overwrite_diff: bool, n_dim: int):
    # fI shape: (1,) or (1, dim) or (1, ...)
    return fI[0]


# TODO: testen und überarbeiten
_RT0_TRI_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
_RT0_TRI_OPPOSITE_VERTS = np.array([2, 0, 1], dtype=int)

def _rt0_orientation(settings: Dict[str, Any], field_key: str, elem_number: int, set: int, n_dofs: int):
    orientation = settings.get("orientation", None)
    if orientation is None:
        return jnp.ones((n_dofs,))
    signs_by_field = orientation[set]
    if field_key not in signs_by_field:
        return jnp.ones((n_dofs,))
    return signs_by_field[field_key][elem_number]

def _rt0_triangle_mapping(xi, xI, settings, n_dim):
    return fem_line_tri_tet(xi, xI, xI, settings, False, n_dim)

def _rt0_triangle_basis_phys(xi, xI, settings, n_dim):
    ref_verts = jnp.asarray(_RT0_TRI_VERTS, dtype=xi.dtype)
    phi_ref = xi[None, :] - ref_verts[_RT0_TRI_OPPOSITE_VERTS]
    jac = jax.jacfwd(lambda z: _rt0_triangle_mapping(z, xI, settings, n_dim))(xi)
    det_jac = jnp.linalg.det(jac)
    return (jac @ phi_ref.T).T / det_jac

def _fem_rt0_triangle(
    x: jnp.ndarray,
    xI: jnp.ndarray,
    fI: jnp.ndarray,
    settings: Dict[str, Any],
    overwrite_diff: bool,
    n_dim: int,
    elem_number: int,
    set: int,
    *,
    field_key: str,
):
    if n_dim != 2 or xI.shape[-1] != 2:
        raise ValueError("HDiv(order=0) is implemented for 2D triangles only.")

    coeffs = jnp.ravel(jnp.asarray(fI))
    signs = _rt0_orientation(settings, field_key, elem_number, set, 3)
    coeffs = coeffs * signs

    def value(xi, coeffs_, xI_):
        basis_phys = _rt0_triangle_basis_phys(xi, xI_, settings, n_dim)
        return jnp.einsum("i,ia->a", coeffs_, basis_phys)

    if not overwrite_diff:
        return value(x, coeffs, xI)

    @jax.custom_jvp
    def ansatz(xi, coeffs_, xI_):
        return value(xi, coeffs_, xI_)

    @ansatz.defjvp
    def f_jvp(primals, tangents):
        xi, coeffs_, xI_ = primals
        x_dot, coeffs_dot, _ = tangents
        jac = jax.jacfwd(lambda z: _rt0_triangle_mapping(z, xI_, settings, n_dim))(xi)

        if jac.shape[0] == jac.shape[1]:
            xi_dot = lin_solve(jac, x_dot)
        else:
            _, r = jnp.linalg.qr(jac, mode="reduced")
            xi_dot = lin_solve(r, x_dot)

        primal_out, tangent_out = jax.jvp(lambda z: value(z, coeffs_, xI_), (xi,), (xi_dot,))
        if coeffs_dot is not None:
            tangent_out += jnp.einsum(
                "i,ia->a",
                coeffs_dot,
                _rt0_triangle_basis_phys(xi, xI_, settings, n_dim),
            )
        return primal_out, tangent_out

    return ansatz(x, coeffs, xI)

def _fem_rt0_line_normal_trace(
    x: jnp.ndarray,
    xI: jnp.ndarray,
    fI: jnp.ndarray,
    settings: Dict[str, Any],
    overwrite_diff: bool,
    n_dim: int,
    elem_number: int,
    set: int,
    *,
    field_key: str,
):
    del x, overwrite_diff, n_dim
    coeff = jnp.ravel(jnp.asarray(fI))[0]
    sign = jnp.ravel(_rt0_orientation(settings, field_key, elem_number, set, 1))[0]
    length = jnp.linalg.norm(xI[1] - xI[0])
    return sign * coeff / length



### Spaces defined in the physical configuration, for assembling modes sparse/dense
@jit_with_docstring(static_argnames=["static_settings", "set"])
def solution_space(x, int_point_number, local_dofs, settings, static_settings, set):
    """
    Compute the solution space for a given integration point and local degrees of freedom.

    This function determines the type of solution space based on the provided settings
    and computes it accordingly. The supported types of solution spaces include
    moving least squares (mls), finite element simplices (fem simplex), nodal values,
    and user-defined solution spaces.

    Args:
      x (jnp.ndarray): The coordinates of the evaluation point.
      int_point_number (int): The index of the integration point.
      local_dofs (jnp.ndarray): The local degrees of freedom.
      settings (dict): A dictionary containing various settings required for the computation.
      static_settings (dict): A dictionary containing static settings that define the solution space and other parameters.
      set (int): The index of the current set of settings being used.

    Returns:
      jnp.ndarray: The computed solution space value or shape functions at the evaluation point.
    """
    if isinstance(local_dofs, dict):
        raise TypeError("solution_space does currently not support DOFs as dicts.")

    # Warning if it was defined in static_settings
    assert "connectivity" not in static_settings, \
        "'connectivity' has been moved to 'settings' in order to reduce compile time. \
        Further, you should not transform it to a tuple of tuples anymore."

    space_type = static_settings["solution space"][set]
    if space_type == "mls":
        beta = settings["beta"][set]
        x_nodes = settings["node coordinates"]
        neighbor_list = settings["connectivity"][set]
        support_radius = settings["support radius"][set]
        x_local_nodes = x_nodes[neighbor_list[int_point_number]]
        return moving_least_squares(
            x, x_local_nodes, local_dofs, beta, support_radius, static_settings, set
        )
    elif space_type == "fem simplex":
        x_nodes = settings["node coordinates"]
        connectivity_list = settings["connectivity"][set]
        x_local_nodes = x_nodes[connectivity_list[int_point_number]]
        return fem_ini_simplex(x, x_local_nodes, local_dofs, static_settings, set)
    elif space_type == "nodal values":
        mode = static_settings["shape function mode"]
        if mode == "direct":
            return local_dofs
        elif mode == "compiled":
            return jnp.asarray([1.0])
    elif space_type == "user":
        return static_settings["user solution space function"][set](
            x, int_point_number, local_dofs, settings, static_settings, set
        )
    else:
        assert False, "Solution space not defined!"

@jit_with_docstring(static_argnames=["static_settings", "set"])
def moving_least_squares(x, xI, fI, beta, support_radius, static_settings, set):
    """
    Compute the moving least squares (MLS) approximation for a given set of points and data.

    Args:
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

    Returns:
      jnp.ndarray:
        The computed MLS approximation at the evaluation point, either as shape functions or the evaluated function, depending on wether the compiled mode or direct mode is chosen.
    """
    if isinstance(xI, dict):
        raise TypeError("moving_least_squares does currently not support DOFs as dicts.")

    order = static_settings["order of basis functions"][set]
    n_dim = x.shape[0]
    mode = static_settings["shape function mode"]
    basis_length = _compute_poly_basis_length(n_dim, order)

    # Initial coefficients
    a_0 = jnp.zeros(basis_length)

    # Radial weigth function, smooth Gauß Kernel
    def weight_function(r_squared):
        scaled_r_squared = r_squared / (support_radius**2)
        cond = jnp.where(scaled_r_squared < 1, 1, 0)

        weight_function_type = static_settings["weight function type"][set]
        if weight_function_type == "gaussian":
            return (
                (jnp.exp(-(beta**2) * scaled_r_squared) - jnp.exp(-(beta**2)))
                / (1 - jnp.exp(-(beta**2)))
                * cond
            )  # Gauss kernel
        elif weight_function_type == "bump":
            return (
                jnp.exp(beta**2 * scaled_r_squared / (scaled_r_squared - 1.0))
            ) * cond  # Bump function
        elif weight_function_type == "gaussian perturbed kronecker":
            return (
                (jnp.exp(-(beta**2) * scaled_r_squared) - jnp.exp(-(beta**2)))
                / (1 - jnp.exp(-(beta**2)))
                / (jnp.sqrt(scaled_r_squared) + 1e-6)
            ) * cond  # Gauss kernel
        elif weight_function_type == "bump perturbed kronecker":
            return (
                (jnp.exp(beta**2 * scaled_r_squared / (scaled_r_squared - 1.0)))
                / (jnp.sqrt(scaled_r_squared) + 1e-6)
            ) * cond  # Bump function
        else:
            assert (
                False
            ), "Weight function type has to be either 'gaussian', 'bump', 'gaussian perturbed kronecker' or 'bump perturbed kronecker'."

    # Ansatz of approximation, evaluated at shifted and scaled xi
    def polynomial_ansatz(a, xi):
        return jnp.dot(a, _polynomial_basis((xi - x) / support_radius, order))

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
        eWLS = weighted_squared_error_vmap(
            a, xI, fI0
        ).sum()  # For computation of shape functions one field is sufficient
        return eWLS

    # Compute coefficients via one Newton step
    def compute_mls(fI0):
        residual = jax.jacrev(mls_error_functional)
        tangent = jax.jacfwd(residual)
        residual_0 = residual(a_0, fI0)
        tangent_0 = tangent(a_0, fI0)

        chol, lower = jax.scipy.linalg.cho_factor(tangent_0)
        a_mls = -jax.scipy.linalg.cho_solve((chol, lower), residual_0)
        return polynomial_ansatz(a_mls, x)

    # Filter the shape functions
    tmpI = jnp.ones(fI.shape[0])
    shape_functions = jax.jacrev(compute_mls)(tmpI)

    if mode == "direct":
        # Use shape functions for all fields
        return jnp.dot(shape_functions, fI)
    elif mode == "compiled":
        return shape_functions
    else:
        assert False, "Wrong mode of shape function computation"

@jit_with_docstring(static_argnames=["static_settings", "set"])
def fem_ini_simplex(x, xI, fI, static_settings, set):
    """
    Compute finite element shape functions directly in the initial/physical configuration.

    Args:
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
    if isinstance(xI, dict):
        raise TypeError("fem_ini_simplex does currently not support DOFs as dicts.")

    mode = static_settings["shape function mode"]
    n_dim = xI.shape[-1]
    n_nodes = xI.shape[0]
    match n_dim:  # Compute the polynomial order based on the number of nodes per element
        case 1:
            order = n_nodes - 1
        case 2:
            match n_nodes:
                case 3:
                    order = 1
                case 6:
                    order = 2
                case _:
                    assert (
                        False
                    ), "Order of shape functions not implemented or number of nodes not adequat"
        case 3:
            match n_nodes:
                case 4:
                    order = 1
                case 10:
                    order = 2
                case _:
                    assert (
                        False
                    ), "Order of shape functions not implemented or number of nodes not adequat"
        case _:
            order = 1
            assert (
                n_nodes == n_dim + 1
            ), "Order of shape functions not implemented or number of nodes not adequat"
    basis_length = n_nodes

    # Initial coefficients
    a_0 = jnp.zeros(basis_length)

    # Ansatz of approximation, evaluated at shifted and scaled xi
    xc = jnp.mean(xI, axis=0)
    scaling_length = jnp.mean(xI - xc, axis=0)

    def polynomial_ansatz(a, xi):
        scaling_length = 1
        return jnp.dot(a, _polynomial_basis((xi - x) / scaling_length, order))

    # Squared error at node i
    def squared_error(a, xi, fi):
        e = polynomial_ansatz(a, xi) - fi
        return e**2

    # Summing weighted squared errors
    def error_functional(a, fI0):
        squared_error_vmap = jax.vmap(squared_error, (None, 0, 0), 0)
        eWLS = squared_error_vmap(
            a, xI, fI0
        ).sum()  # For computation of shape functions one field is sufficient
        return eWLS

    # Compute coefficients via one Newton step
    def compute_ls(fI0):
        residual = jax.jacrev(error_functional)
        tangent = jax.jacfwd(residual)
        residual_0 = residual(a_0, fI0)
        tangent_0 = tangent(a_0, fI0)

        chol, lower = jax.scipy.linalg.cho_factor(tangent_0)
        a_mls = -jax.scipy.linalg.cho_solve((chol, lower), residual_0)
        return polynomial_ansatz(a_mls, x)

    # Filter the shape functions
    tmpI = jnp.ones(fI.shape[0])
    shape_functions = jax.jacrev(compute_ls)(tmpI)

    if mode == "direct":
        # Use shape functions for all fields
        return jnp.dot(shape_functions, fI)
    elif mode == "compiled":
        return shape_functions
    else:
        assert False, "Wrong mode of shape function computation"


### Pre-computing shape functions
@jit_with_docstring(static_argnames=["static_settings", "set", "num_diff"])
def precompute_shape_functions(dofs, settings, static_settings, set, num_diff):
    """
    Precompute shape functions and their derivatives for all integration points.

    Args:
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
    if isinstance(dofs, dict):
        raise TypeError("precompute_shape_functions does currently not support DOFs as dicts.")

    # Warning if it was defined in static_settings
    assert "connectivity" not in static_settings, \
        "'connectivity' has been moved to 'settings' in order to reduce compile time. \
        Further, you should not transform it to a tuple of tuples anymore."

    neighbor_list = settings["connectivity"][set]
    local_dofs = dofs[neighbor_list]
    x_int = settings["integration coordinates"][set]
    int_point_numbers = jnp.arange(0, x_int.shape[0], 1)

    # Computing shape functions and derivatives
    shp_i = _shape_funs(
        x_int, int_point_numbers, local_dofs, settings, static_settings, set
    )
    if num_diff == 0:
        return shp_i

    dshp_i = _shape_funs_dx(
        x_int, int_point_numbers, local_dofs, settings, static_settings, set
    )
    if num_diff == 1:
        return (shp_i, dshp_i)

    ddshp_i = _shape_funs_dxx(
        x_int, int_point_numbers, local_dofs, settings, static_settings, set
    )
    if num_diff == 2:
        return (shp_i, dshp_i, ddshp_i)

    assert False, "Number of differentiations not implemented!"




## Spaces for high-level interface
class AbstractSpace(ABC):
    """Abstract local space definition with number of dofs per entity."""

    # Concrete spaces should override this
    available_cell_types = frozenset()

    def __init__(self, order, dim, field_dimension):
        self.order = int(order)
        self.dim = int(dim)
        self.field_dimension = field_dimension

        # Every concrete space must provide these (dicts keyed by cell_type)
        self.num_node_dofs = {}
        self.num_edge_dofs = {}
        self.num_face_dofs = {}
        self.num_cell_dofs = {}
        self.default_integration_order = int

    @abstractmethod
    def basis(self, cell_type, xi):
        """Return basis functions for given cell_type."""
        raise NotImplementedError

    @abstractmethod
    def surface_basis(self, cell_type, xi):
        """Return facet basis functions for given cell_type."""
        raise NotImplementedError

    # TODO: ordering/permutations...?!

class H1(AbstractSpace):
    """
    Continuous (H1-conforming) Lagrange finite element space of degree ``order``.

    DOFs are shared across element boundaries, so the field is C0-continuous:
    1 per vertex node, ``order-1`` per edge, plus face dofs (3D) and interior
    (cell) dofs for higher order. Requires ``order >= 1``.

    Convention:
      - line/triangle/tetra: P_k (total degree)
      - quad/hex: Q_k (tensor-product degree)
    """
    available_cell_types = frozenset({"line", "triangle", "quad", "tetra", "hex"})

    # Function tables (read-only to avoid accidental global mutation)
    _basis = MappingProxyType({
        "line": fem_line_quad_brick,
        "quad": fem_line_quad_brick,
        "hex": fem_line_quad_brick,
        "triangle": fem_line_tri_tet,
        "tetra": fem_line_tri_tet,
    })

    _surface_basis = MappingProxyType({
        "line": fem_line_quad_brick,
        "quad": fem_line_quad_brick,
        "hex": fem_line_quad_brick,
        "triangle": fem_line_tri_tet,
        "tetra": fem_line_tri_tet,
    })

    def __init__(self, order, dim, field_dimension):
        super().__init__(order=order, dim=dim, field_dimension=field_dimension)
        self.default_integration_order = 2 * order
        p = self.order
        if order == 0:
            raise ValueError("Order has to be at least 1 for H1 space.")

        self.num_node_dofs = {ct: 1 for ct in self.available_cell_types}
        self.num_edge_dofs = {ct: max(0, p - 1) for ct in self.available_cell_types}

        # In 3D: faces exist. In 2D: keep face dofs at 0, interior goes to cell.
        self.num_face_dofs = {ct: 0 for ct in self.available_cell_types}
        self.num_cell_dofs = {ct: 0 for ct in self.available_cell_types}

        # 1D (line) as a top-dimensional domain: interior (cell) dofs. This is
        # read only when a line is the domain itself (dim==1); the p-1 dofs of a
        # line acting as an *edge* of a 2D/3D cell come from num_edge_dofs["line"].
        self.num_cell_dofs["line"] = max(0, p - 1)

        # 2D simplex (triangle): interior (cell) dofs
        self.num_cell_dofs["triangle"] = max(0, (p - 1) * (p - 2) // 2)

        # 2D tensor (quad): interior (cell) dofs
        self.num_cell_dofs["quad"] = max(0, (p - 1) ** 2)

        # 3D simplex (tetra): face + cell dofs
        self.num_face_dofs["tetra"] = max(0, (p - 1) * (p - 2) // 2)
        self.num_cell_dofs["tetra"] = max(0, (p - 1) * (p - 2) * (p - 3) // 6)

        # 3D tensor (hex): face + cell dofs
        self.num_face_dofs["hex"] = max(0, (p - 1) ** 2)
        self.num_cell_dofs["hex"] = max(0, (p - 1) ** 3)

    def basis(self, cell_type):
        return self._basis[cell_type]

    def surface_basis(self, cell_type):
        return self._surface_basis[cell_type]

class L2(AbstractSpace):
    """
    Discontinuous L2 polynomial space:
      - Per element: Lagrange polynomials up to degree `order`
      - Discontinuous across element boundaries
      - All dofs live on cells (no node/edge/face dofs)

    Convention:
      - line/triangle/tetra: P_k (total degree)
      - quad/hex: Q_k (tensor-product degree)
    """
    available_cell_types = frozenset({"line", "triangle", "quad", "tetra", "hex"})

    # Reuse your existing basis evaluators (must span the intended polynomial space).
    _basis = MappingProxyType({
        "line": fem_line_quad_brick,
        "quad": fem_line_quad_brick,
        "hex": fem_line_quad_brick,
        "triangle": fem_line_tri_tet,
        "tetra": fem_line_tri_tet,
    })

    # For DG, facet integrals usually need traces of the *cell* basis.
    # If your framework expects a separate facet basis, adjust accordingly.
    _surface_basis = MappingProxyType({
        "line": fem_line_quad_brick,
        "quad": fem_line_quad_brick,
        "hex": fem_line_quad_brick,
        "triangle": fem_line_tri_tet,
        "tetra": fem_line_tri_tet,
    })

    def __init__(self, order, dim, field_dimension):
        super().__init__(order=order, dim=dim, field_dimension=field_dimension)
        p = self.order
        self.default_integration_order = 2 * p

        # No sub-entity dofs for discontinuous L2 spaces
        self.num_node_dofs = {ct: 0 for ct in self.available_cell_types}
        self.num_edge_dofs = {ct: 0 for ct in self.available_cell_types}
        self.num_face_dofs = {ct: 0 for ct in self.available_cell_types}

        # Cell dofs = dimension of polynomial space on the reference cell
        self.num_cell_dofs = {ct: 0 for ct in self.available_cell_types}

        # P_p on simplex cells
        self.num_cell_dofs["line"]     = p + 1
        self.num_cell_dofs["triangle"] = (p + 1) * (p + 2) // 2
        self.num_cell_dofs["tetra"]    = (p + 1) * (p + 2) * (p + 3) // 6

        # Q_p on tensor-product cells
        self.num_cell_dofs["quad"] = (p + 1) ** 2
        self.num_cell_dofs["hex"]  = (p + 1) ** 3

        if order == 0:
            self._basis = MappingProxyType({
                "line": fem_constant,
                "quad": fem_constant,
                "hex": fem_constant,
                "triangle": fem_constant,
                "tetra": fem_constant,
            })
            self._surface_basis = MappingProxyType({
                "line": fem_constant,
                "quad": fem_constant,
                "hex": fem_constant,
                "triangle": fem_constant,
                "tetra": fem_constant,
            })

    def basis(self, cell_type):
        return self._basis[cell_type]

    def surface_basis(self, cell_type):
        return self._surface_basis[cell_type]

class HDiv(AbstractSpace):
    """Experimental. Lowest-order Raviart-Thomas space on 2D triangles."""

    name = "HDiv"
    family = "HDiv"
    orientation_kind = "edge"
    available_cell_types = frozenset({"triangle"})

    def __init__(self, order=0, dim=2, field_dimension=1):
        if int(order) != 0:
            raise NotImplementedError("Only HDiv(order=0) is implemented.")
        if int(dim) != 2:
            raise NotImplementedError("Only 2D Raviart-Thomas triangles are implemented.")
        if field_dimension != 1:
            raise ValueError("RT0 uses scalar edge-flux DOFs; set field_dimension=1.")
        super().__init__(order=order, dim=dim, field_dimension=field_dimension)
        self.default_integration_order = 4
        self.num_node_dofs = {"triangle": 0, "line": 0}
        self.num_edge_dofs = {"triangle": 1, "line": 1}
        self.num_face_dofs = {"triangle": 0, "line": 0}
        self.num_cell_dofs = {"triangle": 0, "line": 0}

        print("Warning: HDiv is experimental and may be subject to changes without notice.")

    def basis(self, cell_type):
        if cell_type != "triangle":
            raise NotImplementedError("RT0 basis is implemented for triangle cells only.")
        return self._basis_for_field(cell_type, "rt")

    def surface_basis(self, cell_type):
        if cell_type != "line":
            raise NotImplementedError("RT0 normal trace is implemented for line facets only.")
        return self._surface_basis_for_field(cell_type, "rt")

    def _basis_for_field(self, cell_type, field_key):
        if cell_type != "triangle":
            raise NotImplementedError("RT0 basis is implemented for triangle cells only.")

        def basis(x, xI, fI, settings, overwrite_diff, n_dim, elem_number, set):
            return _fem_rt0_triangle(
                x, xI, fI, settings, overwrite_diff, n_dim, elem_number, set, field_key=field_key
            )

        return basis

    def _surface_basis_for_field(self, cell_type, field_key):
        if cell_type != "line":
            raise NotImplementedError("RT0 normal trace is implemented for line facets only.")

        def basis(x, xI, fI, settings, overwrite_diff, n_dim, elem_number, set):
            return _fem_rt0_line_normal_trace(
                x, xI, fI, settings, overwrite_diff, n_dim, elem_number, set, field_key=field_key
            )

        return basis

class InternalVariable(AbstractSpace):
    """
    Gauss-point history variable for the high-level SimState interface.

    Internal variables are declared next to FE spaces, but they do not define a
    basis or global dofs. SimState allocates them per materialized model block
    and quadrature point.
    """

    available_cell_types = frozenset()
    is_internal_variable = True

    def __init__(self, field_dimension=1):
        super().__init__(order=0, dim=0, field_dimension=field_dimension)
        self.default_integration_order = 0
        self.num_node_dofs = {}
        self.num_edge_dofs = {}
        self.num_face_dofs = {}
        self.num_cell_dofs = {}

    def basis(self, cell_type):
        raise TypeError("InternalVariable does not define basis functions.")

    def surface_basis(self, cell_type):
        raise TypeError("InternalVariable does not define surface basis functions.")



# TODO: z.b. um integrale werte zu berechnen und zu speichern
# class GlobalConstant:
#     name = "GlobalConstant"
#     def __init__(self):
#         self.order = 0
#         self.default_integration_order = 0
