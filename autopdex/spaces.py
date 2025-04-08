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

from autopdex.utility import jit_with_docstring, matrix_inv


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
                        return jnp.asarray([0.5 - x0, x0 + 0.5])

                case 3:

                    def shape_functions(xi):
                        x0 = 0.5 * xi
                        return jnp.asarray(
                            [x0 * (xi - 1.0), x0 * (xi + 1.0), 1.0 - xi**2]
                        )

                case 4:

                    def shape_functions(xi):
                        x0 = 3.0 * xi
                        x1 = x0 + 1.0
                        x2 = xi + 1.0
                        x3 = x2 * (x0 - 1.0)
                        x4 = 0.5625 * xi - 0.5625
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray([x1 * x3, x3 * x4, x4 * x5, x1 * x5])

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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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

                case 144:

                    def shape_functions(xi):
                        x0 = 11.0 * xi[1]
                        x1 = x0 + 9.0
                        x2 = xi[0] - 1.0
                        x3 = x1 * x2
                        x4 = 11.0 * xi[0]
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
                        x25 = (
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
                            * x23
                            * x24
                            * x5
                            * x6
                            * x7
                            * x8
                            * x9
                        )
                        x26 = 1.8105646200481947198e-20 * x25
                        x27 = xi[0] + 1.0
                        x28 = x1 * x27
                        x29 = xi[1] + 1.0
                        x30 = x13 * x29
                        x31 = x28 * x30
                        x32 = (
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
                            * x23
                            * x24
                            * x5
                            * x6
                            * x7
                            * x8
                            * x9
                        )
                        x33 = 1.8105646200481947198e-20 * x32
                        x34 = x3 * x30
                        x35 = 2.1907831902583156109e-18 * x27
                        x36 = x14 * x3
                        x37 = x35 * x36
                        x38 = (
                            x12
                            * x15
                            * x16
                            * x17
                            * x18
                            * x19
                            * x20
                            * x21
                            * x22
                            * x23
                            * x24
                            * x5
                            * x6
                            * x7
                            * x8
                            * x9
                        )
                        x39 = x36 * x38
                        x40 = x10 * x27
                        x41 = 1.0953915951291578055e-17 * x40
                        x42 = x13 * x41
                        x43 = (
                            x15
                            * x16
                            * x17
                            * x18
                            * x19
                            * x20
                            * x21
                            * x22
                            * x23
                            * x24
                            * x5
                            * x6
                            * x7
                            * x8
                        )
                        x44 = x36 * x43
                        x45 = 3.2861747853874734164e-17 * x11
                        x46 = x12 * x40
                        x47 = x45 * x46
                        x48 = x13 * x47
                        x49 = x20 * x36
                        x50 = x13 * x49
                        x51 = (
                            x15
                            * x16
                            * x17
                            * x18
                            * x19
                            * x21
                            * x22
                            * x23
                            * x24
                            * x5
                            * x6
                            * x8
                        )
                        x52 = x11 * x9
                        x53 = x46 * x52
                        x54 = x51 * x53
                        x55 = 6.5723495707749468328e-17 * x54
                        x56 = (
                            x15 * x16 * x17 * x18 * x21 * x22 * x23 * x24 * x6 * x7 * x8
                        )
                        x57 = x19 * x53
                        x58 = x50 * x57
                        x59 = 9.2012893990849255659e-17 * x58
                        x60 = x16 * x17 * x18 * x21 * x22 * x5 * x6 * x7 * x8
                        x61 = x23 * x24
                        x62 = x60 * x61
                        x63 = x61 * x8
                        x64 = x15 * x16 * x18 * x21 * x22 * x5 * x6 * x7
                        x65 = 6.5723495707749468328e-17 * x64
                        x66 = x63 * x65
                        x67 = x49 * x5
                        x68 = x56 * x9
                        x69 = x16 * x17 * x18 * x22
                        x70 = x6 * x69
                        x71 = x63 * x70
                        x72 = x12 * x7
                        x73 = x15 * x19
                        x74 = x52 * x73
                        x75 = x72 * x74
                        x76 = x71 * x75
                        x77 = x10 * x20
                        x78 = x12 * x77
                        x79 = x24 * x60
                        x80 = x74 * x79
                        x81 = x78 * x80
                        x82 = x29 * x35
                        x83 = x14 * x31
                        x84 = x52 * x83
                        x85 = 1.0953915951291578055e-17 * x10 * x43
                        x86 = x45 * x83
                        x87 = x17 * x61
                        x88 = x19 * x87
                        x89 = x65 * x78 * x88
                        x90 = x21 * x5
                        x91 = x69 * x90
                        x92 = x63 * x91
                        x93 = x75 * x77
                        x94 = x83 * x93
                        x95 = 9.2012893990849255659e-17 * x94
                        x96 = x17 * x22 * x6 * x90
                        x97 = x18 * x63
                        x98 = x96 * x97
                        x99 = x16 * x94
                        x100 = x63 * x96
                        x101 = 6.5723495707749468328e-17 * x100
                        x102 = x51 * x9
                        x103 = x10 * x102 * x72
                        x104 = x17 * x6
                        x105 = x90 * x97
                        x106 = x104 * x105
                        x107 = 1.0953915951291578055e-17 * x106
                        x108 = x23 * x60
                        x109 = 2.1907831902583156109e-18 * x108 * x74 * x78
                        x110 = x20 * x34
                        x111 = x110 * x5
                        x112 = x111 * x76
                        x113 = x110 * x57
                        x114 = 9.2012893990849255659e-17 * x113
                        x115 = x34 * x43
                        x116 = x34 * x38
                        x117 = x14 * x34
                        x118 = x117 * x93
                        x119 = x118 * x16
                        x120 = 9.2012893990849255659e-17 * x118
                        x121 = x117 * x52
                        x122 = x116 * x14
                        x123 = x2 * x29
                        x124 = 2.6508476602125618892e-16 * x14
                        x125 = x124 * x15
                        x126 = x2 * x30
                        x127 = x126 * x20
                        x128 = x57 * x79
                        x129 = x15 * x57
                        x130 = x129 * x29
                        x131 = x126 * x14
                        x132 = 1.3254238301062809446e-15 * x40
                        x133 = 3.9762714903188428338e-15 * x11
                        x134 = x133 * x46
                        x135 = x127 * x14
                        x136 = 7.9525429806376856677e-15 * x135
                        x137 = x135 * x56
                        x138 = 1.1133560172892759935e-14 * x57
                        x139 = x57 * x64
                        x140 = x139 * x63
                        x141 = 3.9762714903188428338e-15 * x53
                        x142 = 1.3254238301062809446e-15 * x7
                        x143 = x129 * x142
                        x144 = x5 * x71
                        x145 = x110 * x14
                        x146 = x145 * x80
                        x147 = x12 * x27
                        x148 = x113 * x14
                        x149 = x148 * x24
                        x150 = 7.9525429806376856677e-15 * x149
                        x151 = 1.1133560172892759935e-14 * x7
                        x152 = x15 * x8
                        x153 = x149 * x152
                        x154 = x151 * x153
                        x155 = x16 * x7
                        x156 = x104 * x142 * x16
                        x157 = x23 * x8
                        x158 = x111 * x14
                        x159 = x158 * x70
                        x160 = x108 * x145
                        x161 = 7.9525429806376856677e-15 * x148
                        x162 = x151 * x21
                        x163 = x148 * x70
                        x164 = x152 * x23
                        x165 = x160 * x73
                        x166 = x46 * x9
                        x167 = x130 * x67
                        x168 = x21 * x97
                        x169 = x167 * x63
                        x170 = x104 * x155
                        x171 = x170 * x22
                        x172 = x171 * x21
                        x173 = x104 * x22
                        x174 = x173 * x97
                        x175 = 6.6271191505314047231e-15 * x40
                        x176 = x115 * x14
                        x177 = x158 * x7
                        x178 = x129 * x158
                        x179 = x106 * x155
                        x180 = x145 * x166
                        x181 = x180 * x73
                        x182 = 1.9881357451594214169e-14 * x40
                        x183 = x11 * x176
                        x184 = x145 * x52
                        x185 = x184 * x40
                        x186 = 3.9762714903188428338e-14 * x185
                        x187 = x19 * x56
                        x188 = 5.5667800864463799674e-14 * x185
                        x189 = x19 * x62
                        x190 = x19 * x63 * x64
                        x191 = x158 * x52 * x56
                        x192 = 1.9881357451594214169e-14 * x27
                        x193 = x129 * x7
                        x194 = x63 * x69
                        x195 = 5.5667800864463799674e-14 * x129 * x177
                        x196 = 1.9881357451594214169e-14 * x117
                        x197 = 1.9881357451594214169e-14 * x179
                        x198 = x145 * x53
                        x199 = x15 * x198
                        x200 = 3.9762714903188428338e-14 * x155
                        x201 = x148 * x15
                        x202 = x201 * x6
                        x203 = x105 * x202
                        x204 = x168 * x201
                        x205 = x16 * x201
                        x206 = x11 * x46
                        x207 = x145 * x206
                        x208 = x207 * x73
                        x209 = x7 * x98
                        x210 = 5.5667800864463799674e-14 * x181
                        x211 = x7 * x92
                        x212 = x64 * x88
                        x213 = 5.9644072354782642507e-14 * x147
                        x214 = 5.9644072354782642507e-14 * x117
                        x215 = x147 * x184
                        x216 = 1.1928814470956528501e-13 * x215
                        x217 = 1.6700340259339139902e-13 * x215
                        x218 = 1.6700340259339139902e-13 * x199
                        x219 = x100 * x155
                        x220 = 1.1928814470956528501e-13 * x219
                        x221 = 1.1928814470956528501e-13 * x117
                        x222 = 1.6700340259339139902e-13 * x117 * x57
                        x223 = 1.6700340259339139902e-13 * x208
                        x224 = 2.3857628941913057003e-13 * x61
                        x225 = x15 * x163
                        x226 = x155 * x22
                        x227 = 3.3400680518678279804e-13 * x7
                        x228 = x227 * x61
                        x229 = 3.3400680518678279804e-13 * x201
                        x230 = 4.6760952726149591726e-13 * x7
                        x231 = 4.6760952726149591726e-13 * x148
                        return jnp.asarray(
                            [
                                x26 * x3,
                                -x26 * x28,
                                x31 * x33,
                                -x33 * x34,
                                -x32 * x37,
                                x39 * x42,
                                -x44 * x48,
                                x50 * x55,
                                -x56 * x59,
                                x59 * x62,
                                -x58 * x66,
                                x48 * x67 * x68,
                                -x42 * x67 * x76,
                                x13 * x37 * x81,
                                x25 * x82,
                                -x84 * x85,
                                x38 * x86,
                                -x84 * x89,
                                x92 * x95,
                                -x95 * x98,
                                x101 * x99,
                                -x103 * x86,
                                x107 * x99,
                                -x109 * x83,
                                -x34 * x35 * x81,
                                x112 * x41,
                                -x111 * x47 * x68,
                                x113 * x66,
                                -x114 * x62,
                                x114 * x56,
                                -x110 * x55,
                                x115 * x47,
                                -x116 * x41,
                                x3 * x32 * x82,
                                x109 * x117,
                                -x107 * x119,
                                x103 * x117 * x45,
                                -x101 * x119,
                                x120 * x98,
                                -x120 * x92,
                                x121 * x89,
                                -x122 * x45,
                                x121 * x85,
                                -2.1907831902583156109e-18 * x123 * x25,
                                x123 * x124 * x27 * x32,
                                -x125 * x127 * x128,
                                x113 * x125 * x60,
                                -2.6508476602125618892e-16 * x108 * x130 * x49,
                                -x131 * x132 * x38,
                                x131 * x134 * x43,
                                -x136 * x54,
                                x137 * x138,
                                -x135 * x138 * x62,
                                x136 * x140,
                                -x137 * x141 * x5,
                                x135 * x143 * x144,
                                x132 * x146,
                                -3.9762714903188428338e-15 * x146 * x147,
                                x150 * x17 * x64,
                                -x154 * x91,
                                x154 * x18 * x96,
                                -x150 * x152 * x155 * x96,
                                3.9762714903188428338e-15 * x117 * x128 * x15,
                                -x153 * x156 * x18 * x90,
                                -x143 * x157 * x159,
                                x141 * x15 * x160,
                                -x157 * x161 * x64,
                                1.1133560172892759935e-14 * x108 * x148,
                                -x162 * x163 * x164,
                                x161 * x164 * x70 * x90,
                                -x134 * x165,
                                1.3254238301062809446e-15 * x165 * x166,
                                x156 * x167 * x168,
                                -3.9762714903188428338e-15 * x29 * x36 * x54 * x7,
                                7.9525429806376856677e-15 * x169 * x172,
                                -x162 * x167 * x174,
                                x162 * x169 * x69,
                                -7.9525429806376856677e-15 * x139 * x29 * x49 * x87,
                                x133 * x27 * x29 * x39,
                                -x132 * x29 * x44 * x52,
                                x175 * x176 * x9,
                                -x175 * x177 * x71 * x74,
                                6.6271191505314047231e-15 * x170 * x178 * x97,
                                -6.6271191505314047231e-15 * x179 * x181,
                                -x182 * x183,
                                x186 * x51,
                                -x187 * x188,
                                x188 * x189,
                                -x186 * x190,
                                x182 * x191,
                                x112 * x14 * x192,
                                -3.9762714903188428338e-14 * x159 * x193 * x61,
                                x194 * x195,
                                -x174 * x195,
                                3.9762714903188428338e-14 * x171 * x178 * x63,
                                -x144 * x193 * x196,
                                -x197 * x199,
                                x200 * x203,
                                -5.5667800864463799674e-14 * x148 * x179,
                                5.5667800864463799674e-14 * x170 * x204,
                                -3.9762714903188428338e-14 * x106 * x205,
                                x197 * x208,
                                x102 * x196 * x46 * x7,
                                -x100 * x181 * x200,
                                x209 * x210,
                                -x210 * x211,
                                3.9762714903188428338e-14 * x180 * x212,
                                -x122 * x192,
                                x183 * x213,
                                -x191 * x213,
                                x214 * x5 * x53 * x56,
                                -x206 * x214 * x51 * x7,
                                -x216 * x51,
                                x187 * x217,
                                -x189 * x217,
                                x190 * x216,
                                1.1928814470956528501e-13 * x198 * x64 * x87,
                                -x211 * x218,
                                x209 * x218,
                                -x199 * x220,
                                -x140 * x221,
                                x222 * x62,
                                -x222 * x56,
                                x221 * x54,
                                x208 * x220,
                                -x209 * x223,
                                x211 * x223,
                                -1.1928814470956528501e-13 * x207 * x212,
                                x224 * x225 * x90,
                                -x148 * x224 * x64,
                                2.3857628941913057003e-13 * x202 * x226 * x63 * x90,
                                -2.3857628941913057003e-13 * x100 * x205,
                                -x21 * x225 * x228,
                                x163 * x228 * x90,
                                x105 * x226 * x229,
                                -x203 * x22 * x227,
                                -3.3400680518678279804e-13 * x148 * x219,
                                x172 * x229 * x63,
                                x229 * x98,
                                -x229 * x92,
                                x194 * x201 * x21 * x230,
                                -x211 * x231,
                                x209 * x231,
                                -x173 * x204 * x230,
                            ]
                        )

                case 169:

                    def shape_functions(xi):
                        x0 = xi[0] - 1.0
                        x1 = xi[1] - 1.0
                        x2 = 2.0 * xi[1]
                        x3 = x2 + 1.0
                        x4 = 3.0 * xi[1]
                        x5 = x4 + 1.0
                        x6 = 6.0 * xi[1]
                        x7 = x6 + 1.0
                        x8 = x4 + 2.0
                        x9 = x6 + 5.0
                        x10 = x2 - 1.0
                        x11 = x4 - 1.0
                        x12 = x6 - 1.0
                        x13 = x4 - 2.0
                        x14 = x6 - 5.0
                        x15 = (
                            x10 * x11 * x12 * x13 * x14 * x3 * x5 * x7 * x8 * x9 * xi[1]
                        )
                        x16 = x1 * x15
                        x17 = 2.0 * xi[0]
                        x18 = x17 + 1.0
                        x19 = 3.0 * xi[0]
                        x20 = x19 + 1.0
                        x21 = 6.0 * xi[0]
                        x22 = x21 + 1.0
                        x23 = x19 + 2.0
                        x24 = x21 + 5.0
                        x25 = x17 - 1.0
                        x26 = x19 - 1.0
                        x27 = x21 - 1.0
                        x28 = x19 - 2.0
                        x29 = x21 - 5.0
                        x30 = (
                            x18
                            * x20
                            * x22
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * xi[0]
                        )
                        x31 = 1.1712674050336387999e-10 * x30
                        x32 = x16 * x31
                        x33 = xi[0] + 1.0
                        x34 = xi[1] + 1.0
                        x35 = x15 * x31 * x34
                        x36 = x16 * x33
                        x37 = x0 * x18 * x20 * x22 * x23 * x25 * x26 * x27 * x28 * xi[0]
                        x38 = 8.4331253162421993591e-9 * x37
                        x39 = x36 * x38
                        x40 = x24 * x36
                        x41 = x0 * x18 * x20 * x22 * x25 * x26 * x27 * x29 * xi[0]
                        x42 = 2.3191094619666048237e-8 * x41
                        x43 = x40 * x42
                        x44 = x23 * x40
                        x45 = x0 * x20 * x22 * x26 * x27 * x28 * x29 * xi[0]
                        x46 = 5.1535765821480107194e-8 * x45
                        x47 = x44 * x46
                        x48 = x18 * x44
                        x49 = x0 * x22 * x25 * x27 * x28 * x29 * xi[0]
                        x50 = 1.7393320964749536178e-7 * x49
                        x51 = x48 * x50
                        x52 = x0 * x25 * x26 * x28 * x29 * xi[0]
                        x53 = x27 * x52
                        x54 = 5.565862708719851577e-7 * x20
                        x55 = x48 * x54
                        x56 = (
                            4199.04 * xi[0] ** 12
                            - 10614.24 * xi[0] ** 10
                            + 9729.72 * xi[0] ** 8
                            - 4002.57 * xi[0] ** 6
                            + 740.74 * xi[0] ** 4
                            - 53.69 * xi[0] ** 2
                            + 1.0
                        )
                        x57 = x1 * x56
                        x58 = 0.000010822510822510822511 * x15
                        x59 = x22 * x52
                        x60 = x10 * x11 * x12 * x13 * x14 * x3 * x34 * x5 * x7 * xi[1]
                        x61 = x1 * x30
                        x62 = x33 * x61
                        x63 = x60 * x62
                        x64 = 8.4331253162421993591e-9 * x8
                        x65 = 2.3191094619666048237e-8 * x9
                        x66 = x10 * x11 * x12 * x13 * x34 * x5 * x7 * x9 * xi[1]
                        x67 = x62 * x66
                        x68 = x14 * x8
                        x69 = 5.1535765821480107194e-8 * x68
                        x70 = x10 * x11 * x12 * x34 * x68 * x7 * xi[1]
                        x71 = x3 * x62
                        x72 = x70 * x71
                        x73 = x13 * x9
                        x74 = 1.7393320964749536178e-7 * x73
                        x75 = x11 * x12 * x34 * x73 * xi[1]
                        x76 = x10 * x68
                        x77 = x75 * x76
                        x78 = x5 * x71
                        x79 = 5.565862708719851577e-7 * x78
                        x80 = (
                            4199.04 * xi[1] ** 12
                            - 10614.24 * xi[1] ** 10
                            + 9729.72 * xi[1] ** 8
                            - 4002.57 * xi[1] ** 6
                            + 740.74 * xi[1] ** 4
                            - 53.69 * xi[1] ** 2
                            + 1.0
                        )
                        x81 = x33 * x80
                        x82 = 0.000010822510822510822511 * x30
                        x83 = x11 * x73
                        x84 = x34 * x76 * xi[1]
                        x85 = x7 * x84
                        x86 = x83 * x85
                        x87 = x12 * x85
                        x88 = x74 * x87
                        x89 = x7 * x75
                        x90 = x69 * x89
                        x91 = x5 * x65
                        x92 = x3 * x64
                        x93 = x24 * x33
                        x94 = x15 * x34 * x93
                        x95 = x23 * x94
                        x96 = x18 * x95
                        x97 = x50 * x96
                        x98 = x54 * x96
                        x99 = x29 * x33
                        x100 = x0 * x61
                        x101 = x100 * x66
                        x102 = x100 * x3
                        x103 = x102 * x70
                        x104 = x102 * x5
                        x105 = 5.565862708719851577e-7 * x104
                        x106 = x100 * x60
                        x107 = x1 * x37
                        x108 = x107 * x99
                        x109 = x60 * x8
                        x110 = 6.0718502276943835385e-7 * x109
                        x111 = x107 * x93
                        x112 = x111 * x66
                        x113 = x3 * x8
                        x114 = 6.0718502276943835385e-7 * x113
                        x115 = x108 * x66
                        x116 = x1 * x93
                        x117 = x109 * x116
                        x118 = 1.6697588126159554731e-6 * x41
                        x119 = x117 * x118
                        x120 = x117 * x23
                        x121 = 3.710575139146567718e-6 * x45
                        x122 = x120 * x121
                        x123 = x120 * x18
                        x124 = 0.000012523191094619666048 * x49
                        x125 = x123 * x124
                        x126 = 0.000040074211502782931354 * x20
                        x127 = x123 * x126
                        x128 = x57 * x60
                        x129 = 0.00077922077922077922078 * x8
                        x130 = 1.6697588126159554731e-6 * x9
                        x131 = x111 * x130
                        x132 = 3.710575139146567718e-6 * x68
                        x133 = x111 * x3
                        x134 = 0.000012523191094619666048 * x73
                        x135 = x133 * x134
                        x136 = x133 * x5
                        x137 = 0.000040074211502782931354 * x136
                        x138 = x24 * x81
                        x139 = 0.00077922077922077922078 * x37
                        x140 = x5 * x87
                        x141 = x132 * x89
                        x142 = x3 * x5
                        x143 = x142 * x70
                        x144 = x116 * x23
                        x145 = x144 * x66
                        x146 = x113 * x145
                        x147 = x146 * x18
                        x148 = x124 * x147
                        x149 = x126 * x147
                        x150 = x3 * x57
                        x151 = x116 * x28
                        x152 = x151 * x66
                        x153 = x108 * x130
                        x154 = x108 * x142
                        x155 = x140 * x3
                        x156 = x108 * x134
                        x157 = 0.000040074211502782931354 * x154
                        x158 = x3 * x70
                        x159 = x151 * x41
                        x160 = x60 * x9
                        x161 = 4.591836734693877551e-6 * x160
                        x162 = x144 * x41
                        x163 = x143 * x9
                        x164 = 4.591836734693877551e-6 * x163
                        x165 = x25 * x45
                        x166 = x144 * x160
                        x167 = 0.000010204081632653061224 * x166
                        x168 = x166 * x18
                        x169 = 0.000034438775510204081633 * x49
                        x170 = x168 * x169
                        x171 = 0.00011020408163265306122 * x20
                        x172 = x168 * x171
                        x173 = 0.0021428571428571428571 * x9
                        x174 = x18 * x45
                        x175 = 0.000010204081632653061224 * x68
                        x176 = x175 * x41
                        x177 = 0.000034438775510204081633 * x73
                        x178 = x162 * x177
                        x179 = x142 * x162
                        x180 = 0.00011020408163265306122 * x179
                        x181 = x138 * x23
                        x182 = 0.0021428571428571428571 * x41
                        x183 = x175 * x89
                        x184 = x144 * x163
                        x185 = 0.000010204081632653061224 * x184
                        x186 = x18 * x184
                        x187 = x169 * x186
                        x188 = x171 * x186
                        x189 = x150 * x5
                        x190 = x142 * x159
                        x191 = x159 * x177
                        x192 = 0.00011020408163265306122 * x190
                        x193 = 0.000022675736961451247166 * x68
                        x194 = x145 * x193
                        x195 = x144 * x174
                        x196 = x142 * x195
                        x197 = x193 * x89
                        x198 = x142 * x144
                        x199 = x165 * x198
                        x200 = x26 * x49
                        x201 = x18 * x68
                        x202 = x145 * x201
                        x203 = 0.000076530612244897959184 * x202
                        x204 = 0.00024489795918367346939 * x20
                        x205 = x202 * x204
                        x206 = 0.0047619047619047619048 * x68
                        x207 = x20 * x49
                        x208 = 0.000076530612244897959184 * x73
                        x209 = x195 * x208
                        x210 = 0.00024489795918367346939 * x196
                        x211 = x18 * x181
                        x212 = 0.0047619047619047619048 * x45
                        x213 = x198 * x207
                        x214 = x201 * x89
                        x215 = 0.000076530612244897959184 * x214
                        x216 = x198 * x59
                        x217 = x204 * x214
                        x218 = x189 * x7
                        x219 = x198 * x53
                        x220 = x198 * x200
                        x221 = x144 * x155
                        x222 = x165 * x208
                        x223 = 0.00024489795918367346939 * x199
                        x224 = x144 * x158
                        x225 = x18 * x73
                        x226 = 0.00025829081632653061224 * x225
                        x227 = x224 * x226
                        x228 = x221 * x226
                        x229 = 0.00082653061224489795918 * x20 * x225
                        x230 = x224 * x229
                        x231 = 0.016071428571428571429 * x73
                        x232 = 0.00082653061224489795918 * x18
                        x233 = x213 * x232
                        x234 = x20 * x211
                        x235 = 0.016071428571428571429 * x49
                        x236 = x221 * x229
                        x237 = x218 * x84
                        x238 = x220 * x232
                        x239 = 0.0026448979591836734694 * x18 * x20
                        x240 = x239 * x77
                        x241 = x239 * x86
                        x242 = 0.051428571428571428571 * x234
                        return jnp.asarray(
                            [
                                x0 * x32,
                                x32 * x33,
                                x33 * x35,
                                x0 * x35,
                                -x29 * x39,
                                x28 * x43,
                                -x25 * x47,
                                x26 * x51,
                                -x53 * x55,
                                x57 * x58,
                                -x55 * x59,
                                x20 * x51,
                                -x18 * x47,
                                x23 * x43,
                                -x24 * x39,
                                -x63 * x64,
                                x63 * x65,
                                -x67 * x69,
                                x72 * x74,
                                -x77 * x79,
                                x81 * x82,
                                -x79 * x86,
                                x78 * x88,
                                -x78 * x90,
                                x72 * x91,
                                -x67 * x92,
                                -x38 * x94,
                                x42 * x95,
                                -x46 * x96,
                                x20 * x97,
                                -x59 * x98,
                                x34 * x56 * x58,
                                -x53 * x98,
                                x26 * x97,
                                -x25 * x46 * x95,
                                x28 * x42 * x94,
                                -x15 * x34 * x38 * x99,
                                -x101 * x92,
                                x103 * x91,
                                -x104 * x90,
                                x104 * x88,
                                -x105 * x86,
                                x0 * x80 * x82,
                                -x105 * x77,
                                x103 * x74,
                                -x101 * x69,
                                x106 * x65,
                                -x106 * x64,
                                x108 * x110,
                                x110 * x111,
                                x112 * x114,
                                x114 * x115,
                                -x119 * x28,
                                x122 * x25,
                                -x125 * x26,
                                x127 * x53,
                                -x128 * x129,
                                x127 * x59,
                                -x125 * x20,
                                x122 * x18,
                                -x119 * x23,
                                -x131 * x60,
                                x112 * x132,
                                -x135 * x70,
                                x137 * x77,
                                -x138 * x139,
                                x137 * x86,
                                -x135 * x140,
                                x136 * x141,
                                -x131 * x143,
                                -x118 * x146,
                                x121 * x147,
                                -x148 * x20,
                                x149 * x59,
                                -x129 * x150 * x66,
                                x149 * x53,
                                -x148 * x26,
                                x121 * x146 * x25,
                                -x113 * x118 * x152,
                                -x143 * x153,
                                x141 * x154,
                                -x155 * x156,
                                x157 * x86,
                                -x139 * x29 * x81,
                                x157 * x77,
                                -x156 * x158,
                                x115 * x132,
                                -x153 * x60,
                                x159 * x161,
                                x161 * x162,
                                x162 * x164,
                                x159 * x164,
                                -x165 * x167,
                                x170 * x26,
                                -x172 * x53,
                                x128 * x173,
                                -x172 * x59,
                                x170 * x20,
                                -x167 * x174,
                                -x145 * x176,
                                x158 * x178,
                                -x180 * x77,
                                x181 * x182,
                                -x180 * x86,
                                x155 * x178,
                                -x179 * x183,
                                -x174 * x185,
                                x187 * x20,
                                -x188 * x59,
                                x173 * x189 * x70,
                                -x188 * x53,
                                x187 * x26,
                                -x165 * x185,
                                -x183 * x190,
                                x155 * x191,
                                -x192 * x86,
                                x138 * x182 * x28,
                                -x192 * x77,
                                x158 * x191,
                                -x152 * x176,
                                x165 * x194,
                                x174 * x194,
                                x196 * x197,
                                x197 * x199,
                                -x200 * x203,
                                x205 * x53,
                                -x206 * x57 * x66,
                                x205 * x59,
                                -x203 * x207,
                                -x158 * x209,
                                x210 * x77,
                                -x211 * x212,
                                x210 * x86,
                                -x155 * x209,
                                -x213 * x215,
                                x216 * x217,
                                -x206 * x218 * x75,
                                x217 * x219,
                                -x215 * x220,
                                -x221 * x222,
                                x223 * x86,
                                -x181 * x212 * x25,
                                x223 * x77,
                                -x222 * x224,
                                x200 * x227,
                                x207 * x227,
                                x207 * x228,
                                x200 * x228,
                                -x230 * x53,
                                x150 * x231 * x70,
                                -x230 * x59,
                                -x233 * x77,
                                x234 * x235,
                                -x233 * x86,
                                -x236 * x59,
                                x12 * x231 * x237,
                                -x236 * x53,
                                -x238 * x86,
                                x211 * x235 * x26,
                                -x238 * x77,
                                x219 * x240,
                                x216 * x240,
                                x216 * x241,
                                x219 * x241,
                                -0.051428571428571428571 * x189 * x77,
                                -x242 * x59,
                                -0.051428571428571428571 * x237 * x83,
                                -x242 * x53,
                                x56 * x80,
                            ]
                        )

                case 196:

                    def shape_functions(xi):
                        x0 = 13.0 * xi[1]
                        x1 = x0 + 11.0
                        x2 = xi[0] - 1.0
                        x3 = x1 * x2
                        x4 = 13.0 * xi[0]
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
                        x29 = (
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
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x5
                            * x6
                            * x7
                            * x8
                            * x9
                        )
                        x30 = 6.4945069302692935024e-26 * x29
                        x31 = xi[0] + 1.0
                        x32 = x1 * x31
                        x33 = xi[1] + 1.0
                        x34 = x15 * x33
                        x35 = x32 * x34
                        x36 = (
                            x10
                            * x11
                            * x12
                            * x13
                            * x14
                            * x17
                            * x18
                            * x19
                            * x20
                            * x21
                            * x22
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x5
                            * x6
                            * x7
                            * x8
                            * x9
                        )
                        x37 = 6.4945069302692935024e-26 * x36
                        x38 = x3 * x34
                        x39 = 1.0975716712155106019e-23 * x31
                        x40 = x16 * x3
                        x41 = x39 * x40
                        x42 = (
                            x10
                            * x11
                            * x14
                            * x17
                            * x18
                            * x19
                            * x20
                            * x21
                            * x22
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x5
                            * x6
                            * x7
                            * x8
                            * x9
                        )
                        x43 = x40 * x42
                        x44 = x12 * x31
                        x45 = 6.5854300272930636114e-23 * x44
                        x46 = x15 * x45
                        x47 = (
                            x10
                            * x17
                            * x18
                            * x19
                            * x20
                            * x21
                            * x22
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x5
                            * x6
                            * x7
                            * x8
                            * x9
                        )
                        x48 = x40 * x47
                        x49 = 2.4146576766741233242e-22 * x13
                        x50 = x14 * x44
                        x51 = x49 * x50
                        x52 = x15 * x51
                        x53 = x24 * x40
                        x54 = x15 * x53
                        x55 = (
                            x10
                            * x17
                            * x18
                            * x19
                            * x20
                            * x21
                            * x22
                            * x23
                            * x25
                            * x26
                            * x27
                            * x28
                            * x5
                            * x6
                            * x7
                            * x8
                        )
                        x56 = x11 * x13
                        x57 = x50 * x56
                        x58 = x55 * x57
                        x59 = 6.0366441916853083105e-22 * x58
                        x60 = (
                            x10
                            * x17
                            * x18
                            * x19
                            * x20
                            * x21
                            * x22
                            * x25
                            * x26
                            * x27
                            * x28
                            * x5
                            * x6
                            * x8
                            * x9
                        )
                        x61 = x23 * x57
                        x62 = x54 * x61
                        x63 = 1.0865959545033554959e-21 * x62
                        x64 = (
                            x10
                            * x17
                            * x18
                            * x19
                            * x20
                            * x21
                            * x22
                            * x25
                            * x26
                            * x6
                            * x7
                            * x8
                            * x9
                        )
                        x65 = x27 * x28
                        x66 = x62 * x65
                        x67 = 1.4487946060044739945e-21 * x66
                        x68 = x10 * x18 * x19 * x20 * x21 * x22 * x5 * x6 * x7 * x8 * x9
                        x69 = x25 * x26
                        x70 = x68 * x69
                        x71 = x10 * x18 * x20 * x22 * x5 * x6 * x69 * x7 * x8 * x9
                        x72 = x17 * x65
                        x73 = x21 * x72
                        x74 = x71 * x73
                        x75 = 6.0366441916853083105e-22 * x19
                        x76 = x17 * x71
                        x77 = x60 * x7
                        x78 = x11 * x77
                        x79 = x23 * x56
                        x80 = x14 * x79
                        x81 = x26 * x68
                        x82 = x72 * x81
                        x83 = x80 * x82
                        x84 = x12 * x80
                        x85 = x24 * x64
                        x86 = x5 * x85
                        x87 = x28 * x86
                        x88 = x84 * x87
                        x89 = x33 * x39
                        x90 = x16 * x35
                        x91 = x12 * x90
                        x92 = 6.5854300272930636114e-23 * x47 * x56
                        x93 = x42 * x49
                        x94 = x18 * x20 * x22 * x5 * x6 * x69 * x7 * x9
                        x95 = x73 * x8
                        x96 = x94 * x95
                        x97 = x84 * x90
                        x98 = x24 * x75
                        x99 = x97 * x98
                        x100 = x73 * x94
                        x101 = x24 * x97
                        x102 = x10 * x19
                        x103 = x101 * x102
                        x104 = 1.0865959545033554959e-21 * x103
                        x105 = x18 * x95
                        x106 = x20 * x22 * x5 * x69 * x7 * x9
                        x107 = 1.4487946060044739945e-21 * x106
                        x108 = x103 * x107
                        x109 = x6 * x95
                        x110 = x6 * x69
                        x111 = x22 * x5 * x7 * x9
                        x112 = x105 * x111
                        x113 = x110 * x112
                        x114 = x10 * x20
                        x115 = x110 * x9
                        x116 = x5 * x7
                        x117 = x105 * x116
                        x118 = x115 * x117
                        x119 = x114 * x118
                        x120 = x55 * x9
                        x121 = x11 * x120
                        x122 = x121 * x14 * x49
                        x123 = x25 * x68
                        x124 = x123 * x72
                        x125 = 6.5854300272930636114e-23 * x124
                        x126 = x27 * x86
                        x127 = 1.0975716712155106019e-23 * x126
                        x128 = x24 * x38
                        x129 = x128 * x83
                        x130 = x38 * x61
                        x131 = x71 * x72
                        x132 = x130 * x131
                        x133 = x128 * x61
                        x134 = 1.0865959545033554959e-21 * x133
                        x135 = 1.4487946060044739945e-21 * x65
                        x136 = x133 * x70
                        x137 = x130 * x85
                        x138 = x38 * x47
                        x139 = x38 * x42
                        x140 = x16 * x84
                        x141 = x140 * x38
                        x142 = x128 * x140
                        x143 = x16 * x38
                        x144 = x12 * x143
                        x145 = x141 * x98
                        x146 = x102 * x142
                        x147 = 1.0865959545033554959e-21 * x146
                        x148 = x107 * x146
                        x149 = x2 * x33
                        x150 = 1.8548961243542129172e-21 * x16
                        x151 = x2 * x34
                        x152 = x151 * x61
                        x153 = x5 * x64
                        x154 = x33 * x53 * x61
                        x155 = x151 * x16
                        x156 = 1.1129376746125277503e-20 * x44
                        x157 = 4.0807714735792684179e-20 * x155
                        x158 = x13 * x50
                        x159 = 1.0201928683948171045e-19 * x24
                        x160 = x152 * x16
                        x161 = x160 * x24
                        x162 = 1.836347163110670788e-19 * x161
                        x163 = 2.4484628841475610507e-19 * x65
                        x164 = x131 * x19
                        x165 = x57 * x77
                        x166 = x143 * x87
                        x167 = 4.0807714735792684179e-20 * x31
                        x168 = 1.0201928683948171045e-19 * x19
                        x169 = x21 * x8
                        x170 = x16 * x28
                        x171 = x133 * x17
                        x172 = x170 * x171
                        x173 = x172 * x94
                        x174 = 1.836347163110670788e-19 * x102
                        x175 = x169 * x18
                        x176 = x172 * x175
                        x177 = x102 * x106
                        x178 = 2.4484628841475610507e-19 * x177
                        x179 = x169 * x6
                        x180 = x110 * x111
                        x181 = x115 * x20
                        x182 = 1.0201928683948171045e-19 * x102
                        x183 = x116 * x182
                        x184 = x16 * x27
                        x185 = x171 * x184
                        x186 = x126 * x143
                        x187 = 4.0807714735792684179e-20 * x186
                        x188 = x133 * x184 * x76
                        x189 = 2.4484628841475610507e-19 * x184
                        x190 = x22 * x5
                        x191 = x175 * x181
                        x192 = x20 * x22
                        x193 = x110 * x192
                        x194 = x158 * x23
                        x195 = x11 * x23 * x50
                        x196 = x118 * x20
                        x197 = x154 * x174
                        x198 = x154 * x178
                        x199 = 6.677626047675166502e-20 * x16
                        x200 = x199 * x44
                        x201 = x128 * x79
                        x202 = x133 * x72
                        x203 = x124 * x128
                        x204 = x16 * x44
                        x205 = 2.4484628841475610507e-19 * x204
                        x206 = x13 * x138
                        x207 = 6.1211572103689026268e-19 * x204
                        x208 = x128 * x56
                        x209 = x208 * x55
                        x210 = x201 * x204
                        x211 = 1.1018082978664024728e-18 * x210
                        x212 = 1.4690777304885366304e-18 * x65
                        x213 = x143 * x85
                        x214 = x208 * x77
                        x215 = 2.4484628841475610507e-19 * x16
                        x216 = x215 * x31
                        x217 = 6.1211572103689026268e-19 * x19
                        x218 = x20 * x26
                        x219 = x133 * x16
                        x220 = x219 * x6
                        x221 = x112 * x220
                        x222 = x102 * x218
                        x223 = x111 * x222
                        x224 = 1.1018082978664024728e-18 * x220
                        x225 = x18 * x73
                        x226 = 1.4690777304885366304e-18 * x219
                        x227 = 1.1018082978664024728e-18 * x221
                        x228 = x220 * x9
                        x229 = 6.1211572103689026268e-19 * x117
                        x230 = x203 * x215
                        x231 = x102 * x25
                        x232 = x20 * x231
                        x233 = 6.1211572103689026268e-19 * x16
                        x234 = x202 * x8
                        x235 = x18 * x234
                        x236 = x192 * x231
                        x237 = 1.4690777304885366304e-18 * x105
                        x238 = x105 * x190
                        x239 = x238 * x9
                        x240 = x128 * x195
                        x241 = x102 * x240
                        x242 = x113 * x16
                        x243 = 1.1018082978664024728e-18 * x241
                        x244 = x16 * x240
                        x245 = x177 * x244
                        x246 = x100 * x16
                        x247 = x16 * x31
                        x248 = x14 * x247
                        x249 = 8.9776972418743905194e-19 * x248
                        x250 = 8.9776972418743905194e-19 * x143
                        x251 = 4.0399637588434757337e-18 * x128
                        x252 = x247 * x80
                        x253 = x251 * x252
                        x254 = 5.3866183451246343116e-18 * x65
                        x255 = x128 * x252
                        x256 = x254 * x70
                        x257 = x128 * x16
                        x258 = x257 * x57
                        x259 = 2.2444243104685976298e-18 * x19
                        x260 = x259 * x96
                        x261 = x102 * x251
                        x262 = x261 * x57
                        x263 = 5.3866183451246343116e-18 * x177
                        x264 = x258 * x263
                        x265 = 2.2444243104685976298e-18 * x102 * x196
                        x266 = x130 * x16
                        x267 = 4.0399637588434757337e-18 * x266
                        x268 = x194 * x257
                        x269 = x194 * x261
                        x270 = x263 * x268
                        x271 = 5.6110607761714940746e-18 * x19
                        x272 = x117 * x219
                        x273 = x16 * x94
                        x274 = 5.6110607761714940746e-18 * x102
                        x275 = x16 * x235
                        x276 = x110 * x272
                        x277 = x19 * x219
                        x278 = x181 * x277
                        x279 = x22 * x7
                        x280 = 1.3466545862811585779e-17 * x105
                        x281 = 1.3466545862811585779e-17 * x65
                        x282 = 1.0099909397108689334e-17 * x219
                        x283 = 1.0099909397108689334e-17 * x102
                        x284 = 1.3466545862811585779e-17 * x177
                        x285 = x102 * x219
                        x286 = x116 * x285
                        x287 = x181 * x285
                        x288 = x102 * x282
                        x289 = x109 * x69
                        x290 = 1.3466545862811585779e-17 * x192
                        x291 = x225 * x287
                        x292 = x10 * x219
                        x293 = 1.8179836914795640802e-17 * x292
                        x294 = x115 * x285
                        x295 = 2.4239782553060854402e-17 * x279
                        x296 = 2.4239782553060854402e-17 * x285
                        x297 = x296 * x65
                        x298 = 2.4239782553060854402e-17 * x106 * x292
                        x299 = x20 * x296
                        x300 = x289 * x9
                        x301 = 3.231971007074780587e-17 * x192 * x285 * x7
                        x302 = 3.231971007074780587e-17 * x177 * x219 * x65
                        return jnp.asarray(
                            [
                                x3 * x30,
                                -x30 * x32,
                                x35 * x37,
                                -x37 * x38,
                                -x36 * x41,
                                x43 * x46,
                                -x48 * x52,
                                x54 * x59,
                                -x60 * x63,
                                x64 * x67,
                                -x67 * x70,
                                x63 * x74,
                                -x66 * x75 * x76,
                                x52 * x53 * x78,
                                -x46 * x53 * x83,
                                x15 * x41 * x88,
                                x29 * x89,
                                -x91 * x92,
                                x90 * x93,
                                -x96 * x99,
                                x100 * x104,
                                -x105 * x108,
                                x108 * x109,
                                -x104 * x113,
                                x119 * x99,
                                -x122 * x91,
                                x101 * x125,
                                -x127 * x97,
                                -x38 * x39 * x88,
                                x129 * x45,
                                -x128 * x51 * x78,
                                x132 * x98,
                                -x134 * x74,
                                x135 * x136,
                                -x135 * x137,
                                x134 * x60,
                                -x128 * x59,
                                x138 * x51,
                                -x139 * x45,
                                x3 * x36 * x89,
                                x127 * x141,
                                -x125 * x142,
                                x122 * x144,
                                -x119 * x145,
                                x113 * x147,
                                -x109 * x148,
                                x105 * x148,
                                -x100 * x147,
                                x145 * x96,
                                -x143 * x93,
                                x144 * x92,
                                -1.0975716712155106019e-23 * x149 * x29,
                                x149 * x150 * x31 * x36,
                                -x150 * x152 * x87,
                                x130 * x150 * x86,
                                -1.8548961243542129172e-21 * x153 * x154 * x27,
                                -x155 * x156 * x42,
                                x157 * x158 * x47,
                                -x155 * x159 * x58,
                                x162 * x60,
                                -x160 * x163 * x85,
                                x161 * x163 * x70,
                                -x162 * x74,
                                x159 * x160 * x164,
                                -x157 * x165 * x24,
                                1.1129376746125277503e-20 * x161 * x82,
                                x156 * x166 * x79,
                                -x166 * x167 * x80,
                                x168 * x169 * x173,
                                -x173 * x174 * x21,
                                x176 * x178,
                                -x172 * x178 * x179,
                                x174 * x176 * x180,
                                -x176 * x181 * x183,
                                4.0807714735792684179e-20 * x130 * x153 * x170,
                                -1.1129376746125277503e-20 * x123 * x172,
                                -1.1129376746125277503e-20 * x185 * x81,
                                x187 * x57,
                                -x168 * x188,
                                1.836347163110670788e-19 * x188 * x21,
                                -x136 * x189,
                                x137 * x189,
                                -x174 * x185 * x190 * x191,
                                x175 * x183 * x185 * x193,
                                -x187 * x194,
                                1.1129376746125277503e-20 * x186 * x195,
                                1.1129376746125277503e-20 * x124 * x154,
                                -4.0807714735792684179e-20 * x33 * x40 * x58 * x9,
                                x154 * x182 * x196,
                                -x113 * x197,
                                x109 * x198,
                                -x105 * x198,
                                x100 * x197,
                                -x154 * x168 * x96,
                                x13 * x167 * x33 * x43,
                                -x156 * x33 * x48 * x56,
                                x11 * x138 * x200,
                                -x200 * x201 * x82,
                                x199 * x202 * x68,
                                -x195 * x199 * x203,
                                -x205 * x206,
                                x207 * x209,
                                -x211 * x60,
                                x212 * x213 * x44 * x79,
                                -x210 * x212 * x70,
                                x211 * x74,
                                -x164 * x201 * x207,
                                x205 * x214,
                                x129 * x216,
                                -x217 * x218 * x221,
                                x223 * x224 * x225,
                                -x112 * x222 * x226,
                                x109 * x223 * x226,
                                -x102 * x227 * x26,
                                x222 * x228 * x229,
                                -x130 * x215 * x82,
                                -x230 * x57,
                                x111 * x232 * x233 * x235 * x6,
                                -x114 * x227 * x25,
                                x123 * x212 * x219,
                                -x228 * x236 * x237 * x7,
                                x224 * x232 * x239,
                                -x220 * x229 * x236,
                                x194 * x230,
                                2.4484628841475610507e-19 * x121 * x143 * x50,
                                -x196 * x233 * x241,
                                x242 * x243,
                                -1.4690777304885366304e-18 * x109 * x245,
                                x237 * x245,
                                -x243 * x246,
                                x217 * x244 * x96,
                                -x139 * x216,
                                x206 * x249,
                                -x214 * x249,
                                x165 * x250,
                                -x120 * x158 * x250,
                                -2.2444243104685976298e-18 * x209 * x248,
                                x253 * x60,
                                -x213 * x254 * x31 * x80,
                                x255 * x256,
                                -x253 * x74,
                                2.2444243104685976298e-18 * x164 * x255,
                                x258 * x260,
                                -x246 * x262,
                                x105 * x264,
                                -x109 * x264,
                                x242 * x262,
                                -x258 * x265,
                                -x132 * x16 * x259,
                                x267 * x74,
                                -x256 * x266,
                                x254 * x266 * x64,
                                -x267 * x60,
                                2.2444243104685976298e-18 * x143 * x58,
                                x265 * x268,
                                -x242 * x269,
                                x109 * x270,
                                -x105 * x270,
                                x246 * x269,
                                -x260 * x268,
                                x193 * x271 * x272,
                                -x234 * x271 * x273,
                                x116 * x181 * x274 * x275,
                                -x20 * x274 * x276,
                                -1.0099909397108689334e-17 * x238 * x278,
                                x278 * x279 * x280,
                                -x169 * x277 * x281 * x94,
                                x282 * x96,
                                x202 * x273 * x283,
                                -x275 * x284,
                                x16 * x234 * x284 * x6,
                                -x180 * x275 * x283,
                                -x119 * x282,
                                x191 * x281 * x286,
                                -x280 * x287 * x7,
                                x105 * x181 * x288 * x5,
                                x22 * x276 * x283,
                                -x286 * x289 * x290,
                                x102 * x272 * x290 * x69,
                                -x116 * x193 * x225 * x288,
                                1.8179836914795640802e-17 * x190 * x291,
                                -x100 * x293,
                                x113 * x293,
                                -1.8179836914795640802e-17 * x238 * x294,
                                -x291 * x295,
                                x21 * x297 * x94,
                                x105 * x298,
                                -x109 * x298,
                                -x175 * x180 * x297,
                                x105 * x294 * x295,
                                x190 * x299 * x300,
                                -x239 * x299 * x69,
                                x105 * x301 * x69 * x9,
                                -x175 * x302,
                                x179 * x302,
                                -x300 * x301,
                            ]
                        )

                case 225:

                    def shape_functions(xi):
                        x0 = xi[0] - 1.0
                        x1 = xi[1] - 1.0
                        x2 = 7.0 * xi[1]
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
                        x15 = (
                            x10
                            * x11
                            * x12
                            * x13
                            * x14
                            * x3
                            * x4
                            * x5
                            * x6
                            * x7
                            * x8
                            * x9
                            * xi[1]
                        )
                        x16 = x1 * x15
                        x17 = 7.0 * xi[0]
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
                        x30 = (
                            x18
                            * x19
                            * x20
                            * x21
                            * x22
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * xi[0]
                        )
                        x31 = 3.1591878896737389867e-19 * x30
                        x32 = x16 * x31
                        x33 = xi[0] + 1.0
                        x34 = xi[1] + 1.0
                        x35 = x15 * x31 * x34
                        x36 = x16 * x33
                        x37 = (
                            x0
                            * x18
                            * x19
                            * x20
                            * x21
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x29
                            * xi[0]
                        )
                        x38 = 3.096004131880264207e-17 * x37
                        x39 = x36 * x38
                        x40 = x22 * x36
                        x41 = (
                            x0
                            * x18
                            * x19
                            * x20
                            * x21
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * xi[0]
                        )
                        x42 = 2.0124026857221717346e-16 * x41
                        x43 = x40 * x42
                        x44 = x23 * x40
                        x45 = x0 * x18 * x19 * x21 * x24 * x25 * x27 * x28 * x29 * xi[0]
                        x46 = 8.0496107428886869382e-16 * x45
                        x47 = x44 * x46
                        x48 = x20 * x44
                        x49 = x0 * x18 * x19 * x24 * x25 * x26 * x28 * x29 * xi[0]
                        x50 = 2.213642954294388908e-15 * x49
                        x51 = x48 * x50
                        x52 = x21 * x48
                        x53 = x0 * x18 * x24 * x26 * x27 * x28 * x29 * xi[0]
                        x54 = 4.427285908588777816e-15 * x53
                        x55 = x52 * x54
                        x56 = x0 * x25 * x26 * x27 * x28 * x29 * xi[0]
                        x57 = x24 * x56
                        x58 = 6.640928862883166724e-15 * x19
                        x59 = x52 * x58
                        x60 = (
                            -26700.013890817901235 * xi[0] ** 14
                            + 76285.753973765432099 * xi[0] ** 12
                            - 82980.21809799382716 * xi[0] ** 10
                            + 43487.464081790123457 * xi[0] ** 8
                            - 11465.29836612654321 * xi[0] ** 6
                            + 1445.3903549382716049 * xi[0] ** 4
                            - 74.078055555555555556 * xi[0] ** 2
                            + 1.0
                        )
                        x61 = x1 * x60
                        x62 = 5.6206653428875651098e-10 * x15
                        x63 = x18 * x56
                        x64 = (
                            x10
                            * x11
                            * x12
                            * x13
                            * x14
                            * x3
                            * x34
                            * x4
                            * x5
                            * x6
                            * x9
                            * xi[1]
                        )
                        x65 = x1 * x30
                        x66 = x33 * x65
                        x67 = x64 * x66
                        x68 = 3.096004131880264207e-17 * x8
                        x69 = 2.0124026857221717346e-16 * x7
                        x70 = (
                            x10 * x11 * x12 * x14 * x3 * x34 * x4 * x6 * x7 * x9 * xi[1]
                        )
                        x71 = x66 * x70
                        x72 = x13 * x8
                        x73 = 8.0496107428886869382e-16 * x72
                        x74 = x10 * x11 * x12 * x3 * x34 * x4 * x72 * x9 * xi[1]
                        x75 = x5 * x66
                        x76 = x74 * x75
                        x77 = x14 * x7
                        x78 = 2.213642954294388908e-15 * x77
                        x79 = x10 * x12 * x3 * x34 * x77 * x9 * xi[1]
                        x80 = x6 * x75
                        x81 = x79 * x80
                        x82 = x11 * x72
                        x83 = 4.427285908588777816e-15 * x82
                        x84 = x10 * x34 * x82 * x9 * xi[1]
                        x85 = x12 * x77
                        x86 = x84 * x85
                        x87 = x4 * x80
                        x88 = 6.640928862883166724e-15 * x87
                        x89 = (
                            -26700.013890817901235 * xi[1] ** 14
                            + 76285.753973765432099 * xi[1] ** 12
                            - 82980.21809799382716 * xi[1] ** 10
                            + 43487.464081790123457 * xi[1] ** 8
                            - 11465.29836612654321 * xi[1] ** 6
                            + 1445.3903549382716049 * xi[1] ** 4
                            - 74.078055555555555556 * xi[1] ** 2
                            + 1.0
                        )
                        x90 = x33 * x89
                        x91 = 5.6206653428875651098e-10 * x30
                        x92 = x10 * x82
                        x93 = x34 * x85 * xi[1]
                        x94 = x3 * x93
                        x95 = x92 * x94
                        x96 = x9 * x94
                        x97 = x83 * x96
                        x98 = x3 * x84
                        x99 = x78 * x98
                        x100 = x4 * x73
                        x101 = x6 * x69
                        x102 = x5 * x68
                        x103 = x22 * x33
                        x104 = x103 * x15 * x34
                        x105 = x104 * x23
                        x106 = x105 * x20
                        x107 = x106 * x21
                        x108 = x107 * x54
                        x109 = x107 * x58
                        x110 = x28 * x33
                        x111 = x0 * x65
                        x112 = x111 * x70
                        x113 = x111 * x5
                        x114 = x113 * x74
                        x115 = x113 * x6
                        x116 = x115 * x79
                        x117 = x115 * x4
                        x118 = 6.640928862883166724e-15 * x117
                        x119 = x111 * x64
                        x120 = x1 * x37
                        x121 = x110 * x120
                        x122 = x64 * x8
                        x123 = 3.0340840492426589229e-15 * x122
                        x124 = x103 * x120
                        x125 = x124 * x70
                        x126 = x5 * x8
                        x127 = 3.0340840492426589229e-15 * x126
                        x128 = x121 * x70
                        x129 = x1 * x103
                        x130 = x122 * x129
                        x131 = 1.9721546320077282999e-14 * x41
                        x132 = x130 * x131
                        x133 = x130 * x23
                        x134 = 7.8886185280309131995e-14 * x45
                        x135 = x133 * x134
                        x136 = x133 * x20
                        x137 = 2.1693700952085011299e-13 * x49
                        x138 = x136 * x137
                        x139 = x136 * x21
                        x140 = 4.3387401904170022597e-13 * x53
                        x141 = x139 * x140
                        x142 = 6.5081102856255033896e-13 * x19
                        x143 = x139 * x142
                        x144 = x61 * x64
                        x145 = 5.5082520360298138076e-8 * x8
                        x146 = 1.9721546320077282999e-14 * x7
                        x147 = x124 * x146
                        x148 = 7.8886185280309131995e-14 * x72
                        x149 = x124 * x5
                        x150 = 2.1693700952085011299e-13 * x77
                        x151 = x149 * x150
                        x152 = x149 * x6
                        x153 = x152 * x79
                        x154 = 4.3387401904170022597e-13 * x82
                        x155 = x152 * x4
                        x156 = 6.5081102856255033896e-13 * x155
                        x157 = x22 * x90
                        x158 = 5.5082520360298138076e-8 * x37
                        x159 = x154 * x96
                        x160 = x4 * x6
                        x161 = x160 * x98
                        x162 = x5 * x6
                        x163 = x162 * x74
                        x164 = x129 * x23
                        x165 = x164 * x70
                        x166 = x126 * x165
                        x167 = x166 * x20
                        x168 = x167 * x21
                        x169 = x140 * x168
                        x170 = x142 * x168
                        x171 = x5 * x61
                        x172 = x129 * x29
                        x173 = x172 * x70
                        x174 = x121 * x146
                        x175 = x121 * x5
                        x176 = x160 * x175
                        x177 = x150 * x175
                        x178 = 6.5081102856255033896e-13 * x176
                        x179 = x162 * x79
                        x180 = x172 * x41
                        x181 = x64 * x7
                        x182 = 1.2819005108050233949e-13 * x181
                        x183 = x164 * x41
                        x184 = x163 * x7
                        x185 = 1.2819005108050233949e-13 * x184
                        x186 = x26 * x45
                        x187 = x164 * x181
                        x188 = 5.1276020432200935797e-13 * x187
                        x189 = x187 * x20
                        x190 = 1.4100905618855257344e-12 * x49
                        x191 = x189 * x190
                        x192 = x189 * x21
                        x193 = 2.8201811237710514688e-12 * x53
                        x194 = x192 * x193
                        x195 = 4.2302716856565772032e-12 * x19
                        x196 = x192 * x195
                        x197 = 3.5803638234193789749e-7 * x7
                        x198 = x20 * x45
                        x199 = 5.1276020432200935797e-13 * x72
                        x200 = x199 * x41
                        x201 = x183 * x5
                        x202 = 1.4100905618855257344e-12 * x77
                        x203 = x201 * x202
                        x204 = 2.8201811237710514688e-12 * x82
                        x205 = x179 * x204
                        x206 = x160 * x201
                        x207 = 4.2302716856565772032e-12 * x206
                        x208 = x157 * x23
                        x209 = 3.5803638234193789749e-7 * x41
                        x210 = x204 * x96
                        x211 = x199 * x79
                        x212 = x164 * x184
                        x213 = 5.1276020432200935797e-13 * x212
                        x214 = x20 * x21
                        x215 = x212 * x214
                        x216 = x193 * x215
                        x217 = x195 * x215
                        x218 = x171 * x6
                        x219 = x20 * x27
                        x220 = x180 * x5
                        x221 = x160 * x220
                        x222 = x202 * x220
                        x223 = 4.2302716856565772032e-12 * x221
                        x224 = 2.0510408172880374319e-12 * x72
                        x225 = x165 * x224
                        x226 = x164 * x198
                        x227 = x226 * x5
                        x228 = x160 * x227
                        x229 = x224 * x79
                        x230 = x164 * x186
                        x231 = x160 * x5
                        x232 = x230 * x231
                        x233 = x219 * x49
                        x234 = x165 * x72
                        x235 = 5.6403622475421029376e-12 * x234
                        x236 = x214 * x234
                        x237 = 1.1280724495084205875e-11 * x53
                        x238 = x236 * x237
                        x239 = 1.6921086742626308813e-11 * x19
                        x240 = x236 * x239
                        x241 = 1.43214552936775159e-6 * x72
                        x242 = x214 * x49
                        x243 = 5.6403622475421029376e-12 * x77
                        x244 = x227 * x243
                        x245 = 1.1280724495084205875e-11 * x82
                        x246 = x179 * x245
                        x247 = 1.6921086742626308813e-11 * x228
                        x248 = x20 * x208
                        x249 = 1.43214552936775159e-6 * x45
                        x250 = x245 * x96
                        x251 = x164 * x242
                        x252 = x231 * x251
                        x253 = x72 * x79
                        x254 = 5.6403622475421029376e-12 * x253
                        x255 = x19 * x214
                        x256 = x164 * x231
                        x257 = x253 * x256
                        x258 = x237 * x257
                        x259 = x214 * x239 * x257
                        x260 = x218 * x4
                        x261 = x214 * x25
                        x262 = x233 * x256
                        x263 = x161 * x5
                        x264 = x230 * x243
                        x265 = 1.6921086742626308813e-11 * x232
                        x266 = x5 * x74
                        x267 = x164 * x233
                        x268 = 1.5510996180740783078e-11 * x77
                        x269 = x266 * x268
                        x270 = x263 * x268
                        x271 = x261 * x53
                        x272 = x164 * x77
                        x273 = x266 * x272
                        x274 = 3.1021992361481566157e-11 * x273
                        x275 = 4.6532988542222349235e-11 * x255
                        x276 = x273 * x275
                        x277 = 3.9384002057613168724e-6 * x77
                        x278 = x255 * x53
                        x279 = 3.1021992361481566157e-11 * x82
                        x280 = x179 * x279
                        x281 = 4.6532988542222349235e-11 * x252
                        x282 = x21 * x248
                        x283 = 3.9384002057613168724e-6 * x49
                        x284 = x279 * x96
                        x285 = x263 * x272
                        x286 = 3.1021992361481566157e-11 * x285
                        x287 = x275 * x285
                        x288 = x260 * x3
                        x289 = 4.6532988542222349235e-11 * x262
                        x290 = 6.2043984722963132314e-11 * x82
                        x291 = x271 * x290
                        x292 = x164 * x179
                        x293 = x278 * x290
                        x294 = x256 * x96
                        x295 = x255 * x57
                        x296 = 9.3065977084444698471e-11 * x82
                        x297 = x292 * x296
                        x298 = 7.8768004115226337449e-6 * x82
                        x299 = x255 * x63
                        x300 = 9.3065977084444698471e-11 * x256
                        x301 = x278 * x300
                        x302 = x19 * x282
                        x303 = 7.8768004115226337449e-6 * x53
                        x304 = x294 * x296
                        x305 = x288 * x93
                        x306 = x271 * x300
                        x307 = 1.3959896562666704771e-10 * x256
                        x308 = x307 * x86
                        x309 = x307 * x95
                        x310 = 0.000011815200617283950617 * x302
                        return jnp.asarray(
                            [
                                x0 * x32,
                                x32 * x33,
                                x33 * x35,
                                x0 * x35,
                                -x28 * x39,
                                x29 * x43,
                                -x26 * x47,
                                x27 * x51,
                                -x25 * x55,
                                x57 * x59,
                                x61 * x62,
                                x59 * x63,
                                -x19 * x55,
                                x21 * x51,
                                -x20 * x47,
                                x23 * x43,
                                -x22 * x39,
                                -x67 * x68,
                                x67 * x69,
                                -x71 * x73,
                                x76 * x78,
                                -x81 * x83,
                                x86 * x88,
                                x90 * x91,
                                x88 * x95,
                                -x87 * x97,
                                x87 * x99,
                                -x100 * x81,
                                x101 * x76,
                                -x102 * x71,
                                -x104 * x38,
                                x105 * x42,
                                -x106 * x46,
                                x107 * x50,
                                -x108 * x19,
                                x109 * x63,
                                x34 * x60 * x62,
                                x109 * x57,
                                -x108 * x25,
                                x106 * x27 * x50,
                                -x105 * x26 * x46,
                                x104 * x29 * x42,
                                -x110 * x15 * x34 * x38,
                                -x102 * x112,
                                x101 * x114,
                                -x100 * x116,
                                x117 * x99,
                                -x117 * x97,
                                x118 * x95,
                                x0 * x89 * x91,
                                x118 * x86,
                                -x116 * x83,
                                x114 * x78,
                                -x112 * x73,
                                x119 * x69,
                                -x119 * x68,
                                x121 * x123,
                                x123 * x124,
                                x125 * x127,
                                x127 * x128,
                                -x132 * x29,
                                x135 * x26,
                                -x138 * x27,
                                x141 * x25,
                                -x143 * x57,
                                -x144 * x145,
                                -x143 * x63,
                                x141 * x19,
                                -x138 * x21,
                                x135 * x20,
                                -x132 * x23,
                                -x147 * x64,
                                x125 * x148,
                                -x151 * x74,
                                x153 * x154,
                                -x156 * x86,
                                -x157 * x158,
                                -x156 * x95,
                                x155 * x159,
                                -x151 * x161,
                                x148 * x153 * x4,
                                -x147 * x163,
                                -x131 * x166,
                                x134 * x167,
                                -x137 * x168,
                                x169 * x19,
                                -x170 * x63,
                                -x145 * x171 * x70,
                                -x170 * x57,
                                x169 * x25,
                                -x137 * x167 * x27,
                                x134 * x166 * x26,
                                -x126 * x131 * x173,
                                -x163 * x174,
                                x148 * x176 * x79,
                                -x161 * x177,
                                x159 * x176,
                                -x178 * x95,
                                -x158 * x28 * x90,
                                -x178 * x86,
                                x121 * x154 * x179,
                                -x177 * x74,
                                x128 * x148,
                                -x174 * x64,
                                x180 * x182,
                                x182 * x183,
                                x183 * x185,
                                x180 * x185,
                                -x186 * x188,
                                x191 * x27,
                                -x194 * x25,
                                x196 * x57,
                                x144 * x197,
                                x196 * x63,
                                -x19 * x194,
                                x191 * x21,
                                -x188 * x198,
                                -x165 * x200,
                                x203 * x74,
                                -x183 * x205,
                                x207 * x86,
                                x208 * x209,
                                x207 * x95,
                                -x206 * x210,
                                x161 * x203,
                                -x206 * x211,
                                -x198 * x213,
                                x190 * x215,
                                -x19 * x216,
                                x217 * x63,
                                x197 * x218 * x74,
                                x217 * x57,
                                -x216 * x25,
                                x190 * x212 * x219,
                                -x186 * x213,
                                -x211 * x221,
                                x161 * x222,
                                -x210 * x221,
                                x223 * x95,
                                x157 * x209 * x29,
                                x223 * x86,
                                -x180 * x205,
                                x222 * x74,
                                -x173 * x200,
                                x186 * x225,
                                x198 * x225,
                                x228 * x229,
                                x229 * x232,
                                -x233 * x235,
                                x238 * x25,
                                -x240 * x57,
                                -x241 * x61 * x70,
                                -x240 * x63,
                                x19 * x238,
                                -x235 * x242,
                                -x244 * x74,
                                x226 * x246,
                                -x247 * x86,
                                -x248 * x249,
                                -x247 * x95,
                                x228 * x250,
                                -x161 * x244,
                                -x252 * x254,
                                x255 * x258,
                                -x259 * x63,
                                -x241 * x260 * x79,
                                -x259 * x57,
                                x258 * x261,
                                -x254 * x262,
                                -x263 * x264,
                                x232 * x250,
                                -x265 * x95,
                                -x208 * x249 * x26,
                                -x265 * x86,
                                x230 * x246,
                                -x264 * x266,
                                x267 * x269,
                                x251 * x269,
                                x251 * x270,
                                x267 * x270,
                                -x271 * x274,
                                x276 * x57,
                                x171 * x277 * x74,
                                x276 * x63,
                                -x274 * x278,
                                -x251 * x280,
                                x281 * x86,
                                x282 * x283,
                                x281 * x95,
                                -x252 * x284,
                                -x278 * x286,
                                x287 * x63,
                                x277 * x288 * x84,
                                x287 * x57,
                                -x271 * x286,
                                -x262 * x284,
                                x289 * x95,
                                x248 * x27 * x283,
                                x289 * x86,
                                -x267 * x280,
                                x291 * x292,
                                x292 * x293,
                                x293 * x294,
                                x291 * x294,
                                -x295 * x297,
                                -x218 * x298 * x79,
                                -x297 * x299,
                                -x301 * x86,
                                -x302 * x303,
                                -x301 * x95,
                                -x299 * x304,
                                -x298 * x305 * x9,
                                -x295 * x304,
                                -x306 * x95,
                                -x25 * x282 * x303,
                                -x306 * x86,
                                x295 * x308,
                                x299 * x308,
                                x299 * x309,
                                x295 * x309,
                                0.000011815200617283950617 * x260 * x86,
                                x310 * x63,
                                0.000011815200617283950617 * x305 * x92,
                                x310 * x57,
                                x60 * x89,
                            ]
                        )

                case 256:

                    def shape_functions(xi):
                        x0 = 15.0 * xi[1]
                        x1 = x0 + 13.0
                        x2 = xi[0] - 1.0
                        x3 = x1 * x2
                        x4 = 3.0 * xi[0]
                        x5 = x4 + 1.0
                        x6 = 3.0 * xi[1]
                        x7 = x6 + 1.0
                        x8 = 5.0 * xi[0]
                        x9 = x8 + 1.0
                        x10 = 5.0 * xi[1]
                        x11 = x10 + 1.0
                        x12 = 15.0 * xi[0]
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
                        x37 = (
                            x11
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
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x31
                            * x32
                            * x33
                            * x34
                            * x35
                            * x36
                            * x5
                            * x7
                            * x9
                        )
                        x38 = 5.0249700898396030198e-25 * x37
                        x39 = xi[0] + 1.0
                        x40 = x1 * x39
                        x41 = xi[1] + 1.0
                        x42 = x21 * x41
                        x43 = x40 * x42
                        x44 = (
                            x11
                            * x13
                            * x14
                            * x15
                            * x16
                            * x17
                            * x18
                            * x19
                            * x20
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x31
                            * x32
                            * x33
                            * x34
                            * x35
                            * x36
                            * x5
                            * x7
                            * x9
                        )
                        x45 = 5.0249700898396030198e-25 * x44
                        x46 = x3 * x42
                        x47 = 1.1306182702139106794e-22 * x39
                        x48 = x22 * x3
                        x49 = x47 * x48
                        x50 = (
                            x11
                            * x13
                            * x14
                            * x15
                            * x17
                            * x18
                            * x20
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x31
                            * x32
                            * x33
                            * x34
                            * x35
                            * x36
                            * x5
                            * x7
                            * x9
                        )
                        x51 = x48 * x50
                        x52 = x16 * x39
                        x53 = 7.9143278914973747561e-22 * x52
                        x54 = x21 * x53
                        x55 = (
                            x11
                            * x13
                            * x14
                            * x17
                            * x18
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x31
                            * x32
                            * x33
                            * x34
                            * x35
                            * x36
                            * x5
                            * x7
                            * x9
                        )
                        x56 = x48 * x55
                        x57 = 1.143180695438509687e-21 * x19
                        x58 = x20 * x52
                        x59 = x57 * x58
                        x60 = x21 * x59
                        x61 = x30 * x48
                        x62 = x21 * x61
                        x63 = (
                            x11
                            * x13
                            * x14
                            * x18
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * x31
                            * x32
                            * x33
                            * x34
                            * x35
                            * x36
                            * x5
                            * x7
                            * x9
                        )
                        x64 = x15 * x19
                        x65 = x58 * x64
                        x66 = x63 * x65
                        x67 = 1.0288626258946587183e-20 * x66
                        x68 = (
                            x11
                            * x13
                            * x14
                            * x17
                            * x18
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x31
                            * x32
                            * x33
                            * x34
                            * x35
                            * x36
                            * x7
                            * x9
                        )
                        x69 = x29 * x65
                        x70 = x62 * x69
                        x71 = 4.5269955539364983605e-21 * x70
                        x72 = (
                            x11
                            * x13
                            * x14
                            * x17
                            * x18
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x31
                            * x32
                            * x33
                            * x34
                            * x5
                            * x7
                        )
                        x73 = x35 * x36
                        x74 = x70 * x73
                        x75 = 1.2574987649823606557e-20 * x74
                        x76 = (
                            x11
                            * x14
                            * x17
                            * x18
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x31
                            * x32
                            * x5
                            * x7
                            * x9
                        )
                        x77 = x33 * x34
                        x78 = x74 * x77
                        x79 = 4.8503523792176768148e-20 * x78
                        x80 = (
                            x11
                            * x13
                            * x14
                            * x23
                            * x24
                            * x25
                            * x26
                            * x28
                            * x31
                            * x32
                            * x5
                            * x7
                            * x9
                        )
                        x81 = x17 * x18
                        x82 = x80 * x81
                        x83 = (
                            x11
                            * x13
                            * x14
                            * x23
                            * x24
                            * x26
                            * x28
                            * x32
                            * x5
                            * x7
                            * x81
                            * x9
                        )
                        x84 = x27 * x77
                        x85 = x31 * x84
                        x86 = x83 * x85
                        x87 = (
                            x11
                            * x14
                            * x24
                            * x25
                            * x26
                            * x28
                            * x32
                            * x5
                            * x81
                            * x85
                            * x9
                        )
                        x88 = x13 * x73
                        x89 = x7 * x88
                        x90 = x87 * x89
                        x91 = 1.0288626258946587183e-20 * x25
                        x92 = x27 * x83
                        x93 = x5 * x68
                        x94 = x15 * x93
                        x95 = x61 * x88
                        x96 = x34 * x76
                        x97 = x29 * x64
                        x98 = x20 * x97
                        x99 = x96 * x98
                        x100 = x16 * x98
                        x101 = x30 * x72
                        x102 = x101 * x9
                        x103 = x102 * x36
                        x104 = x100 * x103
                        x105 = x41 * x47
                        x106 = x22 * x43
                        x107 = x106 * x16
                        x108 = 7.9143278914973747561e-22 * x55 * x64
                        x109 = x50 * x57
                        x110 = x100 * x106
                        x111 = x30 * x73
                        x112 = x111 * x84
                        x113 = x17 * x80
                        x114 = x112 * x113
                        x115 = 1.0288626258946587183e-20 * x114
                        x116 = x30 * x88
                        x117 = x110 * x116
                        x118 = x23 * x87
                        x119 = 4.5269955539364983605e-21 * x118
                        x120 = x14 * x23 * x24 * x26 * x28 * x32 * x5 * x81 * x9
                        x121 = x120 * x85
                        x122 = x25 * x30
                        x123 = x122 * x89
                        x124 = x110 * x123
                        x125 = 1.2574987649823606557e-20 * x124
                        x126 = x28 * x85
                        x127 = x124 * x126
                        x128 = x11 * x23 * x24 * x26 * x32 * x5 * x81 * x9
                        x129 = 4.8503523792176768148e-20 * x128
                        x130 = x14 * x85
                        x131 = x129 * x130
                        x132 = x11 * x23 * x24 * x32 * x5 * x9
                        x133 = x14 * x81
                        x134 = x126 * x133
                        x135 = x132 * x134
                        x136 = x11 * x26 * x5 * x9
                        x137 = x23 * x32
                        x138 = x136 * x137
                        x139 = 4.5269955539364983605e-21 * x138
                        x140 = x30 * x89
                        x141 = x134 * x24
                        x142 = x136 * x23
                        x143 = x141 * x142
                        x144 = x140 * x143
                        x145 = x144 * x91
                        x146 = x17 * x63
                        x147 = x146 * x15
                        x148 = x147 * x20 * x57
                        x149 = x33 * x76
                        x150 = 7.9143278914973747561e-22 * x149
                        x151 = x102 * x35
                        x152 = 1.1306182702139106794e-22 * x151
                        x153 = x46 * x53
                        x154 = x116 * x99
                        x155 = x30 * x46
                        x156 = x46 * x69
                        x157 = x112 * x83
                        x158 = x69 * x90
                        x159 = 4.5269955539364983605e-21 * x155
                        x160 = x111 * x156
                        x161 = x77 * x82
                        x162 = 4.8503523792176768148e-20 * x160
                        x163 = x76 * x77
                        x164 = x101 * x73
                        x165 = x68 * x69
                        x166 = x46 * x55
                        x167 = x22 * x46
                        x168 = x100 * x167
                        x169 = x116 * x168
                        x170 = x16 * x167
                        x171 = x123 * x168
                        x172 = 1.2574987649823606557e-20 * x171
                        x173 = x2 * x41
                        x174 = 2.5438911079812990288e-20 * x22
                        x175 = x2 * x42
                        x176 = x175 * x69
                        x177 = x72 * x9
                        x178 = x35 * x69
                        x179 = x41 * x61
                        x180 = x175 * x22
                        x181 = 1.7807237755869093201e-19 * x52
                        x182 = 2.5721565647366467957e-19 * x180
                        x183 = x19 * x58
                        x184 = x180 * x30
                        x185 = 1.0185739996357121311e-18 * x184
                        x186 = x176 * x22
                        x187 = 2.8293722212103114753e-18 * x186
                        x188 = 1.0913292853239772833e-17 * x163
                        x189 = x111 * x186
                        x190 = 1.0913292853239772833e-17 * x161
                        x191 = x111 * x86
                        x192 = 2.3149409082629821162e-18 * x25
                        x193 = x65 * x93
                        x194 = 1.7807237755869093201e-19 * x96
                        x195 = x103 * x167
                        x196 = 2.5721565647366467957e-19 * x39
                        x197 = x22 * x36
                        x198 = 2.3149409082629821162e-18 * x197
                        x199 = x155 * x69
                        x200 = x113 * x84
                        x201 = 1.0185739996357121311e-18 * x118
                        x202 = x13 * x197
                        x203 = x199 * x202
                        x204 = 2.8293722212103114753e-18 * x121
                        x205 = x122 * x156
                        x206 = x205 * x7
                        x207 = x202 * x206
                        x208 = 1.0913292853239772833e-17 * x128
                        x209 = x207 * x208
                        x210 = 2.8293722212103114753e-18 * x135
                        x211 = x134 * x138
                        x212 = 1.0185739996357121311e-18 * x211
                        x213 = x13 * x206
                        x214 = 1.7807237755869093201e-19 * x149
                        x215 = x155 * x22
                        x216 = x178 * x215
                        x217 = x13 * x216
                        x218 = x151 * x167
                        x219 = 2.5721565647366467957e-19 * x218
                        x220 = x22 * x35
                        x221 = 2.3149409082629821162e-18 * x84
                        x222 = x11 * x137 * x26
                        x223 = x141 * x9
                        x224 = x222 * x223
                        x225 = x18 * x80
                        x226 = x183 * x29
                        x227 = x15 * x29 * x58
                        x228 = x41 * x69 * x95
                        x229 = x179 * x69
                        x230 = x229 * x89
                        x231 = x230 * x25
                        x232 = x208 * x231
                        x233 = 1.2465066429108365241e-18 * x52
                        x234 = x166 * x22
                        x235 = x116 * x167
                        x236 = x156 * x22
                        x237 = x116 * x236
                        x238 = x149 * x235
                        x239 = 1.800509595315652757e-18 * x52
                        x240 = x19 * x234
                        x241 = x215 * x52
                        x242 = x63 * x64
                        x243 = 7.1300179974499849178e-18 * x241 * x97
                        x244 = 1.9805605548472180327e-17 * x167
                        x245 = x52 * x97
                        x246 = x244 * x245
                        x247 = x111 * x163
                        x248 = x167 * x245
                        x249 = 7.6393049972678409834e-17 * x248
                        x250 = x111 * x161
                        x251 = 1.6204586357840874813e-17 * x25
                        x252 = x215 * x64 * x93
                        x253 = 1.800509595315652757e-18 * x167
                        x254 = x253 * x39
                        x255 = x27 * x34
                        x256 = 1.6204586357840874813e-17 * x255
                        x257 = x160 * x22
                        x258 = x113 * x257
                        x259 = x255 * x31
                        x260 = x120 * x259
                        x261 = x237 * x25
                        x262 = x11 * x261
                        x263 = x123 * x236
                        x264 = 1.9805605548472180327e-17 * x263
                        x265 = 7.6393049972678409834e-17 * x128
                        x266 = x259 * x28
                        x267 = x263 * x266
                        x268 = x14 * x263
                        x269 = x133 * x264
                        x270 = 7.1300179974499849178e-18 * x133
                        x271 = x142 * x24
                        x272 = x263 * x28
                        x273 = x133 * x272
                        x274 = x273 * x31
                        x275 = x271 * x274
                        x276 = 1.800509595315652757e-18 * x238
                        x277 = x257 * x33
                        x278 = x277 * x92
                        x279 = x136 * x32
                        x280 = x24 * x279
                        x281 = x272 * x31
                        x282 = x27 * x33
                        x283 = x270 * x281 * x282
                        x284 = x222 * x24
                        x285 = x284 * x5
                        x286 = x284 * x9
                        x287 = x123 * x227
                        x288 = x167 * x287
                        x289 = x244 * x287
                        x290 = x265 * x288
                        x291 = x118 * x235
                        x292 = x114 * x167
                        x293 = x20 * x39
                        x294 = 2.6007360821226095379e-18 * x293
                        x295 = 2.6007360821226095379e-18 * x167
                        x296 = 1.029891488520553377e-17 * x68
                        x297 = x39 * x98
                        x298 = x215 * x297
                        x299 = x167 * x297
                        x300 = 2.8608096903348704917e-17 * x299
                        x301 = 1.1034551662720214754e-16 * x299
                        x302 = 1.029891488520553377e-17 * x90
                        x303 = 2.3406624739103485841e-17 * x25
                        x304 = 2.3406624739103485841e-17 * x65
                        x305 = 1.029891488520553377e-17 * x65
                        x306 = x123 * x167
                        x307 = x306 * x65
                        x308 = 2.8608096903348704917e-17 * x307
                        x309 = 1.1034551662720214754e-16 * x128
                        x310 = x307 * x309
                        x311 = x211 * x306
                        x312 = x143 * x306
                        x313 = x236 * x73
                        x314 = 2.8608096903348704917e-17 * x313
                        x315 = 1.1034551662720214754e-16 * x313
                        x316 = 2.3406624739103485841e-17 * x226
                        x317 = 1.029891488520553377e-17 * x226
                        x318 = x226 * x306
                        x319 = 2.8608096903348704917e-17 * x318
                        x320 = x309 * x318
                        x321 = 2.1065962265193137257e-16 * x84
                        x322 = x17 * x268
                        x323 = x132 * x26
                        x324 = x126 * x18
                        x325 = x268 * x324
                        x326 = x126 * x322
                        x327 = 9.2690233966849803931e-17 * x326
                        x328 = 9.9310964964481932784e-16 * x323
                        x329 = x25 * x257 * x7
                        x330 = x126 * x14 * x17
                        x331 = 9.9310964964481932784e-16 * x77
                        x332 = 2.5747287213013834425e-16 * x236
                        x333 = x120 * x84
                        x334 = 2.5747287213013834425e-16 * x263
                        x335 = 9.9310964964481932784e-16 * x128 * x84
                        x336 = x273 * x84
                        x337 = 2.5747287213013834425e-16 * x132
                        x338 = 9.2690233966849803931e-17 * x138
                        x339 = 9.2690233966849803931e-17 * x263
                        x340 = x11 * x23
                        x341 = x26 * x5
                        x342 = x141 * x341
                        x343 = x223 * x26
                        x344 = x263 * x328
                        x345 = x134 * x263
                        x346 = 4.078370294541391373e-17 * x345
                        x347 = 1.1328806373726087147e-16 * x5
                        x348 = x222 * x347
                        x349 = 4.3696824584372050425e-16 * x77
                        x350 = x120 * x31
                        x351 = 1.1328806373726087147e-16 * x11
                        x352 = x223 * x263
                        x353 = x32 * x352
                        x354 = x263 * x81
                        x355 = 4.3696824584372050425e-16 * x354
                        x356 = x280 * x355
                        x357 = x140 * x236
                        x358 = x286 * x355
                        x359 = x137 * x263
                        x360 = 3.1468906593683575409e-16 * x359
                        x361 = 3.1468906593683575409e-16 * x357
                        x362 = 1.2138006828992236229e-15 * x329
                        x363 = 1.2138006828992236229e-15 * x77
                        x364 = 1.2138006828992236229e-15 * x126
                        x365 = x128 * x357
                        x366 = 1.2138006828992236229e-15 * x130
                        x367 = x285 * x354
                        x368 = 4.6818026340398625455e-15 * x128
                        x369 = x329 * x368
                        x370 = x368 * x77
                        return jnp.asarray(
                            [
                                x3 * x38,
                                -x38 * x40,
                                x43 * x45,
                                -x45 * x46,
                                -x44 * x49,
                                x51 * x54,
                                -x56 * x60,
                                x62 * x67,
                                -x68 * x71,
                                x72 * x75,
                                -x76 * x79,
                                x79 * x82,
                                -x75 * x86,
                                x71 * x90,
                                -x78 * x91 * x92,
                                x60 * x61 * x94,
                                -x54 * x95 * x99,
                                x104 * x21 * x49,
                                x105 * x37,
                                -x107 * x108,
                                x106 * x109,
                                -x110 * x115,
                                x117 * x119,
                                -x121 * x125,
                                x127 * x129,
                                -x124 * x131,
                                x125 * x135,
                                -x127 * x133 * x139,
                                x110 * x145,
                                -x107 * x148,
                                x117 * x150,
                                -x110 * x152,
                                -x104 * x46 * x47,
                                x153 * x154,
                                -x155 * x59 * x94,
                                x156 * x157 * x91,
                                -x158 * x159,
                                1.2574987649823606557e-20 * x160 * x86,
                                -x161 * x162,
                                x162 * x163,
                                -1.2574987649823606557e-20 * x156 * x164,
                                x159 * x165,
                                -x155 * x67,
                                x166 * x59,
                                -x153 * x50,
                                x105 * x3 * x44,
                                x152 * x168,
                                -x150 * x169,
                                x148 * x170,
                                -x145 * x168,
                                x134 * x139 * x171,
                                -x135 * x172,
                                x131 * x171,
                                -x126 * x129 * x171,
                                x121 * x172,
                                -x119 * x169,
                                x115 * x168,
                                -x109 * x167,
                                x108 * x170,
                                -1.1306182702139106794e-22 * x173 * x37,
                                x173 * x174 * x39 * x44,
                                -x103 * x174 * x176,
                                x102 * x156 * x174,
                                -2.5438911079812990288e-20 * x177 * x178 * x179,
                                -x180 * x181 * x50,
                                x182 * x183 * x55,
                                -2.3149409082629821162e-18 * x184 * x66,
                                x165 * x185,
                                -x164 * x187,
                                x188 * x189,
                                -x189 * x190,
                                x187 * x191,
                                -x158 * x185,
                                x157 * x186 * x192,
                                -x182 * x193 * x30,
                                x116 * x186 * x194,
                                x181 * x195 * x97,
                                -x195 * x196 * x98,
                                x198 * x199 * x200,
                                -x201 * x203,
                                x204 * x207,
                                -x126 * x209,
                                x130 * x209,
                                -x207 * x210,
                                x207 * x212,
                                -x143 * x198 * x213,
                                2.5721565647366467957e-19 * x156 * x177 * x197,
                                -x203 * x214,
                                -x194 * x217,
                                x219 * x65,
                                -x205 * x220 * x221 * x83,
                                1.0185739996357121311e-18 * x217 * x7 * x87,
                                -2.8293722212103114753e-18 * x216 * x86,
                                x190 * x216,
                                -x188 * x216,
                                2.8293722212103114753e-18 * x101 * x156 * x220,
                                -1.0185739996357121311e-18 * x213 * x220 * x224,
                                x216 * x221 * x225,
                                -x219 * x226,
                                1.7807237755869093201e-19 * x218 * x227,
                                x214 * x228,
                                -2.5721565647366467957e-19 * x17 * x41 * x48 * x66,
                                x143 * x192 * x230,
                                -x212 * x231,
                                x210 * x231,
                                -x130 * x232,
                                x126 * x232,
                                -x204 * x231,
                                x201 * x228,
                                -2.3149409082629821162e-18 * x200 * x229 * x73,
                                x19 * x196 * x41 * x51,
                                -x181 * x41 * x56 * x64,
                                x15 * x233 * x234,
                                -x233 * x235 * x96 * x97,
                                1.2465066429108365241e-18 * x237 * x76,
                                -1.2465066429108365241e-18 * x227 * x238,
                                -x239 * x240,
                                1.6204586357840874813e-17 * x241 * x242,
                                -x243 * x68,
                                x164 * x246,
                                -x247 * x249,
                                x249 * x250,
                                -x191 * x246,
                                x243 * x90,
                                -x157 * x248 * x251,
                                x239 * x252,
                                x154 * x254,
                                -x256 * x258,
                                7.1300179974499849178e-18 * x260 * x262,
                                -x260 * x264,
                                x265 * x267,
                                -x259 * x265 * x268,
                                x132 * x266 * x269,
                                -x138 * x267 * x270,
                                x256 * x275,
                                -1.800509595315652757e-18 * x236 * x88 * x96,
                                -x276 * x65,
                                x251 * x278,
                                -x280 * x283,
                                1.9805605548472180327e-17 * x278 * x31,
                                -7.6393049972678409834e-17 * x277 * x82,
                                7.6393049972678409834e-17 * x149 * x257,
                                -x269 * x28 * x282 * x285 * x31,
                                x283 * x286,
                                -1.6204586357840874813e-17 * x225 * x27 * x277,
                                x226 * x276,
                                x147 * x253 * x58,
                                -1.6204586357840874813e-17 * x143 * x288,
                                7.1300179974499849178e-18 * x211 * x288,
                                -x135 * x289,
                                x130 * x290,
                                -x126 * x290,
                                x121 * x289,
                                -7.1300179974499849178e-18 * x227 * x291,
                                1.6204586357840874813e-17 * x227 * x292,
                                -x254 * x50,
                                x240 * x294,
                                -x252 * x294,
                                x193 * x295,
                                -x146 * x183 * x295,
                                -2.3406624739103485841e-17 * x215 * x242 * x293,
                                x296 * x298,
                                -x164 * x300,
                                x247 * x301,
                                -x250 * x301,
                                x191 * x300,
                                -x298 * x302,
                                x157 * x299 * x303,
                                x292 * x304,
                                -x291 * x305,
                                x121 * x308,
                                -x126 * x310,
                                x130 * x310,
                                -x135 * x308,
                                x305 * x311,
                                -x304 * x312,
                                -x303 * x313 * x83 * x84,
                                x236 * x302,
                                -x314 * x86,
                                x161 * x315,
                                -x163 * x315,
                                x314 * x72,
                                -x236 * x296,
                                2.3406624739103485841e-17 * x167 * x66,
                                x312 * x316,
                                -x311 * x317,
                                x135 * x319,
                                -x130 * x320,
                                x126 * x320,
                                -x121 * x319,
                                x291 * x317,
                                -x292 * x316,
                                2.1065962265193137257e-16 * x112 * x236 * x80,
                                -x28 * x321 * x322 * x323,
                                x271 * x273 * x321,
                                -2.1065962265193137257e-16 * x271 * x325,
                                -x286 * x327,
                                2.5747287213013834425e-16 * x285 * x326,
                                -x328 * x329 * x330,
                                x258 * x331,
                                -x140 * x323 * x330 * x332,
                                x280 * x327,
                                9.2690233966849803931e-17 * x262 * x333,
                                -x333 * x334,
                                x272 * x335,
                                -x268 * x335,
                                x336 * x337,
                                -x336 * x338,
                                -x136 * x141 * x339,
                                x144 * x332,
                                -x275 * x331,
                                9.9310964964481932784e-16 * x143 * x329,
                                -x334 * x340 * x342,
                                x339 * x340 * x343,
                                x325 * x338,
                                -x325 * x337,
                                x130 * x18 * x344,
                                -x324 * x344,
                                2.5747287213013834425e-16
                                * x137
                                * x24
                                * x325
                                * x341
                                * x9,
                                -9.2690233966849803931e-17 * x14 * x261 * x323 * x324,
                                4.078370294541391373e-17 * x224 * x261,
                                -4.078370294541391373e-17 * x237 * x87,
                                x279 * x346,
                                -x222 * x346 * x9,
                                -x141 * x261 * x348,
                                4.3696824584372050425e-16 * x118 * x257,
                                -x262 * x349 * x350,
                                x121 * x237 * x351,
                                1.1328806373726087147e-16 * x341 * x353,
                                -x126 * x356,
                                x130 * x356,
                                -x11 * x347 * x353,
                                -1.1328806373726087147e-16 * x211 * x357,
                                x138 * x274 * x349,
                                -4.3696824584372050425e-16 * x211 * x329,
                                x345 * x348,
                                x137 * x351 * x352,
                                -x130 * x358,
                                x126 * x358,
                                -1.1328806373726087147e-16 * x343 * x359,
                                x342 * x360,
                                -x121 * x361,
                                x135 * x361,
                                -x11 * x141 * x360 * x5,
                                -x121 * x362,
                                x263 * x350 * x363,
                                x364 * x365,
                                -x365 * x366,
                                -x132 * x274 * x363,
                                x135 * x362,
                                x366 * x367,
                                -x364 * x367,
                                x126 * x369,
                                -x281 * x370,
                                x268 * x31 * x370,
                                -x130 * x369,
                            ]
                        )

                case 289:

                    def shape_functions(xi):
                        x0 = xi[0] - 1.0
                        x1 = xi[1] - 1.0
                        x2 = 2.0 * xi[1]
                        x3 = x2 + 1.0
                        x4 = 4.0 * xi[1]
                        x5 = x4 + 1.0
                        x6 = 8.0 * xi[1]
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
                        x19 = (
                            x10
                            * x11
                            * x12
                            * x13
                            * x14
                            * x15
                            * x16
                            * x17
                            * x18
                            * x3
                            * x5
                            * x7
                            * x8
                            * x9
                            * xi[1]
                        )
                        x20 = x1 * x19
                        x21 = 2.0 * xi[0]
                        x22 = x21 + 1.0
                        x23 = 4.0 * xi[0]
                        x24 = x23 + 1.0
                        x25 = 8.0 * xi[0]
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
                        x38 = (
                            x22
                            * x24
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x31
                            * x32
                            * x33
                            * x34
                            * x35
                            * x36
                            * x37
                            * xi[0]
                        )
                        x39 = 6.1319794541210246319e-19 * x38
                        x40 = x20 * x39
                        x41 = xi[0] + 1.0
                        x42 = xi[1] + 1.0
                        x43 = x19 * x39 * x42
                        x44 = x20 * x41
                        x45 = (
                            x0
                            * x22
                            * x24
                            * x26
                            * x27
                            * x28
                            * x29
                            * x31
                            * x32
                            * x33
                            * x34
                            * x35
                            * x36
                            * xi[0]
                        )
                        x46 = 7.8489337012749115288e-17 * x45
                        x47 = x44 * x46
                        x48 = x30 * x44
                        x49 = (
                            x0
                            * x22
                            * x24
                            * x26
                            * x28
                            * x29
                            * x31
                            * x32
                            * x33
                            * x35
                            * x36
                            * x37
                            * xi[0]
                        )
                        x50 = 2.9433501379780918233e-16 * x49
                        x51 = x48 * x50
                        x52 = x27 * x48
                        x53 = (
                            x0
                            * x22
                            * x24
                            * x26
                            * x28
                            * x31
                            * x32
                            * x33
                            * x34
                            * x35
                            * x37
                            * xi[0]
                        )
                        x54 = 2.7471267954462190351e-15 * x53
                        x55 = x52 * x54
                        x56 = x29 * x52
                        x57 = (
                            x0
                            * x24
                            * x26
                            * x28
                            * x32
                            * x33
                            * x34
                            * x35
                            * x36
                            * x37
                            * xi[0]
                        )
                        x58 = 2.232040521300052966e-15 * x57
                        x59 = x56 * x58
                        x60 = x22 * x56
                        x61 = x0 * x24 * x26 * x31 * x32 * x33 * x34 * x36 * x37 * xi[0]
                        x62 = 2.1427589004480508474e-14 * x61
                        x63 = x60 * x62
                        x64 = x28 * x60
                        x65 = x0 * x26 * x31 * x33 * x34 * x35 * x36 * x37 * xi[0]
                        x66 = 1.9641956587440466101e-14 * x65
                        x67 = x64 * x66
                        x68 = x0 * x31 * x32 * x34 * x35 * x36 * x37 * xi[0]
                        x69 = x33 * x68
                        x70 = 5.6119875964115617431e-14 * x24
                        x71 = x64 * x70
                        x72 = (
                            173140.53095490047871 * xi[0] ** 16
                            - 551885.44241874527589 * xi[0] ** 14
                            + 694168.40804232804233 * xi[0] ** 12
                            - 441984.42698916603679 * xi[0] ** 10
                            + 152107.76187452758881 * xi[0] ** 8
                            - 28012.603597883597884 * xi[0] ** 6
                            + 2562.5271453766691862 * xi[0] ** 4
                            - 97.755011337868480726 * xi[0] ** 2
                            + 1.0
                        )
                        x73 = x1 * x72
                        x74 = 7.8306956613834920713e-10 * x19
                        x75 = x26 * x68
                        x76 = (
                            x10
                            * x12
                            * x13
                            * x14
                            * x15
                            * x16
                            * x17
                            * x18
                            * x3
                            * x42
                            * x5
                            * x7
                            * x9
                            * xi[1]
                        )
                        x77 = x1 * x38
                        x78 = x41 * x77
                        x79 = x76 * x78
                        x80 = 7.8489337012749115288e-17 * x8
                        x81 = 2.9433501379780918233e-16 * x11
                        x82 = (
                            x11
                            * x12
                            * x13
                            * x14
                            * x15
                            * x16
                            * x17
                            * x3
                            * x42
                            * x5
                            * x7
                            * x9
                            * xi[1]
                        )
                        x83 = x78 * x82
                        x84 = x18 * x8
                        x85 = 2.7471267954462190351e-15 * x84
                        x86 = (
                            x12
                            * x13
                            * x14
                            * x16
                            * x17
                            * x42
                            * x5
                            * x7
                            * x84
                            * x9
                            * xi[1]
                        )
                        x87 = x10 * x78
                        x88 = x86 * x87
                        x89 = x11 * x15
                        x90 = 2.232040521300052966e-15 * x89
                        x91 = x12 * x13 * x14 * x16 * x42 * x5 * x7 * x89 * xi[1]
                        x92 = x3 * x87
                        x93 = x91 * x92
                        x94 = x17 * x84
                        x95 = 2.1427589004480508474e-14 * x94
                        x96 = x13 * x14 * x16 * x42 * x7 * x94 * xi[1]
                        x97 = x9 * x92
                        x98 = x96 * x97
                        x99 = x12 * x89
                        x100 = 1.9641956587440466101e-14 * x99
                        x101 = x13 * x14 * x42 * x99 * xi[1]
                        x102 = x16 * x94
                        x103 = x101 * x102
                        x104 = x5 * x97
                        x105 = 5.6119875964115617431e-14 * x104
                        x106 = (
                            173140.53095490047871 * xi[1] ** 16
                            - 551885.44241874527589 * xi[1] ** 14
                            + 694168.40804232804233 * xi[1] ** 12
                            - 441984.42698916603679 * xi[1] ** 10
                            + 152107.76187452758881 * xi[1] ** 8
                            - 28012.603597883597884 * xi[1] ** 6
                            + 2562.5271453766691862 * xi[1] ** 4
                            - 97.755011337868480726 * xi[1] ** 2
                            + 1.0
                        )
                        x107 = x106 * x41
                        x108 = 7.8306956613834920713e-10 * x38
                        x109 = x13 * x99
                        x110 = x102 * x42 * xi[1]
                        x111 = x110 * x7
                        x112 = x109 * x111
                        x113 = x111 * x14
                        x114 = x100 * x113
                        x115 = x101 * x7
                        x116 = x115 * x95
                        x117 = x5 * x90
                        x118 = x85 * x9
                        x119 = x3 * x81
                        x120 = x10 * x80
                        x121 = x30 * x41
                        x122 = x121 * x19 * x42
                        x123 = x122 * x27
                        x124 = x123 * x29
                        x125 = x124 * x22
                        x126 = x125 * x28
                        x127 = x126 * x66
                        x128 = x126 * x70
                        x129 = x37 * x41
                        x130 = x0 * x77
                        x131 = x130 * x82
                        x132 = x10 * x130
                        x133 = x132 * x86
                        x134 = x132 * x3
                        x135 = x134 * x91
                        x136 = x134 * x9
                        x137 = x136 * x96
                        x138 = x136 * x5
                        x139 = 5.6119875964115617431e-14 * x138
                        x140 = x130 * x76
                        x141 = x1 * x45
                        x142 = x129 * x141
                        x143 = x76 * x8
                        x144 = 1.0046635137631886757e-14 * x143
                        x145 = x121 * x141
                        x146 = x145 * x82
                        x147 = x10 * x8
                        x148 = 1.0046635137631886757e-14 * x147
                        x149 = x142 * x82
                        x150 = x1 * x121
                        x151 = x143 * x150
                        x152 = 3.7674881766119575338e-14 * x49
                        x153 = x151 * x152
                        x154 = x151 * x27
                        x155 = 3.5163222981711603649e-13 * x53
                        x156 = x154 * x155
                        x157 = x154 * x29
                        x158 = 2.8570118672640677965e-13 * x57
                        x159 = x157 * x158
                        x160 = x157 * x22
                        x161 = 2.7427313925735050846e-12 * x61
                        x162 = x160 * x161
                        x163 = x160 * x28
                        x164 = 2.5141704431923796609e-12 * x65
                        x165 = x163 * x164
                        x166 = 7.1833441234067990312e-12 * x24
                        x167 = x163 * x166
                        x168 = x73 * x76
                        x169 = 1.0023290446570869851e-7 * x8
                        x170 = 3.7674881766119575338e-14 * x11
                        x171 = x145 * x170
                        x172 = 3.5163222981711603649e-13 * x84
                        x173 = x10 * x145
                        x174 = 2.8570118672640677965e-13 * x89
                        x175 = x173 * x174
                        x176 = x173 * x3
                        x177 = x176 * x91
                        x178 = 2.7427313925735050846e-12 * x94
                        x179 = x176 * x9
                        x180 = 2.5141704431923796609e-12 * x99
                        x181 = x179 * x180
                        x182 = x179 * x5
                        x183 = 7.1833441234067990312e-12 * x182
                        x184 = x107 * x30
                        x185 = 1.0023290446570869851e-7 * x45
                        x186 = x113 * x5
                        x187 = x115 * x178
                        x188 = x3 * x9
                        x189 = x188 * x5
                        x190 = x189 * x96
                        x191 = x10 * x3
                        x192 = x191 * x86
                        x193 = x150 * x27
                        x194 = x193 * x82
                        x195 = x147 * x194
                        x196 = x195 * x29
                        x197 = x196 * x22
                        x198 = x197 * x28
                        x199 = x164 * x198
                        x200 = x166 * x198
                        x201 = x10 * x73
                        x202 = x150 * x34
                        x203 = x202 * x82
                        x204 = x142 * x170
                        x205 = x10 * x142
                        x206 = x188 * x205
                        x207 = x174 * x205
                        x208 = x189 * x205
                        x209 = x180 * x206
                        x210 = 7.1833441234067990312e-12 * x208
                        x211 = x191 * x91
                        x212 = x202 * x49
                        x213 = x11 * x76
                        x214 = 1.4128080662294840752e-13 * x213
                        x215 = x193 * x49
                        x216 = x11 * x192
                        x217 = 1.4128080662294840752e-13 * x216
                        x218 = x36 * x53
                        x219 = x193 * x213
                        x220 = 1.3186208618141851368e-12 * x219
                        x221 = x219 * x29
                        x222 = 1.0713794502240254237e-12 * x57
                        x223 = x221 * x222
                        x224 = x22 * x221
                        x225 = 1.0285242722150644067e-11 * x61
                        x226 = x224 * x225
                        x227 = x224 * x28
                        x228 = 9.4281391619714237284e-12 * x65
                        x229 = x227 * x228
                        x230 = 2.6937540462775496367e-11 * x24
                        x231 = x227 * x230
                        x232 = 3.7587339174640761942e-7 * x11
                        x233 = x29 * x53
                        x234 = 1.3186208618141851368e-12 * x84
                        x235 = x234 * x49
                        x236 = x10 * x215
                        x237 = 1.0713794502240254237e-12 * x89
                        x238 = x236 * x237
                        x239 = 1.0285242722150644067e-11 * x94
                        x240 = x211 * x239
                        x241 = x188 * x236
                        x242 = 9.4281391619714237284e-12 * x99
                        x243 = x241 * x242
                        x244 = x189 * x236
                        x245 = 2.6937540462775496367e-11 * x244
                        x246 = x184 * x27
                        x247 = 3.7587339174640761942e-7 * x49
                        x248 = x115 * x239
                        x249 = x234 * x91
                        x250 = x193 * x216
                        x251 = 1.3186208618141851368e-12 * x250
                        x252 = x22 * x29
                        x253 = x250 * x252
                        x254 = x253 * x28
                        x255 = x228 * x254
                        x256 = x230 * x254
                        x257 = x201 * x3
                        x258 = x29 * x31
                        x259 = x10 * x212
                        x260 = x188 * x259
                        x261 = x237 * x259
                        x262 = x189 * x259
                        x263 = x242 * x260
                        x264 = 2.6937540462775496367e-11 * x262
                        x265 = 1.2307128043599061277e-11 * x84
                        x266 = x194 * x265
                        x267 = x193 * x233
                        x268 = x10 * x267
                        x269 = x188 * x268
                        x270 = x265 * x91
                        x271 = x193 * x218
                        x272 = x10 * x188
                        x273 = x271 * x272
                        x274 = x258 * x57
                        x275 = x194 * x84
                        x276 = 9.9995415354242372877e-12 * x275
                        x277 = x252 * x275
                        x278 = 9.5995598740072677962e-11 * x61
                        x279 = x277 * x278
                        x280 = x277 * x28
                        x281 = 8.7995965511733288132e-11 * x65
                        x282 = x280 * x281
                        x283 = 2.5141704431923796609e-10 * x24
                        x284 = x280 * x283
                        x285 = 3.508151656299804448e-6 * x84
                        x286 = x252 * x57
                        x287 = 9.9995415354242372877e-12 * x89
                        x288 = x268 * x287
                        x289 = 9.5995598740072677962e-11 * x94
                        x290 = x211 * x289
                        x291 = 8.7995965511733288132e-11 * x99
                        x292 = x269 * x291
                        x293 = x189 * x268
                        x294 = 2.5141704431923796609e-10 * x293
                        x295 = x246 * x29
                        x296 = 3.508151656299804448e-6 * x53
                        x297 = x115 * x289
                        x298 = x193 * x286
                        x299 = x272 * x298
                        x300 = x84 * x91
                        x301 = 9.9995415354242372877e-12 * x300
                        x302 = x193 * x272
                        x303 = x252 * x28
                        x304 = x302 * x303
                        x305 = x300 * x304
                        x306 = x281 * x305
                        x307 = x283 * x305
                        x308 = x257 * x9
                        x309 = x252 * x35
                        x310 = x302 * x309
                        x311 = x274 * x302
                        x312 = x10 * x271
                        x313 = x287 * x312
                        x314 = x189 * x312
                        x315 = x273 * x291
                        x316 = 2.5141704431923796609e-10 * x314
                        x317 = x193 * x274
                        x318 = 8.1246274975321927963e-12 * x10 * x89
                        x319 = x318 * x86
                        x320 = x190 * x318
                        x321 = x193 * x61
                        x322 = x309 * x321
                        x323 = 7.7996423976309050844e-11 * x10
                        x324 = x86 * x89
                        x325 = x323 * x324
                        x326 = x32 * x65
                        x327 = x10 * x303
                        x328 = x193 * x327
                        x329 = x324 * x328
                        x330 = 7.1496721978283296607e-11 * x329
                        x331 = 2.0427634850938084745e-10 * x24
                        x332 = x329 * x331
                        x333 = 2.850373220743591114e-6 * x89
                        x334 = x24 * x65
                        x335 = x303 * x321
                        x336 = x298 * x94
                        x337 = 7.7996423976309050844e-11 * x211
                        x338 = 7.1496721978283296607e-11 * x99
                        x339 = x299 * x338
                        x340 = x10 * x189
                        x341 = 2.0427634850938084745e-10 * x340
                        x342 = x298 * x341
                        x343 = x22 * x295
                        x344 = 2.850373220743591114e-6 * x57
                        x345 = x115 * x189
                        x346 = x323 * x345
                        x347 = x190 * x89
                        x348 = x323 * x347
                        x349 = x328 * x347
                        x350 = 7.1496721978283296607e-11 * x349
                        x351 = x331 * x349
                        x352 = x308 * x5
                        x353 = x317 * x94
                        x354 = x311 * x338
                        x355 = x317 * x341
                        x356 = 7.4876567017256688811e-10 * x94
                        x357 = x211 * x356
                        x358 = x321 * x327
                        x359 = x322 * x340
                        x360 = 6.8636853099151964743e-10 * x94
                        x361 = x326 * x360
                        x362 = x193 * x211 * x303
                        x363 = x24 * x69
                        x364 = 1.9610529456900561355e-9 * x94
                        x365 = x362 * x364
                        x366 = 0.000027363582919138474694 * x94
                        x367 = x24 * x75
                        x368 = x334 * x360
                        x369 = 6.8636853099151964743e-10 * x61
                        x370 = x304 * x99
                        x371 = x370 * x96
                        x372 = 1.9610529456900561355e-9 * x103
                        x373 = x189 * x358
                        x374 = x28 * x343
                        x375 = 0.000027363582919138474694 * x61
                        x376 = 1.9610529456900561355e-9 * x112
                        x377 = x186 * x369
                        x378 = x328 * x345
                        x379 = x364 * x378
                        x380 = x352 * x7
                        x381 = x310 * x99
                        x382 = 6.2917115340889301014e-10 * x371
                        x383 = x186 * x370
                        x384 = 6.2917115340889301014e-10 * x383
                        x385 = 1.7976318668825514576e-9 * x371
                        x386 = 0.000025083284342543601803 * x99
                        x387 = x189 * x328
                        x388 = 1.7976318668825514576e-9 * x387
                        x389 = x334 * x388
                        x390 = x24 * x374
                        x391 = 0.000025083284342543601803 * x65
                        x392 = 1.7976318668825514576e-9 * x383
                        x393 = x110 * x380
                        x394 = x326 * x388
                        x395 = 5.1360910482358613073e-9 * x387
                        x396 = x103 * x395
                        x397 = x112 * x395
                        x398 = 0.000071666526692981719437 * x390
                        return jnp.asarray(
                            [
                                x0 * x40,
                                x40 * x41,
                                x41 * x43,
                                x0 * x43,
                                -x37 * x47,
                                x34 * x51,
                                -x36 * x55,
                                x31 * x59,
                                -x35 * x63,
                                x32 * x67,
                                -x69 * x71,
                                x73 * x74,
                                -x71 * x75,
                                x24 * x67,
                                -x28 * x63,
                                x22 * x59,
                                -x29 * x55,
                                x27 * x51,
                                -x30 * x47,
                                -x79 * x80,
                                x79 * x81,
                                -x83 * x85,
                                x88 * x90,
                                -x93 * x95,
                                x100 * x98,
                                -x103 * x105,
                                x107 * x108,
                                -x105 * x112,
                                x104 * x114,
                                -x104 * x116,
                                x117 * x98,
                                -x118 * x93,
                                x119 * x88,
                                -x120 * x83,
                                -x122 * x46,
                                x123 * x50,
                                -x124 * x54,
                                x125 * x58,
                                -x126 * x62,
                                x127 * x24,
                                -x128 * x75,
                                x42 * x72 * x74,
                                -x128 * x69,
                                x127 * x32,
                                -x125 * x35 * x62,
                                x124 * x31 * x58,
                                -x123 * x36 * x54,
                                x122 * x34 * x50,
                                -x129 * x19 * x42 * x46,
                                -x120 * x131,
                                x119 * x133,
                                -x118 * x135,
                                x117 * x137,
                                -x116 * x138,
                                x114 * x138,
                                -x112 * x139,
                                x0 * x106 * x108,
                                -x103 * x139,
                                x100 * x137,
                                -x135 * x95,
                                x133 * x90,
                                -x131 * x85,
                                x140 * x81,
                                -x140 * x80,
                                x142 * x144,
                                x144 * x145,
                                x146 * x148,
                                x148 * x149,
                                -x153 * x34,
                                x156 * x36,
                                -x159 * x31,
                                x162 * x35,
                                -x165 * x32,
                                x167 * x69,
                                -x168 * x169,
                                x167 * x75,
                                -x165 * x24,
                                x162 * x28,
                                -x159 * x22,
                                x156 * x29,
                                -x153 * x27,
                                -x171 * x76,
                                x146 * x172,
                                -x175 * x86,
                                x177 * x178,
                                -x181 * x96,
                                x103 * x183,
                                -x184 * x185,
                                x112 * x183,
                                -x181 * x186,
                                x182 * x187,
                                -x175 * x190,
                                x172 * x177 * x9,
                                -x171 * x192,
                                -x152 * x195,
                                x155 * x196,
                                -x158 * x197,
                                x161 * x198,
                                -x199 * x24,
                                x200 * x75,
                                -x169 * x201 * x82,
                                x200 * x69,
                                -x199 * x32,
                                x161 * x197 * x35,
                                -x158 * x196 * x31,
                                x155 * x195 * x36,
                                -x147 * x152 * x203,
                                -x192 * x204,
                                x172 * x206 * x91,
                                -x190 * x207,
                                x187 * x208,
                                -x186 * x209,
                                x112 * x210,
                                -x107 * x185 * x37,
                                x103 * x210,
                                -x209 * x96,
                                x142 * x178 * x211,
                                -x207 * x86,
                                x149 * x172,
                                -x204 * x76,
                                x212 * x214,
                                x214 * x215,
                                x215 * x217,
                                x212 * x217,
                                -x218 * x220,
                                x223 * x31,
                                -x226 * x35,
                                x229 * x32,
                                -x231 * x69,
                                x168 * x232,
                                -x231 * x75,
                                x229 * x24,
                                -x226 * x28,
                                x22 * x223,
                                -x220 * x233,
                                -x194 * x235,
                                x238 * x86,
                                -x215 * x240,
                                x243 * x96,
                                -x103 * x245,
                                x246 * x247,
                                -x112 * x245,
                                x186 * x243,
                                -x244 * x248,
                                x190 * x238,
                                -x241 * x249,
                                -x233 * x251,
                                x222 * x253,
                                -x225 * x254,
                                x24 * x255,
                                -x256 * x75,
                                x232 * x257 * x86,
                                -x256 * x69,
                                x255 * x32,
                                -x225 * x253 * x35,
                                x222 * x250 * x258,
                                -x218 * x251,
                                -x249 * x260,
                                x190 * x261,
                                -x248 * x262,
                                x186 * x263,
                                -x112 * x264,
                                x184 * x247 * x34,
                                -x103 * x264,
                                x263 * x96,
                                -x212 * x240,
                                x261 * x86,
                                -x203 * x235,
                                x218 * x266,
                                x233 * x266,
                                x269 * x270,
                                x270 * x273,
                                -x274 * x276,
                                x279 * x35,
                                -x282 * x32,
                                x284 * x69,
                                -x285 * x73 * x82,
                                x284 * x75,
                                -x24 * x282,
                                x279 * x28,
                                -x276 * x286,
                                -x288 * x86,
                                x267 * x290,
                                -x292 * x96,
                                x103 * x294,
                                -x295 * x296,
                                x112 * x294,
                                -x186 * x292,
                                x293 * x297,
                                -x190 * x288,
                                -x299 * x301,
                                x278 * x305,
                                -x24 * x306,
                                x307 * x75,
                                -x285 * x308 * x91,
                                x307 * x69,
                                -x306 * x32,
                                x278 * x300 * x310,
                                -x301 * x311,
                                -x190 * x313,
                                x297 * x314,
                                -x186 * x315,
                                x112 * x316,
                                -x246 * x296 * x36,
                                x103 * x316,
                                -x315 * x96,
                                x271 * x290,
                                -x313 * x86,
                                x317 * x319,
                                x298 * x319,
                                x298 * x320,
                                x317 * x320,
                                -x322 * x325,
                                x326 * x330,
                                -x332 * x69,
                                x201 * x333 * x86,
                                -x332 * x75,
                                x330 * x334,
                                -x325 * x335,
                                -x336 * x337,
                                x339 * x96,
                                -x103 * x342,
                                x343 * x344,
                                -x112 * x342,
                                x186 * x339,
                                -x336 * x346,
                                -x335 * x348,
                                x334 * x350,
                                -x351 * x75,
                                x333 * x352 * x96,
                                -x351 * x69,
                                x326 * x350,
                                -x322 * x348,
                                -x346 * x353,
                                x186 * x354,
                                -x112 * x355,
                                x295 * x31 * x344,
                                -x103 * x355,
                                x354 * x96,
                                -x337 * x353,
                                x322 * x357,
                                x335 * x357,
                                x345 * x356 * x358,
                                x115 * x356 * x359,
                                -x361 * x362,
                                x363 * x365,
                                -x257 * x366 * x91,
                                x365 * x367,
                                -x362 * x368,
                                -x369 * x371,
                                x372 * x373,
                                -x374 * x375,
                                x373 * x376,
                                -x370 * x377,
                                -x368 * x378,
                                x367 * x379,
                                -x101 * x366 * x380,
                                x363 * x379,
                                -x361 * x378,
                                -x377 * x381,
                                x359 * x376,
                                -x343 * x35 * x375,
                                x359 * x372,
                                -x369 * x381 * x96,
                                x326 * x382,
                                x334 * x382,
                                x334 * x384,
                                x326 * x384,
                                -x363 * x385,
                                x308 * x386 * x96,
                                -x367 * x385,
                                -x103 * x389,
                                x390 * x391,
                                -x112 * x389,
                                -x367 * x392,
                                x14 * x386 * x393,
                                -x363 * x392,
                                -x112 * x394,
                                x32 * x374 * x391,
                                -x103 * x394,
                                x363 * x396,
                                x367 * x396,
                                x367 * x397,
                                x363 * x397,
                                -0.000071666526692981719437 * x103 * x352,
                                -x398 * x75,
                                -0.000071666526692981719437 * x109 * x393,
                                -x398 * x69,
                                x106 * x72,
                            ]
                        )

                case 324:

                    def shape_functions(xi):
                        x0 = 17.0 * xi[1]
                        x1 = x0 + 15.0
                        x2 = xi[0] - 1.0
                        x3 = x1 * x2
                        x4 = 17.0 * xi[0]
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
                        x37 = (
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
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x31
                            * x32
                            * x33
                            * x34
                            * x35
                            * x36
                            * x5
                            * x6
                            * x7
                            * x8
                            * x9
                        )
                        x38 = 1.3296610891588984942e-37 * x37
                        x39 = xi[0] + 1.0
                        x40 = x1 * x39
                        x41 = xi[1] + 1.0
                        x42 = x19 * x41
                        x43 = x40 * x42
                        x44 = (
                            x10
                            * x11
                            * x12
                            * x13
                            * x14
                            * x15
                            * x16
                            * x17
                            * x18
                            * x21
                            * x22
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x31
                            * x32
                            * x33
                            * x34
                            * x35
                            * x36
                            * x5
                            * x6
                            * x7
                            * x8
                            * x9
                        )
                        x45 = 1.3296610891588984942e-37 * x44
                        x46 = x3 * x42
                        x47 = 3.8427205476692166482e-35 * x39
                        x48 = x20 * x3
                        x49 = x47 * x48
                        x50 = (
                            x10
                            * x11
                            * x12
                            * x13
                            * x14
                            * x15
                            * x18
                            * x21
                            * x22
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x31
                            * x32
                            * x33
                            * x34
                            * x35
                            * x36
                            * x5
                            * x6
                            * x7
                            * x8
                            * x9
                        )
                        x51 = x48 * x50
                        x52 = x16 * x39
                        x53 = 3.0741764381353733186e-34 * x52
                        x54 = x19 * x53
                        x55 = (
                            x10
                            * x11
                            * x12
                            * x13
                            * x14
                            * x21
                            * x22
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x31
                            * x32
                            * x33
                            * x34
                            * x35
                            * x36
                            * x5
                            * x6
                            * x7
                            * x8
                            * x9
                        )
                        x56 = x48 * x55
                        x57 = 1.5370882190676866593e-33 * x17
                        x58 = x18 * x52
                        x59 = x57 * x58
                        x60 = x19 * x59
                        x61 = x32 * x48
                        x62 = x19 * x61
                        x63 = (
                            x10
                            * x11
                            * x12
                            * x14
                            * x21
                            * x22
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x31
                            * x33
                            * x34
                            * x35
                            * x36
                            * x5
                            * x6
                            * x7
                            * x8
                            * x9
                        )
                        x64 = x15 * x17
                        x65 = x58 * x64
                        x66 = x63 * x65
                        x67 = 5.3798087667369033075e-33 * x66
                        x68 = (
                            x10
                            * x12
                            * x13
                            * x14
                            * x21
                            * x22
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x33
                            * x34
                            * x35
                            * x36
                            * x5
                            * x6
                            * x7
                            * x8
                            * x9
                        )
                        x69 = x31 * x65
                        x70 = x62 * x69
                        x71 = 1.39875027935159486e-32 * x70
                        x72 = (
                            x10
                            * x11
                            * x12
                            * x13
                            * x14
                            * x21
                            * x22
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x33
                            * x34
                            * x5
                            * x6
                            * x7
                            * x8
                        )
                        x73 = x35 * x36
                        x74 = x70 * x73
                        x75 = 2.7975005587031897199e-32 * x74
                        x76 = (
                            x10
                            * x11
                            * x12
                            * x13
                            * x14
                            * x21
                            * x22
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x5
                            * x6
                            * x8
                            * x9
                        )
                        x77 = x33 * x34
                        x78 = x74 * x77
                        x79 = 4.3960723065335838456e-32 * x78
                        x80 = (
                            x10
                            * x11
                            * x12
                            * x21
                            * x22
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x6
                            * x7
                            * x8
                            * x9
                        )
                        x81 = x13 * x14
                        x82 = x78 * x81
                        x83 = 5.495090383166979807e-32 * x82
                        x84 = (
                            x10
                            * x11
                            * x12
                            * x22
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x5
                            * x6
                            * x7
                            * x8
                            * x9
                        )
                        x85 = x29 * x30
                        x86 = x84 * x85
                        x87 = (
                            x10
                            * x12
                            * x22
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x5
                            * x6
                            * x7
                            * x8
                            * x85
                            * x9
                        )
                        x88 = x21 * x81
                        x89 = x11 * x88
                        x90 = x87 * x89
                        x91 = (
                            x10
                            * x22
                            * x23
                            * x24
                            * x26
                            * x27
                            * x28
                            * x6
                            * x7
                            * x8
                            * x85
                            * x89
                            * x9
                        )
                        x92 = x5 * x77
                        x93 = x12 * x92
                        x94 = x91 * x93
                        x95 = (
                            x22
                            * x23
                            * x24
                            * x25
                            * x26
                            * x28
                            * x6
                            * x8
                            * x85
                            * x89
                            * x9
                            * x93
                        )
                        x96 = x7 * x73
                        x97 = x10 * x96
                        x98 = x95 * x97
                        x99 = 5.3798087667369033075e-33 * x84
                        x100 = x30 * x99
                        x101 = x11 * x68
                        x102 = x101 * x15
                        x103 = x61 * x96
                        x104 = x34 * x76
                        x105 = x31 * x64
                        x106 = x105 * x18
                        x107 = x104 * x106
                        x108 = x106 * x16
                        x109 = x32 * x72
                        x110 = x109 * x9
                        x111 = x110 * x36
                        x112 = x108 * x111
                        x113 = x41 * x47
                        x114 = x20 * x43
                        x115 = x114 * x16
                        x116 = 3.0741764381353733186e-34 * x55 * x64
                        x117 = x50 * x57
                        x118 = x108 * x114
                        x119 = x32 * x73
                        x120 = x119 * x80
                        x121 = x13 * x92
                        x122 = x120 * x121
                        x123 = 5.3798087667369033075e-33 * x122
                        x124 = x118 * x25
                        x125 = 1.39875027935159486e-32 * x124
                        x126 = x119 * x91
                        x127 = x126 * x92
                        x128 = x118 * x96
                        x129 = x27 * x32
                        x130 = 2.7975005587031897199e-32 * x129
                        x131 = x130 * x95
                        x132 = x22 * x23 * x24 * x26 * x28 * x6 * x85 * x9 * x93
                        x133 = x89 * x97
                        x134 = x129 * x133
                        x135 = x124 * x134
                        x136 = 4.3960723065335838456e-32 * x135
                        x137 = x22 * x23 * x24 * x26 * x28 * x85 * x9
                        x138 = x8 * x93
                        x139 = 5.495090383166979807e-32 * x138
                        x140 = x135 * x139
                        x141 = x23 * x24 * x26 * x28 * x9
                        x142 = x6 * x85
                        x143 = x141 * x142
                        x144 = x142 * x23 * x28 * x9
                        x145 = x138 * x22
                        x146 = x145 * x26
                        x147 = x144 * x146
                        x148 = x133 * x24
                        x149 = x144 * x145
                        x150 = x148 * x149
                        x151 = x130 * x150
                        x152 = x134 * x24
                        x153 = x142 * x146 * x23 * x9
                        x154 = x152 * x153
                        x155 = x119 * x77
                        x156 = x29 * x88
                        x157 = x155 * x156
                        x158 = x157 * x99
                        x159 = x13 * x63
                        x160 = x15 * x159
                        x161 = x160 * x18 * x57
                        x162 = x32 * x76
                        x163 = x162 * x33
                        x164 = 3.0741764381353733186e-34 * x163
                        x165 = x110 * x35
                        x166 = 3.8427205476692166482e-35 * x165
                        x167 = x32 * x46
                        x168 = x167 * x96
                        x169 = x107 * x168
                        x170 = x46 * x69
                        x171 = x155 * x170
                        x172 = x171 * x88
                        x173 = x69 * x98
                        x174 = 1.39875027935159486e-32 * x167
                        x175 = 2.7975005587031897199e-32 * x170
                        x176 = x119 * x94
                        x177 = 4.3960723065335838456e-32 * x171
                        x178 = 5.495090383166979807e-32 * x81
                        x179 = x171 * x86
                        x180 = x120 * x77
                        x181 = x170 * x180
                        x182 = x109 * x73
                        x183 = x68 * x69
                        x184 = x46 * x55
                        x185 = x46 * x50
                        x186 = x20 * x46
                        x187 = x108 * x186
                        x188 = x187 * x96
                        x189 = x16 * x186
                        x190 = x187 * x25
                        x191 = 1.39875027935159486e-32 * x190
                        x192 = x134 * x190
                        x193 = 4.3960723065335838456e-32 * x192
                        x194 = x139 * x192
                        x195 = x2 * x41
                        x196 = 1.1105462382764036113e-32 * x20
                        x197 = x2 * x42
                        x198 = x197 * x69
                        x199 = x72 * x9
                        x200 = x35 * x69
                        x201 = x41 * x61
                        x202 = x197 * x20
                        x203 = 8.8843699062112288907e-32 * x52
                        x204 = 4.4421849531056144454e-31 * x202
                        x205 = x17 * x58
                        x206 = x202 * x32
                        x207 = 4.0423883073261091453e-30 * x206
                        x208 = x198 * x20
                        x209 = 8.0847766146522182906e-30 * x208
                        x210 = x155 * x208
                        x211 = 1.2704648965882057314e-29 * x210
                        x212 = 1.5880811207352571642e-29 * x81
                        x213 = x212 * x86
                        x214 = 1.5547647335869650559e-30 * x84
                        x215 = x30 * x88
                        x216 = x214 * x215
                        x217 = x101 * x65
                        x218 = 8.8843699062112288907e-32 * x104
                        x219 = x111 * x186
                        x220 = 4.4421849531056144454e-31 * x39
                        x221 = x20 * x36
                        x222 = x221 * x69
                        x223 = x167 * x80
                        x224 = 1.5547647335869650559e-30 * x223
                        x225 = x167 * x222
                        x226 = 4.0423883073261091453e-30 * x25
                        x227 = x226 * x91 * x92
                        x228 = x7 * x95
                        x229 = x170 * x221
                        x230 = x129 * x229
                        x231 = 8.0847766146522182906e-30 * x230
                        x232 = x132 * x89
                        x233 = 1.2704648965882057314e-29 * x25
                        x234 = x10 * x7
                        x235 = x230 * x234
                        x236 = x233 * x235
                        x237 = x137 * x138
                        x238 = 1.5880811207352571642e-29 * x237
                        x239 = x25 * x89
                        x240 = x235 * x239
                        x241 = x138 * x143
                        x242 = 1.5880811207352571642e-29 * x241
                        x243 = x149 * x24
                        x244 = x239 * x243
                        x245 = x153 * x226
                        x246 = x156 * x214 * x77
                        x247 = x20 * x200
                        x248 = x167 * x247
                        x249 = x165 * x186
                        x250 = 4.4421849531056144454e-31 * x249
                        x251 = x248 * x77
                        x252 = x170 * x20
                        x253 = x252 * x35
                        x254 = x23 * x87
                        x255 = x254 * x88
                        x256 = x14 * x92
                        x257 = x205 * x31
                        x258 = x15 * x31 * x58
                        x259 = x33 * x69
                        x260 = x103 * x41
                        x261 = x201 * x69
                        x262 = x261 * x73
                        x263 = x261 * x27
                        x264 = x25 * x263
                        x265 = x133 * x147
                        x266 = x233 * x263
                        x267 = x133 * x264
                        x268 = x132 * x133
                        x269 = x69 * x95
                        x270 = x20 * x52
                        x271 = 7.1074959249689831126e-31 * x270
                        x272 = 7.1074959249689831126e-31 * x96
                        x273 = x163 * x186
                        x274 = 3.5537479624844915563e-30 * x270
                        x275 = x17 * x184
                        x276 = x167 * x270
                        x277 = x63 * x64
                        x278 = 3.2339106458608873162e-29 * x105 * x276
                        x279 = 6.4678212917217746324e-29 * x186
                        x280 = x105 * x52
                        x281 = x279 * x280
                        x282 = 1.0163719172705645851e-28 * x76
                        x283 = x186 * x280
                        x284 = x155 * x283
                        x285 = 1.2704648965882057314e-28 * x81
                        x286 = x285 * x86
                        x287 = 1.0163719172705645851e-28 * x90
                        x288 = 1.2438117868695720447e-29 * x84
                        x289 = x215 * x288
                        x290 = x101 * x167 * x64
                        x291 = x20 * x39
                        x292 = 3.5537479624844915563e-30 * x291
                        x293 = x252 * x34
                        x294 = x293 * x5
                        x295 = 1.2438117868695720447e-29 * x120
                        x296 = x25 * x294
                        x297 = 3.2339106458608873162e-29 * x296
                        x298 = x12 * x8
                        x299 = x137 * x298
                        x300 = x129 * x96
                        x301 = x239 * x6
                        x302 = x300 * x301
                        x303 = x299 * x302
                        x304 = x134 * x296
                        x305 = 1.0163719172705645851e-28 * x304
                        x306 = x137 * x6
                        x307 = x12 * x306
                        x308 = 1.2704648965882057314e-28 * x304
                        x309 = x143 * x298
                        x310 = x22 * x298
                        x311 = x144 * x310
                        x312 = x26 * x311
                        x313 = 6.4678212917217746324e-29 * x152
                        x314 = x152 * x9
                        x315 = x142 * x23
                        x316 = x26 * x310 * x315
                        x317 = x314 * x316
                        x318 = 3.5537479624844915563e-30 * x96
                        x319 = x273 * x318
                        x320 = x252 * x33
                        x321 = x119 * x320
                        x322 = x25 * x5
                        x323 = x167 * x20
                        x324 = x133 * x323
                        x325 = x324 * x6
                        x326 = x320 * x5
                        x327 = x186 * x258
                        x328 = x25 * x327
                        x329 = 3.2339106458608873162e-29 * x328
                        x330 = x258 * x279
                        x331 = x152 * x25
                        x332 = x149 * x331
                        x333 = x134 * x328
                        x334 = 1.0163719172705645851e-28 * x333
                        x335 = 1.2704648965882057314e-28 * x333
                        x336 = x300 * x95
                        x337 = x18 * x291
                        x338 = 1.7768739812422457781e-29 * x337
                        x339 = 1.7768739812422457781e-29 * x186
                        x340 = 1.6169553229304436581e-28 * x68
                        x341 = x106 * x167 * x291
                        x342 = 3.2339106458608873162e-28 * x186
                        x343 = x106 * x39
                        x344 = x342 * x343
                        x345 = 5.0818595863528229255e-28 * x76
                        x346 = x186 * x343
                        x347 = x155 * x346
                        x348 = 6.3523244829410286569e-28 * x81
                        x349 = x348 * x86
                        x350 = 5.0818595863528229255e-28 * x90
                        x351 = 1.6169553229304436581e-28 * x98
                        x352 = 6.2190589343478602235e-29 * x84
                        x353 = x215 * x352
                        x354 = x186 * x65
                        x355 = 6.2190589343478602235e-29 * x122
                        x356 = x25 * x354
                        x357 = 1.6169553229304436581e-28 * x356
                        x358 = x342 * x65
                        x359 = x134 * x356
                        x360 = 5.0818595863528229255e-28 * x359
                        x361 = 6.3523244829410286569e-28 * x359
                        x362 = x157 * x352
                        x363 = x252 * x73
                        x364 = x363 * x77
                        x365 = 3.2339106458608873162e-28 * x363
                        x366 = x186 * x257
                        x367 = x25 * x366
                        x368 = 1.6169553229304436581e-28 * x367
                        x369 = x257 * x342
                        x370 = x134 * x367
                        x371 = 5.0818595863528229255e-28 * x370
                        x372 = 6.3523244829410286569e-28 * x370
                        x373 = x252 * x92
                        x374 = 2.1766706270217510782e-28 * x20 * x84
                        x375 = x13 * x21
                        x376 = x171 * x375
                        x377 = x171 * x29
                        x378 = x14 * x21
                        x379 = x20 * x376
                        x380 = x129 * x97
                        x381 = x24 * x380
                        x382 = x28 * x315
                        x383 = x146 * x25
                        x384 = x382 * x383
                        x385 = 1.1318687260513105607e-27 * x252
                        x386 = x11 * x375
                        x387 = x385 * x386
                        x388 = x25 * x252
                        x389 = 1.7786508552234880239e-27 * x388
                        x390 = x132 * x8
                        x391 = x386 * x390
                        x392 = x119 * x27
                        x393 = x10 * x392
                        x394 = 2.2233135690293600299e-27 * x20
                        x395 = x13 * x394
                        x396 = 5.6593436302565528034e-28 * x25
                        x397 = x396 * x69
                        x398 = x373 * x8
                        x399 = x22 * x398
                        x400 = x141 * x30
                        x401 = x134 * x400
                        x402 = x401 * x6
                        x403 = x145 * x385
                        x404 = x22 * x93
                        x405 = 2.2233135690293600299e-27 * x388
                        x406 = x134 * x6
                        x407 = x23 * x28
                        x408 = x30 * x407
                        x409 = 1.7786508552234880239e-27 * x252
                        x410 = x383 * x409
                        x411 = x314 * x6
                        x412 = x252 * x383
                        x413 = x141 * x29
                        x414 = x145 * x413
                        x415 = x406 * x413
                        x416 = x301 * x393
                        x417 = x146 * x331
                        x418 = x407 * x417
                        x419 = 5.6593436302565528034e-28 * x388
                        x420 = x380 * x419
                        x421 = x11 * x378
                        x422 = x153 * x381
                        x423 = x380 * x421
                        x424 = x25 * x385
                        x425 = x389 * x423
                        x426 = x405 * x423
                        x427 = x300 * x390
                        x428 = x306 * x8
                        x429 = 1.4714293438667037289e-27 * x25
                        x430 = x380 * x88
                        x431 = x306 * x398
                        x432 = x429 * x69
                        x433 = x388 * x422
                        x434 = x26 * x399
                        x435 = x331 * x382
                        x436 = 5.7806152794763360777e-27 * x388
                        x437 = x436 * x77
                        x438 = 5.7806152794763360777e-27 * x25
                        x439 = x11 * x81
                        x440 = x380 * x439
                        x441 = 4.6244922235810688622e-27 * x25
                        x442 = x142 * x28
                        x443 = x314 * x442
                        x444 = 2.9428586877334074578e-27 * x252
                        x445 = x323 * x69
                        x446 = x441 * x445
                        x447 = x324 * x438 * x69
                        x448 = x314 * x412
                        x449 = x252 * x393
                        x450 = x239 * x24
                        x451 = x388 * x430
                        x452 = 4.6244922235810688622e-27 * x451
                        x453 = x430 * x436
                        x454 = x388 * x427
                        x455 = 5.8857173754668149155e-27 * x252
                        x456 = x300 * x455
                        x457 = x146 * x450
                        x458 = 9.2489844471621377244e-27 * x252
                        x459 = 1.1561230558952672155e-26 * x252
                        x460 = x459 * x77
                        x461 = x442 * x9
                        x462 = x134 * x458
                        x463 = x134 * x459
                        x464 = x388 * x443
                        x465 = x388 * x440
                        x466 = x26 * x435
                        x467 = x26 * x464
                        x468 = x239 * x449
                        x469 = x134 * x388 * x77
                        x470 = 1.8167648021211341959e-26 * x469
                        x471 = 1.8167648021211341959e-26 * x465
                        x472 = 1.8167648021211341959e-26 * x468
                        x473 = 2.2709560026514177448e-26 * x469
                        x474 = 2.2709560026514177448e-26 * x465
                        return jnp.asarray(
                            [
                                x3 * x38,
                                -x38 * x40,
                                x43 * x45,
                                -x45 * x46,
                                -x44 * x49,
                                x51 * x54,
                                -x56 * x60,
                                x62 * x67,
                                -x68 * x71,
                                x72 * x75,
                                -x76 * x79,
                                x80 * x83,
                                -x83 * x86,
                                x79 * x90,
                                -x75 * x94,
                                x71 * x98,
                                -x100 * x21 * x82,
                                x102 * x60 * x61,
                                -x103 * x107 * x54,
                                x112 * x19 * x49,
                                x113 * x37,
                                -x115 * x116,
                                x114 * x117,
                                -x118 * x123,
                                x125 * x127,
                                -x128 * x131,
                                x132 * x136,
                                -x137 * x140,
                                x140 * x143,
                                -x136 * x147,
                                x124 * x151,
                                -x125 * x154,
                                x118 * x158,
                                -x115 * x161,
                                x128 * x164,
                                -x118 * x166,
                                -x112 * x46 * x47,
                                x169 * x53,
                                -x102 * x167 * x59,
                                x100 * x172,
                                -x173 * x174,
                                x175 * x176,
                                -x177 * x90,
                                x178 * x179,
                                -x178 * x181,
                                x177 * x76,
                                -x175 * x182,
                                x174 * x183,
                                -x167 * x67,
                                x184 * x59,
                                -x185 * x53,
                                x113 * x3 * x44,
                                x166 * x187,
                                -x164 * x188,
                                x161 * x189,
                                -x158 * x187,
                                x154 * x191,
                                -x151 * x190,
                                x147 * x193,
                                -x143 * x194,
                                x137 * x194,
                                -x132 * x193,
                                x131 * x188,
                                -x127 * x191,
                                x123 * x187,
                                -x117 * x186,
                                x116 * x189,
                                -3.8427205476692166482e-35 * x195 * x37,
                                x195 * x196 * x39 * x44,
                                -x111 * x196 * x198,
                                x110 * x170 * x196,
                                -1.1105462382764036113e-32 * x199 * x200 * x201,
                                -x202 * x203 * x50,
                                x204 * x205 * x55,
                                -1.5547647335869650559e-30 * x206 * x66,
                                x183 * x207,
                                -x182 * x209,
                                x211 * x76,
                                -x180 * x208 * x212,
                                x210 * x213,
                                -x211 * x90,
                                x176 * x209,
                                -x173 * x207,
                                x210 * x216,
                                -x204 * x217 * x32,
                                x208 * x218 * x32 * x96,
                                x105 * x203 * x219,
                                -x106 * x219 * x220,
                                x121 * x222 * x224,
                                -x225 * x227,
                                x228 * x231,
                                -x232 * x236,
                                x238 * x240,
                                -x240 * x242,
                                x147 * x236 * x89,
                                -x231 * x234 * x244,
                                x235 * x24 * x245 * x89,
                                -x225 * x246,
                                4.4421849531056144454e-31 * x199 * x229,
                                -8.8843699062112288907e-32 * x163 * x229 * x7,
                                -x218 * x248 * x7,
                                x250 * x65,
                                -x216 * x251,
                                4.0423883073261091453e-30 * x10 * x228 * x248,
                                -8.0847766146522182906e-30 * x248 * x94,
                                1.2704648965882057314e-29 * x251 * x90,
                                -x213 * x251,
                                x212 * x223 * x247 * x77,
                                -1.2704648965882057314e-29 * x162 * x253 * x77,
                                8.0847766146522182906e-30 * x109 * x253,
                                -4.0423883073261091453e-30 * x251 * x255,
                                x224 * x247 * x256,
                                -x250 * x257,
                                8.8843699062112288907e-32 * x249 * x258,
                                8.8843699062112288907e-32 * x259 * x260 * x76,
                                -4.4421849531056144454e-31 * x13 * x41 * x48 * x66,
                                x246 * x262,
                                -x148 * x245 * x263,
                                8.0847766146522182906e-30 * x150 * x264,
                                -x265 * x266,
                                x242 * x267,
                                -x238 * x267,
                                x266 * x268,
                                -8.0847766146522182906e-30 * x260 * x269 * x27,
                                x227 * x262,
                                -1.5547647335869650559e-30 * x121 * x262 * x80,
                                x17 * x220 * x41 * x51,
                                -x203 * x41 * x56 * x64,
                                x15 * x184 * x271,
                                -x104 * x105 * x168 * x271,
                                x162 * x252 * x272,
                                -x258 * x272 * x273,
                                -x274 * x275,
                                1.2438117868695720447e-29 * x276 * x277,
                                -x278 * x68,
                                x182 * x281,
                                -x282 * x284,
                                x180 * x283 * x285,
                                -x284 * x286,
                                x284 * x287,
                                -x176 * x281,
                                x278 * x98,
                                -x284 * x289,
                                x274 * x290,
                                x169 * x292,
                                -x13 * x294 * x295,
                                x126 * x297,
                                -6.4678212917217746324e-29 * x294 * x303,
                                x305 * x307,
                                -x299 * x308,
                                x308 * x309,
                                -x305 * x312,
                                x296 * x311 * x313,
                                -x297 * x317,
                                x119 * x156 * x288 * x293,
                                -x104 * x252 * x318,
                                -x319 * x65,
                                x289 * x321,
                                -3.2339106458608873162e-29 * x259 * x299 * x322 * x325,
                                6.4678212917217746324e-29 * x12 * x126 * x326,
                                -x287 * x321,
                                x286 * x321,
                                -x120 * x285 * x320,
                                x282 * x321,
                                -x28 * x313 * x316 * x320 * x322,
                                3.2339106458608873162e-29 * x255 * x321,
                                -x14 * x295 * x326,
                                x257 * x319,
                                3.5537479624844915563e-30 * x160 * x186 * x58,
                                -x157 * x288 * x327,
                                x154 * x329,
                                -x330 * x332,
                                x147 * x334,
                                -x241 * x335,
                                x237 * x335,
                                -x132 * x334,
                                x330 * x336,
                                -x127 * x329,
                                1.2438117868695720447e-29 * x122 * x327,
                                -x185 * x292,
                                x275 * x338,
                                -x290 * x338,
                                x217 * x339,
                                -x159 * x205 * x339,
                                -6.2190589343478602235e-29 * x167 * x277 * x337,
                                x340 * x341,
                                -x182 * x344,
                                x345 * x347,
                                -x180 * x346 * x348,
                                x347 * x349,
                                -x347 * x350,
                                x176 * x344,
                                -x341 * x351,
                                x347 * x353,
                                x354 * x355,
                                -x127 * x357,
                                x336 * x358,
                                -x132 * x360,
                                x237 * x361,
                                -x241 * x361,
                                x147 * x360,
                                -x332 * x358,
                                x154 * x357,
                                -x354 * x362,
                                -x353 * x364,
                                x252 * x351,
                                -x365 * x94,
                                x350 * x364,
                                -x349 * x364,
                                x348 * x364 * x80,
                                -x345 * x364,
                                x365 * x72,
                                -x252 * x340,
                                6.2190589343478602235e-29 * x186 * x66,
                                x362 * x366,
                                -x154 * x368,
                                x332 * x369,
                                -x147 * x371,
                                x241 * x372,
                                -x237 * x372,
                                x132 * x371,
                                -x336 * x369,
                                x127 * x368,
                                -x355 * x366,
                                2.1766706270217510782e-28 * x120 * x373,
                                -x30 * x374 * x376,
                                x172 * x374,
                                -x374 * x377 * x378,
                                -5.6593436302565528034e-28 * x254 * x379,
                                x381 * x384 * x387,
                                -x389 * x391 * x393,
                                x181 * x395,
                                -x179 * x395,
                                1.7786508552234880239e-27 * x11 * x379 * x87,
                                -x380 * x387 * x390,
                                x323 * x391 * x397 * x97,
                                x396 * x399 * x402,
                                -x302 * x400 * x403,
                                x389 * x402 * x404,
                                -x145 * x401 * x405,
                                x138 * x402 * x405,
                                -x406 * x408 * x410 * x9,
                                x25 * x403 * x408 * x411,
                                -5.6593436302565528034e-28 * x23 * x30 * x411 * x412,
                                -x325 * x397 * x414,
                                x403 * x415,
                                -x28 * x29 * x410 * x411,
                                x377 * x394 * x81 * x84,
                                -x310 * x405 * x415 * x77,
                                x409 * x414 * x416,
                                -x29 * x385 * x418 * x6,
                                x141 * x145 * x156 * x420 * x6,
                                x419 * x421 * x422,
                                -x243 * x423 * x424,
                                x147 * x425,
                                -x241 * x426,
                                x237 * x426,
                                -x132 * x425,
                                x421 * x424 * x427,
                                -x11 * x21 * x256 * x420 * x428,
                                x429 * x430 * x431,
                                -x324 * x428 * x432 * x92,
                                x148 * x153 * x323 * x432,
                                -1.4714293438667037289e-27 * x433 * x88,
                                -2.9428586877334074578e-27 * x434 * x435,
                                4.6244922235810688622e-27 * x137 * x398 * x416,
                                -x126 * x437,
                                x431 * x438 * x440,
                                -x434 * x441 * x443,
                                x127 * x444,
                                2.9428586877334074578e-27 * x168 * x20 * x269,
                                -x268 * x446,
                                x237 * x447,
                                -x241 * x447,
                                x265 * x446,
                                -2.9428586877334074578e-27 * x150 * x25 * x445,
                                -x154 * x444,
                                4.6244922235810688622e-27 * x142 * x448,
                                -5.7806152794763360777e-27 * x433 * x439,
                                x317 * x437,
                                -4.6244922235810688622e-27 * x153 * x449 * x450,
                                x315 * x417 * x444,
                                2.9428586877334074578e-27 * x243 * x451,
                                -x147 * x452,
                                x241 * x453,
                                -x237 * x453,
                                x132 * x452,
                                -2.9428586877334074578e-27 * x454 * x88,
                                x382 * x456 * x457,
                                -x232 * x456 * x8,
                                x149 * x152 * x455,
                                -x145 * x435 * x455,
                                -x392 * x458 * x95,
                                x303 * x460,
                                -1.1561230558952672155e-26 * x439 * x454,
                                x300 * x457 * x458 * x461,
                                x132 * x462,
                                -x237 * x463,
                                x241 * x463,
                                -x147 * x462,
                                -9.2489844471621377244e-27 * x145 * x464,
                                1.1561230558952672155e-26 * x243 * x465,
                                -x311 * x331 * x460,
                                9.2489844471621377244e-27 * x244 * x449,
                                x384 * x462,
                                -x138 * x459 * x466,
                                x418 * x459 * x85,
                                -x404 * x458 * x466,
                                1.4534118416969073567e-26 * x232 * x388 * x393,
                                -1.4534118416969073567e-26 * x404 * x467,
                                1.4534118416969073567e-26 * x134 * x412 * x461,
                                -1.4534118416969073567e-26 * x147 * x468,
                                -x307 * x470,
                                x132 * x471,
                                1.8167648021211341959e-26 * x28 * x448 * x85,
                                -1.8167648021211341959e-26 * x138 * x467,
                                -x147 * x471,
                                x312 * x470,
                                x241 * x472,
                                -x237 * x472,
                                x299 * x473,
                                -x237 * x474,
                                x241 * x474,
                                -x309 * x473,
                            ]
                        )

                case 361:

                    def shape_functions(xi):
                        x0 = xi[0] - 1.0
                        x1 = xi[1] - 1.0
                        x2 = 3.0 * xi[1]
                        x3 = x2 + 1.0
                        x4 = 9.0 * xi[1]
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
                        x20 = (
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
                            * x3
                            * x5
                            * x6
                            * x7
                            * x8
                            * x9
                            * xi[1]
                        )
                        x21 = x1 * x20
                        x22 = 3.0 * xi[0]
                        x23 = x22 + 1.0
                        x24 = 9.0 * xi[0]
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
                        x40 = (
                            x23
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x31
                            * x32
                            * x33
                            * x34
                            * x35
                            * x36
                            * x37
                            * x38
                            * x39
                            * xi[0]
                        )
                        x41 = 1.0501661969785547778e-24 * x40
                        x42 = x21 * x41
                        x43 = xi[0] + 1.0
                        x44 = xi[1] + 1.0
                        x45 = x20 * x41 * x44
                        x46 = x21 * x43
                        x47 = (
                            x0
                            * x23
                            * x25
                            * x26
                            * x27
                            * x28
                            * x30
                            * x31
                            * x32
                            * x33
                            * x34
                            * x35
                            * x36
                            * x38
                            * x39
                            * xi[0]
                        )
                        x48 = 1.70126923910525874e-22 * x47
                        x49 = x46 * x48
                        x50 = x29 * x46
                        x51 = (
                            x0
                            * x23
                            * x25
                            * x26
                            * x27
                            * x28
                            * x30
                            * x32
                            * x33
                            * x34
                            * x35
                            * x36
                            * x37
                            * x38
                            * xi[0]
                        )
                        x52 = 1.446078853239469929e-21 * x51
                        x53 = x50 * x52
                        x54 = x31 * x50
                        x55 = (
                            x0
                            * x23
                            * x25
                            * x27
                            * x28
                            * x30
                            * x32
                            * x33
                            * x35
                            * x36
                            * x37
                            * x38
                            * x39
                            * xi[0]
                        )
                        x56 = 2.570806850203502096e-21 * x55
                        x57 = x54 * x56
                        x58 = x26 * x54
                        x59 = (
                            x0
                            * x23
                            * x25
                            * x27
                            * x28
                            * x32
                            * x33
                            * x34
                            * x35
                            * x36
                            * x37
                            * x39
                            * xi[0]
                        )
                        x60 = 2.892157706478939858e-20 * x59
                        x61 = x58 * x60
                        x62 = x30 * x58
                        x63 = (
                            x0
                            * x23
                            * x25
                            * x27
                            * x32
                            * x33
                            * x34
                            * x35
                            * x37
                            * x38
                            * x39
                            * xi[0]
                        )
                        x64 = 8.0980415781410316025e-20 * x63
                        x65 = x62 * x64
                        x66 = x28 * x62
                        x67 = (
                            x0
                            * x25
                            * x27
                            * x33
                            * x34
                            * x35
                            * x36
                            * x37
                            * x38
                            * x39
                            * xi[0]
                        )
                        x68 = 5.8485855842129672685e-20 * x67
                        x69 = x66 * x68
                        x70 = x23 * x66
                        x71 = x0 * x25 * x32 * x33 * x34 * x36 * x37 * x38 * x39 * xi[0]
                        x72 = 3.0078440147380974524e-19 * x71
                        x73 = x70 * x72
                        x74 = x0 * x32 * x34 * x35 * x36 * x37 * x38 * x39 * xi[0]
                        x75 = x33 * x74
                        x76 = 4.135785520264883997e-19 * x27
                        x77 = x70 * x76
                        x78 = (
                            -1139827.4301937679369 * xi[0] ** 18
                            + 4010503.9210521464445 * xi[0] ** 16
                            - 5723632.7564645448023 * xi[0] ** 14
                            + 4288221.5882976921237 * xi[0] ** 12
                            - 1825541.0625608358578 * xi[0] ** 10
                            + 447065.31380067163584 * xi[0] ** 8
                            - 60894.246929607780612 * xi[0] ** 6
                            + 4228.3941844706632653 * xi[0] ** 4
                            - 124.72118622448979592 * xi[0] ** 2
                            + 1.0
                        )
                        x79 = x1 * x78
                        x80 = 1.0247761692089423182e-12 * x20
                        x81 = x25 * x74
                        x82 = (
                            x10
                            * x12
                            * x13
                            * x14
                            * x15
                            * x16
                            * x17
                            * x18
                            * x19
                            * x3
                            * x44
                            * x5
                            * x6
                            * x7
                            * x8
                            * xi[1]
                        )
                        x83 = x1 * x40
                        x84 = x43 * x83
                        x85 = x82 * x84
                        x86 = 1.70126923910525874e-22 * x11
                        x87 = 1.446078853239469929e-21 * x9
                        x88 = (
                            x10
                            * x12
                            * x13
                            * x14
                            * x15
                            * x16
                            * x18
                            * x19
                            * x3
                            * x44
                            * x5
                            * x7
                            * x8
                            * x9
                            * xi[1]
                        )
                        x89 = x84 * x88
                        x90 = x11 * x17
                        x91 = 2.570806850203502096e-21 * x90
                        x92 = (
                            x12
                            * x13
                            * x14
                            * x15
                            * x16
                            * x18
                            * x3
                            * x44
                            * x5
                            * x7
                            * x8
                            * x90
                            * xi[1]
                        )
                        x93 = x6 * x84
                        x94 = x92 * x93
                        x95 = x19 * x9
                        x96 = 2.892157706478939858e-20 * x95
                        x97 = (
                            x12
                            * x13
                            * x15
                            * x16
                            * x18
                            * x3
                            * x44
                            * x5
                            * x7
                            * x95
                            * xi[1]
                        )
                        x98 = x10 * x93
                        x99 = x97 * x98
                        x100 = x14 * x90
                        x101 = 8.0980415781410316025e-20 * x100
                        x102 = x100 * x12 * x13 * x15 * x16 * x44 * x5 * x7 * xi[1]
                        x103 = x8 * x98
                        x104 = x102 * x103
                        x105 = x18 * x95
                        x106 = 5.8485855842129672685e-20 * x105
                        x107 = x105 * x12 * x13 * x15 * x44 * x5 * xi[1]
                        x108 = x103 * x3
                        x109 = x107 * x108
                        x110 = x100 * x16
                        x111 = 3.0078440147380974524e-19 * x110
                        x112 = x110 * x13 * x15 * x44 * xi[1]
                        x113 = x105 * x12
                        x114 = x112 * x113
                        x115 = x108 * x7
                        x116 = 4.135785520264883997e-19 * x115
                        x117 = (
                            -1139827.4301937679369 * xi[1] ** 18
                            + 4010503.9210521464445 * xi[1] ** 16
                            - 5723632.7564645448023 * xi[1] ** 14
                            + 4288221.5882976921237 * xi[1] ** 12
                            - 1825541.0625608358578 * xi[1] ** 10
                            + 447065.31380067163584 * xi[1] ** 8
                            - 60894.246929607780612 * xi[1] ** 6
                            + 4228.3941844706632653 * xi[1] ** 4
                            - 124.72118622448979592 * xi[1] ** 2
                            + 1.0
                        )
                        x118 = x117 * x43
                        x119 = 1.0247761692089423182e-12 * x40
                        x120 = x110 * x15
                        x121 = x113 * x44 * xi[1]
                        x122 = x121 * x5
                        x123 = x120 * x122
                        x124 = x122 * x13
                        x125 = x111 * x124
                        x126 = x112 * x5
                        x127 = x106 * x126
                        x128 = x101 * x7
                        x129 = x3 * x96
                        x130 = x8 * x91
                        x131 = x10 * x87
                        x132 = x6 * x86
                        x133 = x29 * x43
                        x134 = x133 * x20 * x44
                        x135 = x134 * x31
                        x136 = x135 * x26
                        x137 = x136 * x30
                        x138 = x137 * x28
                        x139 = x138 * x23
                        x140 = x139 * x72
                        x141 = x139 * x76
                        x142 = x37 * x43
                        x143 = x0 * x83
                        x144 = x143 * x88
                        x145 = x143 * x6
                        x146 = x145 * x92
                        x147 = x10 * x145
                        x148 = x147 * x97
                        x149 = x147 * x8
                        x150 = x102 * x149
                        x151 = x149 * x3
                        x152 = x107 * x151
                        x153 = x151 * x7
                        x154 = 4.135785520264883997e-19 * x153
                        x155 = x143 * x82
                        x156 = x1 * x47
                        x157 = x142 * x156
                        x158 = x11 * x82
                        x159 = 2.7560561673505191588e-20 * x158
                        x160 = x133 * x156
                        x161 = x160 * x88
                        x162 = x11 * x6
                        x163 = 2.7560561673505191588e-20 * x162
                        x164 = x157 * x88
                        x165 = x1 * x133
                        x166 = x158 * x165
                        x167 = 2.342647742247941285e-19 * x51
                        x168 = x166 * x167
                        x169 = x166 * x31
                        x170 = 4.1647070973296733956e-19 * x55
                        x171 = x169 * x170
                        x172 = x169 * x26
                        x173 = 4.68529548449588257e-18 * x59
                        x174 = x172 * x173
                        x175 = x172 * x30
                        x176 = 1.3118827356588471196e-17 * x63
                        x177 = x175 * x176
                        x178 = x175 * x28
                        x179 = 9.4747086464250069749e-18 * x67
                        x180 = x178 * x179
                        x181 = x178 * x23
                        x182 = 4.8727073038757178728e-17 * x71
                        x183 = x181 * x182
                        x184 = 6.6999725428291120751e-17 * x27
                        x185 = x181 * x184
                        x186 = x79 * x82
                        x187 = 1.6601373941184865555e-10 * x11
                        x188 = 2.342647742247941285e-19 * x9
                        x189 = x160 * x188
                        x190 = 4.1647070973296733956e-19 * x90
                        x191 = x160 * x6
                        x192 = 4.68529548449588257e-18 * x95
                        x193 = x191 * x192
                        x194 = x10 * x191
                        x195 = x194 * x97
                        x196 = 1.3118827356588471196e-17 * x100
                        x197 = x194 * x8
                        x198 = 9.4747086464250069749e-18 * x105
                        x199 = x197 * x198
                        x200 = x197 * x3
                        x201 = x107 * x200
                        x202 = 4.8727073038757178728e-17 * x110
                        x203 = x200 * x7
                        x204 = 6.6999725428291120751e-17 * x203
                        x205 = x118 * x29
                        x206 = 1.6601373941184865555e-10 * x47
                        x207 = x124 * x202
                        x208 = x3 * x7
                        x209 = x126 * x208
                        x210 = x10 * x8
                        x211 = x210 * x3
                        x212 = x102 * x211
                        x213 = x10 * x6
                        x214 = x213 * x92
                        x215 = x165 * x31
                        x216 = x215 * x88
                        x217 = x162 * x216
                        x218 = x217 * x26
                        x219 = x218 * x30
                        x220 = x219 * x28
                        x221 = x220 * x23
                        x222 = x182 * x221
                        x223 = x184 * x221
                        x224 = x6 * x79
                        x225 = x165 * x39
                        x226 = x225 * x88
                        x227 = x157 * x188
                        x228 = x157 * x6
                        x229 = x210 * x228
                        x230 = x192 * x228
                        x231 = x208 * x229
                        x232 = x198 * x229
                        x233 = 6.6999725428291120751e-17 * x231
                        x234 = x107 * x211
                        x235 = x213 * x97
                        x236 = x225 * x51
                        x237 = x82 * x9
                        x238 = 1.9912505809107500923e-18 * x237
                        x239 = x215 * x51
                        x240 = x214 * x9
                        x241 = 1.9912505809107500923e-18 * x240
                        x242 = x34 * x55
                        x243 = x215 * x237
                        x244 = 3.5400010327302223862e-18 * x243
                        x245 = x243 * x26
                        x246 = 3.9825011618215001845e-17 * x59
                        x247 = x245 * x246
                        x248 = x245 * x30
                        x249 = 1.1151003253100200517e-16 * x63
                        x250 = x248 * x249
                        x251 = x248 * x28
                        x252 = 8.0535023494612559287e-17 * x67
                        x253 = x251 * x252
                        x254 = x23 * x251
                        x255 = 4.1418012082943601919e-16 * x71
                        x256 = x254 * x255
                        x257 = 5.6949766614047452638e-16 * x27
                        x258 = x254 * x257
                        x259 = 1.4111167850007135721e-9 * x9
                        x260 = x26 * x55
                        x261 = 3.5400010327302223862e-18 * x90
                        x262 = x261 * x51
                        x263 = x239 * x6
                        x264 = 3.9825011618215001845e-17 * x95
                        x265 = x263 * x264
                        x266 = 1.1151003253100200517e-16 * x100
                        x267 = x235 * x266
                        x268 = x210 * x263
                        x269 = 8.0535023494612559287e-17 * x105
                        x270 = x268 * x269
                        x271 = 4.1418012082943601919e-16 * x110
                        x272 = x234 * x271
                        x273 = x208 * x268
                        x274 = 5.6949766614047452638e-16 * x273
                        x275 = x205 * x31
                        x276 = 1.4111167850007135721e-9 * x51
                        x277 = x124 * x271
                        x278 = x107 * x266
                        x279 = x261 * x97
                        x280 = x215 * x240
                        x281 = 3.5400010327302223862e-18 * x280
                        x282 = x26 * x30
                        x283 = x280 * x282
                        x284 = x28 * x283
                        x285 = x23 * x284
                        x286 = x255 * x285
                        x287 = x257 * x285
                        x288 = x10 * x224
                        x289 = x26 * x38
                        x290 = x236 * x6
                        x291 = x210 * x290
                        x292 = x264 * x290
                        x293 = x208 * x291
                        x294 = x269 * x291
                        x295 = 5.6949766614047452638e-16 * x293
                        x296 = 6.2933351692981731311e-18 * x90
                        x297 = x216 * x296
                        x298 = x215 * x260
                        x299 = x298 * x6
                        x300 = x210 * x299
                        x301 = x296 * x97
                        x302 = x215 * x242
                        x303 = x210 * x6
                        x304 = x302 * x303
                        x305 = x289 * x59
                        x306 = x216 * x90
                        x307 = 7.0800020654604447725e-17 * x306
                        x308 = x282 * x306
                        x309 = 1.9824005783289245363e-16 * x63
                        x310 = x308 * x309
                        x311 = x28 * x308
                        x312 = 1.4317337510153343873e-16 * x67
                        x313 = x311 * x312
                        x314 = x23 * x311
                        x315 = 7.3632021480788625634e-16 * x71
                        x316 = x314 * x315
                        x317 = 1.0124402953608436025e-15 * x27
                        x318 = x314 * x317
                        x319 = 2.5086520622234907949e-9 * x90
                        x320 = x282 * x59
                        x321 = 7.0800020654604447725e-17 * x95
                        x322 = x299 * x321
                        x323 = 1.9824005783289245363e-16 * x100
                        x324 = x235 * x323
                        x325 = 1.4317337510153343873e-16 * x105
                        x326 = x300 * x325
                        x327 = 7.3632021480788625634e-16 * x110
                        x328 = x234 * x327
                        x329 = x208 * x300
                        x330 = 1.0124402953608436025e-15 * x329
                        x331 = x26 * x275
                        x332 = 2.5086520622234907949e-9 * x55
                        x333 = x124 * x327
                        x334 = x107 * x323
                        x335 = x215 * x320
                        x336 = x303 * x335
                        x337 = x90 * x97
                        x338 = 7.0800020654604447725e-17 * x337
                        x339 = x215 * x303
                        x340 = x28 * x282
                        x341 = x339 * x340
                        x342 = x337 * x341
                        x343 = x23 * x342
                        x344 = x315 * x343
                        x345 = x317 * x343
                        x346 = x288 * x8
                        x347 = x282 * x36
                        x348 = x339 * x347
                        x349 = x305 * x339
                        x350 = x302 * x6
                        x351 = x321 * x350
                        x352 = x208 * x304
                        x353 = x304 * x325
                        x354 = 1.0124402953608436025e-15 * x352
                        x355 = x215 * x305
                        x356 = 7.965002323643000369e-16 * x6 * x95
                        x357 = x356 * x92
                        x358 = x212 * x356
                        x359 = x215 * x63
                        x360 = x347 * x359
                        x361 = x6 * x95
                        x362 = x361 * x92
                        x363 = 2.2302006506200401033e-15 * x362
                        x364 = x32 * x67
                        x365 = x215 * x340
                        x366 = x362 * x365
                        x367 = 1.6107004698922511857e-15 * x366
                        x368 = x23 * x366
                        x369 = 8.2836024165887203838e-15 * x71
                        x370 = x368 * x369
                        x371 = 1.1389953322809490528e-14 * x27
                        x372 = x368 * x371
                        x373 = 2.8222335700014271443e-8 * x95
                        x374 = x23 * x67
                        x375 = x340 * x359
                        x376 = 2.2302006506200401033e-15 * x100
                        x377 = x235 * x376
                        x378 = 1.6107004698922511857e-15 * x105
                        x379 = x336 * x378
                        x380 = 8.2836024165887203838e-15 * x110
                        x381 = x234 * x6
                        x382 = x380 * x381
                        x383 = x208 * x336
                        x384 = 1.1389953322809490528e-14 * x383
                        x385 = x30 * x331
                        x386 = 2.8222335700014271443e-8 * x59
                        x387 = x124 * x380
                        x388 = x107 * x376
                        x389 = x212 * x361
                        x390 = 2.2302006506200401033e-15 * x389
                        x391 = x365 * x389
                        x392 = 1.6107004698922511857e-15 * x391
                        x393 = x23 * x391
                        x394 = x369 * x393
                        x395 = x371 * x393
                        x396 = x3 * x346
                        x397 = x208 * x349
                        x398 = x349 * x378
                        x399 = 1.1389953322809490528e-14 * x397
                        x400 = 6.2445618217361122893e-15 * x100
                        x401 = x235 * x400
                        x402 = x341 * x63
                        x403 = x208 * x402
                        x404 = x107 * x400
                        x405 = x348 * x63
                        x406 = x208 * x405
                        x407 = x364 * x365
                        x408 = 4.5099613156983033201e-15 * x100
                        x409 = x235 * x408
                        x410 = x100 * x23
                        x411 = x235 * x365 * x410
                        x412 = x35 * x71
                        x413 = 2.3194086766448417075e-14 * x412
                        x414 = x27 * x411
                        x415 = 3.1891869303866573478e-14 * x414
                        x416 = 7.902253996003996004e-8 * x100
                        x417 = 2.3194086766448417075e-14 * x71
                        x418 = x365 * x374
                        x419 = 4.5099613156983033201e-15 * x105
                        x420 = x402 * x419
                        x421 = 2.3194086766448417075e-14 * x110
                        x422 = x381 * x421
                        x423 = 3.1891869303866573478e-14 * x403
                        x424 = x28 * x385
                        x425 = 7.902253996003996004e-8 * x63
                        x426 = x124 * x421
                        x427 = x341 * x374
                        x428 = x107 * x208
                        x429 = x408 * x428
                        x430 = x27 * x341
                        x431 = x410 * x428
                        x432 = x430 * x431
                        x433 = 3.1891869303866573478e-14 * x432
                        x434 = x396 * x7
                        x435 = x341 * x364
                        x436 = x405 * x419
                        x437 = 3.1891869303866573478e-14 * x406
                        x438 = 3.2571942835598857312e-15 * x105
                        x439 = x102 * x438
                        x440 = x209 * x438
                        x441 = x341 * x412
                        x442 = x105 * x23
                        x443 = x102 * x442
                        x444 = 1.6751284886879412332e-14 * x443
                        x445 = 2.3033016719459191956e-14 * x430
                        x446 = x443 * x445
                        x447 = 5.7071834415584415584e-8 * x105
                        x448 = x430 * x71
                        x449 = 1.6751284886879412332e-14 * x110
                        x450 = x381 * x449
                        x451 = x208 * x427
                        x452 = 2.3033016719459191956e-14 * x451
                        x453 = x23 * x424
                        x454 = 5.7071834415584415584e-8 * x67
                        x455 = x124 * x449
                        x456 = x209 * x442
                        x457 = 1.6751284886879412332e-14 * x456
                        x458 = x445 * x456
                        x459 = x434 * x5
                        x460 = x208 * x435
                        x461 = 2.3033016719459191956e-14 * x460
                        x462 = x110 * x23
                        x463 = 8.6149465132522691991e-14 * x462
                        x464 = x365 * x381
                        x465 = x463 * x464
                        x466 = x208 * x448
                        x467 = x124 * x463
                        x468 = x208 * x441
                        x469 = 1.1845551455721870149e-13 * x462
                        x470 = x469 * x75
                        x471 = x27 * x464
                        x472 = 2.9351229128014842301e-7 * x110
                        x473 = x469 * x81
                        x474 = 1.1845551455721870149e-13 * x23
                        x475 = x466 * x474
                        x476 = x27 * x453
                        x477 = 2.9351229128014842301e-7 * x71
                        x478 = x208 * x430
                        x479 = x124 * x478
                        x480 = x121 * x459
                        x481 = x468 * x474
                        x482 = 1.6287633251617571455e-13 * x23 * x478
                        x483 = x114 * x482
                        x484 = x123 * x482
                        x485 = 4.0357940051020408163e-7 * x476
                        return jnp.asarray(
                            [
                                x0 * x42,
                                x42 * x43,
                                x43 * x45,
                                x0 * x45,
                                -x37 * x49,
                                x39 * x53,
                                -x34 * x57,
                                x38 * x61,
                                -x36 * x65,
                                x32 * x69,
                                -x35 * x73,
                                x75 * x77,
                                x79 * x80,
                                x77 * x81,
                                -x27 * x73,
                                x23 * x69,
                                -x28 * x65,
                                x30 * x61,
                                -x26 * x57,
                                x31 * x53,
                                -x29 * x49,
                                -x85 * x86,
                                x85 * x87,
                                -x89 * x91,
                                x94 * x96,
                                -x101 * x99,
                                x104 * x106,
                                -x109 * x111,
                                x114 * x116,
                                x118 * x119,
                                x116 * x123,
                                -x115 * x125,
                                x115 * x127,
                                -x109 * x128,
                                x104 * x129,
                                -x130 * x99,
                                x131 * x94,
                                -x132 * x89,
                                -x134 * x48,
                                x135 * x52,
                                -x136 * x56,
                                x137 * x60,
                                -x138 * x64,
                                x139 * x68,
                                -x140 * x27,
                                x141 * x81,
                                x44 * x78 * x80,
                                x141 * x75,
                                -x140 * x35,
                                x138 * x32 * x68,
                                -x137 * x36 * x64,
                                x136 * x38 * x60,
                                -x135 * x34 * x56,
                                x134 * x39 * x52,
                                -x142 * x20 * x44 * x48,
                                -x132 * x144,
                                x131 * x146,
                                -x130 * x148,
                                x129 * x150,
                                -x128 * x152,
                                x127 * x153,
                                -x125 * x153,
                                x123 * x154,
                                x0 * x117 * x119,
                                x114 * x154,
                                -x111 * x152,
                                x106 * x150,
                                -x101 * x148,
                                x146 * x96,
                                -x144 * x91,
                                x155 * x87,
                                -x155 * x86,
                                x157 * x159,
                                x159 * x160,
                                x161 * x163,
                                x163 * x164,
                                -x168 * x39,
                                x171 * x34,
                                -x174 * x38,
                                x177 * x36,
                                -x180 * x32,
                                x183 * x35,
                                -x185 * x75,
                                -x186 * x187,
                                -x185 * x81,
                                x183 * x27,
                                -x180 * x23,
                                x177 * x28,
                                -x174 * x30,
                                x171 * x26,
                                -x168 * x31,
                                -x189 * x82,
                                x161 * x190,
                                -x193 * x92,
                                x195 * x196,
                                -x102 * x199,
                                x201 * x202,
                                -x114 * x204,
                                -x205 * x206,
                                -x123 * x204,
                                x203 * x207,
                                -x199 * x209,
                                x196 * x201 * x7,
                                -x193 * x212,
                                x190 * x195 * x8,
                                -x189 * x214,
                                -x167 * x217,
                                x170 * x218,
                                -x173 * x219,
                                x176 * x220,
                                -x179 * x221,
                                x222 * x27,
                                -x223 * x81,
                                -x187 * x224 * x88,
                                -x223 * x75,
                                x222 * x35,
                                -x179 * x220 * x32,
                                x176 * x219 * x36,
                                -x173 * x218 * x38,
                                x170 * x217 * x34,
                                -x162 * x167 * x226,
                                -x214 * x227,
                                x190 * x229 * x97,
                                -x212 * x230,
                                x107 * x196 * x231,
                                -x209 * x232,
                                x207 * x231,
                                -x123 * x233,
                                -x118 * x206 * x37,
                                -x114 * x233,
                                x202 * x228 * x234,
                                -x102 * x232,
                                x157 * x196 * x235,
                                -x230 * x92,
                                x164 * x190,
                                -x227 * x82,
                                x236 * x238,
                                x238 * x239,
                                x239 * x241,
                                x236 * x241,
                                -x242 * x244,
                                x247 * x38,
                                -x250 * x36,
                                x253 * x32,
                                -x256 * x35,
                                x258 * x75,
                                x186 * x259,
                                x258 * x81,
                                -x256 * x27,
                                x23 * x253,
                                -x250 * x28,
                                x247 * x30,
                                -x244 * x260,
                                -x216 * x262,
                                x265 * x92,
                                -x239 * x267,
                                x102 * x270,
                                -x263 * x272,
                                x114 * x274,
                                x275 * x276,
                                x123 * x274,
                                -x273 * x277,
                                x209 * x270,
                                -x273 * x278,
                                x212 * x265,
                                -x268 * x279,
                                -x260 * x281,
                                x246 * x283,
                                -x249 * x284,
                                x252 * x285,
                                -x27 * x286,
                                x287 * x81,
                                x259 * x288 * x92,
                                x287 * x75,
                                -x286 * x35,
                                x252 * x284 * x32,
                                -x249 * x283 * x36,
                                x246 * x280 * x289,
                                -x242 * x281,
                                -x279 * x291,
                                x212 * x292,
                                -x278 * x293,
                                x209 * x294,
                                -x277 * x293,
                                x123 * x295,
                                x205 * x276 * x39,
                                x114 * x295,
                                -x272 * x290,
                                x102 * x294,
                                -x236 * x267,
                                x292 * x92,
                                -x226 * x262,
                                x242 * x297,
                                x260 * x297,
                                x300 * x301,
                                x301 * x304,
                                -x305 * x307,
                                x310 * x36,
                                -x313 * x32,
                                x316 * x35,
                                -x318 * x75,
                                -x319 * x79 * x88,
                                -x318 * x81,
                                x27 * x316,
                                -x23 * x313,
                                x28 * x310,
                                -x307 * x320,
                                -x322 * x92,
                                x298 * x324,
                                -x102 * x326,
                                x299 * x328,
                                -x114 * x330,
                                -x331 * x332,
                                -x123 * x330,
                                x329 * x333,
                                -x209 * x326,
                                x329 * x334,
                                -x212 * x322,
                                -x336 * x338,
                                x309 * x342,
                                -x312 * x343,
                                x27 * x344,
                                -x345 * x81,
                                -x319 * x346 * x97,
                                -x345 * x75,
                                x344 * x35,
                                -x312 * x32 * x342,
                                x309 * x337 * x348,
                                -x338 * x349,
                                -x212 * x351,
                                x334 * x352,
                                -x209 * x353,
                                x333 * x352,
                                -x123 * x354,
                                -x275 * x332 * x34,
                                -x114 * x354,
                                x328 * x350,
                                -x102 * x353,
                                x302 * x324,
                                -x351 * x92,
                                x355 * x357,
                                x335 * x357,
                                x335 * x358,
                                x355 * x358,
                                -x360 * x363,
                                x364 * x367,
                                -x35 * x370,
                                x372 * x75,
                                x224 * x373 * x92,
                                x372 * x81,
                                -x27 * x370,
                                x367 * x374,
                                -x363 * x375,
                                -x335 * x377,
                                x102 * x379,
                                -x335 * x382,
                                x114 * x384,
                                x385 * x386,
                                x123 * x384,
                                -x383 * x387,
                                x209 * x379,
                                -x383 * x388,
                                -x375 * x390,
                                x374 * x392,
                                -x27 * x394,
                                x395 * x81,
                                x102 * x373 * x396,
                                x395 * x75,
                                -x35 * x394,
                                x364 * x392,
                                -x360 * x390,
                                -x388 * x397,
                                x209 * x398,
                                -x387 * x397,
                                x123 * x399,
                                x331 * x38 * x386,
                                x114 * x399,
                                -x355 * x382,
                                x102 * x398,
                                -x355 * x377,
                                x360 * x401,
                                x375 * x401,
                                x403 * x404,
                                x404 * x406,
                                -x407 * x409,
                                x411 * x413,
                                -x415 * x75,
                                -x288 * x416 * x97,
                                -x415 * x81,
                                x414 * x417,
                                -x409 * x418,
                                -x102 * x420,
                                x375 * x422,
                                -x114 * x423,
                                -x424 * x425,
                                -x123 * x423,
                                x403 * x426,
                                -x209 * x420,
                                -x427 * x429,
                                x417 * x432,
                                -x433 * x81,
                                -x107 * x416 * x434,
                                -x433 * x75,
                                x341 * x413 * x431,
                                -x429 * x435,
                                -x209 * x436,
                                x406 * x426,
                                -x123 * x437,
                                -x36 * x385 * x425,
                                -x114 * x437,
                                x360 * x422,
                                -x102 * x436,
                                x435 * x439,
                                x427 * x439,
                                x427 * x440,
                                x435 * x440,
                                -x441 * x444,
                                x446 * x75,
                                x102 * x346 * x447,
                                x446 * x81,
                                -x444 * x448,
                                -x418 * x450,
                                x114 * x452,
                                x453 * x454,
                                x123 * x452,
                                -x451 * x455,
                                -x448 * x457,
                                x458 * x81,
                                x112 * x447 * x459,
                                x458 * x75,
                                -x441 * x457,
                                -x455 * x460,
                                x123 * x461,
                                x32 * x424 * x454,
                                x114 * x461,
                                -x407 * x450,
                                x412 * x465,
                                x27 * x465 * x71,
                                x466 * x467,
                                x467 * x468,
                                -x470 * x471,
                                -x107 * x396 * x472,
                                -x471 * x473,
                                -x114 * x475,
                                -x476 * x477,
                                -x123 * x475,
                                -x473 * x479,
                                -x13 * x472 * x480,
                                -x470 * x479,
                                -x123 * x481,
                                -x35 * x453 * x477,
                                -x114 * x481,
                                x483 * x75,
                                x483 * x81,
                                x484 * x81,
                                x484 * x75,
                                4.0357940051020408163e-7 * x114 * x434,
                                x485 * x81,
                                4.0357940051020408163e-7 * x120 * x480,
                                x485 * x75,
                                x117 * x78,
                            ]
                        )

                case 400:

                    def shape_functions(xi):
                        x0 = 19.0 * xi[1]
                        x1 = x0 + 17.0
                        x2 = xi[0] - 1.0
                        x3 = x1 * x2
                        x4 = 19.0 * xi[0]
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
                        x41 = (
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
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x31
                            * x32
                            * x33
                            * x34
                            * x35
                            * x36
                            * x37
                            * x38
                            * x39
                            * x40
                            * x5
                            * x6
                            * x7
                            * x8
                            * x9
                        )
                        x42 = 8.8751995036557687094e-44 * x41
                        x43 = xi[0] + 1.0
                        x44 = x1 * x43
                        x45 = xi[1] + 1.0
                        x46 = x21 * x45
                        x47 = x44 * x46
                        x48 = (
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
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x31
                            * x32
                            * x33
                            * x34
                            * x35
                            * x36
                            * x37
                            * x38
                            * x39
                            * x40
                            * x5
                            * x6
                            * x7
                            * x8
                            * x9
                        )
                        x49 = 8.8751995036557687094e-44 * x48
                        x50 = x3 * x46
                        x51 = 3.2039470208197325041e-41 * x43
                        x52 = x22 * x3
                        x53 = x51 * x52
                        x54 = (
                            x10
                            * x11
                            * x12
                            * x13
                            * x14
                            * x15
                            * x16
                            * x17
                            * x20
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x31
                            * x32
                            * x33
                            * x34
                            * x35
                            * x36
                            * x37
                            * x38
                            * x39
                            * x40
                            * x5
                            * x6
                            * x7
                            * x8
                            * x9
                        )
                        x55 = x52 * x54
                        x56 = x18 * x43
                        x57 = 2.8835523187377592537e-40 * x56
                        x58 = x21 * x57
                        x59 = (
                            x10
                            * x11
                            * x12
                            * x13
                            * x14
                            * x15
                            * x16
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x31
                            * x32
                            * x33
                            * x34
                            * x35
                            * x36
                            * x37
                            * x38
                            * x39
                            * x40
                            * x5
                            * x6
                            * x7
                            * x8
                            * x9
                        )
                        x60 = x52 * x59
                        x61 = 1.6340129806180635771e-39 * x19
                        x62 = x20 * x56
                        x63 = x61 * x62
                        x64 = x21 * x63
                        x65 = x36 * x52
                        x66 = x21 * x65
                        x67 = (
                            x10
                            * x11
                            * x12
                            * x13
                            * x14
                            * x16
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x31
                            * x32
                            * x33
                            * x34
                            * x35
                            * x37
                            * x38
                            * x39
                            * x40
                            * x5
                            * x6
                            * x7
                            * x8
                            * x9
                        )
                        x68 = x17 * x19
                        x69 = x62 * x68
                        x70 = x67 * x69
                        x71 = 6.5360519224722543084e-39 * x70
                        x72 = (
                            x10
                            * x11
                            * x12
                            * x14
                            * x15
                            * x16
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x31
                            * x32
                            * x33
                            * x34
                            * x37
                            * x38
                            * x39
                            * x40
                            * x5
                            * x6
                            * x7
                            * x8
                            * x9
                        )
                        x73 = x35 * x69
                        x74 = x66 * x73
                        x75 = 1.9608155767416762925e-38 * x74
                        x76 = (
                            x10
                            * x12
                            * x13
                            * x14
                            * x15
                            * x16
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x31
                            * x32
                            * x33
                            * x34
                            * x37
                            * x38
                            * x5
                            * x6
                            * x7
                            * x8
                            * x9
                        )
                        x77 = x39 * x40
                        x78 = x74 * x77
                        x79 = 4.5752363457305780158e-38 * x78
                        x80 = (
                            x10
                            * x11
                            * x12
                            * x13
                            * x14
                            * x15
                            * x16
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x31
                            * x32
                            * x33
                            * x34
                            * x5
                            * x6
                            * x7
                            * x8
                        )
                        x81 = x37 * x38
                        x82 = x78 * x81
                        x83 = 8.4968674992139306009e-38 * x82
                        x84 = (
                            x10
                            * x11
                            * x12
                            * x13
                            * x14
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x31
                            * x32
                            * x33
                            * x34
                            * x5
                            * x6
                            * x8
                            * x9
                        )
                        x85 = x15 * x16
                        x86 = x82 * x85
                        x87 = 1.2745301248820895901e-37 * x86
                        x88 = (
                            x10
                            * x11
                            * x12
                            * x13
                            * x14
                            * x23
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x31
                            * x32
                            * x6
                            * x7
                            * x8
                            * x9
                        )
                        x89 = x86 * x88
                        x90 = x33 * x34
                        x91 = 1.5577590415225539435e-37 * x90
                        x92 = (
                            x10
                            * x11
                            * x12
                            * x24
                            * x25
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x31
                            * x32
                            * x5
                            * x6
                            * x7
                            * x8
                            * x9
                        )
                        x93 = x13 * x14
                        x94 = x91 * x93
                        x95 = x92 * x94
                        x96 = (
                            x10
                            * x11
                            * x12
                            * x24
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x31
                            * x32
                            * x5
                            * x7
                            * x8
                            * x9
                        )
                        x97 = x23 * x90
                        x98 = x6 * x93
                        x99 = x97 * x98
                        x100 = x96 * x99
                        x101 = x25 * x85
                        x102 = (
                            x10
                            * x101
                            * x11
                            * x12
                            * x26
                            * x28
                            * x29
                            * x30
                            * x31
                            * x32
                            * x5
                            * x7
                            * x8
                            * x9
                        )
                        x103 = x24 * x99
                        x104 = x102 * x103
                        x105 = (
                            x10
                            * x101
                            * x103
                            * x11
                            * x26
                            * x27
                            * x28
                            * x30
                            * x31
                            * x32
                            * x5
                            * x8
                            * x9
                        )
                        x106 = x7 * x81
                        x107 = x106 * x12
                        x108 = x105 * x107
                        x109 = (
                            x101
                            * x103
                            * x107
                            * x11
                            * x26
                            * x27
                            * x28
                            * x29
                            * x30
                            * x32
                            * x5
                            * x8
                        )
                        x110 = x77 * x9
                        x111 = x10 * x110
                        x112 = x109 * x111
                        x113 = 6.5360519224722543084e-39 * x5
                        x114 = x113 * x34
                        x115 = x13 * x72
                        x116 = x115 * x17
                        x117 = x110 * x65
                        x118 = x38 * x80
                        x119 = x35 * x68
                        x120 = x119 * x20
                        x121 = x118 * x120
                        x122 = x120 * x18
                        x123 = x36 * x76
                        x124 = x11 * x123
                        x125 = x124 * x40
                        x126 = x122 * x125
                        x127 = x45 * x51
                        x128 = x22 * x47
                        x129 = x128 * x18
                        x130 = 2.8835523187377592537e-40 * x59 * x68
                        x131 = x54 * x61
                        x132 = x122 * x128
                        x133 = x36 * x77
                        x134 = x133 * x84
                        x135 = x106 * x15
                        x136 = x134 * x135
                        x137 = 6.5360519224722543084e-39 * x136
                        x138 = x132 * x133
                        x139 = x13 * x97
                        x140 = x81 * x85
                        x141 = x140 * x92
                        x142 = x139 * x141
                        x143 = 1.9608155767416762925e-38 * x142
                        x144 = 4.5752363457305780158e-38 * x29
                        x145 = x105 * x106
                        x146 = x144 * x145
                        x147 = x110 * x36
                        x148 = x132 * x147
                        x149 = x109 * x31
                        x150 = 8.4968674992139306009e-38 * x149
                        x151 = (
                            x101 * x103 * x11 * x26 * x27 * x28 * x30 * x31 * x32 * x5
                        )
                        x152 = x107 * x151
                        x153 = x29 * x36
                        x154 = x111 * x153
                        x155 = x132 * x154
                        x156 = 1.2745301248820895901e-37 * x155
                        x157 = x101 * x96
                        x158 = x23 * x81
                        x159 = x158 * x94
                        x160 = x138 * x159
                        x161 = x102 * x27
                        x162 = x161 * x6
                        x163 = x101 * x103 * x11 * x27 * x30 * x31 * x32 * x5 * x8
                        x164 = x107 * x28
                        x165 = x163 * x164
                        x166 = x155 * x26
                        x167 = x107 * x163
                        x168 = 8.4968674992139306009e-38 * x167
                        x169 = x36 * x5
                        x170 = x111 * x26
                        x171 = x101 * x103 * x11 * x164 * x27 * x31 * x8
                        x172 = x171 * x32
                        x173 = x170 * x172
                        x174 = x169 * x173
                        x175 = x144 * x174
                        x176 = x30 * x5
                        x177 = x171 * x176
                        x178 = 1.9608155767416762925e-38 * x177
                        x179 = x140 * x88
                        x180 = x179 * x33
                        x181 = x113 * x180
                        x182 = x15 * x67
                        x183 = x17 * x182
                        x184 = x183 * x20 * x61
                        x185 = x37 * x80
                        x186 = 2.8835523187377592537e-40 * x185
                        x187 = x124 * x39
                        x188 = 3.2039470208197325041e-41 * x187
                        x189 = x50 * x57
                        x190 = x121 * x147
                        x191 = x36 * x50
                        x192 = x50 * x73
                        x193 = x133 * x192
                        x194 = x179 * x193
                        x195 = x112 * x73
                        x196 = 1.9608155767416762925e-38 * x191
                        x197 = x104 * x81
                        x198 = 8.4968674992139306009e-38 * x193
                        x199 = x140 * x193
                        x200 = x134 * x140
                        x201 = x80 * x81
                        x202 = x123 * x77
                        x203 = x72 * x73
                        x204 = x50 * x59
                        x205 = x22 * x50
                        x206 = x122 * x205
                        x207 = x147 * x206
                        x208 = x18 * x205
                        x209 = x133 * x206
                        x210 = x154 * x206
                        x211 = x210 * x26
                        x212 = 1.2745301248820895901e-37 * x210
                        x213 = x159 * x209
                        x214 = x2 * x45
                        x215 = 1.156624874515923434e-38 * x22
                        x216 = x2 * x46
                        x217 = x216 * x73
                        x218 = x11 * x76
                        x219 = x39 * x73
                        x220 = x45 * x65
                        x221 = x216 * x22
                        x222 = 1.0409623870643310906e-37 * x56
                        x223 = 5.8987868600312095133e-37 * x221
                        x224 = x19 * x62
                        x225 = x221 * x36
                        x226 = 7.078544232037451416e-36 * x225
                        x227 = x217 * x22
                        x228 = 1.6516603208087386637e-35 * x227
                        x229 = x133 * x227
                        x230 = 3.0673691672162289469e-35 * x201
                        x231 = x179 * x229
                        x232 = 5.623510139896419736e-35 * x90
                        x233 = x141 * x93
                        x234 = x232 * x233
                        x235 = 4.6010537508243434204e-35 * x140
                        x236 = x100 * x235
                        x237 = 3.0673691672162289469e-35 * x197
                        x238 = x108 * x133
                        x239 = 2.3595147440124838053e-36 * x34
                        x240 = x115 * x69
                        x241 = 1.0409623870643310906e-37 * x118
                        x242 = x125 * x205
                        x243 = 5.8987868600312095133e-37 * x43
                        x244 = x22 * x40
                        x245 = x191 * x84
                        x246 = 2.3595147440124838053e-36 * x73
                        x247 = x135 * x246
                        x248 = 7.078544232037451416e-36 * x244
                        x249 = x191 * x73
                        x250 = 1.6516603208087386637e-35 * x145
                        x251 = x153 * x192
                        x252 = x244 * x251
                        x253 = x244 * x249
                        x254 = 3.0673691672162289469e-35 * x9
                        x255 = 4.6010537508243434204e-35 * x152
                        x256 = x10 * x252
                        x257 = x256 * x9
                        x258 = 5.623510139896419736e-35 * x81
                        x259 = x253 * x258
                        x260 = x157 * x93
                        x261 = x260 * x97
                        x262 = x161 * x99
                        x263 = 4.6010537508243434204e-35 * x165
                        x264 = x167 * x26
                        x265 = 1.6516603208087386637e-35 * x5
                        x266 = x172 * x26
                        x267 = x10 * x9
                        x268 = x177 * x26
                        x269 = x192 * x244
                        x270 = 1.0409623870643310906e-37 * x185
                        x271 = x219 * x22
                        x272 = x191 * x271
                        x273 = x187 * x205
                        x274 = 5.8987868600312095133e-37 * x273
                        x275 = x192 * x22
                        x276 = x275 * x39
                        x277 = 7.078544232037451416e-36 * x272
                        x278 = x245 * x271
                        x279 = x14 * x97
                        x280 = x106 * x16
                        x281 = x224 * x35
                        x282 = x17 * x35 * x62
                        x283 = x117 * x45 * x73
                        x284 = x220 * x77
                        x285 = x180 * x5
                        x286 = x29 * x73
                        x287 = x220 * x286
                        x288 = x170 * x287
                        x289 = x111 * x287
                        x290 = x284 * x73
                        x291 = x258 * x290
                        x292 = 9.3686614835789798152e-37 * x56
                        x293 = x204 * x22
                        x294 = x147 * x205
                        x295 = x147 * x275
                        x296 = x185 * x294
                        x297 = 5.308908174028088562e-36 * x56
                        x298 = x19 * x293
                        x299 = 2.1235632696112354248e-35 * x56
                        x300 = x191 * x22
                        x301 = x300 * x68
                        x302 = x301 * x67
                        x303 = x300 * x72
                        x304 = x119 * x56
                        x305 = 6.3706898088337062743e-35 * x304
                        x306 = x205 * x304
                        x307 = 1.4864942887278647973e-34 * x306
                        x308 = x133 * x306
                        x309 = 2.7606322504946060522e-34 * x308
                        x310 = 5.0611591259067777624e-34 * x90
                        x311 = x308 * x310
                        x312 = 4.1409483757419090783e-34 * x100
                        x313 = x112 * x300
                        x314 = x34 * x5
                        x315 = x133 * x205
                        x316 = x179 * x315
                        x317 = x314 * x316
                        x318 = x115 * x301
                        x319 = 5.308908174028088562e-36 * x205
                        x320 = x319 * x43
                        x321 = x38 * x7
                        x322 = x134 * x275
                        x323 = x15 * x322
                        x324 = x193 * x22
                        x325 = x324 * x38
                        x326 = x325 * x85
                        x327 = x139 * x92
                        x328 = x29 * x321
                        x329 = 1.4864942887278647973e-34 * x105 * x324
                        x330 = x12 * x151
                        x331 = x295 * x8
                        x332 = x330 * x331
                        x333 = x154 * x275
                        x334 = x321 * x333
                        x335 = 4.1409483757419090783e-34 * x334
                        x336 = 5.0611591259067777624e-34 * x325
                        x337 = x12 * x163
                        x338 = x28 * x337
                        x339 = x26 * x334
                        x340 = 1.4864942887278647973e-34 * x5
                        x341 = x103 * x27 * x31 * x32 * x8
                        x342 = x101 * x11
                        x343 = x28 * x342
                        x344 = x12 * x343
                        x345 = x339 * x344
                        x346 = x176 * x27 * x31
                        x347 = x103 * x8
                        x348 = 6.3706898088337062743e-35 * x347
                        x349 = 2.1235632696112354248e-35 * x88
                        x350 = x33 * x5
                        x351 = 5.308908174028088562e-36 * x296
                        x352 = x324 * x37
                        x353 = x352 * x85
                        x354 = x37 * x7
                        x355 = x176 * x26
                        x356 = x32 * x333
                        x357 = x355 * x356
                        x358 = x27 * x357
                        x359 = x12 * x354
                        x360 = x310 * x353
                        x361 = x92 * x93
                        x362 = x101 * x333
                        x363 = x341 * x355
                        x364 = x28 * x363
                        x365 = x362 * x364
                        x366 = x279 * x92
                        x367 = x282 * x315
                        x368 = x205 * x282
                        x369 = x154 * x368
                        x370 = 4.1409483757419090783e-34 * x369
                        x371 = x262 * x81
                        x372 = 5.0611591259067777624e-34 * x367
                        x373 = x261 * x81
                        x374 = x149 * x294
                        x375 = x145 * x29
                        x376 = x20 * x43
                        x377 = 3.0083812986159168518e-35 * x376
                        x378 = 3.0083812986159168518e-35 * x205
                        x379 = x120 * x43
                        x380 = 3.6100575583391002221e-34 * x379
                        x381 = x205 * x379
                        x382 = 8.423467636124567185e-34 * x381
                        x383 = x315 * x379
                        x384 = 1.5643582752802767629e-33 * x201
                        x385 = 2.8679901713471740654e-33 * x90
                        x386 = x233 * x385
                        x387 = 2.3465374129204151444e-33 * x140
                        x388 = x100 * x387
                        x389 = 1.5643582752802767629e-33 * x197
                        x390 = x205 * x69
                        x391 = 1.2033525194463667407e-34 * x136
                        x392 = x315 * x69
                        x393 = 3.6100575583391002221e-34 * x142
                        x394 = 8.423467636124567185e-34 * x375
                        x395 = 1.5643582752802767629e-33 * x374
                        x396 = x154 * x390
                        x397 = 2.3465374129204151444e-33 * x396
                        x398 = 2.8679901713471740654e-33 * x392
                        x399 = 1.5643582752802767629e-33 * x264
                        x400 = x266 * x5
                        x401 = 8.423467636124567185e-34 * x400
                        x402 = 3.6100575583391002221e-34 * x268
                        x403 = 1.2033525194463667407e-34 * x285
                        x404 = x275 * x77
                        x405 = x179 * x404
                        x406 = 3.6100575583391002221e-34 * x275
                        x407 = 8.423467636124567185e-34 * x404
                        x408 = x281 * x315
                        x409 = x205 * x281
                        x410 = x154 * x409
                        x411 = 2.3465374129204151444e-33 * x410
                        x412 = 2.8679901713471740654e-33 * x408
                        x413 = x15 * x324
                        x414 = x413 * x81
                        x415 = 4.8134100777854669628e-34 * x88
                        x416 = x324 * x81
                        x417 = x16 * x416
                        x418 = x333 * x363
                        x419 = 3.369387054449826874e-33 * x164
                        x420 = x15 * x25
                        x421 = x419 * x420
                        x422 = x10 * x29
                        x423 = x11 * x164
                        x424 = 6.2574331011211070517e-33 * x423
                        x425 = x25 * x424
                        x426 = 1.1471960685388696261e-32 * x414 * x90
                        x427 = x347 * x420
                        x428 = x31 * x357
                        x429 = x11 * x341
                        x430 = x169 * x170 * x275 * x30
                        x431 = 1.4440230233356400889e-33 * x358
                        x432 = x23 * x34
                        x433 = 1.4440230233356400889e-33 * x432
                        x434 = x141 * x324
                        x435 = x13 * x434
                        x436 = x106 * x343
                        x437 = x24 * x98
                        x438 = x437 * x8
                        x439 = x26 * x346
                        x440 = x356 * x439
                        x441 = x432 * x440
                        x442 = x438 * x441
                        x443 = 6.2574331011211070517e-33 * x342
                        x444 = x164 * x439
                        x445 = x443 * x444
                        x446 = x29 * x32 * x331
                        x447 = x437 * x446
                        x448 = x164 * x342
                        x449 = 9.3861496516816605776e-33 * x448
                        x450 = x158 * x324
                        x451 = 1.1471960685388696261e-32 * x34 * x450
                        x452 = x161 * x98
                        x453 = x356 * x438
                        x454 = x346 * x453
                        x455 = x23 * x419
                        x456 = x26 * x342
                        x457 = x27 * x31
                        x458 = x453 * x457
                        x459 = x342 * x438
                        x460 = x333 * x444
                        x461 = x459 * x460
                        x462 = x23 * x33
                        x463 = x438 * x462
                        x464 = x33 * x455
                        x465 = x32 * x430 * x457
                        x466 = x199 * x22 * x96
                        x467 = 1.1471960685388696261e-32 * x324
                        x468 = x438 * x440
                        x469 = x324 * x422
                        x470 = x32 * x469
                        x471 = x16 * x25
                        x472 = x333 * x347
                        x473 = x423 * x439
                        x474 = x472 * x473
                        x475 = x333 * x471
                        x476 = x26 * x5
                        x477 = x107 * x11 * x418
                        x478 = 9.3861496516816605776e-33 * x423
                        x479 = x176 * x341
                        x480 = x440 * x471
                        x481 = x8 * x99
                        x482 = x423 * x481
                        x483 = x93 * x97
                        x484 = x103 * x446
                        x485 = x11 * x333 * x364
                        x486 = 4.3320690700069202666e-33 * x448
                        x487 = x139 * x6
                        x488 = x24 * x8
                        x489 = x487 * x488
                        x490 = x355 * x472
                        x491 = x279 * x6
                        x492 = x342 * x488
                        x493 = x491 * x492
                        x494 = x440 * x489
                        x495 = x101 * x164
                        x496 = 1.8772299303363321155e-32 * x342
                        x497 = x444 * x470
                        x498 = x344 * x81
                        x499 = 2.8158448955044981733e-32 * x498
                        x500 = x448 * x489
                        x501 = x26 * x30
                        x502 = 3.4415882056166088784e-32 * x501
                        x503 = x356 * x457
                        x504 = 3.4415882056166088784e-32 * x90
                        x505 = x102 * x416
                        x506 = x24 * x505
                        x507 = x347 * x358
                        x508 = 1.0108161163349480622e-32 * x436
                        x509 = x358 * x448
                        x510 = 3.4415882056166088784e-32 * x509
                        x511 = x483 * x488
                        x512 = 2.8158448955044981733e-32 * x448
                        x513 = x356 * x512
                        x514 = x27 * x347
                        x515 = x448 * x476
                        x516 = 1.0108161163349480622e-32 * x515
                        x517 = x31 * x448
                        x518 = x488 * x491
                        x519 = x107 * x440
                        x520 = 3.4415882056166088784e-32 * x279 * x416
                        x521 = x24 * x491
                        x522 = x341 * x476
                        x523 = x164 * x362
                        x524 = 4.3802031707847749362e-32 * x469
                        x525 = 8.030372479772087383e-32 * x436
                        x526 = x333 * x341
                        x527 = 6.5703047561771624043e-32 * x85
                        x528 = 4.3802031707847749362e-32 * x347
                        x529 = x22 * x249
                        x530 = 6.5703047561771624043e-32 * x111 * x529
                        x531 = 8.030372479772087383e-32 * x511
                        x532 = x448 * x465
                        x533 = 8.030372479772087383e-32 * x481
                        x534 = x333 * x522
                        x535 = x440 * x495
                        x536 = 1.2201994547186158751e-31 * x81
                        x537 = 1.491354889100530514e-31 * x30
                        x538 = 1.491354889100530514e-31 * x90
                        x539 = 1.2201994547186158751e-31 * x85
                        x540 = x103 * x448
                        x541 = 1.491354889100530514e-31 * x483
                        x542 = 1.2201994547186158751e-31 * x469
                        x543 = 1.8302991820779238126e-31 * x333
                        x544 = x543 * x81
                        x545 = x423 * x85
                        x546 = x501 * x503
                        x547 = 2.237032333650795771e-31 * x440
                        x548 = x448 * x90
                        x549 = x498 * x547
                        x550 = 2.7341506300176392756e-31 * x448 * x546
                        x551 = 2.7341506300176392756e-31 * x416 * x90
                        return jnp.asarray(
                            [
                                x3 * x42,
                                -x42 * x44,
                                x47 * x49,
                                -x49 * x50,
                                -x48 * x53,
                                x55 * x58,
                                -x60 * x64,
                                x66 * x71,
                                -x72 * x75,
                                x76 * x79,
                                -x80 * x83,
                                x84 * x87,
                                -x89 * x91,
                                x86 * x95,
                                -x100 * x87,
                                x104 * x83,
                                -x108 * x79,
                                x112 * x75,
                                -x114 * x89,
                                x116 * x64 * x65,
                                -x117 * x121 * x58,
                                x126 * x21 * x53,
                                x127 * x41,
                                -x129 * x130,
                                x128 * x131,
                                -x132 * x137,
                                x138 * x143,
                                -x138 * x146,
                                x148 * x150,
                                -x152 * x156,
                                x157 * x160,
                                -x160 * x162,
                                x156 * x165,
                                -x166 * x168,
                                x132 * x175,
                                -x166 * x178,
                                x138 * x181,
                                -x129 * x184,
                                x148 * x186,
                                -x132 * x188,
                                -x126 * x50 * x51,
                                x189 * x190,
                                -x116 * x191 * x63,
                                x114 * x194,
                                -x195 * x196,
                                4.5752363457305780158e-38 * x108 * x193,
                                -x197 * x198,
                                1.2745301248820895901e-37 * x100 * x199,
                                -x199 * x95,
                                x194 * x91,
                                -1.2745301248820895901e-37 * x192 * x200,
                                x198 * x201,
                                -4.5752363457305780158e-38 * x192 * x202,
                                x196 * x203,
                                -x191 * x71,
                                x204 * x63,
                                -x189 * x54,
                                x127 * x3 * x48,
                                x188 * x206,
                                -x186 * x207,
                                x184 * x208,
                                -x181 * x209,
                                x178 * x211,
                                -x175 * x206,
                                x168 * x211,
                                -x165 * x212,
                                x162 * x213,
                                -x157 * x213,
                                x152 * x212,
                                -x150 * x207,
                                x146 * x209,
                                -x143 * x209,
                                x137 * x206,
                                -x131 * x205,
                                x130 * x208,
                                -3.2039470208197325041e-41 * x214 * x41,
                                x214 * x215 * x43 * x48,
                                -x125 * x215 * x217,
                                x124 * x192 * x215,
                                -1.156624874515923434e-38 * x218 * x219 * x220,
                                -x221 * x222 * x54,
                                x223 * x224 * x59,
                                -2.3595147440124838053e-36 * x225 * x70,
                                x203 * x226,
                                -x202 * x228,
                                x229 * x230,
                                -4.6010537508243434204e-35 * x200 * x227,
                                x231 * x232,
                                -x229 * x234,
                                x229 * x236,
                                -x229 * x237,
                                x228 * x238,
                                -x195 * x226,
                                x231 * x239 * x5,
                                -x223 * x240 * x36,
                                x147 * x227 * x241,
                                x119 * x222 * x242,
                                -x120 * x242 * x243,
                                x244 * x245 * x247,
                                -x142 * x248 * x249,
                                x250 * x252,
                                -x149 * x253 * x254,
                                x255 * x257,
                                -x259 * x261,
                                x259 * x262,
                                -x257 * x263,
                                x254 * x256 * x264,
                                -x257 * x265 * x266,
                                x248 * x251 * x267 * x268,
                                -2.3595147440124838053e-36 * x169 * x180 * x269,
                                5.8987868600312095133e-37 * x218 * x269,
                                -x253 * x270 * x9,
                                -x241 * x272 * x9,
                                x274 * x69,
                                -x169 * x179 * x239 * x276,
                                x109 * x267 * x277,
                                -1.6516603208087386637e-35 * x108 * x272,
                                x237 * x272,
                                -x236 * x272,
                                x234 * x272,
                                -x179 * x232 * x272,
                                x235 * x278,
                                -x230 * x272,
                                1.6516603208087386637e-35 * x123 * x276,
                                -x141 * x277 * x279,
                                2.3595147440124838053e-36 * x278 * x280,
                                -x274 * x281,
                                1.0409623870643310906e-37 * x273 * x282,
                                x270 * x283,
                                -5.8987868600312095133e-37 * x15 * x45 * x52 * x70,
                                x246 * x284 * x285,
                                -7.078544232037451416e-36 * x177 * x288,
                                x173 * x265 * x287,
                                -3.0673691672162289469e-35 * x167 * x288,
                                x263 * x289,
                                -x262 * x291,
                                x261 * x291,
                                -x255 * x289,
                                3.0673691672162289469e-35 * x149 * x283,
                                -x250 * x284 * x286,
                                7.078544232037451416e-36 * x142 * x290,
                                -x247 * x284 * x84,
                                x19 * x243 * x45 * x55,
                                -x222 * x45 * x60 * x68,
                                x17 * x292 * x293,
                                -x118 * x119 * x292 * x294,
                                9.3686614835789798152e-37 * x295 * x80,
                                -9.3686614835789798152e-37 * x282 * x296,
                                -x297 * x298,
                                x299 * x302,
                                -x303 * x305,
                                x202 * x307,
                                -x201 * x309,
                                4.1409483757419090783e-34 * x200 * x306,
                                -x179 * x311,
                                x233 * x311,
                                -x140 * x308 * x312,
                                x197 * x309,
                                -x238 * x307,
                                x305 * x313,
                                -x119 * x299 * x317,
                                x297 * x318,
                                x190 * x320,
                                -2.1235632696112354248e-35 * x321 * x323,
                                6.3706898088337062743e-35 * x326 * x327,
                                -x328 * x329,
                                2.7606322504946060522e-34 * x328 * x332,
                                -x330 * x335,
                                x261 * x336,
                                -x262 * x336,
                                x335 * x338,
                                -2.7606322504946060522e-34 * x337 * x339,
                                x340 * x341 * x345,
                                -x345 * x346 * x348,
                                x326 * x349 * x350,
                                -5.308908174028088562e-36 * x110 * x118 * x275,
                                -x351 * x69,
                                x314 * x349 * x353,
                                -x344 * x348 * x354 * x358,
                                x329 * x359,
                                -2.7606322504946060522e-34 * x104 * x352,
                                x312 * x353,
                                -x360 * x361,
                                x360 * x88,
                                -4.1409483757419090783e-34 * x322 * x37 * x85,
                                2.7606322504946060522e-34 * x185 * x324,
                                -1.4864942887278647973e-34 * x359 * x365,
                                6.3706898088337062743e-35 * x353 * x366,
                                -2.1235632696112354248e-35 * x16 * x322 * x354,
                                x281 * x351,
                                x183 * x319 * x62,
                                -2.1235632696112354248e-35 * x285 * x367,
                                6.3706898088337062743e-35 * x268 * x369,
                                -x266 * x340 * x369,
                                2.7606322504946060522e-34 * x264 * x369,
                                -x165 * x370,
                                x371 * x372,
                                -x372 * x373,
                                x152 * x370,
                                -2.7606322504946060522e-34 * x282 * x374,
                                1.4864942887278647973e-34 * x367 * x375,
                                -6.3706898088337062743e-35 * x142 * x367,
                                2.1235632696112354248e-35 * x136 * x368,
                                -x320 * x54,
                                x298 * x377,
                                -x318 * x377,
                                x240 * x378,
                                -x182 * x224 * x378,
                                -1.2033525194463667407e-34 * x302 * x376,
                                x303 * x380,
                                -x202 * x382,
                                x383 * x384,
                                -2.3465374129204151444e-33 * x200 * x381,
                                x316 * x379 * x385,
                                -x383 * x386,
                                x383 * x388,
                                -x383 * x389,
                                x238 * x382,
                                -x313 * x380,
                                1.2033525194463667407e-34 * x317 * x379,
                                x390 * x391,
                                -x392 * x393,
                                x392 * x394,
                                -x395 * x69,
                                x152 * x397,
                                -x373 * x398,
                                x371 * x398,
                                -x165 * x397,
                                x396 * x399,
                                -x396 * x401,
                                x396 * x402,
                                -x392 * x403,
                                -1.2033525194463667407e-34 * x314 * x405,
                                x112 * x406,
                                -x108 * x407,
                                x389 * x404,
                                -x388 * x404,
                                x386 * x404,
                                -x385 * x405,
                                x387 * x404 * x84,
                                -x384 * x404,
                                x407 * x76,
                                -x406 * x72,
                                1.2033525194463667407e-34 * x205 * x70,
                                x403 * x408,
                                -x402 * x410,
                                x401 * x410,
                                -x399 * x410,
                                x165 * x411,
                                -x371 * x412,
                                x373 * x412,
                                -x152 * x411,
                                x281 * x395,
                                -x394 * x408,
                                x393 * x408,
                                -x391 * x409,
                                4.8134100777854669628e-34 * x106 * x322,
                                -x314 * x414 * x415,
                                4.8134100777854669628e-34 * x194 * x22 * x5,
                                -x350 * x415 * x417,
                                -1.4440230233356400889e-33 * x366 * x414,
                                x418 * x421,
                                -x363 * x413 * x422 * x425,
                                9.3861496516816605776e-33 * x323 * x81,
                                -x426 * x88,
                                x361 * x426,
                                -9.3861496516816605776e-33 * x100 * x414,
                                x424 * x427 * x428,
                                -x421 * x429 * x430,
                                x423 * x427 * x431,
                                x433 * x435,
                                -3.369387054449826874e-33 * x436 * x442,
                                x432 * x445 * x447,
                                -x437 * x441 * x449,
                                x260 * x451,
                                -x451 * x452,
                                x432 * x449 * x454,
                                -x107 * x442 * x443,
                                x314 * x455 * x456 * x458,
                                -x433 * x461,
                                -x431 * x448 * x463,
                                x459 * x464 * x465,
                                -6.2574331011211070517e-33 * x102 * x33 * x437 * x450,
                                9.3861496516816605776e-33 * x462 * x466 * x98,
                                -x233 * x33 * x467,
                                x180 * x467,
                                -9.3861496516816605776e-33 * x158 * x33 * x344 * x468,
                                x445 * x463 * x470,
                                -x101 * x464 * x468,
                                1.4440230233356400889e-33 * x14 * x434 * x462,
                                1.4440230233356400889e-33 * x471 * x474,
                                -x419 * x429 * x475 * x476,
                                6.2574331011211070517e-33 * x471 * x477,
                                -x475 * x478 * x479,
                                1.1471960685388696261e-32 * x480 * x482,
                                -1.1471960685388696261e-32 * x25 * x417 * x483 * x96,
                                x103 * x478 * x480,
                                -x16 * x425 * x439 * x484,
                                3.369387054449826874e-33 * x25 * x280 * x485,
                                -1.4440230233356400889e-33 * x327 * x417,
                                4.3320690700069202666e-33 * x434 * x97,
                                -x358 * x486 * x489,
                                x27 * x486 * x490,
                                -4.3320690700069202666e-33 * x460 * x493,
                                -1.0108161163349480622e-32 * x494 * x495,
                                x489 * x496 * x497,
                                -x494 * x499,
                                x500 * x502 * x503,
                                -x435 * x504,
                                2.8158448955044981733e-32 * x466 * x487,
                                -1.8772299303363321155e-32 * x487 * x506,
                                1.0108161163349480622e-32 * x465 * x500,
                                x507 * x508,
                                -1.8772299303363321155e-32 * x109 * x295,
                                2.8158448955044981733e-32 * x103 * x509,
                                -x510 * x511,
                                x481 * x510,
                                -x176 * x513 * x514,
                                x107 * x496 * x507,
                                -x356 * x514 * x516,
                                -1.0108161163349480622e-32 * x171 * x430,
                                1.8772299303363321155e-32 * x490 * x517,
                                -2.8158448955044981733e-32 * x474 * x85,
                                x461 * x504,
                                -x171 * x333 * x502,
                                x439 * x472 * x499,
                                -1.8772299303363321155e-32 * x268 * x469,
                                1.0108161163349480622e-32 * x347 * x362 * x444,
                                x503 * x516 * x518,
                                -1.8772299303363321155e-32 * x493 * x519,
                                x346 * x513 * x518,
                                -x162 * x520,
                                x157 * x520,
                                -x440 * x512 * x521,
                                x444 * x446 * x496 * x521,
                                -x440 * x508 * x518,
                                2.3585709381148788118e-32 * x106 * x365,
                                -2.3585709381148788118e-32 * x145 * x324,
                                2.3585709381148788118e-32 * x174 * x275,
                                -2.3585709381148788118e-32 * x522 * x523,
                                -x106 * x151 * x524 * x8,
                                6.5703047561771624043e-32 * x105 * x29 * x416,
                                -x501 * x525 * x526,
                                x468 * x525 * x90,
                                -x106 * x485 * x527,
                                x428 * x436 * x528,
                                4.3802031707847749362e-32 * x152 * x331,
                                -x152 * x530,
                                x531 * x532,
                                -x532 * x533,
                                x165 * x530,
                                -4.3802031707847749362e-32 * x167 * x170 * x529,
                                -x31 * x356 * x515 * x528,
                                x423 * x527 * x534,
                                -8.030372479772087383e-32 * x458 * x515 * x90,
                                8.030372479772087383e-32 * x266 * x333,
                                -6.5703047561771624043e-32 * x498 * x534,
                                x400 * x524,
                                4.3802031707847749362e-32 * x107 * x362 * x363,
                                -6.5703047561771624043e-32 * x479 * x523,
                                x533 * x535,
                                -x531 * x535,
                                6.5703047561771624043e-32 * x103 * x535,
                                -4.3802031707847749362e-32 * x101 * x444 * x484,
                                8.1346630314574391672e-32 * x149 * x324,
                                -8.1346630314574391672e-32 * x355 * x484 * x517,
                                8.1346630314574391672e-32 * x107 * x342 * x347 * x428,
                                -8.1346630314574391672e-32 * x264 * x469,
                                -x29 * x332 * x536,
                                x266 * x29 * x295 * x537,
                                -x342 * x444 * x447 * x538,
                                x473 * x484 * x539,
                                1.2201994547186158751e-31 * x428 * x540,
                                -x506 * x541,
                                1.491354889100530514e-31 * x505 * x99,
                                -1.2201994547186158751e-31 * x176 * x347 * x356 * x517,
                                -x477 * x539,
                                x459 * x519 * x538,
                                -x107 * x456 * x526 * x537,
                                x26 * x333 * x337 * x536,
                                x165 * x542,
                                -1.491354889100530514e-31 * x342 * x481 * x497,
                                x492 * x497 * x541,
                                -x152 * x542,
                                x330 * x544,
                                -1.8302991820779238126e-31 * x103 * x440 * x545,
                                x479 * x543 * x545,
                                -x338 * x544,
                                -2.237032333650795771e-31 * x540 * x546,
                                x437 * x547 * x548,
                                2.237032333650795771e-31 * x466 * x483,
                                -x482 * x547 * x85,
                                -2.237032333650795771e-31 * x454 * x548,
                                2.237032333650795771e-31 * x172 * x30 * x333,
                                x481 * x549,
                                -x511 * x549,
                                x511 * x550,
                                -x260 * x551,
                                x452 * x551,
                                -x481 * x550,
                            ]
                        )

                case 441:

                    def shape_functions(xi):
                        x0 = xi[0] - 1.0
                        x1 = xi[1] - 1.0
                        x2 = 2.0 * xi[1]
                        x3 = x2 + 1.0
                        x4 = 5.0 * xi[1]
                        x5 = x4 + 1.0
                        x6 = 10.0 * xi[1]
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
                            * x3
                            * x5
                            * x7
                            * x8
                            * x9
                            * xi[1]
                        )
                        x24 = x1 * x23
                        x25 = 2.0 * xi[0]
                        x26 = x25 + 1.0
                        x27 = 5.0 * xi[0]
                        x28 = x27 + 1.0
                        x29 = 10.0 * xi[0]
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
                        x46 = (
                            x26
                            * x28
                            * x30
                            * x31
                            * x32
                            * x33
                            * x34
                            * x35
                            * x36
                            * x37
                            * x38
                            * x39
                            * x40
                            * x41
                            * x42
                            * x43
                            * x44
                            * x45
                            * xi[0]
                        )
                        x47 = 6.9200736110635268372e-26 * x46
                        x48 = x24 * x47
                        x49 = xi[0] + 1.0
                        x50 = xi[1] + 1.0
                        x51 = x23 * x47 * x50
                        x52 = x24 * x49
                        x53 = (
                            x0
                            * x26
                            * x28
                            * x30
                            * x31
                            * x32
                            * x33
                            * x34
                            * x35
                            * x37
                            * x38
                            * x39
                            * x40
                            * x41
                            * x42
                            * x43
                            * x44
                            * xi[0]
                        )
                        x54 = 1.3840147222127053674e-23 * x53
                        x55 = x52 * x54
                        x56 = x36 * x52
                        x57 = (
                            x0
                            * x26
                            * x28
                            * x30
                            * x31
                            * x33
                            * x34
                            * x35
                            * x37
                            * x38
                            * x39
                            * x40
                            * x42
                            * x43
                            * x44
                            * x45
                            * xi[0]
                        )
                        x58 = 6.5740699305103504953e-23 * x57
                        x59 = x56 * x58
                        x60 = x32 * x56
                        x61 = (
                            x0
                            * x26
                            * x28
                            * x30
                            * x31
                            * x33
                            * x34
                            * x37
                            * x38
                            * x39
                            * x40
                            * x41
                            * x42
                            * x43
                            * x45
                            * xi[0]
                        )
                        x62 = 7.8888839166124205944e-22 * x61
                        x63 = x60 * x62
                        x64 = x35 * x60
                        x65 = (
                            x0
                            * x26
                            * x28
                            * x30
                            * x31
                            * x34
                            * x37
                            * x38
                            * x39
                            * x40
                            * x41
                            * x43
                            * x44
                            * x45
                            * xi[0]
                        )
                        x66 = 1.6763878322801393763e-21 * x65
                        x67 = x64 * x66
                        x68 = x33 * x64
                        x69 = (
                            x0
                            * x28
                            * x30
                            * x31
                            * x34
                            * x38
                            * x39
                            * x40
                            * x41
                            * x42
                            * x43
                            * x44
                            * x45
                            * xi[0]
                        )
                        x70 = 2.1457764253185784017e-21 * x69
                        x71 = x68 * x70
                        x72 = x26 * x68
                        x73 = (
                            x0
                            * x28
                            * x30
                            * x34
                            * x37
                            * x38
                            * x39
                            * x41
                            * x42
                            * x43
                            * x44
                            * x45
                            * xi[0]
                        )
                        x74 = 1.341110265824111501e-20 * x73
                        x75 = x72 * x74
                        x76 = x31 * x72
                        x77 = (
                            x0
                            * x28
                            * x30
                            * x37
                            * x38
                            * x39
                            * x40
                            * x41
                            * x42
                            * x44
                            * x45
                            * xi[0]
                        )
                        x78 = 5.3644410632964460042e-20 * x77
                        x79 = x76 * x78
                        x80 = x34 * x76
                        x81 = (
                            x0
                            * x30
                            * x37
                            * x39
                            * x40
                            * x41
                            * x42
                            * x43
                            * x44
                            * x45
                            * xi[0]
                        )
                        x82 = 4.3586083639283623784e-20 * x81
                        x83 = x80 * x82
                        x84 = x0 * x37 * x38 * x40 * x41 * x42 * x43 * x44 * x45 * xi[0]
                        x85 = x39 * x84
                        x86 = 1.1622955637142299676e-19 * x28
                        x87 = x80 * x86
                        x88 = (
                            7594058.4281266233059 * xi[0] ** 20
                            - 29237124.948287499728 * xi[0] ** 18
                            + 46662451.417466849566 * xi[0] ** 16
                            - 40202717.496749499983 * xi[0] ** 14
                            + 20418933.234909475907 * xi[0] ** 12
                            - 6274158.9818744284408 * xi[0] ** 10
                            + 1153141.6151619398331 * xi[0] ** 8
                            - 121028.00916570045942 * xi[0] ** 6
                            + 6598.7171853566529492 * xi[0] ** 4
                            - 154.97677311665406904 * xi[0] ** 2
                            + 1.0
                        )
                        x89 = x1 * x88
                        x90 = 2.6306032789197855094e-13 * x23
                        x91 = x30 * x84
                        x92 = (
                            x10
                            * x11
                            * x12
                            * x14
                            * x15
                            * x16
                            * x17
                            * x18
                            * x19
                            * x20
                            * x21
                            * x22
                            * x3
                            * x5
                            * x50
                            * x7
                            * x8
                            * xi[1]
                        )
                        x93 = x1 * x46
                        x94 = x49 * x93
                        x95 = x92 * x94
                        x96 = 1.3840147222127053674e-23 * x9
                        x97 = 6.5740699305103504953e-23 * x13
                        x98 = (
                            x10
                            * x11
                            * x13
                            * x14
                            * x15
                            * x16
                            * x17
                            * x18
                            * x19
                            * x20
                            * x21
                            * x3
                            * x5
                            * x50
                            * x7
                            * x8
                            * xi[1]
                        )
                        x99 = x94 * x98
                        x100 = x22 * x9
                        x101 = 7.8888839166124205944e-22 * x100
                        x102 = (
                            x100
                            * x11
                            * x14
                            * x15
                            * x16
                            * x17
                            * x19
                            * x20
                            * x21
                            * x3
                            * x5
                            * x50
                            * x7
                            * x8
                            * xi[1]
                        )
                        x103 = x12 * x94
                        x104 = x102 * x103
                        x105 = x13 * x18
                        x106 = 1.6763878322801393763e-21 * x105
                        x107 = (
                            x105
                            * x11
                            * x14
                            * x15
                            * x16
                            * x17
                            * x19
                            * x20
                            * x5
                            * x50
                            * x7
                            * x8
                            * xi[1]
                        )
                        x108 = x10 * x103
                        x109 = x107 * x108
                        x110 = x100 * x21
                        x111 = 2.1457764253185784017e-21 * x110
                        x112 = (
                            x11
                            * x110
                            * x14
                            * x15
                            * x16
                            * x17
                            * x20
                            * x5
                            * x50
                            * x7
                            * xi[1]
                        )
                        x113 = x108 * x3
                        x114 = x112 * x113
                        x115 = x105 * x19
                        x116 = 1.341110265824111501e-20 * x115
                        x117 = x115 * x15 * x16 * x17 * x20 * x5 * x50 * x7 * xi[1]
                        x118 = x113 * x8
                        x119 = x117 * x118
                        x120 = x110 * x14
                        x121 = 5.3644410632964460042e-20 * x120
                        x122 = x120 * x15 * x16 * x20 * x50 * x7 * xi[1]
                        x123 = x11 * x118
                        x124 = x122 * x123
                        x125 = x115 * x17
                        x126 = 4.3586083639283623784e-20 * x125
                        x127 = x125 * x15 * x16 * x50 * xi[1]
                        x128 = x120 * x20
                        x129 = x127 * x128
                        x130 = x123 * x5
                        x131 = 1.1622955637142299676e-19 * x130
                        x132 = (
                            7594058.4281266233059 * xi[1] ** 20
                            - 29237124.948287499728 * xi[1] ** 18
                            + 46662451.417466849566 * xi[1] ** 16
                            - 40202717.496749499983 * xi[1] ** 14
                            + 20418933.234909475907 * xi[1] ** 12
                            - 6274158.9818744284408 * xi[1] ** 10
                            + 1153141.6151619398331 * xi[1] ** 8
                            - 121028.00916570045942 * xi[1] ** 6
                            + 6598.7171853566529492 * xi[1] ** 4
                            - 154.97677311665406904 * xi[1] ** 2
                            + 1.0
                        )
                        x133 = x132 * x49
                        x134 = 2.6306032789197855094e-13 * x46
                        x135 = x125 * x15
                        x136 = x128 * x50 * xi[1]
                        x137 = x136 * x7
                        x138 = x135 * x137
                        x139 = x137 * x16
                        x140 = x126 * x139
                        x141 = x127 * x7
                        x142 = x121 * x141
                        x143 = x116 * x5
                        x144 = x11 * x111
                        x145 = x106 * x8
                        x146 = x101 * x3
                        x147 = x10 * x97
                        x148 = x12 * x96
                        x149 = x36 * x49
                        x150 = x149 * x23 * x50
                        x151 = x150 * x32
                        x152 = x151 * x35
                        x153 = x152 * x33
                        x154 = x153 * x26
                        x155 = x154 * x31
                        x156 = x155 * x34
                        x157 = x156 * x82
                        x158 = x156 * x86
                        x159 = x45 * x49
                        x160 = x0 * x93
                        x161 = x160 * x98
                        x162 = x12 * x160
                        x163 = x102 * x162
                        x164 = x10 * x162
                        x165 = x107 * x164
                        x166 = x164 * x3
                        x167 = x112 * x166
                        x168 = x166 * x8
                        x169 = x117 * x168
                        x170 = x11 * x168
                        x171 = x122 * x170
                        x172 = x170 * x5
                        x173 = 1.1622955637142299676e-19 * x172
                        x174 = x160 * x92
                        x175 = x1 * x53
                        x176 = x159 * x175
                        x177 = x9 * x92
                        x178 = 2.7680294444254107349e-21 * x177
                        x179 = x149 * x175
                        x180 = x179 * x98
                        x181 = x12 * x9
                        x182 = 2.7680294444254107349e-21 * x181
                        x183 = x176 * x98
                        x184 = x1 * x149
                        x185 = x177 * x184
                        x186 = 1.3148139861020700991e-20 * x57
                        x187 = x185 * x186
                        x188 = x185 * x32
                        x189 = 1.5777767833224841189e-19 * x61
                        x190 = x188 * x189
                        x191 = x188 * x35
                        x192 = 3.3527756645602787526e-19 * x65
                        x193 = x191 * x192
                        x194 = x191 * x33
                        x195 = 4.2915528506371568033e-19 * x69
                        x196 = x194 * x195
                        x197 = x194 * x26
                        x198 = 2.6822205316482230021e-18 * x73
                        x199 = x197 * x198
                        x200 = x197 * x31
                        x201 = 1.0728882126592892008e-17 * x77
                        x202 = x200 * x201
                        x203 = x200 * x34
                        x204 = 8.7172167278567247568e-18 * x81
                        x205 = x203 * x204
                        x206 = 2.3245911274284599351e-17 * x28
                        x207 = x203 * x206
                        x208 = x89 * x92
                        x209 = 5.2612065578395710189e-11 * x9
                        x210 = 1.3148139861020700991e-20 * x13
                        x211 = x179 * x210
                        x212 = 1.5777767833224841189e-19 * x100
                        x213 = x12 * x179
                        x214 = 3.3527756645602787526e-19 * x105
                        x215 = x213 * x214
                        x216 = x10 * x213
                        x217 = x107 * x216
                        x218 = 4.2915528506371568033e-19 * x110
                        x219 = x216 * x3
                        x220 = 2.6822205316482230021e-18 * x115
                        x221 = x219 * x220
                        x222 = x219 * x8
                        x223 = x117 * x222
                        x224 = 1.0728882126592892008e-17 * x120
                        x225 = x11 * x222
                        x226 = 8.7172167278567247568e-18 * x125
                        x227 = x225 * x226
                        x228 = x225 * x5
                        x229 = 2.3245911274284599351e-17 * x228
                        x230 = x133 * x36
                        x231 = 5.2612065578395710189e-11 * x53
                        x232 = x139 * x5
                        x233 = x141 * x224
                        x234 = x11 * x8
                        x235 = x234 * x5
                        x236 = x122 * x235
                        x237 = x10 * x3
                        x238 = x237 * x8
                        x239 = x112 * x238
                        x240 = x10 * x12
                        x241 = x102 * x240
                        x242 = x184 * x32
                        x243 = x242 * x98
                        x244 = x181 * x243
                        x245 = x244 * x35
                        x246 = x245 * x33
                        x247 = x246 * x26
                        x248 = x247 * x31
                        x249 = x248 * x34
                        x250 = x204 * x249
                        x251 = x206 * x249
                        x252 = x12 * x89
                        x253 = x184 * x41
                        x254 = x253 * x98
                        x255 = x176 * x210
                        x256 = x12 * x176
                        x257 = x237 * x256
                        x258 = x214 * x256
                        x259 = x234 * x257
                        x260 = x220 * x257
                        x261 = x235 * x257
                        x262 = x226 * x259
                        x263 = 2.3245911274284599351e-17 * x261
                        x264 = x117 * x238
                        x265 = x107 * x240
                        x266 = x253 * x57
                        x267 = x13 * x92
                        x268 = 6.2453664339848329705e-20 * x267
                        x269 = x242 * x57
                        x270 = x13 * x241
                        x271 = 6.2453664339848329705e-20 * x270
                        x272 = x44 * x61
                        x273 = x242 * x267
                        x274 = 7.4944397207817995646e-19 * x273
                        x275 = x273 * x35
                        x276 = 1.5925684406661324075e-18 * x65
                        x277 = x275 * x276
                        x278 = x275 * x33
                        x279 = 2.0384876040526494816e-18 * x69
                        x280 = x278 * x279
                        x281 = x26 * x278
                        x282 = 1.274054752532905926e-17 * x73
                        x283 = x281 * x282
                        x284 = x281 * x31
                        x285 = 5.096219010131623704e-17 * x77
                        x286 = x284 * x285
                        x287 = x284 * x34
                        x288 = 4.1406779457319442595e-17 * x81
                        x289 = x287 * x288
                        x290 = 1.1041807855285184692e-16 * x28
                        x291 = x287 * x290
                        x292 = 2.499073114973796234e-10 * x13
                        x293 = x35 * x61
                        x294 = 7.4944397207817995646e-19 * x100
                        x295 = x294 * x57
                        x296 = x12 * x269
                        x297 = 1.5925684406661324075e-18 * x105
                        x298 = x296 * x297
                        x299 = 2.0384876040526494816e-18 * x110
                        x300 = x265 * x299
                        x301 = x237 * x296
                        x302 = 1.274054752532905926e-17 * x115
                        x303 = x301 * x302
                        x304 = 5.096219010131623704e-17 * x120
                        x305 = x264 * x304
                        x306 = x234 * x301
                        x307 = 4.1406779457319442595e-17 * x125
                        x308 = x306 * x307
                        x309 = x235 * x301
                        x310 = 1.1041807855285184692e-16 * x309
                        x311 = x230 * x32
                        x312 = 2.499073114973796234e-10 * x57
                        x313 = x141 * x304
                        x314 = x117 * x299
                        x315 = x107 * x294
                        x316 = x242 * x270
                        x317 = 7.4944397207817995646e-19 * x316
                        x318 = x33 * x35
                        x319 = x316 * x318
                        x320 = x26 * x319
                        x321 = x31 * x320
                        x322 = x321 * x34
                        x323 = x288 * x322
                        x324 = x290 * x322
                        x325 = x10 * x252
                        x326 = x35 * x42
                        x327 = x12 * x266
                        x328 = x237 * x327
                        x329 = x297 * x327
                        x330 = x234 * x328
                        x331 = x302 * x328
                        x332 = x235 * x328
                        x333 = x307 * x330
                        x334 = 1.1041807855285184692e-16 * x332
                        x335 = 8.9933276649381594776e-18 * x100
                        x336 = x243 * x335
                        x337 = x242 * x293
                        x338 = x12 * x337
                        x339 = x237 * x338
                        x340 = x107 * x335
                        x341 = x242 * x272
                        x342 = x12 * x237
                        x343 = x341 * x342
                        x344 = x326 * x65
                        x345 = x100 * x243
                        x346 = 1.911082128799358889e-17 * x345
                        x347 = x318 * x345
                        x348 = 2.4461851248631793779e-17 * x69
                        x349 = x347 * x348
                        x350 = x26 * x347
                        x351 = 1.5288657030394871112e-16 * x73
                        x352 = x350 * x351
                        x353 = x31 * x350
                        x354 = 6.1154628121579484447e-16 * x77
                        x355 = x353 * x354
                        x356 = x34 * x353
                        x357 = 4.9688135348783331114e-16 * x81
                        x358 = x356 * x357
                        x359 = 1.325016942634222163e-15 * x28
                        x360 = x356 * x359
                        x361 = 2.9988877379685554807e-9 * x100
                        x362 = x318 * x65
                        x363 = 1.911082128799358889e-17 * x105
                        x364 = x338 * x363
                        x365 = 2.4461851248631793779e-17 * x110
                        x366 = x265 * x365
                        x367 = 1.5288657030394871112e-16 * x115
                        x368 = x339 * x367
                        x369 = 6.1154628121579484447e-16 * x120
                        x370 = x264 * x369
                        x371 = x234 * x339
                        x372 = 4.9688135348783331114e-16 * x125
                        x373 = x371 * x372
                        x374 = x235 * x339
                        x375 = 1.325016942634222163e-15 * x374
                        x376 = x311 * x35
                        x377 = 2.9988877379685554807e-9 * x61
                        x378 = x141 * x369
                        x379 = x117 * x365
                        x380 = x242 * x362
                        x381 = x342 * x380
                        x382 = x100 * x107
                        x383 = 1.911082128799358889e-17 * x382
                        x384 = x242 * x342
                        x385 = x26 * x318
                        x386 = x384 * x385
                        x387 = x382 * x386
                        x388 = x31 * x387
                        x389 = x34 * x388
                        x390 = x357 * x389
                        x391 = x359 * x389
                        x392 = x3 * x325
                        x393 = x318 * x37
                        x394 = x384 * x393
                        x395 = x344 * x384
                        x396 = x12 * x341
                        x397 = x363 * x396
                        x398 = x234 * x343
                        x399 = x343 * x367
                        x400 = x235 * x343
                        x401 = x372 * x398
                        x402 = 1.325016942634222163e-15 * x400
                        x403 = x242 * x344
                        x404 = 4.0610495236986376391e-17 * x105 * x12
                        x405 = x102 * x404
                        x406 = x239 * x404
                        x407 = x242 * x69
                        x408 = x393 * x407
                        x409 = x105 * x12
                        x410 = x102 * x409
                        x411 = 5.198143390334256178e-17 * x410
                        x412 = x40 * x73
                        x413 = x242 * x385
                        x414 = x410 * x413
                        x415 = 3.2488396189589101113e-16 * x414
                        x416 = x31 * x414
                        x417 = 1.2995358475835640445e-15 * x77
                        x418 = x416 * x417
                        x419 = x34 * x416
                        x420 = 1.0558728761616457862e-15 * x81
                        x421 = x419 * x420
                        x422 = 2.8156610030977220964e-15 * x28
                        x423 = x419 * x422
                        x424 = 6.3726364431831803966e-9 * x105
                        x425 = x31 * x73
                        x426 = x385 * x407
                        x427 = 5.198143390334256178e-17 * x110
                        x428 = x265 * x427
                        x429 = 3.2488396189589101113e-16 * x115
                        x430 = x381 * x429
                        x431 = 1.2995358475835640445e-15 * x120
                        x432 = x12 * x264
                        x433 = x431 * x432
                        x434 = x234 * x381
                        x435 = 1.0558728761616457862e-15 * x125
                        x436 = x434 * x435
                        x437 = x235 * x381
                        x438 = 2.8156610030977220964e-15 * x437
                        x439 = x33 * x376
                        x440 = 6.3726364431831803966e-9 * x65
                        x441 = x141 * x431
                        x442 = x117 * x427
                        x443 = x239 * x409
                        x444 = 5.198143390334256178e-17 * x443
                        x445 = x413 * x443
                        x446 = 3.2488396189589101113e-16 * x445
                        x447 = x31 * x34
                        x448 = x445 * x447
                        x449 = x420 * x448
                        x450 = x422 * x448
                        x451 = x392 * x8
                        x452 = x31 * x43
                        x453 = x234 * x395
                        x454 = x395 * x429
                        x455 = x235 * x395
                        x456 = x435 * x453
                        x457 = 2.8156610030977220964e-15 * x455
                        x458 = 6.6536235396278479079e-17 * x110
                        x459 = x265 * x458
                        x460 = x386 * x69
                        x461 = x234 * x460
                        x462 = x117 * x458
                        x463 = x394 * x69
                        x464 = x234 * x463
                        x465 = x412 * x413
                        x466 = 4.1585147122674049424e-16 * x110
                        x467 = x265 * x466
                        x468 = x110 * x265 * x413
                        x469 = x452 * x77
                        x470 = 1.663405884906961977e-15 * x469
                        x471 = x447 * x468
                        x472 = 1.3515172814869066063e-15 * x81
                        x473 = x471 * x472
                        x474 = 3.6040460839650842834e-15 * x28
                        x475 = x471 * x474
                        x476 = 8.1569746472744709076e-9 * x110
                        x477 = 1.663405884906961977e-15 * x77
                        x478 = x413 * x425
                        x479 = 4.1585147122674049424e-16 * x115
                        x480 = x460 * x479
                        x481 = 1.663405884906961977e-15 * x120
                        x482 = x432 * x481
                        x483 = 1.3515172814869066063e-15 * x125
                        x484 = x461 * x483
                        x485 = x235 * x460
                        x486 = 3.6040460839650842834e-15 * x485
                        x487 = x26 * x439
                        x488 = 8.1569746472744709076e-9 * x69
                        x489 = x141 * x481
                        x490 = x386 * x425
                        x491 = x117 * x234
                        x492 = x466 * x491
                        x493 = x386 * x447
                        x494 = x110 * x491
                        x495 = x493 * x494
                        x496 = x472 * x495
                        x497 = x474 * x495
                        x498 = x11 * x451
                        x499 = x386 * x412
                        x500 = x463 * x479
                        x501 = x235 * x463
                        x502 = x464 * x483
                        x503 = 3.6040460839650842834e-15 * x501
                        x504 = 2.599071695167128089e-15 * x115
                        x505 = x112 * x504
                        x506 = x236 * x504
                        x507 = x386 * x469
                        x508 = x112 * x115
                        x509 = 1.0396286780668512356e-14 * x508
                        x510 = x38 * x81
                        x511 = x493 * x508
                        x512 = 8.4469830092931662893e-15 * x511
                        x513 = 2.2525288024781776771e-14 * x28
                        x514 = x511 * x513
                        x515 = 5.0981091545465443173e-8 * x115
                        x516 = x28 * x81
                        x517 = x493 * x77
                        x518 = 1.0396286780668512356e-14 * x120
                        x519 = x432 * x518
                        x520 = x125 * x234
                        x521 = 8.4469830092931662893e-15 * x520
                        x522 = x490 * x521
                        x523 = x235 * x490
                        x524 = 2.2525288024781776771e-14 * x523
                        x525 = x31 * x487
                        x526 = 5.0981091545465443173e-8 * x73
                        x527 = x141 * x518
                        x528 = x115 * x236
                        x529 = 1.0396286780668512356e-14 * x528
                        x530 = x493 * x528
                        x531 = 8.4469830092931662893e-15 * x530
                        x532 = x513 * x530
                        x533 = x498 * x5
                        x534 = x235 * x499
                        x535 = x499 * x521
                        x536 = 2.2525288024781776771e-14 * x534
                        x537 = 4.1585147122674049424e-14 * x120
                        x538 = x413 * x432
                        x539 = x537 * x538
                        x540 = x235 * x517
                        x541 = x141 * x537
                        x542 = x235 * x507
                        x543 = x447 * x538
                        x544 = 3.3787932037172665157e-14 * x120
                        x545 = x510 * x544
                        x546 = x28 * x85
                        x547 = 9.0101152099127107086e-14 * x120
                        x548 = x543 * x547
                        x549 = 2.0392436618186177269e-7 * x120
                        x550 = x28 * x91
                        x551 = x516 * x544
                        x552 = 3.3787932037172665157e-14 * x520
                        x553 = x517 * x552
                        x554 = 9.0101152099127107086e-14 * x540
                        x555 = x34 * x525
                        x556 = 2.0392436618186177269e-7 * x77
                        x557 = x235 * x493
                        x558 = x141 * x557
                        x559 = x547 * x558
                        x560 = x533 * x7
                        x561 = x507 * x552
                        x562 = 9.0101152099127107086e-14 * x542
                        x563 = x493 * x520
                        x564 = 2.745269478020279044e-14 * x563
                        x565 = x122 * x564
                        x566 = x232 * x564
                        x567 = 7.3207186080540774507e-14 * x563
                        x568 = x122 * x567
                        x569 = 1.6568854752276269031e-7 * x125
                        x570 = 7.3207186080540774507e-14 * x557
                        x571 = x516 * x570
                        x572 = x28 * x555
                        x573 = 1.6568854752276269031e-7 * x81
                        x574 = x232 * x567
                        x575 = x136 * x560
                        x576 = x510 * x570
                        x577 = 1.9521916288144206535e-13 * x557
                        x578 = x129 * x577
                        x579 = x138 * x577
                        x580 = 4.4183612672736717416e-7 * x572
                        return jnp.asarray(
                            [
                                x0 * x48,
                                x48 * x49,
                                x49 * x51,
                                x0 * x51,
                                -x45 * x55,
                                x41 * x59,
                                -x44 * x63,
                                x42 * x67,
                                -x37 * x71,
                                x40 * x75,
                                -x43 * x79,
                                x38 * x83,
                                -x85 * x87,
                                x89 * x90,
                                -x87 * x91,
                                x28 * x83,
                                -x34 * x79,
                                x31 * x75,
                                -x26 * x71,
                                x33 * x67,
                                -x35 * x63,
                                x32 * x59,
                                -x36 * x55,
                                -x95 * x96,
                                x95 * x97,
                                -x101 * x99,
                                x104 * x106,
                                -x109 * x111,
                                x114 * x116,
                                -x119 * x121,
                                x124 * x126,
                                -x129 * x131,
                                x133 * x134,
                                -x131 * x138,
                                x130 * x140,
                                -x130 * x142,
                                x124 * x143,
                                -x119 * x144,
                                x114 * x145,
                                -x109 * x146,
                                x104 * x147,
                                -x148 * x99,
                                -x150 * x54,
                                x151 * x58,
                                -x152 * x62,
                                x153 * x66,
                                -x154 * x70,
                                x155 * x74,
                                -x156 * x78,
                                x157 * x28,
                                -x158 * x91,
                                x50 * x88 * x90,
                                -x158 * x85,
                                x157 * x38,
                                -x155 * x43 * x78,
                                x154 * x40 * x74,
                                -x153 * x37 * x70,
                                x152 * x42 * x66,
                                -x151 * x44 * x62,
                                x150 * x41 * x58,
                                -x159 * x23 * x50 * x54,
                                -x148 * x161,
                                x147 * x163,
                                -x146 * x165,
                                x145 * x167,
                                -x144 * x169,
                                x143 * x171,
                                -x142 * x172,
                                x140 * x172,
                                -x138 * x173,
                                x0 * x132 * x134,
                                -x129 * x173,
                                x126 * x171,
                                -x121 * x169,
                                x116 * x167,
                                -x111 * x165,
                                x106 * x163,
                                -x101 * x161,
                                x174 * x97,
                                -x174 * x96,
                                x176 * x178,
                                x178 * x179,
                                x180 * x182,
                                x182 * x183,
                                -x187 * x41,
                                x190 * x44,
                                -x193 * x42,
                                x196 * x37,
                                -x199 * x40,
                                x202 * x43,
                                -x205 * x38,
                                x207 * x85,
                                -x208 * x209,
                                x207 * x91,
                                -x205 * x28,
                                x202 * x34,
                                -x199 * x31,
                                x196 * x26,
                                -x193 * x33,
                                x190 * x35,
                                -x187 * x32,
                                -x211 * x92,
                                x180 * x212,
                                -x102 * x215,
                                x217 * x218,
                                -x112 * x221,
                                x223 * x224,
                                -x122 * x227,
                                x129 * x229,
                                -x230 * x231,
                                x138 * x229,
                                -x227 * x232,
                                x228 * x233,
                                -x221 * x236,
                                x11 * x218 * x223,
                                -x215 * x239,
                                x212 * x217 * x3,
                                -x211 * x241,
                                -x186 * x244,
                                x189 * x245,
                                -x192 * x246,
                                x195 * x247,
                                -x198 * x248,
                                x201 * x249,
                                -x250 * x28,
                                x251 * x91,
                                -x209 * x252 * x98,
                                x251 * x85,
                                -x250 * x38,
                                x201 * x248 * x43,
                                -x198 * x247 * x40,
                                x195 * x246 * x37,
                                -x192 * x245 * x42,
                                x189 * x244 * x44,
                                -x181 * x186 * x254,
                                -x241 * x255,
                                x107 * x212 * x257,
                                -x239 * x258,
                                x117 * x218 * x259,
                                -x236 * x260,
                                x233 * x261,
                                -x232 * x262,
                                x138 * x263,
                                -x133 * x231 * x45,
                                x129 * x263,
                                -x122 * x262,
                                x224 * x256 * x264,
                                -x112 * x260,
                                x176 * x218 * x265,
                                -x102 * x258,
                                x183 * x212,
                                -x255 * x92,
                                x266 * x268,
                                x268 * x269,
                                x269 * x271,
                                x266 * x271,
                                -x272 * x274,
                                x277 * x42,
                                -x280 * x37,
                                x283 * x40,
                                -x286 * x43,
                                x289 * x38,
                                -x291 * x85,
                                x208 * x292,
                                -x291 * x91,
                                x28 * x289,
                                -x286 * x34,
                                x283 * x31,
                                -x26 * x280,
                                x277 * x33,
                                -x274 * x293,
                                -x243 * x295,
                                x102 * x298,
                                -x269 * x300,
                                x112 * x303,
                                -x296 * x305,
                                x122 * x308,
                                -x129 * x310,
                                x311 * x312,
                                -x138 * x310,
                                x232 * x308,
                                -x309 * x313,
                                x236 * x303,
                                -x306 * x314,
                                x239 * x298,
                                -x301 * x315,
                                -x293 * x317,
                                x276 * x319,
                                -x279 * x320,
                                x282 * x321,
                                -x285 * x322,
                                x28 * x323,
                                -x324 * x91,
                                x102 * x292 * x325,
                                -x324 * x85,
                                x323 * x38,
                                -x285 * x321 * x43,
                                x282 * x320 * x40,
                                -x279 * x319 * x37,
                                x276 * x316 * x326,
                                -x272 * x317,
                                -x315 * x328,
                                x239 * x329,
                                -x314 * x330,
                                x236 * x331,
                                -x313 * x332,
                                x232 * x333,
                                -x138 * x334,
                                x230 * x312 * x41,
                                -x129 * x334,
                                x122 * x333,
                                -x305 * x327,
                                x112 * x331,
                                -x266 * x300,
                                x102 * x329,
                                -x254 * x295,
                                x272 * x336,
                                x293 * x336,
                                x339 * x340,
                                x340 * x343,
                                -x344 * x346,
                                x349 * x37,
                                -x352 * x40,
                                x355 * x43,
                                -x358 * x38,
                                x360 * x85,
                                -x361 * x89 * x98,
                                x360 * x91,
                                -x28 * x358,
                                x34 * x355,
                                -x31 * x352,
                                x26 * x349,
                                -x346 * x362,
                                -x102 * x364,
                                x337 * x366,
                                -x112 * x368,
                                x338 * x370,
                                -x122 * x373,
                                x129 * x375,
                                -x376 * x377,
                                x138 * x375,
                                -x232 * x373,
                                x374 * x378,
                                -x236 * x368,
                                x371 * x379,
                                -x239 * x364,
                                -x381 * x383,
                                x348 * x387,
                                -x351 * x388,
                                x354 * x389,
                                -x28 * x390,
                                x391 * x91,
                                -x107 * x361 * x392,
                                x391 * x85,
                                -x38 * x390,
                                x354 * x388 * x43,
                                -x351 * x387 * x40,
                                x348 * x382 * x394,
                                -x383 * x395,
                                -x239 * x397,
                                x379 * x398,
                                -x236 * x399,
                                x378 * x400,
                                -x232 * x401,
                                x138 * x402,
                                -x311 * x377 * x44,
                                x129 * x402,
                                -x122 * x401,
                                x370 * x396,
                                -x112 * x399,
                                x341 * x366,
                                -x102 * x397,
                                x403 * x405,
                                x380 * x405,
                                x380 * x406,
                                x403 * x406,
                                -x408 * x411,
                                x412 * x415,
                                -x418 * x43,
                                x38 * x421,
                                -x423 * x85,
                                x102 * x252 * x424,
                                -x423 * x91,
                                x28 * x421,
                                -x34 * x418,
                                x415 * x425,
                                -x411 * x426,
                                -x380 * x428,
                                x112 * x430,
                                -x380 * x433,
                                x122 * x436,
                                -x129 * x438,
                                x439 * x440,
                                -x138 * x438,
                                x232 * x436,
                                -x437 * x441,
                                x236 * x430,
                                -x434 * x442,
                                -x426 * x444,
                                x425 * x446,
                                -x417 * x448,
                                x28 * x449,
                                -x450 * x91,
                                x112 * x424 * x451,
                                -x450 * x85,
                                x38 * x449,
                                -x417 * x445 * x452,
                                x412 * x446,
                                -x408 * x444,
                                -x442 * x453,
                                x236 * x454,
                                -x441 * x455,
                                x232 * x456,
                                -x138 * x457,
                                x376 * x42 * x440,
                                -x129 * x457,
                                x122 * x456,
                                -x403 * x433,
                                x112 * x454,
                                -x403 * x428,
                                x408 * x459,
                                x426 * x459,
                                x461 * x462,
                                x462 * x464,
                                -x465 * x467,
                                x468 * x470,
                                -x38 * x473,
                                x475 * x85,
                                -x107 * x325 * x476,
                                x475 * x91,
                                -x28 * x473,
                                x471 * x477,
                                -x467 * x478,
                                -x112 * x480,
                                x426 * x482,
                                -x122 * x484,
                                x129 * x486,
                                -x487 * x488,
                                x138 * x486,
                                -x232 * x484,
                                x485 * x489,
                                -x236 * x480,
                                -x490 * x492,
                                x477 * x495,
                                -x28 * x496,
                                x497 * x91,
                                -x117 * x476 * x498,
                                x497 * x85,
                                -x38 * x496,
                                x386 * x470 * x494,
                                -x492 * x499,
                                -x236 * x500,
                                x489 * x501,
                                -x232 * x502,
                                x138 * x503,
                                -x37 * x439 * x488,
                                x129 * x503,
                                -x122 * x502,
                                x408 * x482,
                                -x112 * x500,
                                x499 * x505,
                                x490 * x505,
                                x490 * x506,
                                x499 * x506,
                                -x507 * x509,
                                x510 * x512,
                                -x514 * x85,
                                x112 * x392 * x515,
                                -x514 * x91,
                                x512 * x516,
                                -x509 * x517,
                                -x478 * x519,
                                x122 * x522,
                                -x129 * x524,
                                x525 * x526,
                                -x138 * x524,
                                x232 * x522,
                                -x523 * x527,
                                -x517 * x529,
                                x516 * x531,
                                -x532 * x91,
                                x122 * x515 * x533,
                                -x532 * x85,
                                x510 * x531,
                                -x507 * x529,
                                -x527 * x534,
                                x232 * x535,
                                -x138 * x536,
                                x40 * x487 * x526,
                                -x129 * x536,
                                x122 * x535,
                                -x465 * x519,
                                x469 * x539,
                                x447 * x539 * x77,
                                x540 * x541,
                                x541 * x542,
                                -x543 * x545,
                                x546 * x548,
                                -x117 * x451 * x549,
                                x548 * x550,
                                -x543 * x551,
                                -x122 * x553,
                                x129 * x554,
                                -x555 * x556,
                                x138 * x554,
                                -x232 * x553,
                                -x551 * x558,
                                x550 * x559,
                                -x127 * x549 * x560,
                                x546 * x559,
                                -x545 * x558,
                                -x232 * x561,
                                x138 * x562,
                                -x43 * x525 * x556,
                                x129 * x562,
                                -x122 * x561,
                                x510 * x565,
                                x516 * x565,
                                x516 * x566,
                                x510 * x566,
                                -x546 * x568,
                                x122 * x498 * x569,
                                -x550 * x568,
                                -x129 * x571,
                                x572 * x573,
                                -x138 * x571,
                                -x550 * x574,
                                x16 * x569 * x575,
                                -x546 * x574,
                                -x138 * x576,
                                x38 * x555 * x573,
                                -x129 * x576,
                                x546 * x578,
                                x550 * x578,
                                x550 * x579,
                                x546 * x579,
                                -4.4183612672736717416e-7 * x129 * x533,
                                -x580 * x91,
                                -4.4183612672736717416e-7 * x135 * x575,
                                -x580 * x85,
                                x132 * x88,
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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

    shape_functions = jax.jit(shape_functions)

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
            dX_dxi = jax.jacfwd(initial_coor)(xi)

            fun = lambda xi: jnp.einsum('i, i...-> ...', shape_functions(xi), fI)
            primal_out = fun(xi)
            df_dxi = jax.jacfwd(fun)(xi)

            tangent_out = jnp.einsum('...i, i-> ...', df_dxi, matrix_inv(dX_dxi) @ x_dot)

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
                        return jnp.asarray([0.5 - x0, x0 + 0.5])

                case 3:

                    def shape_functions(xi):
                        x0 = 0.5 * xi
                        return jnp.asarray(
                            [x0 * (xi - 1.0), x0 * (xi + 1.0), 1.0 - xi**2]
                        )

                case 4:

                    def shape_functions(xi):
                        x0 = 3.0 * xi
                        x1 = x0 + 1.0
                        x2 = xi + 1.0
                        x3 = x2 * (x0 - 1.0)
                        x4 = 0.5625 * xi - 0.5625
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                        return jnp.asarray(
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
                
                        return jnp.asarray([-xi[0] - xi[1] + 1, xi[0], xi[1]])

                case 6:
                    def shape_functions(xi):
                        x0 = 4*xi[0]
                        x1 = x0*xi[1]
                        x2 = -xi[0] - xi[1] + 1
                        return jnp.asarray([x1 + 2*xi[0]**2 - 3*xi[0] + 2*xi[1]**2 - 3*xi[1] + 1, xi[0]*(2*xi[0] - 1), xi[1]*(2*xi[1] - 1), x0*x2, x1, 4*x2*xi[1]])

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
                        return jnp.asarray([-27*x0*xi[1]/2 + 9*x0 - 27*x1*xi[0]/2 + 9*x1 - 9*xi[0]**3/2 + 18*xi[0]*xi[1] - 11*xi[0]/2 - 9*xi[1]**3/2 - 11*xi[1]/2 + 1, xi[0]*(9*x0 - 9*xi[0] + 2)/2, xi[1]*(9*x1 - 9*xi[1] + 2)/2, x5*x6, x6*(-x3 - x8 + 4*xi[0] + xi[1]), x9*(x7 - 1), x9*(3*xi[1] - 1), x10*(-x4 - x8 + xi[0] + 4*xi[1]), x10*x5, 27*x2*(-xi[0] - xi[1] + 1)])

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
                        return jnp.asarray([140*x0/3 + 64*x1*x3 + 70*x1/3 + 128*x2*xi[1]/3 - 80*x2/3 + 70*x3/3 + 128*x4*xi[0]/3 - 80*x4/3 - 80*x5 - 80*x6 + 32*xi[0]**4/3 - 25*xi[0]/3 + 32*xi[1]**4/3 - 25*xi[1]/3 + 1, xi[0]*(-48*x1 + 32*x2 + 22*xi[0] - 3)/3, xi[1]*(-48*x3 + 32*x4 + 22*xi[1] - 3)/3, x10*x11, x15*(x12 + x13*xi[1] - x13 - x14 + 16*x2 + 16*x5 + x9 + 19*xi[0]), x11*(14*x1 - x16 - x17*xi[1] - x18 - x7 - xi[1]), x19*(x17 - 6*xi[0] + 1), x21*(16*x0 - x15 - x20 + 1), x19*(x22 - 6*xi[1] + 1), x23*(-x12 - x18 - x22*xi[0] + 14*x3 - x8 - xi[0]), x20*(x16 + x24*xi[0] - x24 - x25 + 16*x4 + 16*x6 + x9 + 19*xi[1]), x10*x23, x26*(8*x0 - x12 + x14 - x16 + x25 + 3), x26*(-x25 - x27 + 5*xi[0] + xi[1]), x26*(-x14 - x27 + xi[0] + 5*xi[1])])

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
                        return jnp.asarray([1875*x0*x3/4 - 3125*x0*x4/12 + 375*x0/8 - 3125*x1*x3/12 + 625*x1*xi[1]/2 - 2125*x1/24 - 3125*x2*xi[1]/24 + 625*x2/8 + 375*x3/8 + 625*x4*xi[0]/2 - 2125*x4/24 - 3125*x5*xi[0]/24 + 625*x5/8 - 2125*x6/8 - 2125*x7/8 - 625*xi[0]**5/24 + 375*xi[0]*xi[1]/4 - 137*xi[0]/12 - 625*xi[1]**5/24 - 137*xi[1]/12 + 1, xi[0]*(875*x0 - 1250*x1 + 625*x2 - x8 + 24)/24, xi[1]*(875*x3 - 1250*x4 + 625*x5 - x9 + 24)/24, x16*x17, x25*(675*x0*xi[1] - 295*x0 + 325*x1 - x11 - 375*x14 - x19 - x20 + x21 + x22 - x23 - x24 + 107*xi[0]), x25*(245*x0 + x1*x9 - 300*x1 + x11 + x26 - x27 + x28 + 10*x3 - 78*xi[0] - 18*xi[1]), x17*(150*x0*xi[1] - 205*x0 + 275*x1 - x11 + x29 - x31 - x34 + 61*xi[0]), x35*(-150*x0 + x30 + x32 - 6), x38*(-x36 + x37 + 125*x7 + 15*xi[0] + 10*xi[1]), x38*(x37 - x39 + 125*x6 + 10*xi[0] + 15*xi[1]), x35*(x18 - 150*x3 + 55*xi[1] - 6), x41*(-x12 - x19 + 150*x3*xi[0] - 205*x3 - x34 + 275*x4 + x40 + 61*xi[1]), x43*(10*x0 + x12 + x20 + x28 + 245*x3 + x4*x8 - 300*x4 - x42 - 18*xi[0] - 78*xi[1]), x43*(-x12 - 375*x13 - x24 - x26 + 675*x3*xi[0] - 295*x3 - x31 + 325*x4 + x44 + x45 - x46 + 107*xi[1]), x16*x41, x47*(-x21 - x22 + x23 - x27 - x42 - x44 - x45 + x46 + 120*xi[0]*xi[1] + 12), x47*(40*x0 - x45 - x48 - x49 - 17*xi[0] - 2*xi[1]), x47*(-x22 + 40*x3 - x49 - x50 - 2*xi[0] - 17*xi[1]), x54*(x45 + x51*xi[1] - x51 + x52 + x53 + 29*xi[0] + 9*xi[1]), x54*(-x29 - x40 - x52 - x55 + 35*xi[0]*xi[1] + 1), x54*(x22 + x53 + x55 + x56*xi[0] - x56 + 9*xi[0] + 29*xi[1])])

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
                        return jnp.asarray([812*x0/5 + x1*x17 + 406*x1/5 + 1260*x10 - 1134*x11 - 1323*x12/2 + 1260*x13 - 1134*x14 + 1890*x15 - 2268*x16 - 2268*x18 + x19*x5 + 1296*x2*x6 - 441*x2/2 + 315*x3 + 1944*x4*xi[1]/5 - 1134*x4/5 + 406*x5/5 - 441*x6/2 + 315*x7 + 1944*x8*xi[0]/5 - 1134*x8/5 - 1323*x9/2 + 324*xi[0]**6/5 - 147*xi[0]/10 + 324*xi[1]**6/5 - 147*xi[1]/10 + 1, xi[0]*(-675*x1 + 1530*x2 - 1620*x3 + 648*x4 + 137*xi[0] - 10)/10, xi[1]*(-675*x5 + 1530*x6 - 1620*x7 + 648*x8 + 137*xi[1] - 10)/10, x23*x24, x30*(-461*x1 - 792*x10 + 216*x11 + 1752*x12 - 2088*x13 + 864*x14 + 864*x16 + 1296*x18 + 822*x2 + x22 + x25 - x26 - x27 + x28 + x29 - 684*x3 + 1038*x9 + 117*xi[0]), 4*xi[0]*(558*x1 - 1530*x12 - 324*x16 - 972*x18 - x19*xi[1] + 2106*x2*xi[1] - 1089*x2 + 972*x3 + x31 - x32 - x33 - x34 - 324*x4 + 162*x6*xi[0] - 459*x9 - 127*xi[0]), x30*(-307*x1 + 528*x12 - 828*x13 + 432*x14 + 216*x18 + 642*x2 + x25 - 612*x3 - x35 + x36 + x37*x5 + x37 + x39), x24*(130*x1 + 180*x2*xi[1] - 285*x2 - x20 + 288*x3 - x40 - x41*xi[1] - x42*xi[1] - x43 - 2*xi[1]), x44*(-180*x2 + x41 + x42 - 25*xi[0] + 2), x51*(36*x1 - 216*x12 + x46 + x47 + x50 - 6*xi[1]), 4*x0*(81*x0 + 18*x1 - 162*x12 + 18*x5 + x52 - 162*x9 - 9*xi[0] - 9*xi[1] + 1), x51*(x47 + 36*x5 + x53 + x56 - 216*x9 - 6*xi[0]), x44*(x57 + x58 - 180*x6 - 25*xi[1] + 2), x59*(-x21 - x43 + 130*x5 - x57*xi[0] - x58*xi[0] + 180*x6*xi[0] - 285*x6 + 288*x7 - 2*xi[0] - 27*xi[1]), x63*(x1*x61 - 828*x10 + 432*x11 + 216*x16 + x39 + x45 - 307*x5 + 642*x6 - x60 + x61 + x62 - 612*x7 + 528*x9), 4*xi[1]*(-459*x12 - 972*x16 - x17*xi[0] - 324*x18 + 162*x2*xi[1] - x34 + 558*x5 + 2106*x6*xi[0] - 1089*x6 + x64 - x65 - x66 + 972*x7 - 324*x8 - 1530*x9 - 127*xi[1]), x63*(-2088*x10 + 864*x11 + 1038*x12 - 792*x13 + 216*x14 + 1296*x16 + 864*x18 + x22 - 461*x5 + 822*x6 + x62 - x67 - x68 + x69 - 684*x7 + x70 + 1752*x9 + 117*xi[1]), x23*x59, x73*(238*x0 + 144*x10 + 144*x13 + x26 + x27 - x28 - x29 + x38 + x67 + x68 - x69 - x70 + x71 + x72), x73*(36*x1*xi[1] - 47*x1 + 72*x2 - x48*xi[1] - x68 - x74 + 12*xi[0] + xi[1]), x73*(-x27 + 36*x5*xi[0] - 47*x5 - x54*xi[0] + 72*x6 - x74 + xi[0] + 12*xi[1]), x77*(594*x1*xi[1] - 267*x1 - 324*x13 + 288*x2 - x31 + x33 - x42 - x72 - x75 - x76 + 97*xi[0]), x77*(195*x1 - 306*x12 - 252*x2 + x42 + x49 + x53 + x78 + x80 - 56*xi[0]), x77*(180*x1*xi[1] - 27*x1 - x78 - x81 - x82 + 10*xi[0] + 7*xi[1]), x77*(180*x5*xi[0] - 27*x5 - x75 - x82 - x83 + 7*xi[0] + 10*xi[1]), x77*(x46 + 195*x5 + x55 + x58 - 252*x6 + x80 + x83 - 306*x9 - 56*xi[1]), x77*(-324*x10 + 594*x5*xi[0] - 267*x5 - x58 + 288*x6 - x64 + x66 - x71 - x76 - x81 + 97*xi[1] - 10), x40*xi[1]*(324*x0 + 72*x1 - 504*x12 + 432*x15 + 72*x5 + x50 + x56 - 504*x9 - 41*xi[0] - 41*xi[1] + 5)])

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
                        return jnp.asarray([16807*x0*x5/3 + 117649*x0*x7/12 - 823543*x0*x8/240 + 22981*x0/180 + 117649*x1*x6/9 - 823543*x1*x7/144 + 33614*x1*xi[1]/9 - 331681*x1/720 - 331681*x10/240 - 386561*x11/72 - 331681*x12/240 - 386561*x13/72 - 386561*x14/36 - 386561*x15/36 + 117649*x2*x5/12 - 823543*x2*x6/144 + 16807*x2/18 - 823543*x3*x5/240 + 117649*x3*xi[1]/30 - 386561*x3/360 - 823543*x4*xi[1]/720 + 117649*x4/180 + 22981*x5/180 + 33614*x6*xi[0]/9 - 331681*x6/720 + 16807*x7/18 + 117649*x8*xi[0]/30 - 386561*x8/360 - 823543*x9*xi[0]/720 + 117649*x9/180 - 117649*xi[0]**7/720 + 22981*xi[0]*xi[1]/90 - 363*xi[0]/20 - 117649*xi[1]**7/720 - 363*xi[1]/20 + 1, xi[0]*(79576*x0 - 252105*x1 + 420175*x2 - 352947*x3 + 117649*x4 - 12348*xi[0] + 720)/720, xi[1]*(79576*x5 - 252105*x6 + 420175*x7 - 352947*x8 + 117649*x9 - 12348*xi[1] + 720)/720, x27*x28, x37*(264110*x0*x6 + 151165*x0*xi[1] - 27503*x0 + 384160*x1*x5 + 69580*x1 - x17 - 118335*x19 + 252105*x2*xi[1] - 92610*x2 - 286405*x21 - 84035*x22 - 84035*x24 - 168070*x26 + 62426*x3 - x30 + x31 + x32 + x33 - x34 - x35 - x36 + 93590*x5*xi[0] + 72030*x7*xi[0] + 5274*xi[0]), x41*(21784*x0 - 59731*x1 - 32781*x10 - 90356*x12 - 187278*x13 - 81634*x14 - 201684*x15 + x17 + 25382*x19 + 84721*x2 + 193452*x21 + 67228*x22 + 100842*x26 - 60025*x3 + x38 - x39 + x40 + 2506*x5 - 2156*x6 + 686*x7 - 3796*xi[0] - 1276*xi[1]), x41*(14406*x0*x6 + 51744*x0*xi[1] - 17815*x0 + 86436*x1*x5 + 51744*x1 - x17 - 3773*x19 + 129654*x2*xi[1] - 77518*x2 - 122108*x21 - 50421*x22 - 50421*x26 + 57624*x3 - x42 + 10584*x5*xi[0] - 756*x5 + 294*x6 + 2952*xi[0] + 642*xi[1]), x37*(15008*x0 - 45325*x1 - 2450*x10 - 27195*x12 - 79233*x13 + x17 + 71001*x2 + 68600*x21 + 33614*x22 - 55223*x3 - x43 + x44 + x45 + 168*x5 - 2412*xi[0] - 312*xi[1]), x28*(11025*x0*xi[1] - 12943*x0 + 40180*x1 - x17 + 36015*x2*xi[1] - 65170*x2 + 52822*x3 - x47 - x48*xi[1] - x50 + 2038*xi[0] + 120*xi[1]), x51*(-11025*x0 - 36015*x2 + x46 + x48 + x49 - 120), x57*(3430*x1 + 16807*x13 - x52 - x53 + x54 + x55 + x56 + 350*xi[0] + 168*xi[1]), x60*(-588*x0 + 686*x1 + 3773*x10 + 6174*x12 + 16807*x15 - 7203*x21 - 294*x5 + x59 + 154*xi[0] + 126*xi[1]), x60*(-294*x0 + 6174*x10 + 3773*x12 + 16807*x14 - 7203*x19 - 588*x5 + x59 + 686*x6 + 126*xi[0] + 154*xi[1]), x57*(16807*x11 + x56 + 3430*x6 - x61 - x62 + x63 + x64 + 168*xi[0] + 350*xi[1]), x51*(x29 - 11025*x5 + x65 - 36015*x7 + 1918*xi[1] - 120), x66*(-x18 - x30 + 11025*x5*xi[0] - 12943*x5 - x50 + 40180*x6 - x65*xi[0] + 36015*x7*xi[0] - 65170*x7 + 52822*x8 + 120*xi[0] + 2038*xi[1]), x68*(168*x0 - 27195*x10 - 79233*x11 - 2450*x12 + x18 + 68600*x19 + 33614*x20 + x38 + x45 + 15008*x5 - 45325*x6 - x67 + 71001*x7 - 55223*x8 - 312*xi[0] - 2412*xi[1]), x69*(86436*x0*x6 + 10584*x0*xi[1] - 756*x0 + 14406*x1*x5 + 294*x1 - x18 - 122108*x19 - 50421*x20 - 3773*x21 - 50421*x24 - x42 + 51744*x5*xi[0] - 17815*x5 + 51744*x6 + 129654*x7*xi[0] - 77518*x7 + 57624*x8 + 642*xi[0] + 2952*xi[1]), x69*(2506*x0 - 2156*x1 - 90356*x10 - 187278*x11 - 32781*x12 - 201684*x14 - 81634*x15 + x18 + 193452*x19 + 686*x2 + 67228*x20 + 25382*x21 + 100842*x24 + x40 + x44 + 21784*x5 - 59731*x6 + 84721*x7 - x70 - 60025*x8 - 1276*xi[0] - 3796*xi[1]), x68*(384160*x0*x6 + 93590*x0*xi[1] + 264110*x1*x5 - x18 - 286405*x19 + 72030*x2*xi[1] - 84035*x20 - 118335*x21 - 168070*x24 - 84035*x26 - x36 - x47 + 151165*x5*xi[0] - 27503*x5 + 69580*x6 + 252105*x7*xi[0] - 92610*x7 + x71 + x72 + x73 - x74 - x75 + 62426*x8 + 5274*xi[1]), x27*x66, x76*(51450*x0*x5 + 34300*x1*xi[1] - 36015*x10 - 12005*x11 - 36015*x12 - 12005*x13 - x31 - x32 - x33 + x34 + x35 - x43 + 34300*x6*xi[0] - x67 - x71 - x72 - x73 + x74 + x75 + 16450*xi[0]*xi[1] + 360), x76*(2065*x0 + 3430*x1*xi[1] - 5145*x1 + 5831*x2 - x53*xi[1] - x73 - x77 - x78 - 374*xi[0] - 24*xi[1]), x76*(-x33 + 2065*x5 + 3430*x6*xi[0] - 5145*x6 - x62*xi[0] + 5831*x7 - x78 - x79 - 24*xi[0] - 374*xi[1]), x82*(-5719*x0 + 9849*x1 + 20776*x12 + 9604*x13 + 9604*x14 + 14406*x15 - 8918*x19 - 7889*x2 - 1253*x5 + x54 + 1078*x6 + x64 + x73 + x80 + x81 + 1478*xi[0] + 638*xi[1]), x87*(3969*x0 + 15435*x1*xi[1] - 7987*x1 - 2940*x10 - 10829*x12 + 7203*x2 + 252*x5 - x70 - x73 - x83 - x85 - x86 - 844*xi[0] - 214*xi[1]), x82*(-2807*x0 + 6419*x1 + 4900*x12 + 4802*x13 - 6517*x2 - 8575*x21 + x73 + x89 + x91 + 540*xi[0] + 78*xi[1]), x82*(371*x0 + 4802*x1*xi[1] - 637*x1 - 2891*x12 - x89 - x92 - x93 - 83*xi[0] - 48*xi[1]), x87*(4459*x0*x5 + 140*x0 - 1568*x10 - 1568*x12 + 140*x5 - x85 - x94 + 525*xi[0]*xi[1] - 46*xi[0] - 46*xi[1] + 4), x82*(-2891*x10 + 371*x5 + 4802*x6*xi[0] - 637*x6 - x80 - x93 - x95 - 48*xi[0] - 83*xi[1]), x82*(4900*x10 + 4802*x11 - 8575*x19 + x33 - 2807*x5 + 6419*x6 - 6517*x7 + x91 + x95 + 78*xi[0] + 540*xi[1]), x87*(252*x0 - 10829*x10 - 2940*x12 - x33 - x39 + 3969*x5 + 15435*x6*xi[0] - 7987*x6 + 7203*x7 - x86 - x94 - x96 - 214*xi[0] - 844*xi[1]), x82*(-1253*x0 + 1078*x1 + 20776*x10 + 9604*x11 + 14406*x14 + 9604*x15 - 8918*x21 + x33 - 5719*x5 + x55 + 9849*x6 + x63 - 7889*x7 + x81 + x92 + 638*xi[0] + 1478*xi[1]), x97*(875*x0 + 7546*x1*xi[1] - 931*x1 - 8036*x10 - 8036*x12 + 875*x5 - x58 + 7546*x6*xi[0] - 931*x6 - x80 - x83 - x92 - x96 + 3220*xi[0]*xi[1] - 317*xi[0] - 317*xi[1] + 30), x97*(-581*x0 + 784*x1 + 2254*x10 + 4998*x12 + 4802*x15 - 6174*x21 - 196*x5 + x85 + x92 + x98 + 152*xi[0] + 110*xi[1]), x97*(-196*x0 + 4998*x10 + 2254*x12 + 4802*x14 - 6174*x19 - 581*x5 + 784*x6 + x80 + x94 + x98 + 110*xi[0] + 152*xi[1])])

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
                        return jnp.asarray([118124*x0/315 + 524288*x1*x11/45 + 59062*x1/315 + 1048576*x10*x2/45 - 18432*x10/5 + 53248*x11/15 + 1048576*x12*xi[0]/315 - 65536*x12/35 - 12816*x13/5 + 136832*x14/15 - 18432*x15 + 106496*x16/5 - 65536*x17/5 - 12816*x18/5 + 136832*x19/15 - 4272*x2/5 - 18432*x20 + 106496*x21/5 - 65536*x22/5 + 68416*x23/5 - 36864*x24 + 53248*x25 - 196608*x26/5 - 36864*x27 + 212992*x28/3 + 262144*x3*x9/9 + 34208*x3/15 - x30 + 53248*x31 - x33 - 196608*x34/5 + 1048576*x4*x8/45 - 18432*x4/5 + 524288*x5*x7/45 + 53248*x5/15 + 1048576*x6*xi[1]/315 - 65536*x6/35 + 59062*x7/315 - 4272*x8/5 + 34208*x9/15 + 131072*xi[0]**8/315 - 761*xi[0]/35 + 131072*xi[1]**8/315 - 761*xi[1]/35 + 1, xi[0]*(-52528*x1 + 216608*x2 - 501760*x3 + 659456*x4 - 458752*x5 + 131072*x6 + 6534*xi[0] - 315)/315, xi[1]*(659456*x10 - 458752*x11 + 131072*x12 - 52528*x7 + 216608*x8 - 501760*x9 + 6534*xi[1] - 315)/315, x38*x39, x47*(-36706*x1 + 172472*x13 - 314560*x14 + 312320*x15 - 159744*x16 + 32768*x17 + 269704*x18 - 715840*x19 + 122312*x2 + 995840*x20 - 700416*x21 + 196608*x22 + 1080320*x24 - 737280*x25 + 196608*x26 + 1536000*x27 + 491520*x29 - 229120*x3 - 1413120*x31 + 655360*x32 + 491520*x34 + x37 + 244736*x4 + x40 - x41 - x42 - x43 + x44 + x45 + x46 - 139264*x5 + 5589*xi[0]), x54*(92160*x1*x9 + 14346*x1 + 6144*x10*xi[0] - 33360*x13 - 25600*x15 - 81976*x18 + 242400*x2*xi[1] - 51456*x2 - 367360*x20 - 81920*x22 - 188160*x24 - 16384*x26 - 416000*x27 - 81920*x29 + 430080*x3*x7 + 102240*x3 - 163840*x32 - 163840*x34 - x35 + 276480*x4*xi[1] - 114432*x4 + x48 + x49 + 67584*x5 - x50 - x51 - x52 - x53 + 41760*x8*xi[0] - 2003*xi[0]), 4*xi[0]*(-46624*x1 + 51664*x13 - 39680*x14 + 11264*x15 + 198176*x18 - 634880*x19 + 175888*x2 + 1038336*x20 - 835584*x21 + 204800*x24 + 803840*x27 - 366592*x3 + x30 - 933888*x31 + 262144*x32 + 393216*x34 + 428032*x4 + x55*xi[1] - x55 - x56 + x57 + 65536*x6 - 3012*x7 + 2496*x8 - 768*x9 + 6219*xi[0] + 1599*xi[1])/9, x54*(9782*x1 - 4488*x13 - 29128*x18 + 98560*x2*xi[1] - 38176*x2 - 171776*x20 - 49152*x22 - 8960*x24 - 80640*x27 + 104448*x3*x7 + 82592*x3 - 16384*x32 - 49152*x34 - x35 + 147456*x4*xi[1] - 100096*x4 + 63488*x5 + x58 - x59 - x60 - x61 + 1600*x8*xi[0] - 1269*xi[0]), x47*(-16830*x1 + 2192*x13 + 31384*x18 - 110400*x19 + 67272*x2 + 202240*x20 - 184320*x21 + 65536*x22 + 43520*x27 - 149760*x3 + 32768*x34 + 187392*x4 + x40 - 122880*x5 - x62 + x63 - 120*x7 + 2143*xi[0] + 225*xi[1]), x39*(7378*x1 + 23520*x2*xi[1] - 30016*x2 + 68320*x3 - x35 + 43008*x4*xi[1] - 87808*x4 + 59392*x5 - x64 - x65*xi[1] - x66*xi[1] - x67*xi[1] - x68 - 927*xi[0]), x69*(-23520*x2 - 43008*x4 + x65 + x66 + x67 - 882*xi[0] + 45), x76*(1800*x1 - 14400*x18 + 43520*x19 + 7680*x3 - x70 - x71 - x72 + x73 + x74 + x75 - 120*xi[1]), x78*(280*x1 - 1600*x13 - 3360*x18 + 7680*x19 - 640*x2 - 6144*x20 - 20480*x27 + 512*x3 + 16384*x31 + 96*x7 + x77 - 50*xi[0] - 36*xi[1]), 4*x0*(1936*x0 + 576*x1 - 8448*x13 + 11264*x14 - 8448*x18 + 11264*x19 - 768*x2 + 36864*x23 - 49152*x24 - 49152*x27 + 65536*x28 + 576*x7 - 768*x8 - 132*xi[0] - 132*xi[1] + 9)/9, x78*(96*x1 - 3360*x13 + 7680*x14 - 6144*x15 - 1600*x18 - 20480*x24 + 16384*x25 + 280*x7 + x77 - 640*x8 + 512*x9 - 36*xi[0] - 50*xi[1]), x76*(-14400*x13 + 43520*x14 + 1800*x7 + x75 - x79 - x80 + x81 + x82 + 7680*x9 - 120*xi[0] - 274*xi[1]), x69*(-43008*x10 - 23520*x8 + x83 + x84 + x85 - 882*xi[1] + 45), x87*(43008*x10*xi[0] - 87808*x10 + 59392*x11 - x36 - x68 + 7378*x7 + 23520*x8*xi[0] - 30016*x8 - x83*xi[0] - x84*xi[0] - x85*xi[0] - x86 + 68320*x9 - 927*xi[1]), x90*(-120*x1 + 187392*x10 - 122880*x11 + 31384*x13 - 110400*x14 + 202240*x15 - 184320*x16 + 65536*x17 + 2192*x18 + 43520*x24 + 32768*x26 + x63 - 16830*x7 + 67272*x8 + x88 - x89 - 149760*x9 + 225*xi[0] + 2143*xi[1]), x94*(104448*x1*x9 + 147456*x10*xi[0] - 100096*x10 + 63488*x11 - 29128*x13 - 171776*x15 - 49152*x17 - 4488*x18 + 1600*x2*xi[1] - 80640*x24 - 49152*x26 - 8960*x27 - 16384*x29 - x36 - x61 + 9782*x7 + 98560*x8*xi[0] - 38176*x8 + 82592*x9 + x91 - x92 - x93 - 1269*xi[1]), 4*xi[1]*(-3012*x1 + 428032*x10 + 65536*x12 + 198176*x13 - 634880*x14 + 1038336*x15 - 835584*x16 + 51664*x18 - 39680*x19 + 2496*x2 + 11264*x20 + 803840*x24 - 933888*x25 + 393216*x26 + 204800*x27 + 262144*x29 - 768*x3 + x33 + x57 - 46624*x7 + 175888*x8 - 366592*x9 + x95*xi[0] - x95 - x96 + 1599*xi[0] + 6219*xi[1])/9, x94*(430080*x1*x9 + 276480*x10*xi[0] - 114432*x10 - x100 - x101 + 67584*x11 - 81976*x13 - 367360*x15 - 81920*x17 - 33360*x18 + 41760*x2*xi[1] - 25600*x20 - 416000*x24 - 163840*x26 - 188160*x27 - 163840*x29 + 92160*x3*x7 - 81920*x32 - 16384*x34 - x36 + 6144*x4*xi[1] - x53 + 14346*x7 + 242400*x8*xi[0] - 51456*x8 + 102240*x9 + x97 + x98 - x99 - 2003*xi[1]), x90*(244736*x10 - x102 - x103 - x104 + x105 + x106 + x107 - 139264*x11 + 269704*x13 - 715840*x14 + 995840*x15 - 700416*x16 + 196608*x17 + 172472*x18 - 314560*x19 + 312320*x20 - 159744*x21 + 32768*x22 + 1536000*x24 - 1413120*x25 + 491520*x26 + 1080320*x27 + 655360*x29 - 737280*x31 + 491520*x32 + 196608*x34 + x37 - 36706*x7 + 122312*x8 + x88 - 229120*x9 + 5589*xi[1]), x38*x87, x110*(24308*x0 + x102 + x103 + x104 - x105 - x106 - x107 + x108 + x109 - 75240*x13 + 113920*x14 - 84480*x15 - 75240*x18 + 113920*x19 - 84480*x20 + 170880*x23 - 168960*x24 - 168960*x27 + 81920*x28 + x41 + x42 + x43 - x44 - x45 - x46 + x62 + x89 + 315), x110*(1800*x1*xi[1] - 2074*x1 - x104 - x111 - x112 + 7240*x2 + 7680*x3*xi[1] - 13120*x3 + 11776*x4 - x70*xi[1] + 289*xi[0] + 15*xi[1]), x110*(11776*x10 - x112 - x113 - x43 + 1800*x7*xi[0] - 2074*x7 - x79*xi[0] + 7240*x8 + 7680*x9*xi[0] - 13120*x9 + 15*xi[0] + 289*xi[1]), x117*(66560*x1*x8 + 41640*x1*xi[1] - 8014*x1 - x104 - x114 - x115 - x116 - 30400*x14 - 75840*x19 + 97280*x2*x7 + 19400*x2 - 20480*x21 - 20480*x25 + 64000*x3*xi[1] - 24640*x3 - 40960*x31 + 15872*x4 - x48 - x49 + x51 + x52 + 25080*x7*xi[0] + 1583*xi[0]), x122*(10760*x1 + x118 + x120 + x121 - 14544*x13 + 11008*x14 - 43648*x18 + 95232*x19 - 29936*x2 - 92160*x20 - 38912*x24 - 98304*x27 + 42368*x3 - 29696*x4 + 1004*x7 + x74 - 832*x8 + x96 - 1793*xi[0] - 533*xi[1]), x122*(21696*x1*xi[1] - 7496*x1 - x109 - x118 - x123 - x124 - x126 - 55168*x19 + 39936*x2*x7 + 23184*x2 - 36224*x3 + 27648*x4 - x58 + x60 + 3984*x7*xi[0] - x73 + 1143*xi[0]), x117*(2734*x1 + x104 + x127 + x128 + x129 + x131 - 5000*x18 + 14080*x19 - 9080*x2 + 15424*x3 - 12800*x4 - x64 - 395*xi[0]), x117*(2920*x1*xi[1] - 330*x1 - x128 - x132 - x133 - 8000*x19 + 920*x2 + 9728*x3*xi[1] - 1152*x3 + 53*xi[0] + 27*xi[1]), x122*(3024*x1*xi[1] - 236*x1 - x124 - x135 - x136 - 5632*x19 + 17408*x2*x7 + 448*x2 + 2032*x7*xi[0] - 132*x7 + 47*xi[0] + 39*xi[1]), x122*(17408*x1*x8 + 2032*x1*xi[1] - 132*x1 - x120 - x136 - x137 - 5632*x14 + 3024*x7*xi[0] - 236*x7 + 448*x8 + 39*xi[0] + 47*xi[1]), x117*(-x115 - x133 - x138 - 8000*x14 + 2920*x7*xi[0] - 330*x7 + 920*x8 + 9728*x9*xi[0] - 1152*x9 + 27*xi[0] + 53*xi[1]), x117*(-12800*x10 + x114 - 5000*x13 + x131 + x138 + x139 + 14080*x14 + x43 + 2734*x7 - 9080*x8 - x86 + 15424*x9 - 395*xi[1]), x122*(39936*x1*x8 + 3984*x1*xi[1] + 27648*x10 - x108 - x126 - x137 - 55168*x14 - x140 - x141 + 21696*x7*xi[0] - 7496*x7 + 23184*x8 - x81 - 36224*x9 - x91 + x93 + 1143*xi[1]), x122*(1004*x1 - 29696*x10 + x121 - 43648*x13 + x135 + 95232*x14 + x140 - 92160*x15 - 14544*x18 + 11008*x19 - 832*x2 - 98304*x24 - 38912*x27 + x56 + 10760*x7 - 29936*x8 + x82 + 42368*x9 - 533*xi[0] - 1793*xi[1]), x117*(97280*x1*x8 + 25080*x1*xi[1] + 15872*x10 + x100 + x101 - x116 - x127 - x132 - 75840*x14 - 20480*x16 - 30400*x19 + 66560*x2*x7 - 40960*x25 - 20480*x31 - x43 + 41640*x7*xi[0] - 8014*x7 + 19400*x8 + 64000*x9*xi[0] - 24640*x9 - x97 - x98 + 1583*xi[1]), x144*(17256*x0 + 5268*x1 - 60704*x13 + 91904*x14 + x142 + x143 - 63488*x15 - 60704*x18 + 91904*x19 - 8864*x2 - 63488*x20 + 169984*x23 - 180224*x24 + 65536*x25 - 180224*x27 + 98304*x28 + 6912*x3 + 65536*x31 + 5268*x7 - 8864*x8 + 6912*x9 - 1373*xi[0] - 1373*xi[1] + 105), x144*(2028*x1 - 6016*x13 + 2816*x14 + x142 + x146 - 19808*x18 + 47104*x19 - 5024*x2 - 47104*x20 - 12288*x24 - 57344*x27 + 5376*x3 + 32768*x31 + 384*x7 - 192*x8 - 353*xi[0] - 213*xi[1]), x144*(384*x1 - 19808*x13 + 47104*x14 + x143 + x146 - 47104*x15 - 6016*x18 + 2816*x19 - 192*x2 - 57344*x24 + 32768*x25 - 12288*x27 + 2028*x7 - 5024*x8 + 5376*x9 - 213*xi[0] - 353*xi[1]), x148*(31744*x1*x8 + 17504*x1*xi[1] - 1632*x1 - x120 - x123 - x129 - x134 - 9216*x14 - x147 - 33536*x19 + 55296*x2*x7 + 3376*x2 + 27648*x3*xi[1] + 1024*x4 + 9456*x7*xi[0] - 668*x7 + 704*x8 + 325*xi[0] + 241*xi[1]), x148*(1384*x0 + 412*x1 + x120 - 5616*x13 + x135 + 7424*x14 + x145 - 5616*x18 + 7424*x19 - 576*x2 + 21504*x23 - 24576*x24 - 24576*x27 + 412*x7 - 576*x8 - 99*xi[0] - 99*xi[1] + 7), x148*(55296*x1*x8 + 9456*x1*xi[1] - 668*x1 + 1024*x10 - x119 - x135 - x139 - 33536*x14 - x141 - x147 - 9216*x19 + 31744*x2*x7 + 704*x2 + 17504*x7*xi[0] - 1632*x7 + 3376*x8 + 27648*x9*xi[0] + 241*xi[0] + 325*xi[1])])

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
                        return jnp.asarray([4782969*x0*x11/32 - 43046721*x0*x12/1120 + 1869885*x0*x7/64 + 13286025*x0*x9/64 + 58635*x0/224 + 4782969*x1*x10/16 - 14348907*x1*x11/160 + 4428675*x1*x8/16 + 623295*x1*xi[1]/32 - 40707*x1/28 - 43046721*x10*x2/320 + 2657205*x10*xi[0]/32 - 6589431*x10/640 + 885735*x11/64 + 4782969*x12*xi[0]/112 - 5137263*x12/448 - 43046721*x13*xi[0]/4480 + 4782969*x13/896 - 122121*x14/28 - 6589431*x15/128 - 5137263*x16/64 - 122121*x17/28 - 6589431*x18/128 - 5137263*x19/64 + 13286025*x2*x7/64 + 23914845*x2*x9/64 + 623295*x2/128 - 6589431*x20/64 - 15411789*x21/64 - 6589431*x22/64 - 25686315*x23/64 - 25686315*x24/64 - 15411789*x25/64 + 4782969*x3*x8/16 - 43046721*x3*x9/320 + 2657205*x3*xi[1]/32 - 6589431*x3/640 + 4782969*x4*x7/32 - 14348907*x4*x8/160 + 885735*x4/64 - 43046721*x5*x7/1120 + 4782969*x5*xi[1]/112 - 5137263*x5/448 - 43046721*x6*xi[1]/4480 + 4782969*x6/896 + 58635*x7/224 + 623295*x8*xi[0]/32 - 40707*x8/28 + 623295*x9/128 - 4782969*xi[0]**9/4480 + 58635*xi[0]*xi[1]/112 - 7129*xi[0]/280 - 4782969*xi[1]**9/4480 - 7129*xi[1]/280 + 1, xi[0]*(1063116*x0 - 5450004*x1 + 16365321*x2 - x26 + 32240754*x4 - 19131876*x5 + 4782969*x6 - 109584*xi[0] + 4480)/4480, xi[1]*(32240754*x11 - 19131876*x12 + 4782969*x13 - x27 + 1063116*x7 - 5450004*x8 + 16365321*x9 - 109584*xi[1] + 4480)/4480, x44*x45, x58*(16120377*x0*x10 + 26229420*x0*x8 + 3500847*x0*xi[1] - 375066*x0 + 36639540*x1*x7 + 39267585*x1*x9 + 1568763*x1 + 2893401*x11*xi[0] + 51667875*x2*x8 + 23524830*x2*xi[1] - 3848229*x2 - x29 + 38440899*x3*x7 + 5745978*x3 - 5583249*x31 - 6521634*x32 - 12123027*x34 - 25994682*x35 - 3720087*x36 - 28474740*x38 - 3720087*x39 + 15293691*x4*xi[1] - 5143824*x4 - 52816050*x41 - 11160261*x43 - x47 + x48 + x49 + 2539107*x5 + x50 + x51 - x52 - x53 - x54 - 11160261*x55 - 18600435*x56 - x57 + 2295405*x7*xi[0] + 7909650*x9*xi[0] + 46952*xi[0]), x62*(870828*x0 - 3900636*x1 - 170586*x10 + 39366*x11 - 2843424*x14 - 4953555*x15 - 6459750*x17 - 51799095*x18 - 37732311*x19 + 10113417*x2 - 29622915*x20 - 10097379*x21 - 60853275*x22 - 42515280*x23 - 79716150*x24 - 77058945*x25 - 15785766*x3 + 5053185*x31 + 2539107*x32 + 24672519*x34 + 60958251*x35 + 9565938*x36 + 24406920*x38 + 1594323*x39 + 14644152*x4 + 97430850*x41 + 23914845*x43 - 7440174*x5 + 9565938*x55 + 31886460*x56 + x59 - x60 + x61 + 147444*x7 - 284310*x8 + 303750*x9 - 100624*xi[0] - 40144*xi[1]), x70*(354294*x0*x10 + 3287790*x0*x8 + 1345716*x0*xi[1] - 234684*x0 + 10515825*x1*x7 + 3838185*x1*x9 + 1100241*x1 + 11809800*x2*x8 + 12356550*x2*xi[1] - 2978289*x2 - x29 + 15943230*x3*x7 + 4830354*x3 - 500175*x31 - 5509539*x34 - 15418350*x35 - 2657205*x36 - 1738665*x38 + 10038330*x4*xi[1] - 4632066*x4 - 18534825*x41 - 5314410*x43 + 2421009*x5 - 531441*x55 - 5314410*x56 + x63 + x64 + x65 - x66 - x67 - x68*xi[0] - x69 + 407745*x7*xi[0] + 302535*x9*xi[0] + 25996*xi[0]), x70*(196380*x0 - 949140*x1 - 170190*x14 - 36450*x15 - 840690*x17 - 8529300*x18 - 7676370*x19 + 2654613*x2 - 911250*x20 - 4957200*x22 - 590490*x23 - 4133430*x24 - 8857350*x25 + x29 - 4446900*x3 + 129276*x31 + 3608226*x34 + 11219310*x35 + 2125764*x36 + 229635*x38 + 4395870*x4 + 9480645*x41 + 3188646*x43 - 2361960*x5 + 2125764*x56 + 8040*x7 + x71 - 6480*x8 + 1944*x9 - 21200*xi[0] - 4400*xi[1]), x62*(492075*x0*x8 + 1525149*x0*xi[1] - 505842*x0 + 5937705*x1*x7 + 2497797*x1 + 2657205*x2*x8 + 16664940*x2*xi[1] - 7160967*x2 + 12223143*x3*x7 + 12321558*x3 - 66582*x31 - 6769251*x34 - 22950378*x35 - 4782969*x36 + 16474671*x4*xi[1] - 12518388*x4 - 12105045*x41 - 4782969*x43 + 6908733*x5 - 1594323*x56 - x59 + 187272*x7*xi[0] - 8640*x7 - x72 + 3240*x8 + 53672*xi[0] + 7640*xi[1]), x58*(147636*x0 - 740628*x1 - 15876*x14 - 280224*x17 - 3240405*x18 - 3483891*x19 + 2164239*x2 - 535815*x22 + x29 - 3806838*x3 + 1275183*x34 + 4638627*x35 + 1062882*x36 + 3962844*x4 + 1148175*x41 + 531441*x43 - 2243862*x5 + 720*x7 - x73 + x74 - 15472*xi[0] - 1360*xi[1]), x45*(118188*x0*xi[1] - 131256*x0 + 666477*x1 + 1428840*x2*xi[1] - 1977129*x2 - x29 + 3541482*x3 + 1653372*x4*xi[1] - 3766014*x4 + 2184813*x5 - x76 - x77*xi[1] - x78*xi[1] - x80 + 13628*xi[0] + 560*xi[1]), x81*(-118188*x0 - 1428840*x2 - 1653372*x4 + x75 + x77 + x78 + x79 - 560), x87*(59535*x1 + 131544*x17 + 1148175*x18 + 137781*x3 - 535815*x34 - 1240029*x35 - x82 - x83 - x84 + x85 + x86 + 1764*xi[0] + 720*xi[1]), x89*(-12150*x0 + 41310*x1 + 66582*x14 + 164025*x17 + 885735*x18 - 65610*x2 + 1673055*x22 + 1594323*x25 + 39366*x3 - 557685*x34 - 531441*x35 - 2657205*x41 - 3240*x7 + x88 + 1644*xi[0] + 1080*xi[1]), x93*(-1890*x0 + 4860*x1 + 24300*x14 + 31185*x17 + 72171*x18 - 4374*x2 + 229635*x20 + 393660*x22 - 36450*x31 - 80190*x34 - 354294*x41 + 1944*x8 + x90 + x91 + x92 + 300*xi[0] + 264*xi[1]), x93*(1944*x1 + 31185*x14 + 72171*x15 + 24300*x17 + 393660*x20 + 229635*x22 - 80190*x31 - 36450*x34 - 354294*x38 - 1890*x7 + 4860*x8 - 4374*x9 + x92 + x94 + x95 + 264*xi[0] + 300*xi[1]), x89*(-3240*x0 + 39366*x10 + 164025*x14 + 885735*x15 + 66582*x17 + 1673055*x20 + 1594323*x21 - 557685*x31 - 531441*x32 - 2657205*x38 - 12150*x7 + 41310*x8 + x88 - 65610*x9 + 1080*xi[0] + 1644*xi[1]), x87*(137781*x10 + 131544*x14 + 1148175*x15 - 535815*x31 - 1240029*x32 + x60 + 59535*x8 + x86 - x96 - x97 - x98 + 720*xi[0] + 1764*xi[1]), x81*(x100 - 1653372*x11 + x46 - 118188*x7 - 1428840*x9 + x99 + 13068*xi[1] - 560), x101*(3541482*x10 - x100*xi[0] + 1653372*x11*xi[0] - 3766014*x11 + 2184813*x12 - x30 - x47 + 118188*x7*xi[0] - 131256*x7 + 666477*x8 - x80 + 1428840*x9*xi[0] - 1977129*x9 - x99*xi[0] + 560*xi[0] + 13628*xi[1]), x103*(720*x0 - 3806838*x10 - x102 + 3962844*x11 - 2243862*x12 - 280224*x14 - 3240405*x15 - 3483891*x16 - 15876*x17 - 535815*x20 + x30 + 1275183*x31 + 4638627*x32 + 1062882*x33 + 1148175*x38 + 531441*x39 + 147636*x7 + x74 - 740628*x8 + 2164239*x9 - 1360*xi[0] - 15472*xi[1]), x105*(12223143*x0*x10 + 5937705*x0*x8 + 187272*x0*xi[1] - 8640*x0 + 492075*x1*x7 + 2657205*x1*x9 + 3240*x1 + 12321558*x10 - x104 + 16474671*x11*xi[0] - 12518388*x11 + 6908733*x12 - 6769251*x31 - 22950378*x32 - 4782969*x33 - 66582*x34 - 12105045*x38 - 4782969*x39 - 1594323*x55 + 1525149*x7*xi[0] - 505842*x7 - x72 + 2497797*x8 + 16664940*x9*xi[0] - 7160967*x9 + 7640*xi[0] + 53672*xi[1]), x106*(8040*x0 - 6480*x1 - 4446900*x10 + 4395870*x11 - 2361960*x12 - 840690*x14 - 8529300*x15 - 7676370*x16 - 170190*x17 - 36450*x18 + 1944*x2 - 4957200*x20 - 8857350*x21 - 911250*x22 - 4133430*x23 - 590490*x24 + x30 + 3608226*x31 + 11219310*x32 + 2125764*x33 + 129276*x34 + 9480645*x38 + 3188646*x39 + 229635*x41 + 2125764*x55 + 196380*x7 + x71 - 949140*x8 + 2654613*x9 - 4400*xi[0] - 21200*xi[1]), x106*(15943230*x0*x10 + 10515825*x0*x8 + 407745*x0*xi[1] + 3287790*x1*x7 + 11809800*x1*x9 + 4830354*x10 + x107 + x108 + x109 + 10038330*x11*xi[0] - 4632066*x11 - x110 - x111 - x112*xi[1] + 2421009*x12 + 3838185*x2*x8 + 302535*x2*xi[1] + 354294*x3*x7 - x30 - 5509539*x31 - 15418350*x32 - 2657205*x33 - 500175*x34 - 18534825*x38 - 5314410*x39 - 1738665*x41 - 5314410*x55 - 531441*x56 - x69 + 1345716*x7*xi[0] - 234684*x7 + 1100241*x8 + 12356550*x9*xi[0] - 2978289*x9 + 25996*xi[1]), x105*(147444*x0 - 284310*x1 - 15785766*x10 + x104 + 14644152*x11 - 7440174*x12 - 6459750*x14 - 51799095*x15 - 37732311*x16 - 2843424*x17 - 4953555*x18 + 303750*x2 - 60853275*x20 - 77058945*x21 - 29622915*x22 - 79716150*x23 - 42515280*x24 - 10097379*x25 - 170586*x3 + 24672519*x31 + 60958251*x32 + 9565938*x33 + 5053185*x34 + 2539107*x35 + 97430850*x38 + 23914845*x39 + 39366*x4 + 24406920*x41 + 1594323*x43 + 31886460*x55 + 9565938*x56 + x61 + 870828*x7 - 3900636*x8 - x85 + 10113417*x9 - 40144*xi[0] - 100624*xi[1]), x103*(38440899*x0*x10 + 36639540*x0*x8 + 2295405*x0*xi[1] + 26229420*x1*x7 + 51667875*x1*x9 + 5745978*x10 + 15293691*x11*xi[0] - 5143824*x11 + x113 + x114 + x115 + x116 - x117 - x118 - x119 + 2539107*x12 + 39267585*x2*x8 + 7909650*x2*xi[1] + 16120377*x3*x7 - x30 - 12123027*x31 - 25994682*x32 - 3720087*x33 - 5583249*x34 - 6521634*x35 - 52816050*x38 - 11160261*x39 + 2893401*x4*xi[1] - 28474740*x41 - 3720087*x43 - 18600435*x55 - 11160261*x56 - x57 + 3500847*x7*xi[0] - 375066*x7 - x76 + 1568763*x8 + 23524830*x9*xi[0] - 3848229*x9 + 46952*xi[1]), x101*x44, x122*(3470040*x0*x7 + 5511240*x1*x8 + 2313360*x1*xi[1] + 1653372*x10*xi[0] - x102 - x113 - x114 - x115 - x116 + x117 + x118 + x119 - x120 - x121 - 2704590*x15 - 413343*x16 - 2704590*x18 - 413343*x19 - 5409180*x20 - 5409180*x22 - 2066715*x23 - 2066715*x24 + 1653372*x3*xi[1] - x48 - x49 - x50 - x51 + x52 + x53 + x54 - x73 + 2313360*x8*xi[0] + 267876*xi[0]*xi[1]), x122*(16380*x0 + 59535*x1*xi[1] - 74151*x1 - x115 - x123 + 187110*x2 + 137781*x3*xi[1] - 265356*x3 + 196830*x4 - x82*xi[1] - x83*xi[1] - x84*xi[1] - 1844*xi[0] - 80*xi[1]), x122*(137781*x10*xi[0] - 265356*x10 + 196830*x11 - x123 - x50 + 16380*x7 + 59535*x8*xi[0] - 74151*x8 + 187110*x9 - x96*xi[0] - x97*xi[0] - x98*xi[0] - 80*xi[0] - 1844*xi[1]), x127*(-254370*x0 + 805653*x1 + 85293*x10 + x120 + x124 + x125 + x126 + 1793340*x15 + 1753461*x17 + 5893965*x18 + 1062882*x19 - 1431270*x2 + 6320430*x20 + 1062882*x21 + 9054180*x22 + 2657205*x23 + 3542940*x24 + 2657205*x25 + 1452168*x3 - 1886895*x31 - 885735*x32 - 4445685*x34 - 3956283*x35 - 787320*x4 - 7971615*x41 - 73722*x7 + 142155*x8 - 151875*x9 + 40232*xi[0] + 20072*xi[1]), x133*(984150*x0*x9 + 166776*x0 + 2744685*x1*xi[1] - 599913*x1 - x124 - x128 - x131 - x132 - 347895*x14 - 251505*x15 - 921618*x17 - 4122495*x18 - 885735*x19 + 4723920*x2*x7 + 1178550*x2 - 2022975*x20 - 4603635*x22 - 885735*x23 - 1771470*x25 + 3050865*x3*xi[1] - 1294704*x3 + 747954*x4 - x63 - x64 + x66 + x67 + 421605*x8*xi[0] - 22636*xi[0]), x138*(-38304*x0 + 151101*x1 + x115 + x134 + x135*xi[1] - x135 + x136 + x137 + 38025*x14 + 160290*x17 + 914166*x18 - 323676*x2 + 163296*x20 + 685989*x22 + 236196*x24 + 383454*x3 - 28674*x31 - 541404*x34 - 747954*x35 - 826686*x41 - 2010*x7 + 1620*x8 + 4796*xi[0] + 1100*xi[1]), x133*(83298*x0 + 908820*x1*xi[1] - 349623*x1 - x124 - x139 - 34344*x14 - x143 - x144 - 247239*x17 - 1697112*x18 + 1062882*x2*x7 + 802872*x2 - 754515*x22 + 1535274*x3*xi[1] - 1019142*x3 + 669222*x4 + 1728*x7 - x85 - 9928*xi[0] - 1528*xi[1]), x127*(-62934*x0 + 275913*x1 + x124 + x146 + x148 + 118071*x17 + 929475*x18 + 354294*x19 - 668250*x2 + 901044*x3 - 460485*x34 - 925101*x35 - 629856*x4 + 7256*xi[0] + 680*xi[1]), x127*(6897*x0 + 261225*x1*xi[1] - 26730*x1 - x146 - x149 - x150 - 68148*x17 - 513945*x18 + 53460*x2 + 492075*x3*xi[1] - 52488*x3 - 862*xi[0] - 400*xi[1]), x133*(2190*x0 + 95985*x1*xi[1] - 6750*x1 - 16848*x14 - x142 - x152 - x153 - 31455*x17 - 129033*x18 + 433026*x2*x7 + 9234*x2 - 338985*x22 + 864*x7 - 316*xi[0] - 232*xi[1]), x138*(36936*x0*x7 + 390*x0 + 137781*x1*x8 + 13851*x1*xi[1] - 810*x1 - x136 - 6759*x14 - x154 - 6759*x17 - 73629*x20 - 73629*x22 + 390*x7 + 13851*x8*xi[0] - 810*x8 + 1221*xi[0]*xi[1] - 70*xi[0] - 70*xi[1] + 4), x133*(433026*x0*x9 + 864*x0 - x131 - 31455*x14 - 129033*x15 - x153 - x156 - 16848*x17 - 338985*x20 + 2190*x7 + 95985*x8*xi[0] - 6750*x8 + 9234*x9 - 232*xi[0] - 316*xi[1]), x127*(492075*x10*xi[0] - 52488*x10 - x125 - 68148*x14 - 513945*x15 - x150 - x157 + 6897*x7 + 261225*x8*xi[0] - 26730*x8 + 53460*x9 - 400*xi[0] - 862*xi[1]), x127*(901044*x10 - 629856*x11 + 118071*x14 + x148 + 929475*x15 + x157 + x158 + 354294*x16 - 460485*x31 - 925101*x32 - 62934*x7 + 275913*x8 - 668250*x9 + 680*xi[0] + 7256*xi[1]), x133*(1062882*x0*x9 + 1728*x0 + 1535274*x10*xi[0] - 1019142*x10 + 669222*x11 - 247239*x14 - x144 - 1697112*x15 - x158 - x159 - x160 - 34344*x17 - 754515*x20 - x60 + 83298*x7 + 908820*x8*xi[0] - 349623*x8 + 802872*x9 - 1528*xi[0] - 9928*xi[1]), x138*(-2010*x0 + 1620*x1 + 383454*x10 + x137 + 160290*x14 + 914166*x15 + x154 + x161 + x162*xi[0] - x162 + 38025*x17 + 685989*x20 + 163296*x22 + 236196*x23 - 541404*x31 - 747954*x32 - 28674*x34 - 826686*x38 + x50 - 38304*x7 + 151101*x8 - 323676*x9 + 1100*xi[0] + 4796*xi[1]), x133*(4723920*x0*x9 + 421605*x1*xi[1] + 3050865*x10*xi[0] - 1294704*x10 - x107 - x108 + 747954*x11 + x110 + x111 - x132 - 921618*x14 - 4122495*x15 - x152 - x158 - 885735*x16 - x163 - 347895*x17 - 251505*x18 + 984150*x2*x7 - 4603635*x20 - 1771470*x21 - 2022975*x22 - 885735*x24 + 166776*x7 + 2744685*x8*xi[0] - 599913*x8 + 1178550*x9 - 22636*xi[1]), x127*(-73722*x0 + 142155*x1 + 1452168*x10 - 787320*x11 + x121 + x126 + 1753461*x14 + x149 + 5893965*x15 + x158 + 1062882*x16 + 1793340*x18 - 151875*x2 + 9054180*x20 + 2657205*x21 + 6320430*x22 + 3542940*x23 + 2657205*x24 + 1062882*x25 + 85293*x3 - 4445685*x31 - 3956283*x32 - 1886895*x34 - 885735*x35 - 7971615*x38 - 254370*x7 + 805653*x8 - 1431270*x9 + 20072*xi[0] + 40232*xi[1] - 2240), x164*(2374110*x0*x7 + 3050865*x0*x9 + 43797*x0 + 4527090*x1*x8 + 1313415*x1*xi[1] - 102870*x1 + 807003*x10*xi[0] - 78732*x10 - x125 - x128 - 613008*x14 - x149 - 1454355*x15 - x163 - 613008*x17 - 1454355*x18 + 3050865*x2*x7 + 126360*x2 - 3969405*x20 - 885735*x21 - 3969405*x22 - 885735*x25 + 807003*x3*xi[1] - 78732*x3 + 43797*x7 + 1313415*x8*xi[0] - 102870*x8 + 126360*x9 + 132954*xi[0]*xi[1] - 8798*xi[0] - 8798*xi[1] + 560), x164*(-10182*x0 + 36855*x1 + x134 + 25596*x14 + x143 + x149 + x151 + x165 + 110403*x17 + 686718*x18 - 67311*x2 + 546750*x22 - 390015*x34 - 570807*x35 - 747954*x41 + x90 + 1336*xi[0] + 712*xi[1]), x164*(x125 + x130 + 110403*x14 + 686718*x15 + x160 + x161 + x165 + 25596*x17 + 546750*x20 - 390015*x31 - 570807*x32 - 747954*x38 - 10182*x7 + 36855*x8 - 67311*x9 + x94 + 712*xi[0] + 1336*xi[1]), x168*(-25932*x0 + 73305*x1 + x112 + x131 + 201825*x14 + x149 + 220887*x15 + x166 + x167 + 337068*x17 + 1156923*x18 - 103761*x2 + 1441233*x20 + 2309472*x22 + 708588*x23 + 1062882*x24 + 708588*x25 - 311769*x31 - 890109*x34 - 728271*x35 - 846369*x38 - 11910*x7 + 19710*x8 - 15066*x9 + 4124*xi[0] + 3116*xi[1]), x168*(15807*x0 + 590976*x1*xi[1] - 51840*x1 - x139 - 72117*x14 - x149 - x169 - 187947*x17 - x170 - 901044*x18 + 1358127*x2*x7 + 84078*x2 - 400221*x20 - 1211598*x22 + 649539*x3*xi[1] - 65610*x3 + 3870*x7 + 71199*x8*xi[0] - 4050*x8 - x91 - 2230*xi[0] - 1390*xi[1]), x168*(-3750*x0 + 9990*x1 + 40851*x14 + x152 + x169 + 57321*x17 + x171 + 159651*x18 - 11178*x2 + 310554*x20 + 599238*x22 + 354294*x24 - 56376*x31 - 148959*x34 - 570807*x41 - 2250*x7 + 3240*x8 + 596*xi[0] + 500*xi[1]), x168*(-2250*x0 + 3240*x1 + x131 + 57321*x14 + 159651*x15 + 40851*x17 + x171 + x172 + 599238*x20 + 310554*x22 + 354294*x23 - 148959*x31 - 56376*x34 - 570807*x38 - 3750*x7 + 9990*x8 - 11178*x9 + 500*xi[0] + 596*xi[1]), x168*(1358127*x0*x9 + 3870*x0 + 71199*x1*xi[1] - 4050*x1 + 649539*x10*xi[0] - 65610*x10 - x125 - 187947*x14 - 901044*x15 - x159 - 72117*x17 - x170 - x172 - 1211598*x20 - 400221*x22 + 15807*x7 + 590976*x8*xi[0] - 51840*x8 + 84078*x9 - x95 - 1390*xi[0] - 2230*xi[1]), x168*(-11910*x0 + 19710*x1 + x125 + 337068*x14 + 1156923*x15 + x152 + x167 + 201825*x17 + x173 + 220887*x18 - 15066*x2 + 2309472*x20 + 708588*x21 + 1441233*x22 + 1062882*x23 + 708588*x24 - 890109*x31 - 728271*x32 - 311769*x34 - 846369*x41 + x68 - 25932*x7 + 73305*x8 - 103761*x9 + 3116*xi[0] + 4124*xi[1] - 224), 27*x28*(1614006*x0*x7 + 20250*x0 + 3897234*x1*x8 + 662661*x1*xi[1] - 42930*x1 + 177147*x10*xi[0] - 13122*x10 - x139 - 328617*x14 - 570807*x15 - x159 - x166 - 328617*x17 - x173 - 570807*x18 + 39366*x2 - 2899962*x20 - 2899962*x22 - 1594323*x23 - 1594323*x24 + 177147*x3*xi[1] - 13122*x3 + 20250*x7 + 662661*x8*xi[0] - 42930*x8 + 39366*x9 + 63180*xi[0]*xi[1] - 3788*xi[0] - 3788*xi[1])/8])

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
                        return jnp.asarray([177133*x0/252 + 7812500*x1*x15/63 + 177133*x1/504 + 62500000*x10*x6/189 - 10511875*x10/4536 + 15625000*x11*x5/27 + 42711625*x11/4536 + 6250000*x12*x4/9 - 5369375*x12/216 + 15625000*x13*x3/27 + 4695625*x13/108 + 62500000*x14*x2/189 - 9453125*x14/189 + 6875000*x15/189 + 15625000*x16*xi[0]/567 - 8593750*x16/567 - 10511875*x17/1512 + 42711625*x18/1134 - 26846875*x19/216 - 10511875*x2/4536 + 4695625*x20/18 - 9453125*x21/27 + 55000000*x22/189 - 8593750*x23/63 - 10511875*x24/1512 + 42711625*x25/1134 - 26846875*x26/216 + 4695625*x27/18 - 9453125*x28/27 + 55000000*x29/189 + 42711625*x3/4536 - 8593750*x30/63 + 42711625*x31/756 - 26846875*x32/108 + 23478125*x33/36 - 9453125*x34/9 + 27500000*x35/27 - 34375000*x36/63 - 26846875*x37/108 + 23478125*x38/27 - 47265625*x39/27 - 5369375*x4/216 + 55000000*x40/27 - 34375000*x41/27 + 23478125*x42/36 - 47265625*x43/27 + 68750000*x44/27 - 17187500*x45/9 - 9453125*x46/9 + 55000000*x47/27 - 17187500*x48/9 + 27500000*x49/27 + 4695625*x5/108 - 34375000*x50/27 - 34375000*x51/63 - 9453125*x6/189 + 7812500*x7*x9/63 + 6875000*x7/189 + 15625000*x8*xi[1]/567 - 8593750*x8/567 + 177133*x9/504 + 1562500*xi[0]**10/567 - 7381*xi[0]/252 + 1562500*xi[1]**10/567 - 7381*xi[1]/252 + 1, xi[0]*(-1465875*x1 + 9046000*x2 - 33665625*x3 + 79091250*x4 - 118125000*x5 + 108750000*x6 - 56250000*x7 + 12500000*x8 + 128322*xi[0] - 4536)/4536, xi[1]*(9046000*x10 - 33665625*x11 + 79091250*x12 - 118125000*x13 + 108750000*x14 - 56250000*x15 + 12500000*x16 - 1465875*x9 + 128322*xi[1] - 4536)/4536, x55*x56, x66*(-1043307*x1 + 7983480*x17 - 24617600*x18 + 46142250*x19 + 5295340*x2 - 53830000*x20 + 38150000*x21 - 15000000*x22 + 2500000*x23 + 11934750*x24 - 51499000*x25 + 129969000*x26 - 199430000*x27 + 183400000*x28 - 93000000*x29 - 16234925*x3 + 20000000*x30 + 148169000*x32 - 225575000*x33 + 201600000*x34 - 98000000*x35 + 20000000*x36 + 204053500*x37 + 481250000*x39 + 31582250*x4 - 287000000*x40 + 70000000*x41 - 407575000*x42 + 626500000*x43 + 140000000*x45 + 463050000*x46 - 469000000*x47 + 175000000*x48 - 280000000*x49 - 39305000*x5 + 140000000*x50 + 70000000*x51 + x54 + x57 - x58 - x59 + 30350000*x6 - x60 - x61 + x62 + x63 + x64 + x65 - 13250000*x7 + 110178*xi[0]), x76*(35262500*x1*x11 + 8750000*x1*x13 - x1*x74 + 400579*x1 + 135625000*x10*x4 + 4047400*x10*xi[0] + 44625000*x12*x2 + 4541250*x12*xi[0] + 375000*x14*xi[0] - 1726515*x17 - 5586875*x19 + 17488100*x2*xi[1] - 2168695*x2 - 2012500*x21 - 3702150*x24 - 47500250*x26 - 74637500*x28 + 126262500*x3*x9 + 7008225*x3 - 8750000*x30 - 29463000*x32 - 24237500*x34 - 57405250*x37 - 91875000*x39 + 77341250*x4*xi[1] - 14224875*x4 - 8750000*x41 - 164500000*x43 - 26250000*x45 - 154962500*x46 - 43750000*x48 + 99750000*x5*x9 + 18322500*x5 - 43750000*x50 - 26250000*x51 - x52 + 39375000*x6*xi[1] - 14550000*x6 + x67 + x68 + x69 + 6500000*x7 - x70 - x71 - x72 - x73 - x75 - 39246*xi[0]), x78*(-645201*x1 + 122625*x10 - 125250*x11 + 67500*x12 - 15000*x13 + 1592285*x17 - 2749125*x18 + 2633750*x19 + 3642935*x2 - 1327500*x20 + 275000*x21 + 4734595*x24 - 23799075*x25 + 68572000*x26 - 117690000*x27 + 118900000*x28 - 65250000*x29 - 12248475*x3 + 15000000*x30 + 22122500*x32 - 18937500*x33 + 8400000*x34 - 1500000*x35 + 62371000*x37 + 58750000*x39 + 25757250*x4 - 20250000*x40 + 2500000*x41 - 149062500*x42 + 156500000*x43 + 15000000*x45 + 196875000*x46 - 142500000*x47 + 37500000*x48 - 135000000*x49 - 34215000*x5 + 50000000*x50 + 37500000*x51 + x57 + 27900000*x6 - 12750000*x7 + x77 - 66786*x9 + 60759*xi[0] + 19179*xi[1]), xi[0]*(10312500*x1*x11 + 1346625*x1 + 162500000*x10*x4 + 2224375*x10*xi[0] - 95250*x10 + 60000*x11 + 312500*x12*xi[0] - 1848250*x17 - 1325000*x19 + 40270625*x2*xi[1] - 7818625*x2 - 7681625*x24 - 121449375*x26 - 230000000*x28 + 205937500*x3*x9 + 27074375*x3 - 19040625*x32 - 2187500*x34 - 80571875*x37 - 35937500*x39 + 218125000*x4*xi[1] - 58608125*x4 - 161562500*x43 - 6250000*x45 - 290937500*x46 - 31250000*x48 + 212500000*x5*x9 + 80000000*x5 - 62500000*x50 - 62500000*x51 + 131250000*x6*xi[1] - 66875000*x6 - 31250000*x7*xi[1] + 31250000*x7 - x79 - 6250000*x8 + x80 - x81 + 75000*x9 - 123786*xi[0] - 29286*xi[1])/9, x78*(-461789*x1 + 325835*x17 - 244900*x18 + 68500*x19 + 2734310*x2 + 1978945*x24 - 10689200*x25 + 33381500*x26 - 62285000*x27 + 68300000*x28 - 40500000*x29 - 9680025*x3 + 10000000*x30 + 2186500*x32 - 562500*x33 + 15307250*x37 + 2125000*x39 + 21452250*x4 - 41212500*x42 + 21250000*x43 + 61775000*x46 - 23500000*x47 + 2500000*x48 - 48000000*x49 - 29985000*x5 + 10000000*x50 + 15000000*x51 + x57 + 25650000*x6 - 12250000*x7 - x82 - x83 + x84 + x85 + x87 + 41766*xi[0]), x76*(201951*x1 + 2625000*x10*x4 + 22050*x10*xi[0] - 900*x10 - 62235*x17 + 3385725*x2*xi[1] - 1213195*x2 - 613030*x24 - 10864000*x26 - 23762500*x28 + 8662500*x3*x9 + 4368525*x3 - 3750000*x30 - 203000*x32 - 3089625*x37 + 20921250*x4*xi[1] - 9867375*x4 - 2187500*x43 - 13650000*x46 + 11250000*x5*x9 + 14077500*x5 - 1250000*x50 - 3750000*x51 - x52 + 14625000*x6*xi[1] - 12300000*x6 + 6000000*x7 - x88 + 2430*x9 - 18054*xi[0] - 2178*xi[1]), x66*(-358803*x1 + 32670*x17 + 2179465*x2 + 689110*x24 - 3871875*x25 + 12694500*x26 - 25095000*x27 + 29400000*x28 - 18750000*x29 - 7953575*x3 + 5000000*x30 + 1692250*x37 + 18247250*x4 - 4900000*x42 + 8050000*x46 - 26495000*x5 + 2500000*x51 + x57 + 23600000*x6 - 11750000*x7 - x89 - 1260*x9 + x90 + 31797*xi[0] + 2394*xi[1]), x56*(161353*x1 + 841050*x2*xi[1] - 988705*x2 + 3647175*x3 + 5670000*x4*xi[1] - 8476125*x4 + 12495000*x5 - x52 + 4500000*x6*xi[1] - 11325000*x6 + 5750000*x7 - x91*xi[1] - x92*xi[1] - x93*xi[1] - x94*xi[1] - x95 - 14202*xi[0] - 504*xi[1]), x96*(-841050*x2 - 5670000*x4 - 4500000*x6 + x91 + x92 + x93 + x94 - 13698*xi[0] + 504), x103*(32830*x1 + x100 + x102 - 328300*x24 + 1692250*x25 - 4900000*x26 + 8050000*x27 - 7000000*x28 + 490000*x3 + 700000*x5 - x97 - x98 - x99 - 1260*xi[1]), x107*(4060*x1 + x104 + x105 + x106 - 22050*x17 - 18375*x2 - 60900*x24 + 275625*x25 - 656250*x26 + 787500*x27 + 43750*x3 - 918750*x37 - 52500*x4 - 2625000*x46 + 900*x9 - 441*xi[0] - 270*xi[1]), x113*(3375*x1 - 3000*x10 + x108 + x109 + x112 - 41100*x17 + 68500*x18 - 12750*x2 - 61875*x24 + 233750*x25 - 412500*x26 - 562500*x32 - 1275000*x37 + 2250000*x42 - 3750000*x43 + 1800*x9 - 411*xi[0] - 330*xi[1]), x0*(15625*x0 + 5250*x1 - 15000*x10 + x104 + x114 + x115 + x116 + x117 - 109375*x17 + 312500*x18 - 312500*x19 - 15000*x2 - 109375*x24 + 312500*x25 - 312500*x26 + 765625*x31 - 2187500*x32 - 2187500*x37 + 6250000*x38 - 6250000*x39 - 6250000*x43 + 5250*x9 - 750*xi[0] - 750*xi[1] + 36)/9, x113*(1800*x1 - 12750*x10 + x109 + x118 + x120 - 61875*x17 + 233750*x18 - 412500*x19 - 3000*x2 - 41100*x24 + 68500*x25 - 1275000*x32 + 2250000*x33 - 562500*x37 - 3750000*x39 + 3375*x9 - 330*xi[0] - 411*xi[1]), x107*(900*x1 - 18375*x10 + x105 + 43750*x11 + x116 - 52500*x12 + x121 - 60900*x17 + 275625*x18 - 656250*x19 + 787500*x20 - 22050*x24 - 918750*x32 - 2625000*x34 + 4060*x9 - 270*xi[0] - 441*xi[1]), x103*(x100 + 490000*x11 - x122 - x123 + x125 + 700000*x13 - 328300*x17 + 1692250*x18 - 4900000*x19 + 8050000*x20 - 7000000*x21 + 32830*x9 - 1260*xi[0] - 3267*xi[1]), x96*(-841050*x10 - 5670000*x12 + x126 + x127 + x128 + x129 - 4500000*x14 - 13698*xi[1] + 504), x130*(841050*x10*xi[0] - 988705*x10 + 3647175*x11 + 5670000*x12*xi[0] - 8476125*x12 - x126*xi[0] - x127*xi[0] - x128*xi[0] - x129*xi[0] + 12495000*x13 + 4500000*x14*xi[0] - 11325000*x14 + 5750000*x15 - x53 + 161353*x9 - x95 - 504*xi[0] - 14202*xi[1]), x133*(-1260*x1 + 2179465*x10 - 7953575*x11 + 18247250*x12 - 26495000*x13 + x131 - x132 + 23600000*x14 - 11750000*x15 + 689110*x17 - 3871875*x18 + 12694500*x19 - 25095000*x20 + 29400000*x21 - 18750000*x22 + 5000000*x23 + 32670*x24 + 1692250*x32 - 4900000*x33 + 8050000*x34 + 2500000*x36 - 358803*x9 + x90 + 2394*xi[0] + 31797*xi[1]), x134*(8662500*x1*x11 + 11250000*x1*x13 + 2430*x1 + 3385725*x10*xi[0] - 1213195*x10 + 4368525*x11 + 2625000*x12*x2 + 20921250*x12*xi[0] - 9867375*x12 + 14077500*x13 + 14625000*x14*xi[0] - 12300000*x14 + 6000000*x15 - 613030*x17 - 10864000*x19 + 22050*x2*xi[1] - 900*x2 - 23762500*x21 - 3750000*x23 - 62235*x24 - 3089625*x32 - 13650000*x34 - 3750000*x36 - 203000*x37 - 2187500*x39 - 1250000*x41 - x53 - x88 + 201951*x9 - 2178*xi[0] - 18054*xi[1]), x139*(2734310*x10 - 9680025*x11 + 21452250*x12 - 29985000*x13 + x131 - x135 - x136 + x137 + x138 + 25650000*x14 - 12250000*x15 + 1978945*x17 - 10689200*x18 + 33381500*x19 - 62285000*x20 + 68300000*x21 - 40500000*x22 + 10000000*x23 + 325835*x24 - 244900*x25 + 68500*x26 + 15307250*x32 - 41212500*x33 + 61775000*x34 - 48000000*x35 + 15000000*x36 + 2186500*x37 + 21250000*x39 - 23500000*x40 + 10000000*x41 - 562500*x42 + 2125000*x43 + 2500000*x45 + x87 - 461789*x9 + 41766*xi[1]), xi[1]*(205937500*x1*x11 + 212500000*x1*x13 + 75000*x1 + 40270625*x10*xi[0] - 7818625*x10 + 27074375*x11 - x110 + 162500000*x12*x2 + 218125000*x12*xi[0] - 58608125*x12 + 80000000*x13 + 131250000*x14*xi[0] - 66875000*x14 + x140 - 31250000*x15*xi[0] + 31250000*x15 - 6250000*x16 - 7681625*x17 - 121449375*x19 + 2224375*x2*xi[1] - 95250*x2 - 230000000*x21 - 1848250*x24 - 1325000*x26 + 10312500*x3*x9 + 60000*x3 - 80571875*x32 - 290937500*x34 - 62500000*x36 - 19040625*x37 - 161562500*x39 + 312500*x4*xi[1] - 62500000*x41 - 35937500*x43 - 31250000*x45 - 2187500*x46 - 6250000*x48 - x81 + 1346625*x9 - 29286*xi[0] - 123786*xi[1])/9, x139*(-66786*x1 + 3642935*x10 - 12248475*x11 + 25757250*x12 - 34215000*x13 + x131 + 27900000*x14 - 12750000*x15 + 4734595*x17 - 23799075*x18 + 68572000*x19 + 122625*x2 - 117690000*x20 + 118900000*x21 - 65250000*x22 + 15000000*x23 + 1592285*x24 - 2749125*x25 + 2633750*x26 - 1327500*x27 + 275000*x28 - 125250*x3 + 62371000*x32 - 149062500*x33 + 196875000*x34 - 135000000*x35 + 37500000*x36 + 22122500*x37 + 156500000*x39 + 67500*x4 - 142500000*x40 + 50000000*x41 - 18937500*x42 + 58750000*x43 + 37500000*x45 + 8400000*x46 - 20250000*x47 + 15000000*x48 - 1500000*x49 - 15000*x5 + 2500000*x50 + x77 - 645201*x9 + 19179*xi[0] + 60759*xi[1]), x134*(126262500*x1*x11 + 99750000*x1*x13 + 44625000*x10*x4 + 17488100*x10*xi[0] - 2168695*x10 + 7008225*x11 + 135625000*x12*x2 + 77341250*x12*xi[0] - 14224875*x12 + 18322500*x13 + 39375000*x14*xi[0] - 14550000*x14 + x141 + x142 + x143 - x144 - x145 - x146 - x147 - x148*x9 + 6500000*x15 - 3702150*x17 - 47500250*x19 + 4047400*x2*xi[1] - 74637500*x21 - 8750000*x23 - 1726515*x24 - 5586875*x26 - 2012500*x28 + 35262500*x3*x9 - 57405250*x32 - 154962500*x34 - 26250000*x36 - 29463000*x37 - 164500000*x39 + 4541250*x4*xi[1] - 43750000*x41 - 91875000*x43 - 43750000*x45 - 24237500*x46 - 26250000*x48 + 8750000*x5*x9 - 8750000*x50 - x53 + 375000*x6*xi[1] - x75 + 400579*x9 - 39246*xi[1]), x133*(5295340*x10 - 16234925*x11 + 31582250*x12 - 39305000*x13 + x131 + 30350000*x14 - x149 - 13250000*x15 - x150 - x151 - x152 + x153 + x154 + x155 + x156 + 11934750*x17 - 51499000*x18 + 129969000*x19 - 199430000*x20 + 183400000*x21 - 93000000*x22 + 20000000*x23 + 7983480*x24 - 24617600*x25 + 46142250*x26 - 53830000*x27 + 38150000*x28 - 15000000*x29 + 2500000*x30 + 204053500*x32 - 407575000*x33 + 463050000*x34 - 280000000*x35 + 70000000*x36 + 148169000*x37 + 626500000*x39 - 469000000*x40 + 140000000*x41 - 225575000*x42 + 481250000*x43 + 175000000*x45 + 201600000*x46 - 287000000*x47 + 140000000*x48 - 98000000*x49 + 70000000*x50 + 20000000*x51 + x54 - 1043307*x9 + 110178*xi[1]), x130*x55, x157*(790254*x0 + x132 + x149 + x150 + x151 + x152 - x153 - x154 - x155 - x156 - 4032210*x17 + 11176900*x18 - 18200000*x19 + 17430000*x20 - 9100000*x21 + 2000000*x22 - 4032210*x24 + 11176900*x25 - 18200000*x26 + 17430000*x27 - 9100000*x28 + 2000000*x29 + 16765350*x31 - 36400000*x32 + 43575000*x33 - 27300000*x34 - 36400000*x37 + 58100000*x38 - 45500000*x39 + 14000000*x40 + 43575000*x42 - 45500000*x43 + 17500000*x44 - 27300000*x46 + 14000000*x47 + x58 + x59 + x60 + x61 - x62 - x63 - x64 - x65 + x89 + 4536), x157*(32830*x1*xi[1] - 36097*x1 - x152 - x158 - x159 + 202055*x2 + 490000*x3*xi[1] - 659225*x3 + 1295000*x4 + 700000*x5*xi[1] - 1505000*x5 + 950000*x6 - x97*xi[1] - x98*xi[1] + 3393*xi[0] + 126*xi[1]), x157*(202055*x10 + 490000*x11*xi[0] - 659225*x11 + 1295000*x12 - x122*xi[0] - x123*xi[0] + 700000*x13*xi[0] - 1505000*x13 + 950000*x14 - x159 - x160 - x61 + 32830*x9*xi[0] - 36097*x9 + 126*xi[0] + 3393*xi[1]), x163*(14052500*x1*x10 + 7875000*x1*x12 + 2108960*x1*xi[1] - 242149*x1 + 25375000*x10*x3 + 19250000*x11*x2 + 4147500*x11*xi[0] + 1400000*x13*xi[0] - x152 - x161 - x162 - 3082100*x18 + 19810000*x2*x9 + 957950*x2 - 3272500*x20 - 6943300*x25 - 13422500*x27 - 1750000*x29 + 12783750*x3*xi[1] - 2218475*x3 - 14525000*x33 - 1750000*x35 + 18900000*x4*x9 + 3132500*x4 - 5250000*x40 - 27212500*x42 - 8750000*x47 - 5250000*x49 + 7525000*x5*xi[1] - 2660000*x5 + 1250000*x6 - x67 - x68 - x69 + x70 + x72 + x73 + 1344070*x9*xi[0] + 31686*xi[0]), x169*(155957*x1 - 40875*x10 + 41750*x11 - 22500*x12 + x152 + x164 + x165 + x167 + x168 - 456555*x17 + 780125*x18 - 738750*x19 - 694455*x2 + 367500*x20 - 1110135*x24 + 4232575*x25 - 8748750*x26 + 10067500*x27 - 6075000*x28 + 1500000*x29 + 1767975*x3 - 4773750*x32 + 3850000*x33 - 1575000*x34 - 10113750*x37 - 6750000*x39 - 2692500*x4 + 1500000*x40 + 15975000*x42 - 12750000*x43 - 12375000*x46 + 2430000*x5 - 1200000*x6 + 22262*x9 - 17733*xi[0] - 6393*xi[1]), x173*(7037500*x1*x10 + 2982175*x1*xi[1] - 532755*x1 + 27500000*x10*x3 + 47625*x10 + 8750000*x11*x2 + 587500*x11*xi[0] - 30000*x11 - x170 - x171 - x172 - 993125*x18 + 23912500*x2*x9 + 2577425*x2 - 12679875*x25 - 36500000*x27 - 6250000*x29 + 29025000*x3*xi[1] - 7093625*x3 - 3687500*x33 + 37500000*x4*x9 + 11570000*x4 - 43187500*x42 - 12500000*x49 + 23750000*x5*xi[1] - 11075000*x5 + 5750000*x6 + 830375*x9*xi[0] - 37500*x9 - x92 + 56223*xi[0] + 14643*xi[1]), x173*(384305*x1 + x165 - 299975*x17 + x174 + x178 + x179 + 224500*x18 - 1965700*x2 - 1625475*x24 + 7438250*x25 - 18505000*x26 + 25275000*x27 - 17750000*x28 + 5000000*x29 + 5748625*x3 - 1737500*x32 - 10225000*x37 - 9955000*x4 + 20762500*x42 - 20250000*x46 + 7500000*x49 + 10075000*x5 - 5500000*x6 + x82 - x84 - x85 + x92 - 38742*xi[0]), x169*(174015*x1*xi[1] - 57887*x1 - x152 - x180 - x182 + 742500*x2*x9 + 307920*x2 - 838550*x25 - 3267500*x27 - 750000*x29 + 2223750*x3*xi[1] - 942975*x3 + 1800000*x4*x9 + 1717500*x4 - 1650000*x42 - 750000*x49 + 2475000*x5*xi[1] - 1830000*x5 + 1050000*x6 + 19395*x9*xi[0] - 810*x9 + 5658*xi[0] + 726*xi[1]), x163*(45099*x1 + x152 + x183 - x184 + x185 + x187 + x189 - 246925*x2 - 85960*x24 + 430325*x25 - 1198750*x26 + 1872500*x27 - 1525000*x28 + 783475*x3 + 1660000*x5 - 1000000*x6 - 4311*xi[0] - 342*xi[1]), x163*(49070*x1*xi[1] - 4501*x1 - x187 - x190 - x191 - x192 + 22435*x2 - 242725*x25 - 1015000*x27 + 665000*x3*xi[1] - 62125*x3 + 96250*x4 + 800000*x5*xi[1] - 77500*x5 + 459*xi[0] + 198*xi[1]), x169*(20055*x1*xi[1] - 1262*x1 - x180 - x193 - x194 + 332500*x2*x9 + 5375*x2 - 84875*x25 - 192500*x27 + 183750*x3*xi[1] - 11750*x3 + 700000*x4*x9 + 12500*x4 - 700000*x42 + 9205*x9*xi[0] - 390*x9 + 143*xi[0] + 96*xi[1]), x173*(762500*x1*x10 + 57625*x1*xi[1] - 3000*x1 + 3250000*x10*x3 + 4800*x10 - x114 - x177 - 103000*x18 - x195 - x196 - x197 + 1150000*x2*x9 + 10125*x2 - 193125*x25 + 282500*x3*xi[1] - 1637500*x42 + 46175*x9*xi[0] - 2130*x9 + 393*xi[0] + 348*xi[1]), x173*(1150000*x1*x10 + 46175*x1*xi[1] - 2130*x1 + 10125*x10 + 3250000*x11*x2 + 282500*x11*xi[0] - x115 - x171 - 193125*x18 - x190 - x196 - x198 + 762500*x2*x9 + 4800*x2 - 103000*x25 - 1637500*x33 + 57625*x9*xi[0] - 3000*x9 + 348*xi[0] + 393*xi[1]), x169*(332500*x1*x10 + 700000*x1*x12 + 9205*x1*xi[1] - 390*x1 + 5375*x10 + 183750*x11*xi[0] - 11750*x11 + 12500*x12 - x167 - 84875*x18 - x194 - x199 - 192500*x20 - 700000*x33 + 20055*x9*xi[0] - 1262*x9 + 96*xi[0] + 143*xi[1]), x163*(22435*x10 + 665000*x11*xi[0] - 62125*x11 + 96250*x12 + 800000*x13*xi[0] - 77500*x13 - x161 - 242725*x18 - x192 - x195 - 1015000*x20 - x200 + 49070*x9*xi[0] - 4501*x9 + 198*xi[0] + 459*xi[1]), x163*(-246925*x10 + 783475*x11 + 1660000*x13 - 1000000*x14 - 85960*x17 + x175 + 430325*x18 + x189 - 1198750*x19 + 1872500*x20 + x200 - x201 + x202 - 1525000*x21 + x61 + 45099*x9 - 342*xi[0] - 4311*xi[1]), x169*(742500*x1*x10 + 1800000*x1*x12 + 19395*x1*xi[1] - 810*x1 + 307920*x10 + 2223750*x11*xi[0] - 942975*x11 + 1717500*x12 + 2475000*x13*xi[0] - 1830000*x13 + 1050000*x14 - 838550*x18 - x182 - x199 - 3267500*x20 - 750000*x22 - 1650000*x33 - 750000*x35 - x61 + 174015*x9*xi[0] - 57887*x9 + 726*xi[0] + 5658*xi[1]), x173*(-1965700*x10 + 5748625*x11 - 9955000*x12 + x129 + 10075000*x13 + x135 - x137 - x138 - 5500000*x14 - 1625475*x17 + x179 + 7438250*x18 - 18505000*x19 + 25275000*x20 + x203 + x204 + x205 - 17750000*x21 + 5000000*x22 - 299975*x24 + 224500*x25 - 10225000*x32 + 20762500*x33 - 20250000*x34 + 7500000*x35 - 1737500*x37 + 384305*x9 - 38742*xi[1]), x173*(23912500*x1*x10 + 37500000*x1*x12 + 830375*x1*xi[1] - 37500*x1 + 8750000*x10*x3 + 2577425*x10 + 27500000*x11*x2 + 29025000*x11*xi[0] - 7093625*x11 + 11570000*x12 - x129 + 23750000*x13*xi[0] - 11075000*x13 + 5750000*x14 - x172 - 12679875*x18 - x197 + 7037500*x2*x9 + 47625*x2 - 36500000*x20 - x206 - 6250000*x22 - 993125*x25 + 587500*x3*xi[1] - 30000*x3 - 43187500*x33 - 12500000*x35 - 3687500*x42 + 2982175*x9*xi[0] - 532755*x9 + 14643*xi[0] + 56223*xi[1]), x169*(22262*x1 - 694455*x10 + 1767975*x11 - 2692500*x12 + 2430000*x13 - 1200000*x14 + x168 - 1110135*x17 + 4232575*x18 - 8748750*x19 + x193 - 40875*x2 + 10067500*x20 + x204 + x207 - 6075000*x21 + 1500000*x22 - 456555*x24 + 780125*x25 - 738750*x26 + 367500*x27 + 41750*x3 - 10113750*x32 + 15975000*x33 - 12375000*x34 - 4773750*x37 - 12750000*x39 - 22500*x4 + 3850000*x42 - 6750000*x43 - 1575000*x46 + 1500000*x47 + x61 + 155957*x9 - 6393*xi[0] - 17733*xi[1]), x163*(19810000*x1*x10 + 18900000*x1*x12 + 1344070*x1*xi[1] + 19250000*x10*x3 + 957950*x10 + 25375000*x11*x2 + 12783750*x11*xi[0] - 2218475*x11 + 3132500*x12 + 7525000*x13*xi[0] - 2660000*x13 + 1250000*x14 - x141 - x142 - x143 + x144 + x146 + x147 - x162 - 6943300*x18 - x191 + 14052500*x2*x9 - 13422500*x20 - 1750000*x22 - 3082100*x25 - 3272500*x27 + 4147500*x3*xi[1] - 27212500*x33 - 5250000*x35 + 7875000*x4*x9 - 8750000*x40 - 14525000*x42 - 5250000*x47 - 1750000*x49 + 1400000*x5*xi[1] - x61 + 2108960*x9*xi[0] - 242149*x9 + 31686*xi[1]), x211*(247984*x0 + 86192*x1 - 263495*x10 + 450500*x11 - 440000*x12 + 230000*x13 - 1429785*x17 + 4028200*x18 - 6287500*x19 - 263495*x2 + 5555000*x20 + x208 + x209 - 2600000*x21 + x210 - 1429785*x24 + 4028200*x25 - 6287500*x26 + 5555000*x27 - 2600000*x28 + 450500*x3 + 7155400*x31 - 16662500*x32 + 20150000*x33 - 12300000*x34 + 3000000*x35 - 16662500*x37 + 29650000*x38 - 24250000*x39 - 440000*x4 + 7500000*x40 + 20150000*x42 - 24250000*x43 - 12300000*x46 + 7500000*x47 + 3000000*x49 + 230000*x5 + 86192*x9 - 13953*xi[0] - 13953*xi[1]), x211*(12788*x1 - 600*x10 - 28600*x17 + 13700*x18 - 60995*x2 + x208 + x212*x9 + x213 - 151995*x24 + 713200*x25 - 1822500*x26 + 2545000*x27 - 1800000*x28 + 159500*x3 - 112500*x32 - 1075000*x37 - 230000*x4 + 2350000*x42 - 750000*x43 - 2500000*x46 + 500000*x47 + 170000*x5 + 1200*x9 - 1347*xi[0] - 654*xi[1]), x211*(x1*x214 + 1200*x1 - 60995*x10 + 159500*x11 - 230000*x12 + 170000*x13 - 151995*x17 + 713200*x18 - 1822500*x19 - 600*x2 + 2545000*x20 + x209 - 1800000*x21 + x213 - 28600*x24 + 13700*x25 - 1075000*x32 + 2350000*x33 - 2500000*x34 - 112500*x37 - 750000*x39 + 500000*x40 + 12788*x9 - 654*xi[0] - 1347*xi[1]), x217*(15412500*x1*x10 + 7000000*x1*x12 + 1901825*x1*xi[1] - 123515*x1 + 35000000*x10*x3 + 140875*x10 + 24375000*x11*x2 + 2943750*x11*xi[0] - 168750*x11 + 102500*x12 - x121 - x170 - 2629375*x18 + 23350000*x2*x9 + 447425*x2 - 1662500*x20 - x215 - x216 - 6478625*x25 - 11650000*x27 + 11725000*x3*xi[1] - 873125*x3 - 14875000*x33 + 23125000*x4*x9 + 946250*x4 - 33312500*x42 - 6250000*x49 + 6000000*x5*xi[1] - 537500*x5 - x80 + 1217275*x9*xi[0] - 61310*x9 + 16221*xi[0] + 12441*xi[1]), x219*(148450*x1 - 69750*x10 + x102 + 52500*x11 + x120 - 926975*x17 + 1437750*x18 - 1022500*x19 - 611725*x2 + x212 + x218 - 2098950*x24 + 8300250*x25 - 17245000*x26 + 19300000*x27 - 11000000*x28 + 1345000*x3 - 9980000*x32 + 6350000*x33 - 24602500*x37 - 14750000*x39 - 1615000*x4 + 41400000*x42 - 35500000*x43 - 33000000*x46 + 15000000*x47 + 10000000*x49 + 42675*x9 - 17481*xi[0] - 11181*xi[1]), x217*(1412500*x1*x10 + 614875*x1*xi[1] - 47435*x1 + 7250000*x10*x3 + 8400*x10 - x164 - x178 - 184000*x18 + 6043750*x2*x9 + 213050*x2 - x215 - x220 - x221 - 2688000*x25 - 7887500*x27 + 6277500*x3*xi[1] - 516875*x3 + 10875000*x4*x9 + 683750*x4 - 11800000*x42 + 5000000*x5*xi[1] - 462500*x5 + 185225*x9*xi[0] - 8070*x9 + 5226*xi[0] + 2886*xi[1]), x217*(10310*x1 - 6600*x10 + x106 + x111 - 100625*x17 + x178 + 143500*x18 - 40375*x2 + x222 - 173775*x24 + 670625*x25 - 1278750*x26 + 1137500*x27 + 78750*x3 - 1087500*x32 - 3018750*x37 - 72500*x4 + 5425000*x42 - 5250000*x43 - 4375000*x46 + 4470*x9 - 1239*xi[0] - 924*xi[1]), x219*(33675*x0 + 11325*x1 - 32250*x10 + 37500*x11 + x112 + x120 - 228025*x17 + x174 + 637750*x18 - 717500*x19 - 32250*x2 + x203 - 228025*x24 + 637750*x25 - 717500*x26 + 37500*x3 + 1515500*x31 - 4092500*x32 + 4300000*x33 - 4092500*x37 + 10300000*x38 + 4300000*x42 + 5000000*x44 + 11325*x9 - 1656*xi[0] - 1656*xi[1] + 81), x217*(4470*x1 - 40375*x10 + 78750*x11 + x119 - 72500*x12 + x121 - 173775*x17 + 670625*x18 - 1278750*x19 - 6600*x2 + 1137500*x20 + x205 + x222 - 100625*x24 + 143500*x25 - 3018750*x32 + 5425000*x33 - 4375000*x34 - 1087500*x37 - 5250000*x39 + 10310*x9 - 924*xi[0] - 1239*xi[1]), x217*(6043750*x1*x10 + 10875000*x1*x12 + 185225*x1*xi[1] - 8070*x1 + 213050*x10 + 7250000*x11*x2 + 6277500*x11*xi[0] - 516875*x11 + 683750*x12 + 5000000*x13*xi[0] - 462500*x13 - 2688000*x18 + 1412500*x2*x9 + 8400*x2 - 7887500*x20 - x205 - x207 - x221 - x223 - x224 - 184000*x25 - 11800000*x33 + 614875*x9*xi[0] - 47435*x9 + 2886*xi[0] + 5226*xi[1]), x219*(42675*x1 - 611725*x10 + 1345000*x11 + x112 - 1615000*x12 + x125 - 2098950*x17 + 8300250*x18 - 17245000*x19 - 69750*x2 + 19300000*x20 - 11000000*x21 + x214 + x218 - 926975*x24 + 1437750*x25 - 1022500*x26 + 52500*x3 - 24602500*x32 + 41400000*x33 - 33000000*x34 + 10000000*x35 - 9980000*x37 - 35500000*x39 + 15000000*x40 + 6350000*x42 - 14750000*x43 + 148450*x9 - 11181*xi[0] - 17481*xi[1]), x217*(23350000*x1*x10 + 23125000*x1*x12 + 1217275*x1*xi[1] - 61310*x1 + 24375000*x10*x3 + 447425*x10 - x106 + 35000000*x11*x2 + 11725000*x11*xi[0] - 873125*x11 + 946250*x12 + 6000000*x13*xi[0] - 537500*x13 - x140 - 6478625*x18 + 15412500*x2*x9 + 140875*x2 - 11650000*x20 - x206 - x216 - x224 - 2629375*x25 - 1662500*x27 + 2943750*x3*xi[1] - 168750*x3 - 33312500*x33 - 6250000*x35 + 7000000*x4*x9 + 102500*x4 - 14875000*x42 + 1901825*x9*xi[0] - 123515*x9 + 12441*xi[0] + 16221*xi[1]), x169*(97220*x0 + 32860*x1 - 94375*x10 + x106 + 133750*x11 - 92500*x12 + x121 + x165 - 623000*x17 + 1709625*x18 + x184*xi[1] - 2291250*x19 - 94375*x2 + x201*xi[0] + x204 - 623000*x24 + 1709625*x25 - 2291250*x26 + 133750*x3 + 3781750*x31 - 9428750*x32 + 11112500*x33 - 6125000*x34 - 9428750*x37 + 19300000*x38 - 16500000*x39 - 92500*x4 + 11112500*x42 - 16500000*x43 + 7500000*x44 - 6125000*x46 + 32860*x9 - 4987*xi[0] - 4987*xi[1] + 252), x169*(3326250*x1*x10 + 319500*x1*xi[1] - 17860*x1 + 10125000*x10*x3 + 24375*x10 - x106 + 6000000*x11*x2 + 435000*x11*xi[0] - x118 - x171 - 494125*x18 + 5452500*x2*x9 + 61875*x2 - x220 - x225 - 1077875*x25 - 1312500*x27 + 1736250*x3*xi[1] - 103750*x3 - 2662500*x33 + 5250000*x4*x9 + 82500*x4 - 7962500*x42 + 237000*x9*xi[0] - 11250*x9 + 2343*xi[0] + 1983*xi[1]), x169*(5452500*x1*x10 + 5250000*x1*x12 + 237000*x1*xi[1] - 11250*x1 + 6000000*x10*x3 + 61875*x10 - x108 + 10125000*x11*x2 + 1736250*x11*xi[0] - 103750*x11 + 82500*x12 - x121 - 1077875*x18 - x197 + 3326250*x2*x9 + 24375*x2 - 1312500*x20 - x223 - x225 - 494125*x25 + 435000*x3*xi[1] - 7962500*x33 - 2662500*x42 + 319500*x9*xi[0] - 17860*x9 + 1983*xi[0] + 2343*xi[1])])
                case _:
                    assert (
                        False
                    ), "Order of shape functions not implemented or number of nodes not adequat"
        case 3:
            match n_nodes:
                case 4:
                    def shape_functions(xi):
                
                        return jnp.asarray([-xi[0] - xi[1] - xi[2] + 1, xi[0], xi[1], xi[2]])

                case 10:
                    def shape_functions(xi):
                        x0 = 4*xi[0]
                        x1 = x0*xi[1]
                        x2 = x0*xi[2]
                        x3 = 4*xi[1]
                        x4 = x3*xi[2]
                        x5 = -xi[0] - xi[1] - xi[2] + 1
                        return jnp.asarray([x1 + x2 + x4 + 2*xi[0]**2 - 3*xi[0] + 2*xi[1]**2 - 3*xi[1] + 2*xi[2]**2 - 3*xi[2] + 1, xi[0]*(2*xi[0] - 1), xi[1]*(2*xi[1] - 1), xi[2]*(2*xi[2] - 1), x0*x5, x1, x3*x5, 4*x5*xi[2], x2, x4])

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
                        return jnp.asarray([-x0*x7 - x0*x8 + 9*x0 - x1*x6 - x1*x8 + 9*x1 - x2*x6 - x2*x7 + 9*x2 - x5 - 9*xi[0]**3/2 + 18*xi[0]*xi[1] + 18*xi[0]*xi[2] - 11*xi[0]/2 - 9*xi[1]**3/2 + 18*xi[1]*xi[2] - 11*xi[1]/2 - 9*xi[2]**3/2 - 11*xi[2]/2 + 1, xi[0]*(9*x0 - 9*xi[0] + 2)/2, xi[1]*(9*x1 - 9*xi[1] + 2)/2, xi[2]*(9*x2 - 9*xi[2] + 2)/2, x13*x14, x14*(-x10 - x16 - x17 + 4*xi[0]), x18*x19, x19*x21, x23*(-x11 - x16 - x22 + 4*xi[1]), x13*x23, x13*x24, x24*(-x12 - x17 - x22 + 4*xi[2] - 1), x18*x25, x25*x26, x21*x27, x26*x27, x29*xi[1], x29*xi[2], 27*x28*x3, x5])

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
                        return jnp.asarray([x0*xi[1] + x0*xi[2] + 128*x1*x2 + 140*x1/3 + x10*x5 + x10*x7 - x11*x2 - x11*x6 - x12*x2 - x12*x4 + x13*x3 + x13*x7 + x14*x3 + x14*x5 + x15*x4*xi[2] + x15*x6*xi[1] + x16*x4 + x16*x6 + 70*x2/3 - 80*x3/3 + 64*x4*x6 - x4*x9 + 70*x4/3 - 80*x5/3 - x6*x9 + 70*x6/3 - 80*x7/3 - 160*x8 + 32*xi[0]**4/3 - 25*xi[0]/3 + 32*xi[1]**4/3 - 25*xi[1]/3 + 32*xi[2]**4/3 - 25*xi[2]/3 + 1, xi[0]*(-48*x2 + 32*x3 + 22*xi[0] - 3)/3, xi[1]*(-48*x4 + 32*x5 + 22*xi[1] - 3)/3, xi[2]*(-48*x6 + 32*x7 + 22*xi[2] - 3)/3, x28*x29, x40*(x27 + 16*x3 + x30*xi[1] + x30*xi[2] - x30 - x31 + x32*x4 + x32*x6 + x36 + x39 + 19*xi[0]), x29*(-x17 + 14*x2 - x41 - x42 - x43 - x44*xi[1] - x44*xi[2] - x45), x46*x47, x51*(x32*xi[1] + x49 + x50), x47*x53, x55*(-x18 - x37 + 14*x4 - x42 - x52*xi[0] - x52*xi[2] - x54 - xi[2]), x48*(x2*x59 + x25 + x36 + 16*x5 + x56*xi[0] + x56*xi[2] - x56 - x58 + x59*x6 + x61 + 19*xi[1]), x28*x55, x28*x62, x66*(x2*x65 + x26 + x35 + x39 + x4*x65 + x61 + x63*xi[0] + x63*xi[1] - x63 - x64 + 16*x7 + 19*xi[2]), x62*(-x19 - x33 - x43 - x54 + 14*x6 - x67*xi[0] - x67*xi[1] - xi[1]), x46*x68, x70*(x32*xi[2] + x50 + x69), x68*x71, x53*x72, x73*(16*x1 + x49 + x69 + 1), x71*x72, x74*x76, x76*x79, x76*x81, x74*x82, x79*x82, x82*x83, x74*x84, x83*x84, x81*x84, x35*(x40 - 1), x35*(x48 - 1), x35*(x66 - 1), 256*x8*(-x45 - xi[0])])

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
                        return jnp.asarray([-x0*x13 - x0*x14 - 3125*x0*x22/4 + 1875*x0*x3/4 + 1875*x0*x6/4 + 1875*x0*xi[1]*xi[2]/2 + 375*x0/8 - 3125*x1*x9/6 + 625*x1*xi[1]/2 + 625*x1*xi[2]/2 - 2125*x1/24 - 2125*x10/4 - x11*x3 - x11*x6 - x12*x5 - x12*x8 - x13*x6 - x14*x3 - x15*x2 - x15*x8 - x16*x2 - x16*x5 - 3125*x17*x7/6 - 3125*x18*x4/6 - x19*x4 - x19*x7 + 625*x2/8 - x20*x3 - x20*x6 - 3125*x21*xi[0]/4 - 3125*x23*x3/4 + 1875*x3*x6/4 - 3125*x3*x7/12 + 1875*x3*xi[0]*xi[2]/2 + 375*x3/8 - 3125*x4*x6/12 + 625*x4*xi[0]/2 + 625*x4*xi[2]/2 - 2125*x4/24 + 625*x5/8 + 1875*x6*xi[0]*xi[1]/2 + 375*x6/8 + 625*x7*xi[0]/2 + 625*x7*xi[1]/2 - 2125*x7/24 + 625*x8/8 - 625*xi[0]**5/24 + 375*xi[0]*xi[1]/4 + 375*xi[0]*xi[2]/4 - 137*xi[0]/12 - 625*xi[1]**5/24 + 375*xi[1]*xi[2]/4 - 137*xi[1]/12 - 625*xi[2]**5/24 - 137*xi[2]/12 + 1, xi[0]*(875*x0 - 1250*x1 + 625*x2 - x24 + 24)/24, xi[1]*(-x25 + 875*x3 - 1250*x4 + 625*x5 + 24)/24, xi[2]*(-x26 + 875*x6 - 1250*x7 + 625*x8 + 24)/24, x41*x42, x69*(675*x0*xi[1] + 675*x0*xi[2] - 295*x0 + 325*x1 - x28 - x40*x9 - x44 - x46 - x47 - x48*xi[1] - x48*xi[2] - x53 - x57 - x64 - x68 + 107*xi[0]), x69*(245*x0 + x1*x25 + x1*x26 - 300*x1 + x23*x25 + x28 + x71 + x73 + x78 + x79 + 20*x9 - 78*xi[0]), x42*(150*x0*xi[1] + 150*x0*xi[2] - 205*x0 + 275*x1 - x28 - x81 - x82 - x87 - x91 + 61*xi[0]), x92*x93, x100*(x74*xi[1] + x94 + x96 + x99), x100*(x101 + x104 + 125*x38 + x96), x106*x93, x112*(-x107 - x111 - x29 + 150*x3*xi[0] + 150*x3*xi[2] - 205*x3 + 275*x4 - x44 - x87 + 61*xi[1]), x118*(x113*x24 + x116 + x117 + 20*x18 + x24*x4 + x26*x4 + x29 + 245*x3 - 300*x4 + x53 + x78 - 78*xi[1]), x118*(-x114*x4 - x119 - x120 - x123 - x127 - x29 + 675*x3*xi[0] + 675*x3*xi[2] - 295*x3 - 750*x39 - x4*x49 + 325*x4 - x64 - x71 - x81 + 107*xi[1]), x112*x41, x128*x41, x130*(-x107 - x116 - x121*x7 - x127 - x129 - x30 - 750*x37 - x49*x7 + 675*x6*xi[0] + 675*x6*xi[1] - 295*x6 - x63 - x68 + 325*x7 - x73 - x82 + 107*xi[2] - 12), x130*(x117 + x123 + 20*x17 + x22*x24 + x24*x7 + x25*x7 + x30 + x57 + 245*x6 - 300*x7 + x77 + x79 - 78*xi[2] + 8), x128*(-x111 - x119 - x30 - x46 + 150*x6*xi[0] + 150*x6*xi[1] - 205*x6 + 275*x7 - x91 + 61*xi[2] - 6), x131*x92, x134*(x132 + x133 + x74*xi[2] + x99), x134*(x101 + x133 + x135*xi[0] + x138), x131*x139, x106*x140, x142*(x104 + 125*x113 + x132 + x141), x142*(x135*xi[1] + x138 + x141 + x94), x139*x140, x143*x144, x144*x149, x144*x153, x168*x169, x169*(-x158 - x172 - x177 - x180 + 35*xi[0]*xi[1] - xi[2]), x169*x187, x143*x188, x149*x188, x188*x189, x168*x190, x190*(-x162 - x177 - x191 - x193 + 35*xi[0]*xi[2] - xi[1]), x190*x197, x143*x198, x189*x198, x153*x198, x197*x199, x199*(-x176 - x180 - x184 - x193 - x196 - xi[0] + 35*xi[1]*xi[2] + 1), x187*x199, x200*(x97 - x98 + 2), x200*(x102 - x103 + 2), x200*(x136 - x137 + 2), x203*(25*x17 + x201 + x202), x203*(x175 + x202 + x204), x203*(25*x18 + x201 + x204 + 1), x205*(x155 + x156 + x160 - x163 - x166 + x170 + x182 - x185 + x195 + 4), x205*(-x156 - x206 - x207 - x89), x205*(-x160 - x206 - x208 - x84), x205*(-x109 - x170 - x207 - x208 - 1)])

        case _:
            assert False, "Dimensionality not implemented."

    shape_functions = jax.jit(shape_functions)

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
            dX_dxi = jax.jacfwd(initial_coor)(xi)

            fun = lambda xi: jnp.einsum('i, i...-> ...', shape_functions(xi), fI)
            primal_out = fun(xi)
            df_dxi = jax.jacfwd(fun)(xi)

            tangent_out = jnp.einsum('...i, i-> ...', df_dxi, matrix_inv(dX_dxi) @ x_dot)

            # Add tangent with respect to fI
            if fI_dot is not None:
                tangent_out += jnp.einsum('i, i...-> ...', shape_functions(xi), fI_dot)

            return primal_out, tangent_out

        return ansatz(x, fI, xI)
    else:
        return jnp.einsum('i, i...-> ...', shape_functions(x), fI)


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

