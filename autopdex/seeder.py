# seeder.py
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
The key functionalities of the seeder module include:
  - Generation of seed points within a specified box regularly or using quasi-random methods.
  - Filtering seed points based on signed distance functions.
  - Calculation of integration points and weights for standard geometric elements such as lines, triangles, and tetrahedrons in specific configurations.
  - Tensor product Gauß-Quadrature
"""

import sys
import math

import jax
import jax.numpy as jnp
from jax import random

from autopdex import geometry
from autopdex.utility import jit_with_docstring

key = random.key(0)

### Generation of nodes
@jit_with_docstring()
def _get_dim_step(min, max, spacing):
    """
    Computes the dimension and step size for generating a regular grid.

    Args:
      min (array): Minimum coordinates.
      max (array): Maximum coordinates.
      spacing (float): Spacing between points.

    Returns:
      tuple: Dimension and step size.
    """
    dim = min.shape[0]
    step = jax.lax.complex(0.0, 1.0 + (max - min) / spacing)
    return (dim, step)


def regular_in_psdf(psdf, min, max, spacing, atol=1e-8):
    """
    Generates regular seed points within a box and filters them using psdf.

    Args:
      psdf (function): Positive smooth distance function.
      min (array): Minimum coordinates.
      max (array): Maximum coordinates.
      spacing (float): Spacing between points.
      atol (float, optional): Absolute tolerance. Defaults to 1e-8 (no points on boundary).

    Returns:
      tuple: Filtered seed points, number of seeds, and estimated region size.
    """
    (min, max) = (jnp.asarray(min), jnp.asarray(max))
    (x_seeds, n_seeds) = regular_in_box(min, max, spacing)

    x_seeds = just_in_psdf(psdf, x_seeds, n_seeds, atol)
    fill_ratio = x_seeds.shape[0] / n_seeds
    n_seeds = x_seeds.shape[0]

    region_size = estimate_size(min, max, fill_ratio)
    print("Number of seeds in domain: ", n_seeds)
    return (x_seeds, n_seeds, region_size)


def regular_in_box(min, max, spacing):
    """
    Generates regular seed points within a specified box.

    Args:
      min (array): Minimum coordinates.
      max (array): Maximum coordinates.
      spacing (float): Spacing between points.

    Returns:
      tuple: Seed points and number of seeds.
    """
    (min, max) = (jnp.asarray(min), jnp.asarray(max))
    (dim, step) = _get_dim_step(min, max, spacing)
    match dim:
        case 1:
            X = jnp.mgrid[min[0] : max[0] : step[0]]
            x_seeds = jnp.stack([X.flatten()]).transpose()
        case 2:
            X, Y = jnp.mgrid[min[0] : max[0] : step[0], min[1] : max[1] : step[1]]
            x_seeds = jnp.stack([X.flatten(), Y.flatten()]).transpose()
        case 3:
            X, Y, Z = jnp.mgrid[
                min[0] : max[0] : step[0],
                min[1] : max[1] : step[1],
                min[2] : max[2] : step[2],
            ]
            x_seeds = jnp.stack([X.flatten(), Y.flatten(), Z.flatten()]).transpose()
        case 4:
            X, Y, Z, T = jnp.mgrid[
                min[0] : max[0] : step[0],
                min[1] : max[1] : step[1],
                min[2] : max[2] : step[2],
                min[3] : max[3] : step[3],
            ]
            x_seeds = jnp.stack(
                [X.flatten(), Y.flatten(), Z.flatten(), T.flatten()]
            ).transpose()
    n_seeds = x_seeds.shape[0]
    return (x_seeds, n_seeds)


def quasi_random_in_psdf(
    psdf, min, max, n_seeds, mode, atol=1e-8, n_init=10000, approx=True
):
    """
    Generates quasi-random seed points and filters them using psdf.

    Args:
      psdf (function): Positive smooth distance function.
      min (array): Minimum coordinates.
      max (array): Maximum coordinates.
      n_seeds (int): Number of seed points.
      mode (str): Sampling mode ('hammersley' or 'halton').
      atol (float, optional): Absolute tolerance. Defaults to 1e-8.
      n_init (int, optional): Initial number of points for fill ratio estimation. Defaults to 10000.
      approx (bool, optional): Whether to approximate the number of seeds. Defaults to True.

    Returns:
      tuple: Filtered seed points, number of seeds, and estimated region size.
    """
    # Estimaion of fill in ratio
    (x_init, _) = quasi_random_in_box(min, max, n_init, mode)
    x_init = just_in_psdf(psdf, x_init, n_init, atol)
    fill_ratio = x_init.shape[0] / n_init
    if fill_ratio < 0.001:
        print("Bad fill ratio (", fill_ratio, "). Consider increasing n_init.")

    # Generation of enough seeds
    safety = 1.0
    if not approx:
        safety = 1.01
    n_try = math.ceil(safety * n_seeds / fill_ratio)
    (x_try, _) = quasi_random_in_box(min, max, n_try, mode)
    x_try = just_in_psdf(psdf, x_try, n_try, atol)

    # Return first n_seeds that are in psdf of randomly shuffled list x_try
    x_seeds = random.permutation(key, x_try)
    if not approx:
        x_seeds = x_seeds[:n_seeds]

    region_size = estimate_size(min, max, fill_ratio)
    print("Number of seeds in domain: ", x_seeds.shape[0])
    return (x_seeds, x_seeds.shape[0], region_size)


def quasi_random_in_box(min, max, n_seeds, mode):
    """
    Generates quasi-random seed points within a specified box using the given mode.

    Args:
      min (array): Minimum coordinates.
      max (array): Maximum coordinates.
      n_seeds (int): Number of seed points.
      mode (str): Sampling mode ('hammersley' or 'halton').

    Returns:
      tuple: Seed points and number of seeds.
    """
    import skopt

    space = skopt.space.Space(list(map(lambda x, y: (x, y), min, max)))

    if mode == "hammersley":
        sampler = skopt.sampler.Hammersly()
    elif mode == "halton":
        sampler = skopt.sampler.Halton()
    else:
        sys.exit("Sampling mode not defined!")

    pts = sampler.generate(space.dimensions, n_seeds)
    x_seeds = jax.numpy.asarray(pts)
    return (x_seeds, x_seeds.shape[0])


def gauss_points_in_psdf(
    psdf, min, max, spacing, order, type="gauss legendre", atol=1e-8
):
    """
    Quadrature in background cells.

    type=='gauss legendre' uses tensor product rule up to order 10 in up to 4 dimensions

    Returns:
      x_int: positions of integration points
      w_int: weights of integration points
      n_int: number of integration poitns
    """
    # Seeds for integration cells
    (min, max) = (jnp.asarray(min), jnp.asarray(max))
    (x_seeds, n_seeds) = regular_in_box(min, max, spacing)
    n_dim = min.shape[0]

    # Quadrature rule on reference cell in the interval of [0, 1]
    if type == "gauss legendre":
        roots_1d, weights_1d = gauss_legendre_1d(order)
        if n_dim == 1:
            roots_reference, weights_reference = roots_1d, weights_1d
        else:
            roots_reference, weights_reference = tensor_product_rule(
                roots_1d, weights_1d, n_dim
            )
    else:
        assert False, "Quadrature rule not implemented."

    # Transform quadrature rule using actual size of integration cells
    roots_local = spacing * roots_reference
    weights_local = spacing ** (min.shape[0]) * weights_reference

    # Use all seeds to generate global list of integration points and weights
    w_int = jnp.tile(weights_local, n_seeds).flatten()

    def local_int_coor(x_seed):
        return roots_local + jnp.tile(x_seed, roots_local.shape[0]).reshape(
            roots_local.shape
        )

    x_int = jax.jit(jax.vmap(local_int_coor, (0,), 0))(x_seeds).reshape(
        (w_int.shape[0], min.shape[0])
    )

    # Filter integration points with positive smooth distance function
    (x_int, w_int) = just_in_psdf(psdf, x_int, x_int.shape[0], atol, w_int)

    n_int = x_int.shape[0]
    print("Number of integration points in domain: ", n_int)
    return (x_int, w_int, n_int)


@jit_with_docstring()
def estimate_size(min, max, fill_ratio):
    """
    Estimates the region size based on the fill ratio.

    Args:
      min (array): Minimum coordinates.
      max (array): Maximum coordinates.
      fill_ratio (float): Ratio of filled region.

    Returns:
      float: Estimated region size.
    """
    box_lengths = jnp.asarray(max) - jnp.asarray(min)
    box_size = jnp.prod(box_lengths)
    region_size = fill_ratio * box_size
    return region_size


def just_in_psdf(psdf, x_seeds, n_seeds, atol, w_int=None):
    """
    Filters seed points based on a positive smooth distance function (psdf).

    Args:
      psdf (function): Positive smooth distance function.
      x_seeds (array): Seed points.
      n_seeds (int): Number of seed points.
      atol (float): Absolute filtering tolerance (functions in the geometry module may produce nans directly at the boundary).
      w_int (array, optional): Weights. Defaults to None.

    Returns:
      array: Filtered seed points (and weights if provided).
    """

    # Compute psdf at seed points and make a list of bools specifiying whether psdf exceeds atol
    psdf_at_seeds = jax.jit(jax.vmap(psdf, (0), 0))(x_seeds)

    @jax.jit
    def get_condlist():
        return jnp.logical_not(jnp.less_equal(psdf_at_seeds, atol * jnp.ones(n_seeds)))

    cond_list = get_condlist()

    # Delete all seed points that are not in domain
    x_seeds = jnp.compress(cond_list, x_seeds, axis=0)

    if w_int == None:
        return x_seeds
    else:
        w_int = jnp.compress(cond_list, w_int, axis=0)
        return (x_seeds, w_int)


### Numerical integration
@jit_with_docstring()
def tensor_product_two_coordinate_arrays(xi, yi):
    """
    Computes the tensor product of two coordinate arrays.

    Args:
      xi (array): First coordinate array.
      yi (array): Second coordinate array.

    Returns:
      tuple: Two arrays representing the tensor product of the input coordinates.
    """
    xi_new = jnp.tile(xi, yi.shape[0])
    yi_new = (
        jnp.tile(yi, xi.shape[0])
        .reshape((xi.shape[0], yi.shape[0]))
        .transpose()
        .flatten()
    )
    return xi_new, yi_new


def tensor_product_rule(roots_1d, weights_1d, dim):
    """
    Generates integration points and weights for tensor product quadrature rules.

    Args:
      roots_1d (array): 1D quadrature roots.
      weights_1d (array): 1D quadrature weights.
      dim (int): Dimensionality of the quadrature rule (2, 3, or 4).

    Returns:
      tuple: Integration points and weights for the specified dimensionality.
    """
    match dim:
        case 2:
            xi, yi = tensor_product_two_coordinate_arrays(roots_1d, roots_1d)
            x_int = jnp.asarray([xi, yi]).transpose()
            w_int = jnp.outer(weights_1d, weights_1d).flatten()
            return x_int, w_int
        case 3:
            xi_2d, yi_2d = tensor_product_two_coordinate_arrays(roots_1d, roots_1d)

            xi, zi = tensor_product_two_coordinate_arrays(xi_2d, roots_1d)
            yi, _ = tensor_product_two_coordinate_arrays(yi_2d, roots_1d)

            x_int = jnp.asarray([xi, yi, zi]).transpose()
            wi_2d = jnp.outer(weights_1d, weights_1d).flatten()
            w_int = jnp.outer(wi_2d, weights_1d).flatten()
            return x_int, w_int
        case 4:
            xi_2d, yi_2d = tensor_product_two_coordinate_arrays(roots_1d, roots_1d)
            xi_3d, zi_3d = tensor_product_two_coordinate_arrays(xi_2d, roots_1d)
            yi_3d, _ = tensor_product_two_coordinate_arrays(yi_2d, roots_1d)

            xi, ti = tensor_product_two_coordinate_arrays(xi_3d, roots_1d)
            yi, _ = tensor_product_two_coordinate_arrays(yi_3d, roots_1d)
            zi, _ = tensor_product_two_coordinate_arrays(zi_3d, roots_1d)

            x_int = jnp.asarray([xi, yi, zi, ti]).transpose()
            wi_2d = jnp.outer(weights_1d, weights_1d).flatten()
            wi_3d = jnp.outer(wi_2d, weights_1d).flatten()
            w_int = jnp.outer(wi_3d, weights_1d).flatten()
            return x_int, w_int
        case _:
            assert False, "Not implemented for this dimensionality!"


def gauss_legendre_1d(order):
    """
    Returns positions and weights for Gauss-Legendre quadrature for the interval [0,1]

    - Interval [0, 1]
    - accurate for polynomials up to order
    - returns jnp.ceil((order + 1) / 2) integration points
    """

    match order:
        case 1:
            return (jnp.asarray([0.5]), jnp.asarray([1.0]))
        case 2 | 3:
            return (
                jnp.asarray([0.21132486540518713, 0.7886751345948129]),
                jnp.asarray([0.5, 0.5]),
            )
        case 4 | 5:
            return (
                jnp.asarray([0.1127016653792583, 0.5, 0.8872983346207417]),
                jnp.asarray(
                    [0.2777777777777777, 0.44444444444444453, 0.2777777777777777]
                ),
            )
        case 6 | 7:
            return (
                jnp.asarray(
                    [
                        0.06943184420297377,
                        0.33000947820757187,
                        0.6699905217924281,
                        0.9305681557970262,
                    ]
                ),
                jnp.asarray(
                    [
                        0.1739274225687273,
                        0.3260725774312727,
                        0.3260725774312727,
                        0.1739274225687273,
                    ]
                ),
            )
        case 8 | 9:
            return (
                jnp.asarray(
                    [
                        0.046910077030668074,
                        0.2307653449471585,
                        0.5,
                        0.7692346550528415,
                        0.9530899229693319,
                    ]
                ),
                jnp.asarray(
                    [
                        0.11846344252809497,
                        0.23931433524968299,
                        0.28444444444444406,
                        0.23931433524968299,
                        0.11846344252809497,
                    ]
                ),
            )
        case 10 | 11:
            return (
                jnp.asarray(
                    [
                        0.033765242898423975,
                        0.16939530676686776,
                        0.3806904069584015,
                        0.6193095930415985,
                        0.8306046932331322,
                        0.966234757101576,
                    ]
                ),
                jnp.asarray(
                    [
                        0.08566224618958529,
                        0.18038078652406922,
                        0.2339569672863454,
                        0.2339569672863454,
                        0.18038078652406922,
                        0.08566224618958529,
                    ]
                ),
            )
        case 12 | 13:
            return (
                jnp.asarray(
                    [
                        0.025446043828620812,
                        0.12923440720030277,
                        0.29707742431130146,
                        0.5,
                        0.7029225756886985,
                        0.8707655927996972,
                        0.9745539561713792,
                    ]
                ),
                jnp.asarray(
                    [
                        0.06474248308443546,
                        0.13985269574463816,
                        0.1909150252525593,
                        0.20897959183673434,
                        0.1909150252525593,
                        0.13985269574463816,
                        0.06474248308443546,
                    ]
                ),
            )
        case 14 | 15:
            return (
                jnp.asarray(
                    [
                        0.019855071751231912,
                        0.10166676129318664,
                        0.2372337950418355,
                        0.40828267875217505,
                        0.591717321247825,
                        0.7627662049581645,
                        0.8983332387068134,
                        0.9801449282487681,
                    ]
                ),
                jnp.asarray(
                    [
                        0.05061426814518863,
                        0.11119051722668714,
                        0.15685332293894347,
                        0.18134189168918077,
                        0.18134189168918077,
                        0.15685332293894347,
                        0.11119051722668714,
                        0.05061426814518863,
                    ]
                ),
            )
        case 16 | 17:
            return (
                jnp.asarray(
                    [
                        0.015919880246186957,
                        0.08198444633668212,
                        0.19331428364970482,
                        0.3378732882980955,
                        0.5,
                        0.6621267117019045,
                        0.8066857163502952,
                        0.9180155536633179,
                        0.984080119753813,
                    ]
                ),
                jnp.asarray(
                    [
                        0.04063719418078741,
                        0.09032408034742868,
                        0.13030534820146766,
                        0.15617353852000132,
                        0.1651196775006298,
                        0.15617353852000132,
                        0.13030534820146766,
                        0.09032408034742868,
                        0.04063719418078741,
                    ]
                ),
            )
        case 18 | 19:
            return (
                jnp.asarray(
                    [
                        0.013046735741414128,
                        0.06746831665550768,
                        0.16029521585048778,
                        0.2833023029353764,
                        0.4255628305091844,
                        0.5744371694908156,
                        0.7166976970646236,
                        0.8397047841495122,
                        0.9325316833444923,
                        0.9869532642585859,
                    ]
                ),
                jnp.asarray(
                    [
                        0.03333567215434388,
                        0.07472567457528995,
                        0.10954318125799113,
                        0.13463335965499829,
                        0.14776211235737668,
                        0.14776211235737668,
                        0.13463335965499829,
                        0.10954318125799113,
                        0.07472567457528995,
                        0.03333567215434388,
                    ]
                ),
            )
        case 20 | 21:
            return (
                jnp.asarray(
                    [
                        0.010885670926971514,
                        0.05646870011595234,
                        0.13492399721297532,
                        0.2404519353965941,
                        0.36522842202382755,
                        0.50000000000000000000,
                        0.6347715779761725,
                        0.7595480646034058,
                        0.8650760027870247,
                        0.9435312998840477,
                        0.9891143290730284,
                    ]
                ),
                jnp.asarray(
                    [
                        0.027834283558086717,
                        0.06279018473245102,
                        0.09314510546386234,
                        0.11659688229599743,
                        0.13140227225512333,
                        0.13646254338895031536,
                        0.13140227225512333,
                        0.11659688229599743,
                        0.09314510546386234,
                        0.06279018473245102,
                        0.027834283558086717,
                    ]
                ),
            )
        case 22 | 23:
            return (
                jnp.asarray(
                    [
                        0.009219682876640378,
                        0.047941371814762546,
                        0.11504866290284765,
                        0.20634102285669126,
                        0.31608425050090994,
                        0.43738329574426554,
                        0.5626167042557344,
                        0.6839157494990901,
                        0.7936589771433087,
                        0.8849513370971523,
                        0.9520586281852375,
                        0.9907803171233596,
                    ]
                ),
                jnp.asarray(
                    [
                        0.023587668193255553,
                        0.053469662997670135,
                        0.08003916427166385,
                        0.10158371336153928,
                        0.11674626826917785,
                        0.12457352290670139,
                        0.12457352290670139,
                        0.11674626826917785,
                        0.10158371336153928,
                        0.08003916427166385,
                        0.053469662997670135,
                        0.023587668193255553,
                    ]
                ),
            )
        case 24 | 25:
            return (
                jnp.asarray(
                    [
                        0.007908472640705932,
                        0.04120080038851104,
                        0.09921095463334506,
                        0.17882533027982989,
                        0.27575362448177654,
                        0.3847708420224326,
                        0.50000000000000000000,
                        0.6152291579775674,
                        0.7242463755182235,
                        0.8211746697201701,
                        0.9007890453666549,
                        0.958799199611489,
                        0.9920915273592941,
                    ]
                ),
                jnp.asarray(
                    [
                        0.020242002382658532,
                        0.04606074991880237,
                        0.0694367551098878,
                        0.08907299038098014,
                        0.10390802376844523,
                        0.11314159013144862,
                        0.11627577661543695510,
                        0.11314159013144862,
                        0.10390802376844523,
                        0.08907299038098014,
                        0.0694367551098878,
                        0.04606074991880237,
                        0.020242002382658532,
                    ]
                ),
            )
        case 26 | 27:
            return (
                jnp.asarray(
                    [
                        0.006858095651593843,
                        0.03578255816821324,
                        0.08639934246511749,
                        0.15635354759415726,
                        0.24237568182092295,
                        0.3404438155360551,
                        0.44597252564632817,
                        0.5540274743536718,
                        0.6595561844639448,
                        0.757624318179077,
                        0.8436464524058427,
                        0.9136006575348825,
                        0.9642174418317868,
                        0.9931419043484062,
                    ]
                ),
                jnp.asarray(
                    [
                        0.01755973016589133,
                        0.040079043579841746,
                        0.060759285343926696,
                        0.07860158357908102,
                        0.09276919873897065,
                        0.1025992318606474,
                        0.10763192673157891,
                        0.10763192673157891,
                        0.1025992318606474,
                        0.09276919873897065,
                        0.07860158357908102,
                        0.060759285343926696,
                        0.040079043579841746,
                        0.01755973016589133,
                    ]
                ),
            )
        case 28 | 29:
            return (
                jnp.asarray(
                    [
                        0.006003740989757311,
                        0.031363303799647024,
                        0.0758967082947864,
                        0.13779113431991497,
                        0.21451391369573058,
                        0.3029243264612183,
                        0.39940295300128276,
                        0.50000000000000000000,
                        0.6005970469987173,
                        0.6970756735387817,
                        0.7854860863042694,
                        0.862208865680085,
                        0.9241032917052137,
                        0.968636696200353,
                        0.9939962590102427,
                    ]
                ),
                jnp.asarray(
                    [
                        0.015376620998021777,
                        0.03518302374405937,
                        0.053579610233607466,
                        0.06978533896314536,
                        0.08313460290850949,
                        0.09308050000778012,
                        0.09921574266355579,
                        0.10128912096278063644,
                        0.09921574266355579,
                        0.09308050000778012,
                        0.08313460290850949,
                        0.06978533896314536,
                        0.053579610233607466,
                        0.03518302374405937,
                        0.015376620998021777,
                    ]
                ),
            )
        case 30 | 31:
            return (
                jnp.asarray(
                    [
                        0.005299532504175031,
                        0.0277124884633837,
                        0.06718439880608412,
                        0.1222977958224985,
                        0.19106187779867811,
                        0.2709916111713863,
                        0.35919822461037054,
                        0.4524937450811813,
                        0.5475062549188188,
                        0.6408017753896295,
                        0.7290083888286136,
                        0.8089381222013219,
                        0.8777022041775016,
                        0.9328156011939159,
                        0.9722875115366163,
                        0.994700467495825,
                    ]
                ),
                jnp.asarray(
                    [
                        0.013576229706024828,
                        0.03112676196928269,
                        0.047579255841828275,
                        0.0623144856277606,
                        0.07479799440824579,
                        0.08457825969749927,
                        0.09130170752246201,
                        0.09472530522753422,
                        0.09472530522753422,
                        0.09130170752246201,
                        0.08457825969749927,
                        0.07479799440824579,
                        0.0623144856277606,
                        0.047579255841828275,
                        0.03112676196928269,
                        0.013576229706024828,
                    ]
                ),
            )
        case 32 | 33:
            return (
                jnp.asarray(
                    [
                        0.004712262342791318,
                        0.024662239115616102,
                        0.059880423136507044,
                        0.10924299805159932,
                        0.17116442039165464,
                        0.24365473145676153,
                        0.32438411827306185,
                        0.41075790925207606,
                        0.50000000000000000000,
                        0.5892420907479239,
                        0.6756158817269382,
                        0.7563452685432385,
                        0.8288355796083453,
                        0.8907570019484007,
                        0.9401195768634929,
                        0.9753377608843838,
                        0.9952877376572087,
                    ]
                ),
                jnp.asarray(
                    [
                        0.01207415143431202,
                        0.027729764686325483,
                        0.0425180741584208,
                        0.05594192359702783,
                        0.06756818423424478,
                        0.077022880538419,
                        0.08400205107822517,
                        0.08828135268349636,
                        0.089723235178103262729,
                        0.08828135268349636,
                        0.08400205107822517,
                        0.077022880538419,
                        0.06756818423424478,
                        0.05594192359702783,
                        0.0425180741584208,
                        0.027729764686325483,
                        0.01207415143431202,
                    ]
                ),
            )
        case 34 | 35:
            return (
                jnp.asarray(
                    [
                        0.004217415789534551,
                        0.022088025214301144,
                        0.05369876675122215,
                        0.09814752051373843,
                        0.1541564784698234,
                        0.22011458446302623,
                        0.2941244192685787,
                        0.37405688715424723,
                        0.45761249347913235,
                        0.5423875065208676,
                        0.6259431128457528,
                        0.7058755807314213,
                        0.7798854155369738,
                        0.8458435215301766,
                        0.9018524794862616,
                        0.9463012332487779,
                        0.9779119747856988,
                        0.9957825842104655,
                    ]
                ),
                jnp.asarray(
                    [
                        0.010808006763319273,
                        0.024857274446330582,
                        0.038212865130172066,
                        0.050471022053329796,
                        0.061277603355901905,
                        0.07032145733530425,
                        0.07734233756313325,
                        0.08213824187291588,
                        0.08457119148157177,
                        0.08457119148157177,
                        0.08213824187291588,
                        0.07734233756313325,
                        0.07032145733530425,
                        0.061277603355901905,
                        0.050471022053329796,
                        0.038212865130172066,
                        0.024857274446330582,
                        0.010808006763319273,
                    ]
                ),
            )
        case 36 | 37:
            return (
                jnp.asarray(
                    [
                        0.003796578078207824,
                        0.01989592393258499,
                        0.04842204819259105,
                        0.08864267173142859,
                        0.1395169113323853,
                        0.1997273476691595,
                        0.2677146293120195,
                        0.34171795001818506,
                        0.4198206771798873,
                        0.50000000000000000000,
                        0.5801793228201126,
                        0.6582820499818149,
                        0.7322853706879805,
                        0.8002726523308406,
                        0.8604830886676147,
                        0.9113573282685714,
                        0.951577951807409,
                        0.980104076067415,
                        0.9962034219217921,
                    ]
                ),
                jnp.asarray(
                    [
                        0.00973089411478413,
                        0.022407113383627688,
                        0.03452227136614014,
                        0.04574501080956803,
                        0.05578332277478876,
                        0.06437698126939818,
                        0.07130335108681893,
                        0.07638302103293432,
                        0.07948442169697709,
                        0.080527224924391847990,
                        0.07948442169697709,
                        0.07638302103293432,
                        0.07130335108681893,
                        0.06437698126939818,
                        0.05578332277478876,
                        0.04574501080956803,
                        0.03452227136614014,
                        0.022407113383627688,
                        0.00973089411478413,
                    ]
                ),
            )
        case 38 | 39:
            return (
                jnp.asarray(
                    [
                        0.0034357004074525577,
                        0.018014036361043095,
                        0.04388278587433703,
                        0.08044151408889061,
                        0.1268340467699246,
                        0.1819731596367425,
                        0.24456649902458644,
                        0.3131469556422902,
                        0.38610707442917747,
                        0.46173673943325133,
                        0.5382632605667487,
                        0.6138929255708225,
                        0.6868530443577098,
                        0.7554335009754136,
                        0.8180268403632576,
                        0.8731659532300754,
                        0.9195584859111094,
                        0.956117214125663,
                        0.981985963638957,
                        0.9965642995925474,
                    ]
                ),
                jnp.asarray(
                    [
                        0.008807003569250137,
                        0.02030071490320808,
                        0.031336024163181396,
                        0.04163837077894665,
                        0.050965059909388766,
                        0.05909726598040116,
                        0.06584431922464036,
                        0.0710480546591908,
                        0.0745864932363021,
                        0.07637669356536296,
                        0.07637669356536296,
                        0.0745864932363021,
                        0.0710480546591908,
                        0.06584431922464036,
                        0.05909726598040116,
                        0.050965059909388766,
                        0.04163837077894665,
                        0.031336024163181396,
                        0.02030071490320808,
                        0.008807003569250137,
                    ]
                ),
            )
        case 40 | 41:
            return (
                jnp.asarray(
                    [
                        0.003123914689805274,
                        0.016386580716846844,
                        0.0399503329247996,
                        0.07331831770834135,
                        0.11578001826216106,
                        0.16643059790129383,
                        0.2241905820563901,
                        0.2878289398962806,
                        0.35598934159879947,
                        0.42721907291955247,
                        0.50000000000000000000,
                        0.5727809270804476,
                        0.6440106584012005,
                        0.7121710601037194,
                        0.7758094179436099,
                        0.8335694020987061,
                        0.8842199817378389,
                        0.9266816822916586,
                        0.9600496670752003,
                        0.9836134192831532,
                        0.9968760853101948,
                    ]
                ),
                jnp.asarray(
                    [
                        0.00800861413046309,
                        0.018476894874559474,
                        0.028567212727933538,
                        0.03805005679723173,
                        0.046722211724564325,
                        0.05439864958576761,
                        0.060915708026943044,
                        0.06613446931663151,
                        0.06994369739553497,
                        0.07226220199498506,
                        0.073040566824845213596,
                        0.07226220199498506,
                        0.06994369739553497,
                        0.06613446931663151,
                        0.060915708026943044,
                        0.05439864958576761,
                        0.046722211724564325,
                        0.03805005679723173,
                        0.028567212727933538,
                        0.018476894874559474,
                        0.00800861413046309,
                    ]
                ),
            )
        case _:
            assert False, "Quadrature order not implemented"


def gauss_legendre_nd(dimension, order):
    """
    Returns positions and weights for Gauss-Legendre quadrature in higher dimensions based on tensor product rules

    - Interval [-1, 1]^n_dim
    """

    def gauss_legendre_1d_scaled(order):
        position, weights = gauss_legendre_1d(order)
        scaled_weights = 2 * weights
        scaled_positions = 2 * position - 1
        return scaled_positions, scaled_weights

    if dimension == 1:
        return gauss_legendre_1d_scaled(order)
    else:
        return tensor_product_rule(*gauss_legendre_1d_scaled(order), dimension)


def gauss_lobatto_1d(order):
    """
    Returns positions and weights for Gauss-Lobatto quadrature for the interval [0,1]

    - Interval [0, 1]
    - accurate for polynomials up to order
    """

    match order:
        case 1:
            return (jnp.asarray([0.0, 1.0]), jnp.asarray([0.5, 0.5]))
        case 2 | 3:
            return (
                jnp.asarray([0.0, 0.5, 1.0]),
                jnp.asarray(
                    [
                        0.16666666666666666667,
                        0.66666666666666666667,
                        0.16666666666666666667,
                    ]
                ),
            )
        case 4 | 5:
            return (
                jnp.asarray(
                    [
                        0.0,
                        0.27639320225002106,
                        0.7236067977499789,
                        1.0000000000000000000,
                    ]
                ),
                jnp.asarray(
                    [
                        0.083333333333333333333,
                        0.41666666666666663,
                        0.41666666666666663,
                        0.083333333333333333333,
                    ]
                ),
            )
        case 6 | 7:
            return (
                jnp.asarray(
                    [
                        0.0,
                        0.1726731646460114,
                        0.50000000000000000000,
                        0.8273268353539887,
                        1.0000000000000000000,
                    ]
                ),
                jnp.asarray(
                    [
                        0.050000000000000000000,
                        0.27222222222222225,
                        0.35555555555555555556,
                        0.27222222222222225,
                        0.050000000000000000000,
                    ]
                ),
            )
        case 8 | 9:
            return (
                jnp.asarray(
                    [
                        0.0,
                        0.11747233803526763,
                        0.3573842417596774,
                        0.6426157582403226,
                        0.8825276619647324,
                        1.0000000000000000000,
                    ]
                ),
                jnp.asarray(
                    [
                        0.033333333333333333333,
                        0.18923747814892347,
                        0.2774291885177431,
                        0.2774291885177431,
                        0.18923747814892347,
                        0.033333333333333333333,
                    ]
                ),
            )
        case 10 | 11:
            return (
                jnp.asarray(
                    [
                        0.0,
                        0.08488805186071652,
                        0.2655756032646429,
                        0.50000000000000000000,
                        0.7344243967353571,
                        0.9151119481392835,
                        1.0000000000000000000,
                    ]
                ),
                jnp.asarray(
                    [
                        0.023809523809523809524,
                        0.13841302368078293,
                        0.21587269060493122,
                        0.24380952380952380952,
                        0.21587269060493122,
                        0.13841302368078293,
                        0.023809523809523809524,
                    ]
                ),
            )
        case 12 | 13:
            return (
                jnp.asarray(
                    [
                        0.0,
                        0.06412992574519671,
                        0.20414990928342885,
                        0.3953503910487606,
                        0.6046496089512394,
                        0.7958500907165711,
                        0.9358700742548033,
                        1.0000000000000000000,
                    ]
                ),
                jnp.asarray(
                    [
                        0.017857142857142857143,
                        0.10535211357175307,
                        0.17056134624175218,
                        0.20622939732935183,
                        0.20622939732935183,
                        0.17056134624175218,
                        0.10535211357175307,
                        0.017857142857142857143,
                    ]
                ),
            )
        case 14 | 15:
            return (
                jnp.asarray(
                    [
                        0.0,
                        0.05012100229426991,
                        0.16140686024463113,
                        0.3184412680869109,
                        0.50000000000000000000,
                        0.6815587319130891,
                        0.8385931397553689,
                        0.94987899770573,
                        1.0000000000000000000,
                    ]
                ),
                jnp.asarray(
                    [
                        0.013888888888888888889,
                        0.0827476807804028,
                        0.1372693562500807,
                        0.17321425548652308,
                        0.18575963718820861678,
                        0.17321425548652308,
                        0.1372693562500807,
                        0.0827476807804028,
                        0.013888888888888888889,
                    ]
                ),
            )
        case 16 | 17:
            return (
                jnp.asarray(
                    [
                        0.0,
                        0.04023304591677057,
                        0.1306130674472475,
                        0.26103752509477773,
                        0.4173605211668065,
                        0.5826394788331936,
                        0.7389624749052223,
                        0.8693869325527526,
                        0.9597669540832294,
                        1.0000000000000000000,
                    ]
                ),
                jnp.asarray(
                    [
                        0.011111111111111111111,
                        0.06665299542553504,
                        0.11244467103156322,
                        0.1460213418398419,
                        0.16376988059194872,
                        0.16376988059194872,
                        0.1460213418398419,
                        0.11244467103156322,
                        0.06665299542553504,
                        0.011111111111111111111,
                    ]
                ),
            )
        case 18 | 19:
            return (
                jnp.asarray(
                    [
                        0.0,
                        0.03299928479597042,
                        0.10775826316842779,
                        0.21738233650189748,
                        0.3521209322065303,
                        0.50000000000000000000,
                        0.6478790677934697,
                        0.7826176634981026,
                        0.8922417368315723,
                        0.9670007152040296,
                        1.0000000000000000000,
                    ]
                ),
                jnp.asarray(
                    [
                        0.0090909090909090909091,
                        0.05480613663349739,
                        0.0935849408901527,
                        0.12402405213201415,
                        0.14343956238950406,
                        0.15010879772784534689,
                        0.14343956238950406,
                        0.12402405213201415,
                        0.0935849408901527,
                        0.05480613663349739,
                        0.0090909090909090909091,
                    ]
                ),
            )
        case 20 | 21:
            return (
                jnp.asarray(
                    [
                        0.0,
                        0.027550363888558915,
                        0.09036033917799668,
                        0.18356192348406963,
                        0.30023452951732554,
                        0.43172353357253623,
                        0.5682764664274638,
                        0.6997654704826745,
                        0.8164380765159304,
                        0.9096396608220033,
                        0.9724496361114411,
                        1.0000000000000000000,
                    ]
                ),
                jnp.asarray(
                    [
                        0.0075757575757575757576,
                        0.04584225870659809,
                        0.07898735278218508,
                        0.10625420888051049,
                        0.12563780159960058,
                        0.13570262045534812,
                        0.13570262045534812,
                        0.12563780159960058,
                        0.10625420888051049,
                        0.07898735278218508,
                        0.04584225870659809,
                        0.0075757575757575757576,
                    ]
                ),
            )
        case 22 | 23:
            return (
                jnp.asarray(
                    [
                        0.0,
                        0.023345076678918053,
                        0.07682621767406383,
                        0.15690576545912127,
                        0.2585450894543319,
                        0.37535653494688004,
                        0.50000000000000000000,
                        0.62464346505312,
                        0.741454910545668,
                        0.8430942345408787,
                        0.9231737823259362,
                        0.976654923321082,
                        1.0000000000000000000,
                    ]
                ),
                jnp.asarray(
                    [
                        0.0064102564102564102564,
                        0.038900843373409474,
                        0.0674909633448042,
                        0.09182343260177493,
                        0.11038389678305509,
                        0.12200789515333813,
                        0.12596542466672336802,
                        0.12200789515333813,
                        0.11038389678305509,
                        0.09182343260177493,
                        0.0674909633448042,
                        0.038900843373409474,
                        0.0064102564102564102564,
                    ]
                ),
            )
        case 24 | 25:
            return (
                jnp.asarray(
                    [
                        0.0,
                        0.02003247736636954,
                        0.06609947308482639,
                        0.1355657004543369,
                        0.22468029853567645,
                        0.3286379933286436,
                        0.4418340655581481,
                        0.5581659344418519,
                        0.6713620066713564,
                        0.7753197014643236,
                        0.8644342995456631,
                        0.9339005269151737,
                        0.9799675226336304,
                        1.0000000000000000000,
                    ]
                ),
                jnp.asarray(
                    [
                        0.0054945054945054945055,
                        0.03341864224884061,
                        0.05829332794935596,
                        0.08001092588147597,
                        0.09741307468670805,
                        0.10956312650488534,
                        0.11580639723422842,
                        0.11580639723422842,
                        0.10956312650488534,
                        0.09741307468670805,
                        0.08001092588147597,
                        0.05829332794935596,
                        0.03341864224884061,
                        0.0054945054945054945055,
                    ]
                ),
            )
        case 26 | 27:
            return (
                jnp.asarray(
                    [
                        0.0,
                        0.01737703674808072,
                        0.05745897788851184,
                        0.11824015502409241,
                        0.19687339726507713,
                        0.2896809726431637,
                        0.3923230223181029,
                        0.50000000000000000000,
                        0.6076769776818971,
                        0.7103190273568363,
                        0.8031266027349229,
                        0.8817598449759076,
                        0.9425410221114882,
                        0.9826229632519192,
                        1.0000000000000000000,
                    ]
                ),
                jnp.asarray(
                    [
                        0.0047619047619047619048,
                        0.029014946514300685,
                        0.0508300351628591,
                        0.07025584990121399,
                        0.08639482362680036,
                        0.09849361798230656,
                        0.10598679296341042,
                        0.10852405817440782476,
                        0.10598679296341042,
                        0.09849361798230656,
                        0.08639482362680036,
                        0.07025584990121399,
                        0.0508300351628591,
                        0.029014946514300685,
                        0.0047619047619047619048,
                    ]
                ),
            )
        case 28 | 29:
            return (
                jnp.asarray(
                    [
                        0.0,
                        0.015215976864891012,
                        0.05039973345326393,
                        0.10399585406909245,
                        0.17380564855875347,
                        0.25697028905643116,
                        0.3500847655496184,
                        0.4493368632390253,
                        0.5506631367609747,
                        0.6499152344503816,
                        0.7430297109435688,
                        0.8261943514412465,
                        0.8960041459309076,
                        0.949600266546736,
                        0.984784023135109,
                        1.0000000000000000000,
                    ]
                ),
                jnp.asarray(
                    [
                        0.0041666666666666666667,
                        0.025425180502959957,
                        0.04469684866296535,
                        0.06212769106625707,
                        0.07701349040358203,
                        0.08874595669585207,
                        0.09684501191260185,
                        0.10097915408911484,
                        0.10097915408911484,
                        0.09684501191260185,
                        0.08874595669585207,
                        0.07701349040358203,
                        0.06212769106625707,
                        0.04469684866296535,
                        0.025425180502959957,
                        0.0041666666666666666667,
                    ]
                ),
            )
        case 30 | 31:
            return (
                jnp.asarray(
                    [
                        0.0,
                        0.013433911684290867,
                        0.04456000204221322,
                        0.09215187438911487,
                        0.15448550968615765,
                        0.22930730033494923,
                        0.31391278321726146,
                        0.4052440132408413,
                        0.50000000000000000000,
                        0.5947559867591588,
                        0.6860872167827385,
                        0.7706926996650507,
                        0.8455144903138423,
                        0.9078481256108851,
                        0.9554399979577868,
                        0.9865660883157091,
                        1.0000000000000000000,
                    ]
                ),
                jnp.asarray(
                    [
                        0.0036764705882352941176,
                        0.02246097027162709,
                        0.039599135251843595,
                        0.05529645450351409,
                        0.0689938731009633,
                        0.08019733099881075,
                        0.08850212675782891,
                        0.09360816983880964,
                        0.095330937376734716650,
                        0.09360816983880964,
                        0.08850212675782891,
                        0.08019733099881075,
                        0.0689938731009633,
                        0.05529645450351409,
                        0.039599135251843595,
                        0.02246097027162709,
                        0.0036764705882352941176,
                    ]
                ),
            )
        case 32 | 33:
            return (
                jnp.asarray(
                    [
                        0.0,
                        0.011947221293900745,
                        0.039675407326233036,
                        0.0822032323909549,
                        0.13816033535837868,
                        0.20574758284066913,
                        0.282792481543938,
                        0.3668186735608595,
                        0.45512545325767395,
                        0.544874546742326,
                        0.6331813264391405,
                        0.717207518456062,
                        0.7942524171593308,
                        0.8618396646416213,
                        0.917796767609045,
                        0.9603245926737669,
                        0.9880527787060993,
                        1.0000000000000000000,
                    ]
                ),
                jnp.asarray(
                    [
                        0.0032679738562091503268,
                        0.019985314405457075,
                        0.03531858344281686,
                        0.04950813585875143,
                        0.06210526656648344,
                        0.07270598078690102,
                        0.08096975861880114,
                        0.08663105474472804,
                        0.08950793171985152,
                        0.08950793171985152,
                        0.08663105474472804,
                        0.08096975861880114,
                        0.07270598078690102,
                        0.06210526656648344,
                        0.04950813585875143,
                        0.03531858344281686,
                        0.019985314405457075,
                        0.0032679738562091503268,
                    ]
                ),
            )
        case 34 | 35:
            return (
                jnp.asarray(
                    [
                        0.0,
                        0.010694116888959937,
                        0.0355492359237069,
                        0.07376971110167696,
                        0.1242528987236935,
                        0.18554593136738973,
                        0.2558853571596432,
                        0.3332475760877507,
                        0.4154069882953592,
                        0.50000000000000000000,
                        0.5845930117046407,
                        0.6667524239122493,
                        0.7441146428403568,
                        0.8144540686326103,
                        0.8757471012763065,
                        0.926230288898323,
                        0.9644507640762932,
                        0.9893058831110401,
                        1.0000000000000000000,
                    ]
                ),
                jnp.asarray(
                    [
                        0.0029239766081871345029,
                        0.017896682593088343,
                        0.03169094588131481,
                        0.04456587854960347,
                        0.05615767073865242,
                        0.06613364022437529,
                        0.07420697129796938,
                        0.08014546202203052,
                        0.08377829226357139,
                        0.085000959642413617322,
                        0.08377829226357139,
                        0.08014546202203052,
                        0.07420697129796938,
                        0.06613364022437529,
                        0.05615767073865242,
                        0.04456587854960347,
                        0.03169094588131481,
                        0.017896682593088343,
                        0.0029239766081871345029,
                    ]
                ),
            )
        case 36 | 37:
            return (
                jnp.asarray(
                    [
                        0.0,
                        0.00962814755304292,
                        0.03203275059366728,
                        0.06656101095502492,
                        0.11231586952397205,
                        0.16811179885484434,
                        0.23250356798405686,
                        0.3038234081430453,
                        0.38022414703850677,
                        0.45972703138058907,
                        0.5402729686194109,
                        0.6197758529614933,
                        0.6961765918569547,
                        0.7674964320159432,
                        0.8318882011451556,
                        0.8876841304760279,
                        0.933438989044975,
                        0.9679672494063327,
                        0.9903718524469571,
                        1.0000000000000000000,
                    ]
                ),
                jnp.asarray(
                    [
                        0.0026315789473684210526,
                        0.0161185615942445,
                        0.028590901063783362,
                        0.04031588199805985,
                        0.05099574984972543,
                        0.06035461381433739,
                        0.06815024117936207,
                        0.07418077703545842,
                        0.07829005132373776,
                        0.08037164319392284,
                        0.08037164319392284,
                        0.07829005132373776,
                        0.07418077703545842,
                        0.06815024117936207,
                        0.06035461381433739,
                        0.05099574984972543,
                        0.04031588199805985,
                        0.028590901063783362,
                        0.0161185615942445,
                        0.0026315789473684210526,
                    ]
                ),
            )
        case 38 | 39:
            return (
                jnp.asarray(
                    [
                        0.0,
                        0.008713851697725983,
                        0.029011851520127252,
                        0.060352622338204764,
                        0.10199903696114382,
                        0.15297448696888838,
                        0.21208401986908465,
                        0.27794210836049893,
                        0.34900507174561757,
                        0.42360724209890727,
                        0.50000000000000000000,
                        0.5763927579010928,
                        0.6509949282543824,
                        0.7220578916395011,
                        0.7879159801309153,
                        0.8470255130311116,
                        0.8980009630388561,
                        0.9396473776617953,
                        0.9709881484798728,
                        0.991286148302274,
                        1.0000000000000000000,
                    ]
                ),
                jnp.asarray(
                    [
                        0.0023809523809523809524,
                        0.01459242004925281,
                        0.025921584500424793,
                        0.03663695909253707,
                        0.04649273397894307,
                        0.05525854160956163,
                        0.06272906059543436,
                        0.06872923143002062,
                        0.07311843122398869,
                        0.07579378755584067,
                        0.076692595166087474276,
                        0.07579378755584067,
                        0.07311843122398869,
                        0.06872923143002062,
                        0.06272906059543436,
                        0.05525854160956163,
                        0.04649273397894307,
                        0.03663695909253707,
                        0.025921584500424793,
                        0.01459242004925281,
                        0.0023809523809523809524,
                    ]
                ),
            )
        case 40 | 41:
            return (
                jnp.asarray(
                    [
                        0.0,
                        0.007923780771176892,
                        0.026397858000385632,
                        0.05496885490454778,
                        0.09302553619403942,
                        0.13975638001939894,
                        0.1941652808578705,
                        0.25509256240504885,
                        0.32123964493054025,
                        0.3911967074203575,
                        0.4634727299945508,
                        0.5365272700054492,
                        0.6088032925796425,
                        0.6787603550694598,
                        0.7449074375949511,
                        0.8058347191421296,
                        0.8602436199806011,
                        0.9069744638059606,
                        0.9450311450954523,
                        0.9736021419996144,
                        0.992076219228823,
                        1.0000000000000000000,
                    ]
                ),
                jnp.asarray(
                    [
                        0.0021645021645021645022,
                        0.013272873841250856,
                        0.023607232646870323,
                        0.03343280293227656,
                        0.04254503019591922,
                        0.05075028740082377,
                        0.05787382232696945,
                        0.06376384832671507,
                        0.06829484430687072,
                        0.07137024613568065,
                        0.07292450972212082,
                        0.07292450972212082,
                        0.07137024613568065,
                        0.06829484430687072,
                        0.06376384832671507,
                        0.05787382232696945,
                        0.05075028740082377,
                        0.04254503019591922,
                        0.03343280293227656,
                        0.023607232646870323,
                        0.013272873841250856,
                        0.0021645021645021645022,
                    ]
                ),
            )
        case _:
            assert False, "Quadrature order not implemented"


def gauss_lobatto_nd(dimension, order):
    """
    Returns positions and weights for Gauss-Lobatto quadrature in higher dimensions based on tensor product rules

    - Interval [-1, 1]^n_dim
    """

    def gauss_lobatto_1d_scaled(order):
        position, weights = gauss_lobatto_1d(order)
        scaled_weights = 2 * weights
        scaled_positions = 2 * position - 1
        return scaled_positions, scaled_weights

    if dimension == 1:
        return gauss_lobatto_1d_scaled(order)
    else:
        return tensor_product_rule(*gauss_lobatto_1d_scaled(order), dimension)


def int_pts_ref_tri(order):
    """
    Returns tuple with integration point coordinates and weights on a reference triangle, accurate up to polynomial order

    Rules from: https://mathsfromnothing.au/triangle-quadrature-rules/
    """
    match order:
        case 1:
            return (jnp.asarray([[1 / 3, 1 / 3]]), jnp.asarray([1 / 2]))
        case 2:
            return (
                jnp.asarray(
                    [
                        [0.166666666666667, 0.166666666666667, 0.666666666666667],
                        [0.666666666666667, 0.166666666666667, 0.166666666666667],
                    ]
                ).transpose(),
                (1 / 2)
                * jnp.asarray(
                    [0.333333333333333, 0.333333333333333, 0.333333333333333]
                ),
            )
        case 3:
            return (
                jnp.asarray(
                    [
                        [
                            0.445948490915965,
                            0.445948490915965,
                            0.108103018168070,
                            0.091576213509771,
                            0.091576213509771,
                            0.816847572980459,
                        ],
                        [
                            0.108103018168070,
                            0.445948490915965,
                            0.445948490915965,
                            0.816847572980459,
                            0.091576213509771,
                            0.091576213509771,
                        ],
                    ]
                ).transpose(),
                (1 / 2)
                * jnp.asarray(
                    [
                        0.223381589678011,
                        0.223381589678011,
                        0.223381589678011,
                        0.109951743655322,
                        0.109951743655322,
                        0.109951743655322,
                    ]
                ),
            )
        case 4:
            return (
                jnp.asarray(
                    [
                        [
                            0.445948490915965,
                            0.445948490915965,
                            0.108103018168070,
                            0.091576213509771,
                            0.091576213509771,
                            0.816847572980459,
                        ],
                        [
                            0.108103018168070,
                            0.445948490915965,
                            0.445948490915965,
                            0.816847572980459,
                            0.091576213509771,
                            0.091576213509771,
                        ],
                    ]
                ).transpose(),
                (1 / 2)
                * jnp.asarray(
                    [
                        0.223381589678011,
                        0.223381589678011,
                        0.223381589678011,
                        0.109951743655322,
                        0.109951743655322,
                        0.109951743655322,
                    ]
                ),
            )
        case 5:
            return (
                jnp.asarray(
                    [
                        [
                            0.333333333333333,
                            0.470142064105115,
                            0.470142064105115,
                            0.059715871789770,
                            0.101286507323456,
                            0.101286507323456,
                            0.797426985353087,
                        ],
                        [
                            0.333333333333333,
                            0.059715871789770,
                            0.470142064105115,
                            0.470142064105115,
                            0.797426985353087,
                            0.101286507323456,
                            0.101286507323456,
                        ],
                    ]
                ).transpose(),
                (1 / 2)
                * jnp.asarray(
                    [
                        0.225,
                        0.132394152788506,
                        0.132394152788506,
                        0.132394152788506,
                        0.125939180544827,
                        0.125939180544827,
                        0.125939180544827,
                    ]
                ),
            )
        case 6:
            return (
                jnp.asarray(
                    [
                        [
                            0.063089014491502,
                            0.063089014491502,
                            0.873821971016996,
                            0.053145049844817,
                            0.310352451033784,
                            0.636502499121399,
                            0.310352451033784,
                            0.053145049844817,
                            0.636502499121399,
                            0.249286745170910,
                            0.249286745170910,
                            0.501426509658179,
                        ],
                        [
                            0.873821971016996,
                            0.063089014491502,
                            0.063089014491502,
                            0.636502499121399,
                            0.053145049844817,
                            0.310352451033784,
                            0.636502499121399,
                            0.310352451033784,
                            0.053145049844817,
                            0.501426509658179,
                            0.249286745170910,
                            0.249286745170910,
                        ],
                    ]
                ).transpose(),
                (1 / 2)
                * jnp.asarray(
                    [
                        0.050844906370207,
                        0.050844906370207,
                        0.050844906370207,
                        0.082851075618374,
                        0.082851075618374,
                        0.082851075618374,
                        0.082851075618374,
                        0.082851075618374,
                        0.082851075618374,
                        0.116786275726379,
                        0.116786275726379,
                        0.116786275726379,
                    ]
                ),
            )
        case 7:
            return (
                jnp.asarray(
                    [
                        [
                            0.333333333333333,
                            0.459292588292723,
                            0.459292588292723,
                            0.081414823414554,
                            0.170569307751760,
                            0.170569307751760,
                            0.658861384496480,
                            0.008394777409958,
                            0.263112829634638,
                            0.728492392955404,
                            0.263112829634638,
                            0.008394777409958,
                            0.728492392955404,
                            0.050547228317031,
                            0.050547228317031,
                            0.898905543365938,
                        ],
                        [
                            0.333333333333333,
                            0.081414823414554,
                            0.459292588292723,
                            0.459292588292723,
                            0.658861384496480,
                            0.170569307751760,
                            0.170569307751760,
                            0.728492392955404,
                            0.008394777409958,
                            0.263112829634638,
                            0.728492392955404,
                            0.263112829634638,
                            0.008394777409958,
                            0.898905543365938,
                            0.050547228317031,
                            0.050547228317031,
                        ],
                    ]
                ).transpose(),
                (1 / 2)
                * jnp.asarray(
                    [
                        0.144315607677787,
                        0.095091634267285,
                        0.095091634267285,
                        0.095091634267285,
                        0.103217370534718,
                        0.103217370534718,
                        0.103217370534718,
                        0.027230314174435,
                        0.027230314174435,
                        0.027230314174435,
                        0.027230314174435,
                        0.027230314174435,
                        0.027230314174435,
                        0.032458497623198,
                        0.032458497623198,
                        0.032458497623198,
                    ]
                ),
            )
        case 8:
            return (
                jnp.asarray(
                    [
                        [
                            0.333333333333333,
                            0.459292588292723,
                            0.459292588292723,
                            0.081414823414554,
                            0.170569307751760,
                            0.170569307751760,
                            0.658861384496480,
                            0.008394777409958,
                            0.263112829634638,
                            0.728492392955404,
                            0.263112829634638,
                            0.008394777409958,
                            0.728492392955404,
                            0.050547228317031,
                            0.050547228317031,
                            0.898905543365938,
                        ],
                        [
                            0.333333333333333,
                            0.081414823414554,
                            0.459292588292723,
                            0.459292588292723,
                            0.658861384496480,
                            0.170569307751760,
                            0.170569307751760,
                            0.728492392955404,
                            0.008394777409958,
                            0.263112829634638,
                            0.728492392955404,
                            0.263112829634638,
                            0.008394777409958,
                            0.898905543365938,
                            0.050547228317031,
                            0.050547228317031,
                        ],
                    ]
                ).transpose(),
                (1 / 2)
                * jnp.asarray(
                    [
                        0.144315607677787,
                        0.095091634267285,
                        0.095091634267285,
                        0.095091634267285,
                        0.103217370534718,
                        0.103217370534718,
                        0.103217370534718,
                        0.027230314174435,
                        0.027230314174435,
                        0.027230314174435,
                        0.027230314174435,
                        0.027230314174435,
                        0.027230314174435,
                        0.032458497623198,
                        0.032458497623198,
                        0.032458497623198,
                    ]
                ),
            )
        case 9:
            return (
                jnp.asarray(
                    [
                        [
                            0.333333333333333,
                            0.489682519198738,
                            0.489682519198738,
                            0.020634961602525,
                            0.437089591492937,
                            0.437089591492937,
                            0.125820817014127,
                            0.188203535619033,
                            0.188203535619033,
                            0.623592928761935,
                            0.036838412054736,
                            0.221962989160766,
                            0.741198598784498,
                            0.221962989160766,
                            0.036838412054736,
                            0.741198598784498,
                            0.044729513394453,
                            0.044729513394453,
                            0.910540973211095,
                        ],
                        [
                            0.333333333333333,
                            0.020634961602525,
                            0.489682519198738,
                            0.489682519198738,
                            0.125820817014127,
                            0.437089591492937,
                            0.437089591492937,
                            0.623592928761935,
                            0.188203535619033,
                            0.188203535619033,
                            0.741198598784498,
                            0.036838412054736,
                            0.221962989160766,
                            0.741198598784498,
                            0.221962989160766,
                            0.036838412054736,
                            0.910540973211095,
                            0.044729513394453,
                            0.044729513394453,
                        ],
                    ]
                ).transpose(),
                (1 / 2)
                * jnp.asarray(
                    [
                        0.097135796282799,
                        0.031334700227139,
                        0.031334700227139,
                        0.031334700227139,
                        0.077827541004774,
                        0.077827541004774,
                        0.077827541004774,
                        0.079647738927210,
                        0.079647738927210,
                        0.079647738927210,
                        0.043283539377289,
                        0.043283539377289,
                        0.043283539377289,
                        0.043283539377289,
                        0.043283539377289,
                        0.043283539377289,
                        0.025577675658698,
                        0.025577675658698,
                        0.025577675658698,
                    ]
                ),
            )
        case 10:
            return (
                jnp.asarray(
                    [
                        [
                            0.333333333333333,
                            0.485577633383657,
                            0.485577633383657,
                            0.028844733232685,
                            0.141707219414880,
                            0.307939838764121,
                            0.550352941820999,
                            0.307939838764121,
                            0.141707219414880,
                            0.550352941820999,
                            0.025003534762686,
                            0.246672560639903,
                            0.728323904597411,
                            0.246672560639903,
                            0.025003534762686,
                            0.728323904597411,
                            0.009540815400299,
                            0.066803251012200,
                            0.923655933587500,
                            0.066803251012200,
                            0.009540815400299,
                            0.923655933587500,
                            0.109481575485037,
                            0.109481575485037,
                            0.781036849029926,
                        ],
                        [
                            0.333333333333333,
                            0.028844733232685,
                            0.485577633383657,
                            0.485577633383657,
                            0.550352941820999,
                            0.141707219414880,
                            0.307939838764121,
                            0.550352941820999,
                            0.307939838764121,
                            0.141707219414880,
                            0.728323904597411,
                            0.025003534762686,
                            0.246672560639903,
                            0.728323904597411,
                            0.246672560639903,
                            0.025003534762686,
                            0.923655933587500,
                            0.009540815400299,
                            0.066803251012200,
                            0.923655933587500,
                            0.066803251012200,
                            0.009540815400299,
                            0.781036849029926,
                            0.109481575485037,
                            0.109481575485037,
                        ],
                    ]
                ).transpose(),
                (1 / 2)
                * jnp.asarray(
                    [
                        0.090817990382754,
                        0.036725957756467,
                        0.036725957756467,
                        0.036725957756467,
                        0.072757916845420,
                        0.072757916845420,
                        0.072757916845420,
                        0.072757916845420,
                        0.072757916845420,
                        0.072757916845420,
                        0.028327242531057,
                        0.028327242531057,
                        0.028327242531057,
                        0.028327242531057,
                        0.028327242531057,
                        0.028327242531057,
                        0.009421666963733,
                        0.009421666963733,
                        0.009421666963733,
                        0.009421666963733,
                        0.009421666963733,
                        0.009421666963733,
                        0.045321059435528,
                        0.045321059435528,
                        0.045321059435528,
                    ]
                ),
            )
        case _:
            assert False, "Integration rule not implemented!"


def int_pts_ref_tet(order):
    """
    Returns tuple with integration point coordinates and weights on a reference tetrahedron accurate up to polynomial order

    Rules from Jaśkowiec and Sukumar (2020):  https://doi.org/10.1002/nme.6313
    """
    match order:
        case 1:
            tmp = jnp.asarray(
                [
                    [
                        1 / 4,
                        1 / 4,
                        1 / 4,
                        1.0,
                    ]
                ]
            )
        case 2:
            tmp = jnp.asarray(
                [
                    0.1285157070717654,
                    0.1395716909679451,
                    0.6217058655734218,
                    0.2130740063197066,
                    0.6080544789917290,
                    0.1373527300771633,
                    0.1458986947266748,
                    0.2255331922680785,
                    0.1617347277547459,
                    0.1593401922468276,
                    0.1606122963421696,
                    0.3424848235779030,
                    0.1374481016885109,
                    0.6153805112897389,
                    0.1353005423383199,
                    0.2189079778343118,
                ]
            )
            tmp = tmp.reshape((4, 4))
        case 3:
            tmp = jnp.asarray(
                [
                    0.0701026973651683,
                    0.1666606997304260,
                    0.5965757414636450,
                    0.1803964076327240,
                    0.1686953714777434,
                    0.5846526266988473,
                    0.1686952187085394,
                    0.1933162942936617,
                    0.4181757892411656,
                    0.0803037256707292,
                    0.4181013481561183,
                    0.1335947988703586,
                    0.5965762888488200,
                    0.1666726351059825,
                    0.0700786026978241,
                    0.1803710791466976,
                    0.1648085473528479,
                    0.0851134715186107,
                    0.1648086847553500,
                    0.2010167209445907,
                    0.0731475667668616,
                    0.4403391369182129,
                    0.0731570614497775,
                    0.1113046991119671,
                ]
            )
            tmp = tmp.reshape((6, 4))
        case 4:
            tmp = jnp.asarray(
                [
                    0.1187280740805765,
                    0.4881393122183348,
                    0.3108120274441792,
                    0.1418665725465847,
                    0.0054157130719655,
                    0.0286201080244150,
                    0.1144811451032983,
                    0.0247320824310707,
                    0.4469931106692654,
                    0.1201149311199702,
                    0.3324881246726454,
                    0.1529190224329338,
                    0.0126872452591733,
                    0.1901505501887097,
                    0.5852910754713261,
                    0.0548181473315376,
                    0.0427958329505531,
                    0.8193450524383626,
                    0.0267193517141522,
                    0.0316967233772400,
                    0.0919178668344217,
                    0.3372182623043156,
                    0.1126491331073949,
                    0.1515632978457940,
                    0.3402164177011169,
                    0.0865574846798659,
                    0.0793918773870708,
                    0.1169682461544246,
                    0.7509779430720097,
                    0.0796925241935777,
                    0.0632965941138890,
                    0.0569456717776583,
                    0.1450477588516786,
                    0.0612068933310174,
                    0.4429974680874531,
                    0.1171324626635184,
                    0.3940913370297614,
                    0.4395788784711709,
                    0.0577415505657756,
                    0.1112955667737245,
                    0.1316834262016172,
                    0.0852268026657049,
                    0.7742840020908915,
                    0.0400622066655128,
                ]
            )
            tmp = tmp.reshape((11, 4))
        case 5:
            tmp = jnp.asarray(
                [
                    0.3108859192633006,
                    0.3108859192633006,
                    0.3108859192633006,
                    0.1126879257180158,
                    0.4544962958743503,
                    0.4544962958743503,
                    0.0455037041256496,
                    0.0425460207770814,
                    0.0455037041256496,
                    0.4544962958743503,
                    0.4544962958743503,
                    0.0425460207770814,
                    0.4544962958743503,
                    0.0455037041256496,
                    0.0455037041256496,
                    0.0425460207770814,
                    0.0927352503108912,
                    0.0927352503108912,
                    0.7217942490673263,
                    0.0734930431163619,
                    0.3108859192633006,
                    0.3108859192633006,
                    0.0673422422100981,
                    0.1126879257180158,
                    0.0927352503108912,
                    0.0927352503108912,
                    0.0927352503108912,
                    0.0734930431163619,
                    0.0927352503108912,
                    0.7217942490673263,
                    0.0927352503108912,
                    0.0734930431163619,
                    0.3108859192633006,
                    0.0673422422100981,
                    0.3108859192633006,
                    0.1126879257180158,
                    0.7217942490673263,
                    0.0927352503108912,
                    0.0927352503108912,
                    0.0734930431163619,
                    0.0673422422100981,
                    0.3108859192633006,
                    0.3108859192633006,
                    0.1126879257180158,
                    0.0455037041256496,
                    0.0455037041256496,
                    0.4544962958743503,
                    0.0425460207770814,
                    0.0455037041256496,
                    0.4544962958743503,
                    0.0455037041256496,
                    0.0425460207770814,
                    0.4544962958743503,
                    0.0455037041256496,
                    0.4544962958743503,
                    0.0425460207770814,
                ]
            )
            tmp = tmp.reshape((14, 4))
        case 6:
            tmp = jnp.asarray(
                [
                    0.8746168885670683,
                    0.0016502414396875,
                    0.1042983037802942,
                    0.0072051333503377,
                    0.1624377743364805,
                    0.0486578256639518,
                    0.5554865546922777,
                    0.0647230234809383,
                    0.6255956733270350,
                    0.2167840595851656,
                    0.1335729717161918,
                    0.0417561289814202,
                    0.0485590349854997,
                    0.3152341342565913,
                    0.0777799447906708,
                    0.0468866591949777,
                    0.2123913908430850,
                    0.5512046215526779,
                    0.1911298253738990,
                    0.0642626547049837,
                    0.0802502491180986,
                    0.8735128127614880,
                    0.0233398227879653,
                    0.0097107199140436,
                    0.0020452843713277,
                    0.0619185737602200,
                    0.3821105213368406,
                    0.0228306507632138,
                    0.7430454470768171,
                    0.1116735876122876,
                    0.0014009714927136,
                    0.0203977040996243,
                    0.2912491656449773,
                    0.2201903067498265,
                    0.0156772928654807,
                    0.0396800192984899,
                    0.5120512410177051,
                    0.0114046111258490,
                    0.4147866887697848,
                    0.0213303550447111,
                    0.0011957289270105,
                    0.6803017849056812,
                    0.2350505385746095,
                    0.0211633942380320,
                    0.4182556428001524,
                    0.5029243848539261,
                    0.0014823492588063,
                    0.0236703215432329,
                    0.0756495057730943,
                    0.3323211418716275,
                    0.5403722579539822,
                    0.0490532969018372,
                    0.5480249213620508,
                    0.0596672164470000,
                    0.1545963494454863,
                    0.0690790231378501,
                    0.1410085046735627,
                    0.0508937313254425,
                    0.8062812451359968,
                    0.0143311064677154,
                    0.2166008062191251,
                    0.0686003114916989,
                    0.2168670911641968,
                    0.0836591697123374,
                    0.4246910368665688,
                    0.0084801219195986,
                    0.0026145928698761,
                    0.0108464892755928,
                    0.3340496585496776,
                    0.3098946335446950,
                    0.1284572624659843,
                    0.1025240624805954,
                    0.3337613494669106,
                    0.1731672255755042,
                    0.4173503478363065,
                    0.0830129194149179,
                    0.0767039739434372,
                    0.3017356805716037,
                    0.3274922350236289,
                    0.1017105458079781,
                    0.0718764126292230,
                    0.0628888824345865,
                    0.0670609374784439,
                    0.0295527967164588,
                    0.0067396276812792,
                    0.0904745895996605,
                    0.7668964455753671,
                    0.0200739965400799,
                    0.0840955445837731,
                    0.6211204831496858,
                    0.0555145530701308,
                    0.0525398289306307,
                ]
            )
            tmp = tmp.reshape((23, 4))
        case 7:
            tmp = jnp.asarray(
                [
                    0.3002416572973887,
                    0.0343958764090097,
                    0.0363574344558375,
                    0.0174634934965629,
                    0.4562748748421011,
                    0.1770949336338915,
                    0.2023800436826119,
                    0.0649822804683734,
                    0.0435156707057980,
                    0.1907080463902385,
                    0.5156119554737180,
                    0.0432909419411822,
                    0.0674704268652055,
                    0.6248052088506927,
                    0.2864704911297047,
                    0.0196966183096976,
                    0.0351757962004400,
                    0.3009324536669736,
                    0.6263899456052338,
                    0.0180366045230775,
                    0.0465586231823697,
                    0.2782694670147702,
                    0.1904686063806941,
                    0.0465789789500725,
                    0.0270400007458065,
                    0.0592553513707679,
                    0.3085538422115337,
                    0.0208255919856109,
                    0.2646801005318440,
                    0.1971493272516444,
                    0.0649765422855217,
                    0.0575425216335696,
                    0.0584308870280592,
                    0.8178995621576657,
                    0.0603892349556298,
                    0.0211946516501725,
                    0.2777673159649886,
                    0.0476736573398107,
                    0.4793739967382065,
                    0.0471883811637115,
                    0.0555146373286111,
                    0.2869217065138451,
                    0.0272494808480173,
                    0.0196501591898121,
                    0.0203746164592836,
                    0.6125583966450518,
                    0.0623018163296930,
                    0.0184820837344789,
                    0.1922244985520259,
                    0.0430284938334260,
                    0.2485713056693554,
                    0.0427447011008647,
                    0.1991087918995903,
                    0.2642325286400432,
                    0.4718676792912370,
                    0.0576407228457824,
                    0.6235525826354381,
                    0.2894310610359156,
                    0.0514831192758012,
                    0.0223576139136417,
                    0.0594225690096152,
                    0.0588796306801409,
                    0.0637892658968220,
                    0.0212191384526022,
                    0.5028328636834030,
                    0.0311436212022759,
                    0.1603144456632615,
                    0.0338728685733351,
                    0.1646301258774454,
                    0.5201765728647110,
                    0.0228729142743443,
                    0.0267949357580908,
                    0.2891312880999799,
                    0.6241982800589735,
                    0.0347318401180657,
                    0.0221039803504898,
                    0.4579592178379015,
                    0.3051966920830379,
                    0.0380160294851374,
                    0.0373559578225954,
                    0.5226788734283342,
                    0.1641956191205870,
                    0.2936439912332349,
                    0.0251715053465831,
                    0.8177970251994465,
                    0.0583606514742920,
                    0.0635452593856531,
                    0.0212351686583526,
                    0.3053135448784433,
                    0.4603995934951201,
                    0.1965711945189183,
                    0.0372351990949498,
                    0.6106947282193217,
                    0.0202214855799755,
                    0.3055925317265655,
                    0.0187255497284155,
                    0.0585696402354899,
                    0.0598781112217250,
                    0.8184045279916535,
                    0.0210410571517633,
                    0.1947183776661725,
                    0.1980390771723790,
                    0.3015283379031818,
                    0.0605315353356269,
                    0.2858333457907975,
                    0.0547185573312177,
                    0.6296884514679080,
                    0.0204610032387520,
                    0.0304091940099104,
                    0.5012486403424117,
                    0.3042244651274234,
                    0.0335109231569724,
                    0.0597170815760428,
                    0.0259590154076463,
                    0.6093786510419171,
                    0.0203462573214307,
                    0.1765706521407181,
                    0.4553614944072237,
                    0.1676294047317003,
                    0.0635654486690692,
                    0.6262307245732673,
                    0.0680576181199523,
                    0.0198791666412824,
                    0.0191541264343591,
                ]
            )
            tmp = tmp.reshape((31, 4))
        case 8:
            tmp = jnp.asarray(
                [
                    0.4145770165127839,
                    0.3830837692811989,
                    0.1912186370119430,
                    0.0181699231929644,
                    0.0475649866949658,
                    0.0384413738702507,
                    0.8871119506366457,
                    0.0056105823767546,
                    0.3022392039475506,
                    0.3096755189223818,
                    0.0162164527045346,
                    0.0246181135925707,
                    0.0424444184990879,
                    0.0478548656064025,
                    0.7264185551267094,
                    0.0180540181960190,
                    0.4438540306537173,
                    0.1593000599996238,
                    0.3626667438035576,
                    0.0267538807284043,
                    0.1349985069363986,
                    0.4329948100296093,
                    0.4065632160324652,
                    0.0228798879438206,
                    0.2702491895036498,
                    0.0084328712544561,
                    0.2125111291650657,
                    0.0197924000090930,
                    0.2141588954260017,
                    0.0502038576356240,
                    0.4936462128941918,
                    0.0407999077508595,
                    0.1456793207565755,
                    0.1420799277732693,
                    0.2654961248080462,
                    0.0560483550063074,
                    0.4800422639855878,
                    0.0245515962790342,
                    0.4555570662781256,
                    0.0116445350263073,
                    0.0324892081368812,
                    0.2126005989136902,
                    0.7103849921091126,
                    0.0139412815633558,
                    0.6991852897392833,
                    0.0092681532935099,
                    0.0732695871629151,
                    0.0120434677827530,
                    0.6912707213620969,
                    0.2590322769551171,
                    0.0365506847922414,
                    0.0080583052512232,
                    0.0111092826144552,
                    0.5019172513337366,
                    0.4337572382922951,
                    0.0111640226072287,
                    0.1589177658747160,
                    0.3525635612848826,
                    0.1335110842049796,
                    0.0489016994980424,
                    0.0636195353247279,
                    0.2425694992116304,
                    0.4950048247755332,
                    0.0426345969640420,
                    0.7512107984274565,
                    0.0361307917305478,
                    0.1814260426523001,
                    0.0110411629216581,
                    0.4416365362117750,
                    0.4543312862588164,
                    0.0241279520865387,
                    0.0187297023322842,
                    0.4793998925970132,
                    0.0412114636044895,
                    0.2900348230690179,
                    0.0332672437302112,
                    0.1526196910014446,
                    0.0110494239827689,
                    0.0394703558422039,
                    0.0067548714706999,
                    0.8812579575152194,
                    0.0493545924930924,
                    0.0238161892780926,
                    0.0062425385458276,
                    0.6348065288024368,
                    0.1477269775427559,
                    0.0143211909945740,
                    0.0175258602591282,
                    0.0161537768362078,
                    0.0533158863351681,
                    0.0254279957055490,
                    0.0034888937203578,
                    0.1571350802815834,
                    0.6346254701849137,
                    0.0392987001563497,
                    0.0293516666266539,
                    0.2221434793473390,
                    0.0373928531070166,
                    0.6865530348199036,
                    0.0167428782322345,
                    0.0474197661230861,
                    0.8920202885533636,
                    0.0135582016025303,
                    0.0048379630536228,
                    0.1971257374399104,
                    0.1263766406253084,
                    0.0534773781630199,
                    0.0300046030935462,
                    0.0557948612177992,
                    0.4698411775898500,
                    0.0309681567564886,
                    0.0184700507451081,
                    0.6316261452724231,
                    0.1564233918934593,
                    0.1315762321268426,
                    0.0314910850439838,
                    0.1941392655923862,
                    0.5222754374782991,
                    0.1993778513250609,
                    0.0373796919907812,
                    0.0534266429077665,
                    0.0204967533561221,
                    0.4580407029017354,
                    0.0145927845595434,
                    0.4151950751920222,
                    0.1206348095358562,
                    0.1356578205019857,
                    0.0494418796474225,
                    0.3896011380561273,
                    0.3332724856599186,
                    0.1115717539936369,
                    0.0425522878715154,
                    0.4405519657828008,
                    0.0426237277184632,
                    0.0298879430672583,
                    0.0157971106054455,
                    0.1901087733754798,
                    0.1815142502860739,
                    0.5904777744426020,
                    0.0241420024501854,
                    0.0445285256167664,
                    0.0511955588285995,
                    0.1838336361599616,
                    0.0198958586192624,
                    0.0086177998987947,
                    0.7022002301888654,
                    0.0663989044948063,
                    0.0103744025685826,
                    0.0116040900657666,
                    0.1433209858915356,
                    0.4412164737903684,
                    0.0143584515136082,
                    0.2455987399683951,
                    0.6929759446465587,
                    0.0525285915493202,
                    0.0098824472518686,
                    0.0500610272937231,
                    0.2173800961200812,
                    0.0395908150328984,
                    0.0157464797494312,
                    0.2671415544232039,
                    0.2376839659731869,
                    0.3370228174476876,
                    0.0618810767085517,
                    0.0480506603423729,
                    0.4742858045797399,
                    0.2518463964414198,
                    0.0391796049957540,
                    0.0130528118009615,
                    0.2822707494912109,
                    0.1848591801453774,
                    0.0188293816931540,
                    0.0475469996186544,
                    0.7516869678701615,
                    0.1600276135512940,
                    0.0168830425098297,
                ]
            )
            tmp = tmp.reshape((44, 4))
        case 9:
            tmp = jnp.asarray(
                [
                    0.0228771921259469,
                    0.3854277790362540,
                    0.2037937874247621,
                    0.0184541040950682,
                    0.1779013709110090,
                    0.4091789152201282,
                    0.3768320144226283,
                    0.0263688441379337,
                    0.0153346288504854,
                    0.4395414314627663,
                    0.3856915922907349,
                    0.0129843629997352,
                    0.0410266143487432,
                    0.8702665707058525,
                    0.0524007500943351,
                    0.0075369887473240,
                    0.1221435077921257,
                    0.1703632113240204,
                    0.6931483893283681,
                    0.0112902145957498,
                    0.0000130064788249,
                    0.1691795878401991,
                    0.7840636802171264,
                    0.0041974594510122,
                    0.6539546088038215,
                    0.0358013898335501,
                    0.1084429676794620,
                    0.0195297481245139,
                    0.0423720274411046,
                    0.0462330391334404,
                    0.0369443137669500,
                    0.0070701283089204,
                    0.1676767943964418,
                    0.0276106056520789,
                    0.2633783150667404,
                    0.0182295586188933,
                    0.0300416545621096,
                    0.1684488866191471,
                    0.2257928860667317,
                    0.0189742499799463,
                    0.2969101972787469,
                    0.3463378689772197,
                    0.0243387829090815,
                    0.0124790299046609,
                    0.4389309398965947,
                    0.0242752693603188,
                    0.3599485361902332,
                    0.0188590028529837,
                    0.5060894004176661,
                    0.1129766995689154,
                    0.2848034194058732,
                    0.0237485508817650,
                    0.3093454158727241,
                    0.3249822555407154,
                    0.1774721052174033,
                    0.0408665382283079,
                    0.2003288579260044,
                    0.6356109460036826,
                    0.1459507330961254,
                    0.0122781434129651,
                    0.6409443148321043,
                    0.1669655587902913,
                    0.0003804307100489,
                    0.0070809717367823,
                    0.8684348906155640,
                    0.0459524179304327,
                    0.0419438579369183,
                    0.0082426830370143,
                    0.0353244186624793,
                    0.0342602781758433,
                    0.1926456994684344,
                    0.0101967955729838,
                    0.2082525047814187,
                    0.0074989005445734,
                    0.0587480405674917,
                    0.0071690731469955,
                    0.0424979321902079,
                    0.0385222523881031,
                    0.7122430931384430,
                    0.0140560164595015,
                    0.3992138743685673,
                    0.1757543676917409,
                    0.0509527826206183,
                    0.0288231137270553,
                    0.1976857681788971,
                    0.7343371859702916,
                    0.0196302740281513,
                    0.0089077375556437,
                    0.2099687083368246,
                    0.1044992464136398,
                    0.0000007540268307,
                    0.0066457198867808,
                    0.0754927954561146,
                    0.1955780460642323,
                    0.5953204920779403,
                    0.0299035265454547,
                    0.0348267971938675,
                    0.6603573627255149,
                    0.1200651834331449,
                    0.0199599794625485,
                    0.0283787085262725,
                    0.7846734722712416,
                    0.0000004032714584,
                    0.0032902912355902,
                    0.4599570448520207,
                    0.0329395219054931,
                    0.4785665294057564,
                    0.0102526426958416,
                    0.3521798518257496,
                    0.1751588071180923,
                    0.4612374007141584,
                    0.0124745897808344,
                    0.2171825728901714,
                    0.0486159871981612,
                    0.6083076242759426,
                    0.0164536070069318,
                    0.1769835041486189,
                    0.1221421889557460,
                    0.1093851510436900,
                    0.0343219353927922,
                    0.0385688396366762,
                    0.6737565304587229,
                    0.2508341644461082,
                    0.0137454976739527,
                    0.5711357877921358,
                    0.1727855794196116,
                    0.1030022318532128,
                    0.0290870601716308,
                    0.2646596673044796,
                    0.1884660494181322,
                    0.4291259275953451,
                    0.0399256448541824,
                    0.1571292828045067,
                    0.5805461406075157,
                    0.0342398448552173,
                    0.0203806642850829,
                    0.6635380829388825,
                    0.1217610687738568,
                    0.1969586385674539,
                    0.0117955663639247,
                    0.2136593684439528,
                    0.0308486614875299,
                    0.7171530557196868,
                    0.0089396778343122,
                    0.1566773917997997,
                    0.3175181736034333,
                    0.0000002460050490,
                    0.0074257791160959,
                    0.1079976673827427,
                    0.4017631489702297,
                    0.2986812946355224,
                    0.0384850997221568,
                    0.4409806880459452,
                    0.4970119774390469,
                    0.0399506026934863,
                    0.0097366180168374,
                    0.4254477514318533,
                    0.3401421454318485,
                    0.1980448837883506,
                    0.0283415503514634,
                    0.6946928384569798,
                    0.2319317735985255,
                    0.0388553433791697,
                    0.0123670195019076,
                    0.0356505810548022,
                    0.4036293430763501,
                    0.5294685040698826,
                    0.0116694504410301,
                    0.0334874006569261,
                    0.0361860934439792,
                    0.4581708108037080,
                    0.0134792719091988,
                    0.0454430366647997,
                    0.0335914400765359,
                    0.8803952330519741,
                    0.0060256669416746,
                    0.0010448185548277,
                    0.1916915089545268,
                    0.5074181318482486,
                    0.0098393888842789,
                    0.3163270208687076,
                    0.1363678337579023,
                    0.2413790612332155,
                    0.0428438886881625,
                    0.3896405492708102,
                    0.0277742454193376,
                    0.1878688449107907,
                    0.0191177190654798,
                    0.7406364104471735,
                    0.0271695025648547,
                    0.0010088448625600,
                    0.0033485451295063,
                    0.1477794601830333,
                    0.3339861060406962,
                    0.1171280597097285,
                    0.0410612411191106,
                    0.1886404904119914,
                    0.0321791910367441,
                    0.4831348472084820,
                    0.0219039985248497,
                    0.7281357192410779,
                    0.0112968130870926,
                    0.2175637092104645,
                    0.0065044292862879,
                    0.0324051277612772,
                    0.4924708008762203,
                    0.0381180575913889,
                    0.0132819482627586,
                    0.4108404244356825,
                    0.4161327723147283,
                    0.0341551329223098,
                    0.0223305853762450,
                    0.1878404308389233,
                    0.5631872407093852,
                    0.1376501472148007,
                    0.0251380636981340,
                    0.1089137999932238,
                    0.1703763474658899,
                    0.3671064610617264,
                    0.0438140659597697,
                    0.4539162997518049,
                    0.0382704480727342,
                    0.0376101832415742,
                    0.0143183572522290,
                    0.0350490173040331,
                    0.2229861824768844,
                    0.0465507141469340,
                    0.0144475939872048,
                ]
            )
            tmp = tmp.reshape((57, 4))
        case 10:
            tmp = jnp.asarray(
                [
                    0.7108211811635610,
                    0.1680938151486082,
                    0.0910432615060082,
                    0.0130941001192225,
                    0.5790076670015544,
                    0.0754133419096396,
                    0.1931731991654421,
                    0.0232364268442544,
                    0.0978942952808300,
                    0.5193266649058177,
                    0.3520237533382442,
                    0.0158879701144934,
                    0.0711742779743134,
                    0.2990859942693232,
                    0.5748644395468989,
                    0.0197029139925291,
                    0.0178807490892937,
                    0.4111564572188602,
                    0.0271003428758123,
                    0.0060870155115520,
                    0.0037141359332596,
                    0.4870271497277897,
                    0.5086671624419364,
                    0.0016810511093043,
                    0.1653019287626687,
                    0.0322142586483131,
                    0.1856060969044481,
                    0.0185758175795591,
                    0.1548345938765296,
                    0.1665886064354638,
                    0.1008521679205936,
                    0.0258659485047256,
                    0.5053653467106303,
                    0.3052749158008759,
                    0.0035297452623257,
                    0.0074298918266025,
                    0.3497214515955615,
                    0.0650406291975330,
                    0.3704875867283166,
                    0.0279808105781397,
                    0.3621294571184972,
                    0.2230479437288030,
                    0.2111222297561496,
                    0.0397743106074694,
                    0.0015190811552245,
                    0.4112670915387675,
                    0.4585708068459480,
                    0.0075782147439984,
                    0.3676710158884832,
                    0.0442304309693195,
                    0.5653369031608720,
                    0.0093429310289290,
                    0.0497097720547089,
                    0.8388588477211602,
                    0.1076048046045897,
                    0.0031066647153829,
                    0.1325393365313179,
                    0.0851962921160148,
                    0.6044123655429240,
                    0.0222960392583066,
                    0.0167930550889742,
                    0.1467379830061631,
                    0.7198942577861557,
                    0.0093263274933274,
                    0.3846449765100088,
                    0.0778308544281304,
                    0.1338208423184902,
                    0.0339825945050728,
                    0.6107193475414290,
                    0.0314930542293789,
                    0.0256459239867768,
                    0.0077405253830618,
                    0.0800098091809266,
                    0.9082336711181266,
                    0.0106069851164333,
                    0.0010222807980556,
                    0.1722115445736072,
                    0.0041711369451297,
                    0.7881404292621583,
                    0.0031584712640255,
                    0.0697353269021379,
                    0.4653246578925895,
                    0.2797431303256196,
                    0.0305175844620499,
                    0.6113701604897331,
                    0.1367777657504910,
                    0.0511401890283743,
                    0.0208421583972557,
                    0.2211161924187932,
                    0.5356748817939830,
                    0.0108118255726109,
                    0.0107918342350895,
                    0.5754435811494772,
                    0.3828233507637828,
                    0.0177643675730773,
                    0.0053208751782787,
                    0.1158392818921408,
                    0.7727886692399550,
                    0.0242195079409199,
                    0.0086956685251635,
                    0.7827988815314628,
                    0.1619283845747259,
                    0.0016173737233016,
                    0.0035598178068562,
                    0.4121634149061806,
                    0.1202901254531915,
                    0.0095274339951231,
                    0.0086596613547201,
                    0.1814261795878509,
                    0.1096295391696643,
                    0.3301733234164556,
                    0.0332884040430397,
                    0.2928680693443875,
                    0.6537012944960302,
                    0.0291479784247406,
                    0.0065622796262284,
                    0.4907127714152918,
                    0.3114372654169192,
                    0.1006675664676695,
                    0.0245083525594874,
                    0.0293215616139512,
                    0.0414023329173425,
                    0.1095147189589663,
                    0.0074213079787744,
                    0.3276033530938000,
                    0.0000741494887285,
                    0.2954875351026062,
                    0.0083861267242499,
                    0.3600178613386471,
                    0.5012978987010694,
                    0.0432508226806463,
                    0.0140808470761840,
                    0.7755311879383773,
                    0.0301485018665602,
                    0.1481879111466897,
                    0.0080821088401726,
                    0.2126225190120122,
                    0.4825350362748294,
                    0.1303547520984165,
                    0.0323199578468694,
                    0.2443494082536043,
                    0.0103183829221147,
                    0.5903498037553033,
                    0.0090724796272051,
                    0.3016505415459932,
                    0.1430011264098975,
                    0.4807196239735930,
                    0.0229206911684140,
                    0.0262051276767974,
                    0.1956862242710102,
                    0.0981613717356904,
                    0.0102681519608962,
                    0.0454997987570612,
                    0.1988251256663947,
                    0.4768646629265832,
                    0.0257187414584072,
                    0.0314383493935819,
                    0.0359326317838278,
                    0.8977091528373697,
                    0.0039281812985419,
                    0.0291222887265170,
                    0.0005751516278394,
                    0.3188433055232422,
                    0.0033857046660901,
                    0.2272222363946760,
                    0.0902056725553408,
                    0.0251612240414210,
                    0.0098859590235499,
                    0.1773693932713253,
                    0.2659421383325723,
                    0.4053675049542561,
                    0.0361274972395152,
                    0.6431484774270714,
                    0.0428271260641956,
                    0.3113671388220656,
                    0.0037838247969592,
                    0.0002651387752094,
                    0.8943497560407246,
                    0.0342172350951809,
                    0.0022283163404585,
                    0.0343611478688745,
                    0.6780058688832242,
                    0.0151425513695827,
                    0.0060333234046835,
                    0.4999879882641083,
                    0.1742036398751961,
                    0.2886008529574142,
                    0.0232148598477992,
                    0.4444501765779667,
                    0.4135053284871942,
                    0.1419288711528731,
                    0.0072239212399219,
                    0.0943385702418137,
                    0.4872400695815088,
                    0.0708137941505284,
                    0.0216762198118419,
                    0.6011876812681009,
                    0.0001683512016895,
                    0.1454413486084074,
                    0.0065216561617285,
                    0.3249680995582529,
                    0.2796271780202718,
                    0.0557281529108646,
                    0.0323193527386097,
                    0.1089238586456045,
                    0.0231673460679721,
                    0.4749489083627679,
                    0.0144301473923800,
                    0.0453722167712841,
                    0.1577014452371543,
                    0.0134062255036993,
                    0.0054215483403461,
                    0.0206278403148913,
                    0.3071952605835924,
                    0.2815097419109018,
                    0.0135457897808562,
                    0.0416618045025143,
                    0.3402925074117468,
                    0.1419441168969618,
                    0.0131380726063402,
                    0.5232246056946657,
                    0.0171939835068094,
                    0.3747867167288931,
                    0.0102880832307780,
                    0.9333826965414616,
                    0.0355665200086333,
                    0.0307178049764193,
                    0.0013317330487139,
                    0.0368794531682594,
                    0.1068733463428279,
                    0.2878274336814129,
                    0.0200733495555370,
                    0.2390695873803809,
                    0.2707750397116681,
                    0.4900314794039396,
                    0.0084917149996748,
                    0.0098774540167762,
                    0.0515942816051371,
                    0.5576518513346611,
                    0.0070285749956164,
                    0.1468723669075991,
                    0.3163196386441845,
                    0.0140964655837443,
                    0.0118299853728404,
                    0.8260093606523856,
                    0.0306167495791246,
                    0.0314637036026320,
                    0.0065145044115713,
                    0.1380115817340293,
                    0.1031336966538923,
                    0.7246848432623155,
                    0.0131989551030418,
                    0.2852351403960840,
                    0.3927043788424667,
                    0.2677783805703437,
                    0.0280602646278701,
                    0.0194806033226319,
                    0.6891648157458625,
                    0.2406622508912331,
                    0.0079622366404425,
                    0.1832473550535123,
                    0.6410981463475119,
                    0.1440710813942678,
                    0.0148591816036919,
                    0.0605493511843967,
                    0.7035992796137526,
                    0.1064038967839555,
                    0.0149896057693298,
                    0.0005913322473979,
                    0.5834193040821614,
                    0.1419049463214869,
                    0.0068406602525445,
                    0.0197382202205803,
                    0.1983391941632586,
                    0.7814887334628181,
                    0.0021661954307135,
                    0.0020217951217635,
                    0.0265606896774049,
                    0.0002569174707227,
                    0.0004963334735625,
                    0.1583679664182448,
                    0.2740685756121609,
                    0.2078803717179151,
                    0.0364479214739948,
                    0.1164605173320014,
                    0.0255374164732954,
                    0.0335456317293547,
                    0.0055892270592715,
                    0.0414332994179814,
                    0.0205559406284286,
                    0.7571094133732397,
                    0.0069408493672644,
                    0.3572328788470244,
                    0.0061464777334794,
                    0.0465779255700504,
                    0.0061388940465607,
                ]
            )
            tmp = tmp.reshape((74, 4))
        case _:
            assert False, "Integration rule not implemented!"

    coordinates = tmp[:, :3]
    weights = (1 / 6) * tmp[:, 3]
    return (coordinates, weights)


@jit_with_docstring(static_argnames=["order"])
def int_pts_line(x_i, order):
    """
    Linear mapping of integration points on reference line to integration points on one physical line

    Args:
      x_i: nodal coordinates of one physical line (if more than 2 nodes are given, it uses the first 2 nodes for the mapping)
      order: order of polynomial accuracy of the integration rule

    Return: (coordinates, weights)
      - coordinates of the integration points
      - weights of the integration points
    """
    x_0 = jnp.asarray(x_i[0])
    x_1 = jnp.asarray(x_i[1])

    dx01 = x_1 - x_0

    length = jnp.linalg.norm(dx01)
    area = length
    area_ratio = area

    (ref_coor, ref_weights) = gauss_legendre_1d(order)

    def map_one(x_ref, w_ref):
        coor = x_0 + x_ref * dx01
        weight = area_ratio * w_ref
        return (coor, weight)

    return jax.vmap(map_one, (0, 0), (0, 0))(ref_coor, ref_weights)


@jit_with_docstring(static_argnames=["order"])
def int_pts_tri(x_i, order):
    """
    Linear mapping of integration points in reference triangle to integration points in one physical triangle

    Args:
      x_i: nodal coordinates of one physical triangle (if more than 3 nodes are given, it uses the first 3 nodes for the mapping)
      order: order of polynomial accuracy of the integration rule

    Returns (coordinates, weights):
      - coordinates of the integration points
      - weights of the integration points
    """
    x_0 = jnp.asarray(x_i[0])
    x_1 = jnp.asarray(x_i[1])
    x_2 = jnp.asarray(x_i[2])

    dx01 = x_1 - x_0
    dx02 = x_2 - x_0

    length = jnp.linalg.norm(dx02)
    height = jnp.linalg.norm(geometry.project_on_line(x_1, x_0, x_2) - x_1)
    area = (1 / 2) * length * height
    area_ratio = area / (1 / 2)

    (ref_coor, ref_weights) = int_pts_ref_tri(order)

    def map_one(x_ref, w_ref):
        coor = x_0 + x_ref[0] * dx01 + x_ref[1] * dx02
        weight = area_ratio * w_ref
        return (coor, weight)

    return jax.vmap(map_one, (0, 0), (0, 0))(ref_coor, ref_weights)


@jit_with_docstring(static_argnames=["order"])
def int_pts_tet(x_i, order):
    """
    Linear mapping of integration points in reference tetrahedron to integration points in one physical tetrahedron

    Args:
      x_i: nodal coordinates of one physical tetrahedron (if more than 4 nodes are given, it uses the first 4 nodes for the mapping)
      order: order of polynomial accuracy of the integration rule

    Return: (coordinates, weights)
      - coordinates of the integration points
      - weights of the integration points
    """
    x_0 = jnp.asarray(x_i[0])
    x_1 = jnp.asarray(x_i[1])
    x_2 = jnp.asarray(x_i[2])
    x_3 = jnp.asarray(x_i[3])

    dx01 = x_1 - x_0
    dx02 = x_2 - x_0
    dx03 = x_3 - x_0

    area = (1 / 6) * jnp.sqrt(jnp.dot(jnp.cross(dx01, dx02), dx03) ** 2)
    area_ratio = area / (1 / 6)

    (ref_coor, ref_weights) = int_pts_ref_tet(order)

    def map_one(x_ref, w_ref):
        coor = x_0 + x_ref[0] * dx01 + x_ref[1] * dx02 + x_ref[2] * dx03
        weight = area_ratio * w_ref
        return (coor, weight)

    return jax.vmap(map_one, (0, 0), (0, 0))(ref_coor, ref_weights)


@jit_with_docstring(static_argnames=["order"])
def int_pts_in_line_mesh(x_nodes, elem, order):
    """
    Given a mesh of lines, it linearly maps integration points from a reference line onto the actual lines

    Args:
      x_nodes: nodal coordinates of all nodes of the mesh
      elem: nodal indices of each line element (if an element has more than 2 nodes, the first 2 nodes are used for the mapping)
      order: polynomial accuracy order of integration rule within each line

    Returns (coor, connectivity, weights):
      - coordinates of integration points
      - connectivity of integration points, i.e. list of all nodes that have a contribution
      - weights of integration points
    """
    x_nodes = jnp.asarray(x_nodes)
    elem = jnp.asarray(elem)

    (x_int, weights) = jax.vmap(int_pts_line, (0, None), (0, 0))(x_nodes[elem], order)
    ints_per_elem = weights.shape[-1]
    nods_per_elem = elem.shape[-1]
    weights = weights.flatten()
    n_int = weights.shape[0]
    n_dim = x_int.shape[-1]
    x_int = x_int.reshape((n_int, n_dim))
    connectivity = jnp.tile(elem, ints_per_elem).reshape((n_int, nods_per_elem))
    jax.debug.print("Number of elements: {x}", x=elem.shape[0])
    jax.debug.print("Number of integration points: {x}", x=n_int)

    return (x_int, weights, n_int, connectivity)


@jit_with_docstring(static_argnames=["order"])
def int_pts_in_tri_mesh(x_nodes, elem, order):
    """
    Given a mesh of triangles, it linearly maps integration points from a reference triangle into the actual triangles

    Args:
      x_nodes: nodal coordinates of all nodes of the mesh
      elem: nodal indices of each triangular element (if an element has more than 3 nodes, the first 3 nodes are used for the mapping)
      order: polynomial accuracy order of integration rule within each triangle

    Returns (coor, connectivity, weights):
      - coordinates of integration points
      - connectivity of integration points, i.e. list of all nodes that have a contribution
      - weights of integration points
    """
    x_nodes = jnp.asarray(x_nodes)
    elem = jnp.asarray(elem)

    (x_int, weights) = jax.vmap(int_pts_tri, (0, None), (0, 0))(x_nodes[elem], order)
    ints_per_elem = weights.shape[-1]
    nods_per_elem = elem.shape[-1]
    weights = weights.flatten()
    n_int = weights.shape[0]
    n_dim = x_int.shape[-1]
    x_int = x_int.reshape((n_int, n_dim))
    connectivity = jnp.tile(elem, ints_per_elem).reshape((n_int, nods_per_elem))
    jax.debug.print("Number of elements: {x}", x=elem.shape[0])
    jax.debug.print("Number of integration points: {x}", x=n_int)

    return (x_int, weights, n_int, connectivity)


@jit_with_docstring(static_argnames=["order"])
def int_pts_in_tet_mesh(x_nodes, elem, order):
    """
    Given a mesh of tetrahedrons, it linearly maps integration points from a reference tetrahedron into the actual tetrahedrons

    Args:
      x_nodes: nodal coordinates of all nodes of the mesh
      elem: nodal indices of each tetrahedral element (if an element has more than 4 nodes, the first 4 nodes are used for the mapping)
      order: polynomial accuracy order of integration rule within each tetrahedron

    Returns (coor, connectivity, weights):
      - coordinates of integration points
      - connectivity of integration points, i.e. list of all nodes that have a contribution
      - weights of integration points
    """
    x_nodes = jnp.asarray(x_nodes)
    elem = jnp.asarray(elem)

    (x_int, weights) = jax.vmap(int_pts_tet, (0, None), (0, 0))(x_nodes[elem], order)
    ints_per_elem = weights.shape[-1]
    nods_per_elem = elem.shape[-1]
    weights = weights.flatten()
    n_int = weights.shape[0]
    n_dim = x_int.shape[-1]
    x_int = x_int.reshape((n_int, n_dim))
    connectivity = jnp.tile(elem, ints_per_elem).reshape((n_int, nods_per_elem))
    jax.debug.print("Number of elements: {x}", x=elem.shape[0])
    jax.debug.print("Number of integration points: {x}", x=n_int)

    return (x_int, weights, n_int, connectivity)
