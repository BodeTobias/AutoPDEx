# geometry.py
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
This module contains geometry related functions that can be used for seeding nodes and integration points, 
impose boundary conditions and to select nodes and elements. The main functionality relies on distance functions. 
For the construction of user-specific distance functions, there are the Rvachev operations, 
but there are also some pre-made signed distance functions and positive smooth distance functions. 
The distance functions are suitable, for example, for verifying whether nodes lie within certain regions or 
on specified surfaces. Additionally, the smooth distance functions can be used for constructing solution structures.
"""

import jax
import jax.numpy as jnp

from autopdex.utility import jit_with_docstring

# ToDo:
# - reader for stl data


### Rvachev function operations
def r_equivalence(o1, o2):
    """
    Computes the logical NOT XOR operation by combining two positively evaluated smooth distance functions.

    Args:
      o1 (float): The first operand for the logical "not xor" operation.
      o2 (float): The second operand for the logical "not xor" operation.

    Returns:
      float: The result of the logical "not xor" operation, calculated using
          the formula o1 * o2 / (o1 + o2).
    """
    return o1 * o2 / (o1 + o2)


def r_conjunction(o1, o2):
    """
    Computes the logical AND operation using smooth distance functions.

    Args:
      o1 (float): The first operand for the logical "and" operation.
      o2 (float): The second operand for the logical "and" operation.

    Returns:
      float: The result of the logical "and" operation, calculated using
          the formula o1 + o2 - sqrt(o1**2 + o2**2).
    """
    return o1 + o2 - jnp.sqrt(o1**2 + o2**2)


def r_disjunction(o1, o2):
    """
    Computes the logical OR operation using smooth distance functions.

    Args:
      o1 (float): The first operand for the logical "or" operation.
      o2 (float): The second operand for the logical "or" operation.

    Returns:
      float: The result of the logical "or" operation, calculated using
          the formula o1 + o2 + sqrt(o1**2 + o2**2).
    """
    return o1 + o2 + jnp.sqrt(o1**2 + o2**2)


def r_trimming(o, t):
    """
    Trimming operations.

    Args:
      o (float): (Possibly signed) smooth distance function.
      t (float): Trimming function.

    Returns:
      float: Positive smooth distance function with zero set
          as intersection of zero set of o and positive part of t.
    """
    return jnp.sqrt(o**2 + (jnp.sqrt(t**2 + o**4) - t) ** 2 / 4)


def signed_to_positive(o):
    """Absolute value using the formula jnp.sqrt(o**2)."""
    return jnp.sqrt(o**2)


def only_positive(o):
    """Set negative regions to zero by manipulating values using the formula (o + jnp.sqrt(o**2)) / 2."""
    return (o + jnp.sqrt(o**2)) / 2


def first_order_normalization(o, grad):
    """Normalizes an unnormalized distance function o with its gradient grad."""
    return o / jnp.sqrt(o**2 + jnp.dot(grad, grad))


def normals_from_normalized_sdf(x, sdf):
    """
    Returns a smooth vector field that equals the surface normals where the normalized smooth distance function sdf equals 0

    Args:
      x (jnp.ndarray): The position at which the vector field shall be evaluated
      sdf (callable): The normalized smooth distance function with argument x

    Returns:
      jnp.ndarray
    """
    return -jax.jacrev(sdf)(x)


### Signed distance functions
def sdf_infinite_line(x, x_p1, x_p2):
    """
    Normalized signed smooth distance function of an infinite line going through x_p1 and x_p2.

    Args:
      x (jnp.ndarray): The point where the distance is evaluated.
      x_p1 (jnp.ndarray): First point on the infinite line.
      x_p2 (jnp.ndarray): Second point on the infinite line.

    Returns:
      float: The signed distance from the point x to the infinite line defined by x_p1 and x_p2.
    """
    x_p1 = jnp.asarray(x_p1)
    x_p2 = jnp.asarray(x_p2)
    line_length = jnp.linalg.norm(x_p2 - x_p1)
    return (
        (x[0] - x_p1[0]) * (x_p2[1] - x_p1[1]) - (x[1] - x_p1[1]) * (x_p2[0] - x_p1[0])
    ) / line_length


def sdf_nd_sphere(x, xc, r):
    """
    Normalized signed distance function to n-dimensional sphere, positive in the interior.

    Args:
      x (jnp.ndarray): The point where the distance is evaluated.
      xc (jnp.ndarray): Center of the sphere.
      r (float): Radius of the sphere.

    Returns:
      float: The normalized signed distance from the point x to the n-dimensional sphere defined by center xc and radius r.
    """
    xc = jnp.asarray(xc)
    dxc = x - xc
    return (r**2 - jnp.dot(dxc, dxc)) / (2 * r)


def sdf_nd_planes(x, x_p1, x_p2):
    """
    Normalized signed smooth distance function, positive between points (1D), lines (2D), and n-dimensional planes (nD).

    Args:
      x (jnp.ndarray): The point where the distance is evaluated.
      x_p1 (jnp.ndarray): Point on the first plane.
      x_p2 (jnp.ndarray): Point on the second plane.

    Returns:
      float: The signed distance from the point x to the planes defined by x_p1 and x_p2.
    """
    x_p1 = jnp.asarray(x_p1)
    x_p2 = jnp.asarray(x_p2)

    xc = centroid(x_p1, x_p2)
    len = jnp.linalg.norm(x_p2 - x_p1)
    dxc = x - xc

    n_plane = normal(x_p1, x_p2)
    dist_to_mid_plane = jnp.dot(dxc, n_plane)

    return (len**2 / 4 - dist_to_mid_plane**2) / len


def sdf_normal_to_direction(x, x_p1, normal):
    """
    Normalized signed smooth distance function to a plane defined by one point and a normal vector, positive in the direction of the normal vector.

    Args:
      x (jnp.ndarray): The point where the distance is evaluated.
      x_p1 (jnp.ndarray): A point on the plane.
      normal (jnp.ndarray): The normal vector of the plane.

    Returns:
      float: The distance from the point x to the plane defined by x_p1 and the normal vector.
    """
    x_p1 = jnp.asarray(x_p1)
    normal = jnp.asarray(normal)

    dist = x - x_p1
    return jnp.dot(dist, normal)


def sdf_infinite_cylinder(x, x_p1, normal, radius):
    """
    Normalized signed smooth distance function, positive within an infinite cylinder in 3D.

    Args:
      x (jnp.ndarray): The point where the distance is evaluated.
      x_p1 (jnp.ndarray): A point on the center line of the cylinder.
      normal (jnp.ndarray): The direction vector of the cylinder.
      radius (float): The radius of the cylinder.

    Returns:
      float: The signed distance from the point x to the infinite cylinder.
    """
    x_p1 = jnp.asarray(x_p1)
    normal = jnp.asarray(normal)

    x_projected_on_centerline = project_on_line(x, x_p1, x_p1 + normal)
    d = x - x_projected_on_centerline
    return (radius**2 - jnp.dot(d, d)) / (2 * radius)


def sdf_cylinder(x, x_0, normal, radius, length):
    """
    Normalized signed smooth distance function, positive within a cylinder in 3D.

    Args:
      x (jnp.ndarray): The point where the distance is evaluated.
      x_0 (jnp.ndarray): The starting point on the center line of the cylinder.
      normal (jnp.ndarray): The direction vector of the cylinder.
      radius (float): The radius of the cylinder.
      length (float): The length of the cylinder.

    Returns:
      float: The signed distance from the point x to the cylinder.
    """
    x_0 = jnp.asarray(x_0)
    normal = jnp.asarray(normal)

    inf_cylinder = sdf_infinite_cylinder(x, x_0, normal, radius)
    region_between_end_pts = sdf_nd_planes(x, x_0, x_0 + length * normal)
    return r_conjunction(inf_cylinder, region_between_end_pts)


def sdf_cylinder_extruded(x, x_0, normal, radius, length, t0, t1):
    """
    Normalized signed smooth distance function of a cylinder that is extruded in time (fourth dimension).

    Args:
      x (jnp.ndarray): The point where the distance is evaluated.
      x_0 (jnp.ndarray): The starting point on the center line of the cylinder.
      normal (jnp.ndarray): The direction vector of the cylinder.
      radius (float): The radius of the cylinder.
      length (float): The length of the cylinder.
      t0 (float): The starting time of the extrusion.
      t1 (float): The ending time of the extrusion.

    Returns:
      float: The signed distance from the point x to the extruded cylinder.
    """
    x_0 = jnp.asarray(x_0)
    normal = jnp.asarray(normal)

    cylinder_in_3d = sdf_cylinder(x[:3], x_0, normal, radius, length)
    region_between_t0_t1 = sdf_nd_planes(
        x, jnp.asarray([0.0, 0.0, 0.0, t0]), jnp.asarray([0.0, 0.0, 0.0, t1])
    )
    return r_conjunction(cylinder_in_3d, region_between_t0_t1)


def sdf_triangle_2d(x, x_p1, x_p2, x_p3):
    """
    Normalized signed smooth distance function to a triangle (2D) or infinite triprism (3D), positive in the interior.

    Args:
      x (jnp.ndarray): The point where the distance is evaluated.
      x_p1 (jnp.ndarray): First vertex of the triangle.
      x_p2 (jnp.ndarray): Second vertex of the triangle.
      x_p3 (jnp.ndarray): Third vertex of the triangle.

    Returns:
      float: The signed distance from the point x to the triangle or triprism.
    """
    x_p1 = jnp.asarray(x_p1)
    x_p2 = jnp.asarray(x_p2)
    x_p3 = jnp.asarray(x_p3)

    projected_1 = project_on_line(x_p1, x_p2, x_p3)
    projected_2 = project_on_line(x_p2, x_p3, x_p1)
    projected_3 = project_on_line(x_p3, x_p1, x_p2)

    o1 = sdf_nd_planes(x, x_p1, projected_1)
    o2 = sdf_nd_planes(x, x_p2, projected_2)
    o3 = sdf_nd_planes(x, x_p3, projected_3)

    return r_conjunction(r_conjunction(o1, o2), o3)


def sdf_convex_polygon_2d(x, x_p_list):
    """
    Normalized signed smooth distance function to a convex polygon (2D), positive in the interior.

    Args:
      x (jnp.ndarray): The point where the distance is evaluated.
      x_p_list (list of jnp.ndarray): List of vertices of the polygon in counter-clockwise order.

    Returns:
      float: The signed distance from the point x to the convex polygon.
    """
    n_segments = len(x_p_list)
    psdf_d = (lambda t: sdf_infinite_line(t, x_p_list[1], x_p_list[0]))(x)

    for i in range(n_segments - 1):
        x_0 = x_p_list[-i]
        x_1 = x_p_list[-(i + 1)]
        psdf_i = (lambda t: sdf_infinite_line(t, x_0, x_1))(x)
        psdf_d = r_conjunction(psdf_d, psdf_i)
    return psdf_d


def sdf_infinite_triprism_3d(x, x_p1, x_p2, x_p3):
    """
    Normalized signed smooth distance function to a triangle (2D) or infinite triprism (3D), positive in the interior.

    Args:
      x (jnp.ndarray): The point where the distance is evaluated.
      x_p1 (jnp.ndarray): First vertex of the triangle.
      x_p2 (jnp.ndarray): Second vertex of the triangle.
      x_p3 (jnp.ndarray): Third vertex of the triangle.

    Returns:
      float: The signed distance from the point x to the triangle or triprism.
    """
    return sdf_triangle_2d(x, x_p1, x_p2, x_p3)


# def sdf_triangle_3d(x, x_p1, x_p2, x_p3):
#   x_p1 = jnp.asarray(x_p1)
#   x_p2 = jnp.asarray(x_p2)
#   x_p3 = jnp.asarray(x_p3)
#   return #ToDo

# def sdf_triprism(x, x_p1, x_p2, x_p3, length):
#   x_p1 = jnp.asarray(x_p1)
#   x_p2 = jnp.asarray(x_p2)
#   x_p3 = jnp.asarray(x_p3)
#   return #ToDo

# def sdf_tetrahedron_3d(x, x_p1, x_p2, x_p3, x_p4):
#   x_p1 = jnp.asarray(x_p1)
#   x_p2 = jnp.asarray(x_p2)
#   x_p3 = jnp.asarray(x_p3)
#   x_p4 = jnp.asarray(x_p4)
#   return #ToDo

# def sdf_tetrahedron_extruded(x, x_p1, x_p2, x_p3, x_p4, t0, t1):
#   x_p1 = jnp.asarray(x_p1)
#   x_p2 = jnp.asarray(x_p2)
#   x_p3 = jnp.asarray(x_p3)
#   x_p4 = jnp.asarray(x_p4)
#   return #ToDo


def sdf_parallelogram_2d(x, x_p1, x_p2, x_p3):
    """
    Normalized signed smooth distance function to a parallelogram, positive in the interior.

    Usage for rectangle with vertices [[0,0],[0,1],[1,1],[1,0]]:
    sdf_parallelogram(x, [0,0], [0,1], [1,0])

    Args:
      x (jnp.ndarray): The point where the distance is evaluated.
      x_p1 (jnp.ndarray): First vertex of the parallelogram.
      x_p2 (jnp.ndarray): Second vertex of the parallelogram.
      x_p3 (jnp.ndarray): Third vertex of the parallelogram.

    Returns:
      float: The signed distance from the point x to the parallelogram.
    """
    x_p1 = jnp.asarray(x_p1)
    x_p2 = jnp.asarray(x_p2)
    x_p3 = jnp.asarray(x_p3)

    projected_2 = project_on_line(x_p2, x_p3, x_p1)
    projected_3 = project_on_line(x_p3, x_p1, x_p2)

    o2 = sdf_nd_planes(x, x_p2, projected_2)
    o3 = sdf_nd_planes(x, x_p3, projected_3)

    return r_conjunction(o2, o3)


# def sdf_infinite_parallelogram_prism(x, x_p1, x_p2, x_p3):
#   x_p1 = jnp.asarray(x_p1)
#   x_p2 = jnp.asarray(x_p2)
#   x_p3 = jnp.asarray(x_p3)
#   return #ToDo


def sdf_cuboid(x, x_p1, x_p2):
    """
    Normalized signed smooth distance function to a cuboid, positive in the interior.

    Args:
      x (jnp.ndarray): The point where the distance is evaluated.
      x_p1 (jnp.ndarray): The point with minimum coordinates of the cuboid.
      x_p2 (jnp.ndarray): The point with maximum coordinates of the cuboid.

    Returns:
      float: The signed distance from the point x to the cuboid.
    """
    x_p1 = jnp.asarray(x_p1)
    x_p2 = jnp.asarray(x_p2)

    project_x = project_on_line(x_p2, x_p1, x_p1 + jnp.asarray([1.0, 0.0, 0.0]))
    project_y = project_on_line(x_p2, x_p1, x_p1 + jnp.asarray([0.0, 1.0, 0.0]))
    project_z = project_on_line(x_p2, x_p1, x_p1 + jnp.asarray([0.0, 0.0, 1.0]))

    o1 = sdf_nd_planes(x, x_p1, project_x)
    o2 = sdf_nd_planes(x, x_p1, project_y)
    o3 = sdf_nd_planes(x, x_p1, project_z)

    return r_conjunction(r_conjunction(o1, o2), o3)


# def sdf_parallelepiped_extruded(x, x_p1, x_p2, x_p3, x_p4, t0, t1):
#   x_p1 = jnp.asarray(x_p1)
#   x_p2 = jnp.asarray(x_p2)
#   x_p3 = jnp.asarray(x_p3)
#   x_p4 = jnp.asarray(x_p4)
#   return #ToDo

# def sdf_3d_ball_extruded(x, xc, r):
#   xc = jnp.asarray(xc)
#   return #ToDo


### Positive smooth distance functions
def psdf_infinite_line(x, x_p1, x_p2):
    """
    Positive normalized smooth distance function of an infinite line going through x_p1 and x_p2.

    Args:
      x (jnp.ndarray): The point where the distance is evaluated.
      x_p1 (jnp.ndarray): First point on the infinite line.
      x_p2 (jnp.ndarray): Second point on the infinite line.

    Returns:
      float: The positive distance from the point x to the infinite line defined by x_p1 and x_p2.
    """
    x_p1 = jnp.asarray(x_p1)
    x_p2 = jnp.asarray(x_p2)

    dx1 = x - x_p1
    normal_vector_in_line_direction = normal(x_p1, x_p2)
    normal_component = dx1 - project(dx1, normal_vector_in_line_direction)
    return jnp.linalg.norm(normal_component)


def psdf_trimmed_line(x, x_p1, x_p2):
    """
    Positive normalized smooth distance function of a line segment going from x_p1 to x_p2.

    Args:
      x (jnp.ndarray): The point where the distance is evaluated.
      x_p1 (jnp.ndarray): Starting point of the line segment.
      x_p2 (jnp.ndarray): Ending point of the line segment.

    Returns:
      float: The positive distance from the point x to the line segment defined by x_p1 and x_p2.
    """
    x_p1 = jnp.asarray(x_p1)
    x_p2 = jnp.asarray(x_p2)

    omega = sdf_infinite_line(x, x_p1, x_p2)

    xc = centroid(x_p1, x_p2)
    d = x_p2 - x_p1
    len = jnp.linalg.norm(d)
    trim = sdf_nd_sphere(x, xc, len / 2)

    return r_trimming(omega, trim)


def psdf_nd_sphere(x, xc, r):
    """
    Interior positive normalized smooth distance function of an n-dimensional sphere.

    Args:
      x (jnp.ndarray): The point where the distance is evaluated.
      xc (jnp.ndarray): Center of the sphere.
      r (float): Radius of the sphere.

    Returns:
      float: The positive distance from the point x to the region outside of a n-dimensional sphere.
    """
    xc = jnp.asarray(xc)
    return only_positive(sdf_nd_sphere(x, xc, r))


def psdf_nd_sphere_cutout(x, xc, r):
    """
    Exterior positive normalized smooth distance function of an n-dimensional sphere.

    Args:
      x (jnp.ndarray): The point where the distance is evaluated.
      xc (jnp.ndarray): Center of the sphere.
      r (float): Radius of the sphere.

    Returns:
      float: The positive distance from the point x to the filled n-dimensional sphere.
    """
    xc = jnp.asarray(xc)
    return only_positive(-sdf_nd_sphere(x, xc, r))


def psdf_infinite_cylinder(x, x_p1, normal, radius):
    """
    Positive normalized smooth distance function to an infinite cylinder in 3D.

    Args:
      x (jnp.ndarray): The point where the distance is evaluated.
      x_p1 (jnp.ndarray): A point on the center line of the cylinder.
      normal (jnp.ndarray): The direction vector of the cylinder.
      radius (float): The radius of the cylinder.

    Returns:
      float: The positive distance from the point x to the infinite cylinder. Zero outside of the cylinder.
    """
    x_p1 = jnp.asarray(x_p1)
    normal = jnp.asarray(normal)
    return only_positive(sdf_infinite_cylinder(x, x_p1, normal, radius))


def psdf_cylinder(x, x_0, normal, radius, length):
    """
    Positive normalized smooth distance function to a cylinder in 3D.

    Args:
      x (jnp.ndarray): The point where the distance is evaluated.
      x_0 (jnp.ndarray): The starting point on the center line of the cylinder.
      normal (jnp.ndarray): The direction vector of the cylinder.
      radius (float): The radius of the cylinder.
      length (float): The length of the cylinder.

    Returns:
      float: The positive distance from the point x to the cylinder. Zero outside of the cylinder.
    """
    x_0 = jnp.asarray(x_0)
    normal = jnp.asarray(normal)
    return only_positive(sdf_cylinder(x, x_0, normal, radius, length))


def psdf_cylinder_extruded(x, x_0, normal, radius, length, t0, t1):
    """
    Positive normalized smooth distance function of a cylinder that is extruded in time.

    Args:
      x (jnp.ndarray): The point where the distance is evaluated.
      x_0 (jnp.ndarray): The starting point on the center line of the cylinder.
      normal (jnp.ndarray): The direction vector of the cylinder.
      radius (float): The radius of the cylinder.
      length (float): The length of the cylinder.
      t0 (float): The starting time of the extrusion.
      t1 (float): The ending time of the extrusion.

    Returns:
      float: The positive distance from the point x to the extruded cylinder. Zero outside of the cylinder.
    """
    x_0 = jnp.asarray(x_0)
    normal = jnp.asarray(normal)
    return only_positive(sdf_cylinder_extruded(x, x_0, normal, radius, length, t0, t1))


def psdf_triangle_2d(x, x_p1, x_p2, x_p3):
    """
    Interior positive normalized smooth distance function of a triangle in 2D and a tri-prism in 3D.

    Args:
      x (jnp.ndarray): The point where the distance is evaluated.
      x_p1 (jnp.ndarray): First vertex of the triangle.
      x_p2 (jnp.ndarray): Second vertex of the triangle.
      x_p3 (jnp.ndarray): Third vertex of the triangle.

    Returns:
      float: The positive distance from the point x to the region outside of the triangle or tri-prism.
    """
    x_p1 = jnp.asarray(x_p1)
    x_p2 = jnp.asarray(x_p2)
    x_p3 = jnp.asarray(x_p3)
    return only_positive(sdf_triangle_2d(x, x_p1, x_p2, x_p3))


def psdf_infinite_triprism_3d(x, x_p1, x_p2, x_p3):
    """
    Interior positive normalized smooth distance function of a triangle in 2D and a tri-prism in 3D.

    Args:
      x (jnp.ndarray): The point where the distance is evaluated.
      x_p1 (jnp.ndarray): First vertex of the triangle.
      x_p2 (jnp.ndarray): Second vertex of the triangle.
      x_p3 (jnp.ndarray): Third vertex of the triangle.

    Returns:
      float: The positive distance from the point x to the region outside of the triangle or tri-prism.
    """
    x_p1 = jnp.asarray(x_p1)
    x_p2 = jnp.asarray(x_p2)
    x_p3 = jnp.asarray(x_p3)
    return psdf_triangle_2d(x, x_p1, x_p2, x_p3)


# def psdf_triprism(x, x_p1, x_p2, x_p3, length):
#   x_p1 = jnp.asarray(x_p1)
#   x_p2 = jnp.asarray(x_p2)
#   x_p3 = jnp.asarray(x_p3)
#   return #ToDo


def psdf_parallelogram(x, x_p1, x_p2, x_p3):
    """
    Positive normalized smooth distance function to a parallelogram, positive in the interior.

    Usage for rectangle with vertices [[0,0],[0,1],[1,1],[1,0]]:
    psdf_parallelogram(x, [0,0], [0,1], [1,0])

    Args:
      x (jnp.ndarray): The point where the distance is evaluated.
      x_p1 (jnp.ndarray): First vertex of the parallelogram.
      x_p2 (jnp.ndarray): Second vertex of the parallelogram.
      x_p3 (jnp.ndarray): Third vertex of the parallelogram.

    Returns:
      float: The positive distance from the point x to the region outside of the parallelogram.
    """
    x_p1 = jnp.asarray(x_p1)
    x_p2 = jnp.asarray(x_p2)
    x_p3 = jnp.asarray(x_p3)
    return only_positive(sdf_parallelogram_2d(x, x_p1, x_p2, x_p3))


def psdf_arc_in_2d(x, xc, x_p1, x_p2):
    """
    First order normalized positive smooth distance function of a circular arc in 2D.

    Args:
      xc (jnp.ndarray): Center of the arc.
      x_p1 (jnp.ndarray): Start point of the arc.
      x_p2 (jnp.ndarray): End point of the arc (counter-clockwise).

    Returns:
      float: The normalized smooth distance from the point x to the circular arc.
    """
    x_p1 = jnp.asarray(x_p1)
    x_p2 = jnp.asarray(x_p2)
    xc = jnp.asarray(xc)
    [x1, y1] = x_p1
    [x2, y2] = x_p2
    radius = jnp.linalg.norm(xc - x_p1)

    # Distance and trimming function
    o = sdf_nd_sphere(x, xc, radius)
    t = ((x[0] - x1) * (y2 - y1) - (x[1] - y1) * (x2 - x1)) / jnp.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2
    )

    return r_trimming(o, t)


def psdf_polygon(x, x_p_list):
    """
    Positive normalized smooth distance function for a polygon in 2D.

    Args:
      x (jnp.ndarray): The point where the distance is evaluated.
      x_p_list (list of jnp.ndarray): List of edge coordinates of the polygon.

    Returns:
      float: The positive distance from the point x to the polygon boundary. (Positive inside and outside)
    """
    x_p_list = jnp.asarray(x_p_list)
    n_segments = x_p_list.shape[0]
    psdf_d = (lambda t: psdf_trimmed_line(t, x_p_list[0], x_p_list[1]))(x)

    def body_fun(i, psdf_d):
        x_0 = x_p_list[-i]
        x_1 = x_p_list[-(i + 1)]
        psdf_i = (lambda t: psdf_trimmed_line(t, x_0, x_1))(x)
        return r_equivalence(psdf_d, psdf_i)

    return jax.lax.fori_loop(0, n_segments - 1, body_fun, psdf_d)


### Helper functions
def centroid(x_p1, x_p2):
    """
    Mean of two points in n-dimensions:

    result = (x_p1 + x_p2) / 2
    """
    return (x_p1 + x_p2) / 2


def normal(x_p1, x_p2):
    """
    Normalized vector showing from x_p1 to x_p2.

    result = (x_p2 - x_p1) / jnp.linalg.norm(x_p2 - x_p1)
    """
    return (x_p2 - x_p1) / jnp.linalg.norm(x_p2 - x_p1)


def project(x, normal):
    """
    Projects vector x in the direction of vector normal.

    result = jnp.dot(x,normal) * normal
    """
    return jnp.dot(x, normal) * normal


def project_on_line(x_p1, x_p2, x_p3):
    """
    Projects x_p1 on the line going through x_p2 and x_p3.

    result = x_p2 + project(x_p1 - x_p2, normal(x_p2, x_p3))
    """
    return x_p2 + project(x_p1 - x_p2, normal(x_p2, x_p3))



### Transfinite interpolation
def transfinite_interpolation(x, boundary_psdfs, boundary_conditions):
    """
    Transfinite interpolation using positive smooth distance functions (PSDFs) and boundary conditions.

    Args:
      x (jnp.ndarray): The point where the interpolation is evaluated.
      boundary_psdfs (list of callable): List of PSDF functions, each defining a boundary.
      boundary_conditions (list of callable): List of boundary condition functions corresponding to the PSDF functions.

    Returns:
      jnp.ndarray: The interpolated value at point x based on the boundary PSDFs and boundary conditions.

    Notes:
      This function uses an unstable version of transfinite interpolation at the boundary.
      Usually one can simply delete all integration points (and in case of collocation also nodes) on the boundary.
      A more stable but inefficient version (commented out) can be used if stability issues arise.
    """
    psdf_evaluated = jnp.stack([psdf_i(x) for psdf_i in boundary_psdfs])
    bc_evaluated = jnp.stack([bc_i(x) for bc_i in boundary_conditions])

    # Instable version (at boundary)
    psdf_inv = 1.0 / psdf_evaluated
    psdf_inv_sum = psdf_inv.sum()
    bc_d = jnp.dot(psdf_inv, bc_evaluated) / psdf_inv_sum
    return bc_d

    # # Stable version
    # denom = 0
    # for k in range(len(psdf_evaluated)):
    #   wk = 1
    #   for (j,psdf_j) in enumerate(psdf_evaluated):
    #     if k != j:
    #       wk = wk * psdf_j
    #   denom = denom + wk

    # interpolated = 0
    # for i in range(len(psdf_evaluated)):
    #   wi = 1
    #   for (j,psdf_j) in enumerate(psdf_evaluated):
    #     if i != j:
    #       wi = wi * psdf_j
    #   interpolated = interpolated + wi * bc_evaluated[i]
    # return interpolated / denom


def psdf_unification(x, boundary_psdfs):
    """
    Unification of multiple positive smooth distance functions (PSDFs).

    Args:
      x (jnp.ndarray): The point where the unification is evaluated.
      boundary_psdfs (list of callable): List of PSDF functions to be unified.

    Returns:
      float: The unified PSDF value at point x.

    Notes:
      The unification process combines the PSDFs using the r_equivalence relation.
    """
    psdf_d = boundary_psdfs[0](x)
    for psdf_i in boundary_psdfs[1:]:
        psdf_d = r_equivalence(psdf_d, psdf_i(x))

    return psdf_d


### Mesh related functions
@jit_with_docstring()
def triangle_area(x_edges):
    """
    Compute the area of a triangle given its vertices (jitted).

    Args:
      x_edges (jnp.ndarray): Array of shape (3, 2) containing the coordinates of the triangle's vertices.

    Returns:
      float: The area of the triangle.
    """
    x_0 = x_edges[0]
    x_1 = x_edges[1]
    x_2 = x_edges[2]

    x_2_projected = project_on_line(x_2, x_0, x_1)
    l = jnp.linalg.norm(x_1 - x_0)
    h = jnp.linalg.norm(x_2 - x_2_projected)
    area = h * l / 2
    return area


@jit_with_docstring()
def triangle_areas(x_nodes, elements):
    """
    Compute the areas of multiple triangles given their vertices and connectivity (jitted).

    Args:
      x_nodes (jnp.ndarray): Array of shape (n, 2) containing the coordinates of the nodes.
      elements (jnp.ndarray): Array of shape (m, 3) containing the indices of the nodes forming each triangle.

    Returns:
      jnp.ndarray: Array of shape (m,) containing the areas of the triangles.
    """
    areas = jax.vmap(triangle_area, (0), 0)(x_nodes[elements])
    return areas


def in_sdf(x, sdf_fun, tol = 1e-6):
    """
    Check if a point lies on the surface defined by a positive or signed distance function.

    Args:
      x (jnp.ndarray): The point to be checked.
      sdf_fun (callable): The distance function.
      tol (float): The tolerance for isclose check.

    Returns:
      bool: True if the point lies on the surface, False otherwise.
    """
    x = jnp.asarray(x)
    sdf = sdf_fun(x)
    return jnp.where(jnp.isclose(sdf, 0, atol=tol), True, False)


@jit_with_docstring(static_argnames=["sdf_fun"])
def in_sdfs(x, sdf_fun, tol = 1e-6):
    """
    Check if multiple points lie on the surface defined by a positive or signed distance function.

    Args:
      x (jnp.ndarray): Array of points to be checked.
      sdf_fun (callable): The signed distance function.
      tol (float): The tolerance for isclose check.

    Returns:
        jnp.ndarray: Array of booleans indicating whether each point lies on the surface.
    """
    x = jnp.asarray(x)
    return jax.vmap(in_sdf, (0, None, None), 0)(x, sdf_fun, tol)


def select_in_sdfs(x, sdf_fun):
    """
    Select the indices of points that lie on the surface defined by a positive or signed distance function.

    Args:
      x (jnp.ndarray): Array of points to be checked.
      sdf_fun (callable): The signed distance function.

    Returns:
      jnp.ndarray: Array of indices of points that lie on the surface.
    """
    x = jnp.asarray(x)
    return jnp.compress(
        in_sdfs(x, sdf_fun), jnp.arange(x.shape[0], dtype=jnp.int_), axis=0
    )


@jit_with_docstring()
def in_plane(x, x_p1, normal):
    """
    Check if a point lies in a plane defined by a point and a normal vector (jitted).

    Args:
      x (jnp.ndarray): The point to be checked.
      x_p1 (jnp.ndarray): A point on the plane.
      normal (jnp.ndarray): The normal vector of the plane.

    Returns:
      bool: True if the point lies in the plane, False otherwise.
    """
    sdf_fun = lambda x: sdf_normal_to_direction(x, x_p1, normal)
    return in_sdf(jnp.asarray(x), sdf_fun)


@jit_with_docstring()
def in_planes(x_q, x_p1, normal):
    """
    Check if multiple points lie in a plane defined by a point and a normal vector (jitted).

    Args:
      x_q (jnp.ndarray): Array of points to be checked.
      x_p1 (jnp.ndarray): A point on the plane.
      normal (jnp.ndarray): The normal vector of the plane.

    Returns:
      jnp.ndarray: Array of booleans indicating whether each point lies in the plane.
    """
    return jax.vmap(in_plane, (0, None, None), 0)(x_q, x_p1, normal)


def select_in_plane(x_q, x_p1, normal):
    """
    Select the indices of points that lie in a plane (non-jittable).

    Args:
      x_q (jnp.ndarray): Array of points to be checked.
      x_p1 (jnp.ndarray): A point on the plane.
      normal (jnp.ndarray): The normal vector of the plane.

    Returns:
      jnp.ndarray: Array of indices of points that lie in the plane.
    """
    return jnp.compress(
        in_planes(x_q, x_p1, normal), jnp.arange(x_q.shape[0], dtype=jnp.int_), axis=0
    )


@jit_with_docstring()
def on_point(x_q, x_p1):
    """
    Check if points are close to a given point (jitted).

    Args:
      x_q (jnp.ndarray): Array of points to be checked.
      x_p1 (jnp.ndarray): The reference point.

    Returns:
      jnp.ndarray: Array of booleans indicating whether each point is close to the reference point.
    """
    x_p1 = jnp.asarray(x_p1)
    x_q = jnp.asarray(x_q)

    def one_check(x_q, x_p1):
        sdf = jnp.linalg.norm(x_q - x_p1)
        return jnp.where(jnp.isclose(sdf, 0), True, False)

    map_check = lambda x_q, x_p1: jax.jit(jax.vmap(one_check, (0, None), 0))(x_q, x_p1)
    return map_check(x_q, x_p1)


def select_point(x_q, x_p1):
    """
    Select the index of a point that is close to a given point (non-jittable).

    Args:
      x_q (jnp.ndarray): Array of points to be checked.
      x_p1 (jnp.ndarray): The reference point.

    Returns:
      int: The index of the point that is close to the reference point.
    """
    return jnp.compress(
        on_point(x_q, x_p1), jnp.arange(x_q.shape[0], dtype=jnp.int_), axis=0
    )[0]


@jit_with_docstring()
def on_line(x, x_p1, x_p2):
    """
    Check if a point lies on a line segment defined by two points (jitted).

    Args:
      x (jnp.ndarray): The point to be checked.
      x_p1 (jnp.ndarray): The starting point of the line segment.
      x_p2 (jnp.ndarray): The ending point of the line segment.

    Returns:
      bool: True if the point lies on the line segment, False otherwise.
    """
    sdf_fun = lambda x: psdf_trimmed_line(x, x_p1, x_p2)
    return in_sdf(jnp.asarray(x), sdf_fun)


@jit_with_docstring()
def on_lines(x_q, x_p1, x_p2):
    """
    Check if multiple points lie on a line segment defined by two points (jitted).

    Args:
      x_q (jnp.ndarray): Array of points to be checked.
      x_p1 (jnp.ndarray): The starting point of the line segment.
      x_p2 (jnp.ndarray): The ending point of the line segment.

    Returns:
      jnp.ndarray: Array of booleans indicating whether each point lies on the line segment.
    """
    return jax.vmap(on_line, (0, None, None), 0)(x_q, x_p1, x_p2)


def select_on_line(x_q, x_p1, x_p2):
    """
    Select the indices of points that lie on a line segment defined by two points (non-jittable).

    Args:
      x_q (jnp.ndarray): Array of points to be checked.
      x_p1 (jnp.ndarray): The starting point of the line segment.
      x_p2 (jnp.ndarray): The ending point of the line segment.

    Returns:
      jnp.ndarray: Array of indices of points that lie on the line segment.
    """
    return jnp.compress(
        on_lines(x_q, x_p1, x_p2), jnp.arange(x_q.shape[0], dtype=jnp.int_), axis=0
    )


def elem_in_plane(x_nodes, x_p1, normal):
    """
    Check if an element lies in a plane defined by a point and a normal vector.

    Args:
      x_nodes (jnp.ndarray): Array of nodes of the element.
      x_p1 (jnp.ndarray): A point on the plane.
      normal (jnp.ndarray): The normal vector of the plane.

    Returns:
      bool: True if the element lies in the plane, False otherwise.
    """
    x_nodes = jnp.asarray(x_nodes)
    x_p1 = jnp.asarray(x_p1)
    normal = jnp.asarray(normal)
    return in_planes(x_nodes, x_p1, normal).all()


def select_elements_in_plane(x_nodes, surface_elements, x_p1, normal):
    """
    Select the indices of elements that lie in a plane.

    Args:
      x_nodes (jnp.ndarray): Array of nodes of the elements.
      surface_elements (jnp.ndarray): Array of surface elements.
      x_p1 (jnp.ndarray): A point on the plane.
      normal (jnp.ndarray): The normal vector of the plane.

    Returns:
      jnp.ndarray: Array of indices of elements that lie in the plane.
    """
    x_nodes = jnp.asarray(x_nodes)
    surface_elements = jnp.asarray(surface_elements)
    normal = jnp.asarray(normal)
    x_p1 = jnp.asarray(x_p1)
    return jax.vmap(elem_in_plane, (0, None, None), 0)(
        x_nodes[surface_elements], x_p1, normal
    )


def elem_on_line(x_nodes, x_p1, x_p2):
    """
    Check if an element lies on a line segment defined by two points.

    Args:
      x_nodes (jnp.ndarray): Array of nodes of the element.
      x_p1 (jnp.ndarray): The starting point of the line segment.
      x_p2 (jnp.ndarray): The ending point of the line segment.

    Returns:
      bool: True if the element lies on the line segment, False otherwise.
    """
    x_nodes = jnp.asarray(x_nodes)
    x_p1 = jnp.asarray(x_p1)
    x_p2 = jnp.asarray(x_p2)
    return on_lines(x_nodes, x_p1, x_p2).all()


def select_elements_on_line(x_nodes, surface_elements, x_p1, x_p2):
    """
    Select the indices of elements that lie on a line segment defined by two points.

    Args:
      x_nodes (jnp.ndarray): Array of nodes of the elements.
      surface_elements (jnp.ndarray): Array of surface elements.
      x_p1 (jnp.ndarray): The starting point of the line segment.
      x_p2 (jnp.ndarray): The ending point of the line segment.

    Returns:
      jnp.ndarray: Array of indices of elements that lie on the line segment.
    """
    x_nodes = jnp.asarray(x_nodes)
    surface_elements = jnp.asarray(surface_elements)
    x_p1 = jnp.asarray(x_p1)
    x_p2 = jnp.asarray(x_p2)
    return jax.vmap(elem_on_line, (0, None, None), 0)(
        x_nodes[surface_elements], x_p1, x_p2
    )


@jit_with_docstring()
def subelem_in_elem(subelem_to_find, in_elem):
    """
    Check if a sub-element (e.g. a line element) is contained within an element (e.g. triangular element) (jitted).

    Args:
      subelem_to_find (jnp.ndarray): The sub-element to be checked.
      in_elem (jnp.ndarray): The element to be checked against.

    Returns:
      bool: True if the sub-element is contained within the element, False otherwise.
    """
    return jnp.all(jnp.isin(subelem_to_find, in_elem))


def subelem_in_elems(subelem_to_find, in_elems):
    """
    Find the element that contains a given sub-element. Returns only one element (Only to be used for these cases).

    Args:
      subelem_to_find (jnp.ndarray): The sub-element to be found.
      in_elems (jnp.ndarray): Array of elements to be checked.

    Returns:
      jnp.ndarray: The element that contains the sub-element.
    """
    selection = jax.jit(jax.vmap(subelem_in_elem, (None, 0), 0))(
        subelem_to_find, in_elems
    )
    compression = jnp.compress(selection, in_elems, axis=0)[0]
    return compression


def subelems_in_elems(subelems_to_find, in_elems):
    """
    Find the elements that contain given sub-elements (one elem per sub_elem).

    Can be used for detecting the domain elements that belong the given surface elements.

    Args:
      subelems_to_find (jnp.ndarray): Array of sub-elements to be found.
      in_elems (jnp.ndarray): Array of elements to be checked.

    Returns:
      jnp.ndarray: Array of elements that contain the sub-elements.
    """
    return jnp.asarray(
        [subelem_in_elems(subelem, in_elems) for subelem in subelems_to_find]
    )
    # return jax.vmap(subelem_in_elems, (0, None), 0)(subelems_to_find, in_elems)
