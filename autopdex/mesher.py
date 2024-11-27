# mesher.py
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
Module for generation of meshes. 

Currently only support first order quadrilateral meshes on quadrilateral domains.
"""

import jax
import jax.numpy as jnp

from autopdex.utility import jit_with_docstring

@jit_with_docstring(static_argnames=["n_elements", "type"])
def structured_mesh(n_elements, vertices, type, order=1):
    """Generates a structured 2D mesh over a quadrilateral domain.

    Args:
      n_elements (tuple of int): A tuple `(nx, ny)` specifying the number of elements (divisions) along the x-axis (`nx`) and y-axis (`ny`).
      vertices (array-like): A 4x2 array defining the coordinates of the domain's corner points in anti-clockwise order.
      type (str): The element type to use for the mesh. Currently, only quadrilateral meshes (`'quad'`) are supported.
      order (int, optional): The polynomial order of the elements. Higher orders are not yet implemented.

    Returns:
      (coords, elements)
        - coords (jnp.ndarray): A 2D array of shape `((nx + 1) * (ny + 1), 2)` containing the (x, y) coordinates of each node in the mesh.
        - elements (jnp.ndarray): A 2D array of shape `(nx * ny, 4)` containing the vertex indices for each quadrilateral element in the mesh. The indices refer to the positions in the `coords` array.
    """
    (nx, ny) = n_elements
    vertices = jnp.asarray(vertices)

    # Generate x and y coordinates for the grid points
    x = jnp.linspace(-1, 1, nx + 1)
    y = jnp.linspace(-1, 1, ny + 1)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    coords = jnp.column_stack([X.ravel(), Y.ravel()])

    # Define a function for bilinear interpolation
    def bilinear_interpolate(coors):
        s, t = coors
        return (
            (1 - s) * (1 - t) * vertices[0]
            + (1 + s) * (1 - t) * vertices[1]
            + (1 + s) * (1 + t) * vertices[2]
            + (1 - s) * (1 + t) * vertices[3]
        ) / 4

    coords = jax.vmap(bilinear_interpolate)(coords)

    # Function to generate the vertex indices for a quadrilateral element
    def create_quad(j, i):
        return jnp.array(
            [
                j * (nx + 1) + i,
                j * (nx + 1) + (i + 1),
                (j + 1) * (nx + 1) + (i + 1),
                (j + 1) * (nx + 1) + i,
            ]
        )

    # Create a grid of indices for all quadrilateral elements
    I, J = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing="ij")  # Shape (nx, ny)
    I_flat = I.flatten()
    J_flat = J.flatten()

    # Apply the create_quad function to each (j, i) pair using vectorized mapping
    elements = jax.vmap(create_quad)(J_flat, I_flat)

    return coords, elements
