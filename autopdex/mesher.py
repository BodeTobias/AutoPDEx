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

Currently only support first order meshes on quadrilateral and hexahedral domains.
For complex meshes, consider using e.g. GMSH. The necessary information that is needed 
by autopdex is the connectivity of the mesh, i.e. a list of node IDs that form each element 
and the coordinates of the nodes in the order of the IDs.
"""

import jax
import jax.numpy as jnp
import numpy as np

from autopdex.utility import jit_with_docstring

@jit_with_docstring(static_argnames=["n_elements", "element_type", "order"])
def structured_mesh(n_elements, vertices, element_type, order=1):
  """
    Generate a structured 2D or 3D mesh over a quadrilateral or hexahedral domain,
    with an optional subdivision into simplex (triangular/tetrahedral) elements.

    Parameters
    ----------
    n_elements : tuple of int
        For 2D, a tuple (nx, ny) specifying the number of elements along the x-
        and y-directions. For 3D, a tuple (nx, ny, nz).
    vertices : array-like
        For 2D, a 4x2 array; for 3D, an 8x3 array defining the coordinates of the
        domain's corner points.
    element_type : str
        For 2D, either "quad" or "tri"; for 3D, either "brick" or "tet".
    order : int, optional
        The polynomial order of the elements. Currently, only linear elements
        (order == 1) are supported. Default is 1.

    Returns
    -------
    coords : jnp.ndarray
        An array of node coordinates. Its shape is ((n+1)*... x dim), where 'dim'
        represents the spatial dimension (2 or 3).
    elements : jnp.ndarray
        An array of element connectivity. For quadrilaterals/brick elements each row
        lists the indices of the nodes forming the element, while for simplex
        elements each row lists the indices forming a triangle (3 nodes) or tetrahedron (4 nodes).

    Notes
    -----
    For 2D:
        - The vertices should be provided as a 4x2 array in anti-clockwise order.
        - The mapping from a reference square ([-1, 1] x [-1, 1]) to the physical
          domain is performed using bilinear interpolation.
    
    For 3D:
        - The vertices should be provided as an 8x3 array, with the ordering corresponding
          to a standard hexahedron (e.g., starting with (-1, -1, -1) and proceeding in an
          anti-clockwise fashion on the bottom face, then defining the top face).
        - The mapping from a reference cube ([-1, 1]^3) to the physical domain is done via
          trilinear interpolation.

    Currently, only linear (order == 1) elements are supported.
  """
  vertices = jnp.asarray(vertices)
  dim = vertices.shape[1]

  if order != 1:
    raise NotImplementedError("Only order==1 is implemented at this moment.")

  # ----- 2D Mesh Generation -----
  if dim == 2:
    if element_type not in ["quad", "tri"]:
      raise NotImplementedError("For 2D, element_type must be either 'quad' or 'tri'.")
    nx, ny = n_elements

    # Create a reference grid in [-1,1] x [-1,1]
    s = jnp.linspace(-1, 1, nx + 1)
    t = jnp.linspace(-1, 1, ny + 1)
    S, T = jnp.meshgrid(s, t, indexing="ij")
    ref_coords = jnp.column_stack([S.ravel(), T.ravel()])

    # Bilinear mapping from reference coordinates to physical coordinates.
    def bilinear_interpolate(pt):
      s, t = pt
      return ((1 - s) * (1 - t) * vertices[0] + (1 + s) * (1 - t) * vertices[1] + (1 + s) * (1 + t) * vertices[2] +
              (1 - s) * (1 + t) * vertices[3]) / 4

    coords = jax.vmap(bilinear_interpolate)(ref_coords)

    # Function to generate connectivity for one quadrilateral element.
    def quad_indices(i, j):
      return jnp.array([
          i * (ny + 1) + j,
          i * (ny + 1) + (j + 1),
          (i + 1) * (ny + 1) + (j + 1),
          (i + 1) * (ny + 1) + j,
      ])

    I, J = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), indexing="ij")
    I_flat = I.ravel()
    J_flat = J.ravel()
    quads = jax.vmap(quad_indices)(I_flat, J_flat)

    if element_type == "quad":
      elements = quads
    else:  # element_type == "tri"
      # Split each quadrilateral into two triangles.
      # Here we split along the diagonal from the first to the third node.
      tri1 = quads[:, [0, 1, 2]]
      tri2 = quads[:, [0, 2, 3]]
      elements = jnp.concatenate([tri1, tri2], axis=0)

    return coords, elements

  # ----- 3D Mesh Generation -----
  elif dim == 3:
    if element_type not in ["brick", "tet"]:
      raise NotImplementedError("For 3D, element_type must be either 'brick' or 'tet'.")
    nx, ny, nz = n_elements

    # Create a reference grid in [-1,1] x [-1,1] x [-1,1]
    s = jnp.linspace(-1, 1, nx + 1)
    t = jnp.linspace(-1, 1, ny + 1)
    u = jnp.linspace(-1, 1, nz + 1)
    S, T, U = jnp.meshgrid(s, t, u, indexing="ij")
    ref_coords = jnp.column_stack([S.ravel(), T.ravel(), U.ravel()])

    # Trilinear mapping from reference coordinates to physical coordinates.
    def trilinear_interpolate(pt):
      s, t, u = pt
      return ((1 - s) * (1 - t) * (1 - u) * vertices[0] + (1 + s) * (1 - t) * (1 - u) * vertices[1] + (1 + s) *
              (1 + t) * (1 - u) * vertices[2] + (1 - s) * (1 + t) * (1 - u) * vertices[3] + (1 - s) * (1 - t) *
              (1 + u) * vertices[4] + (1 + s) * (1 - t) * (1 + u) * vertices[5] + (1 + s) * (1 + t) *
              (1 + u) * vertices[6] + (1 - s) * (1 + t) * (1 + u) * vertices[7]) / 8

    coords = jax.vmap(trilinear_interpolate)(ref_coords)

    # Function to generate connectivity for one brick element.
    # We assume the local node ordering for the brick is:
    # n0: (i, j, k)
    # n1: (i+1, j, k)
    # n2: (i+1, j+1, k)
    # n3: (i, j+1, k)
    # n4: (i, j, k+1)
    # n5: (i+1, j, k+1)
    # n6: (i+1, j+1, k+1)
    # n7: (i, j+1, k+1)
    def brick_indices(i, j, k):
      n0 = i * (ny + 1) * (nz + 1) + j * (nz + 1) + k
      n1 = (i + 1) * (ny + 1) * (nz + 1) + j * (nz + 1) + k
      n2 = (i + 1) * (ny + 1) * (nz + 1) + (j + 1) * (nz + 1) + k
      n3 = i * (ny + 1) * (nz + 1) + (j + 1) * (nz + 1) + k
      n4 = i * (ny + 1) * (nz + 1) + j * (nz + 1) + (k + 1)
      n5 = (i + 1) * (ny + 1) * (nz + 1) + j * (nz + 1) + (k + 1)
      n6 = (i + 1) * (ny + 1) * (nz + 1) + (j + 1) * (nz + 1) + (k + 1)
      n7 = i * (ny + 1) * (nz + 1) + (j + 1) * (nz + 1) + (k + 1)
      return jnp.array([n0, n1, n2, n3, n4, n5, n6, n7])

    I, J, K = jnp.meshgrid(jnp.arange(nx), jnp.arange(ny), jnp.arange(nz), indexing="ij")
    I_flat = I.ravel()
    J_flat = J.ravel()
    K_flat = K.ravel()
    bricks = jax.vmap(brick_indices)(I_flat, J_flat, K_flat)

    if element_type == "brick":
      elements = bricks
    else:  # element_type == "tet"
      # For each brick element, subdivide it into 6 tetrahedra.
      # One common partition uses the opposite corners n0 and n6 as common vertices:
      #   tet1: [n0, n1, n2, n6]
      #   tet2: [n0, n2, n3, n6]
      #   tet3: [n0, n3, n7, n6]
      #   tet4: [n0, n7, n4, n6]
      #   tet5: [n0, n4, n5, n6]
      #   tet6: [n0, n5, n1, n6]
      def brick_to_tets(brick):
        n0, n1, n2, n3, n4, n5, n6, n7 = brick
        tet1 = jnp.array([n0, n1, n2, n6])
        tet2 = jnp.array([n0, n2, n3, n6])
        tet3 = jnp.array([n0, n3, n7, n6])
        tet4 = jnp.array([n0, n7, n4, n6])
        tet5 = jnp.array([n0, n4, n5, n6])
        tet6 = jnp.array([n0, n5, n1, n6])
        return jnp.stack([tet1, tet2, tet3, tet4, tet5, tet6])

      # Map the subdivision over all brick elements.
      tets_list = jax.vmap(brick_to_tets)(bricks)
      # Reshape so that each tetrahedron is a row.
      n_bricks = bricks.shape[0]
      elements = tets_list.reshape(n_bricks * 6, 4)

    return coords, elements

  else:
    raise ValueError("Unsupported dimension: vertices must have 2 or 3 columns.")

def _elevate_order_triangle(coords, elements):
  """
    Elevates a linear (3-node) triangular mesh to quadratic (6-node) triangles.
    
    Args:
        coords (array-like): (N,2) array of node coordinates.
        elements (array-like): (M,3) array of triangle connectivity (node indices).
    
    Returns:
        new_coords (np.ndarray): Updated node coordinates.
        new_elements (np.ndarray): (M,6) connectivity for quadratic triangles.
    """
  coords = np.array(coords)
  elements = np.array(elements)

  new_coords = list(coords)  # start with the original nodes
  edge_dict = {}  # to store computed midpoints
  new_elements = []

  for tri in elements:
    n0, n1, n2 = tri

    def get_midpoint(i, j):
      key = tuple(sorted((i, j)))
      if key not in edge_dict:
        mid = 0.5 * (coords[i] + coords[j])
        edge_dict[key] = len(new_coords)
        new_coords.append(mid)
      return edge_dict[key]

    m01 = get_midpoint(n0, n1)
    m12 = get_midpoint(n1, n2)
    m20 = get_midpoint(n2, n0)

    new_elements.append([n0, n1, n2, m01, m12, m20])

  return np.array(new_coords), np.array(new_elements)

def _elevate_order_brick(coords, elements):
  """
    Elevates a linear (8-node) brick mesh to quadratic bricks (27 nodes)
       
    Args:
        coords (array-like): (N,3) array of node coordinates.
        elements (array-like): (M,8) array of brick connectivity (node indices).
    
    Returns:
        new_coords (np.ndarray): Updated node coordinates including extra nodes.
        new_elements (np.ndarray): (M,27) connectivity for quadratic bricks.
    """
  coords = np.array(coords)
  elements = np.array(elements)

  new_coords = list(coords)  # original nodes
  edge_dict = {}  # global dictionary for edge midpoints
  face_dict = {}  # global dictionary for face nodes
  new_elements = []

  for brick in elements:
    local = brick  # original 8 corner indices

    # Helper functions
    def get_edge(i, j):
      key = tuple(sorted((local[i], local[j])))
      if key not in edge_dict:
        mid = 0.5 * (coords[local[i]] + coords[local[j]])
        edge_dict[key] = len(new_coords)
        new_coords.append(mid)
      return edge_dict[key]

    def get_face(face_nodes):
      key = tuple(sorted(face_nodes))
      if key not in face_dict:
        face_coord = np.mean([coords[idx] for idx in face_nodes], axis=0)
        face_dict[key] = len(new_coords)
        new_coords.append(face_coord)
      return face_dict[key]

    # Define corner nodes (order-1 nodes) according to the assumed ordering.
    c0 = local[0]  # (-1,-1,-1)
    c1 = local[1]  # ( 1,-1,-1)
    c2 = local[2]  # ( 1, 1,-1)
    c3 = local[3]  # (-1, 1,-1)
    c4 = local[4]  # (-1,-1, 1)
    c5 = local[5]  # ( 1,-1, 1)
    c6 = local[6]  # ( 1, 1, 1)
    c7 = local[7]  # (-1, 1, 1)

    # Compute edge nodes in Gmsh ordering:
    e0  = get_edge(0, 1)  # edge from c0 to c1
    e1  = get_edge(1, 2)  # edge from c1 to c2
    e2  = get_edge(2, 3)  # edge from c2 to c3
    e3  = get_edge(3, 0)  # edge from c3 to c0
    e4  = get_edge(4, 5)  # edge from c4 to c5
    e5  = get_edge(5, 6)  # edge from c5 to c6
    e6  = get_edge(6, 7)  # edge from c6 to c7
    e7  = get_edge(7, 4)  # edge from c7 to c4
    e8  = get_edge(0, 4)  # edge from c0 to c4
    e9  = get_edge(1, 5)  # edge from c1 to c5
    e10 = get_edge(2, 6)  # edge from c2 to c6
    e11 = get_edge(3, 7)  # edge from c3 to c7

    # Compute face nodes in Gmsh ordering:
    f_bottom = get_face((local[0], local[1], local[2], local[3]))  # bottom face (c0,c1,c2,c3)
    f_top    = get_face((local[4], local[5], local[6], local[7]))  # top face (c4,c5,c6,c7)
    f_front  = get_face((local[0], local[1], local[5], local[4]))  # front face (c0,c1,c5,c4)
    f_right  = get_face((local[1], local[2], local[6], local[5]))  # right face (c1,c2,c6,c5)
    f_back   = get_face((local[2], local[3], local[7], local[6]))  # back face (c2,c3,c7,c6)
    f_left   = get_face((local[3], local[0], local[4], local[7]))  # left face (c3,c0,c4,c7)

    # Compute interior node (average of the 8 corners)
    interior_coord = np.mean([coords[idx] for idx in local], axis=0)
    interior_idx = len(new_coords)
    new_coords.append(interior_coord)

    # Corners, then edges, then faces, then interior.
    new_connectivity = [c0, c1, c2, c3, c4, c5, c6, c7,
                        e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11,
                        f_left, f_right, f_front, f_back, f_bottom, f_top,
                        interior_idx]
    new_elements.append(new_connectivity)

  return np.array(new_coords), np.array(new_elements)

def elevate_mesh_order(coords, elements):
  """
    Upgrades a mesh from order 1 to order 2.
    
    Supported cases:
      - 2D triangles (3 nodes -> 6 nodes)
      - 3D bricks   (8 nodes -> 27 nodes)
    
    Args:
        coords (array-like): Array of node coordinates.
        elements (array-like): Element connectivity array.
    
    Returns:
        new_coords, new_elements: The upgraded mesh.
    
    Raises:
        NotImplementedError: If the element type is not supported.
    """
  coords = np.array(coords)
  elements = np.array(elements)
  dim = coords.shape[1]
  n_nodes_per_elem = elements.shape[1]

  if dim == 2 and n_nodes_per_elem == 3:
    return _elevate_order_triangle(coords, elements)
  elif dim == 3 and n_nodes_per_elem == 8:
    return _elevate_order_brick(coords, elements)
  else:
    raise NotImplementedError("Mesh elevation to order 2 not implemented for this element type.")
