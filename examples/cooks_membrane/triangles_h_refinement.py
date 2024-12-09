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

import time
import sys

import jax
from jax import lax, config
import jax.numpy as jnp
import flax
import pygmsh
import meshio

from autopdex import seeder, geometry, solver, solution_structures, plotter, utility, models, spaces, assembler

config.update("jax_enable_x64", True)
config.update("jax_compilation_cache_dir", './cache')


### Cook's membrane problem with simplex elements and two domains, linear elasticity
# First domain: domain integration
# Second domain: surface integration (tractions)


# h-refinement
post_values = []
for i in range(1,3):
  d_h = 10 / (2**i)

  ### Definition of geometry and boundary conditions
  pts = [[0.,0.], [48.,44.], [48.,60.], [0.,44.]]
  with pygmsh.geo.Geometry() as geom:
      geom.add_polygon(pts,mesh_size=d_h)
      mesh = geom.generate_mesh(order=2)

  # Mesh debugging
  mesh.write("mesh.vtk")

  # Import region mesh
  n_dim = 2
  x_nodes = jnp.asarray(mesh.points[:,:n_dim])
  n_nodes = x_nodes.shape[0]
  elements = jnp.asarray(mesh.cells_dict['triangle6'])
  surface_elements = jnp.asarray(mesh.cells_dict['line3'])

  # Selection of nodes for boundary condition
  dirichlet_nodes = geometry.in_planes(x_nodes, pts[0], [1.,0.])
  dirichlet_dofs = utility.dof_select(dirichlet_nodes, jnp.asarray([True, True]))
  dirichlet_conditions = jnp.zeros_like(dirichlet_dofs, dtype=jnp.float64)

  # Import surface mesh for inhomogeneous Neumann conditions
  neumann_selection_1 = geometry.select_elements_in_plane(x_nodes, surface_elements, pts[1], [1.,0.])
  neumann_elements_1 = surface_elements[neumann_selection_1]

  # Generate domain integration points in mesh
  (x_int, w_int, n_int, domain_connectivity) = seeder.int_pts_in_tri_mesh(x_nodes, elements, order=2)
  max_neighbors_1 = domain_connectivity.shape[-1]


  # Generate surface integration points on surface mesh and select domain elements that belong to the surface elements
  (x_surf_int, w_surf_int, n_surf_int, nods_of_surf_elem) = seeder.int_pts_in_line_mesh(x_nodes, neumann_elements_1, order=2)
  surf_connectivity = geometry.subelems_in_elems(nods_of_surf_elem, elements)
  max_neighbors_2 = surf_connectivity.shape[-1]

  # Definition of pde
  lam = 100.
  mu = 40.
  Em = mu * (3 * lam + 2 * mu) / (lam + mu)
  nu = lam / (2 * (lam + mu))
  youngs_mod_fun = lambda x: Em
  poisson_ratio_fun = lambda x: nu
  pde_1 = models.linear_elasticity_weak(youngs_mod_fun, poisson_ratio_fun, 'plain strain')

  # Tractions
  q_0 = 4.0
  traction_fun = lambda x: jnp.asarray([0., q_0])
  pde_2 = models.neumann_weak(traction_fun)


  ### Setting
  n_fields = 2
  dofs0 = jnp.zeros((n_nodes, n_fields))
  static_settings = flax.core.FrozenDict({
    'solution space': ('fem simplex', 'fem simplex'),
    'shape function mode': 'compiled',
    'number of fields': (n_fields, n_fields),
    'assembling mode': ('sparse', 'sparse'),
    'maximal number of neighbors': (max_neighbors_1, max_neighbors_2),
    'variational scheme': ('weak form galerkin', 'weak form galerkin'),
    'model': (pde_1, pde_2),
    'solution structure': ('nodal imposition', 'nodal imposition'),
    'solver type': 'newton',

    # 'solver backend': 'jax',
    # 'solver': 'cg',
    # 'hvp type': 'fwdrev',

    'solver backend': 'pardiso',
    'solver': 'lu',

    # 'solver backend': 'petsc',
    # 'solver': 'cg',
    # 'type of preconditioner': 'ilu',

    'verbose': 2,
  })

  settings = {
    'dirichlet dofs': dirichlet_dofs,
    'connectivity': (domain_connectivity, surf_connectivity),
    'node coordinates': x_nodes,
    'dirichlet conditions': dirichlet_conditions,
    'integration coordinates': (x_int, x_surf_int),
    'integration weights': (w_int, w_surf_int),
  }

  # Precompute shape functions
  settings = solution_structures.precompile(dofs0, settings, static_settings)


  # Call solver
  start = time.time()
  dofs, _ = solver.solver(dofs0, settings, static_settings)
  print("Analysis time: ", time.time() - start)
  print("Number of unknowns: ", dofs.flatten().shape)


  # Paraview postprocessing
  points = mesh.points
  cells = mesh.cells_dict["triangle6"]
  mesh = meshio.Mesh(
      points,
      {'triangle6': cells},
      point_data={
          "u": dofs,
      },
  )
  mesh.write("cooks_membrane.vtk")

  # Displacement at upper right edge
  idx = geometry.select_point(x_nodes, pts[2])
  v_ul = dofs[idx, 1].item()
  post_values.append([n_nodes * n_dim, v_ul]) # number of dofs over edge displacement in y-direction

print("Vertical displacements at upper right edge for each h-order:\n", post_values)
