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
from jax import config
import jax.numpy as jnp
import flax
import pygmsh
import meshio

from autopdex import seeder, geometry, solver, utility, models, spaces

config.update("jax_enable_x64", True)
config.update("jax_compilation_cache_dir", './cache')


### Cook's membrane, see e.g. https://link.springer.com/article/10.1007/s00466-017-1405-4
# Refinement with isoparametric quadrilaterls of order p

# p refinement
post_values = []
for order in range(1,10):

  ### Definition of geometry and boundary conditions
  pts = [[0.,0.], [48.,44.], [48.,60.], [0.,44.]]
  with pygmsh.occ.Geometry() as geom:
      region = geom.add_polygon(pts, mesh_size=10.0)
      geom.set_recombined_surfaces([region.surface])
      mesh = geom.generate_mesh(order=order)
      print("Mesh generation finished.")

  # Import region mesh
  n_dim = 2
  x_nodes = jnp.asarray(mesh.points[:,:n_dim])
  n_nodes = x_nodes.shape[0]
  elements = jnp.asarray([ v for k, v in mesh.cells_dict.items() if 'quad' in k])[0]
  surface_elements = jnp.asarray([ v for k, v in mesh.cells_dict.items() if 'line' in k])[0]
  print("Number of elements: ", elements.shape[0])
  print("Number of unknowns: ", n_nodes * n_dim)

  # Selection of nodes for boundary condition
  dirichlet_nodes = geometry.in_planes(x_nodes, pts[0], [1.,0.])
  dirichlet_dofs = utility.dof_select(dirichlet_nodes, jnp.asarray([True, True]))
  dirichlet_conditions = jnp.zeros_like(dirichlet_dofs, dtype=jnp.float64)

  # Import surface mesh for inhomogeneous Neumann conditions
  neumann_selection = geometry.select_elements_in_plane(x_nodes, surface_elements, pts[1], [1.,0.])
  neumann_elements= surface_elements[neumann_selection]

  # Definition of pde
  lam = 100.
  mu = 40.
  Em = mu * (3 * lam + 2 * mu) / (lam + mu)
  nu = lam / (2 * (lam + mu))
  youngs_mod_fun = lambda x: Em
  poisson_ratio_fun = lambda x: nu
  weak_form_fun_1 = models.linear_elasticity_weak(youngs_mod_fun, poisson_ratio_fun, 'plain strain') # small deformations
  # weak_form_fun_1 = models.hyperelastic_steady_state_weak(models.neo_hooke, youngs_mod_fun, poisson_ratio_fun, 'plain strain') # large deformations
  user_elem_1 = models.isoparametric_domain_element_galerkin(weak_form_fun_1, 
                                                          spaces.fem_iso_line_quad_brick,
                                                          *seeder.gauss_legendre_nd(dimension = 2, order = 2 * order))

  # Tractions
  q_0 = 4.0e+0
  # traction_fun = lambda x: jnp.asarray([0., q_0])
  traction_fun = lambda x, settings: jnp.asarray([0., settings['load multiplier']])
  weak_form_fun_2 = models.neumann_weak(traction_fun)
  user_elem_2 = models.isoparametric_surface_element_galerkin(weak_form_fun_2, 
                                                          spaces.fem_iso_line_quad_brick,
                                                          *seeder.gauss_legendre_nd(dimension = 1, order = 2 * order),
                                                          tangent_contributions=False)

  ### Settings
  n_fields = 2
  dofs0 = jnp.zeros((n_nodes, n_fields))
  static_settings = flax.core.FrozenDict({
    'number of fields': (n_fields, n_fields),
    'assembling mode': ('user element', 'user element'),
    'solution structure': ('nodal imposition', 'nodal imposition'),
    'model': (user_elem_1, user_elem_2),
    'solver type': 'newton',
    'solver backend': 'pardiso',
    'solver': 'lu',
    'verbose': 0,
  })

  settings = {
    'dirichlet dofs': dirichlet_dofs,
    'connectivity': (elements, neumann_elements),
    'load multiplier': q_0,
    'node coordinates': x_nodes,
    'dirichlet conditions': dirichlet_conditions,
  }

  ### Manual load stepping
  start = time.time()
  dofs = dofs0
  multiplier = 0.0
  n_step = 2
  for i in range(n_step):
    # Update boundary conditions
    print('Start load step ', i+1)
    multiplier = multiplier + 1 / n_step
    settings['load multiplier'] = multiplier * q_0

    # Call newton solver
    dofs, _ = solver.solver(dofs, settings, static_settings)


  # Paraview postprocessing
  points = mesh.points
  cells = jnp.asarray([ v for k, v in mesh.cells_dict.items() if 'quad' in k])[0]
  mesh = meshio.Mesh(
      points,
      {'quad': cells[:,:4]},
      point_data={
          "u": dofs,
      },
  )
  mesh.write("cooks_membrane.vtk")

  # Displacement at upper right edge
  idx = geometry.select_point(x_nodes, pts[2])
  v_ul = dofs[idx, 1].item()
  post_values.append([n_nodes * n_dim, v_ul]) # number of dofs over edge displacement in y-direction

print("Vertical displacements at upper right edge for each p-order:\n", post_values)
