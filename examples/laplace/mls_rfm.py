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
Dirichlet problem of the Laplace equation in 2d: [0,1]x[0,1] with cutout circle (midpoint=[0.5,0.5], radius=0.25)
Demonstration of quasi-random sampling in regions defined by signed distance functions
"""

if __name__ == "__main__":

  import jax
  from jax import lax, config
  import jax.numpy as jnp
  import flax

  from autopdex import seeder, geometry, solver, solution_structures, plotter, utility, models

  config.update("jax_enable_x64", True)
  config.update("jax_compilation_cache_dir", './cache')


  # Analytic solution for calculation of L2 error norm
  def analytic_solution(x):
    return jnp.stack([lax.sinh(jnp.pi * x[0] / 2) * lax.sin(jnp.pi * x[1] / 2)])

  # Positive smooth distance functions that select the specific boundary segments
  psdf_bottom = lambda x: geometry.psdf_trimmed_line(x, [0.,0.], [1.,0.])
  psdf_right = lambda x: geometry.psdf_trimmed_line(x, [1.,0.], [1.,1.])
  psdf_top = lambda x: geometry.psdf_trimmed_line(x, [1.,1.], [0.,1.])
  psdf_left = lambda x: geometry.psdf_trimmed_line(x, [0.,1.], [0.,0.])
  psdf_cutout = lambda x: geometry.psdf_nd_sphere_cutout(x, [0.5,0.5], 0.25)
  psdf_rect = lambda x: geometry.psdf_parallelogram(x, [0.,0.], [0.,1.], [1.,0.])
  psdfs = [psdf_bottom, psdf_right, psdf_top, psdf_left, psdf_cutout]


  # Positive smooth distance funcion of full domain (used for seeding nodes and integration points)
  psdf_domain = lambda x: geometry.r_equivalence(psdf_rect(x), psdf_cutout(x))

  # Boundary condition functions
  def homogeneous(x): # Bottom, right and top boundaries
    return 0
  def bc_0(x): # Left boundary
    return jax.lax.sin(jnp.pi * x[0])


  # Assignment of boundary conditions
  dirichlet_conditions = [bc_0, homogeneous, homogeneous, homogeneous, lambda x: 1.0]


  ### Transfinite interpolation of boundary segments
  b_cond = lambda x: geometry.transfinite_interpolation(x, psdfs, dirichlet_conditions)
  psdf = psdf_domain

  ### Initialization and generation of nodes and integration points
  # Quasi-random node placement
  n_nodes = 10000
  # (x_nodes, n_nodes, region_size) = seeder.quasi_random_in_psdf(psdf_domain, [0.,0.], [1.,1.], n_nodes, 'hammersley')
  (x_nodes, n_nodes, region_size) = seeder.quasi_random_in_psdf(psdf_domain, [0.,0.], [1.,1.], n_nodes, 'halton')

  # Integration points
  # Collocation, Note: this can in some cases cause bad conditioning, unphysical low-energy modes and divergence. 
  # If so, use more integration points, e.g. Gauss integration on a background grid (seeder.gauss_points_in_psdf)
  x_int = x_nodes
  n_int = n_nodes

  # Support radius for moving least square ansatz
  support_radius = 3.01 * lax.sqrt(region_size/n_nodes)

  # Integration point weights
  w_int = (region_size / n_int) * jnp.ones(n_int)

  # Definition of pde
  pde = models.poisson(source_fun=None)


  ### Neighborhood search
  (num_neighbors, max_neighbors, min_neighbors, neighbor_list) = utility.search_neighborhood(x_nodes, x_int, support_radius)


  ### Settings
  n_fields = 1
  dofs0 = jnp.zeros((n_nodes, n_fields))
  static_settings = flax.core.FrozenDict({
    'solution space': ('mls',),
    'shape function mode': 'compiled',
    'number of fields': (n_fields,),
    'assembling mode': ('sparse',),
    'maximal number of neighbors': (max_neighbors,),
    'order of basis functions': (2,),
    'weight function type': ('gaussian',),
    'variational scheme': ('least square pde loss',),
    # 'variational scheme': ('strong form galerkin',),
    'model': (pde,),
    'solution structure': ('dirichlet',),
    'boundary conditions': (b_cond,),
    'psdf': (psdf,),
    'solver type': 'linear',
    'solver backend': 'pardiso',
    'solver': 'lu',
    # 'solver backend': 'jax',
    # 'solver': 'bicgstab',
    # 'type of preconditioner': 'jacobi',
    # 'hvp type': 'linearize',
    'verbose': 1,
    'connectivity': (utility.jnp_to_tuple(neighbor_list),),
  })
  settings = {
    'beta': (3,),
    'node coordinates': x_nodes,
    'integration coordinates': (x_int,),
    'integration weights': (w_int,),
    'support radius': (support_radius,),
  }

  # Precompute solution structure
  settings = solution_structures.precompile(dofs0, settings, static_settings)

  ### Call solver
  dofs, _ = solver.solver(dofs0, settings, static_settings, tol=1e-8, newton_tol=1e-8)

  ### Preperation of postprocessing data
  local_dofs = dofs[neighbor_list]
  post_fun_vj = jax.jit(jax.vmap(solution_structures.solution_structure, (0,0,0,None,None,None)), static_argnames=['static_settings', 'set'])
  data = post_fun_vj(x_int, jnp.arange(n_int), local_dofs, settings, static_settings, 0)

  # Plotting with pyvista
  plotter.pv_plot(x_int, data[:,0], export_vtk=False, show=True)

  # Export to Paraview
  import meshio
  import numpy as np
  points = x_int
  cells = [("vertex", np.array([[i] for i in range(len(points))]))]
  mesh = meshio.Mesh(
      points,
      cells,
      point_data={
          "temperature": data[:,0],
      },
  )
  mesh.write("laplace.vtk")
