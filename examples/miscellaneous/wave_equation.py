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
In this example, the wave equation is solved on the square domain 
:math:`(x, t)` with :math:`x \in (0, 1)` and :math:`t \in (0, 1)`.
The problem is formulated as a first-order system, and the solution 
is searched using a least-squares variational method. For the solution space, 
meshfree moving least squares (MLS) shape functions of second order are used, 
with integration performed on a background grid (as in the Element-Free 
Galerkin (EFG) method) using Gaussian quadrature. The application of boundary
conditions is achieved through the solution structure for first-order
systems using Rvachev distance functions to the respective boundary segments.
"""

if __name__ == "__main__":
  import sys

  import jax
  from jax import config
  import jax.numpy as jnp
  import flax

  from autopdex import seeder, geometry, solver, solution_structures, plotter, utility, models

  config.update("jax_enable_x64", True)
  config.update("jax_compilation_cache_dir", './cache')



  ### Definition of geometry and boundary conditions
  # Positive smooth distance functions that select the specific boundary segment
  # Time dimension is always the last dimension (here the second)
  psdf_bottom = lambda x: geometry.psdf_trimmed_line(x, jnp.stack([0.,0.]), jnp.stack([1.,0.]))
  psdf_right = lambda x: geometry.psdf_trimmed_line(x, jnp.stack([1.,0.]), jnp.stack([1.,1.]))
  psdf_left = lambda x: geometry.psdf_trimmed_line(x, jnp.stack([0.,1.]), jnp.stack([0.,0.]))
  psdfs_1 = [psdf_bottom, psdf_right, psdf_left]
  psdfs_2 = [psdf_bottom, psdf_right, psdf_left]


  # Initial condition
  def bc_bottom1(x): # Initial value
    return jax.lax.exp(-100. * (x[0]-0.3)**2)
  def bc_bottom2(x): # Initial speed
    return 0.

  # Dirichlet boundary conditions on left an right side
  def bc_left(x):
    return 0.
  def bc_right(x):
    return 0.
  bcs_1 = [bc_bottom1, bc_right, bc_left]
  bcs_2 = [bc_bottom2]

  # Field selector for boundary conditions
  def dirichlet(x):
    return jnp.asarray([1., 0., 0.])
  def neumann_x(x):
    return jnp.asarray([0., 1., 0.])
  def neumann_t(x):
    return jnp.asarray([0., 0., 1.])
  ai_1 = [dirichlet, dirichlet, neumann_x]
  ai_2 = [neumann_t, neumann_t, neumann_t] # Ensure orthogonality of the coefficients along the boundary segments!

  # Transfinite interpolation of boundary segments
  b_coeff = lambda x: jnp.asarray([geometry.transfinite_interpolation(x, psdfs_1, ai_1),
                                    geometry.transfinite_interpolation(x, psdfs_2, ai_2)])
  b_cond = lambda x: jnp.asarray([geometry.transfinite_interpolation(x, psdfs_1, bcs_1),
                                  bcs_2[0](x)])
  psdf = lambda x: jnp.asarray([geometry.psdf_unification(x, psdfs_1),
                                psdf_bottom(x)]) # Only select the region where boundary conditions shall be enforced!

  # Positive smooth distance funcion of full domain (used for seeding nodes and integration points)
  psdf_rect = lambda x: geometry.psdf_parallelogram(x, jnp.stack([0.,0.]), jnp.stack([0.,1.]), jnp.stack([1.,0.]))
  psdf_domain = psdf_rect

  ### Initialization and generation of nodes and integration points
  nod_spacing = 0.02
  # nod_spacing = 0.01
  (x_nodes, n_nodes, region_size) = seeder.regular_in_psdf(psdf_domain, [0.,0.], [1.,1.], nod_spacing)

  # Gauss integration on background grid
  int_spacing = nod_spacing
  (x_int, w_int, n_int) = seeder.gauss_points_in_psdf(psdf_domain, [0.,0.], [1.,1.], int_spacing, order=2)

  # Support radius for moving least square ansatz
  support_radius = 3.01 * nod_spacing

  # ### Visualizer for boundary functions
  # fun = lambda x: psdf(x)[0]
  # plotter.pv_function_plot(x_int, fun)
  # sys.exit()


  # Definition of pde
  wave_number = lambda x: 1.0
  pde = models.d_alembert_fos(wave_number_fun=wave_number, spacing=nod_spacing)

  ### Neighborhood search
  (num_neighbors, max_neighbors, min_neighbors, neighbor_list) = utility.search_neighborhood(x_nodes, x_int, support_radius)


  ### Settings
  n_fields = 3
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
    'model': (pde,),
    'solution structure': ('first order set',),
    'boundary coefficients': (b_coeff,),
    'boundary conditions': (b_cond,),
    'psdf': (psdf,),
    'solver type': 'linear',
    'solver backend': 'pardiso',
    'solver': 'lu',
    'verbose': 1,
  })
  settings = {
    'connectivity': (neighbor_list,),
    'beta': (3,),
    'node coordinates': x_nodes,
    'integration coordinates': (x_int,),
    'integration weights': (w_int,),
    'support radius': (support_radius,),
  }


  # Precompute shape functions
  settings = solution_structures.precompile(dofs0, settings, static_settings)


  ### Call solver
  dofs, _ = solver.solver(dofs0, settings, static_settings, tol=1e-8)


  ### Preperation of postprocessing data
  local_dofs = dofs[neighbor_list]
  post_fun_vj = jax.jit(jax.vmap(solution_structures.solution_structure, (0,0,0,None,None,None)), static_argnames=['static_settings', 'set'])
  data = post_fun_vj(x_int, jnp.arange(n_int), local_dofs, settings, static_settings, 0)

  # Plotting with pyvista
  plotter.pv_plot(x_int, data[:,0], export_vtk=False, show=True)
  plotter.pv_plot(x_int, data[:,1], export_vtk=False, show=True)
  plotter.pv_plot(x_int, data[:,2], export_vtk=False, show=True)

  # Export to Paraview
  import meshio
  import numpy as np
  points = x_int
  cells = [("vertex", np.array([[i] for i in range(len(points))]))]
  mesh = meshio.Mesh(
      points,
      cells,
      point_data={
          "field": data[:,0],
          "gradient": data[:,1:],
      },
  )
  mesh.write("wave_equation.vtk")
