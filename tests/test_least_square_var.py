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

def test_example_circle_in_channel():
  from jax import config
  import jax.numpy as jnp
  import flax

  from autopdex import seeder, geometry, solution_structures, utility, models, assembler

  config.update("jax_enable_x64", True)

  # Domain selection
  channel = lambda x: geometry.sdf_parallelogram_2d(x, [-6.732,-1.], [6.732,-1.], [-6.732,1.])
  outer_circle = lambda x: geometry.sdf_nd_sphere(x, [0.,0.], 2.)
  inner_circle_cut = lambda x: -geometry.sdf_nd_sphere(x, [0.,0.], 1.)
  outer_and_channel = lambda x: geometry.r_disjunction(channel(x), outer_circle(x))
  sdf_domain = lambda x: geometry.r_conjunction(outer_and_channel(x), inner_circle_cut(x))

  # Smooth distance functions that select the specific boundary segments
  psdf_left = lambda x: geometry.psdf_trimmed_line(x, [-6.732,-1.], [-6.732,1.])
  psdf_right = lambda x: geometry.psdf_trimmed_line(x, [6.732,-1.], [6.732,1.])
  channel_2 = lambda x: geometry.sdf_nd_planes(x, [-6.732,-1.], [-6.732,1.])
  outer_and_channel_2 = lambda x: geometry.r_disjunction(channel_2(x), outer_circle(x))
  sdf_remaining = lambda x: geometry.r_conjunction(outer_and_channel_2(x), inner_circle_cut(x))
  psdfs = [psdf_left, psdf_right, sdf_remaining]

  # Smooth distance funcion of full domain
  sdf = lambda x: jnp.asarray([sdf_domain(x), sdf_domain(x)])

  # Boundary condition functions
  homogeneous = lambda x: 0.

  # Notice the different flow field for changing inlet velocity
  v_left = lambda x: 0.5 * (1 - x[1]**2)
  # v_left = lambda x: 0.01 * (1 - x[1]**2)

  # Field selectors for boundary conditions
  selection_dirichlet_1 = lambda x: jnp.asarray([0.,1.,0.])
  selection_dirichlet_2 = lambda x: jnp.asarray([0.,0.,1.])
  selection_pressure = lambda x: jnp.asarray([1.,0.,0.])

  # Assignment of boundary conditions (same ordering as in psdfs)
  bc_1 = [v_left, homogeneous, homogeneous] # first boundary condition at left, right and remaining boundaries
  bc_2 = [homogeneous, homogeneous, homogeneous] # second bc

  # Selectors that define for which field the boundary conditions shall be imposed
  ai_1 = [selection_dirichlet_1, selection_pressure, selection_dirichlet_1]
  ai_2 = [selection_dirichlet_2, selection_dirichlet_2, selection_dirichlet_2]

  ### Transfinite interpolation of boundary segments
  b_coeff = lambda x: jnp.asarray([geometry.transfinite_interpolation(x, psdfs, ai_1),
                                  geometry.transfinite_interpolation(x, psdfs, ai_2)])
  b_cond = lambda x: jnp.asarray([geometry.transfinite_interpolation(x, psdfs, bc_1),
                                  geometry.transfinite_interpolation(x, psdfs, bc_2)])


  ### Initialization and generation of nodes and integration points
  # Regular node placement
  nod_spacing = 1.
  # nod_spacing = 0.05
  (x_nodes, n_nodes, region_size) = seeder.regular_in_psdf(sdf_domain, [-6.732,-2.], [6.732,2.], nod_spacing, atol=-1e-8)

  # Gauss integration on background grid
  int_spacing = nod_spacing
  (x_int, w_int, n_int) = seeder.gauss_points_in_psdf(sdf_domain, [-6.732,-2.], [6.732,2.], int_spacing, order=2)

  # Support radius for moving least square ansatz
  support_radius = 3.01 * nod_spacing

  # Definition of pde
  pde = models.navier_stokes_incompressible_steady(dynamic_viscosity=0.01, incompressible_weighting=10.)

  ### Neighborhood search
  (num_neighbors, max_neighbors, min_neighbors, neighbor_list) = utility.search_neighborhood(x_nodes, x_int, support_radius)

  ### Setting
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

    # The least square variational approach minimizes a functional with respect to the 
    # degrees of freedom. As a consequence, also nonlinear minimizers can be utilized.
    'variational scheme': ('least square pde loss',),
    # 'solver type': 'minimize',
    # 'solver': 'lbfgs',
    'solver type': 'newton',
    'solver backend': 'scipy',
    'solver': 'lapack',

    'model': (pde,),
    'solution structure': ('second order set',),
    'boundary coefficients': (b_coeff,),
    'boundary conditions': (b_cond,),
    'psdf': (sdf,),
    'verbose': 1,
  })
  settings = {
    'connectivity': (neighbor_list,),
    'beta': (3.,),
    'node coordinates': x_nodes,
    'integration coordinates': (x_int,),
    'integration weights': (w_int,),
    'support radius': (support_radius,),
  }


  # Precompute shape functions
  settings = solution_structures.precompile(dofs0, settings, static_settings)

  tangent = assembler.assemble_tangent(dofs0, settings, static_settings)
  check = tangent.data.flatten().sum()
#   print(check)
  assert jnp.isclose(check, 1427.552335434657)
