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

def test_example_lid_driven_cavity():

    from jax import config
    import jax.numpy as jnp
    import flax

    from autopdex import seeder, geometry, solution_structures, utility, models, assembler

    config.update("jax_enable_x64", True)

    ### Definition of geometry and boundary conditions
    # Positive smooth distance functions that select the specific boundary segments
    pts = [[0.,0.], [0.1,0.], [0.1,0.1], [0.,0.1]]
    psdf_bottom = lambda x: geometry.psdf_trimmed_line(x, pts[0], pts[1])
    psdf_right = lambda x: geometry.psdf_trimmed_line(x, pts[1], pts[2])
    psdf_top = lambda x: geometry.psdf_trimmed_line(x, pts[2], pts[3])
    psdf_left = lambda x: geometry.psdf_trimmed_line(x, pts[3], pts[0])
    psdf_rect = lambda x: geometry.psdf_parallelogram(x, pts[0], pts[3], pts[1])
    psdfs = [psdf_bottom, psdf_right, psdf_top, psdf_left]

    # Positive smooth distance funcion of full domain
    psdf = lambda x: jnp.asarray([psdf_rect(x), psdf_rect(x)])

    # Field selectors for boundary conditions
    selection_dirichlet_1 = lambda x: jnp.asarray([0.,1.,0.])
    selection_dirichlet_2 = lambda x: jnp.asarray([0.,0.,1.])

    # Assignment of boundary conditions (same ordering as in psdfs)
    homogeneous = lambda x: 0.
    one = lambda x: 1.
    bc_1 = [homogeneous, homogeneous, one, homogeneous]

    ### Transfinite interpolation of boundary segments
    b_coeff = lambda x: jnp.asarray([selection_dirichlet_1(x),
                                    selection_dirichlet_2(x)])
    b_cond = lambda x: jnp.asarray([geometry.transfinite_interpolation(x, psdfs, bc_1),
                                    homogeneous(x)])


    ### Initialization and generation of nodes and integration points
    # Regular node placement
    nod_spacing = 0.02
    (x_nodes, n_nodes, region_size) = seeder.regular_in_psdf(psdf_rect, pts[0], pts[2], nod_spacing)

    # Gauss integration on background grid
    int_spacing = nod_spacing
    (x_int, w_int, n_int) = seeder.gauss_points_in_psdf(psdf_rect, pts[0], pts[2], int_spacing, order=2)

    # Support radius for moving least square ansatz
    support_radius = 3.01 * nod_spacing

    # Definition of pde
    pde = models.navier_stokes_incompressible_steady(dynamic_viscosity=0.01)

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

    # Note the inaccuracy in the pressure field with least square variational scheme
    # 'variational scheme': ('least square pde loss',),
    'variational scheme': ('strong form galerkin',),

    'model': (pde,),
    'solution structure': ('second order set',),
    'boundary coefficients': (b_coeff,),
    'boundary conditions': (b_cond,),
    'psdf': (psdf,),
    'solver type': 'newton',
    'solver backend': 'scipy',
    'solver': 'lapack',
    'verbose': -1,
    })
    settings = {
    'connectivity': (neighbor_list,),
    'beta': (3,),
    'node coordinates': x_nodes,
    'integration coordinates': (x_int,),
    'integration weights': (w_int,),
    'support radius': (support_radius,),
    }


    ### Optional: Precompute shape functions and pass them to e.g. static_settings...
    settings = solution_structures.precompile(dofs0, settings, static_settings)

    tangent = assembler.assemble_tangent(dofs0, settings, static_settings)
    assert jnp.isclose(tangent.data.flatten().sum(), 0.008454379736301122)
