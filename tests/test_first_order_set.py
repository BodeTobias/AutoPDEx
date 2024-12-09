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

def test_example_coupled_neumann():

    import jax
    from jax import config
    import jax.numpy as jnp
    import flax

    from autopdex import seeder, geometry, solver, solution_structures, utility, models

    config.update("jax_enable_x64", True)


    # Positive smooth distance functions that select the specific boundary segments
    psdf_bottom = lambda x: geometry.psdf_trimmed_line(x, jnp.stack([0.,0.]), jnp.stack([1.,0.]))
    psdf_right = lambda x: geometry.psdf_trimmed_line(x, jnp.stack([1.,0.]), jnp.stack([1.,1.]))
    psdf_top = lambda x: geometry.psdf_trimmed_line(x, jnp.stack([1.,1.]), jnp.stack([0.,1.]))
    psdf_left = lambda x: geometry.psdf_trimmed_line(x, jnp.stack([0.,1.]), jnp.stack([0.,0.]))
    psdf_cutout = lambda x: geometry.psdf_nd_sphere_cutout(x, jnp.stack([0.5,0.5]), 0.25)
    psdf_rect = lambda x: geometry.psdf_parallelogram(x, jnp.stack([0.,0.]), jnp.stack([0.,1.]), jnp.stack([1.,0.]))
    psdfs = [psdf_bottom, psdf_right, psdf_top, psdf_left, psdf_cutout]

    # Positive smooth distance funcion of full domain (used for seeding nodes and integration points)
    psdf_domain = lambda x: geometry.r_equivalence(psdf_rect(x), psdf_cutout(x))

    # Boundary condition functions
    def homogeneous(x): # Bottom, right and top boundaries
        return 0
    def bc_0(x): # Left boundary
        return jax.lax.sin(jnp.pi * x[0])
    def neumann_at_circ(x): # Normal derivative at circular cutout for a specific analytic solution, may be used for convergence study
        return (2. * jnp.pi / jnp.sinh(jnp.pi)) * ((-1.+2.*x[1])*jnp.cosh(jnp.pi-jnp.pi*x[1]) + (1.-2.*x[0])*jnp.cos(jnp.pi*x[0])*jnp.sinh(jnp.pi-jnp.pi*x[1]))

    # Field selectors for boundary conditions
    def dirichlet(x):
        return jnp.asarray([1., 0., 0.])
    def coupled_neumann(x):
        centroid = jnp.asarray([0.5,0.5])
        dxc = x - centroid
        normal = - dxc / jnp.sqrt(jnp.dot(dxc, dxc))
        return jnp.concatenate([jnp.asarray([0.]), normal])

    # Assignment of boundary conditions
    dirichlet_conditions = [bc_0, homogeneous, homogeneous, homogeneous, neumann_at_circ]
    a_i = [dirichlet, dirichlet, dirichlet, dirichlet, coupled_neumann]

    ### Transfinite interpolation of boundary segments
    b_coeff = lambda x: jnp.asarray([geometry.transfinite_interpolation(x, psdfs, a_i)])
    b_cond = lambda x: jnp.asarray([geometry.transfinite_interpolation(x, psdfs, dirichlet_conditions)])
    psdf = lambda x: jnp.asarray([psdf_domain(x)])


    ### Initialization and generation of nodes and integration points
    # Regular node placement
    nod_spacing = 0.1
    (x_nodes, n_nodes, region_size) = seeder.regular_in_psdf(psdf_domain, [0.,0.], [1.,1.], nod_spacing)

    # Integration points
    # # Regular spacing
    # int_spacing = nod_spacing
    # (x_int, w_int, n_int) = seeder.gauss_points_in_psdf(psdf_domain, [0.,0.], [1.,1.], int_spacing, order=2)
    # Collocation
    x_int = x_nodes
    n_int = n_nodes
    w_int = (region_size / n_int) * jnp.ones(n_int)
    print("Estimated region size: ", w_int.sum())
    assert jnp.isclose(w_int.sum(), 0.49586776859504134), "Region size wrong"

    # Support radius for moving least square ansatz
    support_radius = 2.01 * nod_spacing

    # Definition of pde
    pde = models.poisson_fos(support_radius)

    # Neighborhood search
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
    'order of basis functions': (1,),
    'weight function type': ('gaussian',),
    'variational scheme': ('least square pde loss',),
    'model': (pde,),
    'solution structure': ('first order set',),
    'boundary coefficients': (b_coeff,),
    'boundary conditions': (b_cond,),
    'psdf': (psdf,),
    'solver type': 'linear',
    'solver backend': 'scipy',
    'solver': 'lapack',
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

    ### Optional: Precompute shape functions and pass them to e.g. static_settings...
    settings = solution_structures.precompile(dofs0, settings, static_settings)


    ### Call solver
    dofs, _ = solver.solver(dofs0, settings, static_settings, tol=1e-8)

    check = dofs.flatten().sum()
    # print(check)
    assert jnp.isclose(check, -22.904093385453102), "Solution wrong"
