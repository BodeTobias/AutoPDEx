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

def test_nonhomogeneous_dirichlet():

    import jax
    from jax import config
    import jax.numpy as jnp
    import numpy as np
    import flax
    import pygmsh

    from autopdex import seeder, geometry, solver, utility, models

    config.update("jax_enable_x64", True)


    ### Cook's membrane problem with non-homogeneous Dirichlet conditions nodally imposed

    ### Definition of geometry and boundary conditions
    pts = [[0.,0.], [48.,44.], [48.,60.], [0.,44.]]
    with pygmsh.geo.Geometry() as geom:
        geom.add_polygon(pts,mesh_size=50.0)
        mesh = geom.generate_mesh(order=1)

    # Import region mesh
    n_dim = 2
    x_nodes = jnp.asarray(mesh.points[:,:n_dim])
    n_nodes = x_nodes.shape[0]
    elements = jnp.asarray(mesh.cells_dict['triangle'])
    surface_elements = jnp.asarray(mesh.cells_dict['line'])

    # Selection of nodes for boundary condition
    dirichlet_left = geometry.in_planes(x_nodes, pts[0], [1.,0.])
    dirichlet_dofs_left = utility.dof_select(dirichlet_left, jnp.asarray([True, True]))
    dirichlet_conditions = jnp.zeros_like(dirichlet_dofs_left, dtype=jnp.float64)

    # Assignment of boundary conditions at right edge
    v_bar = lambda x: 0.1
    dirichlet_right = geometry.in_planes(x_nodes, pts[1], [1.,0.])
    dirichlet_dofs_right = utility.dof_select(dirichlet_right, jnp.asarray([False, True])) # Only impose y-direction
    x_right = x_nodes[dirichlet_right]
    v_right = jax.vmap(v_bar)(x_right)
    dirichlet_conditions = utility.mask_set(dirichlet_conditions, dirichlet_dofs_right, v_right)

    # All dirichlet dofs
    dirichlet_dofs = dirichlet_dofs_left + dirichlet_dofs_right

    # Generate domain integration points in mesh
    (x_int, w_int, n_int, domain_connectivity) = seeder.int_pts_in_tri_mesh(x_nodes, elements, order=1)
    max_neighbors_1 = domain_connectivity.shape[-1]

    # Definition of pde
    lam = 100.
    mu = 40.
    Em = mu * (3 * lam + 2 * mu) / (lam + mu)
    nu = lam / (2 * (lam + mu))
    youngs_mod_fun = lambda x: Em
    poisson_ratio_fun = lambda x: nu
    # pde_1 = models.linear_elasticity_weak(youngs_mod_fun, poisson_ratio_fun, 'plain strain')
    pde_1 = models.hyperelastic_steady_state_weak(models.neo_hooke, youngs_mod_fun, poisson_ratio_fun, 'plain strain')


    ### Setting
    n_fields = 2
    dofs0 = jnp.zeros((n_nodes, n_fields))
    static_settings = flax.core.FrozenDict({
    'solution space': ('fem simplex',),
    'shape function mode': 'direct',
    'number of fields': (n_fields,),
    'assembling mode': ('sparse',),
    'maximal number of neighbors': (max_neighbors_1,),
    'variational scheme': ('weak form galerkin',),
    'model': (pde_1,),
    'solution structure': ('nodal imposition',),
    'solver type': 'newton',
    'solver backend': 'scipy',
    'solver': 'lapack',
    'verbose': 0,
    })

    settings = {
    'dirichlet dofs': dirichlet_dofs,
    'connectivity': (domain_connectivity,),
    'node coordinates': x_nodes,
    'dirichlet conditions': dirichlet_conditions,
    'integration coordinates': (x_int,),
    'integration weights': (w_int,),
    }

    # Call solver
    dofs, _ = solver.solver(dofs0, settings, static_settings, tol=1e-8)

    # print(repr(dofs))
    assert jnp.allclose(dofs, jnp.asarray([[ 0.        ,  0.        ],
       [-0.01806664,  0.1       ],
       [-0.05176078,  0.1       ],
       [ 0.        ,  0.        ],
       [ 0.00834901,  0.02592409],
       [-0.02664544,  0.03728282]]))

# test_nonhomogeneous_dirichlet()
