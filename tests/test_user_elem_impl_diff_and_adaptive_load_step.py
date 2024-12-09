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

def test_implicit_diff():
    import time
    from functools import partial

    import jax
    from jax import config
    import jax.numpy as jnp
    import flax

    from autopdex import seeder, geometry, solver, utility, models, spaces

    config.update("jax_enable_x64", True)

    start = time.time()

    order = 2
    pts = [[0.,0.], [48.,44.], [48.,60.], [0.,44.]]
    n_dim = 2
    x_nodes = jnp.array([
        [ 0., 0.], [48., 44.], [48., 60.], [ 0., 44.], [12., 11.], [24., 22.], [36., 33.], [ 6., 5.5], [18., 16.5], 
        [30., 27.5], [42., 38.5], [48., 52.], [48., 48.], [48., 56.], [36., 56.], [24., 52.], [12., 48.], [42., 58.], 
        [30., 54.], [18., 50.], [ 6., 46.], [ 0., 33.], [ 0., 22.], [ 0., 11.], [ 0., 38.5], [ 0., 27.5], [ 0., 16.5], 
        [ 0., 5.5], [22.78099159, 39.33936105], [10.25815892, 28.78971905], [40.00000145, 47.99999935], [11.44621914, 36.55774821], 
        [30.81792875, 42.11576444], [ 7.62152819, 16.21744291], [23.3904958, 30.66968052], [17.11360537, 37.94855463], 
        [10.85218903, 32.67373363], [17.12907946, 25.39485952], [17.12134241, 31.67170708], [ 8.93984355, 22.50358098], 
        [ 9.81076409, 13.60872146], [13.46992178, 19.50179049], [38.00000072, 51.99999968], [44.00000072, 49.99999968], 
        [43.00000036, 53.99999984], [33.40896438, 37.55788222], [26.79946017, 40.72756274], [28.39973009, 34.11378137], 
        [38.00000072, 40.49999968], [43.00000036, 44.24999984], [11.72310957, 42.2788741], [23.3904958, 45.66968052], 
        [17.55680268, 43.97427731], [ 5.72310957, 34.7788741], [ 5.86155479, 40.38943705], [33.40896438, 49.05788222], 
        [35.70448255, 44.77894095], [ 5.12907946, 25.39485952], [ 5.42609452, 30.08686681], [28.39973009, 47.36378137], 
        [ 3.81076409, 13.60872146], [ 4.46992178, 19.50179049], [ 4.90538205, 9.55436073]
    ])
    n_nodes = x_nodes.shape[0]
    elements = jnp.array([
        [ 5, 28, 31, 29, 34, 35, 36, 37, 38], [ 4,  5, 29, 33,  8, 37, 39, 40, 41], [ 2, 14, 30, 11, 17, 42, 43, 13, 44],
        [ 5,  6, 32, 28,  9, 45, 46, 34, 47], [ 6,  1, 11, 30, 10, 12, 43, 48, 49], [15, 16, 31, 28, 19, 50, 35, 51, 52], 
        [16,  3, 21, 31, 20, 24, 53, 50, 54], [ 6, 30, 14, 32, 48, 42, 55, 45, 56], [22, 29, 31, 21, 57, 36, 53, 25, 58], 
        [14, 15, 28, 32, 18, 51, 46, 55, 59], [23, 33, 29, 22, 60, 39, 57, 26, 61], [33, 23,  0,  4, 60, 27,  7, 40, 62]
    ])
    surface_elements = jnp.array([
        [ 0,  4,  7], [ 4,  5,  8], [ 5,  6,  9], [ 6,  1, 10], [ 1, 11, 12], [11,  2, 13], [ 2, 14, 17], 
        [14, 15, 18], [15, 16, 19], [16,  3, 20], [ 3, 21, 24], [21, 22, 25], [22, 23, 26], [23,  0, 27]
    ])
    # import sys
    # sys.exit()

    # Selection of nodes for boundary condition
    dirichlet_nodes = geometry.in_planes(x_nodes, pts[0], [1.,0.])
    dirichlet_dofs = utility.dof_select(dirichlet_nodes, jnp.asarray([True, True]))
    dirichlet_conditions = jnp.zeros_like(dirichlet_dofs, dtype=jnp.float64)

    # Import surface mesh for inhomogeneous Neumann conditions
    neumann_selection = geometry.select_elements_in_plane(x_nodes, surface_elements, pts[1], [1.,0.])
    neumann_elements= surface_elements[neumann_selection]


    ### Built-in autopdex element
    weak_form_fun_1 = models.hyperelastic_steady_state_weak(
    models.neo_hooke, 
    lambda x, settings: settings['youngs modulus'],
    lambda x, settings: settings['poisson ratio'], 
    'plain strain')
    user_elem_1 = models.isoparametric_domain_element_galerkin(
    weak_form_fun_1, 
    spaces.fem_iso_line_quad_brick,
    *seeder.gauss_legendre_nd(dimension = 2, order = 2 * order))

    # Tractions
    q_0 = 2.0e+1
    traction_fun = lambda x, settings: jnp.asarray([0., settings['load multiplier']])
    weak_form_fun_2 = models.neumann_weak(traction_fun)
    user_elem_2 = models.isoparametric_surface_element_galerkin(
    weak_form_fun_2, 
    spaces.fem_iso_line_quad_brick,
    *seeder.gauss_legendre_nd(dimension = 1, order = 2 * order),
    tangent_contributions=False)

    # Adaptive load stepping
    def multiplier_settings(settings, multiplier):
        settings['load multiplier'] = multiplier * q_0
        return settings

    ### Settings
    n_fields = 2
    dofs_0 = jnp.zeros((n_nodes, n_fields))
    static_settings = flax.core.FrozenDict({
    'number of fields': (n_fields, n_fields),
    'assembling mode': ('user element', 'user element'),
    'solution structure': ('nodal imposition', 'nodal imposition'),
    'model': (user_elem_1, user_elem_2),
    'solver type': 'newton',
    'solver backend': 'scipy',
    'solver': 'lapack',
    'verbose': -1,
    })
    settings = {
    'dirichlet dofs': dirichlet_dofs,
    'connectivity': (elements, neumann_elements),
    'load multiplier': q_0,
    'node coordinates': x_nodes,
    'dirichlet conditions': dirichlet_conditions,
    'youngs modulus': 100.,
    'poisson ratio': 0.3,
    }

    youngs_mod_mean = 100.
    nu_mean = 0.3

    # Compare backward and forward mode implicit diff with precomputed values
    @partial(jax.jit, static_argnames=['static_settings'])
    def test_results(dofs_0, settings, static_settings):
        # Second order mixed sensitivities, forward mode
        def fun(dofs, settings, Em, nu):
            settings['youngs modulus'] = Em
            settings['poisson ratio'] = nu
            dofs = solver.adaptive_load_stepping(dofs, settings, static_settings, multiplier_settings, False, 'forward', newton_tol=1e-8)[0]
            return dofs.flatten() @ dofs.flatten()
        forward_and_sensitivity = lambda v: fun(dofs_0, settings, v[0], v[1])
        result = utility.jacfwd_upto_n_one_vector_arg(forward_and_sensitivity, jnp.asarray([youngs_mod_mean, nu_mean]), 2)
        dofs, du_dEm, du_dnu, d2u_dEm2, d2u_dEmdnu, d2u_dnudEm, d2u_dnu2 = result[0], result[1][0], result[1][1], result[2][0][0], result[2][0][1], result[2][1][0], result[2][1][1]

        # Reverse mode implicit diff:
        def fun_bwd(dofs, settings, Em, nu):
            settings['youngs modulus'] = Em
            settings['poisson ratio'] = nu
            dofs = solver.adaptive_load_stepping(dofs, settings, static_settings, multiplier_settings, False, 'backward', newton_tol=1e-8)[0]
            return dofs.flatten() @ dofs.flatten()
        def compute_sensitivities_backward(v):
            sol = fun_bwd(dofs_0, settings, v[0], v[1])
            du_dnu = jax.jacrev(fun_bwd, argnums=3)(dofs_0, settings, v[0], v[1])
            d2u_dnu2 = jax.jacrev(jax.jacrev(fun_bwd, argnums=3), argnums=3)(dofs_0, settings, v[0], v[1])
            return sol, du_dnu, d2u_dnu2
        sol_rev, dudnu_rev, d2udnu2_rev = compute_sensitivities_backward(jnp.asarray([youngs_mod_mean, nu_mean]))

        return dofs, du_dEm, du_dnu, d2u_dEm2, d2u_dEmdnu, d2u_dnudEm, d2u_dnu2, sol_rev, dudnu_rev, d2udnu2_rev

    dofs, du_dEm, du_dnu, d2u_dEm2, d2u_dEmdnu, d2u_dnudEm, d2u_dnu2, sol_rev, dudnu_rev, d2udnu2_rev = test_results(dofs_0, settings, static_settings)

        # Precomputed values:
    test1, test2, test3, test4, test5, test6, test7 = (
        jnp.array(19390.35027108, dtype=jnp.float64), 
        jnp.array(-216.0310416, dtype=jnp.float64), 
        jnp.array(-1859.43760286, dtype=jnp.float64), 
        jnp.array(4.17157824, dtype=jnp.float64), 
        jnp.array(-51.42695148, dtype=jnp.float64), 
        jnp.array(-51.42695148, dtype=jnp.float64), 
        jnp.array(-45854.7002203, dtype=jnp.float64))

    # Tests
    assert jnp.allclose(dofs, test1), 'Incorrect solution.'
    assert jnp.allclose(du_dEm, test2), 'Incorrect first order sensitivities in forward mode'
    assert jnp.allclose(du_dnu, test3), 'Incorrect first order sensitivities in forward mode'
    assert jnp.allclose(d2u_dEm2, test4), 'Incorrect second order sensitivities in forward mode'
    assert jnp.allclose(d2u_dEmdnu, test5), 'Incorrect second order sensitivities in forward mode'
    assert jnp.allclose(d2u_dnudEm, test6), 'Incorrect second order sensitivities in forward mode'
    assert jnp.allclose(d2u_dnu2, test7), 'Incorrect second order sensitivities in forward mode'

    assert jnp.allclose(sol_rev, test1), 'Incorrect solution.'
    assert jnp.allclose(dudnu_rev, test3), 'Incorrect in first order sensitivity in backward mode.'
    assert jnp.allclose(d2udnu2_rev, test7), 'Incorrect in first order sensitivity in backward mode.'

#     computing_time = time.time() - start
#     print('All tests passed. Compute time: ', computing_time)

# test_implicit_diff()
