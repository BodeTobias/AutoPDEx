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
This example addresses Cook's membrane problem where the Young's modulus 
and Poisson's ratio follow a multivariate normal distribution. The goal 
is to compute the expectation and standard deviation of the displacement 
field. Two approaches are compared: the Monte Carlo method and an approach
based on a Taylor series expansion. For the Taylor series expansion, 
the derivatives of the displacement degrees of freedom with respect to 
Young's modulus and Poisson's ratio are required. These sensitivities 
are calculated using automatic implicit differentiation based on the 
implicit function theorem.
"""


if __name__ == "__main__":
    import time
    import sys

    import numpy as np
    import meshio
    import pygmsh
    import jax
    from jax import config
    import jax.numpy as jnp
    import flax

    from autopdex import assembler, seeder, geometry, solver, utility, models, spaces, implicit_diff

    config.update("jax_enable_x64", True)
    config.update("jax_compilation_cache_dir", './cache')

    # Order of ansatz
    order = 2

    ### Definition of geometry and boundary conditions
    pts = [[0.,0.], [48.,44.], [48.,60.], [0.,44.]]
    with pygmsh.occ.Geometry() as geom:
        # region = geom.add_polygon(pts, mesh_size=10.0)
        region = geom.add_polygon(pts, mesh_size=1.0)
        geom.set_recombined_surfaces([region.surface])
        mesh = geom.generate_mesh(order=order)
        print("Mesh generation finished.")

    # Import mesh
    n_dim = 2
    x_nodes = jnp.asarray(mesh.points[:,:n_dim])
    n_nodes = x_nodes.shape[0]
    print("Number of nodes: ", n_nodes)
    elements = jnp.asarray([ v for k, v in mesh.cells_dict.items() if 'quad' in k])[0]
    surface_elements = jnp.asarray([ v for k, v in mesh.cells_dict.items() if 'line' in k])[0]

    # Selection of nodes for boundary condition
    dirichlet_nodes = geometry.in_planes(x_nodes, pts[0], [1.,0.])
    dirichlet_dofs = utility.dof_select(dirichlet_nodes, jnp.asarray([True, True]))
    dirichlet_conditions = jnp.zeros_like(dirichlet_dofs, dtype=jnp.float_)

    # Import surface mesh for inhomogeneous Neumann conditions
    neumann_selection = geometry.select_elements_in_plane(x_nodes, surface_elements, pts[1], [1.,0.])
    neumann_elements= surface_elements[neumann_selection]


    ### Built-in autopdex element
    weak_form_fun_1 = models.hyperelastic_steady_state_weak(
    models.neo_hooke, 
    lambda x, settings: settings['youngs modulus'], # Youngs modulus loaded from settings
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
    'solver backend': 'pardiso',
    'solver': 'lu',
    'verbose': 0,
    'dirichlet dofs': utility.jnp_to_tuple(dirichlet_dofs),
    'connectivity': (utility.jnp_to_tuple(elements), utility.jnp_to_tuple(neumann_elements)),
    })
    settings = {
    'load multiplier': q_0,
    'node coordinates': x_nodes,
    'dirichlet conditions': dirichlet_conditions,
    }



    ### Switch: Monte Carlo/ Taylor series
    monte_carlo = False
    key = jax.random.PRNGKey(123)
    if monte_carlo:
        n_monte_carlo = 1000
    else:
        n_monte_carlo = 10000

    # Generate samples
    means = jnp.asarray([100., 0.3])
    covs = jnp.asarray([[10., 0.0001], [0.0001, 0.001]])
    samples = jax.random.multivariate_normal(key, means, covs, (n_monte_carlo,))
    youngs_mod_mc = samples[:,0]
    nu_mc = samples[:,1]

    ### Estimate uncertainties with Monte Carlo
    start = time.time()
    if monte_carlo:
        key = jax.random.PRNGKey(123)

        # Generate samples
        means = jnp.asarray([100., 0.3])
        covs = jnp.asarray([[10., 0.0001], [0.0001, 0.001]])
        samples = jax.random.multivariate_normal(key, means, covs, (n_monte_carlo,))
        youngs_mod_mc = samples[:,0]
        nu_mc = samples[:,1]

        # Check the stochasticity of the material parameters
        mean_Em_test = youngs_mod_mc.sum()/n_monte_carlo
        variance_Em_test = jnp.sum((youngs_mod_mc - mean_Em_test)**2) / n_monte_carlo
        print("Expectation value of youngs modulus: ", mean_Em_test)
        print("Variance of youngs modulus: ", variance_Em_test)

        mean_nu_test = nu_mc.sum()/n_monte_carlo
        variance_nu_test = jnp.sum((nu_mc - mean_nu_test)**2) / n_monte_carlo
        print("Expectation value of Poisson ratio: ", mean_nu_test)
        print("Variance of Poisson ratio: ", variance_nu_test)

        @jax.jit
        def compute_dofs(dofs, settings, Em, nu):
            settings['youngs modulus'] = Em
            settings['poisson ratio'] = nu
            return solver.adaptive_load_stepping(dofs, 
                                                 settings, 
                                                 static_settings, 
                                                 multiplier_settings, 
                                                 False, 
                                                 'reverse',
                                                 min_increment=0.001, 
                                                 newton_tol=1e-8)[0]

        # Batched forward analysis and uncertainty estimation via Monte Carlo
        dofs_mc = jnp.asarray([compute_dofs(dofs_0, settings, youngs_mod_mc[i], nu_mc[i]) for i in range(youngs_mod_mc.shape[0])])
        mean_dofs = jnp.sum(dofs_mc, axis=0) / n_monte_carlo
        variance_dofs = jnp.sum((dofs_mc - mean_dofs)**2, axis=0) / n_monte_carlo
        std_dofs = jnp.sqrt(variance_dofs)

        skewness_dofs = jnp.sum((dofs_mc - mean_dofs)**3, axis=0) / n_monte_carlo / std_dofs**3
        kurtosis_dofs = jnp.sum((dofs_mc - mean_dofs)**4, axis=0) / n_monte_carlo / std_dofs**4

        print("Time needed for Monte Carlo analysis: ", time.time() - start)

        ## Postprocessing
        points = mesh.points
        cells = jnp.asarray([ v for k, v in mesh.cells_dict.items() if 'quad' in k])[0]
        mesh = meshio.Mesh(
            points,
            {'quad': cells[:,:4]},
            point_data={
                "expectation(u)": mean_dofs,
                "variance(u)": variance_dofs,
                "std_deviation(u)": std_dofs,
                "skewness(u)": skewness_dofs,
                "kurtosis(u)": kurtosis_dofs,
            },
        )
        mesh.write("uncertainties_monte_carlo.vtk")

    ### Estimate uncertainties via Taylor series with implicit differentiation
    else:
        youngs_mod_mean = means[0]
        nu_mean = means[1]

        def compute_dofs(dofs, settings, Em, nu):
            settings['youngs modulus'] = Em
            settings['poisson ratio'] = nu
            return solver.adaptive_load_stepping(dofs, settings, static_settings, multiplier_settings, False, 'forward', newton_tol=1e-8)[0]


        ## Sensitivity analysis
        # # Version 1 using jacfwd_upto_n_scalar_args
        # forward_and_sensitivity = lambda Em, nu: compute_dofs(dofs_0, settings, Em, nu)
        # result = utility.jacfwd_upto_n_scalar_args(forward_and_sensitivity, (youngs_mod_mean, nu_mean), 1, (0, 1))
        # dofs, du_dEm, du_dnu = result[0,0], result[0,1], result[1,1]

        # # Higher order sensitivities (needs longer compile time...)
        # forward_and_sensitivity = lambda Em, nu: compute_dofs(dofs_0, settings, Em, nu)
        # result = utility.jacfwd_upto_n_scalar_args(forward_and_sensitivity, (youngs_mod_mean, nu_mean), 3, (0, 1))
        # dofs, du_dEm, du_dnu, d2u_dEm2, d2u_dnu2, d3u_dEm3, d3u_dnu3 = result[0,0], result[1,0], result[1,1], result[2,0], result[2,1], result[3,0], result[3,1]

        # # Version 2 using jacfwd_upto_n_one_vector_arg
        # forward_and_sensitivity = lambda v: compute_dofs(dofs_0, settings, v[0], v[1])
        # result = utility.jacfwd_upto_n_one_vector_arg(forward_and_sensitivity, jnp.asarray([youngs_mod_mean, nu_mean]), 1)
        # dofs, du_dEm, du_dnu = result[0], result[1][0], result[1][1]

        # Second order mixed sensitivities
        forward_and_sensitivity = lambda v: compute_dofs(dofs_0, settings, v[0], v[1])
        result = utility.jacfwd_upto_n_one_vector_arg(forward_and_sensitivity, jnp.asarray([youngs_mod_mean, nu_mean]), 2)
        dofs, du_dEm, du_dnu, d2u_dEm2, d2u_dEmdnu, d2u_dnudEm, d2u_dnu2 = result[0], result[1][0], result[1][1], result[2][0][0], result[2][0][1], result[2][1][0], result[2][1][1]

        # # Third order mixed sensitivities (needs longer compile time...)
        # forward_and_sensitivity = lambda v: compute_dofs(dofs_0, settings, v[0], v[1])
        # result = utility.jacfwd_upto_n_one_vector_arg(forward_and_sensitivity, jnp.asarray([youngs_mod_mean, nu_mean]), 3)
        # dofs, du_dEm, du_dnu, d2u_dEm2, d2u_dEmdnu, d2u_dnudEm, d2u_dnu2, d3u_dnu3, d3u_dnu2dEm = result[0], result[1][0], result[1][1], result[2][0][0], result[2][0][1], result[2][1][0], result[2][1][1], result[3][1][1][1], result[3][1][1][0]

        print("Sensitivity analysis finished. Starting uncertainty estimation...")

        ## First order Taylor series expansion
        # # Estimate expectation of displacement field via first-order Taylor expansions
        # def dofs_i(dofs, du_dEm, du_dnu, youngs_mod_i, nu_i):
        #     return dofs + du_dEm * (youngs_mod_i - youngs_mod_mean) + du_dnu * (nu_i - nu_mean)
        # mean_dofs = jnp.sum(jax.jit(jax.vmap(dofs_i, (None, None, None, 0, 0), 0))(dofs, du_dEm, du_dnu, youngs_mod_mc, nu_mc), axis=0) / n_monte_carlo

        # # Estimate variance, standard deviation, skewness, and kurtosis of displacement field via first-order Taylor expansions with monte carlo integration
        # def moments_i(dofs, du_dEm, du_dnu, youngs_mod_i, nu_i):
        #     delta = du_dEm * (youngs_mod_i - youngs_mod_mean) + du_dnu * (nu_i - nu_mean) + dofs - mean_dofs
        #     var = delta**2
        #     skew = delta**3
        #     kurt = delta**4
        #     return var, skew, kurt

        # # Calculate moments
        # var_skew_kurt = jax.jit(jax.vmap(moments_i, (None, None, None, 0, 0), 0))(dofs, du_dEm, du_dnu, youngs_mod_mc, nu_mc)

        # # Variance and standard deviation
        # variance_dofs = jnp.sum(var_skew_kurt[0], axis=0) / n_monte_carlo
        # std_dofs = jnp.sqrt(variance_dofs)

        # # Skewness: normalized third central moment
        # skewness_dofs = jnp.sum(var_skew_kurt[1], axis=0) / n_monte_carlo / (std_dofs**3)
        
        # # Kurtosis: normalized fourth central moment
        # kurtosis_dofs = jnp.sum(var_skew_kurt[2], axis=0) / n_monte_carlo / (std_dofs**4)

        ## Second order Taylor series expansion
        @jax.jit
        def compute_mean_var_skew_kurst(dofs, du_dEm, du_dnu, d2u_dEm2, d2u_dnu2, d2u_dEmdnu, youngs_mod_mc, nu_mc):
            # Estimate expectation of displacement field via second-order Taylor expansions
            def dofs_i(dofs, du_dEm, du_dnu, d2u_dEm2, d2u_dnu2, d2u_dEmdnu, youngs_mod_i, nu_i):
                delta_Em = youngs_mod_i - youngs_mod_mean
                delta_nu = nu_i - nu_mean
                linear_term = du_dEm * delta_Em + du_dnu * delta_nu
                quadratic_term = 0.5 * (d2u_dEm2 * delta_Em**2 + d2u_dnu2 * delta_nu**2 + 2 * d2u_dEmdnu * delta_Em * delta_nu)
                return dofs + linear_term + quadratic_term
            mean_dofs = jnp.sum(jax.jit(jax.vmap(dofs_i, (None, None, None, None, None, None, 0, 0), 0))(
                dofs, du_dEm, du_dnu, d2u_dEm2, d2u_dnu2, d2u_dEmdnu, youngs_mod_mc, nu_mc), axis=0) / n_monte_carlo

            # Estimate variance, standard deviation, skewness, and kurtosis of displacement field via second-order Taylor expansions with monte carlo integration
            def moments_i(dofs, du_dEm, du_dnu, d2u_dEm2, d2u_dnu2, d2u_dEmdnu, youngs_mod_i, nu_i):
                delta_Em = youngs_mod_i - youngs_mod_mean
                delta_nu = nu_i - nu_mean
                linear_term = du_dEm * delta_Em + du_dnu * delta_nu
                quadratic_term = 0.5 * (d2u_dEm2 * delta_Em**2 + d2u_dnu2 * delta_nu**2 + 2 * d2u_dEmdnu * delta_Em * delta_nu)
                delta = linear_term + quadratic_term + dofs - mean_dofs
                var = delta**2
                skew = delta**3
                kurt = delta**4
                return var, skew, kurt

            # Calculate moments with second-order Taylor expansion
            var_skew_kurt = jax.jit(jax.vmap(moments_i, (None, None, None, None, None, None, 0, 0), 0))(
                dofs, du_dEm, du_dnu, d2u_dEm2, d2u_dnu2, d2u_dEmdnu, youngs_mod_mc, nu_mc)

            # Variance and standard deviation
            variance_dofs = jnp.sum(var_skew_kurt[0], axis=0) / n_monte_carlo
            std_dofs = jnp.sqrt(variance_dofs)

            # Skewness: normalized third central moment
            skewness_dofs = jnp.sum(var_skew_kurt[1], axis=0) / n_monte_carlo / (std_dofs**3)

            # Kurtosis: normalized fourth central moment
            kurtosis_dofs = jnp.sum(var_skew_kurt[2], axis=0) / n_monte_carlo / (std_dofs**4)
            return mean_dofs, variance_dofs, std_dofs, skewness_dofs, kurtosis_dofs
        mean_dofs, variance_dofs, std_dofs, skewness_dofs, kurtosis_dofs = compute_mean_var_skew_kurst(dofs, du_dEm, du_dnu, d2u_dEm2, d2u_dnu2, d2u_dEmdnu, youngs_mod_mc, nu_mc)

        ## Postprocessing
        print("Time needed for Taylor series/ implicit differentiation analysis: ", time.time() - start)
        points = mesh.points
        cells = jnp.asarray([ v for k, v in mesh.cells_dict.items() if 'quad' in k])[0]
        mesh = meshio.Mesh(
            points,
            {'quad': cells[:,:4]},
            point_data={
                "u": dofs,
                "expectation(u)": mean_dofs,
                "variance(u)": variance_dofs,
                "std_deviation(u)": std_dofs,
                "skewness(u)": skewness_dofs,
                "kurtosis(u)": kurtosis_dofs,
                "du/dEm": du_dEm,
                "du/dnu": du_dnu,

                # "d2u/dnu2": d2u_dnu2,
                # "d2u/dEm2": d2u_dEm2,
                # "d3u/dnu3": d3u_dnu3,
                # "d3u/dEm3": d3u_dEm3,

                "d2u/dnu2": d2u_dnu2,
                "d2u/dEmdnu": d2u_dEmdnu,
                "d2u/dnudEm": d2u_dnudEm,
                "d2u/dEm2": d2u_dEm2,
                
                # "d3u/dnu3": d3u_dnu3,
                # "d3u/dnu2dEm": d3u_dnu2dEm,
            },
        )
        mesh.write("uncertainties_taylor.vtk")
