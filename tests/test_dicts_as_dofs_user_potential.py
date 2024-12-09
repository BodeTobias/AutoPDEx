def test_example():

  # Imports
  import jax
  import jax.numpy as jnp
  import flax

  from autopdex import seeder, geometry, solver, utility, models, spaces, mesher
  jax.config.update("jax_enable_x64", True)

  # Generate mesh (or import the node coordinates and connectivity)
  pts = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
  coords, elems = mesher.structured_mesh((5, 5), pts, 'quad')
  node_coordinates = {'phi': coords,}
  connectivity = {'phi': elems,}

  # Selection of nodes and degrees of freedom for Dirichlet conditions
  sdf = lambda x: geometry.psdf_polygon(x, pts)
  dirichlet_nodes = geometry.in_sdfs(node_coordinates['phi'], sdf)
  dirichlet_dofs = {'phi': dirichlet_nodes,}
  dirichlet_conditions = utility.dict_zeros_like(dirichlet_dofs, dtype=jnp.float64)

  # Define the variational problem
  def integrand_fun(x_int, ansatz_fun, settings, static_settings, elem_number, set):
      # Definition of custom functional    
      x = ansatz_fun['physical coor'](x_int)
      phi_fun = ansatz_fun['phi']
      phi = phi_fun(x_int)
      dphi_dx = jax.jacrev(phi_fun)(x_int)

      x_1 = x
      x_2 = x - jnp.array([1., 0.5])
      source_term = 20 * (jnp.sin(10 * x_1 @ x_1) - jnp.cos(10 * x_2 @ x_2))
      return (1/2) * dphi_dx @ dphi_dx - source_term * phi

  # Set up the finite element, here Q1 elements for the field 'phi'
  user_potential = models.mixed_reference_domain_potential(
      integrand_fun,
      {'phi': spaces.fem_iso_line_quad_brick,},
      *seeder.gauss_legendre_nd(dimension = 2, order = 2),
      'phi')

  # Prepare the settings for autopdex
  static_settings = flax.core.FrozenDict({
    'assembling mode': ('user potential',),
    'solution structure': ('nodal imposition',),
    'model': (user_potential, ),
    'solver type': 'newton',
    'solver backend': 'scipy',
    'solver': 'lapack',
    'verbose': -1,
  })
  settings = {
    'connectivity': (connectivity,),
    'dirichlet dofs': dirichlet_dofs,
    'node coordinates': node_coordinates,
    'dirichlet conditions': dirichlet_conditions,
  }

  # Compile, assemble and solve linear system
  initial_guess = utility.dict_zeros_like(dirichlet_dofs, dtype=jnp.float64)
  test = solver.solver(initial_guess, settings, static_settings)[0]['phi'].sum()
  assert jnp.isclose(test, 1.9066412530282952)

# test_example()