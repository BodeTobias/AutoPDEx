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

from autopdex import seeder, geometry, solver, solution_structures, plotter, utility, models

config.update("jax_enable_x64", True)
config.update("jax_compilation_cache_dir", './cache')



### Definition of geometry and boundary conditions

# Positive smooth distance functions that select the specific boundary segments
# Just inital condition
psdf_bottom = lambda x: geometry.psdf_trimmed_line(x, [x_0,0.], [x_1,0.])
psdf_left = lambda x: geometry.psdf_trimmed_line(x, [x_0,0.], [x_0,t_1])
psdf_right = lambda x: geometry.psdf_trimmed_line(x, [x_1,0.], [x_1,t_1])
psdfs = [psdf_bottom, psdf_left, psdf_right]
psdf = lambda x: geometry.psdf_unification(x, psdfs)

# Positive smooth distance funcion of full domain (used for seeding nodes and integration points)
psdf_rect = lambda x: geometry.psdf_parallelogram(x, [x_0,0.], [x_0,t_1], [x_1,0.])
psdf_domain = psdf_rect

# Boundary condition functions
t_1 = 2.0
x_0 = -2.0
x_1 = 2.0
initial_condition = lambda x: (jnp.pi / 4) - jnp.arctan(x[0])
left_clamping = lambda x: (jnp.pi / 4) - jnp.arctan(x_0)
right_clamping = lambda x: (jnp.pi / 4) - jnp.arctan(x_1)
b_cond = lambda x: geometry.transfinite_interpolation(x, psdfs, [initial_condition, left_clamping, right_clamping])


### Initialization and generation of nodes and integration points
# Regular node placement
nod_spacing = 0.05
(x_nodes, n_nodes, region_size) = seeder.regular_in_psdf(psdf_domain, [x_0,0.], [x_1,t_1], nod_spacing)

# Gauss integration on background grid
int_spacing = nod_spacing
(x_int, w_int, n_int) = seeder.gauss_points_in_psdf(psdf_domain, [x_0,0.], [x_1,t_1], int_spacing, order=2)
# # Collocation
# x_int = x_nodes
# n_int = n_nodes
# w_int = (region_size / n_int) * jnp.ones(n_int)
# print("Estimated region size: ", w_int.sum())

# Support radius for moving least square ansatz
support_radius = 2.51 * nod_spacing

# Definition of pde
pde = models.burgers_equation_inviscid()


# ### Visualizer for boundary functions
# fun = lambda x: b_cond(x)
# plotter.pv_function_plot(x_int, fun)
# sys.exit()


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
  'order of basis functions': (1,),
  'weight function type': ('gaussian',),
  'variational scheme': ('least square pde loss',),
  'model': (pde,),
  'solution structure': ('dirichlet',),
  'boundary conditions': (b_cond,),
  'psdf': (psdf,),
  'solver type': 'minimize',
  'solver': 'lbfgs',
  'verbose': 1,
  })
settings = {
  'connectivity': (neighbor_list,),
  'node coordinates': x_nodes,
  'integration coordinates': (x_int,),
  'integration weights': (w_int,),
  'beta': (3,),
  'support radius': (support_radius,),
  }


### Optional: Precompute shape functions and pass them to e.g. static_settings...
settings = solution_structures.precompile(dofs0, settings, static_settings)


### Call solver
start = time.time()
dofs, _ = solver.solver(dofs0, settings, static_settings, tol=1e-8, maxiter=10000)
print("Computing time of solver: ", time.time() - start)


### Preperation of postprocessing data
local_dofs = dofs[neighbor_list]
def post_fun(x, itt, local_dofs, settings, static_settings):
  return solution_structures.solution_structure(x, itt, local_dofs, settings, static_settings, 0)

post_fun_vj = jax.jit(jax.vmap(post_fun, (0,0,0,None,None)), static_argnames=['static_settings'])
data = post_fun_vj(x_int, jnp.arange(n_int), local_dofs, settings, static_settings)

# Plotting with pyvista
plotter.pv_plot(x_int, data[:,0], export_vtk=False, show=True, scale_range=0.2)
