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
from jax import lax, config
import jax.numpy as jnp
import flax

from autopdex import assembler, seeder, geometry, solver, solution_structures, plotter, utility, models, spaces

config.update("jax_enable_x64", True)
config.update("jax_compilation_cache_dir", './cache')


### Transport equation as moving least square solution structure with least square variational method


### Definition of geometry and boundary conditions
# Linear transport problem

# Positive smooth distance functions that select the specific boundary segments
# Just inital condition
psdf_left = lambda x: geometry.psdf_trimmed_line(x, jnp.stack([0.,0.]), jnp.stack([0.,1.]))
psdf_bottom = lambda x: geometry.psdf_trimmed_line(x, jnp.stack([0.,0.]), jnp.stack([1.,0.]))
psdfs = [psdf_bottom, psdf_left]
psdf = lambda x: geometry.psdf_unification(x, psdfs)

# Positive smooth distance funcion of full domain (used for seeding nodes and integration points)
psdf_rect = lambda x: geometry.psdf_parallelogram(x, jnp.stack([0.,0.]), jnp.stack([0.,1.]), jnp.stack([1.,0.]))
psdf_domain = psdf_rect

# Boundary condition functions
initial_condition = lambda x: jnp.sin(2 * jnp.pi * x[0])
homogeneous = lambda x: 0.
b_cond = lambda x: geometry.transfinite_interpolation(x, psdfs, [initial_condition, homogeneous])

### Initialization and generation of nodes and integration points
# Regular node placement
nod_spacing = 0.02
(x_nodes, n_nodes, region_size) = seeder.regular_in_psdf(psdf_domain, [0.,0.], [1.,1.], nod_spacing)

# Gauss integration on background grid
int_spacing = nod_spacing
(x_int, w_int, n_int) = seeder.gauss_points_in_psdf(psdf_domain, [0.,0.], [1.,1.], int_spacing, order=2)

# Support radius for moving least square ansatz
support_radius = 3.01 * nod_spacing

# Definition of pde
pde = models.transport_equation(c = 0.5)

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
  'order of basis functions': (2,),
  'weight function type': ('gaussian',),
  'variational scheme': ('least square pde loss',),
  'model': (pde,),
  'solution structure': ('dirichlet',),
  'boundary conditions': (b_cond,),
  'psdf': (psdf,),
  'solver type': 'linear',
  'solver backend': 'pardiso',
  'solver': 'lu',
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
