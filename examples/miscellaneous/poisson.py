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

from autopdex import seeder, geometry, solver, solution_structures, plotter, utility, models, spaces

config.update("jax_enable_x64", True)
config.update("jax_compilation_cache_dir", './cache')


# Dirichlet problem of the Poisson equation in 2d: [0,1]x[0,1]
# Demonstration of compile time reduction by manually pre-compiling solution structure


# Positive smooth distance functions that select the specific boundary segments
psdf_bottom = lambda x: geometry.psdf_trimmed_line(x, [0.,0.], [1.,0.])
psdf_right = lambda x: geometry.psdf_trimmed_line(x, [1.,0.], [1.,1.])
psdf_top = lambda x: geometry.psdf_trimmed_line(x, [1.,1.], [0.,1.])
psdf_left = lambda x: geometry.psdf_trimmed_line(x, [0.,1.], [0.,0.])
psdf_rect = lambda x: geometry.psdf_parallelogram(x, [0.,0.], [0.,1.], [1.,0.])
psdfs = [psdf_bottom, psdf_right, psdf_top, psdf_left]

# Positive smooth distance funcion of full domain (used for seeding nodes and integration points)
psdf_domain = psdf_rect

# Boundary condition functions
def homogeneous(x): # Bottom, right and top boundaries
  return 0

# Assignment of boundary conditions
dirichlet_conditions = [homogeneous, homogeneous, homogeneous, homogeneous]


### Transfinite interpolation of boundary segments
b_cond = lambda x: geometry.transfinite_interpolation(x, psdfs, dirichlet_conditions)
psdf = psdf_domain

### Initialization and generation of nodes and integration points
# Regular node placement
nod_spacing = 0.02
(x_nodes, n_nodes, region_size) = seeder.regular_in_psdf(psdf_domain, [0.,0.], [1.,1.], nod_spacing)

# Integration points
# Collocation, Note: this can in some cases cause bad conditioning, unphysical low-energy modes and divergence. 
# If so, use more integration points, e.g. Gauss integration on a background grid (seeder.gauss_points_in_psdf)
x_int = x_nodes
n_int = n_nodes

# Support radius for moving least square ansatz
support_radius = 2.51 * nod_spacing

# Integration point weights
w_int = (region_size / n_int) * jnp.ones(n_int)

# Definition of pde
pde = models.poisson(source_fun = lambda x: 1)


### Neighborhood search
(num_neighbors, max_neighbors, min_neighbors, neighbor_list) = utility.search_neighborhood(x_nodes, x_int, support_radius)


### Settings
n_fields = 1
dofs0 = jnp.zeros((n_nodes, n_fields))
static_settings = flax.core.FrozenDict({

  # Precompilation of the solution structure at the integration points can considerable improve compile time
  # 'shape function mode': 'direct',
  'shape function mode': 'compiled', # precompilation is done in line 107
  
  'solution space': ('mls',),
  'number of fields': (n_fields,),
  'assembling mode': ('sparse',),
  'maximal number of neighbors': (max_neighbors,),
  'order of basis functions': (2,),
  'weight function type': ('gaussian',),
  'variational scheme': ('strong form galerkin',),
  'model': (pde,),
  'solution structure': ('dirichlet',),
  'boundary conditions': (b_cond,),
  'psdf': (psdf,),
  'solver type': 'linear',

  'solver backend': 'pardiso',
  'solver': 'lu',
  # 'solver backend': 'jax',
  # 'solver': 'bicgstab',
  # 'type of preconditioner': 'jacobi',
  # 'hvp type': 'fwdrev',

  'verbose': 2,
  'connectivity': (utility.jnp_to_tuple(neighbor_list),),
})
settings = {
  'node coordinates': x_nodes,
  'beta': (3,),
  'integration coordinates': (x_int,),
  'integration weights': (w_int,),
  'support radius': (support_radius,),
}

### Precompilation of solution structure
start = time.time()
settings = solution_structures.precompile(dofs0, settings, static_settings)


### Call solver
dofs, _ = solver.solver(dofs0, settings, static_settings)
print("Time needed for compiling and solving: ", time.time() - start)

### Preperation of postprocessing data
local_dofs = dofs[neighbor_list]
post_fun_vj = jax.jit(jax.vmap(solution_structures.solution_structure, (0,0,0,None,None,None)), static_argnames=['static_settings', 'set'])
data = post_fun_vj(x_int, jnp.arange(n_int), local_dofs, settings, static_settings, 0)

# Plotting with pyvista
plotter.pv_plot(x_int, data[:,0], export_vtk=False, show=True)
