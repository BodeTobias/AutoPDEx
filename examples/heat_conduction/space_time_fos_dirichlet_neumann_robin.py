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
import os

import jax
from jax import config
config.update("jax_enable_x64", True)
config.update("jax_compilation_cache_dir", './cache')

import jax.numpy as jnp
import flax

from autopdex import seeder, geometry, solver, solution_structures, plotter, utility, models


### First order system of stsationary heat conduction, mls solution structure, dirichlet/neumann/robin conditions


# Positive smooth distance functions that select the specific boundary segment
# Time dimension is always the last dimension (here the second)
psdf_bottom = lambda x: geometry.psdf_trimmed_line(x, [0.,0.], [1.,0.])
psdf_right = lambda x: geometry.psdf_trimmed_line(x, [1.,0.], [1.,1.])
psdf_left = lambda x: geometry.psdf_trimmed_line(x, [0.,1.], [0.,0.])
psdfs = [psdf_bottom, psdf_right, psdf_left]

# Positive smooth distance funcion of full domain (used for seeding nodes and integration points)
psdf_rect = lambda x: geometry.psdf_parallelogram(x, [0.,0.], [0.,1.], [1.,0.])
psdf_domain = psdf_rect

# Initial condition
def bc_bottom(x):
  return 0.5 * (1 - jax.lax.cos(2 * jnp.pi * x[0]))

# Dirichlet boundary conditions on left an right side
def bc_left(x):
  return 0.
def bc_right(x):
  return 0.
dirichlet_conditions = [bc_bottom, bc_right, bc_left]

# Field selector for boundary conditions
def dirichlet(x):
  return jnp.asarray([1., 0.])  # 1 * theta + 0 * dtheta / dx = bc_bottom
def neumann(x):
  return jnp.asarray([0., 1.]) # 0 * theta + 1 * dtheta / dx = bc_right
def robin(x):
  return jnp.asarray([0.980581, -0.196116]) # 0.980581 * theta - 0.196116 * dtheta / dx = bc_left
a_i = [dirichlet, neumann, robin]

# Transfinite interpolation of boundary segments
b_coeff = lambda x: jnp.asarray([geometry.transfinite_interpolation(x, psdfs, a_i)])
b_cond = lambda x: jnp.asarray([geometry.transfinite_interpolation(x, psdfs, dirichlet_conditions)])
psdf = lambda x: jnp.asarray([geometry.psdf_unification(x, psdfs)])

### Initialization and generation of nodes and integration points
nod_spacing = 0.01
(x_nodes, n_nodes, region_size) = seeder.regular_in_psdf(psdf_domain, [0.,0.], [1.,1.], nod_spacing)

# Collocation
x_int = x_nodes
n_int = n_nodes
w_int = (region_size / n_int) * jnp.ones(n_int)
print("Estimated region size: ", w_int.sum())

# Support radius for moving least square ansatz
support_radius = 2.51 * nod_spacing

# Definition of pde
diffusivity = lambda x: 0.1
pde = models.heat_equation_fos(diffusivity_fun=diffusivity)

### Neighborhood search
(num_neighbors, max_neighbors, min_neighbors, neighbor_list) = utility.search_neighborhood(x_nodes, x_int, support_radius)


### Settings
n_fields = 2
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
  'solution structure': ('first order set',),
  'boundary coefficients': (b_coeff,),
  'boundary conditions': (b_cond,),
  'psdf': (psdf,),
  'solver type': 'linear',
  'solver backend': 'pardiso',
  'solver': 'lu',
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

# Precompute shape functions and pass them to e.g. static_settings...
settings = solution_structures.precompile(dofs0, settings, static_settings)

# Call solver
dofs, _ = solver.solver(dofs0, settings, static_settings, tol=1e-8)

# Preperation of postprocessing data
local_dofs = dofs[neighbor_list]
def post_fun(x, itt, local_dofs, settings, static_settings):
  return solution_structures.solution_structure(x, itt, local_dofs, settings, static_settings, 0)

post_fun_vj = jax.jit(jax.vmap(post_fun, (0,0,0,None,None)), static_argnames=['static_settings'])
data = post_fun_vj(x_int, jnp.arange(n_int), local_dofs, settings, static_settings)

# Plotting with pyvista
plotter.pv_plot(x_int, data[:,0], export_vtk=False, show=True)
