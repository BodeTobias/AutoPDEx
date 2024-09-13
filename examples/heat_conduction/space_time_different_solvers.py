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

import sys
import time

import jax
from jax import config
import jax.numpy as jnp
import flax

from autopdex import seeder, geometry, solver, solution_structures, plotter, utility, models, assembler

config.update("jax_enable_x64", True)
config.update("jax_compilation_cache_dir", './cache')

### Dirichlet problem of transient heat conduction equation (moving least square solution structure)
# Comparison of different linear solvers


# Positive smooth distance functions that select the specific boundary segment
# Time dimension is always the last dimension (here the second)
psdf_bottom = lambda x: geometry.psdf_trimmed_line(x, jnp.stack([0.,0.]), jnp.stack([1.,0.]))
psdf_right = lambda x: geometry.psdf_trimmed_line(x, jnp.stack([1.,0.]), jnp.stack([1.,1.]))
psdf_left = lambda x: geometry.psdf_trimmed_line(x, jnp.stack([0.,1.]), jnp.stack([0.,0.]))
psdfs = [psdf_bottom, psdf_right, psdf_left]

# Positive smooth distance funcion of full domain (used for seeding nodes and integration points)
psdf_rect = lambda x: geometry.psdf_parallelogram(x, jnp.stack([0.,0.]), jnp.stack([0.,1.]), jnp.stack([1.,0.]))
psdf_domain = psdf_rect

# Initial condition
def bc_bottom(x):
  return jax.lax.sin(jnp.pi * x[0])

# Dirichlet boundary conditions on left an right side
def bc_left(x):
  return 0.
def bc_right(x):
  return 0.

# Assignment of boundary conditions (has to be in the same order as the psdfs)
dirichlet_conditions = [bc_bottom, bc_right, bc_left]

# Transfinite interpolation of boundary segments
b_cond = lambda x: geometry.transfinite_interpolation(x, psdfs, dirichlet_conditions)
psdf = lambda x: geometry.psdf_unification(x, psdfs)

### Initialization and generation of nodes and integration points
nod_spacing = 0.02
(x_nodes, n_nodes, region_size) = seeder.regular_in_psdf(psdf_domain, [0.,0.], [1.,1.], nod_spacing)

# Integration points
int_spacing = nod_spacing
(x_int, w_int, n_int) = seeder.gauss_points_in_psdf(psdf_domain, [0.,0.], [1.,1.], int_spacing, order=2)

# Support radius for moving least square ansatz
support_radius = 3.51 * nod_spacing

# Definition of pde
diffusivity = lambda x: 0.5
pde = models.heat_equation(diffusivity_fun=diffusivity)

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

  # See different solver options below.
  # Remarks:
  # For small to medium problems on cpu, pardiso is a good choice
  # If tangent can not be loaded in RAM: try matrix-free solvers
  # For very large problems: try multigrid preconditioners on (gpu) cluster (petsc, pyamg)
  # For itterative solvers: setting tol or maxiter to low can reduce the solution accuracy

  # 'solver type': 'linear',
  # 'solver backend': 'pyamg',                # for further options see pyamg documentation
  # 'solver': 'cg',                           # cg, bcgs
  # 'type of preconditioner': 'ruge stuben',  # ruge stuben, smoothed aggregation 

  # 'solver type': 'linear',
  # 'solver backend': 'petsc',        # available options see PETSc documentation
  # 'solver': 'bcgs',
  # 'type of preconditioner': 'ilu',

  # 'solver type': 'linear',
  # 'solver backend': 'jax',              # if possible, uses matrix-free algorithms
  # 'solver': 'cg',                       #cg, bicgstab, gmres(, cg normal, lu, cholesky, qr, jacobi)
  # 'type of preconditioner': 'jacobi',   # 'jacobi', 'none'
  # 'hvp type': 'fwdrev',                 # fwdrev, revrev (only for symmetric matrices), linearize, assemble

  # 'solver type': 'linear',
  # 'solver backend': 'scipy',  # only on cpu
  # 'solver': 'lapack',         # lapack, umfpack

  'solver type': 'linear',
  'solver backend': 'pardiso', # only on cpu
  'solver': 'lu', # lu, qr

  # # Only works for problems where a functional can be minimized (e.g. least square pde loss, least square function approximation)
  # 'solver type': 'minimize',  # maxiter and tol may have to be adjusted (see effect on optimized L2 error); 
  # 'solver': 'lbfgs',          # lbfgs, gradient descent, nonlinear cg, bfgs, gauss newton, levenberg marquart

  'verbose': 2,
  'connectivity': (utility.jnp_to_tuple(neighbor_list),),
})
settings = {
  'beta': (3,),
  'node coordinates': x_nodes,
  'integration coordinates': (x_int,),
  'integration weights': (w_int,),
  'support radius': (support_radius,),
}


### Precompute shape functions and pass them to e.g. static_settings...
settings = solution_structures.precompile(dofs0, settings, static_settings)


### Call solver
dofs, _ = solver.solver(dofs0, settings, static_settings, tol=1e-8, maxiter=1000)


### Preperation of postprocessing data
local_dofs = dofs[neighbor_list]
def post_fun(x, itt, local_dofs, settings, static_settings):
  return solution_structures.solution_structure(x, itt, local_dofs, settings, static_settings, 0)

post_fun_vj = jax.jit(jax.vmap(post_fun, (0,0,0,None,None)), static_argnames=['static_settings'])
data = post_fun_vj(x_int, jnp.arange(n_int), local_dofs, settings, static_settings)


# Plotting with pyvista
plotter.pv_plot(x_int, data[:,0], export_vtk=False, show=True)
