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
import equinox as eqx

from autopdex import seeder, geometry, solver, solution_structures, plotter, models, spaces

config.update("jax_enable_x64", True)
config.update("jax_compilation_cache_dir", './cache')



### Example: Dirichlet problem of the Laplace equation in 2d: [0,1]x[0,1] with cutout circle (midpoint=[0.5,0.5], radius=0.25)
# Neural network as user-defined solution space, boundary conditions through R functions



# Positive smooth distance functions that select the specific boundary segments
psdf_bottom = lambda x: geometry.psdf_trimmed_line(x, [0.,0.], [1.,0.])
psdf_right = lambda x: geometry.psdf_trimmed_line(x, [1.,0.], [1.,1.])
psdf_top = lambda x: geometry.psdf_trimmed_line(x, [1.,1.], [0.,1.])
psdf_left = lambda x: geometry.psdf_trimmed_line(x, [0.,1.], [0.,0.])
psdf_cutout = lambda x: geometry.psdf_nd_sphere_cutout(x, [0.5,0.5], 0.25)
psdf_rect = lambda x: geometry.psdf_parallelogram(x, [0.,0.], [0.,1.], [1.,0.])
psdfs = [psdf_bottom, psdf_right, psdf_top, psdf_left, psdf_cutout]

# Positive smooth distance funcion of full domain (used for seeding nodes and integration points)
psdf_domain = lambda x: geometry.r_equivalence(psdf_rect(x), psdf_cutout(x))

# Boundary condition functions
def homogeneous(x): # Bottom, right and top boundaries
  return 0
def bc_0(x): # Left boundary
  return jax.lax.sin(jnp.pi * x[0])

# Assignment of boundary conditions
dirichlet_conditions = [bc_0, homogeneous, homogeneous, homogeneous, lambda x: 1.0]

# Transfinite interpolation of boundary segments
b_cond = lambda x: geometry.transfinite_interpolation(x, psdfs, dirichlet_conditions)
psdf = psdf_domain

# Integration points
int_spacing = 0.02
(x_int, n_int, region_size) = seeder.regular_in_psdf(psdf_domain, [0.,0.], [1.,1.], int_spacing)
w_int = (region_size / n_int) * jnp.ones(n_int)

# Definition of pde
pde = models.poisson(source_fun=None)


### User defined solution space
key = jax.random.PRNGKey(123)
class CNN(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2, key3 = jax.random.split(key, 3)
        # Neural network with n_dim inputs and n_fields outputs
        self.layers = [
            eqx.nn.Linear(2, 20, key=key1),
            jax.nn.silu,
            eqx.nn.Linear(20, 20, key=key2),
            jax.nn.silu,
            eqx.nn.Linear(20, 1, key=key3),
            jax.nn.silu,
            ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

key, subkey = jax.random.split(key, 2)
model = CNN(subkey)

dofs0, static = eqx.partition(model, eqx.is_array)

# Due to the structure of dofs, in this way it will only work for assembler.integrate_functional and the wrapped solvers 'lbfgs', 'bfgs', 'gradient descent' and 'nonlinear cg' (not for assembler.assemble_residual) 
def neural_network(x, int_point_number, dofs, settings, static_settings, set):
  parametrized_network = eqx.combine(dofs, static)
  return parametrized_network(x)


### Settings
n_fields = 1
static_settings = flax.core.FrozenDict({
  'solution space': ('user',),
  'user solution space function': (neural_network,),
  'shape function mode': 'direct',
  'number of fields': (n_fields,),
  'assembling mode': ('dense',),
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
  'integration coordinates': (x_int,),
  'integration weights': (w_int,),
}


### Here, the built-in wrapper for jaxopt minimizers is used. One may built a user-defined minimizer based on the function assembler.integrate_functional(dofs, settings, static_settings)
start = time.time()
dofs, _ = solver.solver(dofs0, settings, static_settings, tol=1e-12, maxiter=1000)
print("Solver time: ", time.time() - start)


### Preperation of postprocessing data, --> unoptimized solution structure fulfills boundary conditions, but not pde
post_fun_vj = jax.jit(jax.vmap(solution_structures.solution_structure, (0,0,None,None,None,None)), static_argnames=['static_settings', 'set'])
data = post_fun_vj(x_int, jnp.arange(n_int), dofs0, settings, static_settings, 0)
plotter.pv_plot(x_int, data[:,0], export_vtk=False, show=True)


### Preperation of postprocessing data, --> solution structure fulfills boundary conditions, and pde approximately
post_fun_vj = jax.jit(jax.vmap(solution_structures.solution_structure, (0,0,None,None,None,None)), static_argnames=['static_settings', 'set'])
data = post_fun_vj(x_int, jnp.arange(n_int), dofs, settings, static_settings, 0)
plotter.pv_plot(x_int, data[:,0], export_vtk=False, show=True)


### Preperation of postprocessing data, --> solution space does neighter fulfill boundary conditions nor pde
post_fun_vj = jax.jit(jax.vmap(spaces.solution_space, (0,0,None,None,None,None)), static_argnames=['static_settings', 'set'])
data = post_fun_vj(x_int, jnp.arange(n_int), dofs, settings, static_settings, 0)
plotter.pv_plot(x_int, data[:,0], export_vtk=False, show=True)
