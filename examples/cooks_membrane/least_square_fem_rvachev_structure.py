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
import pygmsh

from autopdex import seeder, geometry, solver, solution_structures, plotter, utility, models, spaces, assembler

config.update("jax_enable_x64", True)
config.update("jax_compilation_cache_dir", './cache')
config.update("jax_debug_nans", False)


# Cook's membrane problem: first order system least square finite elements (T2/Tri-6) as Rvachev function solution structure



### Definition of geometry and boundary conditions
# Positive smooth distance functions that select the specific boundary segments
pts = [[0.,0.], [48.,44.], [48.,60.], [0.,44.]]
psdf_bottom = lambda x: geometry.psdf_trimmed_line(x, pts[0], pts[1])
psdf_right = lambda x: geometry.psdf_trimmed_line(x, pts[1], pts[2])
psdf_top = lambda x: geometry.psdf_trimmed_line(x, pts[2], pts[3])
psdf_left = lambda x: geometry.psdf_trimmed_line(x, pts[3], pts[0])
psdfs = [psdf_bottom, psdf_right, psdf_top, psdf_left]

# Positive smooth distance funcion of full domain
sdf_cooks = lambda x: geometry.sdf_convex_polygon_2d(x, pts)
psdf_cooks = lambda x: geometry.only_positive(sdf_cooks(x))
psdf = lambda x: jnp.asarray([psdf_cooks(x), psdf_cooks(x)])

# Boundary condition functions
homogeneous = lambda x: 0.
normals = lambda x: geometry.normals_from_normalized_sdf(x, psdf_cooks)

# Field selectors for boundary conditions
selection_dirichlet_1 = lambda x: jnp.asarray([1.,0.,0.,0.,0.,0.])
selection_dirichlet_2 = lambda x: jnp.asarray([0.,1.,0.,0.,0.,0.])
def selection_neumann_1(x):
  normal_x = normals(x)[0]
  normal_y = normals(x)[1]
  return jnp.asarray([0.,0.,normal_x,normal_y,0.,0.])
def selection_neumann_2(x):
  normal_x = normals(x)[0]
  normal_y = normals(x)[1]
  return jnp.asarray([0.,0.,0.,0.,normal_x,normal_y])

# Assignment of boundary conditions (same ordering as in psdfs)
load = lambda x: 4.0
bc_1 = [homogeneous, homogeneous, homogeneous, homogeneous]
bc_2 = [homogeneous, load, homogeneous, homogeneous]
a_1 = [selection_neumann_1, selection_neumann_1, selection_neumann_1, selection_dirichlet_1]
a_2 = [selection_neumann_2, selection_neumann_2, selection_neumann_2, selection_dirichlet_2]

### Transfinite interpolation of boundary segments
b_coeff = lambda x: jnp.asarray([geometry.transfinite_interpolation(x, psdfs, a_1),
                                   geometry.transfinite_interpolation(x, psdfs, a_2)])
b_cond = lambda x: jnp.asarray([geometry.transfinite_interpolation(x, psdfs, bc_1),
                                geometry.transfinite_interpolation(x, psdfs, bc_2)])

### Discretization
### Definition of geometry and boundary conditions
with pygmsh.geo.Geometry() as geom:
    elem_length = 1.
    geom.add_polygon(pts,mesh_size=elem_length)
    mesh = geom.generate_mesh(order=2)

# Mesh debugging
# mesh.write("mesh.vtk")

# Import region mesh
n_dim = 2
x_nodes = jnp.asarray(mesh.points[:,:n_dim])
n_nodes = x_nodes.shape[0]
elements = mesh.cells_dict['triangle6']

# Generate integration points in mesh
(x_int, w_int, n_int, domain_connectivity) = seeder.int_pts_in_tri_mesh(x_nodes, elements, order=3)

# # ### Visualizer for boundary functions
# fun = lambda x: b_cond(x)[1]
# plotter.pv_function_plot(x_int, fun)
# sys.exit()

# Definition of pde
lam = 100.
mu = 40.
Em = mu * (3 * lam + 2 * mu) / (lam + mu)
nu = lam / (2 * (lam + mu))
youngs_mod_fun = lambda x: Em
poisson_ratio_fun = lambda x: nu
# pde = models.linear_elasticity_fos(youngs_mod_fun = youngs_mod_fun, poisson_ratio_fun = poisson_ratio_fun, mode='plain strain', spacing=elem_length)
pde = models.hyperelastic_steady_state_fos(models.neo_hooke, youngs_mod_fun = youngs_mod_fun, poisson_ratio_fun = poisson_ratio_fun, mode='plain strain', spacing=elem_length)

### Setting
n_fields = 6
dofs0 = jnp.zeros((n_nodes, n_fields))
static_settings = flax.core.FrozenDict({
  'solution space': ('fem simplex',),
  'shape function mode': 'compiled',
  'number of fields': (n_fields,),
  'assembling mode': ('sparse',),
  'maximal number of neighbors': (domain_connectivity.shape[-1],),
  'variational scheme': ('least square pde loss',),
  'model': (pde,),
  'solution structure': ('first order set',),
  'boundary coefficients': (b_coeff,),
  'boundary conditions': (b_cond,),
  'psdf': (psdf,),
  'solver type': 'newton',
  'solver backend': 'pardiso',
  'solver': 'lu',
  'verbose': 0,
})
settings = {
  'connectivity': (domain_connectivity,),
  'node coordinates': x_nodes,
  'integration coordinates': (x_int,),
  'integration weights': (w_int,),
}


# Precompute shape functions
settings = solution_structures.precompile(dofs0, settings, static_settings)


### Load stepping in order to provide better initial guesses for the newton method
start = time.time()
dofs = dofs0
multiplier = 0.0
n_step = 5
for i in range(n_step):
  # Update boundary conditions
  print('Start load step ', i+1)
  multiplier = multiplier + 1 / n_step
  load_mult = lambda x: multiplier * load(x)
  bc_2 = [homogeneous, load_mult, homogeneous, homogeneous]
  b_cond = lambda x: jnp.asarray([geometry.transfinite_interpolation(x, psdfs, bc_1),
                                  geometry.transfinite_interpolation(x, psdfs, bc_2)])
  settings['compiled bc'] = (solution_structures.precompute_coupled_boundary(b_cond, b_coeff, x_int, num_diff=1),)
  dofs, _ = solver.solver(dofs, settings, static_settings)



### Preperation of postprocessing data
local_dofs = dofs[domain_connectivity]

def post_fun(x, itt, local_dofs, settings, static_settings):
  return solution_structures.solution_structure(x, itt, local_dofs, settings, static_settings, 0)

post_fun_vj = jax.jit(jax.vmap(post_fun, (0,0,0,None,None)), static_argnames=['static_settings'])
data = post_fun_vj(x_int, jnp.arange(n_int), local_dofs, settings, static_settings)

x_vis = x_int + jnp.transpose(jnp.asarray([data[:,0],data[:,1]]))

# Plotting with pyvista
plotter.pv_plot(x_vis, data[:,0], export_vtk=False, show=True, data_on_z=False)
plotter.pv_plot(x_vis, data[:,1], export_vtk=False, show=True, data_on_z=False)
plotter.pv_plot(x_vis, data[:,2], export_vtk=False, show=True, data_on_z=False)
plotter.pv_plot(x_vis, data[:,3], export_vtk=False, show=True, data_on_z=False)
plotter.pv_plot(x_vis, data[:,4], export_vtk=False, show=True, data_on_z=False)
plotter.pv_plot(x_vis, data[:,5], export_vtk=False, show=True, data_on_z=False)
