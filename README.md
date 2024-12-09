<div align="center">
    <img src="https://github.com/BodeTobias/AutoPDEx/blob/main/docs/_static/logo.png" width="300"/>
</div>

AutoPDEx is a free open source partial differential equation (PDE) solver based on the automatic code transformation capabilities of [JAX](https://github.com/jax-ml/jax).

The idea of the project is to develop a modular and easily extendable environment for the solution of boundary and initial boundary value problems, which provides automatic sensitivity analysis, allows for good integration with machine learning algorithms and can be executed on accelerators such as GPUs.

The documentation with more examples is available [here](https://bodetobias.github.io/AutoPDEx/index.html).

![](https://github.com/BodeTobias/AutoPDEx/blob/main/docs/_static/demos_small.png)

## Installation

To install AutoPDEx, you can use the following command. Note, that it requires python>=3.10. 

```
pip install --upgrade pip
pip install autopdex
```

## Example

This is a short example for solving Poisson's problem with homogeneous Dirichlet conditions on the domain $[0,1]\times[0,1]$ with the source term 

$$b = 20 \left(\sin{\left(10\ \boldsymbol{x}\cdot\boldsymbol{x}\right)} - \cos{\left(10\  \left(\boldsymbol{x} - \boldsymbol{x}_2\right) \cdot \left(\boldsymbol{x} - \boldsymbol{x}_2\right)\right)}\right)$$ 

where $\boldsymbol{x}_2 = (1, 0.5)^T$.

[Download full example](examples/miscellaneous/short_example.py)

First, we import the necessary packages and enable double precision.

```python
import jax
import jax.numpy as jnp
import flax
import meshio

from autopdex import seeder, geometry, solver, utility, models, spaces, mesher
jax.config.update("jax_enable_x64", True)

```

Generate the mesh (or import the node coordinates and connectivity)

```python
pts = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
coords, elems = mesher.structured_mesh((200, 200), pts, 'quad')
node_coordinates = {'phi': coords,}
connectivity = {'phi': elems,}
```


Selection of nodes and degrees of freedom for Dirichlet conditions

```python
sdf = lambda x: geometry.psdf_polygon(x, pts)
dirichlet_nodes = geometry.in_sdfs(node_coordinates['phi'], sdf)
dirichlet_dofs = {'phi': dirichlet_nodes,}
dirichlet_conditions = utility.dict_zeros_like(dirichlet_dofs, dtype=jnp.float64)
```

The variational problem is defined in terms of the potential $\Pi$. The stationarity condition of this potential gives the Poisson equation. The potential is represented by the integral

$$\Pi(\phi) = \int_\Omega \left( \frac{1}{2} \nabla \phi \cdot \nabla \phi - b \phi \right)\ \mathrm{d}\Omega$$

This leads to the following Euler-Lagrange equation (Poisson's equation):

$$- \Delta \phi = b$$

```python
def integrand_fun(x_int, ansatz_fun, settings, static_settings, elem_number, set):
    # Definition of custom functional    
    x = ansatz_fun['physical coor'](x_int)
    phi_fun = ansatz_fun['phi']
    phi = phi_fun(x_int)
    dphi_dx = jax.jacrev(phi_fun)(x_int)
    x_2 = x - jnp.array([1., 0.5])
    b = 20 * (jnp.sin(10 * x @ x) - jnp.cos(10 * x_2 @ x_2))
    return (1/2) * dphi_dx @ dphi_dx - b * phi
```

Set up the finite element, here Q1 elements for the field 'phi'
```python
user_potential = models.mixed_reference_domain_potential(
    integrand_fun,
    {'phi': spaces.fem_iso_line_quad_brick,},
    *seeder.gauss_legendre_nd(dimension = 2, order = 2),
    'phi')
```

Prepare the settings for autopdex
```python
static_settings = flax.core.FrozenDict({
  'assembling mode': ('user potential',),
  'solution structure': ('nodal imposition',),
  'model': (user_potential, ),
  'solver type': 'newton',
  'solver backend': 'scipy',
  'solver': 'lapack',
  'verbose': 1,
})
settings = {
  'connectivity': (connectivity,),
  'dirichlet dofs': dirichlet_dofs,
  'node coordinates': node_coordinates,
  'dirichlet conditions': dirichlet_conditions,
}
```

Compile, assemble and solve linear system
```python
initial_guess = utility.dict_zeros_like(dirichlet_dofs, dtype=jnp.float64)
dofs = solver.solver(initial_guess, settings, static_settings)[0]
```

Write vtk file for visualization with Paraview
```python
meshio.Mesh(
    coords,
    {'quad': elems},
    point_data={
        "phi": dofs['phi'],
    },
).write("./short_example.vtk")
```

<div align="center">
    <img src="https://github.com/BodeTobias/AutoPDEx/blob/main/docs/_static/short_example.png" width="500"/>
</div>


## Contributions

You are warmly invited to contribute to the project. For larger developments, please get in touch beforehand in order to circumvent double work. 

For detailed information on how to contribute, please see our [Contribution Guidelines](https://github.com/BodeTobias/AutoPDEx/blob/main/CONTRIBUTING.md)

## License

AutoPDEx is licensed under the GNU Affero General Public License, Version 3.
