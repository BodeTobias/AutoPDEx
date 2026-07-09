[![DOI](https://joss.theoj.org/papers/10.21105/joss.07300/status.svg)](https://doi.org/10.21105/joss.07300)

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

or with optional dependencies:

```
pip install --upgrade pip
pip install autopdex[dev]
```


## Example

This is a short example for solving Poisson's problem with homogeneous Dirichlet conditions on the domain $[0,1]\times[0,1]$ with the source term 

$$b = 20 \left(\sin{\left(10\ \boldsymbol{x}\cdot\boldsymbol{x}\right)} - \cos{\left(10\  \left(\boldsymbol{x} - \boldsymbol{x}_2\right) \cdot \left(\boldsymbol{x} - \boldsymbol{x}_2\right)\right)}\right)$$ 

where $\boldsymbol{x}_2 = (1, 0.5)^T$.

[Full high-level interface example notebook](docs/notebooks/short_example_hli.ipynb)

First, we import the necessary packages and enable double precision.

```python
import jax
import jax.numpy as jnp

from autopdex import SimState, dae, spaces

jax.config.update("jax_enable_x64", True)
```

The weak form is provided as a compact ``SimState`` model. In output mode, the same model can also provide derived quantities for postprocessing.

```python
def poisson_model(ctx: SimState.ModelContext):
    phi = ctx.trial_ansatz["phi"]
    x = ctx.trial_ansatz["physical coor"](ctx.x_int)

    def source_term(x):
        x2 = x - jnp.asarray([1.0, 0.5])
        return 20.0 * (jnp.sin(10.0 * x @ x) - jnp.cos(10.0 * x2 @ x2))

    if ctx.mode == "output":
        return {"source": source_term(x)}

    test = ctx.test_ansatz["phi"]
    grad_phi = jax.jacfwd(phi, 0)(ctx.x_int, ctx.t)
    grad_test = jax.jacfwd(test)(ctx.x_int)

    return grad_phi @ grad_test - source_term(x) * test(x)
```

Set up the simulation object, generate a structured mesh, register the model and impose homogeneous Dirichlet conditions on the outer boundary.

```python
vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
n_elements = (1000, 1000)

sim = SimState({"phi": spaces.H1(order=1, dim=2, field_dimension=1)})
sim.add_structured_mesh(n_elements, vertices, "quad")
sim.add_temporal_discretization({"phi": dae.NoTimeDerivative()})
sim.add_model("__all__", "weak form", poisson_model)

def on_outer_boundary(x):
    return jnp.logical_or(
        jnp.logical_or(jnp.isclose(x[0], 0.0), jnp.isclose(x[0], 1.0)),
        jnp.logical_or(jnp.isclose(x[1], 0.0), jnp.isclose(x[1], 1.0)),
    )

sim.add_strong_bc("phi", on_outer_boundary, lambda x, t: 0.0)
sim.set_postprocessing_policy(dae.SaveAllPolicy(), result_folder_name="./short_example.res")
```

Initialize the solver data and run the stationary problem. The result folder contains VTU/PVD files for ParaView.

```python
sim.initialize(verbose=1)
sim.prepare()
sim = sim.run(dt0=1.0, time_span=1.0, num_time_steps=1)
```

<div align="center">
    <img src="https://github.com/BodeTobias/AutoPDEx/blob/main/docs/_static/short_example.png" width="500"/>
</div>

## Citation

If you found this library useful in academic research, please cite the JOSS paper:

```bibtex
@article{Bode_AutoPDEx_An_Automized_2025,
author = {Bode, Tobias},
doi = {10.21105/joss.07300},
journal = {Journal of Open Source Software},
month = apr,
number = {108},
pages = {7300},
title = {{AutoPDEx: An Automized Partial Differential Equation solver based on JAX}},
url = {https://joss.theoj.org/papers/10.21105/joss.07300},
volume = {10},
year = {2025}
}
```

## Contributions

You are warmly invited to contribute to the project. For larger developments, please get in touch beforehand in order to circumvent double work. 

For detailed information on how to contribute, please see our [Contribution Guidelines](https://github.com/BodeTobias/AutoPDEx/blob/main/CONTRIBUTING.md)

## License

AutoPDEx is licensed under the GNU Affero General Public License, Version 3.
