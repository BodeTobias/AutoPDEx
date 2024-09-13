---
title: 'AutoPDEx: An Automized Partial Differential Equation solver based on JAX'
tags:
  - JAX, Partial Differential Equations (PDEs), Automatic Differentiation, Sensitivity Analysis, Machine Learning
authors:
  - name: Tobias Bode
    orcid: 0000-0001-7941-7658
    affiliation: 1
affiliations:
 - name: Institute of Continuum Mechanics, Leibniz University Hannover, An der Universität 1, 30823 Garbsen, Germany
   index: 1
date: 10 September 2024
bibliography: paper.bib

---

# Summary

`AutoPDEx` is a free and open-source software for solving partial differential equations (PDEs) based on the automatic code transformation capabilities of `JAX` [@jax2018github]. It is designed to provide a modular, flexible, and extendable environment for solving boundary and initial value problems, allowing seamless integration with machine learning algorithms and GPU acceleration through the Accelerated Linear Algebra (XLA) compiler.

At its core, AutoPDEx includes a versatile solver module that supports algorithms such as adaptive load stepping, Newton's method, and nonlinear minimizers. The PDEs to be solved and the chosen variational methods and solution spaces can be specified via user-defined JAX-transformable functions. Pre-built models and ansatz functions are available in the `models` and `spaces` modules. In addition to finite element methods, e.g. mesh-free approaches or neural networks can be used as solution spaces. This flexibility makes AutoPDEx suitable for researchers working at the intersection of numerical analysis and machine learning.

The `implicit_diff` module provides a wrapper to make the solution methods differentiable through automatic implicit differentiation [@blondel2022efficient]. This allows adaptive load stepping to be used for arbitrary order sensitivity analyses in forward or reverse mode. For solving linear systems, it integrates with high-performance external solvers such as PARDISO [@schenk2004solving] and PETSc [@balay2019petsc]. Below, the solution of some example test cases available in the documentation is depicted.

![](../docs/_static/demos.png)

# Statement of Need

AutoPDEx is specifically designed to leverage automatic differentiation-based modeling, including sensitivity analysis, which is essential for applications such as material parameter identification, topology optimization, uncertainty estimation, and multi-scale analysis [@korelc2016automation]. By building on JAX, it enables the smooth combination of numerical simulations with machine learning models from the JAX ecosystem, such as those provided by Flax [@flax2020github] and Equinox [@kidger2021equinox]. This capability allows researchers to incorporate data-driven approaches into traditional simulation workflows, while its modular structure allows for rapid prototyping and experimentation, fostering innovative developments in these fields. The analysis in AutoPDEx can be used in combination with established tools such as Gmsh [@geuzaine2009gmsh] for mesh generation and PyVista [@sullivan2019pyvista] or ParaView [@ahrens200536] for visualisation, providing a comprehensive solution from model preparation to analysis.

# Acknowledgements

Fixme: phoenix, siiri?

# References