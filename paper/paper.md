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

`AutoPDEx` is a free and open-source software for solving partial differential equations (PDEs) based on the automatic code transformation capabilities of `JAX` [@jax2018github]. It is designed to provide a modular, flexible and extendable environment for solving boundary and initial value problems, allowing seamless integration with machine learning algorithms and GPU acceleration through the Accelerated Linear Algebra (XLA) compiler.

At its core, AutoPDEx includes a versatile solver module that supports algorithms such as adaptive load stepping, Newton's method, and nonlinear minimizers. The PDEs to be solved and the chosen variational methods and solution spaces can be specified via user-defined JAX-transformable functions. Pre-built models and ansatz functions are available in the `models` and `spaces` modules. In addition to finite element methods, e.g. mesh-free approaches or neural networks can be used as solution spaces. This flexibility makes AutoPDEx suitable for researchers working at the intersection of numerical analysis and machine learning.

The `implicit_diff` module provides a wrapper to make the solution methods differentiable through automatic implicit differentiation [@blondel2022efficient]. This allows adaptive load stepping to be used in combination with arbitrary order sensitivity analyses in forward and reverse mode. For solving linear systems, it integrates with high-performance external solvers as e.g. PARDISO [@schenk2004solving] and PETSc [@balay2019petsc]. Below, the solution of some example test cases available in the documentation is depicted.

![](../docs/_static/demos.png)

# Statement of Need

The efficient and accurate solution of partial differential equations (PDEs) is a central task in many scientific and engineering applications. Python has become a popular platform for PDE solving due to its versatility and the availability of mature libraries such as FEniCS [@alnaes2015fenics] with dolfin-adjoint [@mitusch2019dolfin] for sensitivity analysis. However, many existing tools rely on traditional numerical approaches and require manual implementation of residuals and tangents, or use domain-specific symbolic manipulations.

AutoPDEx uses the code-to-code transformation library JAX to compute local derivatives such as spatial gradients, as well as residual and tangent contributions, through automatic differentiation. By leveraging automatic implicit differentiation, the entire solution procedure can also be differentiated in a unified way. This and the modular structure of AutoPDEx make it attractive for research applications in which having sensitivities is beneficial, as e.g. in material parameter identification, topology optimization, uncertainty estimation, and multi-scale analysis [@korelc2016automation]. By building on JAX, AutoPDEx further enables the smooth combination of numerical simulations with machine learning models from the JAX ecosystem, as for instance those provided by Flax [@flax2020github] and Equinox [@kidger2021equinox]. This capability allows incorporating data-driven approaches into traditional simulation workflows. 

The analysis in AutoPDEx can be used in combination with established tools like Gmsh [@geuzaine2009gmsh] for mesh generation and PyVista [@sullivan2019pyvista] or ParaView [@ahrens200536] for visualization, providing a comprehensive solution from model preparation to analysis.

# Acknowledgements

This research was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) through the project grant TRR 298 (SIIRI, Project No. 426335750), as well as under Germany’s Excellence Strategy within the Cluster of Excellence PhoenixD (EXC 2122, Project ID 390833453).

# References
