Variational_schemes
===================

.. automodule:: autopdex.variational_schemes
    :no-index:

Let's have a short description of the variational schemes in this setting.
The goal is to solve a partial differential equation (PDE), written in its strong form as

.. math::

   \mathbf{r}(\mathbf{u}) = \mathbf{0}

where :math:`\mathbf{u}` is the primary field in the domain :math:`\Omega`, together with some Dirichlet (and Neumann) boundary conditions.

To find an approximate solution, a classical approach is to define a linear combination of shape functions :math:`N_I(\mathbf{X})` and 
unknowns :math:`\mathbf{u}_I` :

.. math::

   \mathbf{u}_h = \sum_{I}^{n_{\mathrm{nod}}} N_I(\mathbf{X}) \mathbf{u}_I

where the shape functions typically have local support to enable efficient solution methods (i.e., sparse tangent matrices). 
All unknowns at the nodes are gathered in a vector :math:`\mathbf{d}`, representing the degrees of freedom (DOFs). 
The objective of the variational method is to find the combination of DOFs that best satisfies the PDE and the boundary conditions.
The accuracy is measured using error norms, typically :math:`L^2` or :math:`H^1` norms, corresponding e.g. to displacement error 
and energy error, respectively. Depending on the variational method, the error in the PDE and boundary conditions may be balanced differently, 
or the boundary conditions may need to be embedded in the shape functions as solution_structures.

Two main types of variational methods are: Galerkin (strong/weak) and least squares.

1. **Least Squares Variational Method**:
   The goal is to minimize the squared residual over the domain :math:`\Omega`

   .. math::
      \Pi_{\mathrm{LS}} = \int_{\Omega} \frac{\mathbf{r}_h(\mathbf{d})^2}{2} \, \mathrm{d}\Omega,

   possibly with additional contributions for taking into account boundary conditions.
   The optimal DOFs (measured in the PDE norm) are found by minimizing the functional:

   .. math::
      \mathbf{d}_{\mathrm{LS}} = \operatorname{argmin}_{\mathbf{d}} \Pi_{\mathrm{LS}}

   This leads to a system of equations of which one has to find the root of -- the residual

   .. math::
      \mathbf{R}_{\mathrm{LS}} := \frac{\partial \Pi_{\mathrm{LS}}}{\partial \mathbf{d}} = \mathbf{0}.

   If we consider, for instance, a second order PDE, the shape functions have to be able to reproduce second order polynomials and 
   should be globally :math:`C^1` continuous. In order to ensure optimality also in the :math:`L^2` and :math:`H^1` norms, the least square functional
   has to be norm-equivalent to them.

2. **Galerkin Method (strong/weak form)**:
   In the Galerkin method, the DOFs are searched in a way, that the PDE is fulfilled in a weighted average sense, 
   such that the residual is orthogonal to a test space. Let's call it the strong form Galerkin method, 
   in case we solve for the DOFs by making the following functional stationary with respect to the test function DOFs:
   
   .. math::
      \delta\Pi_{\mathrm{G}} = \int_{\Omega} \mathbf{r}_h \cdot \mathbf{v} \, \mathrm{d}\Omega

   where the test function is typically:

   .. math::
      \mathbf{v} = \sum_{I}^{n_{\mathrm{nod}}} N_I \delta \mathbf{u}_I.

   In comparison to the least square method, the solution space may need to satisfy Ladyzhenskaya–Babuška–Brezzi 
   conditions in case of multi-field problems.

   By doing integration by parts, the continuity requirements of the solution space can be weakend, 
   such that e.g. piecewise linear shape functions can be used for an approximate solution of a problem with a second order PDE.
   Further, also Neumann boundary conditions can be imposed weakly and do not have to be built into the solution space.
   As a drawback, the integration by parts has to be accurate also in its discrete form, leading to so-called integration constraints
   that may need special care when using non-piecewise-polynomial shape functions. 
   The DOFs of the approximate solution can be found as the stationary point of the functional:

   .. math::
      \mathbf{d}_{\mathrm{G}} = \underset{\delta \mathbf{d}}{\operatorname{argstat}} \, \delta\Pi_{\mathrm{G}}

   This leads to the following set of equations

   .. math::
      \mathbf{R}_{\mathrm{G}} := \frac{\partial \delta\Pi_{\mathrm{G}}}{\partial \delta \mathbf{d}} = \mathbf{0}

The residual :math:`\mathbf{R}_{G/LS}` can be efficiently derived from a potential or pseudo-potential using backward mode
automatic differentiation. For sufficiently smooth and convex problems, the Newton-Raphson method, combined with adaptive 
load-stepping, is often one of the most efficient methods for solving the resulting system of equations. 
The necessary tangent matrix can also be derived using automatic differentiation, where exploiting the sparsity of 
the tangent matrix significantly reduces computational time.



Variational Schemes
-------------------

.. autosummary::
   :toctree: _autosummary

   least_square_pde_loss
   weak_form_galerkin
   strong_form_galerkin
   least_square_function_approximation

Distributive functions
----------------------

.. autosummary::
   :toctree: _autosummary

   functional_at_int_point
   direct_residual_at_int_point
   residual_from_deriv_at_int_point