Models
======

.. automodule:: autopdex.models
    :no-index:

Linear equations
----------------
.. autosummary::
   :toctree: _autosummary

   transport_equation
   poisson
   poisson_weak
   poisson_fos
   heat_equation
   heat_equation_fos
   d_alembert
   d_alembert_fos
   linear_elasticity
   linear_elasticity_weak
   linear_elasticity_fos
   neumann_weak

Nonlinear equations
-------------------
.. autosummary::
   :toctree: _autosummary

   burgers_equation_inviscid
   hyperelastic_steady_state_fos
   hyperelastic_steady_state_weak
   navier_stokes_incompressible_steady
   navier_stokes_incompressible

Strain energy functions
-----------------------
.. autosummary::
   :toctree: _autosummary

   neo_hooke
   isochoric_neo_hooke
   linear_elastic_strain_energy

User potentials/elements
------------------------
.. autosummary::
   :toctree: _autosummary

   mixed_reference_domain_potential
   mixed_reference_domain_residual
   mixed_reference_surface_potential
   mixed_reference_surface_residual
   isoparametric_domain_integrate_potential
   isoparametric_domain_element_galerkin
   isoparametric_surface_element_galerkin

Time integration procedures
---------------------------
.. autosummary::
   :toctree: _autosummary

   forward_backward_euler_weak
   central_differences

