High-level interface
====================

``SimState`` is the high-level interface of AutoPDEx. 
It bundles a finite-element simulation behind a single object:
you declare one spatial and temporal discretization per field, import or generate a mesh,
register the governing equations as compact model functions, and add boundary and
initial conditions. ``SimState`` then builds the global DOF numbering, sets up the
assembly and runs the transient solve. 
The run method is jittable and diffable via implicit differentiation.
Examples for sensitivity analysis applications are under development.


Typical workflow
----------------

Most simulations follow this compact setup sequence:

.. code-block:: python

   sim = SimState(spatial_discretizations_per_field)
   sim.import_mesh(meshio_object)
   sim.add_temporal_discretization(time_integrators)
   sim.add_model("__all__", "weak form", weak_form_fun)
   sim.add_strong_bc(field, on_boundary_fun, value_fun)
   sim.add_weak_bc(field, on_boundary_fun, value_fun)
   sim.set_initial_conditions(value_fun)
   sim.set_postprocessing_policy(postprocessing_policy)
   sim.set_step_size_controller(time_step_controller)
   sim.initialize()
   sim.prepare()
   sim = sim.run(dt0, time_span, num_time_steps) # jax transformable

SimState class
--------------

.. currentmodule:: autopdex

.. autoclass:: autopdex.SimState
   :no-index:

Model context
~~~~~~~~~~~~~

Model functions (weak forms / potentials) registered with ``SimState.add_model(...)`` receive a
``SimState.ModelContext`` argument. It is a data class providing the following attributes:

.. autosummary::
   :toctree: _autosummary

   models.ModelContext

Writing a model function
~~~~~~~~~~~~~~~~~~~~~~~~~~

A model function receives a single :class:`ModelContext` and returns a different
object depending on ``ctx.mode`` which can be ``"weak form"``, ``"potential"``, ``"internal variables"``, or ``"output"``.:

.. code-block:: python

   def model(ctx: SimState.ModelContext):
       phi = ctx.trial_ansatz["phi"]               # trial field callable
       grad_phi = jax.jacfwd(phi)(ctx.x_int, ctx.t)

       if ctx.mode == "output":                    # derived quantities for postprocessing
           return {"my quantity": ...}

       test = ctx.test_ansatz["phi"]               # test field callable (not available in output mode)
       grad_test = jax.jacfwd(test)(ctx.x_int)
       return grad_phi @ grad_test

Setup and execution
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary

   SimState
   SimState.import_mesh
   SimState.add_structured_mesh
   SimState.add_temporal_discretization
   SimState.add_model
   SimState.add_local_subsystem
   SimState.add_strong_bc
   SimState.add_weak_bc
   SimState.set_initial_conditions
   SimState.set_postprocessing_policy
   SimState.set_step_size_controller
   SimState.set_dof_scaling
   SimState.set_root_solver
   SimState.initialize
   SimState.prepare
   SimState.run

Accessing results
~~~~~~~~~~~~~~~~~~

After ``run(...)`` the returned ``SimState`` exposes the solution and the
time-integration statistics as properties. Further transformation of the results 
for sensitivity analysis purposes is under development.

.. autosummary::
   :toctree: _autosummary

   SimState.dofs
   SimState.num_steps
   SimState.num_accepted
   SimState.num_rejected

Spatial discretization
~~~~~~~~~~~~~~~~~~~~~~

The following spatial discretization classes are currently available:

.. autosummary::
   :toctree: _autosummary

   spaces.H1
   spaces.L2
   spaces.InternalVariable

Temporal discretization compatible with SimState
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Current limitations: Time integrators with coupled stages, 
such as Gauss–Legendre Runge–Kutta methods, are not supported for primary fields.
Algebraic constraints don't work with explicit stages.
The time integrators of different fields must be compatible with each other, 
i.e. the stage positions must be the same for all fields.
The multi-step methods (BDF, AdamsMoulton, AdamsBashforth) are only compatible 
with the ConstantStepSizeController.
