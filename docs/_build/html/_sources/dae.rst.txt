Time stepping
=============

.. automodule:: autopdex.dae

The 'dae' module includes time integration methods (Runge-Kutta and multistep methods) for solving coupled ordinary differential equations. The differential equations must be specified in implicit form. In principle, one can choose different integrators for different fields; however, the number and position of the stages must be compatible, as currently only a fully monolithic solution is implemented. Additionally, differential-algebraic systems can be solved. When algebraic equations are present, only integrators with exclusively implicit stages should be used.

Furthermore, by using the keyword 'call pde', it is possible to solve time-dependent PDEs (currently, only time integrators with a single stage). The mechanism works by endowing the nodal degrees of freedom with discrete values and derivatives as functions of time using the chosen time discretization rules. The spatial discretization can be carried out using the same ansatz (for example, isoparametric finite elements) as in time-independent problems. In the case of backward Euler and spatial FE ansatz functions, this would appear as follows:

:math:`\Theta = \sum_{I=1}^{N} \Theta_I\left(t\right) N_I\left(\boldsymbol{x}\right)`,

where :math:`N_I\left(\boldsymbol{x}\right)` are the FE ansatz functions and :math:`\Theta_I\left(t\right)` represent the nodal degrees of freedom. The given PDE is then solved at the step :math:`t_{n+1}` , where the time derivative of the nodal degrees of freedom is defined by a custom_jvp rule as :math:`\frac{\partial\Theta_I}{\partial t}\vert_{t_{n+1}} = \frac{\Theta_I\left(t_{n+1}\right) - \Theta_I\left(t_n\right)}{\Delta t}`. In the 'call pde' mode, one should currently not use jacrev w.r.t. the time.

*WARNING*: This module is currently under development and lacks detailed documentation. However, some example notebooks are already available.

Central class
-------------

.. autosummary::
  :toctree: _autosummary

  TimeSteppingManager
  TimeSteppingManager.run
  TimeSteppingManagerState

Time integrators
----------------

.. autosummary::
  :toctree: _autosummary

  TimeIntegrator
  BackwardEuler
  ForwardEuler
  Newmark
  AdamsMoulton
  AdamsBashforth
  BackwardDiffFormula
  ExplicitRungeKutta
  DiagonallyImplicitRungeKutta
  Kvaerno
  DormandPrince
  GaussLegendreRungeKutta

Helper functions 
----------------

.. autosummary::
  :toctree: _autosummary

  discrete_value_with_derivatives
  detect_stage_dependencies
  invert_butcher_with_order

Step size controllers
---------------------

.. autosummary::
  :toctree: _autosummary

  StepSizeController
  ConstantStepSizeController
  CSSControllerState
  PIDController
  PIDControllerState
  RootIterationController
  RootIterationControllerState

Data saving policies
--------------------

.. autosummary::
  :toctree: _autosummary

  SavePolicy
  HistoryState
  SaveNothingPolicy
  SaveEquidistantPolicy
  SaveEquidistantHistoryState
  SaveAllPolicy
  SaveAllHistoryState

Rootsolvers compatible with the DAE module
------------------------------------------

.. autosummary::
  :toctree: _autosummary

  RootSolverResult
  newton_solver