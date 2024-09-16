Examplary input files
======================

Here is a list of examples demonstrating the functionality of AutoPDEx.



Example 1: Cook's membrane (Quadrilaterals in Hyperelasticity)
--------------------------------------------------------------

**Link:** `quadrilaterals_autopdex_and_acegen.py <../../../examples/cooks_membrane/quadrilaterals_autopdex_and_acegen.py>`_

.. automodule:: examples.cooks_membrane.quadrilaterals_autopdex_and_acegen
    :no-index:

**Figure:** Here is a picture generated with ParaView from the output file of the above code that shows the initial configuration as a wireframe and the deformed configuration with colored horizontal displacements.

.. image:: _static/cooks_membrane.png
   :align: center
   :height: 400px

**Further examples, based on the Cook's membrane include:**

   - `triangles_h_refinement.py <../../../examples/cooks_membrane/triangles_h_refinement.py>`_: Simplex-shaped elements based on least squares, integrated in the initial configuration.
   - `non-homogeneous_dirichlet_conditions.py <../../../examples/cooks_membrane/non-homogeneous_dirichlet_conditions.py>`_: Nodal imposition of non-homogeneous Dirichlet conditions.
   - `quadrilaterals_p_refinement.py <../../../examples/cooks_membrane/quadrilaterals_p_refinement.py>`_: A p-refinement with isoparametric quadrilateral elements.
   - `least_square_fem_rvachev_structure.py <../../../examples/cooks_membrane/least_square_fem_rvachev_structure.py>`_: First order system least square Finite Elements as a Rvachev solution structure.
   - `moving_least_squares_least_square_fos.py <../../../examples/cooks_membrane/moving_least_squares_least_square_fos.py>`_: Moving Least Square solution structure based on a first order system.



Example 2: Transient heat conduction (forward- and backward-Euler)
------------------------------------------------------------------

**Link:** `maze_forward_euler.py <../../../examples/heat_conduction/maze_forward_euler.py>`_

.. automodule:: examples.heat_conduction.maze_forward_euler
    :no-index:

**Figure:** The image shows a snapshot of the temperature distribution within the maze visualized with ParaView

.. image:: _static/maze_t64.png
   :align: center
   :height: 400px

**Further examples related to transient heat transfer:**

   - `maze_backward_euler.py <../../../examples/heat_conduction/maze_backward_euler.py>`_: Same as above, but with implicit backward Euler time discretization.
   - `space_time_fos_dirichlet_neumann_robin.py <../../../examples/heat_conduction/space_time_fos_dirichlet_neumann_robin.py>`_: Space-time solution of transient heat conduction as first order system with combined Dirichlet, Neumann and Robin conditions.
   - `space_time_different_solvers.py <../../../examples/heat_conduction/space_time_different_solvers.py>`_: Comparison of different solver backends.



Example 3: Wave equation (least square method of first order system)
--------------------------------------------------------------------

**Link:** `wave_equation.py <../../../examples/miscellaneous/wave_equation.py>`_

.. automodule:: examples.miscellaneous.wave_equation
    :no-index:

**Figure:** The gauss-point quantities for the field (left), its spatial derivative (middle) and its temporal rate (right) are depicted. One can clearly see the effects of wave propagation, superposition and reflection at the free and clamped end.

.. image:: _static/wave_equation.png
   :align: center
   :height: 400px



Example 4: Steady-state Navier-Stokes equations (adaptive load stepping and nonlinear minimization)
---------------------------------------------------------------------------------------------------

**Link:** `circle_in_channel.py <../../../examples/navier_stokes/circle_in_channel.py>`_

.. automodule:: examples.navier_stokes.circle_in_channel
    :no-index:

**Figure:** Steady-state flow field of a Moving Least Squares Galerkin solution.

.. image:: _static/navier_stokes.png
   :align: center
   :height: 400px

**Related examples:**

   - `lid_driven_cavity.py <../../../examples/navier_stokes/lid_driven_cavity.py>`_: Comparison of Galerkin and least square variational schemes.



Example 5: Implicit differentiation (uncertainty estimation)
------------------------------------------------------------

**Link:** `uncertainty_estimation_hyperelastic.py <../../../examples/cooks_membrane/uncertainty_estimation_hyperelastic.py>`_

.. automodule:: examples.cooks_membrane.uncertainty_estimation_hyperelastic
    :no-index:

**Figure:** The three wireframes represent the scaled standard deviation of the displacement field for three Monte Carlo simulations. With 10,000 samples, the Monte Carlo solution aligns visually with the black nodes of the solution calculated using the Taylor expansion. The coloring represents the sensitivity of the vertical displacement field with respect to Poisson's ratio.

.. image:: _static/uncertainty_estimation.png
   :align: center
   :height: 400px



Example 6: Further examples
---------------------------

**List of further examples:**

   - `mls_rfm.py <../../../examples/laplace/mls_rfm.py>`_: Dirichlet problem of the Laplace equation in 2d: [0,1]x[0,1] with cutout circle (midpoint=[0.5,0.5], radius=0.25). Demonstration of quasi-random sampling in regions defined by signed distance functions.
   - `pinn_rfm.py <../../../examples/laplace/pinn_rfm.py>`_: Neural network as user-defined solution space, boundary conditions through R functions.
   - `fos_coupled_neumann_boundary.py <../../../examples/laplace/fos_coupled_neumann_boundary.py>`_: First order least square system of Laplace equation with coupled Neumann boundary conditions on circular cutout.
   - `poisson.py <../../../examples/miscellaneous/poisson.py>`_: Demonstration of compile time reduction by manually pre-compiling solution structures.
   - `transport.py <../../../examples/miscellaneous/transport.py>`_: Transport equation as moving least square solution structure with least square variational method.
   - `burgers_equation.py <../../../examples/miscellaneous/burgers_equation.py>`_: Solution of the Burgers equation using moving least squares.
   - `geometry_examples.py <../../../examples/miscellaneous/geometry_examples.py>`_: Interactive visualization of smooth distance functions.

**Figure:** Exemplary result of solving the Laplace equation with a Moving Least Square solution structure on a square-shaped domain with cutout (`mls_rfm.py <../../../examples/laplace/mls_rfm.py>`_).

.. image:: _static/laplace.png
   :align: center
   :height: 400px
