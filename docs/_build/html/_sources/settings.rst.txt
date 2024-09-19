Settings and static_settings
============================

In many of the functions, both `settings` and `static_settings` are required as inputs. The specific keywords that need to be defined 
within these settings depend on the particular problem being addressed. Some options necessitate the definition of specific keywords 
to ensure proper function.

Below is an overview of the possible keywords for `settings` and `static_settings`. Most of these keywords are optional, and the 
exact ones to use can be determined from the examples provided. If a keyword is not defined, either a default value will be used, 
or an error will be generated. In the case of tuples, each entry stands for one domain.

In the Examples column, there are links to the source code of examples in which the keywords are used. In addition, user-specific 
keywords can be introduced, which can then be called within a user element, for example.

There are basically two ways to define a model. Firstly, by defining the PDE in strong or weak form (depending on the variational scheme) 
and specifying the variational scheme, the solution space and the integration point coordinates and weights. Secondly, by directly 
providing already integrated element contributions to the total potential, total residual or also the tangent matrix. The latter 
option can be used, for example, for elements that are integrated in a reference configuration.

Configuration Options for `static_settings`
-------------------------------------------

.. list-table::
   :widths: 15 12 55 18
   :header-rows: 1

   * - Keyword
     - Type
     - Description
     - Example
   * - 'assembling mode'
     - Tuple (String)
     - **'sparse'**: Expects the model functions to return a weak or strong form of the pde
     
       **'dense'**: Like sparse, but results in a dense tangent matrix

       **'user potential'**: Expects the model functions to return the element contribution of a potential. 
       
       The residual and tangent is then computed automatically.

       **'user residual'**: Expects the model functions to return the element residual. 
       
       The tangent is computed automatically, if needed.

       **'user element'**: Expects the model functions to return the element residual and tangent
     - `transport.py <https://github.com/BodeTobias/AutoPDEx/tree/main/examples/miscellaneous/transport.py>`_
    
       `pinn_rfm.py <https://github.com/BodeTobias/AutoPDEx/tree/main/examples/laplace/pinn_rfm.py>`_

       `quadrilaterals_p_refinement.py <https://github.com/BodeTobias/AutoPDEx/tree/main/examples/cooks_membrane/quadrilaterals_p_refinement.py>`_
   * - 'solution space'

       (for assembling modes 
       
       'sparse' or 'dense')
     - Tuple (String)
     - **'mls'**: moving least squares

       **'fem simplex'**: triangles, tetrahedrons

       **'nodal values'**: see example

       **'user'**: static_settings['user solution space function']
     - `mls_rfm.py <https://github.com/BodeTobias/AutoPDEx/tree/main/examples/laplace/mls_rfm.py>`_

       `pinn_rfm.py <https://github.com/BodeTobias/AutoPDEx/tree/main/examples/laplace/pinn_rfm.py>`_

       `triangles_h_refinement.py <https://github.com/BodeTobias/AutoPDEx/tree/main/examples/cooks_membrane/triangles_h_refinement.py>`_
       
       `maze_forward_euler.py <https://github.com/BodeTobias/AutoPDEx/tree/main/examples/heat_conduction/maze_forward_euler.py>`_
   * - 'shape function mode'

       (for assembling modes 
       
       'sparse' or 'dense')
     - String
     - **'compiled'**: uses a solution structure that is precompiled in preprocessing

       **'direct'**: no manual precompilation; can be considerably slower for complicated solution structures
     - `poisson.py <https://github.com/BodeTobias/AutoPDEx/tree/main/examples/miscellaneous/poisson.py>`_
   * - 'number of fields'
     - Tuple (Integer)
     - **Number of fields for the respective domain**, e.g.:

       1 for models.poisson_weak

       2 for models.linear_elasticity_weak with plain strain

       6 for models.hyperelastic_steady_state_fos with plain strain
     - `poisson.py <https://github.com/BodeTobias/AutoPDEx/tree/main/examples/miscellaneous/poisson.py>`_

       `least_square_fem_rvachev_structure.py <https://github.com/BodeTobias/AutoPDEx/tree/main/examples/cooks_membrane/least_square_fem_rvachev_structure.py>`_

       `non-homogeneous_dirichlet_conditions.py <https://github.com/BodeTobias/AutoPDEx/tree/main/examples/cooks_membrane/non-homogeneous_dirichlet_conditions.py>`_
   * - 'maximal number of neighbors'
     - Tuple (Integer)
     - **Maximum number of neighbors for the respective domain**, e.g.

       6 for second order triangle,

       max_neighbors for moving least squares
     - 
   * - 'variational scheme'

       (for assembling modes 
       
       'sparse' or 'dense')
     - Tuple (String)
     - **'least square pde loss'**:
       
       requires special treatment of Neumann boundaries and higher smoothness;

       first-order systems should be norm-equivalent

       **'strong form galerkin'**:
       
       requires special treatment of Neumann boundaries and higher smoothness;

       may suffer from locking/ need to consider LBB conditions

       **'weak form galerkin'**: 

       integration is subject to variational consistency conditions;

       may suffer from locking/ need to consider LBB conditions
     - `lid_driven_cavity.py <https://github.com/BodeTobias/AutoPDEx/tree/main/examples/navier_stokes/lid_driven_cavity.py>`_
   * - 'model'
     - Tuple (Function)
     - **Jax-transformable functions** that defines:

       strong form of pde, e.g. models.poisson,

       weak form of pde, e.g. models.poisson_weak,

       user-defined potenial, residual or tangent and residual, e.g. models.isoparametric_domain_element_galerkin
     - 
   * - 'solution structure'
     - Tuple (String)
     - **'nodal imposition'**: 
       
       boundary conditions are imposed nodally;
       
       requires static_settings['dirichlet dofs'] and settings['dirichlet conditions']
       
       **'first order set'**:
       
       boundary conditions are imposed using distance functions;
       
       requires static_settings['boundary coefficients', 'boundary conditions', 'psdf']

       **'second order set'**:
       
       like 'first order set', but with higher differentiability im compiled mode

       **'off'**: 
       
       solution_space is forwarded
     - 
   * - 'boundary coefficients'

       (not for solution structures 
       
       'nodal imposition' or 'off')
     - Tuple (Callable)
     - Functions for calculating boundary coefficients
     - `space_time_fos_dirichlet_neumann_robin.py <https://github.com/BodeTobias/AutoPDEx/tree/main/examples/heat_conduction/space_time_fos_dirichlet_neumann_robin.py>`_
   * - 'boundary conditions'

       (not for solution structures 
       
       'nodal imposition' or 'off')
     - Tuple (Callable)
     - Functions for calculating boundary conditions
     - `space_time_fos_dirichlet_neumann_robin.py <https://github.com/BodeTobias/AutoPDEx/tree/main/examples/heat_conduction/space_time_fos_dirichlet_neumann_robin.py>`_
   * - 'psdf'

       (not for solution structures 
       
       'nodal imposition' or 'off')
     - Tuple (Callable)
     - Functions for calculating positive smooth distance functions
     - `space_time_fos_dirichlet_neumann_robin.py <https://github.com/BodeTobias/AutoPDEx/tree/main/examples/heat_conduction/space_time_fos_dirichlet_neumann_robin.py>`_
   * - 'solver type'
     - String
     - **'linear'**, **'newton'**, **'damped newton'**, **'minimize'**
     - 
   * - 'solver backend'
     - String
     - **'jax'**, **'pardiso'**, **'petsc'**, **'scipy'**, **'pyamg'**
     - `space_time_different_solvers.py <https://github.com/BodeTobias/AutoPDEx/tree/main/examples/heat_conduction/space_time_different_solvers.py>`_
   * - 'solver'
     - String
     - depending on solver backend: **'cg'**, **'lu'**, **'qr'**, **'lbfgs'**, **'gradient descent'**, **'bcgs'**
     - `space_time_different_solvers.py <https://github.com/BodeTobias/AutoPDEx/tree/main/examples/heat_conduction/space_time_different_solvers.py>`_
   * - 'type of preconditioner'

       (for solver backend 'jax')
     - String
     - depending on solver backend: **'jacobi'**, **'ilu'**, **'none'**
     - `space_time_different_solvers.py <https://github.com/BodeTobias/AutoPDEx/tree/main/examples/heat_conduction/space_time_different_solvers.py>`_
   * - 'hvp type'

       (for solver backend 'jax')
     - String
     - Type of hessian vector product: **'fwdrev'**, **'revrev'**, **'linearize'**, **'assemble'**
     - `space_time_different_solvers.py <https://github.com/BodeTobias/AutoPDEx/tree/main/examples/heat_conduction/space_time_different_solvers.py>`_
   * - 'verbose'
     - Integer
     - Level of verbosity of output (-1,0,1,2)
     - 
   * - 'dirichlet dofs'

       (for solution structure

       'nodal imposition')
     - Tuple (Bool)
     - Boolean tuple-Mask for selection DOFs for nodal Dirichlet boundary imposition
     - `non-homogeneous_dirichlet_conditions.py <https://github.com/BodeTobias/AutoPDEx/tree/main/examples/cooks_membrane/non-homogeneous_dirichlet_conditions.py>`_
   * - 'connectivity'
     - Tuple (jnp.ndarray)
     - List of node numbers for each element/neighborhood in a domain
     - 

Configuration Options for `settings`
------------------------------------

.. list-table:: 
   :widths: 18 12 70
   :header-rows: 1

   * - Keyword
     - Type
     - Description
   * - 'node coordinates'
     - jnp.ndarray
     - Coordinates of the nodes
   * - 'dirichlet conditions'

       (for solution structure

       'nodal imposition')
     - jnp.ndarray
     - Dirichlet boundary conditions that shall be nodally imposed
   * - 'integration coordinates'

       (for assembling modes 
       
       'sparse' or 'dense')
     - Tuple (jnp.ndarray)
     - Integration point coordinates for each domain (not necessarry for user elements)
   * - 'integration weights'

       (for assembling modes 
       
       'sparse' or 'dense')
     - Tuple (jnp.ndarray)
     - Weights of integration points for each domain (not necessarry for user elements)
   * - 'compiled bc', 

       'compiled shape functions',
       
       'compiled projection'

       (for shape function mode 
       
       'compiled')
     - Tuple (Function)
     - Precomputed values including spatial derivatives for construction of compiled solution structure
   * - e.g. 'support radius'
     - Tuple (Float)
     - Support radius for moving least square shape functions
   * - e.g. 'beta'
     - Tuple (Float)
     - Parameter for moving least square shape functions
   * - e.g. 'load multiplier'
     - Float
     - a multiplier that can be used within solver.adaptive_load_stepping
   * - e.g. 'youngs modulus' 
   
       or 'poisson ratio'
     - Float
     - differentiable material parameters
