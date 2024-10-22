Assembler
=========

.. automodule:: autopdex.assembler
    :no-index:

General assembling functions
----------------------------
.. autosummary::
   :toctree: _autosummary

   integrate_functional
   assemble_residual
   assemble_tangent_diagonal
   assemble_tangent

Dense assembling
----------------
.. autosummary::
   :toctree: _autosummary

   dense_integrate_functional
   dense_assemble_residual
   dense_assemble_tangent

Sparse assembling
-----------------
.. autosummary::
   :toctree: _autosummary

   sparse_integrate_functional
   sparse_assemble_residual
   sparse_assemble_tangent_diagonal
   sparse_assemble_tangent

Assembling for user potentials
------------------------------
.. autosummary::
   :toctree: _autosummary

   user_potential_integrate_functional
   user_potential_assemble_residual
   user_potential_assemble_tangent_diagonal
   user_potential_assemble_tangent

Assembling for user residuals
-----------------------------
.. autosummary::
   :toctree: _autosummary

   user_residual_assemble_residual
   user_residual_assemble_tangent_diagonal
   user_residual_assemble_tangent

Assembling for user elements
----------------------------
.. autosummary::
   :toctree: _autosummary

   user_element_assemble_residual
   user_element_assemble_tangent_diagonal
   user_element_assemble_tangent

