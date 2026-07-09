# autopdex/sim_state/__init__.py
"""AutoPDEx high-level interface (HLI).

This subpackage provides the orchestrator class :class:`SimState`. The
implementation is split across several modules (``mesh_topology``,
``mesh_info``, ``dofs``, ``projection``, ``state``); externally, only
``SimState`` is re-exported, so ``from autopdex.sim_state import SimState``
remains valid unchanged.
"""

import jax

from .state import SimState

# JAX pytree registration: exactly once when importing the subpackage.
jax.tree_util.register_pytree_node(
    SimState, SimState._tree_flatten, SimState._tree_unflatten
)

__all__ = ["SimState"]
