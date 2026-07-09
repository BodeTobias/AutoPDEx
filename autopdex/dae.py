# dae.py
# Copyright (C) 2025 Tobias Bode
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

"""Module for solving differential algebraic systems and transient PDEs."""

# TODO: make sure, the integrator supports changing the step size if neccessarry! e.g. adams moulton: root iteration controler not possible...
# TODO: translate all comments to english
# TODO: staggered policies, explicit diagonal modes
# TODO: add information about algebraic equations and add different treatements, e.g. projection for explicit modes
# TODO: add accuracy tests for each integrator

from os import environ
from typing import Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import ceil, isclose, sqrt

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import jax
import jax.numpy as jnp
from jax import tree_util, custom_jvp
from jax.experimental import sparse
from flax.core import FrozenDict

from autopdex.utility import dict_flatten, reshape_as, jit_with_docstring, dict_zeros_like
from autopdex import solver, assembler, implicit_diff



## helper functions

@custom_jvp
def _no_derivative(t, q):
  return q

@_no_derivative.defjvp
def _no_derivative_jvp(primals, tangents):
  raise ValueError("\n\nYour chosen integrator does not support derivatives up to the order you are using!\
    Consider using a different integrator or convert your system to a system of lower order.\n")

@custom_jvp
def discrete_value_with_derivatives(t, q, q_derivs):
  """
  Evaluate the discrete state value with custom derivative propagation.

  This function returns the discrete state value `q`, but it is equipped with a custom Jacobian-vector product (JVP)
  rule to correctly propagate derivative information through discrete operations. This custom derivative rule is designed
  to support higher-order derivative calculations by processing a sequence of derivative values provided in `q_derivs`.

  In the absence of derivative information (i.e. when `q_derivs` is empty), the derivative is taken to be `q_dot`.
  When derivative information is available, the first derivative in `q_derivs` is used recursively along with the time
  derivative `t_dot` to compute the overall derivative contribution.

  It is used to construct a differentiable q_fun based on a value and its derivative defined by an integration rule, e.g.:

    .. code-block:: python
    
      def diffable_q_fun(t): # q_ts is a tuple of (q, q_t, q_tt, ...) coming from the integrator
        return {key: discrete_value_with_derivatives(t, q_ts[key][0], q_ts[key][1:]) for key in template.keys()}
  
  Args:
    t: Scalar representing the time variable.
    q: The discrete state value.
    q_derivs: A sequence (e.g., list or tuple) of derivative values corresponding to `q`. The first element represents
              the first derivative, with subsequent elements (if any) representing higher-order derivatives.
  
  Returns:
    The discrete state value `q`. The custom derivative rule ensures that during differentiation the returned derivative follows the form:
    
      - If no derivative information is provided (`q_derivs` is empty): returns `q_dot`.
      - Otherwise: returns `q_dot + (discrete_value_with_derivatives(t, first_deriv, remaining_derivs) * t_dot)`,
        where `first_deriv` is the first element of `q_derivs` and `remaining_derivs` contains any higher-order derivatives.
  """
  return q

@discrete_value_with_derivatives.defjvp
def discrete_value_with_derivatives_jvp(primals, tangents):
  (t, q, q_derivs) = primals
  (t_dot, q_dot, q_derivs_dot) = tangents

  if len(q_derivs) == 0:
    # return _no_derivative(t, q), q_dot # Problematic with multiple fields?!
    return q, q_dot
  else:
    first_derivs = q_derivs[0]
    remaining_derivs = q_derivs[1:]
    return discrete_value_with_derivatives(
        t, q, q_derivs), q_dot + discrete_value_with_derivatives(t, first_derivs, remaining_derivs) * t_dot


def _derivative_tuple(q_der, q, num_derivs):
  if isinstance(q_der, tuple):
    return q_der
  if num_derivs == 0:
    return ()
  if q_der.shape == q.shape:
    return (q_der,)
  if q_der.shape == (num_derivs, *q.shape):
    return tuple(q_der[i] for i in range(num_derivs))
  raise ValueError(f"Derivative shape {q_der.shape} is incompatible with state shape {q.shape}.")


def _canonical_derivatives(q_der, q, num_derivs):
  derivs = _derivative_tuple(q_der, q, num_derivs)
  if len(derivs) == 0:
    return jnp.zeros((0, *q.shape), dtype=getattr(q_der, "dtype", q.dtype))
  return jnp.stack(derivs)


def _stage_tuple(q_ts, stage, stage_shape):
  return tuple(val[stage] if val.shape == stage_shape else val for val in q_ts)


## butcher tableau inversion

def detect_stage_dependencies(A):
  """
  Detects coupled structures (strongly connected components) in the Butcher matrix A and identifies explicit stages.

  Parameters:
    A (ndarray): Butcher matrix of stage coefficients (s x s).

  Returns:
    stage_blocks (list): A list of lists containing the indices of coupled stages.
    explicit_stages (list): A list of indices corresponding to explicit stages.
    block_dependencies (dict): A dictionary mapping each block to its dependent blocks.
  """
  s = A.shape[0]
  dependency_matrix = (A != 0).astype(int)

  # Find SCCs
  graph = csr_matrix(dependency_matrix)
  n_components, labels = connected_components(csgraph=graph, directed=True, connection='strong')

  # Group stages by their SCC labels
  stage_blocks = [[] for _ in range(n_components)]
  for i in range(s):
    stage_blocks[labels[i]].append(i)

  # Explicit stages: a_ii = 0
  explicit_stages = []
  for block in stage_blocks:
    if all(A[i, i] == 0 for i in block):  # All diagonal elements in this block are 0
      explicit_stages.extend(block)

  # Set up block dependencies
  block_dependencies = {i: set() for i in range(n_components)}
  for i in range(s):
    for j in range(s):
      if dependency_matrix[i, j]:
        block_i = labels[i]
        block_j = labels[j]
        if block_i != block_j:
          block_dependencies[block_i].add(block_j)

  return stage_blocks, explicit_stages, block_dependencies

def invert_butcher_with_order(A):
  """
  Computes the blockwise linear mapping matrix ``A_`` that maps U to U_dot without inter-block coupling,
  and determines the execution order of the blocks.

  Parameters:
    ``A`` (ndarray): Butcher matrix of stage coefficients (s x s).

  Returns:
    ``A_`` (ndarray): Matrix mapping U to U_dot (s x s).
    execution_order (list): ``A`` list specifying the order of operations (blocks or explicit stage indices).
  """
  s = A.shape[0]
  A_ = np.zeros_like(A)  # Initialize resulting matrix

  # Detect explicit stages and coupled blocks
  stage_blocks, explicit_stages, block_dependencies = detect_stage_dependencies(A)

  for block in stage_blocks:
    if all(i in explicit_stages for i in block):  # Explicit stages
      continue
    else:
      # Coupled blocks
      A_block = A[np.ix_(block, block)]
      A_block_inv = np.linalg.inv(A_block)

      for i, row_idx in enumerate(block):
        for j, col_idx in enumerate(block):
          A_[row_idx, col_idx] = A_block_inv[i, j]

  # Explicit stages: set diagonal to 1 and invert the sign of the lower triangle
  for i in explicit_stages:
    for j in range(i):
      A_[i, j] = -A[i, j]
    A_[i, i] = 1

  # Determine ordering of the blocks
  execution_order = []
  resolved = set()

  def resolve_block(block_idx):
    if block_idx in resolved:
      return
    for dep in block_dependencies[block_idx]:
      resolve_block(dep)
    resolved.add(block_idx)
    block = stage_blocks[block_idx]
    if all(i in explicit_stages for i in block):
      for i in block:
        execution_order.append((i, "explicit"))
    else:
      execution_order.append((block, "implicit"))

  for block_idx in range(len(stage_blocks)):
    resolve_block(block_idx)

  return A_, execution_order


## integrator class

class TimeIntegrator(ABC):
  """
  Base class for time integrators.
  """

  def __init__(self,
               name,
               value_and_derivatives,
               update,
               stage_list,
               stage_types,
               stage_positions,
               num_steps=1,
               num_derivs=1,
               num_stages=1):
    """
    Initializes the time integrator.

    Parameters:
      name (str): The name of the method.
      value_and_derivatives (callable): Function to compute state values and their derivatives.
      update (callable): Function that updates the state based on stage results.
      stage_list (ndarray): Array containing the indices or order of stages.
      stage_types (tuple): Tuple indicating the type of each stage ('explicit' or 'implicit').
      stage_positions (ndarray): Array of stage positions (e.g., Butcher nodes).
      num_steps (int): Number of previous steps (for multi-step methods).
      num_derivs (int): Highest derivative order that is supported.
      num_stages (int): Number of stages (e.g., in Runge–Kutta methods).
    """
    self.name = name
    self.value_and_derivatives = value_and_derivatives
    self.update = update
    self.stage_list = stage_list
    self.stage_types = stage_types
    self.stage_positions = stage_positions
    self.num_steps = num_steps
    self.num_derivs = num_derivs
    self.num_stages = num_stages

  def _update(self, q_stages, q_n, q_t_n, dt):
    """
    Updating rule after solving the step.

    This method must be implemented by concrete integrator classes.

    Parameters:
      q_stages: Results from the solve, stage results
      q_n: Values of last time steps
      q_t_n: Derivatives of last time steps
      dt: Step size

    Returns:
      A tuple containing the updated state and derivative.
    """
    raise NotImplementedError

  def _rule(self, q, q_n, q_t_n, dt):
    """
    Computes the function value and temporal derivative for an integrator, e.g. q, (q-q_n[0])/dt for backward Euler.

    Parameters:
      q: Values of the stages that are to be determined.
      q_n: Values of last time steps. q_n[0] is of time n, q_n[1] of time n-1, etc.
      q_t_n: Derivatives at last time steps. q_t_n[i, j] is the j+1-th derivative of time n-i.
      dt: Time step size.

    Returns:
      State values and derivatives for the stages or for the next time step.
    """
    raise NotImplementedError

  def _error_estimate(self, q, q_n, q_t_n, dt):
    """
    Computes an error estimate for the integrator. Similar to _update. Check e.g. Kvaerno.
    """
    return None


## specific integrator classes

class NoTimeDerivative(TimeIntegrator):
  """
  No time derivative, i.e., static case.
  """

  def __init__(self):
    super().__init__("no_time_derivative",
                     self._rule,
                     self._update,
                     jnp.asarray([[0]]), ('implicit',),
                     jnp.array([1.]),
                     num_steps=1,
                     num_derivs=0,
                     num_stages=1)
    self.order = 1000

  def _update(self, q_stages, q_n, q_t_n, dt):
    q_n1 = q_stages[0]
    return q_n1, jnp.zeros((0, ) + q_n1.shape)  # No derivatives

  def _rule(self, q, q_n, q_t_n, dt):
    return q, jnp.zeros_like(q)  # No derivatives

class BackwardEuler(TimeIntegrator):
  """
  Backward Euler method.
  
  Accuracy: 1st order.
  Stability: L-stable.
  Number of steps: 1.
  Number of stages: 1, implicit.
  Number of derivatives: 1.
  """

  def __init__(self):
    super().__init__("backward_euler",
                     self._rule,
                     self._update,
                     jnp.asarray([[0]]), ('implicit',),
                     jnp.array([1.]),
                     num_steps=1,
                     num_derivs=1,
                     num_stages=1)
    self.butcher_b = jnp.array([1.])
    self.order = 1

  def _update(self, q_stages, q_n, q_t_n, dt):
    q_n1 = q_stages[0]
    q_t_n1 = self.value_and_derivatives(q_n1, q_n, q_t_n, dt)[1:]
    return q_n1, q_t_n1

  def _rule(self, q, q_n, q_t_n, dt):
    q_t = (q - q_n[0]) / dt
    return q, q_t

class ForwardEuler(TimeIntegrator):
  """
  Forward Euler method.
  
  Accuracy: 1st order.
  Stability: instable for stiff problems.
  Number of steps: 1.
  Number of stages: 1, explicit.
  Number of derivatives: 1.
  """

  def __init__(self):
    super().__init__("forward_euler",
                     self._rule,
                     self._update,
                     jnp.asarray([[0]]), ('explicit',),
                     jnp.array([0.]),
                     num_steps=1,
                     num_derivs=1,
                     num_stages=1)
    self.butcher_b = jnp.array([1.])
    self.order = 1

  def _update(self, q_stages, q_n, q_t_n, dt):
    q_n1 = q_stages[0]
    q_t_n1 = self.value_and_derivatives(q_n1, q_n, q_t_n, dt)[1:]
    return q_n1, q_t_n1

  def _rule(self, q, q_n, q_t_n, dt):
    q_t = (q - q_n[0]) / dt
    return q_n[0], q_t

class Newmark(TimeIntegrator):
  """Newmark-beta method.
  
  Args: 
    gamma (float): Newmark parameter.
    beta (float): Newmark parameter.

  Explicit central differences:
    gamma = 0.5
    beta = 0
  
  Average constant acceleration (middle point rule, unconditional stable):
    gamma = 0.5
    beta = 0.25

  Number of steps: 1.
  Number of stages: 1, explicit or implicit.
  Number of derivatives: 2.      
  """

  def __init__(self, gamma=0.5, beta=0.25):
    if isclose(beta, 0.):
      super().__init__("newmark",
                       self._rule,
                       self._update,
                       jnp.asarray([[0]]), ('explicit',),
                       jnp.array([0.]),
                       num_steps=1,
                       num_derivs=2,
                       num_stages=1)
    else:
      super().__init__("newmark",
                       self._rule,
                       self._update,
                       jnp.asarray([[0]]), ('implicit',),
                       jnp.array([1.]),
                       num_steps=1,
                       num_derivs=2,
                       num_stages=1)
    self.gamma = gamma
    self.beta = beta
    self.butcher_b = jnp.array([1.])
    self.order = 2

  def _update(self, q_stages, q_n, q_t_n, dt):
    q_n1 = q_stages[0]
    q_t_n1 = self.value_and_derivatives(q_n1, q_n, q_t_n, dt)[1:]
    return q_n1, q_t_n1

  def _rule(self, q, q_n, q_t_n, dt):
    gamma = self.gamma
    beta = self.beta

    dq = q - q_n[0]
    if isclose(beta, 0.):
      # Central differences
      q_t = (q - q_n[0]) / dt
      q_tt = (q - q_n[0] - q_t_n[0, 0] * dt) / (dt**2)
      return jnp.asarray([q_n[0]]), q_t, q_tt

    else:
      v_n = q_t_n[0, 0]
      a_n = q_t_n[0, 1]
      q_tt = (dq / dt**2 - v_n / dt - a_n * (1 / 2 - beta)) / beta
      q_t = v_n + dt * ((1 - gamma) * a_n + gamma * q_tt)

      return q, q_t, q_tt

class AdamsMoulton(TimeIntegrator):
  """Adams-Moulton method.
  
  Args:
    num_steps (int): Number of previous steps (1 to 6). 

  Number of stages: 1, implicit.
  """

  def __init__(self, num_steps):
    super().__init__("adams_moulton",
                     self._rule,
                     self._update,
                     jnp.asarray([[0]]), ('implicit',),
                     jnp.array([1.]),
                     num_steps=num_steps,
                     num_derivs=1,
                     num_stages=1)
    self.num_steps = num_steps
    self.butcher_b = jnp.array([1.])
    self.order = num_steps + 1

  def _update(self, q_stages, q_n, q_t_n, dt):
    q_n1 = q_stages[0]
    q_t_n1 = self.value_and_derivatives(q_n1, q_n, q_t_n, dt)[1:]
    return q_n1, q_t_n1

  def _rule(self, q, q_n, q_t_n, dt):
    num_steps = self.num_steps

    # Adams-Moulton coefficients
    adams_moulton_coeffs = {
        1:
            jnp.array([1 / 2, 1 / 2]),
        2:
            jnp.array([5 / 12, 8 / 12, -1 / 12]),
        3:
            jnp.array([9 / 24, 19 / 24, -5 / 24, 1 / 24]),
        4:
            jnp.array([251 / 720, 646 / 720, -264 / 720, 106 / 720, -19 / 720]),
        5:
            jnp.array([475 / 1440, 1427 / 1440, -798 / 1440, 482 / 1440, -173 / 1440, 27 / 1440]),
        6:
            jnp.array([
                19087 / 60480, 65112 / 60480, -46461 / 60480, 37504 / 60480, -20211 / 60480, 6312 / 60480, -863 / 60480
            ]),
    }

    # Ensure num_steps is supported
    if num_steps not in adams_moulton_coeffs:
      raise ValueError(f"num_steps={num_steps} is not supported. Supported: {list(adams_moulton_coeffs.keys())}")

    # Get the coefficients for the specified num_steps
    coeffs = adams_moulton_coeffs[num_steps]
    a_0 = coeffs[0]

    # Compute q_t (implicit derivative) using the Adams-Moulton formula
    q_t = (q - q_n[0]) / dt  # Start with the difference quotient
    q_t = q_t - jnp.einsum("j,j...->...", coeffs[1:], q_t_n[:, 0])  # Subtract weighted previous derivatives
    q_t /= a_0  # Divide by a_0 to solve for q_t

    return q, q_t

class AdamsBashforth(TimeIntegrator):
  """Adams-Bashforth time integrator.
  
  Args:
    num_steps (int): Number of previous steps (1 to 6).

  Number of stages: 1, explicit.
  """

  def __init__(self, num_steps):
    super().__init__("adams_bashforth",
                     self._rule,
                     self._update,
                     jnp.asarray([[0]]), ('explicit',),
                     jnp.array([0.]),
                     num_steps=num_steps,
                     num_derivs=1,
                     num_stages=1)
    self.num_steps = num_steps
    self.butcher_b = jnp.array([1.])
    self.order = num_steps

  def _update(self, q_stages, q_n, q_t_n, dt):
    q_n1 = q_stages[0]
    q_t_n1 = self.value_and_derivatives(q_n1, q_n, q_t_n, dt)[1:]
    return q_n1, q_t_n1

  def _rule(self, q, q_n, q_t_n, dt):
    num_steps = self.num_steps
    # Adams-Bashforth coefficients for different step numbers
    adams_bashforth_coeffs = {
        0: jnp.array([1]),
        1: jnp.array([3 / 2, -1 / 2]),
        2: jnp.array([23 / 12, -16 / 12, 5 / 12]),
        3: jnp.array([55 / 24, -59 / 24, 37 / 24, -9 / 24]),
        4: jnp.array([1901 / 720, -2774 / 720, 2616 / 720, -1274 / 720, 251 / 720]),
        5: jnp.array([4277 / 1440, -7923 / 1440, 9982 / 1440, -7298 / 1440, 2877 / 1440, -475 / 1440]),
    }

    # Ensure num_steps is supported
    if num_steps - 1 not in adams_bashforth_coeffs:
      raise ValueError(f"num_steps={num_steps} is not supported. Supported number of steps: 1 to 6.")

    # Get the coefficients for the specified num_steps
    coeffs = adams_bashforth_coeffs[num_steps - 1]
    a_0 = coeffs[0]

    # Compute q_t for the previous step using the Adams-Bashforth formula
    q_t = (q - q_n[0]) / dt  # Start with the difference quotient
    q_t = q_t - jnp.einsum("j,j...->...", coeffs[1:], q_t_n[1:, 0])  # Subtract weighted previous derivatives
    q_t /= a_0  # Divide by a_0

    return q_n[0], q_t

class BackwardDiffFormula(TimeIntegrator):
  """Backward differentiation formula (BDF).
  
  Args:
    num_steps (int): Number of previous steps (1 to 6).
    
  Number of stages: 1, implicit.
  """

  def __init__(self, num_steps):
    super().__init__("backward_diff_formula",
                     self._rule,
                     self._update,
                     jnp.asarray([[0]]), ('implicit',),
                     jnp.array([1.]),
                     num_steps=num_steps,
                     num_derivs=1,
                     num_stages=1)
    self.num_steps = num_steps
    self.butcher_b = jnp.array([1.])
    self.order = num_steps

  def _update(self, q_stages, q_n, q_t_n, dt):
    q_n1 = q_stages[0]
    q_t_n1 = self.value_and_derivatives(q_n1, q_n, q_t_n, dt)[1:]
    return q_n1, q_t_n1

  def _rule(self, q, q_n, q_t_n, dt):
    num_steps = self.num_steps
    # BDF coefficients for different orders
    bdf_coeffs = {
        1: jnp.array([1, -1]),
        2: jnp.array([3 / 2, -2, 1 / 2]),
        3: jnp.array([11 / 6, -3, 3 / 2, -1 / 3]),
        4: jnp.array([25 / 12, -4, 3, -4 / 3, 1 / 4]),
        5: jnp.array([137 / 60, -5, 5, -10 / 3, 5 / 4, -1 / 5]),
        6: jnp.array([49 / 20, -6, 15 / 2, -20 / 3, 15 / 4, -6 / 5, 1 / 6]),
    }

    # Ensure num_steps is supported
    if num_steps not in bdf_coeffs:
      raise ValueError(
          "Order of BDF method not supported. Supported orders: 1 to 6. From 7 on the BDF method is not stable.")

    # Get the coefficients for the specified order
    coeffs = bdf_coeffs[num_steps]

    # Backward differentiation formula
    q_t = (q * coeffs[0] + jnp.einsum("j,j...->...", coeffs[1:], q_n)) / dt

    return q, q_t

class ExplicitRungeKutta(TimeIntegrator):
  """Explicit Runge-Kutta method.
  
  Args:
    num_stages (int): Number of stages (1, 2, 3, 4, 5, 6, 7, 9, 11).
  """

  def __init__(self, num_stages):
    match num_stages:  # From JC Butcher 2008: Numerical Methods for Ordinary Differential Equations, ISBN: 978-0-470-72335-7
      case 1:  # Forward Euler
        butcher_c = jnp.array([0])
        butcher_b = jnp.array([1])
        butcher_A = jnp.array([[0]])
        self.order = 1
      case 2:  # Heun's method
        butcher_c = jnp.array([0, 1])
        butcher_b = jnp.array([1 / 2, 1 / 2])
        butcher_A = jnp.array([[0, 0], [1, 0]])
        self.order = 2
      case 3:  # Kutta's third-order method
        butcher_c = jnp.array([0, 1 / 2, 1])
        butcher_b = jnp.array([1 / 6, 2 / 3, 1 / 6])
        butcher_A = jnp.array([[0, 0, 0], [1 / 2, 0, 0], [-1, 2, 0]])
        self.order = 3
      case 4:  # Classic Runge-Kutta method
        butcher_c = jnp.array([0, 1 / 2, 1 / 2, 1])
        butcher_b = jnp.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
        butcher_A = jnp.array([[0, 0, 0, 0], [1 / 2, 0, 0, 0], [0, 1 / 2, 0, 0], [0, 0, 1, 0]])
        self.order = 4
      case 6:
        butcher_c = jnp.array([0, 1 / 4, 1 / 4, 1 / 2, 3 / 4, 1])
        butcher_b = jnp.array([7 / 90, 0, 16 / 45, 2 / 15, 16 / 45, 7 / 90])
        butcher_A = jnp.array([[0, 0, 0, 0, 0, 0], [1 / 4, 0, 0, 0, 0, 0], [1 / 8, 1 / 8, 0, 0, 0, 0],
                               [0, 0, 1 / 2, 0, 0, 0], [3 / 16, -3 / 8, 3 / 8, 9 / 16, 0, 0],
                               [-3 / 7, 8 / 7, 6 / 7, -12 / 7, 8 / 7, 0]])
        self.order = 5
      case 7:
        butcher_c = jnp.array([0, 1 / 3, 2 / 3, 1 / 3, 5 / 6, 1 / 6, 1])
        butcher_b = jnp.array([13 / 200, 0, 11 / 40, 11 / 40, 4 / 25, 4 / 25, 13 / 200])
        butcher_A = jnp.array([[0, 0, 0, 0, 0, 0, 0], [1 / 3, 0, 0, 0, 0, 0, 0], [0, 2 / 3, 0, 0, 0, 0, 0],
                               [1 / 12, 1 / 3, -1 / 12, 0, 0, 0, 0], [25 / 48, -55 / 24, 35 / 48, 15 / 8, 0, 0, 0],
                               [3 / 20, -11 / 24, -1 / 8, 1 / 2, 1 / 10, 0, 0],
                               [-261 / 260, 33 / 13, 43 / 156, -118 / 39, 32 / 195, 80 / 39, 0]])
        self.order = 6
      case 9:
        butcher_c = jnp.array([0, 1 / 6, 1 / 3, 1 / 2, 2 / 11, 2 / 3, 6 / 7, 0, 1])
        butcher_b = jnp.array([0, 0, 0, 32 / 105, 1771561 / 6289920, 243 / 2560, 16807 / 74880, 77 / 1440, 11 / 270])
        butcher_A = jnp.array([[0, 0, 0, 0, 0, 0, 0, 0, 0], [1 / 6, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1 / 3, 0, 0, 0, 0, 0, 0, 0], [1 / 8, 0, 3 / 8, 0, 0, 0, 0, 0, 0],
                               [148 / 1331, 0, 150 / 1331, -56 / 1331, 0, 0, 0, 0, 0],
                               [-404 / 243, 0, -170 / 27, 4024 / 1701, 10648 / 1701, 0, 0, 0, 0],
                               [2466 / 2401, 0, 1242 / 343, -19176 / 16807, -51909 / 16807, 1053 / 2401, 0, 0, 0],
                               [5 / 154, 0, 0, 96 / 539, -1815 / 20384, -405 / 2464, 49 / 1144, 0, 0],
                               [-113 / 32, 0, -195 / 22, 32 / 7, 29403 / 3584, -729 / 512, 1029 / 1408, 21 / 16, 0]])
        self.order = 7
      case 11:
        sqrt_21 = jnp.sqrt(21)
        butcher_c = jnp.array([
            0, 1 / 2, 1 / 2, (7 + sqrt_21) / 14, (7 + sqrt_21) / 14, 1 / 2, (7 - sqrt_21) / 14, (7 - sqrt_21) / 14,
            1 / 2, (7 + sqrt_21) / 14, 1
        ])
        butcher_b = jnp.array([1 / 20, 0, 0, 0, 0, 0, 0, 49 / 180, 16 / 45, 49 / 180, 1 / 20])
        butcher_A = jnp.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1 / 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1 / 4, 1 / 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1 / 7, (-7 - 3 * sqrt_21) / 98, (21 + 5 * sqrt_21) / 49, 0, 0, 0, 0, 0, 0, 0, 0],
                               [(11 + sqrt_21) / 84, 0, (18 + 4 * sqrt_21) / 63, (21 - sqrt_21) / 252, 0, 0, 0, 0, 0, 0,
                                0],
                               [(5 + sqrt_21) / 48, 0, (9 + sqrt_21) / 36, (-231 + 14 * sqrt_21) / 360,
                                (63 - 7 * sqrt_21) / 80, 0, 0, 0, 0, 0, 0],
                               [(10 - sqrt_21) / 42, 0, (-432 + 92 * sqrt_21) / 315, (633 - 145 * sqrt_21) / 90,
                                (-504 + 115 * sqrt_21) / 70, (63 - 13 * sqrt_21) / 35, 0, 0, 0, 0, 0],
                               [1 / 14, 0, 0, 0, (14 - 3 * sqrt_21) / 126, (13 - 3 * sqrt_21) / 63, 1 / 9, 0, 0, 0, 0],
                               [
                                   1 / 32, 0, 0, 0, (91 - 21 * sqrt_21) / 576, 11 / 72, (-385 - 75 * sqrt_21) / 1152,
                                   (63 + 13 * sqrt_21) / 128, 0, 0, 0
                               ],
                               [
                                   1 / 14, 0, 0, 0, 1 / 9, (-733 - 147 * sqrt_21) / 2205, (515 + 111 * sqrt_21) / 504,
                                   (-51 - 11 * sqrt_21) / 56, (132 + 28 * sqrt_21) / 245, 0, 0
                               ],
                               [
                                   0, 0, 0, 0, (-42 + 7 * sqrt_21) / 18, (-18 + 28 * sqrt_21) / 45,
                                   (-273 - 53 * sqrt_21) / 72, (301 + 53 * sqrt_21) / 72, (28 - 28 * sqrt_21) / 45,
                                   (49 - 7 * sqrt_21) / 18, 0
                               ]])
        self.order = 8
      case _:
        raise ValueError("num_stages not supported for ExplicitRungeKutta. Supported: 1, 2, 3, 4, 6, 7, 9, 11")

    stages = jnp.asarray([[i] for i in range(num_stages)])
    stage_types = tuple('explicit' for i in range(num_stages))
    super().__init__("explicit_runge_kutta",
                     self._rule,
                     self._update,
                     stages,
                     stage_types,
                     butcher_c,
                     num_steps=1,
                     num_derivs=1,
                     num_stages=num_stages)
    self.num_stages = num_stages
    self.butcher_A_inv = jnp.asarray(invert_butcher_with_order(butcher_A)[0])
    self.butcher_A = butcher_A
    self.butcher_b = butcher_b
    self.butcher_c = butcher_c

  def _update(self, q_stages, q_n, q_t_n, dt):
    # Linear combination of stage results
    q_s_t = self.value_and_derivatives(q_stages, q_n, q_t_n, dt)[1]
    q_n1 = q_n[0] + dt * jnp.einsum("j,j...->...", self.butcher_b, q_s_t)
    q_t_n1 = jnp.einsum("j,j...->...", self.butcher_b, q_s_t)
    return q_n1, q_t_n1

  def _rule(self, q, q_n, q_t_n, dt):
    q_t = (1 / dt) * jnp.einsum("ij,j...->i...", self.butcher_A_inv, (q - jnp.stack([q_n[0]] * self.num_stages)))
    q = jnp.stack([q_n[0]] * self.num_stages) + dt * jnp.einsum("ij,j...->i...", self.butcher_A, q_t)
    return q, q_t

class DiagonallyImplicitRungeKutta(TimeIntegrator):
  """Diagonally implicit Runge-Kutta method.
  
  Args:
    num_stages (int): Number of stages (1, 2, 3).
  """

  def __init__(self, num_stages):
    match num_stages:  # From JC Butcher 2008: Numerical Methods for Ordinary Differential Equations, ISBN: 978-0-470-72335-7
      case 1:  # Implicit midpoint (Gauss-Legendre, 2nd order, symplectic)
        butcher_c = jnp.array([1 / 2])
        butcher_b = jnp.array([1])
        butcher_A = jnp.array([[1 / 2]])
        self.order = 2
      case 2:  # Crouzeix's method (3rd order)
        sqrt_3 = jnp.sqrt(3)
        one_half = 1 / 2
        butcher_c = jnp.array([one_half + sqrt_3 / 6, one_half - sqrt_3 / 6])
        butcher_b = jnp.array([one_half, one_half])
        butcher_A = jnp.array([[one_half + sqrt_3 / 6, 0], [-sqrt_3 / 3, one_half + sqrt_3 / 6]])
        self.order = 3
      case 3:  # Crouzeix's method (4rd order)
        alpha = 2 * jnp.cos(jnp.pi / 18) / jnp.sqrt(3)
        butcher_c = jnp.array([(1 + alpha) / 2, 1 / 2, (1 - alpha) / 2])
        butcher_b = jnp.array([1 / (6 * alpha**2), 1 - 1 / (3 * alpha**2), 1 / (6 * alpha**2)])
        butcher_A = jnp.array([[(1 + alpha) / 2, 0, 0], [-alpha / 2, (1 + alpha) / 2, 0],
                               [1 + alpha, -(1 + 2 * alpha), (1 + alpha) / 2]])
        self.order = 4
      case _:
        raise ValueError("num_stages not supported for ExplicitRungeKutta. Supported: 1, 2, 3")

    stages = jnp.asarray([[i] for i in range(num_stages)])
    stage_types = tuple('implicit' for i in range(num_stages))

    super().__init__("diagonally_implicit_runge_kutta",
                     self._rule,
                     self._update,
                     stages,
                     stage_types,
                     butcher_c,
                     num_steps=1,
                     num_derivs=1,
                     num_stages=num_stages)
    self.num_stages = num_stages
    self.butcher_A_inv = jnp.asarray(invert_butcher_with_order(butcher_A)[0])
    self.butcher_A = butcher_A
    self.butcher_b = butcher_b
    self.butcher_c = butcher_c

  def _update(self, q_stages, q_n, q_t_n, dt):
    # Linear combination of stage results
    q_s_t = self.value_and_derivatives(q_stages, q_n, q_t_n, dt)[1]
    q_n1 = q_n[0] + dt * jnp.einsum("j,j...->...", self.butcher_b, q_s_t)
    q_t_n1 = jnp.einsum("j,j...->...", self.butcher_b, q_s_t)
    return q_n1, q_t_n1

  def _rule(self, q, q_n, q_t_n, dt):  # q contains all stages (first dimension)
    q_t = (1 / dt) * jnp.einsum("ij,j...->i...", self.butcher_A_inv, (q - jnp.stack([q_n[0]] * self.num_stages)))
    q = jnp.stack([q_n[0]] * self.num_stages) + dt * jnp.einsum("ij,j...->i...", self.butcher_A, q_t)

    return q, q_t

class Kvaerno(TimeIntegrator):
  """Kvaerno method (explicit first stage diagonally implicit Runge-Kutta with embedded error estimation).
  
  Args:
    order (int): Order of the method (3, 4, 5).
    
  Supports PID control.
  """

  def __init__(self, order):
    match order:  # Coefficients from https://github.com/patrick-kidger/diffrax/blob/0a59c9dbd34f580efb3505386f38ce9fcedb120b/diffrax/_solver -> kvaerno{3,4,5}.py
      case 3:
        γ = 0.43586652150
        a21 = γ
        a31 = (-4 * γ**2 + 6 * γ - 1) / (4 * γ)
        a32 = (-2 * γ + 1) / (4 * γ)
        a41 = (6 * γ - 1) / (12 * γ)
        a42 = -1 / ((24 * γ - 12) * γ)
        a43 = (-6 * γ**2 + 6 * γ - 1) / (6 * γ - 3)

        butcher_c = jnp.array([0., 2 * γ, 1.0, 1.0])
        butcher_b = jnp.array([a41, a42, a43, γ])
        error_b = jnp.array([a41 - a31, a42 - a32, a43 - γ, γ])
        butcher_A = jnp.array([[0, 0, 0, 0], [a21, γ, 0, 0], [a31, a32, γ, 0], [a41, a42, a43, γ]])
        self.order = 3
      case 4:
        γ = 0.5728160625

        def poly(*args):
          return jnp.polyval(jnp.asarray(args), γ)

        a21 = γ
        a31 = poly(144, -180, 81, -15, 1) * γ / poly(12, -6, 1)**2
        a32 = poly(-36, 39, -15, 2) * γ / poly(12, -6, 1)**2
        a41 = poly(-144, 396, -330, 117, -18, 1) / (12 * γ**2 * poly(12, -9, 2))
        a42 = poly(72, -126, 69, -15, 1) / (12 * γ**2 * poly(3, -1))
        a43 = (poly(-6, 6, -1) * poly(12, -6, 1)**2) / (12 * γ**2 * poly(12, -9, 2) * poly(3, -1))
        a51 = poly(288, -312, 120, -18, 1) / (48 * γ**2 * poly(12, -9, 2))
        a52 = poly(24, -12, 1) / (48 * γ**2 * poly(3, -1))
        a53 = -(poly(12, -6, 1)**3) / (48 * γ**2 * poly(3, -1) * poly(12, -9, 2) * poly(6, -6, 1))
        a54 = poly(-24, 36, -12, 1) / poly(24, -24, 4)
        c2 = γ + a21
        c3 = γ + a31 + a32
        c4 = 1.0
        c5 = 1.0

        butcher_c = jnp.array([0, c2, c3, c4, c5])
        butcher_b = jnp.array([a51, a52, a53, a54, γ])
        error_b = jnp.array([a51 - a41, a52 - a42, a53 - a43, a54 - γ, γ])
        butcher_A = jnp.array([[0, 0, 0, 0, 0], [a21, γ, 0, 0, 0], [a31, a32, γ, 0, 0], [a41, a42, a43, γ, 0],
                               [a51, a52, a53, a54, γ]])
        self.order = 4
      case 5:
        γ = 0.26
        a21 = γ
        a31 = 0.13
        a32 = 0.84033320996790809
        a41 = 0.22371961478320505
        a42 = 0.47675532319799699
        a43 = -0.06470895363112615
        a51 = 0.16648564323248321
        a52 = 0.10450018841591720
        a53 = 0.03631482272098715
        a54 = -0.13090704451073998
        a61 = 0.13855640231268224
        a62 = 0
        a63 = -0.04245337201752043
        a64 = 0.02446657898003141
        a65 = 0.61943039072480676
        a71 = 0.13659751177640291
        a72 = 0
        a73 = -0.05496908796538376
        a74 = -0.04118626728321046
        a75 = 0.62993304899016403
        a76 = 0.06962479448202728

        butcher_c = jnp.array([0, 0.52, 1.230333209967908, 0.8957659843500759, 0.43639360985864756, 1.0, 1.0])
        butcher_b = jnp.array([a71, a72, a73, a74, a75, a76, γ])
        error_b = jnp.array([a71 - a61, a72 - a62, a73 - a63, a74 - a64, a75 - a65, a76 - γ, γ])
        butcher_A = jnp.array([[0, 0, 0, 0, 0, 0, 0], [a21, γ, 0, 0, 0, 0, 0], [a31, a32, γ, 0, 0, 0, 0],
                               [a41, a42, a43, γ, 0, 0, 0], [a51, a52, a53, a54, γ, 0, 0],
                               [a61, a62, a63, a64, a65, γ, 0], [a71, a72, a73, a74, a75, a76, γ]])
        self.order = 5
      case _:
        raise ValueError("order not supported for Kvaerno. Supported: 3, 4, 5")

    num_stages = butcher_c.shape[0]
    stages = jnp.asarray([[i] for i in range(num_stages)])
    stage_types = ('explicit', *tuple('implicit' for i in range(num_stages - 1)))

    super().__init__("Kvaerno",
                     self._rule,
                     self._update,
                     stages,
                     stage_types,
                     butcher_c,
                     num_steps=1,
                     num_derivs=1,
                     num_stages=num_stages)
    self.num_stages = num_stages
    self.butcher_A_inv = jnp.asarray(invert_butcher_with_order(butcher_A)[0])
    self.butcher_A = butcher_A
    self.butcher_b = butcher_b
    self.butcher_c = butcher_c
    self.error_b = error_b

  def _update(self, q_stages, q_n, q_t_n, dt):
    # Linear combination of stage results
    q_s_t = self.value_and_derivatives(q_stages, q_n, q_t_n, dt)[1]
    q_n1 = q_n[0] + dt * jnp.einsum("j,j...->...", self.butcher_b, q_s_t)
    q_t_n1 = jnp.einsum("j,j...->...", self.butcher_b, q_s_t)
    return q_n1, q_t_n1

  def _error_estimate(self, q_stages, q_n, q_t_n, dt):
    # Linear combination of stage results
    q_s_t = self.value_and_derivatives(q_stages, q_n, q_t_n, dt)[1]
    weights = self.error_b
    error_estimate = dt * jnp.einsum("j,j...->...", weights, q_s_t)
    return error_estimate

  def _rule(self, q, q_n, q_t_n, dt):  # q contains all stages (first dimension)
    q_t = (1 / dt) * jnp.einsum("ij,j...->i...", self.butcher_A_inv, (q - jnp.stack([q_n[0]] * self.num_stages)))
    q = jnp.stack([q_n[0]] * self.num_stages) + dt * jnp.einsum("ij,j...->i...", self.butcher_A, q_t)
    return q, q_t

class DormandPrince(TimeIntegrator):
  """Dormand-Prince method (explicit with embedded error estimation).
  
  Args:
    order (int): Order of the method (5, 8).
    
  Supports PID control.
  """

  def __init__(self, order):
    # Coefficients from https://github.com/patrick-kidger/diffrax/blob/0a59c9dbd34f580efb3505386f38ce9fcedb120b/diffrax/_solver -> dopri{5,8}.py
    match order:
      case 5:
        butcher_c = jnp.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0])
        butcher_b = jnp.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
        error_b = jnp.array([
            35 / 384 - 1951 / 21600, 0, 500 / 1113 - 22642 / 50085, 125 / 192 - 451 / 720,
            -2187 / 6784 - -12231 / 42400, 11 / 84 - 649 / 6300, -1.0 / 60.0
        ])
        butcher_A = jnp.array([[0, 0, 0, 0, 0, 0, 0], [1 / 5, 0, 0, 0, 0, 0, 0], [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
                               [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
                               [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
                               [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
                               [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]])
        self.order = 5
      case 8:
        butcher_c = jnp.array([
            0, 1 / 18, 1 / 12, 1 / 8, 5 / 16, 3 / 8, 59 / 400, 93 / 200, 5490023248 / 9719169821, 13 / 20,
            1201146811 / 1299019798, 1, 1, 1
        ])
        butcher_b = jnp.array([
            14005451 / 335480064, 0, 0, 0, 0, -59238493 / 1068277825, 181606767 / 758867731, 561292985 / 797845732,
            -1041891430 / 1371343529, 760417239 / 1151165299, 118820643 / 751138087, -528747749 / 2220607170, 1 / 4, 0
        ])
        error_b = jnp.array([
            14005451 / 335480064 - 13451932 / 455176623, 0, 0, 0, 0, -59238493 / 1068277825 - -808719846 / 976000145,
            181606767 / 758867731 - 1757004468 / 5645159321, 561292985 / 797845732 - 656045339 / 265891186,
            -1041891430 / 1371343529 - -3867574721 / 1518517206, 760417239 / 1151165299 - 465885868 / 322736535,
            118820643 / 751138087 - 53011238 / 667516719, -528747749 / 2220607170 - 2 / 45, 1 / 4, 0
        ])
        butcher_A = jnp.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1 / 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1 / 48, 1 / 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [1 / 32, 0, 3 / 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [5 / 16, 0, -75 / 64, 75 / 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [3 / 80, 0, 0, 3 / 16, 3 / 20, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [
                                   29443841 / 614563906, 0, 0, 77736538 / 692538347, -28693883 / 1125000000,
                                   23124283 / 1800000000, 0, 0, 0, 0, 0, 0, 0, 0
                               ],
                               [
                                   16016141 / 946692911, 0, 0, 61564180 / 158732637, 22789713 / 633445777,
                                   545815736 / 2771057229, -180193667 / 1043307555, 0, 0, 0, 0, 0, 0, 0
                               ],
                               [
                                   39632708 / 573591083, 0, 0, -433636366 / 683701615, -421739975 / 2616292301,
                                   100302831 / 723423059, 790204164 / 839813087, 800635310 / 3783071287, 0, 0, 0, 0, 0,
                                   0
                               ],
                               [
                                   246121993 / 1340847787, 0, 0, -37695042795 / 15268766246, -309121744 / 1061227803,
                                   -12992083 / 490766935, 6005943493 / 2108947869, 393006217 / 1396673457,
                                   123872331 / 1001029789, 0, 0, 0, 0, 0
                               ],
                               [
                                   -1028468189 / 846180014, 0, 0, 8478235783 / 508512852, 1311729495 / 1432422823,
                                   -10304129995 / 1701304382, -48777925059 / 3047939560, 15336726248 / 1032824649,
                                   -45442868181 / 3398467696, 3065993473 / 597172653, 0, 0, 0, 0
                               ],
                               [
                                   185892177 / 718116043, 0, 0, -3185094517 / 667107341, -477755414 / 1098053517,
                                   -703635378 / 230739211, 5731566787 / 1027545527, 5232866602 / 850066563,
                                   -4093664535 / 808688257, 3962137247 / 1805957418, 65686358 / 487910083, 0, 0, 0
                               ],
                               [
                                   403863854 / 491063109, 0, 0, -5068492393 / 434740067, -411421997 / 543043805,
                                   652783627 / 914296604, 11173962825 / 925320556, -13158990841 / 6184727034,
                                   3936647629 / 1978049680, -160528059 / 685178525, 248638103 / 1413531060, 0, 0, 0
                               ],
                               [
                                   14005451 / 335480064, 0, 0, 0, 0, -59238493 / 1068277825, 181606767 / 758867731,
                                   561292985 / 797845732, -1041891430 / 1371343529, 760417239 / 1151165299,
                                   118820643 / 751138087, -528747749 / 2220607170, 1 / 4, 0
                               ]])
        self.order = 8
      case _:
        raise ValueError("order not supported for DormandPrince. Supported: 5, 8")

    num_stages = butcher_c.shape[0]
    stages = jnp.asarray([[i] for i in range(num_stages)])
    stage_types = tuple('explicit' for i in range(num_stages))

    super().__init__("DormandPrince",
                     self._rule,
                     self._update,
                     stages,
                     stage_types,
                     butcher_c,
                     num_steps=1,
                     num_derivs=1,
                     num_stages=num_stages)
    self.num_stages = num_stages
    self.butcher_A_inv = jnp.asarray(invert_butcher_with_order(butcher_A)[0])
    self.butcher_A = butcher_A
    self.butcher_b = butcher_b
    self.butcher_c = butcher_c
    self.error_b = error_b

  def _update(self, q_stages, q_n, q_t_n, dt):
    # Linear combination of stage results
    q_s_t = self.value_and_derivatives(q_stages, q_n, q_t_n, dt)[1]
    q_n1 = q_n[0] + dt * jnp.einsum("j,j...->...", self.butcher_b, q_s_t)
    q_t_n1 = jnp.einsum("j,j...->...", self.butcher_b, q_s_t)
    return q_n1, q_t_n1

  def _error_estimate(self, q_stages, q_n, q_t_n, dt):
    # Linear combination of stage results
    q_s_t = self.value_and_derivatives(q_stages, q_n, q_t_n, dt)[1]
    weights = self.error_b
    error_estimate = dt * jnp.einsum("j,j...->...", weights, q_s_t)
    return error_estimate

  def _rule(self, q, q_n, q_t_n, dt):  # q contains all stages (first dimension)
    q_t = (1 / dt) * jnp.einsum("ij,j...->i...", self.butcher_A_inv, (q - jnp.stack([q_n[0]] * self.num_stages)))
    q = jnp.stack([q_n[0]] * self.num_stages) + dt * jnp.einsum("ij,j...->i...", self.butcher_A, q_t)
    return q, q_t

class GaussLegendreRungeKutta(TimeIntegrator):
  """Gauss-Legendre Runge-Kutta method (fully implicit).
  
  Args:
    num_stages (int): Number of stages.

  Accuracy: 2 * num_stages.
  """

  def __init__(self, num_stages):

    def get_gauss_legendre_coefficients(s):
      # from numpy.polynomial.legendre import leggauss

      # 1. Compute the Gauss-Legendre nodes and weights on [-1, 1]
      nodes, weights = np.polynomial.legendre.leggauss(s)

      # 2. Scale the nodes and weights to [0, 1]
      c = 0.5 * (nodes + 1)
      b = 0.5 * weights

      # 3. Compute matrix A
      A = np.zeros((s, s))
      for i in range(s):
        for j in range(s):
          # Build the Lagrange basis function L_j(x)
          L_j = np.poly1d([1.0])
          for k in range(s):
            if k != j:
              L_j = np.poly1d(np.convolve(L_j.coeffs, [1.0, -c[k]])) / (c[j] - c[k])
          # Integrate L_j(x) from 0 to c_i
          Lj_int = np.polyint(L_j)
          A[i, j] = Lj_int(c[i]) - Lj_int(0.0)

      # 4. Convert A, c, and b to JAX arrays
      A = jnp.array(A)
      c = jnp.array(c)
      b = jnp.array(b)
      return A, b, c

    butcher_A, butcher_b, butcher_c = get_gauss_legendre_coefficients(num_stages)
    butcher_A_inv, stage_order = invert_butcher_with_order(butcher_A)

    stage_types = tuple(st[1] for st in stage_order)
    stage_order = jnp.asarray([st[0] for st in stage_order])

    super().__init__("GaussLegendreRungeKutta",
                     self._rule,
                     self._update,
                     stage_order,
                     stage_types,
                     butcher_c,
                     num_steps=1,
                     num_derivs=1,
                     num_stages=num_stages)
    self.num_stages = num_stages
    self.butcher_A_inv = jnp.asarray(butcher_A_inv)
    self.butcher_A = butcher_A
    self.butcher_b = butcher_b
    self.butcher_c = butcher_c
    self.order = 2 * num_stages

  def _update(self, q_stages, q_n, q_t_n, dt):
    # Linear combination of stage results
    q_s_t = self.value_and_derivatives(q_stages, q_n, q_t_n, dt)[1]
    q_n1 = q_n[0] + dt * jnp.einsum("j,j...->...", self.butcher_b, q_s_t)
    q_t_n1 = jnp.einsum("j,j...->...", self.butcher_b, q_s_t)
    return q_n1, q_t_n1

  def _rule(self, q, q_n, q_t_n, dt):  # q contains all stages (first dimension)
    q_t = (1 / dt) * jnp.einsum("ij,j...->i...", self.butcher_A_inv, (q - jnp.stack([q_n[0]] * self.num_stages)))
    q = jnp.stack([q_n[0]] * self.num_stages) + dt * jnp.einsum("ij,j...->i...", self.butcher_A, q_t)
    return q, q_t


## Saving policies

@jax.tree_util.register_dataclass
@dataclass
class HistoryState:
  """
  Container for storing the history state data.

  Attributes:
      t: Dictionary of time data arrays.
      q: Dictionary of state variable arrays.
      user: Dictionary of additional user data.
  """
  t: dict[str, jnp.ndarray]
  q: dict[str, jnp.ndarray]
  user: Any

class SavePolicy(ABC):
  """Abstract base class for save strategies."""

  @abstractmethod
  def initialize(self, q, t_start, t_final, num_time_steps, user_data={}):
    """
    Initializes the storage.

    Args:
        q: Dictionary of state variables.
        t_start: Initial simulation time.
        t_final: Final simulation time.
        num_time_steps: Number of time steps.
        user_data: Dictionary of additional user data.

    Returns:
        An initial state for the saving strategy.
    """
    pass

  @abstractmethod
  def save_step(self, state, t, q, user_data={}):
    """
    Saves the desired data to the storage.

    Args:
        state: The current history state.
        t: The current time.
        q: The current state dictionary.
        user_data: Dictionary of additional user data.

    Returns:
        The updated history state.
    """
    pass

  @abstractmethod
  def finalize(self, state):
    """
    Finalizes the storage and returns the relevant history data.

    Args:
        state: The current history state.

    Returns:
        The finalized history data.
    """
    pass

class SaveNothingPolicy(SavePolicy):
  """A policy that does not save any data."""

  def initialize(self, q, t_start, t_final, num_time_steps, user_data={}):
    return None

  def save_step(self, state, t, q, user_data={}):
    return None

  def finalize(self, state):
    return None

@jax.tree_util.register_dataclass
@dataclass
class SaveEquidistantHistoryState:
  """
  State for the SaveAllPolicy.
  """
  t_final: float
  num_points: int
  target_times: jnp.ndarray
  t: dict[str, jnp.ndarray]
  q: dict[str, jnp.ndarray]
  current_save_idx: int
  user: Any

class SaveEquidistantPolicy(SavePolicy):
  """
  Saves data at (approximately) equidistant time points using pre-allocated arrays.
  """

  def __init__(self, num_points=None, tol=1e-6):
    self.tol = tol
    self.num_points = num_points

  def initialize(self, q, t_start, t_final, max_steps, user_data={}):
    """
    Initializes the preallocated arrays for time and states.

    Args:
        q_keys: Keys of the state variables.
        q_shapes: Shapes of the state variables.

    Returns:
        A tuple containing:
            - history_t: Array for the time data.
            - history_q: Dictionary of arrays for the state data.
            - target_times: Array of target time points.
            - current_save_idx: Initial index for saving.
    """
    num_points = self.num_points if self.num_points is not None else max_steps
    history_t = jnp.full(num_points + 1, jnp.nan)
    history_q = {key: jnp.full((num_points + 1,) + q[key].shape, jnp.nan) for key in q.keys()}
    history_user = {key: jnp.full((num_points + 1,) + user_data[key].shape, jnp.nan) for key in user_data.keys()}
    target_times = jnp.linspace(t_start, t_final, num_points + 1)
    current_save_idx = 0
    return SaveEquidistantHistoryState(t_final,
                                       num_points,
                                       target_times,
                                       history_t,
                                       history_q,
                                       current_save_idx,
                                       user=history_user)

  def save_step(self, state, t, q, user_data={}):
    """
    Saves the current state when the target time point is reached.

    Args:
      state: A tuple of (history_t, history_q, target_times, current_save_idx).
      t: Current time.
      q: Current state.

    Returns:
      Updated state with saved data and updated index.
    """

    # Condition: has the current time reached target time - tolerance?
    condition = t >= (state.target_times[state.current_save_idx] - self.tol)

    def do_save(state):
      # Save the current time
      state.t = state.t.at[state.current_save_idx].set(t)

      # Save the current states
      for key in state.q:
        state.q[key] = state.q[key].at[state.current_save_idx].set(q[key])

      if user_data is not None:
        for key in state.user:
          state.user[key] = state.user[key].at[state.current_save_idx].set(user_data[key])

      # Increment the save step, but clip it to num_points + 1
      state.current_save_idx = jnp.minimum(state.current_save_idx + 1, state.num_points)
      return state

    def do_nothing(state):
      return state

    new_state = jax.lax.cond(condition, do_save, do_nothing, state)
    return new_state

  def finalize(self, state):
    return HistoryState(state.t, state.q, state.user)

@jax.tree_util.register_dataclass
@dataclass
class SaveAllHistoryState:
  """
  State for the SaveAllPolicy.
  """
  t_final: float
  max_steps: int
  t: dict[str, jnp.ndarray]
  q: dict[str, jnp.ndarray]
  current_save_idx: int
  user: Any

class SaveAllPolicy(SavePolicy):
  """
  Saves data at every accepted time step.
  """

  def unpack_state(self, state):
    return state.t_final, state.max_steps, state.t, state.q, state.current_save_idx

  def initialize(self, q, t_start, t_final, max_steps, user_data={}):
    """Prepares the history state. Initializes the arrays with NaNs."""

    history_t = jnp.full(max_steps + 1, jnp.nan)
    history_q = {key: jnp.full((max_steps + 1,) + q[key].shape, jnp.nan) for key in q.keys()}
    history_user = {key: jnp.full((max_steps + 1,) + user_data[key].shape, jnp.nan) for key in user_data.keys()}
    current_save_idx = 0
    return SaveAllHistoryState(t_final, max_steps, history_t, history_q, current_save_idx, user=history_user)

  def save_step(self, state, t, q, user_data={}):
    """Save current state."""
    state.t = state.t.at[state.current_save_idx].set(t)

    for key in state.q:
      state.q[key] = state.q[key].at[state.current_save_idx].set(q[key])
    if user_data is not None:
      for key in state.user:
        state.user[key] = state.user[key].at[state.current_save_idx].set(user_data[key])

    # Increment the save step, but clip it to max_steps
    state.current_save_idx = jnp.minimum(state.current_save_idx + 1, state.max_steps)
    return state

  def finalize(self, state):
    return HistoryState(state.t, state.q, state.user)


## Step size controler
class StepSizeController(ABC):
  """
  Abstract base class for adaptive step size controllers.
  """

  @abstractmethod
  def initialize(self, initial_error):
    """
      Initialize the controller state based on the initial error.

      Args:
          initial_error (float): The initial scaled error estimate.

      Returns:
          Any: The initial state of the controller.
      """
    pass

  @abstractmethod
  def compute_scaler(self, error, q, q_n, state, order, converged, num_iterations, dt, verbose):
    """
      Compute the scaling factor `step_scaler` and update the controller state.

      Args:
          error (float): The current scaled error estimate.
          state (Any): The current state of the controller.

      Returns:
          Tuple[float, Any]: A tuple containing the scaling factor `step_scaler` and the updated state.
      """
    pass

  @abstractmethod
  def check_accept(self, state, converged, verbose):
    """
      Check whether the current step should be accepted.

      Args:
          state (Any): The current state of the controller.

      Returns:
          Bool: Whether the step should be accepted.
      """
    pass

@jax.tree_util.register_dataclass
@dataclass
class PIDControllerState:
  """
  State for the Proportional-Integral-Derivative Controller.
  """
  step_scaler: float
  e_n: float  # error_n
  e_nn: float  # error_{n-1}
  accept: bool
  interrupt: bool


class PIDController(StepSizeController):
  """Proportional-Integral-Derivative (PID) Step Size Controller.
  
  Inspired by https://docs.kidger.site/diffrax/api/stepsize_controller and https://docs.sciml.ai/DiffEqDocs/stable/extras/timestepping/
  """

  def __init__(self,
               pcoeff: float = 0.0,
               icoeff: float = 1.0,
               dcoeff: float = 0.0,
               limiter: Any = None,
               atol=1e-6,
               rtol=1e-3):
    """
    Initialize the PIDController.

    Args:
      pcoeff (float): The coefficient of the proportional part of the step size control.
      icoeff (float): The coefficient of the integral part of the step size control.
      dcoeff (float): The coefficient of the derivative part of the step size control.
      limiter (callable): Limiter function. If None the limiter is set to `1.0 + jnp.arctan(x - 1.0)`.
      atol (float): Absolute tolerance.
      rtol (float): Relative tolerance.
    """
    self.limiter = limiter if limiter is not None else (lambda x: 1.0 + jnp.arctan(x - 1.0))
    self.atol = atol
    self.rtol = rtol
    self.pcoeff = pcoeff
    self.icoeff = icoeff
    self.dcoeff = dcoeff

  def initialize(self, initial_error):
    """
    TODO: Initialization of time increment? See diffrax
    """
    return PIDControllerState(step_scaler=1., e_n=initial_error, e_nn=initial_error, accept=True, interrupt=False)

  def compute_scaler(self, error, q, q_n, state, order, converged, num_iterations, dt, verbose):
    keys = q.keys()
    atol = self.atol if isinstance(self.atol, dict) else {key: self.atol for key in keys}
    rtol = self.rtol if isinstance(self.rtol, dict) else {key: self.rtol for key in keys}

    # Check that all integrators support error estimation (integrators that does not support return None as error)
    for key in keys:
      if error[key] is None:
        raise ValueError(
            f"\n\nError estimation is not supported for the integrator of field '{key}'!\n Use a different integrator or a time step control that does not require error estimation."
        )

    # Scaled error estimate
    scaled_error = {
        key: jnp.abs(jnp.divide(error[key], atol[key] + jnp.maximum(q[key], q_n[key][0]) * rtol[key])) for key in keys
    }
    scaled_error = dict_flatten(scaled_error)

    # Hairer norm
    inv_error_norm = 1 / jnp.sqrt(jnp.mean(scaled_error**2))

    # PID coefficients
    k = order + 1
    beta1 = (self.pcoeff + self.icoeff + self.dcoeff) / k
    beta2 = -(self.pcoeff + 2 * self.dcoeff) / k
    beta3 = self.dcoeff / k

    # PID rule
    step_scaler = jnp.power(inv_error_norm, beta1) * jnp.power(state.e_n, beta2) * jnp.power(state.e_nn, beta3)

    # Handle zero error
    step_scaler = jnp.where(jnp.isinf(inv_error_norm), 1., step_scaler)

    # Apply limiter function
    step_scaler = self.limiter(step_scaler)

    # Update state
    state = PIDControllerState(step_scaler, state.e_n, state.e_nn, state.accept, state.interrupt)

    # Update previous errors if step is accepted
    accepted = self.check_accept(state, converged, verbose).accept

    def do_accept(state):
      return PIDControllerState(step_scaler, inv_error_norm, state.e_n, state.accept, state.interrupt)

    def do_reject(state):
      return PIDControllerState(step_scaler, state.e_n, state.e_nn, state.accept, state.interrupt)

    state = jax.lax.cond(accepted, do_accept, do_reject, state)

    return state

  def check_accept(self, state, converged, verbose):
    return PIDControllerState(state.step_scaler, state.e_n, state.e_nn, converged, state.interrupt)

@jax.tree_util.register_dataclass
@dataclass
class CSSControllerState:
  """
  State for the Constant Step Size Controller.
  """
  step_scaler: float
  accept: bool
  interrupt: bool

class ConstantStepSizeController(StepSizeController):
  """Constant Step Size Controller."""

  def __init__(self):
    pass

  def initialize(self, initial_error):
    return CSSControllerState(step_scaler=1., accept=True, interrupt=False)

  def compute_scaler(self, error, q, q_n, state, order, converged, num_iterations, dt, verbose):
    return state

  def check_accept(self, state, converged, verbose):
    # Give warning in case not converged and not already warned
    def send_warning(s):
      if verbose >= 0:
        jax.debug.print("Root solver did not converge, but stepsize controller can not reduce step size!")
      return jnp.array(True)

    warn = jnp.logical_and(jnp.logical_not(converged), jnp.logical_not(state.interrupt))
    interrupt = jax.lax.cond(warn, send_warning, lambda s: s, state.interrupt)
    return CSSControllerState(step_scaler=state.step_scaler, accept=converged, interrupt=interrupt)

@jax.tree_util.register_dataclass
@dataclass
class RootIterationControllerState:
  """
  State for the Root Iteration Controller.
  """
  step_scaler: float  # Scaling factor for the step size
  dt: float
  accept: bool
  interrupt: bool

class RootIterationController(StepSizeController):
  """Root Iteration Controller for adjusting step size based on number of root solver iterations.
  
  Tries to achieve a target number of Newton iterations by adjusting the step size.
  Does not consider possible error estimates. Maximal and minimal step sizes can be set.
  """

  def __init__(self,
               target_niters: int = 6,
               gamma: float = 0.5,
               max_step_size: float = 1e20,
               min_step_size: float = 1e-6):
    """
    Initialize the RootIterationController.

    Args:
      target_niters (int): Desired number of Newton iterations.
      gamma (float): Proportionality factor for step size adjustment.
      max_step_size (float): Maximum allowable step size.
      min_step_size (float): Minimum allowable step size.
    """
    self.target_niters = target_niters
    self.gamma = gamma
    self.max_step_size = max_step_size
    self.min_step_size = min_step_size

  def initialize(self, initial_error):
    return RootIterationControllerState(step_scaler=1.0, dt=1.0, accept=True, interrupt=False)

  def compute_scaler(self, error, q, q_n, state, order, converged, num_iterations, dt, verbose):
    # Proportional control based on the deviation from target_niters
    correction = (1 + self.gamma * (self.target_niters - num_iterations) / self.target_niters)
    correction = jnp.clip(correction, 0.5, 2.0)

    # If not converged, devide step size by 2
    correction = jax.lax.cond(converged, lambda x: jnp.astype(x, float), lambda x: 1 / 2, correction)

    # Change step size
    dt_old = dt
    dt = dt * correction

    # Limit step size to bounds
    dt = jnp.clip(dt, self.min_step_size, self.max_step_size)

    # Recalculate scaling factor
    step_scaler = dt / dt_old

    # Update state
    return RootIterationControllerState(step_scaler=step_scaler, dt=dt, accept=state.accept, interrupt=state.interrupt)

  def check_accept(self, state, converged, verbose):
    # Give warning in case not converged and step size can not be reduced further
    def send_warning(s):
      if verbose >= 0:
        jax.debug.print("Root solver did not converge, but minimum step_size is reached!")
      return jnp.array(True)

    warn = jnp.logical_and(jnp.logical_and(jnp.logical_not(converged), jnp.isclose(state.dt, self.min_step_size)),
                           jnp.logical_not(state.interrupt))
    interrupt = jax.lax.cond(warn, send_warning, lambda s: s, state.interrupt)
    return RootIterationControllerState(state.step_scaler, state.dt, converged, interrupt)


## Root solvers
@jax.tree_util.register_dataclass
@dataclass
class RootSolverResult:
  root: Any
  num_iterations: int
  converged: bool
  settings: Any

def newton_solver(
    func,
    x0,
    *,
    atol=1e-10,
    rtol=1e-8,
    max_iter=20,
    damping_factor=1.0,
    tangent_fun=None,
    lin_solve_fun=None,
    constrained_dofs=None,
    constrained_values=None,
    verbose=0,
    settings=None,
    static_settings=None,
    termination_mode='residual',
    _pre_iteration_update=None,
):
  """
    Newton-Raphson solver to find a root of F(x)=0.

    Args:
      func: Function F(x, settings (auxiliary data)) whose zero is sought.
      x0: Initial guess.
      atol: Absolute tolerance.
      rtol: Relative tolerance.
      max_iter: Maximum number of iterations.
      damping_factor: Constant damping factor for the Newton update (0 < damping_factor <= 1; 1 -> no damping).
      tangent_fun: Function to compute the Jacobian. (Default: jax.jacfwd(func)), same signature as func.
      lin_solve_fun: Function to solve the linear system. (Default: jnp.linalg.solve)
      constrained_dofs: Boolean mask for fixed degrees of freedom.
      constrained_values: Fixed values for constrained DOFs.
      verbose: If >=1, prints the residual norm each iteration.
      termination_mode: 'residual' uses the residual norm; 'update' uses the update size.

    Returns:
      A RootSolverResult dataclass with fields:
        - .root: the computed solution,
        - .num_iterations: number of updates performed,
        - .converged: convergence flag.
    """
  if settings is None:
    settings = {}
  if _pre_iteration_update is None:
    def _pre_iteration_update(_x, settings):
      return settings
  else:
    _pre_iteration_update = jax.jit(_pre_iteration_update)

  # Set constraints if provided.
  free_dofs = None
  if constrained_dofs is not None:
    free_dofs = jnp.invert(constrained_dofs)
    if constrained_values is None:
      raise ValueError("constrained_values must be provided if constrained_dofs is not None!")
    if constrained_dofs.shape != x0.shape:
      raise ValueError("constrained_dofs must have the same shape as x0!")
    if constrained_values.shape != x0.shape:
      raise ValueError("constrained_values must have the same shape as x0!")

  if lin_solve_fun is None:
    if constrained_dofs is None:
      def lin_solve_fun(J, b, free_dofs):
        return jnp.linalg.solve(J, b)
    else:
      def lin_solve_fun(J, b, free_dofs):
        free_f = free_dofs.astype(J.dtype)
        I = jnp.eye(J.shape[0], dtype=J.dtype)

        J_mod = J * free_f[:, None] * free_f[None, :] + I * (1.0 - free_f)[:, None]
        b_mod = jnp.where(free_dofs, b, jnp.zeros_like(b))

        return jnp.linalg.solve(J_mod, b_mod)

  use_residual = termination_mode == 'residual'
  n = x0.shape[0]
  n_sqrt = jnp.sqrt(n)

  if tangent_fun is None:
    @jax.jit()
    def residual_and_tangent(x, settings):
      fx, f_lin = jax.linearize(lambda z: func(z, settings), x)
      eye = jnp.eye(n, dtype=x.dtype)
      return fx, jax.vmap(f_lin)(eye).T
  else:
    residual_and_tangent = None
    func = jax.jit(func) # TODO: test

  # Pre-check: allows 0 iterations
  fx0 = residual_and_tangent(x0, settings)[0] if tangent_fun is None else func(x0, settings)
  fx0_norm = jnp.linalg.norm(fx0) / n_sqrt
  stop0 = jnp.any(jnp.logical_or(jnp.isnan(fx0_norm), jnp.isinf(fx0_norm)))
  tol = atol + rtol * jnp.max(jnp.stack([fx0_norm, 1.0]))

  # if verbose >= 1 and use_residual and constrained_dofs is None:
  #   jax.debug.print("Iteration 0: Residual norm = {res}",  res=fx0_norm)

  # 0 iterations only in residual mode and without constraints
  conv0 = fx0_norm < tol if use_residual and constrained_dofs is None else False

  # count starts at -1 (1st update sets count -> 0)
  state = (x0, -1, fx0_norm, conv0, stop0, settings, tol)

  def body(state):
    x, count, fx_norm, converged, stop, settings, tol = state

    # Pass dynamic closures through the lax.cond operand to avoid
    # "No constant handler for DynamicJaxprTracer" errors in nested compilation contexts.
    _cv = constrained_values if constrained_values is not None else jnp.zeros(1, dtype=x.dtype)

    def update_fn(operand):
      state, _cv = operand
      x, count, fx_norm, converged, stop, settings, tol = state
      first_iter = count == -1

      # Apply boundary conditions
      x_mod = x if constrained_dofs is None else jnp.where(constrained_dofs, _cv, x)

      # In first iteration apply boundary conditions after evaluating fx and J
      if constrained_dofs is not None:
        x_mod = jnp.where(first_iter, x, x_mod)

      settings = jax.lax.cond(
          first_iter,
          lambda s: _pre_iteration_update(x_mod, s),
          lambda s: s,
          settings,
      )

      skip_update = fx_norm < tol if use_residual else False
      if constrained_dofs is not None:
        # After at least one update, skipping is allowed; not before.
        skip_update = jnp.logical_and(skip_update, count >= 0)

      # Jacobian + fx at the current point
      if tangent_fun is not None:
        fx_raw = func(x_mod, settings)
        J = tangent_fun(x_mod, settings)
      else:
        fx_raw, J = residual_and_tangent(x_mod, settings)

      # Apply constraints to residual
      if constrained_dofs is not None:
        def first_iter_branch(_):
            dc = jnp.where(constrained_dofs, _cv - x_mod, 0.0)
            fx0 = fx_raw + J @ dc
            fx0 = jnp.where(constrained_dofs, 0.0, fx0)
            # if verbose >= 1 and use_residual:
            #   jax.debug.print("Iteration 0: Residual norm = {res}",  res=jnp.linalg.norm(fx0) / n_sqrt)
            return fx0
        def later_iter_branch(_):
            return jnp.where(constrained_dofs, 0.0, fx_raw)
        fx = jax.lax.cond(first_iter, first_iter_branch, later_iter_branch, operand=None)

      else:
        fx = fx_raw

      fxn = jnp.linalg.norm(fx) / n_sqrt
      stop_now = jnp.any(jnp.logical_or(jnp.isnan(fxn), jnp.isinf(fxn)))
      if constrained_dofs is not None and use_residual:
        # For displacement-driven loads, func(x0) can be zero because
        # the new Dirichlet values are only injected through the first tangent.
        # Scale rtol by that corrected first residual instead.
        tol_candidate = atol + rtol * jnp.max(jnp.stack([fx_norm, fxn, 1.0]))
        tol = jnp.where(jnp.logical_and(first_iter, jnp.logical_not(stop_now)), tol_candidate, tol)

      # Newton step — use jnp.where to avoid nested lax.cond
      solve_data = settings.get("_linear solve data", free_dofs)
      delta_solve = lin_solve_fun(J, -fx, solve_data)
      if use_residual:
        skip_or_stop = jnp.logical_or(skip_update, stop_now)
        delta = jnp.where(skip_or_stop, jnp.zeros_like(x_mod), delta_solve)
      else:
        delta = jnp.where(stop_now, jnp.zeros_like(x_mod), delta_solve)

      x_new = x_mod + damping_factor * delta
      if constrained_dofs is not None:
        x_new = jnp.where(constrained_dofs, _cv, x_new)

      # Convergence check — use jnp.where to avoid nested lax.cond
      settings = _pre_iteration_update(x_new, settings)
      if use_residual:
        fx_new_raw = residual_and_tangent(x_new, settings)[0] if tangent_fun is None else func(x_new, settings)
        fx_new = fx_new_raw if constrained_dofs is None else jnp.where(constrained_dofs, 0.0, fx_new_raw)
        fx_new_norm_full = jnp.linalg.norm(fx_new) / n_sqrt
        stop_new_full = jnp.any(jnp.logical_or(jnp.isnan(fx_new_norm_full), jnp.isinf(fx_new_norm_full)))

        skip_or_stop = jnp.logical_or(skip_update, stop_now)
        fx_new_norm = jnp.where(skip_or_stop, fxn, fx_new_norm_full)
        stop_new = jnp.where(skip_or_stop, stop_now, stop_new_full)

        new_converged = fx_new_norm < tol
        new_fx_norm = fx_new_norm
        new_stop = jnp.logical_or(stop_now, stop_new)
      else:
        if constrained_dofs is not None:
          upd_norm = jnp.linalg.norm(jnp.where(constrained_dofs, 0.0, delta)) / n_sqrt
        else:
          upd_norm = jnp.linalg.norm(delta) / n_sqrt
        new_converged = upd_norm < atol
        new_fx_norm = fxn
        new_stop = stop_now

      new_count = count + 1
      if verbose >= 1:
        jax.debug.print("Iteration {iter}, Residual norm: {res}", iter=new_count+1, res=new_fx_norm, ordered=True)

      return (x_new, new_count, new_fx_norm, new_converged, new_stop, settings, tol), _cv

    new_state, _ = update_fn((state, _cv))
    return new_state

  def cond_fn(state):
    _, count, _, converged, stop, _, _ = state
    return jnp.logical_and(
      count < max_iter - 1,
      jnp.logical_not(jnp.logical_or(converged, stop))
    )

  final_state = jax.lax.while_loop(cond_fn, body, state)
  x_final, iterations, _, conv, _, settings, _ = final_state

  # If diverged, fallback to the initial guess
  is_nan_inf = jnp.any(jnp.logical_or(jnp.isnan(x_final), jnp.isinf(x_final)))
  x_final = jnp.where(
    jnp.logical_or(~conv, is_nan_inf),
    x0,
    x_final
  )
  conv = jnp.logical_and(conv, ~is_nan_inf)
  return RootSolverResult(x_final, iterations+1, conv, settings)


## Time stepping manager
_INITIAL_DERIVATIVES_KEY = "_initial derivatives"
_RESTART_Q_N_KEY = "_time stepping q_n"
_RESTART_Q_DER_N_KEY = "_time stepping q_der_n"
_RESTART_CONTROLLER_STATE_KEY = "_step size controller state"
_RESTART_DT_KEY = "_step size"

@jax.tree_util.register_dataclass
@dataclass
class TimeSteppingManagerState:
  """
  State for the TimeSteppingManager.

  Attributes:
      q (dict[str, jnp.ndarray]): Final state variables after time stepping.
      settings (dict[str, Any]): Simulation settings after the run.
      q_n (dict[str, jnp.ndarray]): Final value history for restart/continuation.
      q_der_n (dict[str, jnp.ndarray]): Final derivative history for restart/continuation.
      dt (float): Suggested next step size after the final loop iteration.
      controller_state (Any): Final step size controller state.
      history (Any): Recorded history data (if a save policy is used).
      num_steps (int): Total number of steps taken.
      num_accepted (int): Number of accepted time steps.
      num_rejected (int): Number of rejected time steps.
  """
  q: dict[str, jnp.ndarray]
  settings: dict[str, Any]
  q_n: dict[str, jnp.ndarray]
  q_der_n: dict[str, jnp.ndarray]
  dt: float
  controller_state: Any
  history: Any
  num_steps: int
  num_accepted: int
  num_rejected: int

class TimeSteppingManager:
  """
  Manages the time stepping procedure for a simulation using multi-stage integration schemes.

  This class orchestrates the simulation by coordinating various components such as:
    - Time integrators for different fields,
    - A root solver for implicit equations,
    - An adaptive step size controller,
    - A save policy for recording history,
    - Pre-step and post-step update functions for custom processing.

  The manager initializes the simulation state from given degrees of freedom (DOFs) and then runs a loop
  over a specified number of time steps. At each step, it computes the new state using multi-stage methods,
  applies error control and adaptive time stepping, and optionally records simulation history. The final state
  along with simulation statistics is returned.
  """

  def __init__(
      self,
      static_settings,
      settings={'current time': 0.0},
      root_solver=newton_solver,
      save_policy=None,
      step_size_controller=ConstantStepSizeController(),
      postprocessing_fun=lambda q_fun, t, settings: {},
      pre_step_updates=None,
      post_step_updates=None,
  ):
    self.integrators = static_settings['time integrators']
    self.root_solver = root_solver
    self.num_time_derivs = {key: integrator.num_derivs for key, integrator in self.integrators.items()}
    self.num_steps = {key: integrator.num_steps for key, integrator in self.integrators.items()}
    self.save_policy = save_policy
    self.step_size_controller = step_size_controller
    self.postprocessing_fun = postprocessing_fun
    self.static_settings = static_settings
    if pre_step_updates is None:
      def pre_step_updates(t, settings):
        return settings
    self.pre_step_updates = pre_step_updates
    if post_step_updates is None:
      def post_step_updates(q_fun, t, settings):
        return settings
    self.post_step_updates = post_step_updates
    self.verbose = static_settings.get('verbose', 0)
    self._global_template = None  # Will be set in _initialize

  @staticmethod
  def _tree_flatten(obj):
    children = ()
    aux_data = (obj.integrators, obj.num_time_derivs, obj.num_steps, obj.root_solver, obj.save_policy,
                obj.step_size_controller, obj.postprocessing_fun, obj.static_settings, obj.verbose,
                obj.pre_step_updates, obj.post_step_updates, obj._global_template)
    return (children, aux_data)

  @staticmethod
  def _tree_unflatten(aux_data, children):
    obj = object.__new__(TimeSteppingManager)
    (obj.integrators, obj.num_time_derivs, obj.num_steps, obj.root_solver, obj.save_policy,
     obj.step_size_controller, obj.postprocessing_fun, obj.static_settings, obj.verbose,
     obj.pre_step_updates, obj.post_step_updates, obj._global_template) = aux_data
    return obj

  def _initialize(self, dofs, settings=None):
    """
      Initializes history for multi-step methods and stores the global DOF structure
      as a template for unflattening.
    """
    q_n = {key: jnp.repeat(dofs[key][None, ...], self.num_steps[key], axis=0) for key in dofs}
    q_der_n = {key: jnp.zeros((self.num_steps[key], self.num_time_derivs[key], *dofs[key].shape)) for key in dofs}
    if settings is not None:
      restart_q_n = settings.get(_RESTART_Q_N_KEY, None)
      if restart_q_n is not None:
        q_n = {key: jnp.asarray(restart_q_n[key]) for key in dofs}

      restart_q_der_n = settings.get(_RESTART_Q_DER_N_KEY, None)
      if restart_q_der_n is not None:
        q_der_n = {key: jnp.asarray(restart_q_der_n[key]) for key in dofs}
      else:
        initial_derivatives = settings.get(_INITIAL_DERIVATIVES_KEY, None)
        if initial_derivatives is not None:
          for key, by_order in initial_derivatives.items():
            if key not in q_der_n:
              raise KeyError(f"Initial derivatives contain unknown field '{key}'.")
            for order, value in by_order.items():
              order_idx = int(order) - 1
              if order_idx < 0 or order_idx >= self.num_time_derivs[key]:
                raise ValueError(
                    f"Initial derivative order {order} for field '{key}' is not supported by its integrator."
                )
              q_der_n[key] = q_der_n[key].at[:, order_idx].set(jnp.asarray(value))
    self._global_template = dofs
    return q_n, q_der_n

  def _assemble_sparse_tangent(self, x_flat, q_stages, q_n, q_t_n, dt, t, current_stages, settings):
    num_domains = len(self.static_settings["assembling mode"])
    num_dofs = sum(v.size for v in self._global_template.values())
    if current_stages.shape[0] != 1:
      raise NotImplementedError("Sparse tangent assembly currently supports only one implicit stage per block.")

    integrated_tangent = sparse.empty((num_dofs, num_dofs), dtype=float, index_dtype=jnp.int_)

    # Make sure all assembling modes are 'user residual' as it is the only one currently supported
    assert all(item == 'user residual'
               for item in self.static_settings["assembling mode"]), "Only 'user residual' assembling mode is supported."

    # Loop over all sets of integration points/ domains
    for domain in range(num_domains):
      integrated_tangent += self._assemble_sparse_tangent_domain(x_flat, q_stages, q_n, q_t_n, dt, t, current_stages, settings, domain)

    return integrated_tangent

  def _assemble_sparse_tangent_domain(self, x_flat, q_stages, q_n, q_t_n, dt, t, current_stages, settings, domain):
    # Todo: other modes? e.g. for potential-based problems (like user potentials in assembler)
    # todo: currently only single stages and blocks possible

    # Reconstruct global DOF structure from flat vector
    global_dofs = reshape_as(x_flat, self._global_template)

    # Get elementwise quantities (model_fun, node coordinates, elem_numbers, connectivity)
    model_fun, x_nodes, elem_numbers, connectivity = assembler._get_element_quantities(
        global_dofs, settings, self.static_settings, domain)

    def extract_dofs(dofs, node_list, axis=0):
      # return jax.tree.map(lambda x, y: jnp.take(x, y, axis=axis), dofs, node_list)
      # A model on a subdomain only carries the fields present in its block's
      # connectivity, so restrict the extraction to those (node_list keys).
      return {key: jnp.take(dofs[key], node_list[key], axis=axis)
              for key in dofs.keys() if key in node_list}

    # Calculate the tangent for each element
    # @jax.jit(inline=True)
    def element_tangent_wrapper(local_dofs, elem_number, node_list):
      # Extract local DOFs for the current element
      local_q_stages = extract_dofs(q_stages, node_list, axis=1)
      local_q_n = extract_dofs(q_n, node_list, axis=1)
      local_q_t_n = extract_dofs(q_t_n, node_list, axis=2)
      stage = current_stages[0]
      for key in local_dofs:
        local_q_stages[key] = local_q_stages[key].at[stage].set(local_dofs[key])

      def diffable_q_fun(t):
        local_diffable = {}
        for key in local_dofs:
          tup = _stage_tuple(
              self.integrators[key].value_and_derivatives(local_q_stages[key], local_q_n[key], local_q_t_n[key], dt),
              stage,
              local_q_stages[key].shape,
          )
          local_diffable[key] = discrete_value_with_derivatives(t, tup[0], tup[1:])  # todo: handle multiple blocks
        return local_diffable

      if isinstance(x_nodes, dict):
        local_node_coor = extract_dofs(x_nodes, node_list)
      else:
        local_node_coor = jnp.take(x_nodes, node_list['physical coor'], axis=0)
      return model_fun(diffable_q_fun, local_node_coor, elem_number, settings, self.static_settings, domain)

    def element_tangent(elem_number, node_list):
      local_dofs = extract_dofs(global_dofs, node_list)
      return jax.jacfwd(lambda x: element_tangent_wrapper(x, elem_number, node_list))(local_dofs)

    num_local_dofs = assembler._get_num_local_dofs(connectivity, global_dofs)

    # tangent_contributions = jax.vmap(element_tangent, in_axes=(0, 0))(elem_numbers, connectivity)
    tangent_contributions = assembler._batched_map(element_tangent, elem_numbers, connectivity, num_local_dofs)

    data = dict_flatten(tangent_contributions)
    indices = assembler._get_indices(connectivity, global_dofs)

    if isinstance(global_dofs, dict):
      num_dofs = sum(v.size for v in global_dofs.values())
    else:
      num_dofs = global_dofs.size

    tangent_matrix = sparse.BCOO((data, indices), shape=(num_dofs, num_dofs))
    return tangent_matrix

  def _multi_stage_step(self, q, t, t_n, dt, q_n, q_t_n, settings):
    # Assumption: All integrators have the same number of stages
    num_stages = next(iter(self.integrators.values())).num_stages
    template = jax.lax.stop_gradient(q)

    stage_list = next(iter(self.integrators.values())).stage_list
    stage_positions = next(iter(self.integrators.values())).stage_positions
    num_blocks = stage_list.shape[0]

    q_stages = {key: jnp.repeat(q[key][None, ...], num_stages, axis=0) for key in template.keys()}

    # Run through all stages
    converged = True
    num_iterations = 0
    state_init = (q_stages, num_iterations, converged, settings)

    def block_body(block_number, state):
      q_stages, num_iterations, converged, settings = state
      current_stages = stage_list[block_number]
      num_coupled_stages = current_stages.shape[0]
      t_stage = t_n + dt * stage_positions[current_stages]

      # Update e.g. boundary conditions and 'current time' for the assembler for PDE problems
      settings['current time'] = t_stage[0]
      apply_bcs = self.static_settings.get("apply dirichlet bcs", None)
      if apply_bcs is not None:
        settings = apply_bcs(t_stage[0], settings)
      if self.pre_step_updates is not None:
        settings = self.pre_step_updates(t_stage[0], settings)

      # Getting some settings for solver and tangent specific settings
      dirichlet_dofs = settings.get('dirichlet dofs', None)
      dirichlet_conditions = settings.get('dirichlet conditions', None)
      if dirichlet_dofs is not None:
        assert dirichlet_conditions is not None, "'dirichlet conditions' must be provided if 'dirichlet dofs' are given."
        dirichlet_dofs = dict_flatten(dirichlet_dofs)
        dirichlet_conditions = dict_flatten(dirichlet_conditions)
      free_dofs = ~dirichlet_dofs if dirichlet_dofs is not None else None

      solver_backend = self.static_settings.get('solver backend', None)
      solver_subtype = self.static_settings.get('solver', None)
      verbose = self.static_settings.get('verbose', 0)
      impl_diff_mode = self.static_settings.get('implicit diff mode', 'forward')

      # Optional: Prepare assembling kernel + template 
      assembling_kernel = self.static_settings.get("assembling kernel", None)
      assembling_template = settings.get("assembling template", None)

      def _split_solve_data(solve_data):
        if isinstance(solve_data, tuple):
          return solve_data
        return solve_data, None

      def _prepare_mat_for_external(mat, solve_data):
        """
        Applies, if available, the JAX assembling kernel and returns
        either the original BCOO matrix or a CSR triple
        (data, col_sorted, indptr) that is directly given to the callback.
        """
        _, solve_template = _split_solve_data(solve_data)
        if assembling_kernel is None or solve_template is None:
          return mat

        if isinstance(mat, jnp.ndarray):
          mat_bcoo = sparse.bcoo_fromdense(mat)
        else:
          mat_bcoo = mat

        assembled_values = assembling_kernel(mat_bcoo.data, solve_template.scatter)

        return (
          assembled_values,
          solve_template.col_sorted,
          solve_template.indptr,
        )

      # Default solver
      if self.static_settings['dae'] == 'call pde' and solver_backend is None:
        try:
          import pypardiso
          solver_backend = 'pardiso'
          num_threads = environ.get('MKL_NUM_THREADS', None)
          if solver_subtype is None:
            solver_subtype = 'lu'
          if verbose >= 1:
            if num_threads is not None:
              print(f"Linear solver: Pardiso({solver_subtype}); {num_threads} threads.")
            else:
              print(f"Linear solver: Pardiso({solver_subtype}).")
        except ImportError:
          if verbose >= 0:
            print("Pardiso solver not available, falling back to scipy sparse solver.")
          solver_backend = 'scipy'
          if solver_subtype is None:
            solver_subtype = 'lapack'
      elif solver_backend is None:
        solver_backend = 'dense'
      if solver_subtype is None:
        solver_subtype = 'lu' if solver_backend == 'pardiso' else 'lapack' if solver_backend == 'scipy' else solver_subtype

      # Custom tangent via assembling for sparse problems
      tangent_fun = None

      if solver_backend == 'pardiso':
        def lin_solve_fun(mat, rhs, solve_data):
          free_dofs, _ = _split_solve_data(solve_data)
          mat = _prepare_mat_for_external(mat, solve_data)
          callback_fun = lambda mat_, rhs_, free_dofs_: solver.linear_solve_pardiso(
              mat_, rhs_, solver=solver_subtype, verbose=verbose, free_dofs=free_dofs_)
          return jax.pure_callback(callback_fun,
                                   jnp.zeros(rhs.shape, rhs.dtype),
                                   mat,
                                   rhs,
                                   free_dofs,
                                   vmap_method='sequential')
      elif solver_backend == 'dense':
        def lin_solve_fun(mat, rhs, free_dofs):
          if free_dofs is None:
            lufac = jax.scipy.linalg.lu_factor(mat)
            return jax.scipy.linalg.lu_solve(lufac, rhs)

          free_f = free_dofs.astype(mat.dtype)
          I = jnp.eye(mat.shape[0], dtype=mat.dtype)

          mat_mod = mat * free_f[:, None] * free_f[None, :] + I * (1.0 - free_f)[:, None]
          rhs_mod = jnp.where(free_dofs, rhs, jnp.zeros_like(rhs))

          lufac = jax.scipy.linalg.lu_factor(mat_mod)
          return jax.scipy.linalg.lu_solve(lufac, rhs_mod)

      elif solver_backend == 'scipy':
        def lin_solve_fun(mat, rhs, solve_data):
          free_dofs, _ = _split_solve_data(solve_data)
          mat = _prepare_mat_for_external(mat, solve_data)
          callback_fun = lambda mat_, rhs_, free_dofs_: solver.linear_solve_scipy(
              mat_, rhs_, free_dofs=free_dofs_, solver=solver_subtype, verbose=verbose)
          return jax.pure_callback(callback_fun,
                                   jnp.zeros(rhs.shape, rhs.dtype),
                                   mat,
                                   rhs,
                                   free_dofs,
                                   vmap_method='sequential')
      elif solver_backend == 'petsc':
        n_fields = self.static_settings["number of fields"]
        pc_type = self.static_settings["type of preconditioner"]

        def lin_solve_fun(mat, rhs, solve_data):
          free_dofs, _ = _split_solve_data(solve_data)
          mat = _prepare_mat_for_external(mat, solve_data)

          callback_fun = lambda mat_, rhs_, free_: solver.linear_solve_petsc(
              mat_,
              rhs_,
              n_fields,
              solver_subtype,
              pc_type,
              verbose,
              free_dofs=free_,
          )
          return jax.pure_callback(
              callback_fun,
              jnp.zeros(rhs.shape, rhs.dtype),
              mat,
              rhs,
              free_dofs,
              vmap_method='sequential',
          )
      elif solver_backend == 'pyamg':
        pc_type = self.static_settings["type of preconditioner"]

        def lin_solve_fun(mat, rhs, solve_data):
          free_dofs, _ = _split_solve_data(solve_data)
          mat = _prepare_mat_for_external(mat, solve_data)

          callback_fun = lambda mat_, rhs_, free_: solver.linear_solve_pyamg(
              mat_,
              rhs_,
              solver_subtype,
              pc_type,
              verbose,
              free_dofs=free_,
          )
          return jax.pure_callback(
              callback_fun,
              jnp.zeros(rhs.shape, rhs.dtype),
              mat,
              rhs,
              free_dofs,
              vmap_method='sequential',
          )
      elif solver_backend == 'dense_cholesky':  
        if dirichlet_dofs is None:
          def lin_solve_fun(mat, rhs, free_dofs):
            chofac = jax.scipy.linalg.cho_factor(mat)
            return jax.scipy.linalg.cho_solve(chofac, rhs)
        else:
          def lin_solve_fun(mat, rhs, free_dofs):
            free_f = free_dofs.astype(mat.dtype)
            I = jnp.eye(mat.shape[0], dtype=mat.dtype)

            mat_mod = mat * free_f[:, None] * free_f[None, :] + I * (1.0 - free_f)[:, None]
            rhs_mod = jnp.where(free_dofs, rhs, jnp.zeros_like(rhs))

            chofac = jax.scipy.linalg.cho_factor(mat_mod)
            return jax.scipy.linalg.cho_solve(chofac, rhs_mod)
      else:
        raise ValueError(f"Unknown solver backend: {solver_backend}")

      # @jax.jit
      def residual_fun_flat(x, settings, q_stages, q_n, q_t_n, current_stages, t_stage, dt, template, dirichlet_conditions, dirichlet_dofs):
        # print("Traced residual")
        q_stages_current = {key: q_stages[key] for key in template.keys()}
        x = jnp.reshape(x, (num_coupled_stages, -1))

        def inner_body(s, q_sc):
          q_s = reshape_as(x[s], template)
          for key in template.keys():
            q_sc[key] = q_sc[key].at[current_stages[s]].set(q_s[key])
          return q_sc

        if num_coupled_stages == 1:
          q_stages_current = inner_body(0, q_stages_current)
        else:
          q_stages_current = jax.lax.fori_loop(0, num_coupled_stages, inner_body, q_stages_current)

        q_ts_all = {
            key: self.integrators[key].value_and_derivatives(q_stages_current[key], q_n[key], q_t_n[key], dt)
            for key in template.keys()
        }

        def _residual_fun(s, t):
          q_ts = {key: _stage_tuple(q_ts_all[key], s, q_stages_current[key].shape) for key in template.keys()}

          # print("shape q_stage:", {key: q_stages_current[key][s].shape for key in q_stages_current})
          # print("shape q_n:", {key: (q_n[key].shape) for key in q_n})
          # print("shape q_t_n:", {key: (q_t_n[key].shape) for key in q_t_n})
          # print("shape q_ts:", {key: (q_ts[key][0].shape, [arr.shape for arr in q_ts[key][1:]]) for key in q_ts})

          def diffable_q_fun(t):
            return {key: discrete_value_with_derivatives(t, q_ts[key][0], q_ts[key][1:]) for key in template.keys()}

          if self.static_settings['dae'] == 'call pde':
            q_fun_state = assembler._q_fun_state(q_ts=q_ts)
            residual = assembler.assemble_residual(q_fun_state, settings, self.static_settings)
            # Optional external nodal loads (e.g. point/Neumann BCs on the
            # endpoints of a 1D field), added directly to the assembled residual.
            nodal_fun = self.static_settings.get('nodal residual fun', None)
            if nodal_fun is not None:
              residual = nodal_fun(residual, settings, t)
            return dict_flatten(residual)

          else:
            return self.static_settings['dae'](diffable_q_fun, t, settings)


        if current_stages.shape[0] == 1:
          return _residual_fun(current_stages[0], t_stage[0]).flatten()
        else:
          residual_fun_vmap = jax.vmap(lambda s, t: _residual_fun(s, t), (0, 0))
          return residual_fun_vmap(current_stages, t_stage).flatten()

      names_by_set = tuple(self.static_settings.get("internal variable names", tuple()))
      has_internal_variables = any(len(tuple(names)) > 0 for names in names_by_set)
      local_solvers = self.static_settings.get("local subsystem solvers", None)
      has_local_subsystems = local_solvers is not None and len(local_solvers) > 0

      def pre_iteration_update(x, settings):
        if self.static_settings['dae'] != 'call pde' or not has_local_subsystems:
          return settings
        if num_coupled_stages != 1:
          return settings

        q_stages_current = {key: q_stages[key] for key in template.keys()}
        x = jnp.reshape(x, (num_coupled_stages, -1))
        q_s = reshape_as(x[0], template)
        for key in template.keys():
          q_stages_current[key] = q_stages_current[key].at[current_stages[0]].set(q_s[key])

        q_ts = {}
        for key in template.keys():
          q_stage = q_stages_current[key][current_stages[0]]
          q_ts[key] = self.integrators[key].value_and_derivatives(q_stage, q_n[key], q_t_n[key], dt)

        settings_updated = dict(settings)
        current_blocks = list(settings_updated["internal variables"])
        q_fun_state = assembler._q_fun_state(q_ts=q_ts)
        for set_index, names in enumerate(names_by_set):
          if len(tuple(names)) == 0:
            continue
          current_blocks[set_index] = assembler.get_int_var_updates(
              q_fun_state, settings_updated, self.static_settings, set_index
          )
        settings_updated["internal variables"] = tuple(current_blocks)
        return settings_updated

      q_flat = dict_flatten(q)
      q_flat = jnp.tile(q_flat[None, ...], (num_coupled_stages, 1)).flatten()

      # Root solve
      if solver_backend in ['pardiso', 'scipy', 'pyamg', 'petsc']:
        assert self.static_settings['dae'] == 'call pde', "Only 'call pde' mode is supported for sparse solvers."

        # @jax.jit
        def tangent_fun(x, settings, q_stages, q_n, q_t_n, current_stages, t_stage, dt, template, dirichlet_conditions, dirichlet_dofs):
          return self._assemble_sparse_tangent(x, q_stages, q_n, q_t_n, dt, t_stage, current_stages, settings)

      elif solver_backend in ['dense', 'dense_cholesky']:
        def tangent_fun(x, settings, q_stages, q_n, q_t_n, current_stages, t_stage, dt, template, dirichlet_conditions, dirichlet_dofs):
          jacobian = jax.jacfwd(lambda y: residual_fun_flat(y, settings, q_stages, q_n, q_t_n, current_stages, t_stage, dt, template, dirichlet_conditions, dirichlet_dofs))(x)
          # jacobian = jax.jacrev(lambda y: residual_fun_flat(y, settings, q_stages, q_n, q_t_n, current_stages, t_stage, dt, template, dirichlet_conditions, dirichlet_dofs))(x)
          return jacobian
      else:
        raise ValueError(f"Unknown solver backend: {solver_backend}")

      use_template_solve_data = assembling_template is not None and solver_backend in ('pardiso', 'scipy', 'petsc', 'pyamg')
      solve_data = (free_dofs, assembling_template) if use_template_solve_data else free_dofs
      settings_for_solve = dict(settings)
      settings_for_solve["_linear solve data"] = solve_data
      root_solver_fun = getattr(self.root_solver, "func", self.root_solver)
      use_pre_iteration_update = (
          root_solver_fun is newton_solver
          and self.static_settings['dae'] == 'call pde'
          and has_internal_variables
          and has_local_subsystems
          and num_coupled_stages == 1
      )
      root_solver_extra = {"_pre_iteration_update": pre_iteration_update} if use_pre_iteration_update else {}
      use_solver_linearize_tangent = (
          root_solver_fun is newton_solver
          and solver_backend in ('dense', 'dense_cholesky')
      )

      if impl_diff_mode in ('forward', 'reverse', 'backward'):
        @implicit_diff.custom_root(
          residual_fun=residual_fun_flat,
          mat_fun=tangent_fun,
          solve=lin_solve_fun,
          free_dofs=free_dofs,
          has_aux=True,
          mode=impl_diff_mode)
        def root_solve(x0, settings, q_stages, q_n, q_t_n, current_stages, t_stage, dt, template, dirichlet_conditions, dirichlet_dofs):
          solver_tangent_fun = None
          if not use_solver_linearize_tangent:
            solver_tangent_fun = lambda x, s: tangent_fun(x, s, q_stages, q_n, q_t_n, current_stages, t_stage, dt, template, dirichlet_conditions, dirichlet_dofs)
          result = self.root_solver(
                # lambda x: residual_fun_flat(x, settings, q_stages, q_n, q_t_n, current_stages, t_stage, dt, template, dirichlet_conditions, dirichlet_dofs),
                lambda x, s: residual_fun_flat(x, s, q_stages, q_n, q_t_n, current_stages, t_stage, dt, template, dirichlet_conditions, dirichlet_dofs),
                x0,
                # tangent_fun=lambda x: tangent_fun(x, settings, q_stages, q_n, q_t_n, current_stages, t_stage, dt, template, dirichlet_conditions, dirichlet_dofs),
                tangent_fun=solver_tangent_fun,
                constrained_dofs=dirichlet_dofs,
                constrained_values=dirichlet_conditions,
                lin_solve_fun=lin_solve_fun,
                verbose=verbose,
                settings=settings,
                static_settings=self.static_settings,
                **root_solver_extra,
            )

          float_iterations = jnp.astype(result.num_iterations, float)
          float_conv = jnp.astype(result.converged, float)
          return result.root, (float_iterations, float_conv)

        q_root, (float_iterations, float_converged) = root_solve(jax.lax.stop_gradient(q_flat), settings_for_solve, q_stages, q_n, q_t_n, current_stages, t_stage, dt, template, dirichlet_conditions, dirichlet_dofs)

      else:
        # Direct call of root solver; derivative through iterations
        def root_solve(x0, settings, q_stages, q_n, q_t_n, current_stages, t_stage, dt, template, dirichlet_conditions, dirichlet_dofs):
          solver_tangent_fun = None
          if not use_solver_linearize_tangent:
            solver_tangent_fun = lambda x, s: tangent_fun(x, s, q_stages, q_n, q_t_n, current_stages, t_stage, dt, template, dirichlet_conditions, dirichlet_dofs)
          result = self.root_solver(
                # lambda x: residual_fun_flat(x, settings, q_stages, q_n, q_t_n, current_stages, t_stage, dt, template, dirichlet_conditions, dirichlet_dofs),
                lambda x, s: residual_fun_flat(x, s, q_stages, q_n, q_t_n, current_stages, t_stage, dt, template, dirichlet_conditions, dirichlet_dofs),
                x0,
                # tangent_fun=lambda x: tangent_fun(x, settings, q_stages, q_n, q_t_n, current_stages, t_stage, dt, template, dirichlet_conditions, dirichlet_dofs),
                tangent_fun=solver_tangent_fun,
                constrained_dofs=dirichlet_dofs,
                constrained_values=dirichlet_conditions,
                lin_solve_fun=lin_solve_fun,
                verbose=verbose,
                settings=settings,
                static_settings=self.static_settings,
                **root_solver_extra,
            )

          float_iterations = jnp.astype(result.num_iterations, float)
          float_conv = jnp.astype(result.converged, float)
          return result.root, (float_iterations, float_conv)

        q_root, (float_iterations, float_converged) = root_solve(jax.lax.stop_gradient(q_flat), settings_for_solve, q_stages, q_n, q_t_n, current_stages, t_stage, dt, template, dirichlet_conditions, dirichlet_dofs)


      num_iterations = jnp.maximum(jnp.astype(float_iterations, jnp.int32), num_iterations)
      converged = jnp.logical_and(jnp.astype(float_converged, jnp.bool), converged)
      q_root = jnp.reshape(q_root, (num_coupled_stages, -1))

      def update_body(s, q_stg):
        q_s = reshape_as(q_root[s], q)
        for key in template.keys():
          q_stg[key] = q_stg[key].at[current_stages[s]].set(q_s[key])
        return q_stg

      if num_coupled_stages == 1:
        q_stages = update_body(0, q_stages)
      else:
        q_stages = jax.lax.fori_loop(0, num_coupled_stages, update_body, q_stages)
      return (q_stages, num_iterations, converged, settings)

    if num_blocks == 1:
      q_stages, num_iterations, converged, settings = block_body(0, state_init)
    else:
      q_stages, num_iterations, converged, settings = jax.lax.fori_loop(0, num_blocks, block_body, state_init)

    # Call integrators for updating the solution and derivatives
    q_n1 = {}
    q_t_n1 = {}
    for key in template.keys():
      q_n1[key], q_t_n1[key] = self.integrators[key].update(q_stages[key], q_n[key], q_t_n[key], dt)
      q_t_n1[key] = _canonical_derivatives(q_t_n1[key], q_n1[key], self.num_time_derivs[key])
    error_estimate = {
        key: self.integrators[key]._error_estimate(q_stages[key], q_n[key], q_t_n[key], dt) for key in template.keys()
    }
    return q_n1, q_t_n1, error_estimate, num_iterations, converged, settings

  def _build_assembling_template(self, initial_guess, settings):
      return assembler._build_assembling_template_from_connectivity(
          initial_guess=initial_guess,
          settings=settings,
          static_settings=self.static_settings,
      )

  def initialize(self, dofs, settings):
    """
    Constructs a template for efficient assembling (not jittable). 
    """

    if self.static_settings['dae'] == 'call pde':

      # Template for reshape_as / indices etc.
      self._global_template = dofs

      # Only needed if you actually use sparse assembly + external solver

      solver_backend = self.static_settings.get("solver backend", None)
      solver_subtype = self.static_settings.get("solver", None)
      if solver_backend is None:
        # check wether pypardiso is installed, if yes use 'pardiso', otherwise 'scipy'
        try:
          import pypardiso
          solver_backend = 'pardiso'
          if solver_subtype is None:
            solver_subtype = 'lu'
        except ImportError:
          solver_backend = 'scipy'
          if solver_subtype is None:
            solver_subtype = 'lapack'
      elif solver_backend is None:
        solver_backend = 'dense'
      needs_sparse = solver_backend in ("pardiso", "scipy", "petsc", "pyamg")

      has_template = ("assembling template" in settings)
      has_kernel = (self.static_settings.get("assembling kernel", None) is not None)

      if needs_sparse and (not has_template or not has_kernel):
          settings, new_static = self._build_assembling_template(
              initial_guess=dofs,
              settings=settings,
          )
          # Important: write FrozenDict update back to the manager
          self.static_settings = new_static

    return settings

  def add_dirichlet_condition(self, field, on_boundary_fun, value_fun, index=None, reset=False):
    """
    Register a Dirichlet BC for `field` (not jittable).

    Required signatures:
      on_boundary_fun(x, t) -> bool
      value_fun(x, t) -> scalar per node OR full value per node

    For vector/tensor fields:
      - scalar value_fun: requires index (int for vectors, tuple for tensors)
      - full value_fun: returns shape (field_shape,) per node
    """

    # Collect BC specs (kept in static_settings so they are static wrt. JIT)
    old_specs = tuple(self.static_settings.get("dirichlet bc specs", ()))
    spec = FrozenDict({
      "field": field,
      "on_boundary_fun": on_boundary_fun,
      "value_fun": value_fun,
      "index": index,
      "reset": reset,
    })

    new_specs = old_specs + (spec,)

    def apply_dirichlet_bcs(t, settings):
      # --- small helpers (kept local to keep surface small) ---
      def _field_shape(dim):
        if dim == 1 or dim is None:
          return ()
        if isinstance(dim, int):
          return (dim,)
        if isinstance(dim, tuple):
          return dim
        raise TypeError(f"static_settings['fields'][..] must be int or tuple, got {type(dim)}")

      # --- ensure required base data ---
      if self.static_settings.get("field dimensions") is None:
        raise ValueError("static_settings must contain a 'fields'-FrozenDict.")
      if settings.get("node coordinates") is None:
        raise ValueError("settings must contain 'node coordinates'.")

      # --- ensure storage exists (per-field node count and shape) ---
      if settings.get("dirichlet dofs") is None:
        dd = {}
        for f, dim in self.static_settings["field dimensions"].items():
          coor_f = settings["node coordinates"]#[f]
          n_f = coor_f.shape[0]
          fshape = _field_shape(dim)
          dd[f] = jnp.zeros((n_f,), dtype=bool) if fshape == () else jnp.zeros((n_f, *fshape), dtype=bool)
        settings["dirichlet dofs"] = dd

      if settings.get("dirichlet conditions") is None:
        settings["dirichlet conditions"] = dict_zeros_like(settings["dirichlet dofs"], dtype=float)

      # --- apply all registered specs ---
      for spec in new_specs:
        f = spec["field"]
        coor = settings["node coordinates"]#[f]
        n_nodes = coor.shape[0]
        fshape = _field_shape(self.static_settings["field dimensions"][f])

        mask = jax.vmap(lambda x: spec["on_boundary_fun"](x, t))(coor).astype(bool)
        vals = jax.vmap(lambda x: spec["value_fun"](x, t))(coor)

        dofs_f = settings["dirichlet dofs"][f]
        cond_f = settings["dirichlet conditions"][f]

        if spec.get("reset", False):
          dofs_f = jnp.zeros_like(dofs_f)
          cond_f = jnp.zeros_like(cond_f)

        # scalar field
        if fshape == ():
          if vals.ndim != 1:
            raise ValueError(f"value_fun must return a scalar per node for scalar field '{f}'.")
          dofs_f = jnp.where(mask, True, dofs_f)
          cond_f = jnp.where(mask, vals, cond_f)

        # vector/tensor field
        else:
          k = len(fshape)
          # node_ids = jnp.where(mask)[0]

          # scalar per node -> set one component
          if vals.ndim == 1:
            idx = spec.get("index", None)
            if idx is None:
              raise ValueError(f"Scalar value_fun for non-scalar field '{f}' requires `index`.")
            idx_t = (idx,) if isinstance(idx, int) else idx
            if not isinstance(idx_t, tuple) or len(idx_t) != k:
              raise ValueError(f"`index` for field '{f}' must be int or tuple of length {k}, got {idx}.")
            for ax, (ii, dim) in enumerate(zip(idx_t, fshape)):
              if (not isinstance(ii, int)) or ii < 0 or ii >= dim:
                raise ValueError(f"index[{ax}]={ii} out of range [0, {dim}) for field '{f}'.")

            sl = (slice(None), *idx_t)         # (all nodes, selected component)
            dofs_comp = dofs_f[sl]             # shape: (n_nodes,)
            cond_comp = cond_f[sl]             # shape: (n_nodes,)

            dofs_comp = jnp.where(mask, True, dofs_comp)
            cond_comp = jnp.where(mask, vals, cond_comp)

            dofs_f = dofs_f.at[sl].set(dofs_comp)
            cond_f = cond_f.at[sl].set(cond_comp)

          # full value per node -> set all components
          else:
            # expected per-node shape: fshape -> vals shape (n_nodes, *fshape)
            if vals.shape[0] != n_nodes or tuple(vals.shape[1:]) != fshape:
              raise ValueError(f"value_fun for '{f}' must return shape (n, {fshape}), got {vals.shape}.")

            mask_nd = mask.reshape((n_nodes,) + (1,) * k)
            dofs_f = jnp.where(mask_nd, True, dofs_f)
            cond_f = jnp.where(mask_nd, vals, cond_f)

        settings["dirichlet dofs"][f] = dofs_f
        settings["dirichlet conditions"][f] = cond_f

      return settings

    # store function + specs back into static_settings
    if isinstance(self.static_settings, FrozenDict):
      self.static_settings = self.static_settings.copy(add_or_replace={
        "dirichlet bc specs": new_specs,
        "apply dirichlet bcs": apply_dirichlet_bcs,
      })
    else:
      self.static_settings["dirichlet bc specs"] = new_specs
      self.static_settings["apply dirichlet bcs"] = apply_dirichlet_bcs

  @jit_with_docstring()
  def _global_residual(self, dofs, dt0, settings={'current time': 0.0}):
    """
    Not tested, only for performance checks... Computes the global residual for the current state (stage 0).

    Semantics:
      - Builds q_n / q_der_n like at the beginning of `run` (repeat current dofs, zero derivatives).
      - Applies Dirichlet BCs (if registered) and pre_step_updates at t = settings['current time'].
      - Evaluates the assembled residual at stage 0.
      - If Dirichlet dofs are present: replaces constrained residual entries by (x - bc),
        consistent with the row-identity modification used in dense solves.

    Returns:
      jnp.ndarray: flat residual vector (global ordering = dict_flatten ordering).
    """
    # Keep the same convention as run()
    assert list(dofs.keys()) == sorted(dofs.keys()), "The keys of the DOFs must be alphabetically sorted."

    # Local copy to avoid unintended external side effects
    settings = dict(settings)
    t = settings.get('current time', 0.0)
    settings['current time'] = t

    # Apply BCs and user updates in the same order as run()
    apply_bcs = self.static_settings.get("apply dirichlet bcs", None)
    if apply_bcs is not None:
      settings = apply_bcs(t, settings)
    if self.pre_step_updates is not None:
      settings = self.pre_step_updates(t, settings)

    # Build history like at initialization (and set global template)
    q_n, q_der_n = self._initialize(dofs)

    # Stage-0 evaluation setup
    num_stages = next(iter(self.integrators.values())).num_stages
    stage_positions = next(iter(self.integrators.values())).stage_positions
    current_stages = jnp.array([0], dtype=jnp.int32)

    # Choose t_n such that t_stage(stage 0) == current time t (robust even if c0 != 0)
    t_n = t - dt0 * stage_positions[current_stages][0]
    t_stage = t_n + dt0 * stage_positions[current_stages]

    template = jax.lax.stop_gradient(dofs)
    q_stages = {k: jnp.repeat(dofs[k][None, ...], num_stages, axis=0) for k in template.keys()}
    x_flat = dict_flatten(dofs)

    def residual_fun_flat(x):
      # x is flat stage unknowns for the coupled stages (here: 1 stage)
      x = jnp.reshape(x, (1, -1))

      # Update stage 0 with x
      q_stages_current = {k: q_stages[k] for k in template.keys()}
      q_s = reshape_as(x[0], template)
      for k in template.keys():
        q_stages_current[k] = q_stages_current[k].at[current_stages[0]].set(q_s[k])

      # Build discrete values + derivatives as required by the dae/assembler
      q_ts = {}
      for k in template.keys():
        q_stage = q_stages_current[k][current_stages[0]]
        tup = self.integrators[k].value_and_derivatives(q_stage, q_n[k], q_der_n[k], dt0)
        q_ts[k] = _stage_tuple(tup, current_stages[0], (num_stages, *q_stage.shape))

      def diffable_q_fun(tt):
        return {
          k: discrete_value_with_derivatives(tt, q_ts[k][0], q_ts[k][1:])
          for k in template.keys()
        }

      if self.static_settings['dae'] == 'call pde':
        q_fun_state = assembler._q_fun_state(q_ts=q_ts)
        return dict_flatten(assembler.assemble_residual(q_fun_state, settings, self.static_settings))

      else:
        res = self.static_settings['dae'](diffable_q_fun, t_stage[0], settings)
        return dict_flatten(res) if isinstance(res, dict) else res

    r = residual_fun_flat(x_flat).reshape(-1)
    return r

  @jit_with_docstring()
  def _global_tangent(self, dofs, dt0, settings={'current time': 0.0}):
    """
    Not tested, only for performance checks... Computes the global tangent (Jacobian) for the current state (stage 0).

    Semantics:
      - Same state preparation as global_residual().
      - If sparse backend (pardiso/scipy/petsc/pyamg) and dae == 'call pde': uses _assemble_sparse_tangent().
      - Else: uses jax.jacfwd(global_residual wrt x_flat) to obtain a dense Jacobian.
      - If assembling template/kernel exist, consolidates sparse tangent to unique NNZ.
      - If Dirichlet dofs exist and tangent is dense: replaces constrained rows by identity.

    Returns:
      - jnp.ndarray for dense backends
      - sparse.BCOO for sparse assembling backends
    """
    # Keep the same convention as run()
    assert list(dofs.keys()) == sorted(dofs.keys()), "The keys of the DOFs must be alphabetically sorted."

    settings = dict(settings)
    t = settings.get('current time', 0.0)
    settings['current time'] = t

    apply_bcs = self.static_settings.get("apply dirichlet bcs", None)
    if apply_bcs is not None:
      settings = apply_bcs(t, settings)
    if self.pre_step_updates is not None:
      settings = self.pre_step_updates(t, settings)

    q_n, q_der_n = self._initialize(dofs)

    num_stages = next(iter(self.integrators.values())).num_stages
    stage_positions = next(iter(self.integrators.values())).stage_positions
    current_stages = jnp.array([0], dtype=jnp.int32)

    t_n = t - dt0 * stage_positions[current_stages][0]
    t_stage = t_n + dt0 * stage_positions[current_stages]

    template = jax.lax.stop_gradient(dofs)
    q_stages = {k: jnp.repeat(dofs[k][None, ...], num_stages, axis=0) for k in template.keys()}
    x_flat = dict_flatten(dofs)

    solver_backend = self.static_settings.get('solver backend', None)
    solver_subtype = self.static_settings.get('solver', None)
    if solver_backend is None:
      # check wether pypardiso is installed, if yes use 'pardiso', otherwise 'scipy'
      try:
        import pypardiso
        solver_backend = 'pardiso'
        if solver_subtype is None:
          solver_subtype = 'lu'
      except ImportError:
        solver_backend = 'scipy'
        if solver_subtype is None:
          solver_subtype = 'lapack'
    elif solver_backend is None:
      solver_backend = 'dense'
      
    # Compute tangent: sparse assembly vs dense autodiff
    if (solver_backend in ('pardiso', 'scipy', 'petsc', 'pyamg')) and (self.static_settings['dae'] == 'call pde'):
      K = self._assemble_sparse_tangent(
        x_flat, q_stages, q_n, q_der_n, dt0, t_stage, current_stages, settings
      )

      # # Optional: consolidate duplicates to unique NNZ if template/kernel are present
      # assembling_kernel = self.static_settings.get("assembling kernel", None)
      # assembling_template = settings.get("assembling template", None)
      # if assembling_kernel is not None and assembling_template is not None:
      #   # Ensure BCOO
      #   K_bcoo = K if isinstance(K, sparse.BCOO) else sparse.bcoo_fromdense(K)
      #   assembled_values = assembling_kernel(K_bcoo.data, assembling_template.scatter)
      #   K = sparse.BCOO(
      #     (assembled_values, assembling_template.indices_unique),
      #     shape=K_bcoo.shape
      #   )

      return K

    else:
      assert solver_backend in ('dense', 'dense_cholesky', None), f"Unknown solver backend: {solver_backend}"

  @jit_with_docstring(static_argnames=['num_time_steps'])
  def run(self, dofs, dt0, t_final, num_time_steps: int, settings = {'current time': 0.0}):
    """
    Executes the time stepping loop for the simulation.

    This method performs the following operations:
      1. Verifies that all time integrators are compatible (i.e., they have the same number of stages,
         identical stage positions, and stage lists).
      2. Initializes simulation variables including the initial DOFs, time (t), step size (dt), and history state.
      3. Iteratively performs time steps using a multi-stage integration method:
           - Updates state with pre-step modifications.
           - Computes the new state and derivative estimates using the multi-stage step procedure.
           - Estimates the error and uses the step size controller to adjust dt.
           - Accepts or rejects the time step based on convergence criteria.
           - Optionally saves the current state using the save policy.
           - Performs post-step updates to settings.
      4. Continues the loop until the simulation time reaches t_final or the maximum number of time steps is reached.
      5. Finalizes and returns the simulation state along with statistics such as the number of accepted
         and rejected steps.

    Args:
        dofs (dict[str, jnp.ndarray]): Initial degrees of freedom for the simulation.
        dt0 (float): Initial time step size.
        t_final (float): Absolute final simulation time.
        num_time_steps (int): Maximum number of time steps to perform.
        settings (dict[str, Any]): Dictionary containing dynamic simulation settings. Default is {'current time': 0.0}.

    Returns:
        TimeSteppingManagerState: An object containing the final state (q), updated settings, simulation history, and step statistics (total steps, accepted steps, rejected steps).
    """

    # Check whether time integrators are compatible
    assert all(integrator.num_stages == next(iter(self.integrators.values())).num_stages for integrator in self.integrators.values()),\
      "Number of stages must be the same for all fields."

    if self.verbose >= 2:
      assert all(np.allclose(integrator.stage_positions, next(iter(self.integrators.values())).stage_positions) for integrator in self.integrators.values()),\
        "Stage positions must be the same for all fields."

    # Alphabetic keywords
    assert list(dofs.keys()) == sorted(dofs.keys()), "The keys of the DOFs must be alphabetically sorted."

    # Some initializations
    dt = settings.get(_RESTART_DT_KEY, dt0)
    t = settings.get('current time', 0.0)
    t_start = t
    q = dofs
    q_n, q_der_n = self._initialize(q, settings)
    controler_state = settings.get(_RESTART_CONTROLLER_STATE_KEY, None)
    if controler_state is None:
      controler_state = self.step_size_controller.initialize(0.0)

    def _make_q_fun_state(q, q_der):
      def _q_tup(key):
        return (q[key],) + _derivative_tuple(q_der[key], q[key], self.num_time_derivs[key])

      return assembler._q_fun_state(q_ts={
          key: _q_tup(key)
          for key in q.keys()
      })

    @jax.jit
    def _postprocessing_fun(t, q, q_der, settings):
      return self.postprocessing_fun(_make_q_fun_state(q, q_der), t, settings)
    q_der = {key: q_der_n[key][0] for key in q_der_n.keys()}
    user_data = _postprocessing_fun(t, q, q_der, settings)

    history_state = (self.save_policy.initialize(q, t_start, t_final, num_time_steps, user_data)
                     if self.save_policy is not None else None)
    
    @jax.jit
    def _save_step(history_state, t, q, user_data):
      return self.save_policy.save_step(history_state, t, q, user_data) \
                      if self.save_policy is not None else history_state
    history_state = _save_step(history_state, t, q, user_data)

    # Prepare the function for one time step
    def loop_body(step, state):
      def step_fun(state):
        t, t_n, dt, _, q_n, q_der_n, history_state, controler_state, num_accepted, num_rejected, settings, last_printed, t_final = state
        t = t_n + dt
        t = jnp.minimum(t, t_final)
        dt = t - t_n
        q = {key: q_n[key][0] for key in q_n.keys()}

        settings['current time'] = t

        # Perform one multi-stage time step
        q, q_der, error_estimate, num_iterations, converged, settings = self._multi_stage_step(
            q, t, t_n, dt, q_n, q_der_n, settings)

        # Call time step controler
        order = min([integrator.order for integrator in self.integrators.values()])
        controler_state = self.step_size_controller.compute_scaler(
          error_estimate, q, q_n, controler_state, order, converged, num_iterations, dt, self.verbose
          )
        controler_state = self.step_size_controller.check_accept(controler_state, converged, self.verbose)

        # Here the step size controling logic is not taken into account for the derivatives
        # Todo: check whether integrators support changing the step size
        controler_state = jax.lax.stop_gradient(controler_state)
        dt_scaler = controler_state.step_scaler
        accept = controler_state.accept
        interrupt = controler_state.interrupt
        accept = jnp.logical_and(accept, jnp.logical_not(interrupt))

        def do_accept(x):
          # Update history data and user-defined postprocessing data and adjust step size

          (history_state, q, q_der, q_n, q_der_n, t, t_n, dt, dt_scaler, num_a, num_r, settings) = x
          settings["current time"] = t
          for key in q.keys():
            q_n[key] = jnp.roll(q_n[key], shift=1, axis=0)
            q_n[key] = q_n[key].at[0].set(q[key])
            q_der_n[key] = jnp.roll(q_der_n[key], shift=1, axis=0)
            q_der_n[key] = q_der_n[key].at[0].set(q_der[key])

          user_data = _postprocessing_fun(t, q, q_der, settings)
          settings = self.post_step_updates(_make_q_fun_state(q, q_der), t, settings)
          history_state = _save_step(history_state, t, q, user_data)

          dt = dt_scaler * dt
          t_n = t

          return history_state, q_n, q_der_n, t, t_n, dt, num_a + 1, num_r, settings

        def do_reject(x):
          # Return to old step and reduce step size
          (history_state, _, _q_der, q_n, q_der_n, t, t_n, dt, dt_scaler, num_a, num_r, settings) = x
          t = t - dt
          dt = dt_scaler * dt
          if "internal variables n" in settings:
            settings = dict(settings)
            settings["internal variables"] = settings["internal variables n"]
          settings["current time"] = t_n
          return (history_state, q_n, q_der_n, t, t_n, dt, num_a, num_r + 1, settings)

        history_state, q_n, q_der_n, t, t_n, dt, num_accepted, num_rejected, settings = jax.lax.cond(accept, do_accept, do_reject,
            (history_state, q, q_der, q_n, q_der_n, t, t_n, dt, dt_scaler, num_accepted, num_rejected, settings))

        # # debug accept/reject logic
        # jax.debug.print("Accept: {x}", x=accept, ordered=True)
        # jax.debug.print("dt: {x}", x=dt, ordered=True)
        # jax.debug.print("t: {x}", x=t, ordered=True)
        # jax.debug.print("t_n: {x}", x=t_n, ordered=True)
        # jax.debug.print("q: {x}", x=q, ordered=True)
        # jax.debug.print("q_n: {x}", x=q_n, ordered=True)

        if self.verbose >= 0:
          progress = (100 * t / t_final).astype(int)
          should_print = jnp.greater_equal(progress - last_printed, 5)

          def print_and_update(args):
            _progress, _t, _dt, _num_iterations, _last_printed = args
            jax.debug.print("Progress: {a}%, Time: {b:.2e}, dt: {c:.2e}, iterations: {e}",
                            a=_progress,
                            b=_t,
                            c=_dt,
                            e=_num_iterations,
                            ordered=True)
            return _progress

          last_printed = jax.lax.cond(should_print, print_and_update, lambda args: args[4],
                                      (progress, t, dt, num_iterations, last_printed))

          if self.verbose >= 1:
            jax.debug.print(" ", ordered=True)

        return (t, t_n, dt, q, q_n, q_der_n, history_state, controler_state, num_accepted, num_rejected, settings, last_printed, t_final)

      def do_nothing(state):
        return state

      t = state[0]
      t_final = state[-1]
      interrupt = state[7].interrupt
      state = jax.lax.cond(jnp.logical_and(t < t_final - 1e-12, jnp.logical_not(interrupt)), step_fun, do_nothing, state)

      # In case of interruption, set all values in q to nan
      interrupt = state[7].interrupt
      state = (*state[:3],
               jax.lax.cond(interrupt, lambda x: jax.lax.stop_gradient(jax.tree.map(lambda a: jnp.full_like(a, jnp.nan), x)), lambda x: x, state[3]),
               *state[4:])

      return state

    # Run time stepping loop
    num_accepted = 0
    num_rejected = 0
    initial_state = (t, t, dt, q, q_n, q_der_n, history_state, controler_state, num_accepted, num_rejected, settings, -2, t_final)
    if num_time_steps == 1:
      final_state = loop_body(0, initial_state)
    elif self.static_settings.get("checkpointing", None) is None:
      final_state = jax.lax.fori_loop(0, num_time_steps, loop_body, initial_state)
    elif self.static_settings.get("checkpointing", None) == "chunking":
      checkpoints = self.static_settings.get("checkpoints", None)
      if checkpoints is None:
        checkpoints = ceil(sqrt(num_time_steps))
      checkpoints = min(int(checkpoints), num_time_steps)
      if checkpoints < 1:
        raise ValueError("static_settings['checkpoints'] must be a positive integer or None.")
      chunk_size = ceil(num_time_steps / checkpoints)

      def chunk_body(chunk, state):
        chunk_start = chunk * chunk_size

        def local_body(local_step, inner_state):
          step = chunk_start + local_step
          return jax.lax.cond(
              step < num_time_steps,
              lambda s: loop_body(step, s),
              lambda s: s,
              inner_state,
          )

        return jax.lax.fori_loop(0, chunk_size, local_body, state)

      final_state = jax.lax.fori_loop(0, checkpoints, jax.checkpoint(chunk_body), initial_state)
    else:
      raise ValueError("Unknown static_settings['checkpointing']; use None or 'chunking'.")
    t, _, dt, q, q_n, q_der_n, history_state, controler_state, num_accepted, num_rejected, settings, _, _ = final_state

    if self.verbose >= 0:
      jax.lax.cond(jnp.isclose(t, t_final),
                  lambda _: None,
                  lambda _: jax.debug.print("Maximum number of steps reached before t_final!"),
                  operand=None)
    history_state = self.save_policy.finalize(history_state) if self.save_policy is not None else history_state

    return TimeSteppingManagerState(
        q=q,
        settings=settings,
        q_n=q_n,
        q_der_n=q_der_n,
        dt=dt,
        controller_state=controler_state,
        history=history_state,
        num_steps=num_accepted + num_rejected,
        num_accepted=num_accepted,
        num_rejected=num_rejected,
    )

# Register as pytree node in order to be able to jit the methods
tree_util.register_pytree_node(TimeSteppingManager, TimeSteppingManager._tree_flatten,
                               TimeSteppingManager._tree_unflatten)


def _make_local_subsystem_solver(
    local_dae: Callable,
    fields: tuple[str, ...],
    time_integrators: dict[str, TimeIntegrator],
    root_solver: Callable = newton_solver,
):
  solve_fields = tuple(sorted(fields))
  local_static_settings = {
      "dae": local_dae,
      "time integrators": {field: time_integrators[field] for field in solve_fields},
      "verbose": -1,
  }
  local_manager = TimeSteppingManager(local_static_settings, root_solver=root_solver)

  def solve_local(internal_vars, settings, inputs=None, *, elem_number, int_point_number, set):
    t_n = settings["last time"]
    t_np1 = settings["current time"]
    dt = t_np1 - t_n
    local_q0 = {field: internal_vars[field] for field in solve_fields}
    has_invalid_q0 = jnp.asarray(False)
    for field in solve_fields:
      has_invalid_q0 = jnp.logical_or(
          has_invalid_q0,
          jnp.any(jnp.logical_not(jnp.isfinite(local_q0[field]))),
      )
    local_settings = {
        "current time": t_n,
        "last time": t_n,
    }
    committed = settings["internal variables n"]
    committed = committed[set] if isinstance(committed, (tuple, list)) else committed
    local_q_n = {
        field: jnp.repeat(
            committed[field][elem_number, int_point_number][None, ...],
            local_manager.num_steps[field],
            axis=0,
        )
        for field in solve_fields
    }
    local_q_t_n = {
        field: jnp.zeros(
            (local_manager.num_steps[field], local_manager.num_time_derivs[field], *local_q0[field].shape),
            dtype=local_q0[field].dtype,
        )
        for field in solve_fields
    }
    if inputs is not None:
      local_settings.update(inputs)

    def solve(_):
      q_n1, _, _, _, converged, _ = local_manager._multi_stage_step(
          local_q0, t_np1, t_n, dt, local_q_n, local_q_t_n, local_settings
      )
      return {
          field: jnp.where(converged, q_n1[field], jnp.full_like(q_n1[field], jnp.nan))
          for field in fields
      }

    return jax.lax.cond(
        has_invalid_q0,
        lambda _: {field: jnp.full_like(local_q0[field], jnp.nan) for field in fields},
        solve,
        operand=None,
    )

  solve_local = jax.jit(solve_local, static_argnames=("set",))
  solve_local.fields = fields
  solve_local.manager = local_manager
  return solve_local
