# utility.py
# Copyright (C) 2024 Tobias Bode
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

"""
This module contains some useful functions, including:

- Wrapper to nearest neighbor algorithm (e.g. for moving least squares)
- Degree of freedom (DOF) selection for boundary conditions
- Compute condition number and check symmetry of tangent matrix
- Functions for manipulating arrays
"""

from functools import wraps

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict
from jax.experimental import sparse


def jit_with_docstring(static_argnames=None):
    """JIT wrapper that keeps the originial docstring of the function."""
    def decorator(fun):
        # Define the jitted function with optional static_argnames
        if static_argnames is not None:
            jitted_fun = jax.tree_util.Partial(
                jax.jit, static_argnames=static_argnames
            )(fun)
        else:
            jitted_fun = jax.jit(fun)

        # Create the wrapper function
        @wraps(fun)
        def wrapper(*args, **kwargs):
            return jitted_fun(*args, **kwargs)

        return wrapper

    return decorator


def dict_zeros_like(arr, **keyargs):
    """Wrapper around zeros_like, that works also for dicts with jnp.ndarray entries."""
    if isinstance(arr, dict):
        res = {}
        fields = list(arr.keys())
        for key in fields:
            res[key] = jnp.zeros_like(arr[key], **keyargs)
        return res
    else:
        return jnp.zeros_like(arr, **keyargs)


def dict_flatten(arr):
    """
    Recursively flattens a nested dict of arrays (np.ndarray or jnp.ndarray) to one flat array.
    If a single array is provided, it simply returns arr.flatten().

    Args:
        arr (dict or array): A nested dictionary of arrays (NumPy or JAX) or a single array.

    Returns:
        array: A single flat array containing all elements, with the same type as the input arrays.
    """
    if isinstance(arr, dict):
        flat_arrays = []
        # Sort keys for consistent ordering
        for key in arr.keys():
            flat_array = dict_flatten(arr[key])
            flat_arrays.append(flat_array)
        if not flat_arrays:
            return np.array([]) if isinstance(flat_array, np.ndarray) else jnp.array([])
        else:
            return (
                jnp.concatenate(flat_arrays)
                if isinstance(flat_arrays[0], jnp.ndarray)
                else np.concatenate(flat_arrays)
            )
    elif isinstance(arr, np.ndarray) or isinstance(arr, jnp.ndarray):
        return arr.flatten()
    else:
        raise TypeError("Input must be a nested dict of arrays or an array.")


def reshape_as(flat_array, signature_array):
    """
    Reshapes a flat array (np.ndarray or jnp.ndarray) into an array or dict of arrays matching the structure of signature_array.

    Args:
        flat_array (np.ndarray or jnp.ndarray): The flat array to be reshaped.
        signature_array (np.ndarray, jnp.ndarray, or dict): An array or dict of arrays whose shapes are used to reshape flat_array.

    Returns:
        Reshaped array or dict of arrays matching the shapes in signature_array.
    """
    if isinstance(signature_array, dict):
        reshaped_arrays = {}
        start = 0
        for key in sorted(signature_array.keys()):
            sig_value = signature_array[key]
            # Calculate the number of elements needed for this array
            size = (
                sig_value.size
                if isinstance(sig_value, (np.ndarray, jnp.ndarray))
                else 0
            )
            end = start + size
            # Extract the relevant slice from flat_array
            slice_flat = flat_array[start:end]
            if isinstance(sig_value, dict):
                # Recursive call for nested dictionaries
                reshaped_arrays[key] = reshape_as(slice_flat, sig_value)
                start = end  # Update start position after processing nested dict
            elif isinstance(sig_value, (np.ndarray, jnp.ndarray)):
                # Reshape the slice to match the shape of the signature array
                reshaped_array = slice_flat.reshape(sig_value.shape)
                # Convert to the same type as sig_value if necessary
                if isinstance(sig_value, np.ndarray) and not isinstance(
                    reshaped_array, np.ndarray
                ):
                    reshaped_array = np.array(reshaped_array)
                elif isinstance(sig_value, jnp.ndarray) and not isinstance(
                    reshaped_array, jnp.ndarray
                ):
                    reshaped_array = jnp.array(reshaped_array)
                reshaped_arrays[key] = reshaped_array
                start = end  # Update start position
            else:
                raise TypeError("Values in signature_array must be arrays or dicts.")
        # Check if all elements were used
        if start != flat_array.size:
            raise ValueError(
                "The size of flat_array does not match the total size of signature_array."
            )
        return reshaped_arrays
    elif isinstance(signature_array, (np.ndarray, jnp.ndarray)):
        # Reshape flat_array to match the shape of signature_array
        if flat_array.size != signature_array.size:
            raise ValueError(
                "The size of flat_array does not match the size of signature_array."
            )
        reshaped_array = flat_array.reshape(signature_array.shape)
        # Convert to the same type as signature_array if necessary
        if isinstance(signature_array, np.ndarray) and not isinstance(
            reshaped_array, np.ndarray
        ):
            reshaped_array = np.array(reshaped_array)
        elif isinstance(signature_array, jnp.ndarray) and not isinstance(
            reshaped_array, jnp.ndarray
        ):
            reshaped_array = jnp.array(reshaped_array)
        return reshaped_array
    else:
        raise TypeError("signature_array must be an array or a dict of arrays.")


def mask_set(array, selection, values):
    """
    Changes values in a JAX array or dict of arrays based on a boolean mask.

    Args:
      array (jnp.ndarray or dict): JAX array or dict of JAX arrays to update.
      selection (jnp.ndarray or dict): Boolean mask or dict of boolean masks.
      values (jnp.ndarray or dict): Values to insert, or dict of values.

    Returns:
      Updated array or dict of arrays.
    """
    if isinstance(array, dict):
        # Ensure selection and values are also dicts with the same keys
        if not isinstance(selection, dict) or not isinstance(values, dict):
            raise TypeError(
                "If 'array' is a dict, 'selection' and 'values' must also be dicts with the same keys."
            )
        if array.keys() != selection.keys() or array.keys() != values.keys():
            raise ValueError(
                "The keys of 'array', 'selection', and 'values' must match."
            )
        # Update each array in the dict
        return {key: mask_set(array[key], selection[key], values[key]) for key in array}
    else:
        # Flatten the array, selection mask, and values
        flat_array = array.flatten()
        flat_selection = selection.flatten()
        idx = jnp.arange(flat_array.shape[0])
        selected_indices = idx[flat_selection]

        # Flatten values
        flat_values = values.flatten()

        # Determine how to select values based on their size
        if flat_values.size == selected_indices.size:
            # Use values directly
            flat_array = flat_array.at[selected_indices].set(flat_values)
        elif flat_values.size == flat_array.size:
            # Use the same indices to select from values
            flat_array = flat_array.at[selected_indices].set(
                flat_values[selected_indices]
            )
        else:
            raise ValueError(
                "Values size must be either equal to the number of selected elements or the size of the array."
            )
        return flat_array.reshape(array.shape)


def mask_select(array, selection):
    """
    Selects elements from a JAX array or dictionary of arrays based on a boolean mask.

    Args:
        array (jnp.ndarray or dict): JAX array or dictionary of JAX arrays.
        selection (jnp.ndarray or dict): Boolean mask or dictionary of boolean masks.

    Returns:
        Selected elements as an array or dictionary of arrays.
    """
    if isinstance(array, dict):
        # Ensure that 'selection' is also a dictionary with the same keys
        if not isinstance(selection, dict):
            raise TypeError(
                "If 'array' is a dictionary, 'selection' must also be a dictionary with the same keys."
            )
        if array.keys() != selection.keys():
            raise ValueError("The keys of 'array' and 'selection' must match.")
        # Recursively apply 'mask_select' to each element in the dictionary
        return {key: mask_select(array[key], selection[key]) for key in array}
    else:
        # 'array' and 'selection' are JAX arrays
        # Ensure that the shapes match
        if array.shape != selection.shape:
            raise ValueError("The shape of 'array' and 'selection' must match.")
        # Select elements where the mask is True
        selected_elements = array[selection]
        return selected_elements


def mask_op(array, selection, values=None, mode="set", ufunc=None):
    """
    Performs an operation on a JAX array or dict of arrays based on a boolean mask.

    Args:
        array (jnp.ndarray or dict): JAX array or dict of JAX arrays to update.
        selection (jnp.ndarray or dict): Boolean mask or dict of boolean masks.
        values (jnp.ndarray or dict, optional): Values to use in the operation. Required for modes except 'get' and 'apply'.
        mode (str): Operation mode. One of 'set', 'add', 'multiply', 'divide', 'power', 'min', 'max', 'apply', 'get'.
        ufunc (callable, optional): A unary function to apply when mode is 'apply'.

    Returns:
        Updated array or dict of arrays, or selected elements if mode is 'get'.
    """
    # Define the allowed modes and their corresponding JAX methods
    mode_methods = {
        "set": "set",
        "add": "add",
        "subtract": "subtract",
        "multiply": "multiply",
        "divide": "divide",
        "power": "power",
        "min": "min",
        "max": "max",
        "apply": "apply",
        "get": "get",
    }

    if mode not in mode_methods:
        raise ValueError(
            f"Invalid mode '{mode}'. Allowed modes are: {list(mode_methods.keys())}"
        )

    if isinstance(array, dict):
        # Ensure selection (and values if needed) are also dicts with the same keys
        if not isinstance(selection, dict):
            raise TypeError(
                "If 'array' is a dict, 'selection' must also be a dict with the same keys."
            )
        if array.keys() != selection.keys():
            raise ValueError("The keys of 'array' and 'selection' must match.")

        if mode not in ("get", "apply") and values is None:
            raise ValueError(f"'values' must be provided for mode '{mode}'.")

        if mode not in ("get", "apply") and not isinstance(values, dict):
            raise TypeError(
                "If 'array' is a dict, 'values' must also be a dict with the same keys."
            )
        if mode not in ("get", "apply") and array.keys() != values.keys():
            raise ValueError("The keys of 'array' and 'values' must match.")

        # Apply the operation recursively to each item in the dict
        result = {}
        for key in array:
            val = values[key] if mode not in ("get", "apply") else None
            result[key] = mask_op(array[key], selection[key], val, mode, ufunc)
        return result
    else:
        if not (
            isinstance(selection, np.ndarray) or isinstance(selection, jnp.ndarray)
        ):
            raise TypeError("'selection' must be a NumPy or JAX array.")

        if mode not in ("get", "apply") and values is None:
            raise ValueError(f"'values' must be provided for mode '{mode}'.")

        # Flatten the array and selection
        flat_array = array.flatten()
        flat_selection = selection.flatten()
        idx = jnp.arange(flat_array.size)
        selected_indices = idx[flat_selection]

        if mode == "get":
            # Return the selected elements
            return flat_array[selected_indices]
        elif mode == "apply":
            if ufunc is None:
                raise ValueError("ufunc must be provided when mode is 'apply'.")
            # Apply the ufunc to the selected elements
            updated_values = ufunc(flat_array[selected_indices])
            flat_array = flat_array.at[selected_indices].set(updated_values)
        else:
            # Flatten values
            flat_values = values.flatten()

            # Determine how to select values based on their size
            if flat_values.size == selected_indices.size:
                values_to_use = flat_values
            elif flat_values.size == flat_array.size:
                values_to_use = flat_values[selected_indices]
            else:
                raise ValueError(
                    "Values size must be either equal to the number of selected elements or the size of the array."
                )

            # Use getattr to get the appropriate method
            method_name = mode_methods[mode]
            op_method = getattr(flat_array.at[selected_indices], method_name)
            flat_array = op_method(values_to_use)

        return flat_array.reshape(array.shape)


def search_neighborhood(x_nodes, x_query, support_radius):
    """
    Neighbor search within a radius based on scipy's KDTree.

    Args:
      x_nodes (array): Coordinates of the nodes.
      x_query (array): Coordinates of the query points.
      support_radius (float): Radius within which to search for neighbors.

    Returns:
      tuple:
        - num_neighbors (array): Number of neighbors for each query point.
        - max_neighbors (int): Maximum number of neighbors.
        - min_neighbors (int): Minimum number of neighbors.
        - neighbor_list (array): List of neighbors for each query point.
    """
    import scipy

    tree = scipy.spatial.cKDTree(x_nodes)
    num_neighbors = jnp.asarray(
        tree.query_ball_point(
            x_query, support_radius, return_sorted=False, return_length=True
        )
    )
    max_neighbors = num_neighbors.max().item()
    min_neighbors = num_neighbors.min().item()
    _, ii = tree.query(x_query, k=max_neighbors)
    neighbor_list = jnp.asarray(ii)

    print(
        "Neighbor search finished. Min/max neighbors: ", (min_neighbors, max_neighbors)
    )
    return (num_neighbors, max_neighbors, min_neighbors, neighbor_list)


@jit_with_docstring(static_argnames=["static_settings"])
def get_condition_number(dofs, settings, static_settings):
    """
    Computes the condition number of the assembled tangent matrix.

    Notes:
     - Computes a dense tangent. Therefore only useful for small problems.

    Args:
      dofs (jnp.ndarray): Degrees of freedom.
      settings (dict): Settings for the assembly process.
      static_settings (dict): Static settings.

    Returns:
      float: Condition number of the tangent matrix.
    """
    from autopdex import assembler

    (eig, _) = jnp.linalg.eig(
        assembler.assemble_tangent(dofs, settings, static_settings).todense()
    )
    conditioning = eig[0] / eig[-1]
    return conditioning


@jit_with_docstring(static_argnames=["static_settings"])
def symmetry_check(dofs, settings, static_settings):
    """
    Checks the symmetry of the assembled tangent matrix.

    Args:
      dofs (jnp.ndarray): Degrees of freedom.
      settings (dict): Settings for the assembly process.
      static_settings (dict): Static settings.

    Returns:
      float: Sum of absolute differences between the matrix and its transpose.
    """
    from autopdex import assembler

    mat = assembler.assemble_tangent(dofs, settings, static_settings)
    return sparse.sparsify(jnp.sum)(
        sparse.sparsify(jnp.abs)(mat - sparse.bcoo_transpose(mat, permutation=(1, 0)))
    )


def dof_select(dirichlet_nodes, selected_fields):
    """
    DOF selection for the nodal imposition of boundary conditions (jitted).

    Args:
      dirichlet_nodes: list of bools, indicating, which nods are dirichlet_nods
      selected_fields: bool or list of bools

    Returns:
      dirichlet_dofs: list of bools, indicating, which dofs are dirichlet_dofs
    """
    if isinstance(selected_fields, bool):
        return dirichlet_nodes * selected_fields
    else:
        return jnp.outer(dirichlet_nodes, jnp.asarray(selected_fields))


def jnp_to_tuple(jnp_array):
    """
    Converts a JAX array to a tuple. Also works for dicts of JAX arrays.

    Args:
      jnp.ndarray: JAX array to convert.

    Returns:
      tuple: Converted tuple or flax.core.FrozenDict of tuples.
    """
    if isinstance(jnp_array, dict):
        return FrozenDict(
            {key: jnp_to_tuple(jnp_array[key]) for key in list(jnp_array.keys())}
        )
    else:
        as_list = np.array(jnp_array).tolist()
        try:
            return tuple(jnp_to_tuple(i) for i in as_list)
        except:
            return as_list


def to_jax_function(subexpr, reduced_expr):
    """
    Generation of a pure JAX function based on subexpressions and reduced return-value. Can be combined with sympys common subexpression eliminator.

    Return value of the function will be an jnp.ndarray
    Warning: Works currently only functions with basic operations. Special functions as exp() may not be jax-conforming...
    Warning: arguments and function name have to be set manually. The function may have to be adjusted manually.
    """
    assignment_statements = []
    for var, expr in subexpr:
        assignment_statements.append(f"{var} = {expr}")

    # Create the function body
    function_body = "\n  ".join(assignment_statements)

    # Combine everything into a function definition
    function_code = "def shape_functions(xi):\n"
    function_code += f"  {function_body}\n"
    function_code += f"  return jnp.asarray({reduced_expr})\n"

    # Write the function code to a Python file
    with open("lagrange_polynomials.py", "w") as f:
        f.write(function_code)

    # Print the generated function definition
    print(function_code)


@jit_with_docstring(static_argnames=["fun", "derivative_order", "argnum"])
def jacfwd_upto_n_scalar_args(fun, args, derivative_order, argnum):
    """
    Computes all up to the nth derivative of a function `f` using a single pass.

    The function is designed for the use in sensitivity analysis, where the primal evaluation can be very expensive.
    By using recursive jvps, the function computes all the derivatives in pass and does not rely on XLA to delete duplicated code.

    This function supports computing derivatives of any order with respect to individual or multiple
    scalar arguments in a tuple. The derivatives are returned either as a single value or as a collection
    of values, depending on the input type of `argnum`.

    Args:
        fun (Callable): The function for which derivatives are to be computed. This function should accept
            a sequence of scalar inputs and return a scalar output or jnp.ndarray.
        args (tuple): A tuple of input values at which the derivatives are evaluated. The length of `args`
            should match the number of arguments expected by `fun`.
        derivative_order (int): The order of the derivatives to compute. For instance, `n=2` computes the value, first and second derivative.
        argnum (Union[int, tuple]): The index or indices of the argument(s) with respect to which the derivative
            is computed. If an integer is provided, the derivative is computed with respect to a single argument.
            If a tuple of integers is provided, the derivative is computed with respect to multiple arguments.

    Returns:
        jnp.ndarray:
        - The jnp.ndarray has the following order: results[i][j][k]
        - i (optional) selects the argument in the argnum tuple. This dimension dissapears in case argnum is a scalar
        - j is the j-th derivative order
        - k an further indices come from the output shape of the example function

    Example:
        >>> def example_function(x, y):
        >>>     return jnp.asarray([x**3, 2*x*y**3, x*y])
        >>> 
        >>> args = (2.0, 3.0)  # Example input values for x and y
        >>> order = 4  # Up to the fourth derivative
        >>> argnum = (0, 1) # Differentiate with respect to both x and y
        >>> result = jacfwd_upto_n_scalar_args(example_function, args, order, argnum)
        >>> print("Derivatives:", result)
        >>> print("Derivatives:", result.shape) # Output: (2, 5, 3), like the dimensions of (argnum, 1+order, output_of_example_fun)
    """
    def nth_derivative_as_tuple(f, x, n, index):
        # Set the direction for differentiation for the specified argumentz
        tangent_directions = jnp.asarray([jnp.zeros_like(xi) for xi in x])
        tangent_directions = tangent_directions.at[index].set(1)

        if n == 0:
            return f(*x)
        elif n == 1:
            return jax.jvp(f, x, tuple(tangent_directions))
        else:  # Recursive call of jvp
            df = lambda *y: nth_derivative_as_tuple(f, y, n - 1, index)
            values = jax.jvp(df, x, tuple(tangent_directions))
            return values[0] + (values[-1][-1],)

    if isinstance(argnum, int):
        return jnp.asarray(nth_derivative_as_tuple(fun, args, derivative_order, argnum))
    elif isinstance(argnum, tuple):
        # Use jax.vmap to compute derivatives for each index in argnum
        fun_vmap = lambda idx: nth_derivative_as_tuple(fun, args, derivative_order, idx)
        return jnp.asarray(jax.vmap(fun_vmap)(jnp.asarray(argnum)))
    else:
        raise TypeError("argnum must be an integer or a tuple of integers.")


@jit_with_docstring(static_argnames=["fun", "n"])
def jacfwd_upto_n_one_vector_arg(fun, x, n):
    """
    Computes up to the nth derivative of a function `fun` with respect to a vector argument `x` using a single pass.

    This function is tailored for sensitivity analysis, where the primal evaluation can be very expensive. By using
    recursive Jacobian-vector products (JVPs), this function computes all derivatives in one pass, avoiding redundant
    computations.

    The function supports computing derivatives of any order with respect to a single `jnp.ndarray` argument `x`. The derivatives
    are returned as a tuple, where each entry corresponds to the function's value or a derivative of increasing order.

    Args:
        fun (Callable): The function for which derivatives are to be computed. This function should accept a single `jnp.ndarray`
            argument `x` and return a scalar output or `jnp.ndarray`.
        x (jnp.ndarray): The vector input at which the derivatives are evaluated.
        n (int): The order of the derivatives to compute. For instance, `n=3` computes the value, first, second, and third derivatives.

    Returns:
        tuple: A tuple containing the function value and its derivatives up to the nth order.
        The structure of the tuple is as follows:
        - The first entry is the value of the function `fun(x)`.
        - Subsequent entries are the first, second, and higher-order derivatives.
        - For a function returning a multi-dimensional array, the array dimensions are appended to derivative dimensions (see example below)

    Example:
        >>> def example_fun(x): # input shape (2,)
        >>>     return jnp.asarray([
        >>>         [x[0], x[1], x[0] + x[1], x[0] * x[1]],
        >>>         [x[1] - x[0], x[0] / (x[1] + 1e-5), x[0]**2, x[1]**2],
        >>>         [x[0] + 1, x[1] + 1, x[0] * x[0], x[1] * x[1]]
        >>>     ]) # output shape (3, 4)
        >>> 
        >>> x = jnp.asarray([1.0, 2.0])  # Example vector input
        >>> n = 3  # Up to the third derivative
        >>> derivatives = jacfwd_upto_n_one_vector_arg(example_fun, x, n)
        >>> 
        >>> print("Function value:", derivatives[0].shape) # shape (3, 4)
        >>> print("First derivative:", derivatives[1].shape)  # shape (2, 3, 4)
        >>> print("Second derivative:", derivatives[2].shape) # shape (2, 2, 3, 4)
        >>> print("Third derivative:", derivatives[3].shape) # shape (2, 2, 2, 3, 4)
    """

    def flatten_tuple(nested_tuple):
        flat_tuple = ()
        for item in nested_tuple:
            if isinstance(item, tuple):
                flat_tuple += flatten_tuple(item)
            else:
                flat_tuple += (item,)
        return flat_tuple

    def jacfwd_upto_n_one_vector_arg_tmp(f, x, n):
        def single_component_derivative(i, f, n):
            # Initialize tangent_directions with zeros like x
            tangent_directions = jnp.zeros_like(x)
            # Set the i-th component to 1
            tangent_directions = tangent_directions.at[i].set(1.0)

            if n == 1:
                # Compute the jvp for this direction
                values = jax.jvp(f, (x,), (tangent_directions,))
                return (values[0],) + (values[-1],)
            else:
                df = lambda y: jacfwd_upto_n_one_vector_arg_tmp(f, y, n - 1)
                values = jax.jvp(df, (x,), (tangent_directions,))
                return (values[0][0],) + (values[-1],)

        if n == 0:
            return (f(x),)
        elif n >= 1:
            # Vectorize over each component of x
            jvp_values = jax.vmap(single_component_derivative, in_axes=(0, None, None))(
                jnp.arange(x.shape[0]), f, n
            )
            return (jvp_values[0][0],) + (jvp_values[1],)
        else:
            raise ValueError("n must be a non-negative integer.")

    return flatten_tuple(jacfwd_upto_n_one_vector_arg_tmp(fun, x, n))
