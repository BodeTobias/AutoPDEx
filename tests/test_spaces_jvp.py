import numpy as np
import pytest

import jax
import jax.numpy as jnp

from autopdex import spaces


@pytest.mark.parametrize(
    "fun, x, xI, n_dim",
    [
        pytest.param(
            spaces.fem_iso_line_quad_brick,
            jnp.asarray(0.0),
            jnp.asarray([[0.0, 0.0], [1.0, 0.0]]),
            1,
            id="iso-tensor-line-in-2d",
        ),
        pytest.param(
            spaces.fem_iso_line_tri_tet,
            jnp.asarray(0.0),
            jnp.asarray([[0.0, 0.0], [1.0, 0.0]]),
            1,
            id="iso-simplex-line-in-2d",
        ),
        pytest.param(
            spaces.fem_iso_line_quad_brick,
            jnp.asarray([0.0, 0.0]),
            jnp.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
            ),
            2,
            id="iso-tensor-surface-in-3d",
        ),
        pytest.param(
            spaces.fem_iso_line_tri_tet,
            jnp.asarray([0.2, 0.3]),
            jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            2,
            id="iso-simplex-surface-in-3d",
        ),
        pytest.param(
            spaces.fem_line_quad_brick,
            jnp.asarray(0.0),
            jnp.asarray([[0.0, 0.0], [1.0, 0.0]]),
            1,
            id="tensor-line-in-2d",
        ),
        pytest.param(
            spaces.fem_line_tri_tet,
            jnp.asarray(0.0),
            jnp.asarray([[0.0, 0.0], [1.0, 0.0]]),
            1,
            id="simplex-line-in-2d",
        ),
        pytest.param(
            spaces.fem_line_quad_brick,
            jnp.asarray([0.0, 0.0]),
            jnp.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
            ),
            2,
            id="tensor-surface-in-3d",
        ),
        pytest.param(
            spaces.fem_line_tri_tet,
            jnp.asarray([0.2, 0.3]),
            jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            2,
            id="simplex-surface-in-3d",
        ),
    ],
)
def test_space_custom_jvp_supports_scalar_and_vector_pullbacks(fun, x, xI, n_dim):
    fI_scalar = jnp.arange(1, xI.shape[0] + 1, dtype=xI.dtype)
    fI_vector = jnp.stack([fI_scalar, jnp.zeros_like(fI_scalar), -fI_scalar], axis=1)

    for fI in (fI_scalar, fI_vector):
        value = fun(x, xI, fI, {}, True, n_dim)
        grad = jax.jacfwd(lambda xi: fun(xi, xI, fI, {}, True, n_dim))(x)

        assert np.shape(grad) == np.shape(value) + np.shape(x)
        assert bool(jnp.all(jnp.isfinite(jnp.asarray(value))))
        assert bool(jnp.all(jnp.isfinite(jnp.asarray(grad))))
