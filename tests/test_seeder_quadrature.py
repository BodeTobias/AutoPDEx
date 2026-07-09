def test_int_pts_in_tri_mesh_runs_and_shapes():
    import jax
    import jax.numpy as jnp

    from autopdex import seeder

    jax.config.update("jax_enable_x64", True)
    x_nodes = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64)
    elem = jnp.asarray([[0, 1, 2]], dtype=int)

    x_int, w_int, n_int, connectivity = seeder.int_pts_in_tri_mesh(x_nodes, elem, order=2)

    assert x_int.shape == (3, 2)
    assert w_int.shape == (3,)
    assert int(n_int) == 3
    assert connectivity.shape == (3, 3)


def test_int_pts_in_tet_mesh_runs_and_shapes():
    import jax
    import jax.numpy as jnp

    from autopdex import seeder

    jax.config.update("jax_enable_x64", True)
    x_nodes = jnp.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=jnp.float64,
    )
    elem = jnp.asarray([[0, 1, 2, 3]], dtype=int)

    x_int, w_int, n_int, connectivity = seeder.int_pts_in_tet_mesh(x_nodes, elem, order=2)

    assert x_int.shape == (4, 3)
    assert w_int.shape == (4,)
    assert int(n_int) == 4
    assert connectivity.shape == (4, 4)
