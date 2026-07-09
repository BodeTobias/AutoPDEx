import pytest

import jax
import jax.numpy as jnp

from autopdex import dae

jax.config.update("jax_enable_x64", True)


def _lotka_volterra_residual(q_fun, t, settings):
    q_t_fun = jax.jacfwd(q_fun)
    q = q_fun(t)
    q_t = q_t_fun(t)

    u = q["u"]
    v = q["v"]
    u_t = q_t["u"]
    v_t = q_t["v"]
    alpha, beta, gamma, delta = (0.1, 0.02, 0.4, 0.02)

    return jnp.array([
        (u_t - (alpha * u - beta * u * v))[0],
        (v_t - (-gamma * v + delta * u * v))[0],
    ])


def _scalar_decay_residual(q_fun, t, settings):
    q, q_t = jax.jvp(q_fun, (t,), (1.0,))
    return jnp.atleast_1d(q_t["rho_p"] + q["rho_p"])


def _constant_rate_residual(q_fun, t, settings):
    q, q_t = jax.jvp(q_fun, (t,), (1.0,))
    return q_t["u"] - settings["rate"]


def _settings_time_residual(q_fun, t, settings):
    q, q_t = jax.jvp(q_fun, (t,), (1.0,))
    return jnp.atleast_1d(q_t["u"] - settings["current time"])


INTEGRATORS = [
    pytest.param("forward_euler", dae.ForwardEuler, id="ForwardEuler"),
    pytest.param("backward_euler", dae.BackwardEuler, id="BackwardEuler"),
    pytest.param("adams_bashforth_4", lambda: dae.AdamsBashforth(4), id="AdamsBashforth4"),
    pytest.param("adams_moulton_1", lambda: dae.AdamsMoulton(1), id="AdamsMoulton1"),
    pytest.param("backward_diff_formula_3", lambda: dae.BackwardDiffFormula(3), id="BackwardDiffFormula3"),
    pytest.param("dirk_3", lambda: dae.DiagonallyImplicitRungeKutta(3), id="DiagonallyImplicitRungeKutta3"),
    pytest.param("kvaerno_5", lambda: dae.Kvaerno(5), id="Kvaerno5"),
    pytest.param("gauss_legendre_14", lambda: dae.GaussLegendreRungeKutta(14), id="GaussLegendreRungeKutta14"),
    pytest.param("dormand_prince_5", lambda: dae.DormandPrince(5), id="DormandPrince5"),
    pytest.param("explicit_rk_11", lambda: dae.ExplicitRungeKutta(11), id="ExplicitRungeKutta11"),
    pytest.param("newmark", dae.Newmark, id="Newmark"),
]


@pytest.mark.parametrize("_name, make_integrator", INTEGRATORS)
def test_lotka_volterra_example_runs_for_listed_integrators_and_newmark(_name, make_integrator):
    static_settings = {
        "dae": _lotka_volterra_residual,
        "time integrators": {
            "u": make_integrator(),
            "v": make_integrator(),
        },
        "verbose": -1,
    }
    manager = dae.TimeSteppingManager(
        static_settings,
        save_policy=dae.SaveEquidistantPolicy(num_points=3),
        step_size_controller=dae.ConstantStepSizeController(),
    )

    num_steps = 20
    result = manager.run(
        {"u": jnp.array([10.0]), "v": jnp.array([10.0])},
        1.0 / num_steps,
        1.0,
        num_steps,
    )

    assert int(result.num_accepted) == num_steps
    assert int(result.num_rejected) == 0
    assert result.q["u"].shape == (1,)
    assert result.q["v"].shape == (1,)
    assert bool(jnp.all(jnp.isfinite(result.q["u"])))
    assert bool(jnp.all(jnp.isfinite(result.q["v"])))


def test_time_stepping_manager_accepts_scalar_dofs_with_dirk_postprocessing():
    def postprocess(q_fun, t, settings):
        q, q_t = jax.jvp(q_fun, (t,), (1.0,))
        assert q["rho_p"].shape == ()
        assert q_t["rho_p"].shape == ()
        return {"rho_p_dot": q_t["rho_p"]}

    manager = dae.TimeSteppingManager(
        {
            "dae": _scalar_decay_residual,
            "time integrators": {"rho_p": dae.DiagonallyImplicitRungeKutta(2)},
            "verbose": -1,
        },
        save_policy=dae.SaveAllPolicy(),
        step_size_controller=dae.ConstantStepSizeController(),
        postprocessing_fun=postprocess,
    )

    result = manager.run({"rho_p": 1.0}, 0.1, 0.1, 1)

    assert result.q["rho_p"].shape == ()
    assert result.q_der_n["rho_p"].shape == (1, 1)
    assert bool(jnp.all(jnp.isfinite(result.history.user["rho_p_dot"])))


def test_time_stepping_manager_preserves_vector_derivative_shape_in_postprocessing():
    rate = jnp.array([1.0, 2.0])

    def postprocess(q_fun, t, settings):
        _, q_t = jax.jvp(q_fun, (t,), (1.0,))
        return {"u_t": q_t["u"]}

    manager = dae.TimeSteppingManager(
        {
            "dae": _constant_rate_residual,
            "time integrators": {"u": dae.DiagonallyImplicitRungeKutta(1)},
            "verbose": -1,
        },
        save_policy=dae.SaveAllPolicy(),
        step_size_controller=dae.ConstantStepSizeController(),
        postprocessing_fun=postprocess,
    )

    result = manager.run({"u": jnp.zeros(2)}, 0.5, 0.5, 1, {"current time": 0.0, "rate": rate})

    assert result.history.user["u_t"].shape == (2, 2)
    assert jnp.allclose(result.history.user["u_t"][1], rate, atol=1e-12, rtol=1e-12)


def test_time_stepping_manager_sets_current_time_to_dirk_stage_time():
    manager = dae.TimeSteppingManager(
        {
            "dae": _settings_time_residual,
            "time integrators": {"u": dae.DiagonallyImplicitRungeKutta(2)},
            "verbose": -1,
        },
        step_size_controller=dae.ConstantStepSizeController(),
    )

    result = manager.run({"u": 0.0}, 1.0, 1.0, 1)

    assert jnp.allclose(result.q["u"], 0.5, atol=1e-12, rtol=1e-12)
    assert jnp.allclose(result.settings["current time"], 1.0, atol=1e-12, rtol=1e-12)
