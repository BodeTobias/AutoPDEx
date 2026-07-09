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

"""
Boundary conditions on fields living on embedded lower-dimensional domains.

The boundary of such a field is one dimension below the field itself (the edges
of a 2D membrane, the endpoints of a 1D wire), i.e. below the mesh facets. These
tests cover:
  - strong (Dirichlet) BC on the dim-1 edges of a 2D membrane in 3D (patch test),
  - weak BC on those edges (assembles and runs),
  - point (0-dim boundary) Dirichlet on the endpoints of a 1D wire (patch test).
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from autopdex import SimState, dae, spaces

jax.config.update("jax_enable_x64", True)
meshio = pytest.importorskip("meshio")


def _sc(fn, *a):
    return jnp.asarray(fn(*a)).reshape(())


def _grad(field, ctx):
    return jax.jacfwd(lambda x: _sc(ctx.trial_ansatz[field], x, ctx.t))(ctx.x_int)


def _flat_membrane_mesh(n=4):
    xs = np.linspace(0.0, 1.0, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])  # flat sheet in 3D
    nid = lambda i, j: j * (n + 1) + i
    quads = np.array([[nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
                      for j in range(n) for i in range(n)], dtype=int)
    return meshio.Mesh(points=pts, cells=[("quad", quads)])


def test_membrane_edge_strong_bc_is_exact():
    """Dirichlet on the dim-1 edges of a 2D membrane embedded in 3D (patch test)."""
    mesh = _flat_membrane_mesh(4)

    def pot(ctx):
        u = _sc(ctx.trial_ansatz["mem"], ctx.x_int, ctx.t)
        if ctx.mode == "output":
            return {"mem": u}
        gu = _grad("mem", ctx)
        return 0.5 * jnp.dot(gu, gu) - (-4.0) * u  # u = x^2 + y^2

    sim = SimState({"mem": spaces.H1(order=2, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({"mem": dae.NoTimeDerivative()})
    sim.add_model("__all__", "potential", pot)
    sim.add_strong_bc(
        "mem",
        on_boundary_fun=lambda x: (jnp.isclose(x[0], 0.0) | jnp.isclose(x[0], 1.0)
                                   | jnp.isclose(x[1], 0.0) | jnp.isclose(x[1], 1.0)),
        value_fun=lambda x, t, s: x[0]**2 + x[1]**2,
    )
    sim.initialize(verbose=-1)
    sim.prepare()
    res = sim.run(dt0=1.0, time_span=1.0, num_time_steps=1)

    node = np.asarray(sim.settings["node coordinates"])
    u = np.asarray(res.q["mem"]).ravel()
    exact = node[:, 0]**2 + node[:, 1]**2
    assert np.max(np.abs(u[:node.shape[0]] - exact)) < 1e-10


def test_membrane_edge_weak_bc_runs():
    """A weak BC on the dim-1 edge of a 2D membrane in 3D assembles and runs."""
    mesh = _flat_membrane_mesh(4)

    def pot(ctx):
        u = _sc(ctx.trial_ansatz["mem"], ctx.x_int, ctx.t)
        if ctx.mode == "output":
            return {"mem": u}
        gu = _grad("mem", ctx)
        return 0.5 * jnp.dot(gu, gu) + 0.5 * u**2 - 1.0 * u  # Helmholtz, well posed

    sim = SimState({"mem": spaces.H1(order=2, dim=2, field_dimension=1)})
    sim.import_mesh(mesh)
    sim.add_temporal_discretization({"mem": dae.NoTimeDerivative()})
    sim.add_model("__all__", "potential", pot)
    sim.add_weak_bc("mem", on_boundary_fun=lambda x: jnp.isclose(x[0], 1.0),
                    value_fun=lambda x, t, s: jnp.asarray(0.5))
    sim.initialize(verbose=-1)
    sim.prepare()
    res = sim.run(dt0=1.0, time_span=1.0, num_time_steps=1)
    assert np.all(np.isfinite(np.asarray(res.q["mem"])))


def test_wire_point_dirichlet_is_exact():
    """Point (0-dim boundary) Dirichlet on the endpoints of a 1D wire in a 2D plate."""
    n = 4
    xs = np.linspace(0.0, 1.0, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    nid = lambda i, j: j * (n + 1) + i
    quads = np.array([[nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
                      for j in range(n) for i in range(n)], dtype=int)
    jw = n // 2
    wire = np.array([[nid(i, jw), nid(i + 1, jw)] for i in range(n)], dtype=int)
    mesh = meshio.Mesh(points=pts, cells=[("quad", quads), ("line", wire)])

    def plate_pot(ctx):
        u = _sc(ctx.trial_ansatz["plate"], ctx.x_int, ctx.t)
        if ctx.mode == "output":
            return {"plate": u}
        gu = _grad("plate", ctx)
        return 0.5 * jnp.dot(gu, gu)

    def wire_pot(ctx):
        u = _sc(ctx.trial_ansatz["wire"], ctx.x_int, ctx.t)
        if ctx.mode == "output":
            return {"wire": u}
        x = ctx.trial_ansatz["physical coor"](ctx.x_int)[0]
        f = -(6.0 - 12.0 * x)  # -u'' = f, u = 3x^2 - 2x^3
        gu = _grad("wire", ctx)
        return 0.5 * jnp.dot(gu, gu) - f * u

    sim = SimState({
        "plate": spaces.H1(order=1, dim=2, field_dimension=1),
        "wire": spaces.H1(order=3, dim=1, field_dimension=1),
    })
    sim.import_mesh(mesh)
    sim.add_domain("wire", lambda x: True, element_type="line")
    sim.set_active_domains({"plate": "__all__", "wire": "wire"})
    sim.add_temporal_discretization({"plate": dae.NoTimeDerivative(), "wire": dae.NoTimeDerivative()})
    sim.add_model("__all__", "potential", plate_pot)
    sim.add_model("wire", "potential", wire_pot)
    # pin the plate so its own system is well posed
    sim.add_strong_bc("plate",
                      on_boundary_fun=lambda x: (jnp.isclose(x[0], 0.0) | jnp.isclose(x[0], 1.0)
                                                 | jnp.isclose(x[1], 0.0) | jnp.isclose(x[1], 1.0)),
                      value_fun=lambda x, t, s: jnp.asarray(0.0))
    # point Dirichlet on the 1D wire endpoints (0-dimensional boundary)
    sim.add_strong_bc("wire",
                      on_boundary_fun=lambda x: (jnp.isclose(x[0], 0.0) | jnp.isclose(x[0], 1.0)),
                      value_fun=lambda x, t, s: 3.0 * x[0]**2 - 2.0 * x[0]**3)
    sim.initialize(verbose=-1)
    sim.prepare()
    res = sim.run(dt0=1.0, time_span=1.0, num_time_steps=1)

    node = np.asarray(sim.settings["node coordinates"])
    uw = np.asarray(res.q["wire"]).ravel()
    on_wire = np.isclose(node[:, 1], 0.5)
    exact = 3.0 * node[on_wire, 0]**2 - 2.0 * node[on_wire, 0]**3
    assert np.max(np.abs(uw[:int(on_wire.sum())] - exact)) < 1e-10


def test_wire_point_load_is_exact():
    """Point (0-dim) weak BC / force on a 1D wire endpoint: bar under an end force."""
    F = 0.37
    n = 4
    xs = np.linspace(0.0, 1.0, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    nid = lambda i, j: j * (n + 1) + i
    quads = np.array([[nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
                      for j in range(n) for i in range(n)], dtype=int)
    jw = n // 2
    wire = np.array([[nid(i, jw), nid(i + 1, jw)] for i in range(n)], dtype=int)
    mesh = meshio.Mesh(points=pts, cells=[("quad", quads), ("line", wire)])

    def plate_pot(ctx):
        u = _sc(ctx.trial_ansatz["plate"], ctx.x_int, ctx.t)
        if ctx.mode == "output":
            return {"plate": u}
        gu = _grad("plate", ctx)
        return 0.5 * jnp.dot(gu, gu)

    def wire_pot(ctx):
        u = _sc(ctx.trial_ansatz["wire"], ctx.x_int, ctx.t)
        if ctx.mode == "output":
            return {"wire": u}
        gu = _grad("wire", ctx)
        return 0.5 * jnp.dot(gu, gu)  # -u'' = 0

    sim = SimState({
        "plate": spaces.H1(order=1, dim=2, field_dimension=1),
        "wire": spaces.H1(order=1, dim=1, field_dimension=1),
    })
    sim.import_mesh(mesh)
    sim.add_domain("wire", lambda x: True, element_type="line")
    sim.set_active_domains({"plate": "__all__", "wire": "wire"})
    sim.add_temporal_discretization({"plate": dae.NoTimeDerivative(), "wire": dae.NoTimeDerivative()})
    sim.add_model("__all__", "potential", plate_pot)
    sim.add_model("wire", "potential", wire_pot)
    sim.add_strong_bc("plate",
                      on_boundary_fun=lambda x: (jnp.isclose(x[0], 0.0) | jnp.isclose(x[0], 1.0)
                                                 | jnp.isclose(x[1], 0.0) | jnp.isclose(x[1], 1.0)),
                      value_fun=lambda x, t, s: jnp.asarray(0.0))
    # wire: Dirichlet at x=0, point force F at the free end x=1
    sim.add_strong_bc("wire", on_boundary_fun=lambda x: jnp.isclose(x[0], 0.0),
                      value_fun=lambda x, t, s: jnp.asarray(0.0))
    sim.add_weak_bc("wire", on_boundary_fun=lambda x: jnp.isclose(x[0], 1.0),
                    value_fun=lambda x, t, s: jnp.asarray(F), sign=-1.0)
    sim.initialize(verbose=-1)
    sim.prepare()
    res = sim.run(dt0=1.0, time_span=1.0, num_time_steps=1)

    node = np.asarray(sim.settings["node coordinates"])
    uw = np.asarray(res.q["wire"]).ravel()
    on_wire = np.isclose(node[:, 1], 0.5)
    exact = F * node[on_wire, 0]  # u = F x
    assert np.max(np.abs(uw[:int(on_wire.sum())] - exact)) < 1e-10
