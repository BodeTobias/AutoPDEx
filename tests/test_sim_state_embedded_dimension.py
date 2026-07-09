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
Tests for fields living on domains of different (embedded) intrinsic dimensions,
mirroring examples/offline/embedded_lower_dim_fields.py.

The integration dimension is a property of the domain (its cells); a field is a
volume field when ``space.dim == domain.dim``. This lets a lower-dimensional
field carry its own DOFs on cells embedded in a higher-dimensional coordinate
space (a 1D wire in a 2D plate, a 2D membrane on a 3D block). Both are checked as
patch tests: manufactured polynomial solutions are reproduced exactly.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from autopdex import SimState, dae, spaces

jax.config.update("jax_enable_x64", True)
meshio = pytest.importorskip("meshio")


def _scalar(fun, *args):
    return jnp.asarray(fun(*args)).reshape(())


def _grad(field, ctx):
    return jax.jacfwd(lambda x: _scalar(ctx.trial_ansatz[field], x, ctx.t))(ctx.x_int)


def test_wire_embedded_in_plate_is_exact():
    """1D field (line cells) embedded in a 2D plate, decoupled, patch test."""
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
        u = _scalar(ctx.trial_ansatz["plate"], ctx.x_int, ctx.t)
        if ctx.mode == "output":
            return {"plate": u}
        gu = _grad("plate", ctx)
        return 0.5 * jnp.dot(gu, gu) - (-4.0) * u  # u = x^2 + y^2

    def wire_pot(ctx):
        u = _scalar(ctx.trial_ansatz["wire"], ctx.x_int, ctx.t)
        if ctx.mode == "output":
            return {"wire": u}
        x = ctx.trial_ansatz["physical coor"](ctx.x_int)[0]
        g = -(6.0 - 12.0 * x) + (3.0 * x**2 - 2.0 * x**3)  # -u'' + u, u = 3x^2 - 2x^3
        gu = _grad("wire", ctx)
        return 0.5 * jnp.dot(gu, gu) + 0.5 * u**2 - g * u

    sim = SimState({
        "plate": spaces.H1(order=2, dim=2, field_dimension=1),
        "wire": spaces.H1(order=3, dim=1, field_dimension=1),
    })
    sim.import_mesh(mesh)
    sim.add_domain("wire", lambda x: True, element_type="line")
    sim.set_active_domains({"plate": "__all__", "wire": "wire"})
    sim.add_temporal_discretization({"plate": dae.NoTimeDerivative(), "wire": dae.NoTimeDerivative()})
    sim.add_model("__all__", "potential", plate_pot)
    sim.add_model("wire", "potential", wire_pot)
    sim.add_strong_bc(
        "plate",
        on_boundary_fun=lambda x: (jnp.isclose(x[0], 0.0) | jnp.isclose(x[0], 1.0)
                                   | jnp.isclose(x[1], 0.0) | jnp.isclose(x[1], 1.0)),
        value_fun=lambda x, t, s: x[0]**2 + x[1]**2,
    )
    sim.initialize(verbose=-1)
    sim.prepare()
    res = sim.run(dt0=1.0, time_span=1.0, num_time_steps=1)

    node = np.asarray(sim.settings["node coordinates"])
    nN = node.shape[0]
    up = np.asarray(res.q["plate"]).ravel()
    uw = np.asarray(res.q["wire"]).ravel()

    # the wire has its own DOFs on the 1D line only -> strictly fewer than the plate
    on_wire = np.isclose(node[:, 1], 0.5)
    assert uw.shape[0] < up.shape[0]

    assert np.max(np.abs(up[:nN] - (node[:, 0]**2 + node[:, 1]**2))) < 1e-10
    exact_w = 3.0 * node[on_wire, 0]**2 - 2.0 * node[on_wire, 0]**3
    assert np.max(np.abs(uw[:int(on_wire.sum())] - exact_w)) < 1e-10


def test_membrane_embedded_on_block_is_exact():
    """2D field (quad cells) on the face of a 3D block, decoupled, patch test."""
    n = 2
    c = np.linspace(0.0, 1.0, n + 1)
    P = np.array([[x, y, z] for k, z in enumerate(c) for j, y in enumerate(c) for i, x in enumerate(c)])
    nid = lambda i, j, k: k * (n + 1) ** 2 + j * (n + 1) + i
    hexes = np.array([[nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                       nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)]
                      for k in range(n) for j in range(n) for i in range(n)], dtype=int)
    quads = np.array([[nid(i, j, n), nid(i + 1, j, n), nid(i + 1, j + 1, n), nid(i, j + 1, n)]
                      for j in range(n) for i in range(n)], dtype=int)
    mesh = meshio.Mesh(points=P, cells=[("hexahedron", hexes), ("quad", quads)])

    f = lambda t: 3 * t**2 - 2 * t**3
    d2 = lambda t: 6 - 12 * t

    def bulk_pot(ctx):
        u = _scalar(ctx.trial_ansatz["bulk"], ctx.x_int, ctx.t)
        if ctx.mode == "output":
            return {"bulk": u}
        gu = _grad("bulk", ctx)
        return 0.5 * jnp.dot(gu, gu) - (-6.0) * u  # u = x^2 + y^2 + z^2

    def membrane_pot(ctx):
        u = _scalar(ctx.trial_ansatz["membrane"], ctx.x_int, ctx.t)
        if ctx.mode == "output":
            return {"membrane": u}
        x = ctx.trial_ansatz["physical coor"](ctx.x_int)
        g = -(d2(x[0]) + d2(x[1])) + (f(x[0]) + f(x[1]))
        gu = _grad("membrane", ctx)
        return 0.5 * jnp.dot(gu, gu) + 0.5 * u**2 - g * u

    sim = SimState({
        "bulk": spaces.H1(order=2, dim=3, field_dimension=1),
        "membrane": spaces.H1(order=3, dim=2, field_dimension=1),
    })
    sim.import_mesh(mesh)
    sim.add_domain("top", lambda x: jnp.isclose(x[2], 1.0), element_type="quad")
    sim.set_active_domains({"bulk": "__all__", "membrane": "top"})
    sim.add_temporal_discretization({"bulk": dae.NoTimeDerivative(), "membrane": dae.NoTimeDerivative()})
    sim.add_model("__all__", "potential", bulk_pot)
    sim.add_model("top", "potential", membrane_pot)
    sim.add_strong_bc(
        "bulk",
        on_boundary_fun=lambda x: (jnp.isclose(x[0], 0.0) | jnp.isclose(x[0], 1.0)
                                   | jnp.isclose(x[1], 0.0) | jnp.isclose(x[1], 1.0)
                                   | jnp.isclose(x[2], 0.0) | jnp.isclose(x[2], 1.0)),
        value_fun=lambda x, t, s: x[0]**2 + x[1]**2 + x[2]**2,
    )
    sim.initialize(verbose=-1)
    sim.prepare()
    res = sim.run(dt0=1.0, time_span=1.0, num_time_steps=1)

    node = np.asarray(sim.settings["node coordinates"])
    nN = node.shape[0]
    ub = np.asarray(res.q["bulk"]).ravel()
    um = np.asarray(res.q["membrane"]).ravel()

    top = np.isclose(node[:, 2], 1.0)
    assert um.shape[0] < ub.shape[0]  # membrane only on the top face
    assert np.max(np.abs(ub[:nN] - (node[:, 0]**2 + node[:, 1]**2 + node[:, 2]**2))) < 1e-10
    exact_m = f(node[top, 0]) + f(node[top, 1])
    assert np.max(np.abs(um[:int(top.sum())] - exact_m)) < 1e-10
