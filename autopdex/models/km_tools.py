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

"""Convenience functions for Kelvin-Mandel notation.

Supported representations
-------------------------

2D matrix:
    shape == (2, 2)

2D Kelvin-Mandel vector:
    shape == (3,)
    [xx, yy, sqrt(2) * xy]

3D matrix:
    shape == (3, 3)

3D Kelvin-Mandel vector with zero z-shear:
    shape == (4,)
    [xx, yy, zz, sqrt(2) * xy]

Full 3D Kelvin-Mandel vector:
    shape == (6,)
    [xx, yy, zz, sqrt(2) * xy, sqrt(2) * yz, sqrt(2) * xz]
"""

import jax.numpy as jnp

from autopdex.utility import matrix_det

__all__ = [
    "eye",
    "trace",
    "isotropic",
    "dev",
    "symmetric_part",
    "pack",
    "pack_full",
    "pack_z_shear_zero",
    "unpack",
    "unpack_to_2d",
    "unpack_to_3d",
    "det",
    "_det_xy",
    "_det_z_shear_zero",
    "invariants",
]

_sqrt2 = jnp.sqrt(2.0)
_inv_sqrt2 = 1.0 / _sqrt2

_eye_2d = jnp.array([1.0, 1.0, 0.0])
_eye_z_shear_zero = jnp.array([1.0, 1.0, 1.0, 0.0])

eye_3d = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
eye_2d = jnp.array([1.0, 1.0, 0.0])
eye = eye_3d

def _unsupported(a, name):
    """Raise a consistent shape error."""
    raise ValueError(f"Unsupported shape for {name}: {a.shape}")

def _zero_like(x):
    """Return a scalar zero with the dtype of x."""
    return 0.0 * x

def _dimension(a):
    """Return the physical dimension inferred from the representation."""
    if a.shape == (2, 2) or a.shape == (3,):
        return 2

    if a.shape == (3, 3) or a.shape == (4,) or a.shape == (6,):
        return 3

    _unsupported(a, "_dimension")

def _identity_like(a):
    """Return the identity tensor in the same representation as a."""
    if a.shape == (2, 2):
        return jnp.eye(2, dtype=a.dtype)

    if a.shape == (3,):
        return _eye_2d.astype(a.dtype)

    if a.shape == (3, 3):
        return jnp.eye(3, dtype=a.dtype)

    if a.shape == (4,):
        return _eye_z_shear_zero.astype(a.dtype)

    if a.shape == (6,):
        return eye.astype(a.dtype)

    _unsupported(a, "_identity_like")

def trace(a):
    """Return the trace inferred from the representation."""
    if a.shape == (2, 2):
        return a[0, 0] + a[1, 1]

    if a.shape == (3,):
        return a[0] + a[1]

    if a.shape == (3, 3):
        return a[0, 0] + a[1, 1] + a[2, 2]

    if a.shape == (4,) or a.shape == (6,):
        return a[0] + a[1] + a[2]

    _unsupported(a, "trace")

def spherical(a):
    """Return the spherical part in the same representation as a."""
    return trace(a) / _dimension(a) * _identity_like(a)

def dev(a):
    """Return the devic part in the same representation as a."""
    return a - spherical(a)

def symmetric_part(A):
    """Return the symmetric part of a 2x2 or 3x3 matrix."""
    if A.shape == (2, 2) or A.shape == (3, 3):
        return 0.5 * (A + A.T)

    _unsupported(A, "symmetric_part")

def _pack_from_2d(A, symmetrize=True):
    """Pack a 2x2 matrix into 2D Kelvin-Mandel notation."""
    if A.shape != (2, 2):
        _unsupported(A, "_pack_from_2d")

    if symmetrize:
        xy = 0.5 * (A[0, 1] + A[1, 0])
    else:
        xy = A[0, 1]

    return jnp.stack([A[0, 0], A[1, 1], _sqrt2 * xy])

def _pack_from_3d(A, symmetrize=True):
    """Pack a 3x3 matrix into full 3D Kelvin-Mandel notation."""
    if A.shape != (3, 3):
        _unsupported(A, "_pack_from_3d")

    if symmetrize:
        A = symmetric_part(A)

    return jnp.stack(
        [
            A[0, 0],
            A[1, 1],
            A[2, 2],
            _sqrt2 * A[0, 1],
            _sqrt2 * A[1, 2],
            _sqrt2 * A[0, 2],
        ]
    )

def pack(A, symmetrize=True):
    """Pack a 2x2 or 3x3 matrix into its natural Kelvin-Mandel notation.

    A 2x2 matrix becomes

        [xx, yy, sqrt(2) * xy]

    A 3x3 matrix becomes

        [xx, yy, zz, sqrt(2) * xy, sqrt(2) * yz, sqrt(2) * xz]
    """
    if A.shape == (2, 2):
        return _pack_from_2d(A, symmetrize=symmetrize)

    if A.shape == (3, 3):
        return _pack_from_3d(A, symmetrize=symmetrize)

    _unsupported(A, "pack")

def pack_z_shear_zero(A, zz=0.0, symmetrize=True):
    """Pack to compact 3D notation with zero z-shear.

    For a 2x2 input, zz is supplied explicitly.

    For a 3x3 input, zz is taken from A[2, 2]. The z-shear entries
    A[0, 2], A[2, 0], A[1, 2] and A[2, 1] are ignored.

    The returned vector is

        [xx, yy, zz, sqrt(2) * xy]
    """
    if A.shape == (2, 2):
        if symmetrize:
            xy = 0.5 * (A[0, 1] + A[1, 0])
        else:
            xy = A[0, 1]

        zero = _zero_like(A[0, 0])
        return jnp.stack([A[0, 0], A[1, 1], zero + zz, _sqrt2 * xy])

    if A.shape == (3, 3):
        if symmetrize:
            xy = 0.5 * (A[0, 1] + A[1, 0])
        else:
            xy = A[0, 1]

        return jnp.stack([A[0, 0], A[1, 1], A[2, 2], _sqrt2 * xy])

    _unsupported(A, "pack_z_shear_zero")

def pack_full(A, zz=0.0, symmetrize=True):
    """Pack a 2x2 or 3x3 matrix into full 3D Kelvin-Mandel notation.

    A 2x2 matrix is embedded into 3D with the supplied zz value and
    zero z-shear components.
    """
    if A.shape == (2, 2):
        a = pack_z_shear_zero(A, zz=zz, symmetrize=symmetrize)
        return _expand_z_shear_zero(a)

    if A.shape == (3, 3):
        return _pack_from_3d(A, symmetrize=symmetrize)

    _unsupported(A, "pack_full")

def unpack_to_2d(a):
    """Unpack the xy block to a 2x2 matrix."""
    if a.shape == (2, 2):
        return a

    if a.shape == (3, 3):
        return a[:2, :2]

    if a.shape == (3,):
        xy = _inv_sqrt2 * a[2]
        return jnp.stack(
            [
                jnp.stack([a[0], xy]),
                jnp.stack([xy, a[1]]),
            ]
        )

    if a.shape == (4,):
        xy = _inv_sqrt2 * a[3]
        return jnp.stack(
            [
                jnp.stack([a[0], xy]),
                jnp.stack([xy, a[1]]),
            ]
        )

    if a.shape == (6,):
        xy = _inv_sqrt2 * a[3]
        return jnp.stack(
            [
                jnp.stack([a[0], xy]),
                jnp.stack([xy, a[1]]),
            ]
        )

    _unsupported(a, "unpack_to_2d")

def unpack_to_3d(a, zz=0.0):
    """Unpack to a 3x3 matrix.

    A 2D input is embedded into 3D with the supplied zz value and
    zero z-shear components.
    """
    if a.shape == (3, 3):
        return a

    if a.shape == (2, 2):
        zero = _zero_like(a[0, 0])
        return jnp.stack(
            [
                jnp.stack([a[0, 0], a[0, 1], zero]),
                jnp.stack([a[1, 0], a[1, 1], zero]),
                jnp.stack([zero, zero, zero + zz]),
            ]
        )

    if a.shape == (3,):
        zero = _zero_like(a[0])
        xy = _inv_sqrt2 * a[2]
        return jnp.stack(
            [
                jnp.stack([a[0], xy, zero]),
                jnp.stack([xy, a[1], zero]),
                jnp.stack([zero, zero, zero + zz]),
            ]
        )

    if a.shape == (4,):
        zero = _zero_like(a[0])
        xy = _inv_sqrt2 * a[3]
        return jnp.stack(
            [
                jnp.stack([a[0], xy, zero]),
                jnp.stack([xy, a[1], zero]),
                jnp.stack([zero, zero, a[2]]),
            ]
        )

    if a.shape == (6,):
        xy = _inv_sqrt2 * a[3]
        yz = _inv_sqrt2 * a[4]
        xz = _inv_sqrt2 * a[5]
        return jnp.stack(
            [
                jnp.stack([a[0], xy, xz]),
                jnp.stack([xy, a[1], yz]),
                jnp.stack([xz, yz, a[2]]),
            ]
        )

    _unsupported(a, "unpack_to_3d")

def unpack(a):
    """Unpack to the natural matrix dimension.

    A 2D representation is unpacked to 2x2.
    A 3D representation is unpacked to 3x3.
    Matrices are returned unchanged.
    """
    if a.shape == (2, 2) or a.shape == (3, 3):
        return a

    if a.shape == (3,):
        return unpack_to_2d(a)

    if a.shape == (4,) or a.shape == (6,):
        return unpack_to_3d(a)

    _unsupported(a, "unpack")

def _expand_z_shear_zero(a):
    """Expand compact zero-z-shear notation to full 3D notation."""
    if a.shape == (4,):
        zero = _zero_like(a[0])
        return jnp.stack([a[0], a[1], a[2], a[3], zero, zero])

    if a.shape == (6,):
        zero = _zero_like(a[0])
        return jnp.stack([a[0], a[1], a[2], a[3], zero, zero])

    _unsupported(a, "_expand_z_shear_zero")

def _det_xy(a):
    """Return the determinant of the xy block."""
    if a.shape == (3,):
        xy = _inv_sqrt2 * a[2]
        return a[0] * a[1] - xy * xy

    A = unpack_to_2d(a)
    return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]

def _det_z_shear_zero(a, zz=0.0):
    """Return the determinant assuming zero z-shear.

    A 2D input is embedded into 3D with the supplied zz value.
    """
    if a.shape == (2, 2) or a.shape == (3,):
        return zz * _det_xy(a)

    if a.shape == (4,):
        xy = _inv_sqrt2 * a[3]
        return a[2] * (a[0] * a[1] - xy * xy)

    if a.shape == (6,):
        xy = _inv_sqrt2 * a[3]
        return a[2] * (a[0] * a[1] - xy * xy)

    if a.shape == (3, 3):
        return a[2, 2] * _det_xy(a)

    _unsupported(a, "_det_z_shear_zero")

def det(a):
    """Return the determinant inferred from the representation."""
    if a.shape == (2, 2) or a.shape == (3,):
        return _det_xy(a)

    if a.shape == (3, 3):
        return matrix_det(a)

    if a.shape == (4,):
        return _det_z_shear_zero(a)

    if a.shape == (6,):
        xx, yy, zz, sxy, syz, sxz = a

        xy2 = 0.5 * sxy * sxy
        yz2 = 0.5 * syz * syz
        xz2 = 0.5 * sxz * sxz

        return (
            xx * yy * zz
            + _inv_sqrt2 * sxy * syz * sxz
            - xx * yz2
            - yy * xz2
            - zz * xy2
        )

    _unsupported(a, "det")

def invariants(a):
    """Return principal invariants.

    For 2D inputs, return (I1, I2).
    For 3D inputs, return (I1, I2, I3).
    """
    I1 = trace(a)

    if a.shape == (2, 2) or a.shape == (3,):
        I2 = det(a)
        return I1, I2

    if a.shape == (3, 3):
        I2 = (
            a[0, 0] * a[1, 1]
            + a[0, 0] * a[2, 2]
            + a[1, 1] * a[2, 2]
            - a[0, 1] * a[1, 0]
            - a[0, 2] * a[2, 0]
            - a[1, 2] * a[2, 1]
        )
        return I1, I2, det(a)

    if a.shape == (4,):
        xx, yy, zz, sxy = a

        I2 = (
            xx * yy
            + xx * zz
            + yy * zz
            - 0.5 * sxy * sxy
        )

        return I1, I2, det(a)

    if a.shape == (6,):
        xx, yy, zz, sxy, syz, sxz = a

        I2 = (
            xx * yy
            + xx * zz
            + yy * zz
            - 0.5 * (sxy * sxy + syz * syz + sxz * sxz)
        )

        return I1, I2, det(a)

    _unsupported(a, "invariants")
