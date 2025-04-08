# plotter.py
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
This module provides some visualization functions mainly for debugging smooth distance functions.
For plotting simulation results, it is recommended to use meshio for writing VTK files and
inspecting them in Paraview. See the quickstart guide as an example.
"""

import os
import jax
import jax.numpy as jnp
import numpy as np

def pv_plot(
    x_vis,
    data,
    export_vtk=False,
    show=True,
    groupname="group",
    filename="frame",
    fieldname="data",
    scale_range=1.0,
    data_on_z=True,
):
    """
    Plots and optionally exports 3D data using PyVista.

    Args:
      - x_vis (np.ndarray): Array of coordinates for visualization points.
      - data (np.ndarray): Data to be visualized at the points.
      - export_vtk (bool, optional): Whether to export the plot to a VTK file. Default is False.
      - show (bool, optional): Whether to show the plot. Default is True.
      - groupname (str, optional): Name of the group folder for VTK export. Default is 'group'.
      - filename (str, optional): Name of the file for VTK export. Default is 'frame'.
      - fieldname (str, optional): Name of the data field in the VTK file. Default is 'data'.
      - scale_range (float, optional): Scale range for data visualization. Default is 1.0.
      - data_on_z (bool, optional): Whether to plot data on the z-axis. Default is True.

    Notes:
      - Only supports 2d and 3d
    """
    import pyvista as pv

    pv.global_theme.cmap = "jet"

    if x_vis.shape[1] not in [2, 3]:
        raise AssertionError(
            "Currently not implemented for dimensions other than 2 or 3."
        )

    if x_vis.shape[1] == 2:
        x_range = jnp.nanmax(x_vis.flatten()) - jnp.nanmin(x_vis.flatten())
        data_range = jnp.nanmax(data) - jnp.nanmin(data)
        if scale_range:
            scaled_data = data
        else:
            scaled_data = (
                scale_range * (x_range / data_range) * data if data_range > 1e-8 else data
            )
        positions = np.column_stack(
            (
                x_vis[:, 0],
                x_vis[:, 1],
                scaled_data if data_on_z else np.zeros_like(scaled_data),
            )
        )
    else:
        positions = np.asarray(x_vis)

    points = pv.PolyData(positions)
    points.point_data[fieldname] = np.asarray(data)

    if show:
        points.plot(point_size=3, style="points")

    if export_vtk:
        os.makedirs(groupname, exist_ok=True)
        points.save(os.path.join(groupname, f"{filename}.vtp"))

def pv_function_plot(
    x_vis, fun, scale_range=1.0, data_on_z=True, isosurf=False, n_surfaces=20
):
    """
    Convenience function for visualizing/debugging smooth distance functions.

    Args:
      - x_vis (np.ndarray): Array containing the coordinates of nodes for visualization.
      - fun (function): Function to be visualized depending on x.
      - scale_range (float, optional): Scale range for data visualization. Default is 1.0.
      - data_on_z (bool, optional): Whether to plot data on the z-axis. Default is True.
      - isosurf (bool, optional): Whether to plot isosurfaces for structured 3D grids. Default is False.
      - n_surfaces (int, optional): Number of isosurfaces to plot. Default is 20.
    """
    data = jax.jit(jax.vmap(fun, (0), 0))(x_vis)
    if isosurf:
        isosurface(x_vis, data, n_surfaces)
    else:
        pv_plot(
            x_vis,
            data,
            export_vtk=False,
            show=True,
            scale_range=scale_range,
            data_on_z=data_on_z,
        )

def isosurface(x_vis, data, n_surfaces=20):
    """
    Plots isosurfaces for structured 3D data using Plotly.

    Args:
      - x_vis (np.ndarray): Array of coordinates for visualization points.
      - data (np.ndarray): Data to be visualized at the points.
      - n_surfaces (int, optional): Number of isosurfaces to plot. Default is 20.
    """
    import plotly.graph_objects as go

    fig = go.Figure(
        data=go.Isosurface(
            x=x_vis[:, 0],
            y=x_vis[:, 1],
            z=x_vis[:, 2],
            value=data,
            opacity=0.5,
            isomin=jnp.nanmin(data).item(),
            isomax=jnp.nanmax(data).item(),
            colorscale="Rainbow",
            surface_count=n_surfaces,
            caps=dict(x_show=False, y_show=False),
        )
    )
    fig.show()
