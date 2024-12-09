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
This example demonstrates the solution of Cook's membrane problem using custom elements from 
the `autopdex.models` library and an exemplary external AceGen-generated C routine.

The main steps include:
    1. Definition of geometry and boundary conditions.
    2. Mesh generation using `pygmsh` and conversion to JAX arrays.
    3. Selection of nodes for boundary conditions and setup of Dirichlet conditions.
    4. Definition of material properties for the neo-Hookean model.
    5. Optional use of an AceGen-generated shared library for custom element computation.
    6. Element definition for hyperelastic steady-state weak form.
    7. Element definition for imposition of Neumann boundary conditions.
    8. Solver settings and configuration using `flax.core.FrozenDict`.
    9. Adaptive load stepping solver for nonlinear analysis.
    10. Export results in VTK format for ParaView.
"""

if __name__ == "__main__":
    import time
    import sys
    import ctypes
    import os

    import jax
    from jax import config
    import numpy as np
    import jax.numpy as jnp
    import flax
    import pygmsh
    import meshio

    from autopdex import seeder, geometry, solver, solution_structures, plotter, utility, models, spaces, assembler

    config.update("jax_enable_x64", True)
    config.update("jax_compilation_cache_dir", './cache')


    ### Definition of geometry and boundary conditions and mesh
    order = 2
    pts = [[0.,0.], [48.,44.], [48.,60.], [0.,44.]]
    with pygmsh.occ.Geometry() as geom:
        region = geom.add_polygon(pts, mesh_size=1.0)
        # region = geom.add_polygon(pts, mesh_size=0.3)
        geom.set_recombined_surfaces([region.surface])
        mesh = geom.generate_mesh(order=order)
        print("Mesh generation finished.")

    # Import mesh
    n_dim = 2
    x_nodes = jnp.asarray(mesh.points[:,:n_dim])
    n_nodes = x_nodes.shape[0]
    elements = jnp.asarray([ v for k, v in mesh.cells_dict.items() if 'quad' in k])[0]
    surface_elements = jnp.asarray([ v for k, v in mesh.cells_dict.items() if 'line' in k])[0]
    print("Number of elements: ", elements.shape[0])
    print("Number of unknowns: ", n_nodes * n_dim)

    # Selection of nodes for boundary condition
    dirichlet_nodes = geometry.in_planes(x_nodes, pts[0], [1.,0.])
    dirichlet_dofs = utility.dof_select(dirichlet_nodes, jnp.asarray([True, True]))
    dirichlet_conditions = jnp.zeros_like(dirichlet_dofs, dtype=jnp.float64)

    # Import surface mesh for inhomogeneous Neumann conditions
    neumann_selection = geometry.select_elements_in_plane(x_nodes, surface_elements, pts[1], [1.,0.])
    neumann_elements= surface_elements[neumann_selection]

    # Definition of material parameters
    lam = 100.
    mu = 40.
    Em = mu * (3 * lam + 2 * mu) / (lam + mu)
    nu = lam / (2 * (lam + mu))


    ### Select element
    element = 'built-in'
    # element = 'acegen'

    if element == 'built-in':
        ### Built-in AutoPDEx element
        weak_form_fun_1 = models.hyperelastic_steady_state_weak(models.neo_hooke, 
                                                            lambda x: Em,
                                                            lambda x: nu, 
                                                            'plain strain')
        user_elem_1 = models.isoparametric_domain_element_galerkin(weak_form_fun_1, 
                                                                spaces.fem_iso_line_quad_brick,
                                                                *seeder.gauss_legendre_nd(dimension = 2, order = 2 * order))
    elif element == 'acegen':
        ### Load the shared library with AceGen generated element 
        # The example element can be generated with the notebook 'external_user_elem_generation.nb' 
        # using AceGen(http://symech.fgg.uni-lj.si/Download.htm) and Mathematica
        # You have to add extern "C" in the beginning of the document, uncomment #include "sms.h" 
        # and move the decleration of the vector v[...] into the function
        # Then, you can compile it e.g. with
        # g++ -shared -o user_elem.dll user_elem.cpp -O2 -s -fPIC
        # and adjust the following pat
        path_to_code = 'examples/cooks_membrane/user_elem.dll'
        lib = ctypes.CDLL(path_to_code)

        # Define the ctypes types corresponding to the function signature
        DoublePtr = ctypes.POINTER(ctypes.c_double)
        Double2DArray = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS')
        Double1DArray = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')

        # Define the function prototype
        lib.user_elem.argtypes = [
            Double2DArray,  # double fI[9][2]
            Double2DArray,  # double xI[9][2]
            DoublePtr,      # double (*Em)
            DoublePtr,      # double (*nu)
            Double1DArray,  # residual
            Double2DArray   # tangent
        ]
        lib.user_elem.restype = None

        # Define a Python callback function
        def my_c_function_callback(fI, xI, Em, nu):
            fI_np = np.ascontiguousarray(np.asarray(fI))
            xI_np = np.ascontiguousarray(np.asarray(xI))

            residual = np.zeros((18,), dtype=np.float64)
            tangent = np.zeros((18, 18), dtype=np.float64)

            # Convert scalars to ctypes pointers
            Em_ct = ctypes.byref(ctypes.c_double(Em))
            nu_ct = ctypes.byref(ctypes.c_double(nu))

            # Call the C function
            lib.user_elem(fI_np, xI_np, Em_ct, nu_ct, residual, tangent)
            return residual, tangent

        def user_elem_1(fI, xI, elem_number, settings, static_settings, mode, set):
            result_shape_dtype = (
                jax.ShapeDtypeStruct(shape=(18,), dtype=np.float64),
                jax.ShapeDtypeStruct(shape=(18, 18), dtype=np.float64)
            )
            arguments = (fI, xI, Em, nu)
            
            # Call the pure_callback with the appropriate arguments
            residual, tangent = jax.pure_callback(my_c_function_callback, result_shape_dtype, *arguments, vmap_method='sequential')

            if mode == 'residual':
                return residual
            elif mode == 'tangent':
                return tangent
            else:
                raise ValueError("User elem mode has to be 'residual' or 'tangent'.")
    else:
        assert False, "Element not set. Use 'acegen' or 'built-in'."

    # Tractions suitable for adaptive load stepping
    q_0 = -1.5e+1
    # traction_fun = lambda x: jnp.asarray([0., q_0])
    traction_fun = lambda x, settings: jnp.asarray([0., settings['load multiplier']])
    weak_form_fun_2 = models.neumann_weak(traction_fun)
    user_elem_2 = models.isoparametric_surface_element_galerkin(weak_form_fun_2, 
                                                            spaces.fem_iso_line_quad_brick,
                                                            *seeder.gauss_legendre_nd(dimension = 1, order = 2 * order),
                                                            tangent_contributions=False)

    ### Settings
    n_fields = 2
    dofs0 = jnp.zeros((n_nodes, n_fields))
    static_settings = flax.core.FrozenDict({
    'number of fields': (n_fields, n_fields),
    'assembling mode': ('user element', 'user element'),
    'solution structure': ('nodal imposition', 'nodal imposition'),
    'model': (user_elem_1, user_elem_2),
    'solver type': 'newton',
    'solver backend': 'pardiso',
    'solver': 'lu',
    'verbose': 1,
    })

    settings = {
    'dirichlet dofs': dirichlet_dofs,
    'connectivity': (elements, neumann_elements),
    'load multiplier': q_0,
    'node coordinates': x_nodes,
    'dirichlet conditions': dirichlet_conditions,
    }


    # ### Timing comparison for computing tangent contributions.
    # start = time.time()
    # tangent_contributions = assembler.user_element_assemble_tangent(dofs0, settings, static_settings, 0)
    # print("Compile time: ", time.time() - start)
    # start = time.time()
    # for i in range(10):
    #     tangent_contributions = assembler.user_element_assemble_tangent(dofs0, settings, static_settings, 0)
    # print("Time needed: ", time.time() - start)
    # sys.exit()


    # Adaptive load stepping
    def multiplier_settings(settings, multiplier):
        settings['load multiplier'] = multiplier * q_0
        return settings
    
    dofs = solver.adaptive_load_stepping(dofs = dofs0, 
                                        settings = settings, 
                                        static_settings = static_settings,
                                        multiplier_settings = multiplier_settings,
                                        path_dependent = False)[0]

    ### Postprocessing
    points = mesh.points
    cells = jnp.asarray([ v for k, v in mesh.cells_dict.items() if 'quad' in k])[0]
    mesh = meshio.Mesh(
        points,
        {'quad': cells[:,:4]},
        point_data={
            "u": dofs,
        },
    )
    mesh.write("cooks_membrane.vtk")
