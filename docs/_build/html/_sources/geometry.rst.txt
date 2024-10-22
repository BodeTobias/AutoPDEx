Geometry
========

.. automodule:: autopdex.geometry
    :no-index:

Rvachev function operations
---------------------------
.. autosummary::
   :toctree: _autosummary

   r_equivalence
   r_conjunction
   r_disjunction
   r_trimming
   signed_to_positive
   only_positive
   first_order_normalization
   normals_from_normalized_sdf

Signed distance functions
-------------------------
.. autosummary::
   :toctree: _autosummary

   sdf_infinite_line
   sdf_nd_sphere
   sdf_nd_planes
   sdf_normal_to_direction
   sdf_infinite_cylinder
   sdf_cylinder
   sdf_cylinder_extruded
   sdf_triangle_2d
   sdf_convex_polygon_2d
   sdf_infinite_triprism_3d
   sdf_parallelogram_2d
   sdf_cuboid

Positive smooth distance functions
----------------------------------
.. autosummary::
   :toctree: _autosummary

   psdf_infinite_line
   psdf_trimmed_line
   psdf_nd_sphere
   psdf_nd_sphere_cutout
   psdf_infinite_cylinder
   psdf_cylinder
   psdf_cylinder_extruded
   psdf_triangle_2d
   psdf_infinite_triprism_3d
   psdf_parallelogram
   psdf_arc_in_2d
   psdf_polygon

Helper functions
----------------
.. autosummary::
   :toctree: _autosummary

   centroid
   normal
   project
   project_on_line

Transfinite interpolation
-------------------------
.. autosummary::
   :toctree: _autosummary

   transfinite_interpolation
   psdf_unification

Mesh related functions
----------------------
.. autosummary::
   :toctree: _autosummary

   triangle_area
   triangle_areas
   in_sdf
   in_sdfs
   select_in_sdfs
   in_plane
   in_planes
   select_in_plane
   on_point
   select_point
   on_line
   on_lines
   select_on_line
   elem_in_plane
   select_elements_in_plane
   elem_on_line
   select_elements_on_line
   subelem_in_elem
   subelem_in_elems
   subelems_in_elems

