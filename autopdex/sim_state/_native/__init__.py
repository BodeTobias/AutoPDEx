"""Native (cffi) acceleration for sim_state mesh topology.

Contains the out-of-line cffi extension ``_topo_hash`` (built from
``_topo_hash.c`` via ``_topo_hash_cffi.py``) used by ``mesh_topology`` /
``mesh_info`` for O(n) entity deduplication, lookup and point merging.
"""
