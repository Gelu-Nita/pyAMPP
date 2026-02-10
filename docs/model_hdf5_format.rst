pyAMPP HDF5 Model Format
========================

This page describes the upgraded HDF5 stage files produced by ``gx-fov2box`` and consumed by downstream tools.

Common groups
-------------

All saved stages carry:

- ``base``:
  - ``bx``, ``by``, ``bz`` (2D)
  - ``ic`` (2D)
  - ``chromo_mask`` (2D)
  - ``index`` (FITS-like header string; canonical base-map header)
- ``refmaps``:
  - per map: ``refmaps/<map_id>/data`` and ``refmaps/<map_id>/wcs_header``
- ``metadata``:
  - ``id`` (stage id string)
  - ``execute`` (full command provenance)
  - ``disambiguation`` (e.g. HMI/SFQ)
  - ``projection`` (CEA/TOP)

Each refmap carries its own ``wcs_header`` by design, so maps can be reconstructed independently.

Stage-specific groups
---------------------

NONE
~~~~

- ``corona`` with zero-field placeholders and ``attrs/model_type = "none"``
- ``grid`` with:
  - ``voxel_id`` (uint32)
  - ``dx``, ``dy``, ``dz``

BND
~~~

- ``bounds`` group with boundary maps:
  - ``bx``, ``by``, ``bz``, ``dr``

POT
~~~

- ``corona`` group with potential extrapolation:
  - ``bx``, ``by``, ``bz``, ``dr``
  - ``attrs/model_type = "pot"``

NAS
~~~

- ``corona`` group with NLFFF extrapolation:
  - ``bx``, ``by``, ``bz``, ``dr``
  - ``attrs/model_type = "nlfff"``

NAS.GEN
~~~~~~~

- ``chromo`` includes generated line metadata needed for CHR completion:
  - ``codes``, ``apex_idx``, ``start_idx``, ``end_idx``, ``seed_idx``
  - ``av_field``, ``phys_length``, ``voxel_status``

NAS.CHR
~~~~~~~

- ``chromo`` full chromospheric model payload (e.g. ``bcube``, ``chromo_bcube``, ``chromo_*``, ``dz``, etc.)
- ``grid`` with final:
  - ``voxel_id`` (uint32)
  - ``dx``, ``dy``, ``dz``

Dimension conventions
---------------------

- ``chromo/dz`` and ``grid/voxel_id`` use the same order: ``(nx, ny, nz)``.
- Internal renderer arrays may be reordered for ABI compatibility; file-level convention remains ``(nx, ny, nz)``.

IDL compatibility notes
-----------------------

- ``base/index`` is intended to be IDL-compatible metadata for base map geometry.
- ``refmaps/*/wcs_header`` stores per-map WCS headers so IDL/Python viewers can reconstruct map coordinates.
