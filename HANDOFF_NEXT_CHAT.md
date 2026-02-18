# Handoff: Next Chat Resume Point (pyAMPP)

## 1) Current Context
- Repository: `pyAMPP`
- Working branch: `feat/fix-pyampp-gxmodel-mismatches`
- Goal in progress: implement and validate the **new stage-aware HDF5 schema** agreed in `pyampp/tests/idl_hdf5_mapping.md`.

## 2) Implemented So Far (Not Committed Yet)
### Schema behavior
- `BND` stage writes magnetic products under `corona/` (not `bounds/`).
- `NAS.GEN` writes line products under `lines/`.
- `NAS.CHR` writes `corona + lines + chromo`.
- `chromo/` appears only at CHR stage.
- Persist explicit vector components (no bcube persistence):
  - `corona/bx`, `corona/by`, `corona/bz`
  - `chromo/bx`, `chromo/by`, `chromo/bz` (at CHR)
- `grid/voxel_id` persisted at stages with `corona`.
- `corona/corona_base` persisted at stages with `corona`.

### Voxel ID / corona base logic
- `gx_box2id` updated to optionally return `(voxel_id, corona_base)`.
- Robust handling added for flattened `start_idx` (IDL-style 1D) when inferring `corona_base`.

### Compatibility and tooling updates
- `read_b3d_h5` legacy normalization maps `bounds -> corona`.
- Viewer dimension inference updated to support new explicit vector fields.
- parity utilities updated for `lines/` + `corona/corona_base` mapping.

## 3) Files Modified (Uncommitted)
- `pyampp/gxbox/gx_fov2box.py`
- `pyampp/gxbox/gxbox_factory.py`
- `pyampp/gxbox/gx_box2id.py`
- `pyampp/gxbox/boxutils.py`
- `pyampp/gxbox/view_h5.py`
- `pyampp/tests/build_h5_from_sav.py`
- `pyampp/tests/compare_idl_hdf5.py`
- `pyampp/tests/idl_hdf5_mapping.md`

## 4) Regenerated Test Models (new schema)
### HMI
- `/tmp/pyampp_models_newschema/2025-11-26/hmi.M_720s.20251126_154752.W28S12CR.CEA.NAS.h5`
- `/tmp/pyampp_models_newschema/2025-11-26/hmi.M_720s.20251126_154752.W28S12CR.CEA.NAS.GEN.h5`
- `/tmp/pyampp_models_newschema/2025-11-26/hmi.M_720s.20251126_154752.W28S12CR.CEA.NAS.CHR.h5`

### SFQ
- `/tmp/pyampp_models_sfq_newschema/2025-11-26/hmi.M_720s.20251126_154752.W28S12CR.CEA.NAS.h5`
- `/tmp/pyampp_models_sfq_newschema/2025-11-26/hmi.M_720s.20251126_154752.W28S12CR.CEA.NAS.GEN.h5`
- `/tmp/pyampp_models_sfq_newschema/2025-11-26/hmi.M_720s.20251126_154752.W28S12CR.CEA.NAS.CHR.h5`

## 5) Verified Already
- CHR files contain: `base`, `corona`, `lines`, `chromo`, `grid`, `metadata`, `refmaps`.
- `corona/corona_base` and `grid/voxel_id` present.
- Legacy persisted `bcube/chromo_bcube` absent from output H5.

## 6) Pending Next Actions (Start Here)
1. Run the same schema validation for NAS and NAS.GEN (both HMI and SFQ), and record a concise per-stage checklist output.
2. Copy selected regenerated CHR files into `gximagecomputing/test_data/` if needed for downstream parity checks.
3. Re-run cross-comparison scripts (HMI and SFQ) using the new schema and confirm no reader breakage in `gximagecomputing`.
4. After user review, prepare a single clean commit (or logical split commits) on this branch.

## 7) Important User Direction
- Do **not** change data format again without explicit consultation and doc alignment.
- `idl_hdf5_mapping.md` is the source-of-truth for schema decisions.

## 8) Notes
- Some untracked artifacts may exist from prior testing (images/reports); do not commit artifacts unless explicitly requested.
