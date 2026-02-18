# TODO For Next Session: pyAMPP Stage Format Contract Repair

Date: 2026-02-11
Context owner: Gelu + Codex
Status: Planning only (no refactor today)

## Agreed Ground Truth
- GX/IDL is the reference behavior.
- pyAMPP must match GX/IDL conventions (including horizontal component ordering and stage semantics).

## Critical Findings From Today

### 1) Horizontal component swap bug (pyAMPP)
- Symptom observed: direct parity was poor for base `bx/by`, but cross-correlation showed `h5_bx ~ sav_by` and `h5_by ~ sav_bx`.
- Patch already applied locally (not yet finalized upstream in this branch workflow):
  - enforce GX convention: `bx := bp`, `by := -bt`, `bz := br`.
- After rerun with regenerated models, direct correlations improved strongly:
  - HMI: `bx corr ~0.967`, `by corr ~0.975`, `bz corr ~0.996`.

### 2) Stage schema semantic drift (major)
- Current `NAS.GEN` writes a `chromo/` group, but GEN stage is not chromosphere-complete.
- Current `NAS.CHR` replaces/discards explicit `corona/` group instead of preserving stage continuity.
- This is misleading and violates intended stage lineage.

## Required Contract (as requested by Gelu)

### Stage lineage requirements
- `NAS`:
  - Keep `corona/` with `bx, by, bz, dr` (and related coronal products)
- `NAS.GEN`:
  - Must **expand under the same `corona/` category** (add line parameters there), not switch to `chromo/`
- `NAS.CHR`:
  - Must preserve `corona/` from NAS/NAS.GEN
  - Add CHR-specific products under `chromo/` (or another explicit CHR namespace)
  - Do not collapse everything into a single misleading namespace

### Naming semantics
- Do not label GEN payload as `chromo`.
- Keep physical meaning aligned with stage name.

## Scope Impact (known)
- pyAMPP writer/reader stage serialization logic
- pyAMPP viewers (`gxbox-view`, `gxrefmap-view`, and related assumptions)
- gximagecomputing Python branch readers/parsers and example/test workflows
- IDL conversion utilities in gximagecomputing (ConvertToGX / LoadGXmodel assumptions)
- Documentation in both repos (format specs, examples, compatibility notes)
- Tests and parity scripts (expected group paths by stage)

## Concrete Work Plan For Tomorrow

### A) Freeze/define schema first
1. Write explicit stage contract table (NONE/BND/POT/NAS/GEN/CHR).
2. Define exact required/optional groups and datasets for each stage.
3. Mark legacy aliases accepted for backward compatibility.

### B) Implement pyAMPP stage writer corrections
1. Keep `corona/` preserved from NAS through GEN and CHR.
2. Move GEN line outputs into `corona/` expansion.
3. Keep CHR additions in `chromo/` only for CHR-specific data.
4. Preserve backward-readable metadata (`execute`, `id`, `projection`, disambiguation).

### C) Backward compatibility layer
1. Reader normalization for legacy files (old GEN/CHR with single `chromo/`).
2. Explicit warning message for legacy layout, not hard failure.

### D) Validate parity and consistency
1. Base map parity (bx/by/bz) against SAV.
2. Check `base` vs coronal cube z=0 consistency in each stage.
3. Full-cube comparison (pyAMPP vs IDL-converted baseline).
4. Re-check HMI vs SFQ behavior.

### E) gximagecomputing alignment
1. Update model readers to new corrected stage contract.
2. Verify RenderExampleMW.py / render_mw path with both SAV and H5.
3. Re-run Python-vs-IDL render comparisons.

### F) Documentation and release hygiene
1. Update format docs in pyAMPP and gximagecomputing.
2. Add migration note: old misleading GEN/CHR layout -> corrected layout.
3. Add changelog entries and tests that lock the contract.

## Files To Review First Tomorrow
- `pyampp/gxbox/gx_fov2box.py`
- `pyampp/gxbox/gxbox_factory.py`
- `pyampp/gxbox/boxutils.py`
- `pyampp/tests/compare_base_maps.py`
- `gximagecomputing/src/gximagecomputing/io/*`
- `gximagecomputing/src/gximagecomputing/workflows/render_mw.py`
- gximagecomputing IDL conversion routines under `idlcode/`

## Note
- No refactor executed for schema contract today by user request (fatigue/break).
- Resume from this file in next session before coding.
