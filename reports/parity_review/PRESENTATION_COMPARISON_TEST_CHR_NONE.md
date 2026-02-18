# pyAMPP vs IDL Comparison Report (Presentation)

- Generated: 2026-02-16 22:58
- Reference SAV: `/Users/gelu/Library/CloudStorage/Dropbox/@Projects/@SUNCAST-ORG/gximagecomputing/test_data/test.chr.sav`

## 1. OBS-Inferred Model vs SAV

- H5: `/Users/gelu/Library/CloudStorage/Dropbox/@Projects/@SUNCAST-ORG/gximagecomputing/test_data/test.chr.h5`
- `metadata/lineage`: `OBS.NONE.POT.BND.NAS.GEN.CHR`
- `metadata/execute`: `gx-fov2box --time 2025-11-26T15:34:31 ... --entry-box ...GEN.h5 --rebuild`

### Plots

![OBS-inferred base](reports/parity_review/figures/presentation_test_chr_h5_vs_sav_base_3x4.png)

![OBS-inferred z0](reports/parity_review/figures/presentation_test_chr_h5_vs_sav_z0_3x4.png)

### Volumetric Metrics (Coronal Cubes vs SAV)

| Component | MAE | RMSE | Max Abs | Corr |
|---|---:|---:|---:|---:|
| BX | 3.32961 | 8.69617 | 372.349 | 0.961160 |
| BY | 2.86350 | 8.36093 | 310.275 | 0.956035 |
| BZ | 2.86396 | 7.46776 | 309.955 | 0.990151 |

## 2. NONE.SAV-Inferred Model vs SAV

- H5: `/Users/gelu/Library/CloudStorage/Dropbox/@Projects/@SUNCAST-ORG/gximagecomputing/test_data/test.chr.none.sav.h5`
- `metadata/lineage`: `ENTRY.NAS.CHR.SAV->GEN.CHR.h5`
- `metadata/execute`: `gx-fov2box ... --entry-box .../test.chr.sav --rebuild-from-none`

### Plots

![NONE.SAV-inferred base](reports/parity_review/figures/presentation_test_chr_none_sav_h5_vs_sav_base_3x4.png)

![NONE.SAV-inferred z0](reports/parity_review/figures/presentation_test_chr_none_sav_h5_vs_sav_z0_3x4.png)

### Volumetric Metrics (Coronal Cubes vs SAV)

| Component | MAE | RMSE | Max Abs | Corr |
|---|---:|---:|---:|---:|
| BX | 1.47131 | 3.90397 | 242.464 | 0.992241 |
| BY | 1.45112 | 4.15369 | 198.583 | 0.989469 |
| BZ | 0.968022 | 3.59781 | 236.185 | 0.997671 |

## 3. Conclusions

The NONE.SAV-inferred build is closer to the IDL SAV reference in 3D volumetric metrics (lower MAE/RMSE and higher correlations for all three components) than the OBS-inferred build.

- OBS-inferred (`OBS.NONE.POT.BND.NAS.GEN.CHR`) is still highly correlated with SAV, but with larger residuals.
- NONE.SAV-inferred (`ENTRY.NAS.CHR.SAV->GEN.CHR.h5`) gives the strongest parity against the IDL SAV reference for this case.

## Artifacts

- `reports/parity_review/PRESENTATION_COMPARISON_TEST_CHR_NONE.md`
- `reports/parity_review/PRESENTATION_COMPARISON_TEST_CHR_NONE.pdf`
- `reports/parity_review/data/presentation_test_chr_h5_vs_sav_cube_metrics.json`
- `reports/parity_review/data/presentation_test_chr_none_sav_h5_vs_sav_cube_metrics.json`
- `reports/parity_review/figures/presentation_test_chr_h5_vs_sav_base_3x4.png`
- `reports/parity_review/figures/presentation_test_chr_h5_vs_sav_z0_3x4.png`
- `reports/parity_review/figures/presentation_test_chr_none_sav_h5_vs_sav_base_3x4.png`
- `reports/parity_review/figures/presentation_test_chr_none_sav_h5_vs_sav_z0_3x4.png`
