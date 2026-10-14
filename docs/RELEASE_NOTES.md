# CEDDAR software v1.1.0

Prepared release; publication and the revised-source DOI are pending.

This software release accompanies the revised GMD manuscript **“CEDDAR v1.0.2:
Evaluating stochastic diffusion-based precipitation downscaling beyond pointwise
skill”**. The manuscript's model identity remains v1.0.2. Trained weights and
architecture are unchanged. Software v1.1.0 includes the previously audited
sigma* implementation fixes and the revised evaluation framework, distinguishing
it from the original v1.0.2 source archive. The repository cleanup preserves those
scientific implementations and the final figure settings.

## Changes

- Retain the completed sigma* audit fixes, sampling/initial-state provenance and
  useful investigation workflows, with historical settings distinguished.
- Add revised deterministic, probabilistic, climatological, spatial and temporal
  evaluation, benchmark comparisons, common-event ISS and sigma* uncertainty panels.
- Provide a portable runner for Figures 2–7 and S5–S10 and tooling for a compact
  derived-data archive with a manifest and SHA256 checksums. Figure 1 and S1–S4
  were prepared manually.
- Refresh installation and reproduction instructions; remove superseded plotting
  layouts, a preview script, an unimplemented smoke placeholder and obsolete notes.

## Validation and known limitation

All 12 Python-generated manuscript figures reproduced from the compact archive.
PNGs match the existing final outputs byte-for-byte; PDFs match apart from creation
and modification timestamps. The run had no attempted access to the original
full-data tree and no undeclared data reads. The archive contains 118 data files
and is 21,787,598 bytes compressed.

The retained suite runs 71 tests: 70 pass and one is an expected failure.
`test_strict_metrics_reject_missing_arrays` records a pre-existing limitation:
the frozen sigma* evaluator skips missing HR observations before its strict-input
guard, and can write header-only outputs if no dates remain. Recomputed metrics
need complete physical inputs and an evaluated-date check. Final manuscript
plotting reads saved products and does not call this evaluator. See the
[sigma* validation notes](SIGMA_STAR_REVISION.md).

A fresh Python 3.11 installation passed `pip check`, the retained test suite and
the synthetic CPU smoke, and generated all 12 figures. The default pip rendering
builds produced different figure bytes and text rendering. A separate clean Intel
macOS environment with the reference rendering builds reproduced all 12 PNGs
exactly and PDFs apart from timestamps, with no undeclared data reads. The tested
[environment specification and installation instructions](../repro/README.md#environment)
are provided for exact rendering on that platform.

## Separate archives

- [Original v1.0.2 source](https://doi.org/10.5281/zenodo.18643186).
- [Trained models and example data](https://doi.org/10.5281/zenodo.18643307).
- [Compact manuscript figure data](https://doi.org/10.5281/zenodo.23369324):
  published. The derived figure data do not replace DANRA or ERA5 source data.
- [Manuscript preprint](https://doi.org/10.5194/egusphere-2026-1139).
  The final GMD article DOI is pending.
- Revised v1.1.0 source DOI: pending.

See the [figure-data guide](../repro/MANUSCRIPT_FIGURE_DATA.md) for archive
contents, checksums and reproduction instructions.
