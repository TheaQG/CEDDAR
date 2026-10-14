# Compact manuscript figure-data archive

This workflow runs the existing figure runner, traces its actual file reads, and
copies only those saved inputs into a separate archive. It never invokes model
inference or recomputes evaluation products. Model, metrics, plotting code and
runner arguments remain unchanged.

Use an activated figure environment with the repository requirements and
`pandas`. All work directories and the staged archive must be new locations
outside the Git checkout and full input tree. The interpreter used to invoke the
tool is also used by every figure command. Paths below are examples to replace.

```bash
python repro/manuscript_figure_data.py trace \
  --data-root /path/to/P1_figures_data \
  --work-dir /path/to/work/source-run

python repro/manuscript_figure_data.py stage \
  --trace /path/to/work/source-run/trace.json \
  --destination /path/to/work/CEDDAR_manuscript_figure_data

python repro/manuscript_figure_data.py verify \
  --archive /path/to/work/CEDDAR_manuscript_figure_data \
  --work-dir /path/to/work/compact-run \
  --forbid-root /path/to/P1_figures_data \
  --reference-output /path/to/work/source-run/figures

python repro/manuscript_figure_data.py pack \
  --archive /path/to/work/CEDDAR_manuscript_figure_data \
  --output /path/to/work/CEDDAR_manuscript_figure_data.tar.gz
```

The trace and verification each run all 12 Python-generated figures (main 2-7
and supplementary S5-S10), producing PNG and PDF outputs. Figure 1 and S1-S4 were
prepared manually. Read `runner.log` if a run fails; missing inputs are diagnosed
without changing any scientific definitions or figures.

The opt-in `figure_data_audit/sitecustomize.py` hook records Python file opens
and directory scans. During verification, it rejects access to `--forbid-root`
(repeatable) and rejects writes to figure inputs. The source tree is left in
place and unchanged. This is a Python audit guard for these known loaders,
not an operating-system sandbox for arbitrary native code. `trace.json` lists
all external reads so that source imports, installed packages, fonts and other
runtime resources can be reviewed separately from archived figure data. Each
run uses its own output, Matplotlib and XDG cache directories.

The tool also resolves the runner's shell `require_file` and `require_dir`
checks. Shell-only inputs, such as the ISS threshold-count table, are included
even when no Python loader opens them.

Staging checks that the sources still match their traced SHA256 values and copies
files byte-for-byte. Unchanged loaders scan some directories broadly, so the
archive retains required tables and metadata. The source run logs identify the
two sigma* bundles actually selected by the final figures; staging omits the
other candidates and records those exclusions in provenance.json. Verification
must still reproduce the same outputs. It excludes unread source
files, old rendered figures, caches and other evaluation families. An empty
bilinear directory is kept solely for the runner's existence check.

The archive contains:

- `README.md`: scope, manual figures, separate archives and plotting command.
- `MANIFEST.csv`: relative paths, bytes, descriptions, generator entry points,
  figure consumers and SHA256. Saved metadata remains the historical authority;
  an entry-point name is not a recovered original command.
- `provenance.json`: source commit, dirty status, runner hash and trace environment.
- `checksums.sha256`: every archive file except the checksum file itself.
- Only traced files under `P1_F_samples`, `P1_legacy`, `P1_revision`, `P1_sigma_star`.

Verification checks archive integrity, compares observed reads with the manifest,
and requires all 24 expected outputs. With `--reference-output`, it requires exact
PNG equality against the fresh full-data run and PDF content equality after
removing only CreationDate/ModDate metadata. Raw PDF byte comparisons are also
recorded. Undeclared external CSV/NPZ/NetCDF/JSON/YAML data reads fail verification;
code and installed-package resources are allowed. The separate
`verification.json` and run logs stay
outside the data archive; preserve them with the release preparation records.

For ordinary plotting after extraction, no audit tool is required:

```bash
DATA_ROOT=/path/to/CEDDAR_manuscript_figure_data \
OUT=/path/to/figures \
bash bash_paper1/run_manuscript_figures.sh
```

Keep large data archives outside Git. Packaging preserves the empty directories
required by the runner and refuses to overwrite an existing tarball.
