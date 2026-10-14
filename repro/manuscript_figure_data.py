"""Trace, stage and verify the saved data read by the final manuscript runner.

Only the figure subprocesses import scientific packages. This tool copies saved
files byte-for-byte and never runs evaluation/data-generation entry points.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile


REPO = Path(__file__).resolve().parent.parent
RUNNER = REPO / "bash_paper1/run_manuscript_figures.sh"
HOOK = REPO / "repro/figure_data_audit"
FIGURES = {
    "fig02_samples_distributions": "Figure 2",
    "fig03_deterministic": "Figure 3",
    "fig04_dry_bias": "Figure 4",
    "fig05_probabilistic": "Figure 5",
    "fig06_spatial_morphology": "Figure 6",
    "fig_sigma_star_outline": "Figure 7",
    "figS05_sigma_star_extended_validation_outline": "Figure S5",
    "figS06_sigma_star_uncertainty": "Figure S6",
    "figS07_iss": "Figure S7",
    "figS08_seasonal_crps": "Figure S8",
    "figS09_temporal_persistence": "Figure S9",
    "figS10_daily_mean_bias": "Figure S10",
}
MODULE_FIGURES = {
    "fig_samples_distributions": "Figure 2", "fig_deterministic": "Figure 3",
    "fig_dry_bias": "Figure 4", "fig_probabilistic": "Figure 5",
    "fig_spatial_morphology": "Figure 6", "figS_sigma_star_uncertainty": "Figure S6",
    "figS_iss": "Figure S7", "figS_seasonal_crps": "Figure S8",
    "figS_temporal_persistence": "Figure S9", "figS_daily_mean_bias": "Figure S10",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pdf_content_digest(path):
    # Matplotlib embeds the run time in PDF metadata; leave every other byte.
    data = Path(path).read_bytes()
    data = re.sub(rb"/(CreationDate|ModDate)\s*\([^)]*\)", rb"/\1 (timestamp omitted)", data)
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def environment():
    versions = {}
    for name in ("numpy", "scipy", "matplotlib", "pandas", "torch", "PyYAML", "Pillow"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return dict(platform=platform.platform(), executable=sys.executable, packages=versions)


def figure_label(module, output):
    if module.endswith(".fig_sigma_star"):
        return "Figure S5" if Path(output).name.startswith("figS05") else "Figure 7"
    return MODULE_FIGURES.get(module.rsplit(".", 1)[-1], "runtime helper")


def shell_preflight(data_root):
    """Resolve the runner's require_file/require_dir calls without running plots.

    The current configuration preamble only assigns paths and defines functions;
    stop before cd/mkdir and ask Bash to expand the existing check arguments.
    """
    text = RUNNER.read_text()
    preamble = text.split('cd "$REPO"', 1)[0]
    checks = re.findall(r'^require_(file|dir) "([^"\n]+)"$', text, flags=re.MULTILINE)
    commands = [f'printf "%s\\t%s\\n" "{kind}" "{expression}"' for kind, expression in checks]
    env = {**os.environ, "REPO": str(REPO), "DATA_ROOT": str(data_root), "OUT": "/unused"}
    result = subprocess.check_output(["bash", "-c", preamble + "\n" + "\n".join(commands)], env=env, text=True)
    paths = []
    for line in result.splitlines():
        kind, value = line.split("\t", 1)
        path = Path(value).resolve()
        if not path.is_relative_to(data_root):
            raise ValueError(f"Runner preflight uses a path outside DATA_ROOT: {path}")
        if not (path.is_file() if kind == "file" else path.is_dir()):
            raise FileNotFoundError(f"Runner preflight requires {kind}: {path}")
        paths.append((kind, path.relative_to(data_root).as_posix()))
    return paths


def lineage(relative):
    """Identify repository entry points; do not claim recovered run provenance."""
    name = Path(relative).name
    if relative.startswith("P1_F_samples/"):
        return "Saved physical example fields or land mask", "sbgm.cli.launch_generation (archived products)"
    if relative.startswith("P1_sigma_star/"):
        return "Saved sigma* evaluation tables, metadata or example bundle", "sbgm.cli.launch_evaluation_sigma_star (archived products)"
    generators = {
        "daily_spatial_correlation.csv": "generate_deterministic_correlation.py",
        "spread_skill_daily.csv": "generate_daily_spread_skill.py",
        "ceddar_member_histograms.npz": "generate_ensemble_hist.py",
        "qm_land_histograms.npz": "generate_qm_hist.py",
    }
    if name in generators:
        return "Derived manuscript diagnostic", "revision_manuscript_plots/data/" + generators[name]
    if "/derived_spatial/" in relative:
        return "Saved 2019 CEDDAR accumulation/mean map", "revision_manuscript_plots/data/generate_spatial_summaries.py"
    for group, generator in (("daily_mean_bias", "generate_daily_mean_bias.py"), ("iss", "generate_iss_common_events.py")):
        if f"/evaluation/{group}/" in relative:
            return f"Saved {group} diagnostic or metadata", "revision_manuscript_plots/data/" + generator
    for group in ("deterministic", "dry_bias", "probabilistic", "morphology"):
        if f"/evaluation/{group}/" in relative:
            return f"Saved {group} evaluation table or provenance", "revision_evaluation/run_" + group + ".py (see saved metadata)"
    return "Saved legacy evaluation table or provenance", "sbgm.evaluate (exact historical invocation: see saved metadata)"


def run_audited(data_root, work, forbidden):
    data_root = data_root.resolve(strict=True)
    work = work.resolve()
    if work.is_relative_to(data_root) or work.is_relative_to(REPO):
        raise ValueError("Use a new work directory outside the data and source trees")
    work.mkdir(parents=True, exist_ok=False)
    audit_dir = work / "audit"
    audit_dir.mkdir()
    config = work / "audit_config.json"
    write_json(config, {"data_root": str(data_root), "audit_dir": str(audit_dir),
                        "forbidden_roots": [str(p.resolve()) for p in forbidden]})
    env = dict(os.environ)
    # Use the selected interpreter even though the unchanged runner calls 'python'.
    binary = work / "bin"
    binary.mkdir()
    (binary / "python").symlink_to(sys.executable)
    env.update(REPO=str(REPO), DATA_ROOT=str(data_root), OUT=str(work / "figures"),
               PATH=str(binary) + os.pathsep + env.get("PATH", ""),
               PYTHONPATH=os.pathsep.join((str(HOOK), str(REPO))),
               PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
               MPLCONFIGDIR=str(work / "matplotlib"), XDG_CACHE_HOME=str(work / "cache"),
               CEDDAR_FIGURE_AUDIT_CONFIG=str(config), PYTHONUNBUFFERED="1")
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONOPTIMIZE", None)
    with (work / "runner.log").open("w") as log:
        process = subprocess.run(["bash", str(RUNNER)], cwd=work, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
    return collect_run(data_root, work, process.returncode)


def collect_run(data_root, work, returncode):
    """Collect audit records, including Python runtime helper subprocesses."""
    audit_dir = work / "audit"
    reads, external, violations, starts = {}, set(), [], []
    for path in sorted(audit_dir.glob("*.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        start = records[0]
        starts.append(start)
        label = figure_label(start["module"], start["output"])
        for event in records[1:]:
            if event["denied"]:
                violations.append(event)
            if event["event"] != "open" or event["writing"]:
                continue
            source = Path(event["path"])
            if not source.is_file():
                continue
            if source.is_relative_to(data_root):
                relative = source.relative_to(data_root).as_posix()
                reads.setdefault(relative, set()).add(label)
            elif not source.is_relative_to(work):
                external.add(str(source))
    preflight = shell_preflight(data_root)
    preflight_only = []
    for kind, relative in preflight:
        if kind == "file" and relative not in reads:
            preflight_only.append(relative)
            reads[relative] = {"Runner preflight (existence check)"}
    dependencies = []
    for relative, labels in sorted(reads.items()):
        source = data_root / relative
        description, generator = lineage(relative)
        dependencies.append(dict(relative_path=relative, size_bytes=source.stat().st_size,
                                 description=description, generated_by=generator,
                                 used_by_figure="; ".join(sorted(labels)), sha256=digest(source)))
    expected = [stem + suffix for stem in FIGURES for suffix in (".png", ".pdf")]
    missing = [name for name in expected if not (work / "figures" / name).is_file()]
    figure_starts = [s for s in starts if s["module"].startswith("revision_manuscript_plots.figures.")]
    selected_bundles = []
    for value in re.findall(r"^Bundle: (.+)$", (work / "runner.log").read_text(), flags=re.MULTILINE):
        selected_bundles.append(Path(value).resolve().relative_to(data_root).as_posix())
    runtime_roots = (REPO, Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve())
    external_data = [p for p in sorted(external) if Path(p).suffix.lower() in
                     (".csv", ".npz", ".npy", ".nc", ".json", ".yaml", ".yml")
                     and not any(Path(p).is_relative_to(root) for root in runtime_roots)]
    report = dict(data_root=str(data_root), work_dir=str(work), returncode=returncode,
                  runner_sha256=digest(RUNNER), git_commit=git("rev-parse", "HEAD"),
                  git_status=git("status", "--short"), python=sys.version, environment=environment(),
                  figure_processes=figure_starts, helper_processes=[s for s in starts if s not in figure_starts], dependencies=dependencies,
                  required_directories=[p for kind, p in preflight if kind == "dir"], preflight_only=preflight_only,
                  selected_example_bundles=selected_bundles,
                  external_reads=sorted(external), violations=violations, missing_outputs=missing,
                  external_data_reads=external_data,
                  output_checksums={name: digest(work / "figures" / name) for name in expected if name not in missing})
    report["success"] = returncode == 0 and len(figure_starts) == 12 and not missing and not violations and not external_data
    write_json(work / "trace.json", report)
    if not report["success"]:
        raise RuntimeError(f"Figure run failed; inspect {work / 'runner.log'} and trace.json")
    return report


ARCHIVE_README = """# CEDDAR manuscript figure data

Saved evaluation and figure products for the revised GMD manuscript:
**CEDDAR v1.0.2: Evaluating stochastic diffusion-based precipitation downscaling
beyond pointwise skill**.

These derived products reproduce the Python-generated main Figures 2-7 and
Supplementary Figures S5-S10. Figure 1 and S1-S4 were prepared manually and are
not generated from this archive. This archive does not replace DANRA or ERA5
source data. Source code is released separately, as are trained models and the
model reproducibility dataset. No weights or training datasets are included.

## Reproduction

Use the matching CEDDAR source checkout and an activated figure environment:

```bash
DATA_ROOT=/absolute/path/to/CEDDAR_manuscript_figure_data \\
OUT=/absolute/path/to/output \\
bash /absolute/path/to/CEDDAR/bash_paper1/run_manuscript_figures.sh
```

The environment needs the repository requirements plus pandas. Consult the
source repository's reproduction instructions for environment details. The
provenance.json records the exact environment used for the dependency trace.

The P1_F_samples, P1_legacy, P1_revision and P1_sigma_star directories preserve
the original relative input layout. Files are copied byte-for-byte; no scientific
values, dates, ensemble members, thresholds or examples are recomputed or reduced.
Broad file scans in the unchanged loaders may read tables and example bundles
that are not displayed. Required tables and provenance are retained. For sigma*,
only the two bundles actually selected by the full-data figure run are retained;
unselected candidate bundles are omitted, with output equality checked separately.
Shell-only preflight dependencies are also retained: the ISS threshold-count
table and an empty bilinear evaluation directory are checked for existence.

MANIFEST.csv records each data file's size, description, repository generator
entry point, figure consumers and SHA256. Generator names identify code entry
points, not proof of the historical command used; consult original metadata.
Original metadata may contain historical local/HPC paths. These are provenance
records, not required filesystem locations for plotting.

Verify all data and documentation with `shasum -a 256 -c checksums.sha256`.
checksums.sha256 covers every archive file except itself.

## Separate releases

- Archived v1.0.2 source: https://doi.org/10.5281/zenodo.18643186
- Trained models and example data: https://doi.org/10.5281/zenodo.18643307
- Revised source release and manuscript-figure-data DOI: to be assigned.

See provenance.json for the source commit, runner hash and figure interpreter.
"""


def safe_relative(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe manifest path: {value}")
    return path


def check_archive(archive):
    rows = list(csv.DictReader((archive / "MANIFEST.csv").open(newline="")))
    for row in rows:
        path = archive / safe_relative(row["relative_path"])
        if path.stat().st_size != int(row["size_bytes"]) or digest(path) != row["sha256"]:
            raise ValueError(f"Manifest mismatch: {path}")
    covered = set()
    for line in (archive / "checksums.sha256").read_text().splitlines():
        checksum, relative = line.split("  ", 1)
        path = archive / safe_relative(relative)
        if digest(path) != checksum:
            raise ValueError(f"Checksum mismatch: {path}")
        covered.add(relative)
    actual = {p.relative_to(archive).as_posix() for p in archive.rglob("*") if p.is_file()}
    if covered != actual - {"checksums.sha256"}:
        raise ValueError("Checksums must cover every archive file except checksums.sha256")
    return rows


def stage(trace_path, destination):
    report = json.loads(trace_path.read_text())
    if not report["success"]:
        raise ValueError("Stage only from a successful audited figure run")
    source = Path(report["data_root"]).resolve(strict=True)
    destination = destination.resolve()
    if destination.is_relative_to(source) or destination.is_relative_to(REPO):
        raise ValueError("Stage outside the full data tree and Git checkout")
    destination.mkdir(parents=True, exist_ok=False)
    selected = set(report["selected_example_bundles"])
    omitted = []
    rows = []
    for row in report["dependencies"]:
        path = Path(row["relative_path"])
        candidate = path.parts[0] == "P1_sigma_star" and path.parent.name in ("selected_examples", "selected_samples") and path.suffix == ".npz"
        if selected and candidate and row["relative_path"] not in selected:
            omitted.append(row["relative_path"])
        else:
            rows.append(row)
    for row in rows:
        relative = safe_relative(row["relative_path"])
        path = source / relative
        if path.stat().st_size != row["size_bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"Source changed since trace: {path}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    # Empty input directories can be required by the shell's preflight checks.
    for item in report["required_directories"]:
        (destination / item).mkdir(parents=True, exist_ok=True)
    (destination / "README.md").write_text(ARCHIVE_README)
    with (destination / "MANIFEST.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["relative_path", "size_bytes", "description", "generated_by", "used_by_figure", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    provenance = {key: report[key] for key in ("git_commit", "git_status", "runner_sha256", "python", "environment")}
    provenance.update(selected_example_bundles=sorted(selected), omitted_unselected_bundles=omitted)
    write_json(destination / "provenance.json", provenance)
    with (destination / "checksums.sha256").open("w") as stream:
        for path in sorted(destination.rglob("*")):
            if path.is_file() and path.name != "checksums.sha256":
                stream.write(f"{digest(path)}  {path.relative_to(destination).as_posix()}\n")
    check_archive(destination)
    return dict(destination=str(destination), data_files=len(rows),
                size_bytes=sum(p.stat().st_size for p in destination.rglob("*") if p.is_file()))


def verify(archive, work, forbidden, reference):
    rows = check_archive(archive)
    report = run_audited(archive, work, forbidden)
    declared = {row["relative_path"] for row in rows}
    observed = {row["relative_path"] for row in report["dependencies"]}
    if observed != declared:
        raise ValueError(f"Archive read mismatch: undeclared={observed-declared}; unused={declared-observed}")
    comparison = {}
    if reference:
        for name, checksum in report["output_checksums"].items():
            path = reference / name
            comparison[name] = dict(reference_exists=path.is_file(),
                                    byte_identical=path.is_file() and digest(path) == checksum)
            if name.endswith(".pdf"):
                comparison[name]["content_identical_except_timestamps"] = path.is_file() and pdf_content_digest(path) == pdf_content_digest(work / "figures" / name)
        # PDF files may contain run-time CreationDate metadata; PNGs have no such date.
        mismatches = [name for name, result in comparison.items() if name.endswith(".png") and not result["byte_identical"]]
        report["comparison"] = comparison
        report["png_mismatches"] = mismatches
        pdf_mismatches = [name for name, result in comparison.items() if name.endswith(".pdf") and not result["content_identical_except_timestamps"]]
        report["pdf_content_mismatches"] = pdf_mismatches
        write_json(work / "verification.json", report)
        if mismatches or pdf_mismatches:
            raise ValueError(f"Figure output mismatch: {mismatches + pdf_mismatches}; inspect verification.json")
    else:
        write_json(work / "verification.json", report)
    check_archive(archive)
    return dict(success=True, figures=12, outputs=len(report["output_checksums"]),
                declared_files=len(rows), violations=len(report["violations"]),
                report=str(work / "verification.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    trace = sub.add_parser("trace", help="Run figures against the full tree and record reads")
    trace.add_argument("--data-root", type=Path, required=True)
    trace.add_argument("--work-dir", type=Path, required=True)
    stage_parser = sub.add_parser("stage", help="Copy only traced files and write manifest/checksums")
    stage_parser.add_argument("--trace", type=Path, required=True)
    stage_parser.add_argument("--destination", type=Path, required=True)
    verify_parser = sub.add_parser("verify", help="Run figures from the archive with source-tree access blocked")
    verify_parser.add_argument("--archive", type=Path, required=True)
    verify_parser.add_argument("--work-dir", type=Path, required=True)
    verify_parser.add_argument("--forbid-root", type=Path, action="append", required=True)
    verify_parser.add_argument("--reference-output", type=Path)
    pack = sub.add_parser("pack", help="Verify checksums and create a new tar.gz, retaining empty directories")
    pack.add_argument("--archive", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "trace":
        report = run_audited(args.data_root, args.work_dir, [])
        result = dict(success=True, files=len(report["dependencies"]), size_bytes=sum(r["size_bytes"] for r in report["dependencies"]), trace=str(args.work_dir / "trace.json"))
    elif args.action == "stage":
        result = stage(args.trace, args.destination)
    elif args.action == "verify":
        result = verify(args.archive.resolve(), args.work_dir, args.forbid_root, args.reference_output)
    else:
        archive = args.archive.resolve(strict=True)
        if args.output.resolve().is_relative_to(archive):
            raise ValueError("Write the tar.gz outside the staged archive")
        check_archive(archive)
        with args.output.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as tar:
            tar.add(archive, arcname=archive.name)
        result = dict(output=str(args.output), size_bytes=args.output.stat().st_size, sha256=digest(args.output))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
