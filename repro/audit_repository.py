"""Static inventory of tracked imports and possible output/path sites (stdlib only).

Run: python repro/audit_repository.py --output /external/path/audit
Includes comments/examples, labelled separately; matches are review candidates,
not proof that a branch executes or that a destination is inside the repository.
"""
import argparse
import ast
import csv
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "machine path": r"/scratch/|/Users/|/home/|/tmp/|/project/",
    "path setting": r"(?:[A-Z_]*(?:DIR|ROOT|PATH)|\w*(?:dir|path|root))\s*[:=]|#SBATCH --(?:output|error)",
    "write/copy site": r"save\w*\(|write\w*\(|dump\w*\(|mkdir|makedirs|to_csv\(|to_netcdf\(|to_zarr\(|open\(.*[\"'][wax]|\b(?:cp|mv|rsync|scp|tee)\b|tempfile|shutil\.(?:copy|move)",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    with (args.output / "paths.tsv").open("w") as f, (args.output / "imports.tsv").open("w") as g:
        paths, imports = csv.writer(f, delimiter="\t"), csv.writer(g, delimiter="\t")
        paths.writerow(["file", "line", "kind", "comment", "source"])
        imports.writerow(["module", "file", "line", "stdlib_or_local"])
        for name in sorted(files):
            p = ROOT / name
            if p.suffix not in {".py", ".sh", ".yaml", ".yml"}:
                continue
            source = p.read_text()
            for lineno, line in enumerate(source.splitlines(), 1):
                for kind, pattern in PATTERNS.items():
                    if re.search(pattern, line):
                        paths.writerow([name, lineno, kind, line.lstrip().startswith("#"), line.strip()])
            if p.suffix != ".py":
                continue
            for node in ast.walk(ast.parse(source, filename=name)):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                    modules = [node.module]
                for module in modules:
                    top = module.split(".")[0]
                    local = (ROOT / top).exists() or top in {"utils", "plot_utils"}
                    imports.writerow([module, name, node.lineno, local or top in sys.stdlib_module_names])
    print(f"Inventory written to {args.output}")


if __name__ == "__main__":
    main()
