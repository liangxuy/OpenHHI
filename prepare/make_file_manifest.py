"""Generate a deterministic TSV inventory for an assembled release."""

import argparse
import os
from pathlib import Path


RUNTIME_DIR_NAMES = {"__pycache__", ".pytest_cache"}
CHECKPOINT_RUNTIME_DIR_NAMES = {"animation", "animation_infer", "eval", "log"}
RUNTIME_SUFFIXES = {".pyc", ".pyo", ".orig", ".rej"}

def classify(relative_path):
    first = relative_path.parts[0] if relative_path.parts else ""
    if first == "data":
        return "data"
    if first == "checkpoints":
        return "model"
    if first in {"assets", "visualization"}:
        return "asset"
    return "code"


def should_include(relative_path):
    parts = relative_path.parts
    if any(part in RUNTIME_DIR_NAMES for part in parts):
        return False
    if relative_path.suffix in RUNTIME_SUFFIXES:
        return False
    if parts and parts[0] == "outputs":
        return False
    if parts and parts[0] == "checkpoints":
        if any(part in CHECKPOINT_RUNTIME_DIR_NAMES for part in parts[1:]):
            return False
    return True


def build_manifest(root, output):
    root = root.resolve()
    output = output.resolve()
    rows = []
    for path in root.rglob("*"):
        if path == output or not path.is_file():
            continue
        relative = path.relative_to(root)
        if not should_include(relative):
            continue
        rows.append((relative.as_posix(), path.stat().st_size, classify(relative)))
    rows.sort(key=lambda row: row[0])

    tmp_output = output.with_suffix(output.suffix + ".tmp")
    with tmp_output.open("w", encoding="utf-8", newline="") as stream:
        stream.write("path\tsize_bytes\tcategory\n")
        for path, size, category in rows:
            stream.write(f"{path}\t{size}\t{category}\n")
    os.replace(tmp_output, output)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.root / "FILE_MANIFEST.tsv"
    rows = build_manifest(args.root, output)
    total_bytes = sum(row[1] for row in rows)
    print(f"Wrote {output}: {len(rows)} files, {total_bytes / 1073741824:.3f} GiB")


if __name__ == "__main__":
    main()
