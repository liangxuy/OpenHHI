"""Split a rot6d HDF5 file using the official Inter-X split lists."""

from __future__ import annotations

import argparse
import os
from contextlib import ExitStack
from pathlib import Path

import h5py
from tqdm import tqdm


SPLIT_NAMES = ("train", "val", "test")


def build_parser() -> argparse.ArgumentParser:
    output_root = Path(os.environ.get("INTERX_OUTPUT_ROOT", "outputs"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=output_root / "h5_files/rot6d/interx_smplx_rot6d.h5")
    parser.add_argument("--splits-dir", type=Path, required=True,
                        help="Directory containing train.txt, val.txt, and test.txt")
    parser.add_argument("--output-dir", type=Path,
                        default=output_root / "h5_files/rot6d/motions")
    parser.add_argument("--strict", action="store_true",
                        help="Fail if a split entry is missing from the input HDF5")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def read_split(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Split file does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input.expanduser().resolve()
    splits_dir = args.splits_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input HDF5 does not exist: {input_path}")

    split_keys = {name: read_split(splits_dir / f"{name}.txt") for name in SPLIT_NAMES}
    ownership: dict[str, str] = {}
    for split_name, keys in split_keys.items():
        for key in keys:
            if key in ownership:
                raise ValueError(
                    f"Sample {key} occurs in both {ownership[key]} and {split_name} splits"
                )
            ownership[key] = split_name

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {name: output_dir / f"{name}.h5" for name in SPLIT_NAMES}
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Output exists: {existing[0]}. Pass --overwrite to replace it.")
    temporary = {
        name: path.with_name(f".{path.name}.tmp") for name, path in outputs.items()
    }

    report: dict[str, tuple[int, int]] = {}
    try:
        with h5py.File(input_path, "r") as source, ExitStack() as stack:
            targets = {
                name: stack.enter_context(h5py.File(path, "w"))
                for name, path in temporary.items()
            }
            for target in targets.values():
                for attr, value in source.attrs.items():
                    target.attrs[attr] = value

            for split_name, keys in split_keys.items():
                saved = missing = 0
                for key in tqdm(keys, desc=f"Writing {split_name}"):
                    if key not in source:
                        missing += 1
                        continue
                    targets[split_name].create_dataset(
                        key, data=source[key][()], dtype="f4"
                    )
                    saved += 1
                report[split_name] = saved, missing

        missing_total = sum(missing for _, missing in report.values())
        if args.strict and missing_total:
            raise ValueError(f"{missing_total} split entries are missing from the input HDF5")
        for name in SPLIT_NAMES:
            temporary[name].replace(outputs[name])
    except Exception:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise

    for name in SPLIT_NAMES:
        saved, missing = report[name]
        print(f"{name}: {saved} saved, {missing} missing")


if __name__ == "__main__":
    main()

