"""Convert paired Inter-X axis-angle NPY files to rot6d HDF5."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

import utils.rotation_conversions as geometry


def build_parser() -> argparse.ArgumentParser:
    output_root = Path(os.environ.get("INTERX_OUTPUT_ROOT", "outputs"))
    parser = argparse.ArgumentParser(
        description="Convert [T, 56, 6] Inter-X axis-angle NPY files to rot6d HDF5."
    )
    parser.add_argument("--input-dir", type=Path,
                        default=output_root / "interx_smplx_npy")
    parser.add_argument("--output", type=Path,
                        default=output_root / "h5_files/rot6d/interx_smplx_rot6d.h5")
    parser.add_argument("--limit", type=int,
                        help="Convert only the first N files (for smoke tests)")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def axis_angle_to_rot6d(pose: torch.Tensor) -> torch.Tensor:
    """Convert [T, 55, 6] two-person axis angles to [T, 55, 12]."""
    person1 = geometry.matrix_to_rotation_6d(
        geometry.axis_angle_to_matrix(pose[..., :3])
    )
    person2 = geometry.matrix_to_rotation_6d(
        geometry.axis_angle_to_matrix(pose[..., 3:])
    )
    return torch.cat((person1, person2), dim=-1)


def convert_motion(data: np.ndarray) -> np.ndarray:
    if data.ndim != 3 or data.shape[1:] != (56, 6):
        raise ValueError(f"Expected [T, 56, 6], got {data.shape}")
    if len(data) == 0 or not np.isfinite(data).all():
        raise ValueError("Input motion is empty or contains non-finite values")

    motion = torch.from_numpy(data.astype(np.float32, copy=False))
    rot6d = axis_angle_to_rot6d(motion[:, :-1])
    translation = motion[:, -1]
    velocity = torch.zeros_like(translation)
    if len(translation) > 1:
        velocity[:-1] = translation[1:] - translation[:-1]

    translation_velocity = torch.cat(
        (
            translation[:, :3],
            velocity[:, :3],
            translation[:, 3:],
            velocity[:, 3:],
        ),
        dim=-1,
    )
    result = torch.cat((rot6d, translation_velocity[:, None]), dim=1)
    return result.numpy().astype(np.float32, copy=False)


def main() -> None:
    args = build_parser().parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    input_dir = args.input_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    files = sorted(input_dir.glob("*.npy"))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise ValueError(f"No NPY files found in {input_dir}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with h5py.File(temporary, "w") as output:
            output.attrs["representation"] = "rot6d"
            output.attrs["shape"] = "[T,56,12]"
            for path in tqdm(files, desc="Converting to rot6d"):
                data = np.load(path, allow_pickle=False)
                output.create_dataset(path.stem, data=convert_motion(data), dtype="f4")
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Saved {len(files)} sequences to {output_path}.")


if __name__ == "__main__":
    main()

