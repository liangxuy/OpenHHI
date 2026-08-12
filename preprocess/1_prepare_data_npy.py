"""Prepare paired Inter-X SMPL-X parameters as axis-angle NPY files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from human_body_prior.body_model.body_model import BodyModel
from tqdm import tqdm


POSE_ROWS = 55
OUTPUT_ROWS = 56
REQUIRED_FIELDS = ("pose_body", "pose_lhand", "pose_rhand", "root_orient", "trans")


def build_parser() -> argparse.ArgumentParser:
    output_root = Path(os.environ.get("INTERX_OUTPUT_ROOT", "outputs"))
    parser = argparse.ArgumentParser(
        description="Convert raw paired Inter-X SMPL-X NPZ files to [T, 56, 6] NPY files."
    )
    parser.add_argument("--motions-dir", type=Path, required=True)
    parser.add_argument("--body-model", type=Path, required=True,
                        help="Path to SMPLX_NEUTRAL.npz")
    parser.add_argument("--output-dir", type=Path,
                        default=output_root / "interx_smplx_npy")
    parser.add_argument("--downsample", type=int, default=4,
                        help="Keep one frame every N frames (default: 4)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="SMPL-X inference batch size (default: 256)")
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, cuda, or cuda:N (default: auto)")
    parser.add_argument("--limit", type=int,
                        help="Process only the first N sequences (for smoke tests)")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {name}")
    return device


def load_person(path: Path, downsample: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        missing = [name for name in REQUIRED_FIELDS if name not in source]
        if missing:
            raise ValueError(f"{path} is missing fields: {', '.join(missing)}")
        body = source["pose_body"][::downsample].astype(np.float32, copy=True)
        left_hand = source["pose_lhand"][::downsample].astype(np.float32, copy=True)
        right_hand = source["pose_rhand"][::downsample].astype(np.float32, copy=True)
        root = source["root_orient"][::downsample].astype(np.float32, copy=True)
        translation = source["trans"][::downsample].astype(np.float32, copy=True)

    frames = len(body)
    expected = {
        "pose_body": (frames, 21, 3),
        "pose_lhand": (frames, 15, 3),
        "pose_rhand": (frames, 15, 3),
        "root_orient": (frames, 3),
        "trans": (frames, 3),
    }
    actual = {
        "pose_body": body.shape,
        "pose_lhand": left_hand.shape,
        "pose_rhand": right_hand.shape,
        "root_orient": root.shape,
        "trans": translation.shape,
    }
    invalid = [f"{name}={actual[name]}" for name in expected if actual[name] != expected[name]]
    if invalid:
        raise ValueError(f"Unexpected arrays in {path}: {', '.join(invalid)}")

    face_placeholders = np.zeros((frames, 3, 3), dtype=np.float32)
    pose = np.concatenate(
        (root[:, None], body, face_placeholders, left_hand, right_hand), axis=1
    )
    if pose.shape != (frames, POSE_ROWS, 3):
        raise AssertionError(f"Unexpected pose shape: {pose.shape}")
    return pose, translation


def align_to_floor(
    pose: np.ndarray,
    translation: np.ndarray,
    body_model: BodyModel,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Place the lowest SMPL-X joint on the ground and append root translation."""
    minimum_y = np.inf
    with torch.no_grad():
        for start in range(0, len(pose), batch_size):
            stop = min(start + batch_size, len(pose))
            batch = torch.from_numpy(pose[start:stop]).to(device)
            trans = torch.from_numpy(translation[start:stop]).to(device)
            body = body_model(
                root_orient=batch[:, 0],
                pose_body=batch[:, 1:22].reshape(stop - start, -1),
                pose_hand=batch[:, 25:55].reshape(stop - start, -1),
            )
            joints = body.Jtr + trans[:, None]
            minimum_y = min(minimum_y, float(joints[..., 1].min().item()))

    translation[:, 1] -= minimum_y
    return np.concatenate((pose, translation[:, None]), axis=1).astype(np.float32)


def prepare_sequence(
    person_files: list[Path],
    body_model: BodyModel,
    device: torch.device,
    downsample: int,
    batch_size: int,
) -> np.ndarray:
    if len(person_files) != 2:
        raise ValueError(f"Expected exactly two person NPZ files, found {len(person_files)}")

    people = []
    for path in person_files:
        pose, translation = load_person(path, downsample)
        people.append(align_to_floor(pose, translation, body_model, device, batch_size))

    if len(people[0]) != len(people[1]):
        raise ValueError(
            f"Person frame counts differ: {len(people[0])} and {len(people[1])}"
        )

    # Express both translations relative to person 1 at the first frame.
    # Only row 55 is translation; pose rows must remain unchanged.
    origin = people[0][0, -1].copy()
    people[0][:, -1] -= origin
    people[1][:, -1] -= origin
    result = np.concatenate(people, axis=-1).astype(np.float32)
    if result.shape[1:] != (OUTPUT_ROWS, 6) or not np.isfinite(result).all():
        raise ValueError(f"Generated invalid motion with shape {result.shape}")
    return result


def save_npy(path: Path, data: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, data, allow_pickle=False)
    temporary.replace(path)


def main() -> None:
    args = build_parser().parse_args()
    if args.downsample < 1 or args.batch_size < 1:
        raise ValueError("--downsample and --batch-size must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    motions_dir = args.motions_dir.expanduser().resolve()
    model_path = args.body_model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not motions_dir.is_dir():
        raise FileNotFoundError(f"Motions directory does not exist: {motions_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(f"SMPL-X model does not exist: {model_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = sorted(path for path in motions_dir.iterdir() if path.is_dir())
    if args.limit is not None:
        sequences = sequences[: args.limit]
    if not sequences:
        raise ValueError(f"No sequence directories found in {motions_dir}")

    device = resolve_device(args.device)
    body_model = BodyModel(bm_fname=str(model_path), num_betas=10).to(device)
    body_model.eval()
    saved = skipped = 0
    for sequence_dir in tqdm(sequences, desc="Preparing motions"):
        output_path = output_dir / f"{sequence_dir.name}.npy"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        person_files = sorted(sequence_dir.glob("*.npz"))
        motion = prepare_sequence(
            person_files, body_model, device, args.downsample, args.batch_size
        )
        save_npy(output_path, motion)
        saved += 1

    print(f"Saved {saved} sequences to {output_dir}; skipped {skipped} existing files.")


if __name__ == "__main__":
    main()

