from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Optional, Sequence

import numpy as np

from .hybrid_pick import detect_white_cylinder
from .pick_calibration import load_pick_calibration, preprocess_camera_image
from .policy_adapter import OpenPIPolicyClient
from .real_loop import EXPECTED_CHECKPOINT_STEP, EXPECTED_DATASET_REPO_ID, EXPECTED_FPS, EXPECTED_PROMPT


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    calibration = load_pick_calibration(args.calibration)
    if calibration is None or calibration.overhead is None:
        raise RuntimeError("replay requires --calibration with overhead geometry")
    payload = json.loads(pathlib.Path(args.observation_json).read_text(encoding="utf-8"))
    state = np.asarray(payload["state"], dtype=np.float32)
    overhead = _read_rgb(args.overhead_image)
    wrist = _read_rgb(args.wrist_image) if args.wrist_image else None
    rectified = preprocess_camera_image(overhead, calibration.overhead)
    variants = {
        "raw": overhead,
        "rectified": rectified,
    }
    variants.update(_shifted_cylinder_variants(rectified, calibration))

    client = OpenPIPolicyClient(args.policy_host, args.policy_port)
    identity = client.validate_identity(
        dataset_repo_id=EXPECTED_DATASET_REPO_ID,
        checkpoint_step=EXPECTED_CHECKPOINT_STEP,
        prompt=EXPECTED_PROMPT,
        fps=EXPECTED_FPS,
    )
    results: dict[str, Any] = {}
    for name, image in variants.items():
        observation = {
            "prompt": EXPECTED_PROMPT,
            "observation/state": state,
            "observation/image": image,
        }
        if wrist is not None:
            observation["observation/wrist_image"] = preprocess_camera_image(wrist, calibration.wrist)
        response = client.client.infer(observation)
        actions = np.asarray(response["actions"], dtype=np.float32)
        results[name] = {
            "mean_first5": actions[:5].mean(axis=0).tolist(),
            "actions": actions.tolist(),
        }

    left = np.asarray(results.get("cylinder_left", {}).get("mean_first5", [0, 0]), dtype=float)[:2]
    right = np.asarray(results.get("cylinder_right", {}).get("mean_first5", [0, 0]), dtype=float)[:2]
    sensitivity = float(np.linalg.norm(left - right))
    report = {
        "dataset_repo_id": EXPECTED_DATASET_REPO_ID,
        "checkpoint_step": EXPECTED_CHECKPOINT_STEP,
        "policy_identity": identity,
        "image_sensitivity_l2": sensitivity,
        "image_sensitivity_pass": sensitivity >= args.sensitivity_threshold,
        "variants": results,
    }
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _shifted_cylinder_variants(image: np.ndarray, calibration: Any) -> dict[str, np.ndarray]:
    detection = detect_white_cylinder(
        image,
        calibration.overhead,
        expected_diameter_m=calibration.cylinder_diameter_m,
    )
    cx, cy = detection.pixel_xy
    radius = max(3.0, (detection.area_px / np.pi) ** 0.5 * 1.25)
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius
    background = np.median(image[~mask], axis=0).astype(np.uint8)
    source = image.copy()
    source[~mask] = 0
    variants = {}
    for name, shift in (("cylinder_left", -20), ("cylinder_right", 20)):
        moved = image.copy()
        moved[mask] = background
        shifted_pixels = np.roll(source, shift, axis=1)
        shifted_mask = np.roll(mask, shift, axis=1)
        moved[shifted_mask] = shifted_pixels[shifted_mask]
        variants[name] = moved
    return variants


def _read_rgb(path_text: str) -> np.ndarray:
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for observation replay") from exc
    return np.asarray(Image.open(path_text).convert("RGB"))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay saved real observations with camera counterfactuals.")
    parser.add_argument("--observation-json", required=True)
    parser.add_argument("--overhead-image", required=True)
    parser.add_argument("--wrist-image", default="")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--policy-host", default="localhost")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--sensitivity-threshold", type=float, default=0.02)
    parser.add_argument("--output", default="logs/pure_vla_image_sensitivity.json")
    return parser.parse_args(argv)


def main() -> None:
    report = run_replay(parse_args())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
