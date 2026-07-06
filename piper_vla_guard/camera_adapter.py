from __future__ import annotations

from typing import Any, Optional

import numpy as np


class CameraCaptureError(RuntimeError):
    pass


def capture_rgb_frame(source: str | int | None) -> Optional[np.ndarray]:
    """Capture one RGB frame from an OpenCV-compatible camera source."""
    if source is None or str(source).strip() == "":
        return None
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise CameraCaptureError("opencv-python is not installed. Install with: pip install -e '.[vision]'") from exc

    cap = cv2.VideoCapture(_normalize_source(source))
    try:
        if not cap.isOpened():
            raise CameraCaptureError(f"Could not open camera source: {source!r}")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise CameraCaptureError(f"Could not read frame from camera source: {source!r}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def _normalize_source(source: Any) -> Any:
    if isinstance(source, int):
        return source
    text = str(source).strip()
    try:
        return int(text)
    except ValueError:
        return text
