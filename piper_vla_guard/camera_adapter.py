from __future__ import annotations

import time
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


class PersistentCamera:
    """OpenCV camera wrapper for repeated VLA loop captures.

    Some V4L devices occasionally timeout when opened and closed every cycle.
    Keeping the descriptor open and falling back to the last good frame makes
    the real-time loop much less brittle without hiding the warning.
    """

    def __init__(
        self,
        source: str | int | None,
        *,
        retries: int = 3,
        retry_sleep_s: float = 0.05,
        label: str = "camera",
    ) -> None:
        self.source = source
        self.retries = max(1, int(retries))
        self.retry_sleep_s = max(0.0, float(retry_sleep_s))
        self.label = label
        self._cap: Any = None
        self._cv2: Any = None
        self._last_rgb: Optional[np.ndarray] = None

    def read_rgb(self) -> Optional[np.ndarray]:
        if self.source is None or str(self.source).strip() == "":
            return None
        self._ensure_open()
        last_error = ""
        for attempt in range(self.retries):
            ok, frame = self._cap.read()
            if ok and frame is not None:
                rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
                self._last_rgb = rgb
                return rgb
            last_error = f"Could not read frame from {self.label} source: {self.source!r}"
            time.sleep(self.retry_sleep_s)
        if self._last_rgb is not None:
            print(f"WARNING: {last_error}; using last good {self.label} frame.")
            return self._last_rgb
        raise CameraCaptureError(last_error)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _ensure_open(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            return
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise CameraCaptureError("opencv-python is not installed. Install with: pip install -e '.[vision]'") from exc
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(_normalize_source(self.source))
        if not self._cap.isOpened():
            raise CameraCaptureError(f"Could not open {self.label} camera source: {self.source!r}")


def _normalize_source(source: Any) -> Any:
    if isinstance(source, int):
        return source
    text = str(source).strip()
    try:
        return int(text)
    except ValueError:
        return text
