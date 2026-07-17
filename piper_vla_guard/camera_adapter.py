from __future__ import annotations

import time
import hashlib
from typing import Any, Optional

import numpy as np


class CameraCaptureError(RuntimeError):
    pass


class InvalidCameraFrame(CameraCaptureError):
    pass


def validate_rgb_frame(
    frame: np.ndarray,
    *,
    min_std: float = 2.0,
    max_dominant_fraction: float = 0.98,
) -> None:
    """Reject blank / single-colour frames before they can reach the policy."""
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise InvalidCameraFrame(f"invalid RGB frame shape: {image.shape}")
    if not np.isfinite(image).all():
        raise InvalidCameraFrame("camera frame contains non-finite values")
    if float(np.std(image.astype(np.float32))) < float(min_std):
        raise InvalidCameraFrame("camera frame is nearly constant")
    # Quantise slightly so compressed single-colour frames are also detected.
    sample = (image[::4, ::4].astype(np.uint16) // 8).reshape(-1, 3)
    _, counts = np.unique(sample, axis=0, return_counts=True)
    dominant = float(counts.max()) / float(len(sample))
    if dominant > float(max_dominant_fraction):
        raise InvalidCameraFrame(
            f"camera frame dominant-colour fraction {dominant:.3f} > {max_dominant_fraction:.3f}"
        )


def frame_digest(frame: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


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
        warmup_frames: int = 5,
        min_std: float = 2.0,
        max_dominant_fraction: float = 0.98,
        max_identical_frames: int = 2,
        allow_stale_fallback: bool = False,
        width: int = 640,
        height: int = 480,
        fps: float = 15.0,
        fourcc: str = "MJPG",
        buffer_size: int = 1,
        read_timeout_ms: int = 3000,
    ) -> None:
        self.source = source
        self.retries = max(1, int(retries))
        self.retry_sleep_s = max(0.0, float(retry_sleep_s))
        self.label = label
        self.warmup_frames = max(0, int(warmup_frames))
        self.min_std = float(min_std)
        self.max_dominant_fraction = float(max_dominant_fraction)
        self.max_identical_frames = max(0, int(max_identical_frames))
        self.allow_stale_fallback = bool(allow_stale_fallback)
        self.width = max(0, int(width))
        self.height = max(0, int(height))
        self.fps = max(0.0, float(fps))
        self.fourcc = str(fourcc).strip().upper()
        self.buffer_size = max(0, int(buffer_size))
        self.read_timeout_ms = max(0, int(read_timeout_ms))
        self._cap: Any = None
        self._cv2: Any = None
        self._last_rgb: Optional[np.ndarray] = None
        self._last_digest: Optional[str] = None
        self._identical_count = 0
        self._warmed_up = False

    def read_rgb(self) -> Optional[np.ndarray]:
        if self.source is None or str(self.source).strip() == "":
            return None
        self._ensure_open()
        if not self._warmed_up:
            self._warm_up()
        last_error = ""
        for attempt in range(self.retries):
            ok, frame = self._cap.read()
            if ok and frame is not None:
                rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
                try:
                    validate_rgb_frame(
                        rgb,
                        min_std=self.min_std,
                        max_dominant_fraction=self.max_dominant_fraction,
                    )
                    digest = frame_digest(rgb)
                    if digest == self._last_digest:
                        self._identical_count += 1
                    else:
                        self._identical_count = 0
                    if self._identical_count > self.max_identical_frames:
                        raise InvalidCameraFrame(
                            f"{self.label} frame repeated {self._identical_count + 1} times"
                        )
                except InvalidCameraFrame as exc:
                    last_error = str(exc)
                    # Repeated/invalid frames often indicate a wedged UVC
                    # stream. Reopen the descriptor instead of repeatedly
                    # reading the same dead capture object.
                    self._reopen()
                    time.sleep(self.retry_sleep_s)
                    continue
                self._last_rgb = rgb
                self._last_digest = digest
                return rgb
            last_error = f"Could not read frame from {self.label} source: {self.source!r}"
            self._reopen()
            time.sleep(self.retry_sleep_s)
        if self.allow_stale_fallback and self._last_rgb is not None:
            print(f"WARNING: {last_error}; using last good {self.label} frame.")
            return self._last_rgb
        raise CameraCaptureError(last_error)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._warmed_up = False

    def _ensure_open(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            return
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise CameraCaptureError("opencv-python is not installed. Install with: pip install -e '.[vision]'") from exc
        self._cv2 = cv2
        normalized = _normalize_source(self.source)
        if isinstance(normalized, int) or str(normalized).startswith("/dev/video"):
            self._cap = cv2.VideoCapture(normalized, cv2.CAP_V4L2)
        else:
            self._cap = cv2.VideoCapture(normalized)
        if not self._cap.isOpened():
            raise CameraCaptureError(f"Could not open {self.label} camera source: {self.source!r}")
        self._configure_capture()

    def _configure_capture(self) -> None:
        if self._cap is None or self._cv2 is None:
            return
        if self.fourcc and len(self.fourcc) == 4:
            self._cap.set(
                self._cv2.CAP_PROP_FOURCC,
                self._cv2.VideoWriter_fourcc(*self.fourcc),
            )
        for prop_name, value in (
            ("CAP_PROP_FRAME_WIDTH", self.width),
            ("CAP_PROP_FRAME_HEIGHT", self.height),
            ("CAP_PROP_FPS", self.fps),
            ("CAP_PROP_BUFFERSIZE", self.buffer_size),
            ("CAP_PROP_READ_TIMEOUT_MSEC", self.read_timeout_ms),
        ):
            if value <= 0 or not hasattr(self._cv2, prop_name):
                continue
            self._cap.set(getattr(self._cv2, prop_name), value)

    def _warm_up(self) -> None:
        if self.warmup_frames <= 0:
            self._warmed_up = True
            return
        good = 0
        attempts = 0
        limit = self.warmup_frames + self.retries
        while good < self.warmup_frames and attempts < limit:
            attempts += 1
            ok, frame = self._cap.read()
            if ok and frame is not None:
                good += 1
            else:
                self._reopen()
            if self.retry_sleep_s:
                time.sleep(self.retry_sleep_s)
        if good < self.warmup_frames:
            raise CameraCaptureError(
                f"Could not warm up {self.label} source {self.source!r}: "
                f"{good}/{self.warmup_frames} frames"
            )
        self._warmed_up = True

    def _reopen(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._warmed_up = False
        if self.retry_sleep_s:
            time.sleep(self.retry_sleep_s)
        self._ensure_open()


def _normalize_source(source: Any) -> Any:
    if isinstance(source, int):
        return source
    text = str(source).strip()
    try:
        return int(text)
    except ValueError:
        return text
