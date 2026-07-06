from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence

import numpy as np


class OpenPIClientError(RuntimeError):
    pass


class OpenPIPolicyClient:
    """Optional OpenPI websocket client wrapper.

    Requires installing OpenPI's packages/openpi-client in the robot runtime.
    """

    def __init__(self, host: str = "localhost", port: int = 8000):
        try:
            from openpi_client import image_tools  # type: ignore
            from openpi_client import websocket_client_policy  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise OpenPIClientError(
                "openpi-client is not installed. In the OpenPI repo, run: "
                "cd packages/openpi-client && pip install -e ."
            ) from exc
        self.image_tools = image_tools
        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=int(port))

    def infer(
        self,
        prompt: str,
        image: Optional[Any] = None,
        wrist_image: Optional[Any] = None,
        state: Optional[Sequence[float]] = None,
        image_size: int = 224,
    ) -> Dict[str, Any]:
        observation: Dict[str, Any] = {"prompt": prompt}
        if image is not None:
            observation["observation/image"] = self._prep_image(image, image_size)
        if wrist_image is not None:
            observation["observation/wrist_image"] = self._prep_image(wrist_image, image_size)
        if state is not None:
            observation["observation/state"] = np.asarray(state, dtype=np.float32)
        return self.client.infer(observation)

    def _prep_image(self, img: Any, image_size: int) -> np.ndarray:
        arr = np.asarray(img)
        return self.image_tools.convert_to_uint8(
            self.image_tools.resize_with_pad(arr, image_size, image_size)
        )


def actions_to_json(actions: Any) -> str:
    if hasattr(actions, "tolist"):
        actions = actions.tolist()
    return json.dumps({"actions": actions}, indent=2)


def response_to_json(response: Any) -> str:
    return json.dumps(_jsonable(response), indent=2)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def parse_state_json(text: str | None) -> Optional[list[float]]:
    if text is None or not text.strip():
        return None
    data = json.loads(text)
    if isinstance(data, dict):
        for key in ("state", "observation/state", "proprio"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("State JSON must be a list or an object containing state")
    return [float(x) for x in data]
