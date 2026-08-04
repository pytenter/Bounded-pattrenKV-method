"""Error types for PatternKV Insight observer hooks."""

from __future__ import annotations

from typing import Any

import torch


def tensor_shapes(values: dict[str, Any]) -> dict[str, Any]:
    """Return JSON-safe tensor shapes for hook error context."""
    out: dict[str, Any] = {}
    for key, value in values.items():
        if torch.is_tensor(value):
            out[key] = list(value.shape)
        elif isinstance(value, (list, tuple)) and all(torch.is_tensor(x) for x in value):
            out[key] = [list(x.shape) for x in value]
        elif value is None:
            out[key] = None
        else:
            out[key] = str(type(value).__name__)
    return out


class InsightHookError(RuntimeError):
    """Observer hook failure with phase/layer/head/tensor context."""

    def __init__(
        self,
        message: str,
        *,
        hook_name: str,
        phase: str,
        kv_type: str,
        layer_idx: int | None,
        kv_head: int | None = None,
        tensor_shapes: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.hook_name = hook_name
        self.phase = phase
        self.kv_type = kv_type
        self.layer_idx = layer_idx
        self.kv_head = kv_head
        self.tensor_shapes = tensor_shapes or {}
        self.exception_type = cause.__class__.__name__ if cause is not None else self.__class__.__name__
        self.exception_message = str(cause) if cause is not None else message
        super().__init__(
            f"{hook_name} failed phase={phase} kv_type={kv_type} layer={layer_idx} "
            f"head={kv_head}: {self.exception_type}: {self.exception_message}; shapes={self.tensor_shapes}"
        )
