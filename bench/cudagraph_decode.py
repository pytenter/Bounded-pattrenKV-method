from __future__ import annotations

import time
import copy
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable

import torch


DecodeFn = Callable[[torch.Tensor, Any], tuple[Any, torch.Tensor, torch.Tensor]]


def tree_clone(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(tree_clone(item) for item in value)
    if isinstance(value, list):
        return [tree_clone(item) for item in value]
    if isinstance(value, dict):
        return {key: tree_clone(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{field.name: tree_clone(getattr(value, field.name)) for field in fields(value)})
    if hasattr(value, "__dict__") and not isinstance(value, type):
        out = copy.copy(value)
        for key, item in vars(value).items():
            setattr(out, key, tree_clone(item))
        return out
    return value


def tree_copy_(dst: Any, src: Any) -> None:
    if torch.is_tensor(dst) and torch.is_tensor(src):
        if dst.shape != src.shape or dst.dtype != src.dtype:
            raise ValueError(f"static tensor mismatch: dst={tuple(dst.shape)} {dst.dtype} src={tuple(src.shape)} {src.dtype}")
        dst.copy_(src)
        return
    if isinstance(dst, tuple) and isinstance(src, tuple):
        if len(dst) != len(src):
            raise ValueError(f"tuple length mismatch: dst={len(dst)} src={len(src)}")
        for left, right in zip(dst, src):
            tree_copy_(left, right)
        return
    if isinstance(dst, list) and isinstance(src, list):
        if len(dst) != len(src):
            raise ValueError(f"list length mismatch: dst={len(dst)} src={len(src)}")
        for left, right in zip(dst, src):
            tree_copy_(left, right)
        return
    if isinstance(dst, dict) and isinstance(src, dict):
        if set(dst) != set(src):
            raise ValueError("dict key mismatch while restoring static graph buffers")
        for key in dst:
            tree_copy_(dst[key], src[key])
        return
    if is_dataclass(dst) and is_dataclass(src):
        if type(dst) is not type(src):
            raise ValueError(f"dataclass type mismatch: dst={type(dst).__name__} src={type(src).__name__}")
        for field in fields(dst):
            tree_copy_(getattr(dst, field.name), getattr(src, field.name))
        return
    if hasattr(dst, "__dict__") and hasattr(src, "__dict__") and type(dst) is type(src):
        for key in vars(dst):
            if hasattr(src, key):
                tree_copy_(getattr(dst, key), getattr(src, key))
        return


def tree_tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, (tuple, list)):
        return sum(tree_tensor_bytes(item) for item in value)
    if isinstance(value, dict):
        return sum(tree_tensor_bytes(item) for item in value.values())
    if is_dataclass(value) and not isinstance(value, type):
        return sum(tree_tensor_bytes(getattr(value, field.name)) for field in fields(value))
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return sum(tree_tensor_bytes(item) for item in vars(value).values())
    return 0


@dataclass
class CausalDecodeGraphSequence:
    graphs: list[torch.cuda.CUDAGraph]
    static_tokens: list[torch.Tensor]
    output_tokens: list[torch.Tensor]
    output_logits: list[torch.Tensor]
    output_caches: list[Any]
    initial_cache: Any
    initial_cache_snapshot: Any
    capture_time_ms: float
    capture_memory_allocated_bytes: int
    capture_memory_reserved_bytes: int

    @property
    def steps(self) -> int:
        return len(self.graphs)

    def reset_initial_state_(self, token: torch.Tensor) -> None:
        self.static_tokens[0].copy_(token.view_as(self.static_tokens[0]))
        tree_copy_(self.initial_cache, self.initial_cache_snapshot)

    def replay(self, token: torch.Tensor) -> tuple[Any, list[torch.Tensor], list[torch.Tensor]]:
        self.reset_initial_state_(token)
        for idx, graph in enumerate(self.graphs):
            graph.replay()
            if idx + 1 < len(self.static_tokens):
                self.static_tokens[idx + 1].copy_(self.output_tokens[idx].view_as(self.static_tokens[idx + 1]))
        return self.output_caches[-1], self.output_tokens, self.output_logits


def capture_causal_decode_graph_sequence(
    decode_fn: DecodeFn,
    initial_token: torch.Tensor,
    initial_cache: Any,
    *,
    steps: int,
    device: torch.device,
) -> CausalDecodeGraphSequence:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    torch.cuda.synchronize(device)
    initial_snapshot = tree_clone(initial_cache)
    static_tokens = [initial_token.detach().clone() for _ in range(steps)]
    graphs: list[torch.cuda.CUDAGraph] = []
    output_tokens: list[torch.Tensor] = []
    output_logits: list[torch.Tensor] = []
    output_caches: list[Any] = []
    current_cache = initial_cache
    current_token = static_tokens[0]
    before_alloc = int(torch.cuda.memory_allocated(device))
    before_reserved = int(torch.cuda.memory_reserved(device))
    started = time.perf_counter()
    for step in range(steps):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            next_cache, next_token, logits = decode_fn(current_token, current_cache)
        graphs.append(graph)
        output_caches.append(next_cache)
        output_tokens.append(next_token)
        output_logits.append(logits)
        if step + 1 < steps:
            static_tokens[step + 1].copy_(next_token.view_as(static_tokens[step + 1]))
            current_token = static_tokens[step + 1]
            current_cache = next_cache
    torch.cuda.synchronize(device)
    capture_ms = (time.perf_counter() - started) * 1000.0
    capture_alloc = max(0, int(torch.cuda.memory_allocated(device)) - before_alloc)
    capture_reserved = max(0, int(torch.cuda.memory_reserved(device)) - before_reserved)
    sequence = CausalDecodeGraphSequence(
        graphs=graphs,
        static_tokens=static_tokens,
        output_tokens=output_tokens,
        output_logits=output_logits,
        output_caches=output_caches,
        initial_cache=initial_cache,
        initial_cache_snapshot=initial_snapshot,
        capture_time_ms=capture_ms,
        capture_memory_allocated_bytes=capture_alloc,
        capture_memory_reserved_bytes=capture_reserved,
    )
    sequence.reset_initial_state_(initial_token)
    torch.cuda.synchronize(device)
    return sequence
