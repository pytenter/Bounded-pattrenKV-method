"""Configuration loading and baseline validation for PatternKV Insight."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


CANONICAL_METHODS = ("fp16", "kivi_paper_g128", "patternkv_paper")
LEGACY_METHODS = ("kivi", "kivi_official", "kivi_original_g32", "patternkv")


@dataclass(frozen=True)
class InsightRuntimeConfig:
    """Runtime observer settings parsed from environment variables."""

    enabled: bool = False
    level: str = "basic"
    sample_tokens: int = 8
    output: Path = Path("results/insight_v2/observer")
    seed: int = 0
    max_sample_records: int = 4096
    oracle_layers: tuple[int, ...] = (0, 7, 15, 23, 31)

    @classmethod
    def from_env(cls) -> "InsightRuntimeConfig":
        """Build config from PATTERNKV_INSIGHT* environment variables."""
        enabled = os.environ.get("PATTERNKV_INSIGHT", "0") == "1"
        level = os.environ.get("PATTERNKV_INSIGHT_LEVEL", "basic")
        if level not in {"basic", "oracle", "attention"}:
            raise ValueError(f"invalid PATTERNKV_INSIGHT_LEVEL={level!r}")
        return cls(
            enabled=enabled,
            level=level,
            sample_tokens=int(os.environ.get("PATTERNKV_INSIGHT_SAMPLE_TOKENS", "8")),
            output=Path(os.environ.get("PATTERNKV_INSIGHT_OUTPUT", "results/insight_v2/observer")),
            seed=int(os.environ.get("PATTERNKV_INSIGHT_SEED", "0")),
            max_sample_records=int(os.environ.get("PATTERNKV_INSIGHT_MAX_SAMPLE_RECORDS", "4096")),
            oracle_layers=tuple(
                int(x)
                for x in os.environ.get("PATTERNKV_INSIGHT_ORACLE_LAYERS", "0,7,15,23,31").split(",")
                if x.strip()
            ),
        )


@dataclass(frozen=True)
class StandardBaselines:
    """Canonical paper-v2 method configuration loaded from YAML."""

    path: Path
    raw: dict[str, Any]
    config_hash: str

    @property
    def canonical_methods(self) -> tuple[str, ...]:
        return tuple(self.raw.get("canonical_methods", ()))

    @property
    def methods(self) -> Mapping[str, Mapping[str, Any]]:
        return self.raw.get("methods", {})

    def assert_canonical_semantics(self) -> None:
        """Assert the YAML still encodes the expected paper-v2 baseline set."""
        if self.canonical_methods != CANONICAL_METHODS:
            raise ValueError(f"canonical_methods changed: {self.canonical_methods}")
        for method in CANONICAL_METHODS:
            if method not in self.methods:
                raise ValueError(f"missing canonical method {method}")
        expected = {
            "fp16": {"backend_method": "fp16", "k_bits": 16, "v_bits": 16, "group_size": 0, "residual_length": 0},
            "kivi_paper_g128": {"backend_method": "kivi_official", "k_bits": 2, "v_bits": 2, "group_size": 128, "residual_length": 128},
            "patternkv_paper": {
                "backend_method": "patternkv",
                "k_bits": 2,
                "v_bits": 2,
                "group_size": 128,
                "residual_length": 128,
                "initial_pattern_count": 32,
                "pattern_group": 128,
            },
        }
        for method, values in expected.items():
            cfg = self.methods[method]
            for key, value in values.items():
                if cfg.get(key) != value:
                    raise ValueError(f"{method}.{key} changed: got {cfg.get(key)!r}, expected {value!r}")

    def validate_record(self, record: Mapping[str, Any]) -> list[str]:
        """Return config validation errors for one result record."""
        method = record.get("method")
        if method not in CANONICAL_METHODS:
            return [f"non_canonical_method={method}"]
        qc = (
            record.get("quantization_config")
            or record.get("paper_config_snapshot")
            or record.get("patternkv_config")
            or {}
        )
        expected = self.methods[str(method)]
        errors: list[str] = []
        checks = ("backend_method", "k_bits", "v_bits", "group_size", "residual_length")
        for key in checks:
            if qc.get(key) != expected.get(key):
                errors.append(f"{key}: got {qc.get(key)!r}, expected {expected.get(key)!r}")
        if method == "patternkv_paper":
            for key in ("initial_pattern_count", "pattern_group"):
                if qc.get(key) != expected.get(key):
                    errors.append(f"{key}: got {qc.get(key)!r}, expected {expected.get(key)!r}")
            pos = str(qc.get("pattern_selection_position", ""))
            if "post-RoPE" not in pos and "post_rope" not in pos:
                errors.append(f"pattern_selection_position: got {pos!r}, expected post-RoPE")
        if method in {"kivi_paper_g128", "patternkv_paper"}:
            stats = record.get("cache_bitwidth_stats") or {}
            if stats.get("persistent_key_heads") not in (None, 8):
                errors.append(f"persistent_key_heads: got {stats.get('persistent_key_heads')!r}, expected 8")
            key_axis = str(qc.get("key_quant_axis", expected.get("key_quant_axis", "")))
            value_axis = str(qc.get("value_quant_axis", expected.get("value_quant_axis", "")))
            if "per-channel" not in key_axis and "per_channel" not in key_axis:
                errors.append(f"key_quant_axis: got {key_axis!r}, expected per-channel")
            if "per-token" not in value_axis and "per_token" not in value_axis:
                errors.append(f"value_quant_axis: got {value_axis!r}, expected per-token")
        return errors


def load_standard_baselines(path: Path = Path("configs/standard_baselines.paper_v2.yaml")) -> StandardBaselines:
    """Load and validate the canonical paper-v2 baseline YAML."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    cfg = StandardBaselines(path=path, raw=raw, config_hash=hashlib.sha256(payload.encode()).hexdigest())
    cfg.assert_canonical_semantics()
    return cfg
