from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from typing import Any


PORTABLE_REFERENCE_GENERATION_SCHEMA = "aime24_reference_generation_semantics_v1"


REFERENCE_USER_PROMPT_TEMPLATE = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{}."
REFERENCE_CHAT_TEMPLATE_SEMANTICS = "tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)"


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_hash32(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:32]


def recompute_legacy_generation_hash(generation_config: dict[str, Any]) -> str:
    return canonical_hash32(generation_config)


def task_seed_map_from_frozen_tasks(tasks: list[dict[str, Any]]) -> list[list[Any]]:
    return [[str(task["task_key"]), int(task["seed"])] for task in tasks]


def ordered_task_identity(tasks: list[dict[str, Any]]) -> list[dict[str, int | str]]:
    return [
        {
            "task_key": str(task["task_key"]),
            "problem_id": int(task["problem_id"]),
            "sample_id": int(task["sample_id"]),
            "seed": int(task["seed"]),
        }
        for task in tasks
    ]


def validate_effective_seed_map(tasks: list[dict[str, Any]], base_seed: int) -> bool:
    for task in tasks:
        expected = int(base_seed) + int(task["problem_id"]) * 1000 + int(task["sample_id"])
        if int(task["seed"]) != expected:
            return False
        expected_key = f"aime24:p{int(task['problem_id'])}:s{int(task['sample_id'])}:seed{expected}"
        if str(task["task_key"]) != expected_key:
            return False
    return True


def build_portable_reference_generation_semantics(
    *,
    task_manifest_sha256: str,
    tasks: list[dict[str, Any]],
    model_name: str,
    model_identity_hash: str,
    tokenizer_identity_hash: str,
    model_dtype: str,
    context_limit: int,
    resolved_pad_token_id: int | None,
    resolved_eos_token_ids: list[int],
    base_seed: int = 42,
    do_sample: bool = True,
    temperature: float = 0.6,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    num_return_sequences: int = 1,
    max_new_tokens: int = 32768,
    force_think_prefix: bool = True,
    add_special_tokens: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": PORTABLE_REFERENCE_GENERATION_SCHEMA,
        "dataset": "aime24",
        "task_manifest_sha256": task_manifest_sha256,
        "task_count": len(tasks),
        "ordered_task_identity": ordered_task_identity(tasks),
        "base_seed": int(base_seed),
        "task_seed_algorithm": "effective_seed = base_seed + problem_id * 1000 + sample_id",
        "task_seed_map": task_seed_map_from_frozen_tasks(tasks),
        "model_name": model_name,
        "model_identity_hash": model_identity_hash,
        "tokenizer_identity_hash": tokenizer_identity_hash,
        "model_dtype": model_dtype,
        "prompt_protocol": "deepseek_r1_recommended",
        "user_prompt_template": REFERENCE_USER_PROMPT_TEMPLATE,
        "chat_template": REFERENCE_CHAT_TEMPLATE_SEMANTICS,
        "force_think_prefix": bool(force_think_prefix),
        "think_prefix": "<think>\n" if force_think_prefix else "",
        "do_sample": bool(do_sample),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "repetition_penalty": float(repetition_penalty),
        "num_return_sequences": int(num_return_sequences),
        "max_new_tokens": int(max_new_tokens),
        "context_limit": int(context_limit),
        "add_special_tokens": bool(add_special_tokens),
        "resolved_pad_token_id": resolved_pad_token_id,
        "resolved_eos_token_ids": [int(x) for x in resolved_eos_token_ids],
    }


def portable_reference_generation_hash(payload: dict[str, Any]) -> str:
    reject_absolute_paths(payload)
    return canonical_hash32(payload)


def build_experiment_config_set_payload(configs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "aime24_pseudodecode_config_set_v1",
        "configs": [
            {
                "config": cfg.get("config"),
                "method": cfg.get("method"),
                "mode_role": cfg.get("mode_role"),
                "sink_length": cfg.get("sink_length"),
                "recent_length": cfg.get("recent_length"),
                "resolved_method_config": cfg.get("resolved_method_config"),
            }
            for cfg in configs
        ],
    }


def experiment_config_set_hash(configs: list[dict[str, Any]]) -> str:
    return canonical_hash32(build_experiment_config_set_payload(configs))


def reject_absolute_paths(payload: Any) -> None:
    paths = find_absolute_path_strings(payload)
    if paths:
        raise ValueError(f"portable reference generation payload contains absolute path(s): {paths[:3]}")


def find_absolute_path_strings(payload: Any) -> list[str]:
    found: list[str] = []
    if isinstance(payload, str):
        if os.path.isabs(payload):
            found.append(payload)
        return found
    if isinstance(payload, dict):
        for value in payload.values():
            found.extend(find_absolute_path_strings(value))
    elif isinstance(payload, list):
        for value in payload:
            found.extend(find_absolute_path_strings(value))
    return found


def mutate_payload(payload: dict[str, Any], **changes: Any) -> dict[str, Any]:
    out = deepcopy(payload)
    out.update(changes)
    return out
