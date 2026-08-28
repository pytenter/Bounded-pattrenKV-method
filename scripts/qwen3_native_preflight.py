#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor/transformers_4_51_runtime"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def render_qwen3_prompt(tokenizer, problem: str) -> tuple[str, dict]:
    user_prompt = f"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    messages = [{"role": "user", "content": user_prompt}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    sig = inspect.signature(tokenizer.apply_chat_template)
    enable_thinking_supported = "enable_thinking" in sig.parameters
    if enable_thinking_supported:
        kwargs["enable_thinking"] = True
    rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if "deepseek_r1" in rendered.lower() or rendered.endswith("<think>\n"):
        raise RuntimeError("manual DeepSeek/R1 think prefix detected in rendered prompt")
    return rendered, {
        "raw_user_message": user_prompt,
        "prompt_protocol": "qwen3_native_thinking_v1",
        "enable_thinking_supported": enable_thinking_supported,
        "enable_thinking_resolved": bool(kwargs.get("enable_thinking", False)),
        "rendered_prompt": rendered,
        "rendered_prompt_sha256": sha256_text(rendered),
        "chat_template_sha256": sha256_text(str(getattr(tokenizer, "chat_template", ""))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/home/qinch2023/modelscope_models/Qwen3-8B")
    ap.add_argument("--dataset", default=str(ROOT / "datasets/aime/aime24.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "reports/qwen3_8b_aime24_native_generalization_v1"))
    ap.add_argument("--gate0", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    model_path = Path(args.model_path).resolve()
    out_dir = Path(args.out_dir)
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    if config.model_type != "qwen3":
        raise RuntimeError(f"config.model_type must be qwen3, got {config.model_type!r}")
    if "Qwen3ForCausalLM" not in list(getattr(config, "architectures", []) or []):
        raise RuntimeError(f"config.architectures missing Qwen3ForCausalLM: {config.architectures!r}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data_path = Path(args.dataset)
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 30:
        raise RuntimeError(f"expected 30 AIME24 rows, got {len(rows)}")
    rendered, prompt_info = render_qwen3_prompt(tokenizer, rows[0]["problem"])
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)

    weights = []
    for p in sorted(model_path.glob("*.safetensors")):
        weights.append({"name": p.name, "size": p.stat().st_size})
    identity = {
        "absolute_model_path": str(model_path),
        "model_directory_basename": model_path.name,
        "model_type": config.model_type,
        "architectures": list(getattr(config, "architectures", []) or []),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        "head_dim": getattr(config, "head_dim", None),
        "rope_scaling": getattr(config, "rope_scaling", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "torch_dtype": str(getattr(config, "torch_dtype", None)),
        "config_json_sha256": sha256_file(model_path / "config.json"),
        "generation_config_json_sha256": sha256_file(model_path / "generation_config.json"),
        "tokenizer_config_json_sha256": sha256_file(model_path / "tokenizer_config.json"),
        "chat_template_hash": prompt_info["chat_template_sha256"],
        "safetensors_index_sha256": sha256_file(model_path / "model.safetensors.index.json"),
        "weight_shards": weights,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "git_head": os.popen("git rev-parse HEAD").read().strip(),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    write_json(out_dir / "model_identity.json", identity)
    write_json(out_dir / "prompt_protocol.json", {**prompt_info, "input_token_hash": sha256_text(json.dumps(encoded.input_ids.tolist()))})

    gate = {
        "MODEL_IDENTITY_GATE": "PASS",
        "QWEN3_NATIVE_CLASS_GATE": "PENDING",
        "PROMPT_GATE": "PASS",
        "LOGITS_PARITY_GATE": "NOT_RUN",
        "CACHE_LIFECYCLE_GATE": "NOT_RUN",
        "PATTERN_IDENTITY_GATE": "NOT_RUN",
        "CAUSAL_IDENTITY_GATE": "NOT_RUN",
        "SHORT_SMOKE_GATE": "NOT_RUN",
        "FORMAL_CAP_SMOKE_GATE": "NOT_RUN",
        "FIXED_SUBSET_GATE": "NOT_RUN",
        "PROVENANCE_SCHEMA_GATE": "PARTIAL",
    }

    if args.gate0:
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "1", "2", "3"}:
            raise RuntimeError("Gate0 must run with CUDA_VISIBLE_DEVICES set to one of physical GPU0-3")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).to("cuda:0")
        model.eval()
        actual_class = model.__class__.__name__
        gate["actual_model_class"] = actual_class
        gate["QWEN3_NATIVE_CLASS_GATE"] = "PASS" if actual_class == "Qwen3ForCausalLM" else "FAIL"
        input_ids = encoded.input_ids.to("cuda:0")
        attention_mask = encoded.attention_mask.to("cuda:0")
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out[0, input_ids.shape[1]:].detach().cpu().tolist()
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        collapse = ("ffffffff" in text) or ("be be be" in text.lower())
        finite = all(torch.isfinite(p).all().item() for p in [next(model.parameters()).detach()])
        gate["GATE0_NATIVE_FP16_ORACLE"] = "PASS" if text.strip() and not collapse and finite else "FAIL"
        write_json(out_dir / "gate0_native_fp16_oracle.json", {
            "actual_model_class": actual_class,
            "config_model_type": model.config.model_type,
            "generated_text_head": text[:2000],
            "generated_tokens": len(gen_ids),
            "collapse_detected": collapse,
            "finite_parameter_check": finite,
        })
    write_json(out_dir / "preflight_gate.json", gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if gate.get("MODEL_IDENTITY_GATE") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
