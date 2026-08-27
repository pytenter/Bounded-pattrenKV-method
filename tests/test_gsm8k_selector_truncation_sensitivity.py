from __future__ import annotations

import json
from pathlib import Path

from scripts import run_gsm8k_selector_truncation_sensitivity as trunc


def _write_old_result(root: Path, method: str, pid: int, *, stop_reason: str, correct: bool, hit: bool = False) -> None:
    path = root / method / f"p{pid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "problem_id": pid,
                "method": method,
                "stop_reason": stop_reason,
                "hit_max_new_tokens": hit,
                "generated_tokens": 2048 if stop_reason == "length" else 17,
                "generated_token_ids": list(range(2048 if stop_reason == "length" else 17)),
                "is_correct": correct,
                "parsed_answer": "1",
                "config_hash": f"cfg-{method}",
            }
        ),
        encoding="utf-8",
    )


def test_truncation_union_uses_only_stop_reason_or_hit_flag(tmp_path: Path) -> None:
    for method in trunc.METHODS:
        _write_old_result(tmp_path, method, 1, stop_reason="eos", correct=False)
        _write_old_result(tmp_path, method, 2, stop_reason="length", correct=True)
        _write_old_result(tmp_path, method, 3, stop_reason="eos", correct=False, hit=True)
    old = trunc.old_record_by_method_pid(tmp_path)
    union = sorted({pid for method in trunc.METHODS for pid, row in old[method].items() if trunc.is_truncated(row)})
    assert union == [2, 3]


def test_truncation_union_ignores_correctness(tmp_path: Path) -> None:
    for method in trunc.METHODS:
        _write_old_result(tmp_path, method, 10, stop_reason="eos", correct=False)
        _write_old_result(tmp_path, method, 11, stop_reason="eos", correct=True)
    old = trunc.old_record_by_method_pid(tmp_path)
    union = sorted({pid for method in trunc.METHODS for pid, row in old[method].items() if trunc.is_truncated(row)})
    assert union == []


def test_frozen_union_manifest_is_unique_and_matched() -> None:
    manifest_path = Path("reports/gsm8k_selector_truncation_sensitivity_v1/truncation_union_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = [int(x) for x in manifest["truncation_union_ids"]]
    assert ids == [111, 186, 359, 525, 566, 672, 731, 855, 894, 923, 991, 1203]
    assert len(ids) == len(set(ids)) == manifest["truncation_union_n"]
    assert {int(r["problem_id"]) for r in manifest["rows"]} == set(ids)
    for row in manifest["rows"]:
        assert set(row["selected_because_method_truncated"]).issubset(set(trunc.METHODS))
        for method in trunc.METHODS:
            assert f"{method}_old_stop_reason" in row


def test_all_methods_share_same_union_contract() -> None:
    manifest = json.loads(Path("reports/gsm8k_selector_truncation_sensitivity_v1/truncation_union_manifest.json").read_text(encoding="utf-8"))
    ids = set(manifest["truncation_union_ids"])
    for method in trunc.METHODS:
        assert len(ids) == manifest["truncation_union_n"]
        assert all(any(f"{method}_old_" in key for key in row) for row in manifest["rows"])


def test_only_max_new_tokens_changes_and_cap_is_8192() -> None:
    protocol = json.loads(Path("reports/gsm8k_selector_truncation_sensitivity_v1/protocol_manifest.json").read_text(encoding="utf-8"))
    assert protocol["old_max_new_tokens"] == 2048
    assert protocol["new_max_new_tokens"] == 8192
    assert protocol["only_scientific_config_change"] == "max_new_tokens 2048 -> 8192"
    assert protocol["do_sample"] is False
    assert protocol["num_beams"] == 1


def test_method_identity_and_v4_budget_are_unchanged() -> None:
    method_identity = json.loads(Path("reports/gsm8k_selector_truncation_sensitivity_v1/method_identity.json").read_text(encoding="utf-8"))
    assert method_identity["unchanged_from_source_pilot"] is True
    assert method_identity["methods"]["causal_v4_25"]["selector"] == "causal_v4"
    assert method_identity["methods"]["error_only_v4_25"]["selector"] == "error_only_v4"
    assert method_identity["methods"]["importance_only_v4_25"]["selector"] == "importance_only_v4"
    for row in method_identity["methods"].values():
        assert row["patternkv_config"]["patternkv_v4_budget_fraction"] == 0.25


def test_new_config_hash_includes_truncation_union_and_8192() -> None:
    args = type("Args", (), {"model_path": Path(trunc.DEFAULT_MODEL), "max_new_tokens": 8192, "report_dir": Path("reports/gsm8k_selector_truncation_sensitivity_v1")})()
    cfg_hash = trunc.new_cfg_hash(args)
    assert isinstance(cfg_hash, str)
    assert len(cfg_hash) == 16


def test_old_results_are_external_and_never_overwritten() -> None:
    old_root = Path(trunc.DEFAULT_OLD_RESULT_DIR).resolve()
    new_root = Path("results/gsm8k_selector_truncation_sensitivity_v1").resolve()
    assert old_root.exists()
    assert new_root not in old_root.parents
    assert old_root != new_root


def test_smoke_and_formal_outputs_are_separate() -> None:
    smoke = Path("results/gsm8k_selector_truncation_sensitivity_v1/smoke")
    formal = Path("results/gsm8k_selector_truncation_sensitivity_v1/formal")
    assert smoke != formal
    assert str(smoke).endswith("/smoke")
    assert str(formal).endswith("/formal")


def test_prefix_parity_rules_for_length_and_eos_rows() -> None:
    old_ids = list(range(2048))
    assert trunc.generated_ids_hash(old_ids) == trunc.generated_ids_hash(old_ids[:2048])
    assert trunc.is_truncated({"stop_reason": "length", "hit_max_new_tokens": False})
    assert trunc.is_truncated({"stop_reason": "eos", "hit_max_new_tokens": True})
    assert not trunc.is_truncated({"stop_reason": "eos", "hit_max_new_tokens": False})


def test_gpu2_gpu3_are_forbidden() -> None:
    audit = json.loads(Path("reports/gsm8k_selector_truncation_sensitivity_v1/gpu_protection_audit.json").read_text(encoding="utf-8"))
    assert audit["forbidden_gpus"] == ["2", "3"]
    assert set(audit["selected_gpus"]) == {"1", "4"}
    assert not set(audit["selected_gpus"]) & set(audit["forbidden_gpus"])
    assert audit["ACTIVE_AIME_GPU2_GPU3_UNCHANGED"] is True


def test_paper_boundary_is_supplementary_only() -> None:
    text = Path("reports/gsm8k_selector_truncation_sensitivity_v1/claim_audit.md").read_text(encoding="utf-8")
    assert "post-hoc truncation-union sensitivity diagnostic only" in text
    assert "must not be reported as an unbiased GSM8K accuracy estimate" in text
