from pathlib import Path

from scripts.run_longbench_paper_8k_single4090 import METHODS, parse_filter, plan, sample_indices


class Args:
    data_dir = "/root/Block-kvcache-experiment/data/LongBench"
    method_filter = None
    task_filter = None
    sample_filter = None


def test_runner_methods_are_three_phase_and_no_g32():
    assert METHODS == ("fp16", "kivi_paper_g128", "patternkv_paper")
    assert "kivi_original_g32" not in METHODS


def test_shell_runner_requires_gpu0_and_8192():
    text = Path("scripts/run_longbench_paper_8k_single4090.sh").read_text(encoding="utf-8")
    assert 'CUDA_VISIBLE_DEVICES:-}" != "0"' in text
    assert 'MAX_INPUT_LENGTH" != "8192"' in text
    assert "longbench_full_strict" in text


def test_runner_plans_full_local_panel_when_data_available():
    if not Path(Args.data_dir).exists():
        return
    p = plan(Args())
    assert len(p["tasks"]) == 21
    assert p["planned_per_method"] == 4750
    assert p["planned_total"] == 4750 * 3


def test_runner_filters_are_exact_tokens():
    assert parse_filter("trec samsum,passage_count") == {"trec", "samsum", "passage_count"}

    args = Args()
    args.task_filter = "trec"
    args.method_filter = "fp16"
    p = plan(args)
    assert p["tasks"] == ["trec"]
    assert p["methods"] == ["fp16"]


def test_sample_filter_supports_ranges():
    data = [{} for _ in range(60)]
    assert sample_indices(data, "0-2,5") == [0, 1, 2, 5]
    assert sample_indices(data, "0-49") == list(range(50))
