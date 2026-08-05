from scripts.audit_passage_retrieval_zh_decode_delta import (
    determine_final_status,
    duplicate_primary_key_count,
    event_row_delta,
    expected_decode_update_positions,
    infer_layer_head_shape,
    row_key,
    rows_per_layer_head,
    sha256_text,
    split_key_sets,
)


def test_split_key_sets_is_order_invariant():
    left = [
        {"task": "t", "phase": "decode", "kv_type": "k", "layer": "1", "kv_head": "0", "bucket": "bucketfirst", "metric": "m1"},
        {"task": "t", "phase": "decode", "kv_type": "k", "layer": "0", "kv_head": "0", "bucket": "bucketfirst", "metric": "m1"},
    ]
    right = [
        {"task": "t", "phase": "decode", "kv_type": "k", "layer": "0", "kv_head": "0", "bucket": "bucketfirst", "metric": "m1"},
        {"task": "t", "phase": "decode", "kv_type": "k", "layer": "2", "kv_head": "0", "bucket": "bucketfirst", "metric": "m1"},
    ]
    key_fields = ["task", "phase", "kv_type", "layer", "kv_head", "bucket", "metric"]
    summary = split_key_sets(left, right, key_fields)
    assert len(summary["common_keys"]) == 1
    assert len(summary["left_only_keys"]) == 1
    assert len(summary["right_only_keys"]) == 1
    assert summary["common_keys"][0] == row_key(left[1], key_fields)


def test_duplicate_primary_key_count_detects_repeat():
    rows = [
        {"task": "t", "phase": "decode", "kv_type": "k", "layer": "0", "kv_head": "0", "bucket": "bucketfirst", "metric": "m1"},
        {"task": "t", "phase": "decode", "kv_type": "k", "layer": "0", "kv_head": "0", "bucket": "bucketfirst", "metric": "m1"},
        {"task": "t", "phase": "decode", "kv_type": "k", "layer": "0", "kv_head": "1", "bucket": "bucketfirst", "metric": "m1"},
    ]
    assert duplicate_primary_key_count(rows, ["task", "phase", "kv_type", "layer", "kv_head", "bucket", "metric"]) == 1


def test_infer_layer_head_shape_and_rows_per_layer_head_are_not_hardcoded():
    rows = []
    for layer in range(4):
        for head in range(3):
            rows.append({"layer": str(layer), "kv_head": str(head)})
            rows.append({"layer": str(layer), "kv_head": str(head)})
    shape = infer_layer_head_shape(rows)
    assert shape["hidden_layers"] == 4
    assert shape["kv_heads"] == 3
    assert shape["layer_head_product"] == 12
    assert rows_per_layer_head(rows) == 2


def test_event_row_delta_matches_structure():
    assert event_row_delta(32 * 8, 1) == 256
    assert event_row_delta(32 * 8, 2) == 512


def test_expected_decode_update_positions_zero_based_and_one_based():
    assert expected_decode_update_positions(127, prefill_length=0, interval=128, residual_length=128, index_origin=1) == []
    assert expected_decode_update_positions(128, prefill_length=0, interval=128, residual_length=128, index_origin=1) == [128]
    assert expected_decode_update_positions(129, prefill_length=0, interval=128, residual_length=128, index_origin=1) == [128]
    assert expected_decode_update_positions(128, prefill_length=0, interval=128, residual_length=128, index_origin=0) == [127]


def test_expected_decode_update_positions_prefill_residual_crossing():
    assert expected_decode_update_positions(5, prefill_length=123, interval=128, residual_length=128, index_origin=1) == [5]
    assert expected_decode_update_positions(32, prefill_length=4986, interval=128, residual_length=128, index_origin=1) == [6]
    assert expected_decode_update_positions(257, prefill_length=0, interval=128, residual_length=128, index_origin=1) == [128, 256]


def test_determine_final_status_prefers_data_insufficient_when_v100_raw_missing():
    evidence = {
        "pattern_gain_extra_rows": 512,
        "dynamic_extra_rows": 256,
        "responsible_samples": ["s1"],
        "v100_event_count": None,
        "gpu4090_event_count": 1,
        "event_count_delta": None,
        "expected_pattern_gain_delta": 512,
        "expected_dynamic_delta": 256,
        "raw_data_available": {
            "v100_generation": False,
            "v100_observer": False,
            "gpu4090_generation": True,
            "gpu4090_observer": True,
        },
        "summarizer_difference_proven": False,
        "runtime_difference_proven": False,
        "partial_alignment": True,
    }
    assert determine_final_status(evidence) == "data_insufficient"


def test_determine_final_status_requires_more_than_row_counts():
    evidence = {
        "pattern_gain_extra_rows": 512,
        "dynamic_extra_rows": 256,
        "responsible_samples": [],
        "v100_event_count": None,
        "gpu4090_event_count": None,
        "event_count_delta": None,
        "expected_pattern_gain_delta": 512,
        "expected_dynamic_delta": 256,
        "raw_data_available": {
            "v100_generation": True,
            "v100_observer": True,
            "gpu4090_generation": True,
            "gpu4090_observer": True,
        },
        "summarizer_difference_proven": False,
        "runtime_difference_proven": False,
        "partial_alignment": False,
    }
    assert determine_final_status(evidence) == "unexplained"


def test_determine_final_status_generation_boundary_case():
    evidence = {
        "pattern_gain_extra_rows": 512,
        "dynamic_extra_rows": 256,
        "responsible_samples": ["s1"],
        "v100_event_count": 0,
        "gpu4090_event_count": 1,
        "event_count_delta": 1,
        "expected_pattern_gain_delta": 512,
        "expected_dynamic_delta": 256,
        "token_hash_equal": True,
        "raw_data_available": {
            "v100_generation": True,
            "v100_observer": True,
            "gpu4090_generation": True,
            "gpu4090_observer": True,
        },
        "summarizer_difference_proven": False,
        "runtime_difference_proven": False,
        "partial_alignment": True,
    }
    assert determine_final_status(evidence) == "explained_by_generation_length_boundary"


def test_determine_final_status_partial_case():
    evidence = {
        "pattern_gain_extra_rows": 512,
        "dynamic_extra_rows": 200,
        "responsible_samples": ["s1"],
        "v100_event_count": 0,
        "gpu4090_event_count": 1,
        "event_count_delta": None,
        "expected_pattern_gain_delta": None,
        "expected_dynamic_delta": None,
        "raw_data_available": {
            "v100_generation": True,
            "v100_observer": True,
            "gpu4090_generation": True,
            "gpu4090_observer": True,
        },
        "summarizer_difference_proven": False,
        "runtime_difference_proven": False,
        "partial_alignment": True,
    }
    assert determine_final_status(evidence) == "partially_explained"


def test_sha256_text_is_reproducible():
    assert sha256_text("passage_retrieval_zh") == sha256_text("passage_retrieval_zh")
