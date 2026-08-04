from bench.paper_config import kivi_quantized_region_bits


def test_affine_bits():
    assert abs(kivi_quantized_region_bits(128, 2) - 2.25) < 1e-9
