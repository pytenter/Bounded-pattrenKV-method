from argparse import Namespace

from bench.paper_config import apply_method_defaults


def test_paper_methods():
    k = Namespace(method="kivi_paper_g128", k_bits=4, v_bits=4, group_size=32, residual_length=64, num_k_base=1, num_v_base=1)
    p = Namespace(method="patternkv_paper", k_bits=4, v_bits=4, group_size=32, residual_length=64, num_k_base=1, num_v_base=1)
    apply_method_defaults(k)
    apply_method_defaults(p)
    assert (k.k_bits, k.v_bits, k.group_size, k.residual_length) == (2, 2, 128, 128)
    assert (p.k_bits, p.v_bits, p.group_size, p.residual_length, p.num_k_base, p.num_v_base) == (2, 2, 128, 128, 32, 32)
