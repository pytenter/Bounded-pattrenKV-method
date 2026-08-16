# Amdahl Bounds

- Bound A, all <10us kernels free: save 23.527 ms/token, TPOT 168.170 ms, speedup 1.140x.
- Bound B, formal Value/cache/softmax launch-orchestration terms free: save 71.618 ms/token, TPOT 120.079 ms, speedup 1.596x.
- Bound C, FP16 Value tail fragmentation free while measured GPU work remains: save 37.342 ms/token, TPOT 154.355 ms.
- Bound D, QK FP16 tail fragmentation free while measured GPU work remains: save 37.246 ms/token, TPOT 154.451 ms.
- Bound E, all measured attention orchestration free: save 108.864 ms/token, TPOT 82.833 ms, speedup 2.314x.

Bounds use formal CAUSAL TPOT 191.697 ms/token. Where previous formal component ranges exist, those wall times are used instead of profiler-overhead TPOT; new PyTorch trace supplies actual GPU kernel time/counts. QK FP16 tail wall time remains profiler-derived and is therefore less reliable than Value/cache/softmax.
