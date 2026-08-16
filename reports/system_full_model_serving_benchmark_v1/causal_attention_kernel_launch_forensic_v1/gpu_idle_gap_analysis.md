# GPU Idle Gap Analysis

- Total positive same-stream kernel gaps, all model kernels: 4914.266 ms/run
- Attention-associated same-stream positive gaps: 2596.367 ms/run (324.546 ms/token)
- Interpretation: approximate same-stream gaps from PyTorch trace timestamps; Nsight is required to prove CPU launch starvation versus dependency scheduling.
- gap_lt_5us: count=4889 total_gap_ms=4.580
- gap_5_10us: count=557 total_gap_ms=3.315
- gap_10_20us: count=16667 total_gap_ms=309.689
- gap_20_50us: count=60002 total_gap_ms=1800.790
- gap_50_100us: count=9550 total_gap_ms=647.154
- gap_gt_100us: count=9126 total_gap_ms=2148.738
