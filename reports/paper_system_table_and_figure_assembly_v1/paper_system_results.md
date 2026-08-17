# Paper System Results

### System Setup

We evaluate full-model decode serving on a single RTX3090 using DeepSeek-R1-Distill-Llama-8B. Measurements use decode-only timing, true batch execution, subprocess isolation, the same model/harness/protocol across all methods, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

### Throughput

At C2048/B4/D8, the throughput ranking is FP16 > KIVI >> PatternKV ~= CAUSAL. FP16 reaches 126.747 tokens/s, KIVI reaches 64.171 tokens/s, PatternKV reaches 25.168 tokens/s, and CAUSAL-V4@25% reaches 24.138 tokens/s. Relative to FP16, KIVI is 0.506x, PatternKV is 0.199x, and CAUSAL is 0.190x.

### Capacity and Memory

At C4096 under the reconciled allocator protocol, FP16 reaches maximum successful batch B4 and first OOM at B8. KIVI, PatternKV, and CAUSAL all reach B8 and first OOM at B16, doubling the observed maximum successful batch size versus FP16 in this tested setup. At matched C4096/B4/D8, full-model peak allocated memory is 19.00 GiB for FP16, 17.14 GiB for KIVI, 17.86 GiB for PatternKV, and 17.94 GiB for CAUSAL. These are full-model memory measurements, not KV-cache-only memory.

### CAUSAL Overhead over PatternKV

CAUSAL adds a small system overhead relative to PatternKV: 4.1% lower throughput at C2048/B4 and 3.2% higher TPOT at C4096/B1/D256. This is the measured runtime cost of CAUSAL's selective heterogeneous V2/V4 mechanism in this harness.

### Interpretation

CAUSAL-V4@25% should be positioned as quality-oriented selective precision built on a PatternKV-like compressed runtime, not as a throughput optimization over KIVI or FP16. Quality benefit must be referenced from the frozen quality evidence.
