# Claim Audit

## Supported

- Same-GPU four-method comparison.
- True batch execution.
- Zero fallback.
- Reconciled allocator protocol.
- CAUSAL 2x FP16 max successful B in the tested C4096 setup.
- PatternKV 2x FP16 max successful B in the tested C4096 setup.
- KIVI 2x FP16 max successful B in the tested C4096 setup.
- CAUSAL ~4% throughput overhead vs PatternKV at C2048/B4.
- CAUSAL ~3% TPOT overhead vs PatternKV at D256.
- Nearly context-flat PatternKV/CAUSAL TPOT over the tested 2K-8K context range.

## Not Supported

- CAUSAL full-model speedup over FP16.
- CAUSAL full-model speedup over KIVI.
- Universal 2x capacity.
- Whole-GPU 2.5-bit memory.
- 84% full-model memory reduction.
- Production vLLM/SGLang integration.
