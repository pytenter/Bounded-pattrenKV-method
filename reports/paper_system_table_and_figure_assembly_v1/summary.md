# Summary

PAPER_SYSTEM_TABLE_AND_FIGURE_ASSEMBLY_V1_SUPPORTED. System experiments remain frozen; no new GPU experiments were run and no experimental numbers were modified.

In our tested RTX3090 / DeepSeek-R1-Distill-Llama-8B setup, FP16 provides the highest matched-B throughput. KIVI is slower than FP16 but substantially faster than PatternKV and CAUSAL. KIVI, PatternKV, and CAUSAL all reach B8 at C4096, compared with FP16 B4. CAUSAL is close to PatternKV at matched throughput, with 4.1% lower C2048/B4 throughput and 3.2% higher C4096/B1/D256 TPOT.

These system results position CAUSAL-V4@25% as a quality-oriented selective-precision runtime built on a PatternKV-like compressed system path, not as a throughput optimization over KIVI or FP16. Quality benefit should be cited from the frozen quality evidence rather than recomputed here.
