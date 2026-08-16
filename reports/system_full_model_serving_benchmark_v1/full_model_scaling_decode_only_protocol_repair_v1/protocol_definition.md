# Decode-Only Protocol Definition

Each benchmark point runs in an independent subprocess.
The worker pre-fills all active requests before timing, synchronizes CUDA, resets decode-window peak memory counters, and then times only fixed-membership decode iterations.
Timed-window hard gates: prefill calls = 0, prefill tokens = 0, refill calls = 0, membership changes = 0.
Initial prefill and full lifecycle peak memory are recorded separately from decode-only TPOT and throughput.
