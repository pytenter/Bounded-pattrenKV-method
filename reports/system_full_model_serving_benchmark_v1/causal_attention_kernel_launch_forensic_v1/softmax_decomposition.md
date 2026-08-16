# Softmax Decomposition

- CUDA kernel calls/token: 928.000
- GPU kernel time/token: 2.062 ms
- Classification: wrapper dominated
- Old CAUSAL path uses the global fixed-split softmax; state merge is absent from production files.
