# FP16 vs CAUSAL Launch Comparison

| Metric | FP16 | CAUSAL |
| --- | ---: | ---: |
| kernels/token | 1342.000 | 12599.000 |
| attention kernels/token | NOT_AVAILABLE | 4768.000 |
| kernels <10us/token | 1012.000 | 12277.000 |
| kernels <20us/token | 1153.000 | 12341.000 |
| CUDA launch API calls/token | 1342.000 | 12503.000 |
| GPU kernel busy ms/token | 22.793 | 72.929 |
| estimated idle/gap ms/token | 33.504 | 324.546 |
