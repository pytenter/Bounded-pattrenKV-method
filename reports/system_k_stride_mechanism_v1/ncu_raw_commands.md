# NCU Raw Commands

- `which ncu`: ``
- `which nsys`: ``

Nsight Compute was not used for final evidence in this phase. The portable fallback commands used were:

```bash
/usr/local/cuda-12.4/bin/cuobjdump --dump-resource-usage quant/patternkv_gemv.cpython-310-x86_64-linux-gnu.so > reports/system_k_stride_mechanism_v1/cuobjdump_resource_usage.txt
/usr/local/cuda-12.4/bin/cuobjdump --dump-sass quant/patternkv_gemv.cpython-310-x86_64-linux-gnu.so > reports/system_k_stride_mechanism_v1/cuobjdump_sass.txt
```
