# Environment

Repository: `pytenter/Bounded-pattrenKV-method`

Local checkout: `/data/zypan/Bounded-pattrenKV-pseudodecode-3090`

Branch: `sys/causal-v4-25-kernel-v1`

Start HEAD: `19128a1515de7a7a5eea6ed18ddd1a26686e2f6f`

Remote policy:

- allowed push: `git push bounded sys/causal-v4-25-kernel-v1`
- forbidden: `git push origin`
- forbidden: force push

Phase: `S6-B.1 - SERVING-NATIVE PATTERNKV BATCH ABI + OPERATOR ADAPTATION STUDY`

Mode: design/audit only. Production CUDA kernels, selector semantics, quantization semantics, K layout, V4 ratio, and default runtime paths were not changed.

Frozen algorithm:

- K: INT2
- Base V: INT2
- selected top 25% eligible V: INT4
- sink: 16
- recent: 128
- residual: 128
- group: 128
- selector: causal importance times positive V2->V4 local gain
- V2/V4: independent affine quantization streams

Validation commands for this phase:

```bash
python -m compileall bench models quant scripts tests
pytest -q
git diff --check
```
