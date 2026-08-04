# Config Audit

Output tag: `longbench_21x50_8k_4090`
Run scope: `21x50` subset
Experiment name: `longbench_paper_v2_8k_single4090`
Description: PatternKV paper-v2 configuration-aligned LongBench reproduction with an 8K input cap. This is not the paper's strict 31.5K reproduction.

Git branch: `repro/patternkv-longbench-8k-single4090`
Git commit: `dbe38273c44c7c1c395eea747a9f0878a32fb504`
Model path: `/root/autodl-tmp/models/meta-llama/Llama-3.1-8B-Instruct`
Hardware target: NVIDIA GeForce RTX 4090 D 24GB, GPU 0 only.
MAX_INPUT_LENGTH: `8192`
Batch size: `1`; decoding: greedy; `use_cache=true`.

Methods:

- `fp16`: KV quantization disabled; backend `fp16`.
- `kivi_paper_g128`: k/v 2-bit, group_size 128, residual_length 128, asymmetric, official KIVI backend, persistent KV heads 8 for Llama GQA.
- `patternkv_paper`: k/v 2-bit, group_size 128, residual_length 128, 32 K patterns, 32 V patterns, G_pattern 128, post-RoPE selection.

Hashes:

```json
{
  "8k_config_sha256": "81aca44a46019c8f440af1e0726b9642866cd5049cd8519b33efce9ba2e961b6",
  "generation_config_sha256": "189fb0c0d7fd8a527db217c0a60a0e013f0394cd8800f9697a666a9e75e5f7fd",
  "maxlen_sha256": "75301d9cacf912e967c775997c94d9f021d176f940cb2d0318b99567220f52fb",
  "model_config_sha256": "29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e",
  "patternkv_extension_sha256": "53ac4566011071cd37bdca8083b9a632266cbaaacf1300bafe69e64fb4457aa6",
  "prompt_sha256": "5c1231ee6c3f198b0021e9c1ab9b66cbbfb52c432e0ed821a5afe89eefa17fd8",
  "scorer_sha256": "49c8d74106b2c191a462fc985f6e871f82de50d3d0e3cdadf7649e39106a7a6a",
  "tokenizer_config_sha256": "177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424"
}
```
