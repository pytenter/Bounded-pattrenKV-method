# PatternKV Paper Audit

## Paper Identity

- Title: "PatternKV: Flattening KV Representation Expands Quantization Headroom"
- arXiv identifier: `2510.05176`
- Version inspected: `v1`
- Authors: Ji Zhang, Yiwei Li, Shaoxiong Feng, Peiwen Yuan, Xinglin Wang, Jiayi Shi, Yueqi Zhang, Chuyi Tan, Boyuan Pan, Yao Hu, Kan Li
- Source files inspected: arXiv PDF and arXiv HTML

## Long-CoT Benchmark Evidence

Section 4.1/4.2 states that reasoning benchmarks include GSM8K, AIME, and AMC. The AMC citation is Li et al. 2024a.

Table 2 is titled:

```text
Overall Results on the Long-CoT Benchmark at 2-bit precision.
```

The Table 2 columns include:

```text
AIME 25
AIME 24
AMC 24
AMC 23
Avg@8
Maj@8
```

Appendix Table 8 reports the same long-CoT benchmark family at 4-bit precision.

## Metric Definitions

The paper explicitly states that for each problem it generates eight independent responses and reports:

- `Avg@8`: per-sample accuracy averaged over the eight responses;
- `Maj@8`: problem-level accuracy under majority voting across the eight responses.

This supports:

```text
responses_per_problem = 8
Avg@8 = correct response observations / (N problems x 8)
Maj@8 = problem-level majority-vote accuracy
```

It does not support:

```text
original seed list
majority tie handling
invalid parse handling
exact answer parser
exact prompt template
exact sampling parameters
exact max generation limit
```

## Reference Numbers

PatternKV Table 2, Llama-8B, INT2, AMC24:

| Method | Avg@8 | Maj@8 |
| --- | ---: | ---: |
| FP16 | 53.06 | 60.22 |
| KIVI | 30.52 | 46.05 |
| PatternKV | 34.44 | 42.11 |

These are `REFERENCE_ONLY` values. They are not local reproduced numbers and must not be mixed into project raw results.

PatternKV Table 8, Llama-8B, INT4, AMC24, was inspected as supplemental context. The text extraction around the table was incomplete enough that only the INT2 AMC24 reference values above are frozen in this audit.

## Missing From Paper

The paper does not provide:

- dataset revision or file path for AMC24;
- row-selection rule for AMC24;
- problem IDs/source row IDs;
- answer-space representation;
- exact parser;
- exact prompt template;
- exact generation configuration;
- original seed list;
- majority tie policy.
