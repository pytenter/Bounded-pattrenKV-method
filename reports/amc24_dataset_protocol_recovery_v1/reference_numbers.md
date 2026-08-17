# Reference Numbers

PatternKV paper values are preserved only as future sanity/reference context.

## INT2 Long-CoT, Table 2, Llama-8B

| Benchmark | Method | Avg@8 | Maj@8 |
| --- | --- | ---: | ---: |
| AMC24 | FP16 | 53.06 | 60.22 |
| AMC24 | KIVI | 30.52 | 46.05 |
| AMC24 | PatternKV | 34.44 | 42.11 |

## Use Constraint

```text
reference_only = true
```

These values are not local reproduced numbers and must not be used to tune sampling, parsing, prompt wording, dataset composition, or method configuration.
