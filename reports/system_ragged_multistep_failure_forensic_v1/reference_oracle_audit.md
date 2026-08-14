# Reference Oracle Audit

Clean B1 references are built independently: A full prefill+16 decode, reset, B full prefill+16 decode, reset, then ragged forced replay with saved tokens/logits.

`REFERENCE_INTERLEAVING_REMOVED=true`

Repeated prefill determinism:

```json
{
  "first_diff": {
    "component": "k_centroid_values",
    "got_hash": "e2c856b5cfd71c96139158b3f7b1beb1f81ed2c73e9d9e14af692f8d43d38d31",
    "layer": 0,
    "ref_hash": "121854cea67cd05693b5e190bf775ed5c247b9f7566c5b2c95e66215967ab6ae"
  },
  "match": false
}
```
