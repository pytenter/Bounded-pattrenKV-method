# Merge State Audit

{
  "fixed_split_boundaries": [
    {
      "end": 128,
      "split": 0,
      "start": 0
    },
    {
      "end": 256,
      "split": 1,
      "start": 128
    },
    {
      "end": 384,
      "split": 2,
      "start": 256
    },
    {
      "end": 385,
      "split": 3,
      "start": 384
    }
  ],
  "note": "fixed split merge is deterministic for identical request-local logits because boundaries are independent of peer requests"
}
