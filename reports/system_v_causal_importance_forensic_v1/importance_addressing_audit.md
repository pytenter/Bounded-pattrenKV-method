# Importance Addressing Audit

The update uses the physical concatenated attention index directly as the destination importance index.

B1 A mapping:
```json
[
  {
    "segment": "sink",
    "physical_offset": 0,
    "physical_length": 16,
    "row_valid_length": 16,
    "logical_offset": 0,
    "logical_length": 16
  },
  {
    "segment": "packed",
    "physical_offset": 16,
    "physical_length": 128,
    "row_valid_length": 128,
    "logical_offset": 16,
    "logical_length": 128
  },
  {
    "segment": "pending",
    "physical_offset": 144,
    "physical_length": 113,
    "row_valid_length": 113,
    "logical_offset": 144,
    "logical_length": 113
  },
  {
    "segment": "recent",
    "physical_offset": 257,
    "physical_length": 128,
    "row_valid_length": 128,
    "logical_offset": 257,
    "logical_length": 128
  }
]
```

Ragged [A,B] row A mapping:
```json
[
  {
    "segment": "sink",
    "physical_offset": 0,
    "physical_length": 16,
    "row_valid_length": 16,
    "logical_offset": 0,
    "logical_length": 16
  },
  {
    "segment": "packed",
    "physical_offset": 16,
    "physical_length": 256,
    "row_valid_length": 128,
    "logical_offset": 16,
    "logical_length": 128
  },
  {
    "segment": "pending",
    "physical_offset": 272,
    "physical_length": 114,
    "row_valid_length": 113,
    "logical_offset": 144,
    "logical_length": 113
  },
  {
    "segment": "recent",
    "physical_offset": 386,
    "physical_length": 128,
    "row_valid_length": 128,
    "logical_offset": 257,
    "logical_length": 128
  }
]
```

The logical valid lengths for A match, but packed/pending/recent physical offsets differ because the ragged batch uses segment-wide physical maxima. Therefore row A's pending/recent attention mass is written to shifted importance indices in ragged execution.
