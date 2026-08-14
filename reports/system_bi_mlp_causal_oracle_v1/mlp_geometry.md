# MLP Geometry

{
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "gate_proj": {
    "M": "batch*tokens",
    "K": 4096,
    "N": 14336
  },
  "up_proj": {
    "M": "batch*tokens",
    "K": 4096,
    "N": 14336
  },
  "down_proj": {
    "M": "batch*tokens",
    "K": 14336,
    "N": 4096
  },
  "oracle_backend": "v2"
}
