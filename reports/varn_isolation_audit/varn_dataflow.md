# KVarN VarN Dataflow

Canonical source path:

```text
hidden
-> K/V projection
-> RoPE on K only
-> Hadamard rotation along head_dim
-> VarN/Sinkhorn scaling
-> asymmetric RTN quantization
-> metadata storage
-> dequant + scale restore
-> rotated-Q QK path and output un-rotation, or equivalent K/V un-rotation
-> attention aggregation
```

RoPE order: K projection -> RoPE -> Hadamard -> VarN/Sinkhorn -> quantization; V has no RoPE and enters Hadamard directly after V projection

Hadamard order: K/V source -> Hadamard rotation along head_dim -> VarN/Sinkhorn scaling -> low-bit quantization

Decode restore: dequantize rotated/normalized tile scales first; rotate Q for QK in decode and un-rotate output, or equivalently un-rotate dequantized K/V before attention
