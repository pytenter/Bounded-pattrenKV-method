# Mask Semantics

Valid token logits are compared only in request-local logical order. Invalid ragged physical positions are masked with `-65504.0` for `torch.float16` before softmax. `exp(invalid)` is zero in fp32, so invalid positions are semantically excluded even though they still participate in the physical reduction width.
