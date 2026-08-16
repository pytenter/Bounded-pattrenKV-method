# Fixed Split Size Design

Use request-local logical valid token indices and fixed split size `128`. Invalid physical padding is excluded before split planning, so peers cannot change request A's softmax reduction boundaries.
