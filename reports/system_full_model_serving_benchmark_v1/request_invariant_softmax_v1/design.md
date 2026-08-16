# Request-Invariant Fixed-Split Softmax V1 Design

## Old Execution Path

The existing correct path builds a logical score tensor, scatters physical sink/packed/pending/recent scores into request-local logical order, pads to split size 128, computes per-split max and denominator states, merges split states left-to-right, computes logical probabilities, then scatters probabilities back to physical order for the existing Value readers. This is correct but launches many PyTorch operations per layer.

## New Experimental Path

`request_invariant_fixed_split_softmax_cuda` consumes physical scores `[B,H,1,T]` plus request-local segment valid lengths. One CUDA block handles one `(request, head)` pair. Inside the block, logical split boundaries are fixed at 128 tokens. Each split computes local max and denominator using block reductions, then thread 0 merges split states in deterministic left-to-right request-local split order. The final probabilities are written back to physical sink/packed/pending/recent order.

## Split Representation

The split signature is `(logical_length, 128, boundaries, left_to_right)`. It depends only on request-local logical length and not on batch size, peer lengths, peer content, or active-row ordering.

## Partial State Representation

The kernel computes `(m_i, l_i)` per split and merges them with max-rescaled online-softmax algebra. It does not yet fuse the weighted Value accumulator `a_i`; existing Value readers still consume the probability tensor.

## Deterministic Merge Topology

The merge order is left-to-right by request-local split index. GPU scheduling changes block execution order, but not a request's logical split boundaries or intra-request merge order.

## B1/B2/B4 Mapping

The CUDA unit test covers B1, B2, and B4 synthetic ragged score tensors. The full-model serving harness B>1 low-copy pre-gate still shows old assemble/split copies, so B>1 full-model performance is not considered closed in this round.

## Workspace Policy

The kernel uses block-local shared reduction storage and no persistent global partial-state workspace. Profile counters report logical partial and merge state sizes for capacity accounting only.

## Correctness Invariants

The implementation does not change selector, V4 ratio, K/V bitwidths, centroid semantics, quantization, page pools, or Value readers. Because ragged hard gates failed with the opt-in CUDA path, the optimized kernel is disabled by default.

## Complexity

Before: many PyTorch tensor operations per layer, including logical scatter/pad/split tensors and Python-level split merge. After opt-in: one CUDA softmax kernel launch per layer, with work proportional to `B * H * logical_length` and split count `ceil(L/128)` per request.
