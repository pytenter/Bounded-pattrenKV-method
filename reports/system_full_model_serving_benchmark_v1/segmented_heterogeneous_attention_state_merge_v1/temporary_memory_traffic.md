# Temporary Memory Traffic

## Before

- Global score concat: present, `[B,Hq,Q,total]`.
- Global normalized probability tensor: present, `[B,Hq,Q,total]`.
- Segment Value output tensors: present, one per non-empty segment before tensor sum.
- Historical FP16 K/V materialization: zero.

## After

- Global score concat: eliminated on rolling state path.
- Global normalized probability tensor: eliminated on rolling state path.
- Segment-local score tensors: still present by V1 scope.
- Segment-local probability tensors: present, one per segment, used by existing Value backends.
- State workspace: `o [B,Hq,Q,D]` plus `m/l [B,Hq,Q]` per segment before merge.
- Causal-importance probabilities: computed per segment by rescaling local probabilities from merged state; no global probability tensor is concatenated.

The memory peak at C2048 B1 did not materially change: old/new full-lifecycle peak allocated was 16.513622016 GB in both formal workers.

