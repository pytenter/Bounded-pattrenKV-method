# Before-Fix Value Planner

Before this round, packed fused-page Value reduction iterated `t < T`, where T was the physical packed attention width, and full precision tails used `torch.matmul` over segment physical width. For ragged batches, peer-induced padded widths changed the reduction domain even when invalid attention probability was zero.
