# Before-Fix Planner

Before this round, segmented decode concatenated sink/packed/pending/recent score parts into a physical attention axis, applied ragged invalid masking, and called `torch.nn.functional.softmax` over `cache.total_tokens`. For request A this made the reduction trajectory depend on peer-induced physical width: B1 width 385 versus ragged B2 width 514.
