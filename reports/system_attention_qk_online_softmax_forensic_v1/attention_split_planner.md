# Attention Split Planner

Production currently performs one `torch.softmax` over the physical concatenated width. The request-invariant oracle uses logical fixed boundaries of 128 valid tokens: `[{'split': 0, 'start': 0, 'end': 128}, {'split': 1, 'start': 128, 'end': 256}, {'split': 2, 'start': 256, 'end': 384}, {'split': 3, 'start': 384, 'end': 385}]`.
