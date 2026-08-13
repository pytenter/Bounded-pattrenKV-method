# K-Means Initialization Audit

Initialization samples token indices from `torch.rand(H,N).topk(k)` with seed 0. Initial token indices diverge: `False`. Initial centroid values still differ because selected token vectors differ.
