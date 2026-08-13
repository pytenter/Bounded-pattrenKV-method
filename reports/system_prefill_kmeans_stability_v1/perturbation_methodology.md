# Perturbation Methodology

Variants use production-equivalent layer0 prefill K-means: `batched_kmeans_fast_compiled(k=32,iters=30,tol=1e-4,seed=0)` followed by `batched_assign_compiled`. Perturbations include exact real delta, random norm-matched noise, and scaled real delta.
