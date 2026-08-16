# Main Promotion Plan

- main HEAD: `c7324746d7447be532bd6bdbe0c8d58dd5e30c67`
- system HEAD: `0ca6debff700f68ae8ff536e77ddb2cb1e68d69d`
- Ahead/behind: main_only=0, system_only=124
- Direct commits that would enter main: 124

## System Commits Not In Main

- `0ca6deb test: profile heterogeneous causal attention path`
- `d695322 test: close full-model serving capacity evidence`
- `359d82b test: record request-local system correctness evidence`
- `d1b2d48 test: add system forensic scripts`
- `5267681 fix: preserve request-local system runtime semantics`
- `cc50fdc test: diagnose PatternKV ragged multistep drift`
- `7099633 fix: preserve PatternKV ragged multistep state`
- `1f418f9 test: record request lifecycle prerequisite block`
- `6b9f32d fix: preserve PatternKV ragged decode semantics`
- `ea34144 feat: support PatternKV ragged K valid lengths`
- `1544904 docs: record generalization and system branch split`
- `b1c0ee0 Revert "Add CAUSAL V4 25 generalization results"`
- `3dcedb4 Add CAUSAL V4 25 generalization results`
- `3a9fa06 feat: assemble PatternKV ragged request caches`
- `c66676c feat: add PatternKV ragged batch decode MVP audit`
- `27d9982 test: close final PatternKV fixed-batch semantic gate`
- `9c86f32 test: validate batch-invariant MLP causal oracle`
- `57716a5 test: trace P2 production first divergence`
- `82d43a1 feat: define PatternKV prefill projection mode policy`
- `b7f7103 Evaluate BI V prefill projection mode`
- `9b49829 test: audit PatternKV V centroid semantic impact`
- `17fad41 test: trace actual-model prefill V centroid divergence`
- `7cada5a feat: integrate batch-invariant K projection into PatternKV prefill`
- `ca0463c perf: optimize batch-invariant K projection with persistent Triton GEMM`
- `9616629 feat: add batch-invariant PatternKV K projection prototype`
- `5722b4a test: evaluate serving-stable PatternKV K centroid construction`
- `03bf5b9 test: diagnose prefill kmeans numerical amplification`
- `a57a9bc test: trace actual-model PatternKV K assignment divergence`
- `3d071f9 test: validate PatternKV fixed batch on actual DeepSeek model`
- `970d1ba test: record request-local centroid performance sanity`
- `d90d930 fix: make PatternKV dynamic centroid state request-local`
- `df737b7 perf: integrate fused page operator into PatternKV decode runtime`
- `0022005 perf: add fused page-centric batch value operator`
- `98ac299 perf: profile page-centric PatternKV batch operator`
- `0ce940a perf: implement page-centric PatternKV batch decode MVP`
- `0d4e208 perf: validate page-centric PatternKV batch ABI`
- `e789a6f perf: design serving-native PatternKV batch ABI`
- `19128a1 perf: evaluate asymmetric PatternKV concurrency`
- `f27e4e0 perf: characterize asymmetric PatternKV runtime`
- `7844c86 perf: diagnose PatternKV K stride regression`
- `5b13e3a perf: evaluate stride-aware PatternKV K reader`
- `aaa1f4b perf: integrate contiguous-capacity PatternKV value cache`
- `5f67a4a perf: add stride-aware PatternKV V2 reader`
- `551fea5 perf: evaluate contiguous-capacity PatternKV cache`
- `139d128 perf: reprofile optimized PatternKV decode bottlenecks`
- `ed4b273 perf: evaluate page-native PatternKV attention reader`
- `92d74dd perf: add fixed-page PatternKV cache ABI`
- `54c4172 perf: evaluate GQA-aware PatternKV value kernels`
- `cf10095 perf: optimize PatternKV centroid table contribution`
- `274e193 perf: reduce PatternKV centroid histogram contention`
- `9cda2cc perf: decompose PatternKV centroid attention cost`
- `da0b6a8 perf: evaluate V2 kernel optimization candidates`
- `c644416 perf: deep-profile mixed V attention and cache mutation`
- `7dc760e perf: profile post-fusion PatternKV decode bottlenecks`
- `f8126bf perf: fuse mixed V2/V4 PatternKV value attention`
- `d59c349 docs: freeze CAUSAL-V4@25 AIME24 v1`
- `c73aeed run: complete Full AIME24 causal25 validation`
- `96e042d feat: allow formal worker seed subsets`
- `c34738b fix: harden AIME24 quality reporting`
- `e3fe0d3 fix: key PatternKV value selector by AIME task`
- `8d57a99 exp: add 4-GPU Full AIME24 quality runner`
- `83c46ed run: complete Value capacity budget study`
- `241b832 run: complete selective Value precision screen`
- `8d2278f exp: add causal and oracle V4 selectors`
- `2c62886 exp: add mixed-precision Value cache`
- `47cf601 run: complete direction-aware Value mechanism screen`
- `b686212 exp: add direction-aware Value assignment objectives`
- `fd6748b exp: audit mechanism-guided Value objectives`
- `0a64485 run: complete routing-value propagation diagnostic`
- `e3970de exp: record routing diagnostic preflight provenance`
- `3573378 exp: fix routing diagnostic provenance filtering`
- `c0bab1e exp: instrument routing and value-direction propagation`
- `f7f6ca9 audit: isolate canonical VarN semantics`
- `6cfed2b exp: diagnose KV norm-tail accumulation`
- `0f93d48 analysis: synthesize pseudo-decode formal decisions`
- `08e8334 run: complete formal pseudo-decode accumulation`
- `c6b6e9c fix: add matched execution-path controls`
- `177c626 exp: freeze AIME24 FP16 references and preflight gates`
- `0d15eb4 fix: resolve portable AIME24 generation config provenance`
- `7dbc224 exp: prepare AIME24 pseudo-decode diagnostics on 8x3090`

## Recommendation

Recommendation: C. later create a clean paper-release branch and merge that. The current system branch contains extensive benchmark/report evidence and raw-adjacent scientific artifacts; direct fast-forwarding `main` would make the reproduction entry noisy. Keep `main` reproduction-focused for now and promote a curated release when the paper-facing artifact boundary is decided.

No merge or fast-forward was executed.
