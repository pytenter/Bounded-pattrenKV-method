# Source Inventory

## Local Repository

- Repository: `/data/zypan/Bounded-pattrenKV-pseudodecode-3090`
- Branch at audit start: `sys/causal-v4-25-kernel-v1`
- HEAD at audit start: `3c6c6710f8b1265477ade76768d310c0c5692739`
- Frozen release branch: `release/causal-v4-25-system-final`
- Frozen release SHA: `8d60485b5d2c93b7c1d478efc449de56d28159c3`

## Git Safety Audit

The required safety audit was run before edits:

```text
pwd
git branch --show-current
git rev-parse HEAD
git status --short
git status --porcelain=v1
git diff --stat
git diff --name-status
git diff --check
git ls-files --others --exclude-standard
git remote -v
```

Observed:

- branch matched `sys/causal-v4-25-kernel-v1`;
- HEAD matched `3c6c6710f8b1265477ade76768d310c0c5692739`;
- no tracked dirty diff existed at audit start;
- unrelated untracked report artifacts existed under prior report directories and were left untouched;
- `bounded` points to `git@github.com:pytenter/Bounded-pattrenKV-method.git`;
- `origin` points to `https://github.com/HCOOOH/PatternKV.git`.

## Local Search

Tracked working tree and git history were searched for:

```text
AMC
AMC23
AMC24
AMC 2024
amc24
amc_24
AMC_24
Avg@8
Maj@8
majority
8 responses
num_samples
n_samples
NuminaMath
Numina
Li et al.
PatternKV
temperature
top_p
top_k
do_sample
max_new_tokens
answer parser
multiple choice
choice
```

Result:

- no hidden AMC24 dataset, AMC runner, AMC parser, AMC result set, AMC row list, or AMC ground-truth mapping was found;
- the only AMC24-local package before this audit was `reports/amc24_long_cot_generalization_v1/`, which correctly classified the previous attempt as blocked;
- AIME24 utilities exist and can inform project protocol choices only after AMC24 dataset identity is established.

## PatternKV Official Repository

- Repository: `https://github.com/HCOOOH/PatternKV`
- Remote ref inspected: `origin/main`
- Commit inspected: `aba09a82e14732f6a0ed1f2b133925ff368d0538`
- Remote heads/tags observed: only `refs/heads/main` at `aba09a82e14732f6a0ed1f2b133925ff368d0538`; no tags.

Result:

- repository tree contains README/example implementation material;
- no AMC/AIME/Long-CoT evaluation script was present;
- no dataset preparation script was present;
- no Avg@8/Maj@8 evaluator was present;
- no prompt/sampling implementation for AMC was present.

## PatternKV Paper

- Title: "PatternKV: Flattening KV Representation Expands Quantization Headroom"
- arXiv: `2510.05176`
- Version inspected: `v1`
- PDF inspected from `https://arxiv.org/pdf/2510.05176`
- HTML inspected from `https://arxiv.org/html/2510.05176v1`

Result:

- paper supports AMC24 presence and high-level Long-CoT metric definitions;
- paper does not publish AMC24 row identity, seeds, tie policy, exact prompt template, or exact generation configuration for AMC.

## NuminaMath

- Dataset family resolved from PatternKV citation: NuminaMath.
- Primary HF dataset inspected: `AI-MO/NuminaMath-CoT`
- Revision pinned from HF API: `9d8d210c9f6a36c8f3cd84045668c9b7800ef517`
- Dataset card URL: `https://huggingface.co/datasets/AI-MO/NuminaMath-CoT`

Result:

- schema is insufficient to isolate AMC24 rows;
- dataset card gives source category counts, including `amc_aime`, but not AMC24 row IDs or an AMC24 split/filter.

The related `AI-MO/NuminaMath-TIR` dataset was also checked as a forensic branch because it is derived from NuminaMath-CoT, but it is not the PatternKV-cited dataset and does not resolve AMC24 identity.
