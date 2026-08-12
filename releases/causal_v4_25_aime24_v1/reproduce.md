# Reproduce CAUSAL-V4@25% AIME24 v1

## Repository

Use algorithm commit:

```bash
git checkout c73aeed3247c136859f695d5b238eeb357434b17
```

The archival metadata commit may be newer, but the frozen algorithm tag `causal-v4-25-aime24-v1` must resolve exactly to `c73aeed3247c136859f695d5b238eeb357434b17`.

## Environment

The recorded environment is in `software_environment.txt` and `hardware_environment.txt`. The run used `/data/zypan/.local/share/mamba/envs/patternkv/bin/python`.

## Model And Dataset

- Model path: `/data/zypan/blockgtq-repro/models/DeepSeek-R1-Distill-Llama-8B`
- Dataset path: `datasets/aime/aime24.jsonl`
- Dataset SHA256: `07ec3f0c489406676be9d6057e2f97c9c32bc18e856d13df1d05c76724cbb08f`
- Model identity and tokenizer/config hashes: `model_and_dataset_identity.json`

## Formal Runner

Runner: `scripts/run_aime24_full_causal25_quality.py`

Recovered launch manifest: `run/aime24_full_causal25_quality_4gpu/logs/formal_launcher_manifest.json`

The formal detached launch used these method/GPU worker commands:

```bash
/data/zypan/.local/share/mamba/envs/patternkv/bin/python scripts/run_aime24_full_causal25_quality.py --worker --phase formal --method-id FP16 --physical-gpu 1
/data/zypan/.local/share/mamba/envs/patternkv/bin/python scripts/run_aime24_full_causal25_quality.py --worker --phase formal --method-id PATTERN_BASE --physical-gpu 2
/data/zypan/.local/share/mamba/envs/patternkv/bin/python scripts/run_aime24_full_causal25_quality.py --worker --phase formal --method-id RANDOM_V4_25 --physical-gpu 3
/data/zypan/.local/share/mamba/envs/patternkv/bin/python scripts/run_aime24_full_causal25_quality.py --worker --phase formal --method-id CAUSAL_V4_25 --physical-gpu 4
```

Equivalent runner-level launch path:

```bash
/data/zypan/.local/share/mamba/envs/patternkv/bin/python scripts/run_aime24_full_causal25_quality.py --preflight --gpus 1,2,3,4
/data/zypan/.local/share/mamba/envs/patternkv/bin/python scripts/run_aime24_full_causal25_quality.py --launch-formal --detach --gpus 1,2,3,4
/data/zypan/.local/share/mamba/envs/patternkv/bin/python scripts/run_aime24_full_causal25_quality.py --aggregate
```

## Expected Workload

- Methods: FP16, PATTERN_BASE, RANDOM_V4_25, CAUSAL_V4_25
- Base seeds: 42, 43, 44
- Problems: AIME24 p00-p29
- Samples per problem/seed: 1
- Expected formal generations: 360
- Generation config: `generation_config.json`
- Formal config hash: `35fdc9fe0d9dda5e`

## Validation

Expected validation for this freeze:

```bash
/data/zypan/.local/share/mamba/envs/patternkv/bin/python -m compileall bench models quant scripts
/data/zypan/.local/share/mamba/envs/patternkv/bin/python -m pytest -q
git diff --check
```

The formal frozen validation observed `443 passed`, compileall passed, and git diff check passed.

## Command Recovery Status

Command recovery is complete for the formal runner, method IDs, seeds, problem count, generation config, model path, dataset path, and GPU worker assignments. Runtime tail acceleration workers were used only to finish missing already-defined formal tasks; they did not change algorithm parameters or result schema.
