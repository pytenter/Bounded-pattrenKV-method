# Pytest

Clean repaired system worktree validation:

```text
/data/zypan/.local/share/mamba/envs/patternkv/bin/python -m compileall bench models quant scripts tests
```

Result: passed.

```text
/data/zypan/.local/share/mamba/envs/patternkv/bin/python -m pytest -q
```

Result: `858 passed in 23.30s`.

The previous dirty-worktree result was `855 passed, 5 failed`. Those 5 method-count failures disappeared after removing the generalization commit from the clean system tree, so the previous failures are confirmed as generalization worktree contamination.
