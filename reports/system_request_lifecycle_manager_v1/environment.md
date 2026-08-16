# Environment

- cwd: `/data/zypan/Bounded-pattrenKV-pseudodecode-3090`
- branch: `sys/causal-v4-25-kernel-v1`
- head: `cc50fdc513181d2137438cc6a7c0dd8322ccf767`
- date: `2026-08-15`
- GPU note: `nvidia-smi` shows GPU0 occupied by an existing python process; GPUs 1-7 are idle except Xorg.
- Python note: `/usr/bin/python3` is available, but `torch`, `pytest`, and `pip` are not installed in this shell.
- Validation impact: compile-only checks can run; pytest and full model/GPU lifecycle regression require the PatternKV runtime Python environment.

