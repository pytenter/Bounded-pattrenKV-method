# Qwen Backend Audit

Model path: `/home/qinch2023/modelscope_models/Qwen3-8B`. Config hash: `f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30`. The available Qwen3 PatternKV adapter imports native Qwen3 classes and is not a Llama class, but it reconstructs historical K/V during decode and is therefore not a valid compressed-domain performance backend.
