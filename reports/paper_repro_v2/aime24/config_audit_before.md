# AIME24 Config Audit Before

Generated: 2026-08-04

| 项目 | 当前仓库状态 | 可复用代码 | 需要新增内容 |
|---|---|---|---|
| AIME24 数据加载 | 无 AIME runner；外部已有 `/data/zypan/kvarn-repro/datasets/aime/aime24.jsonl` | JSONL 读写工具 | 复制/规范化到 `datasets/aime/aime24.jsonl`，校验 30 题和 checksum |
| AIME答案解析 | 无专用 AIME parser | GSM8K boxed parser 可参考 | `bench/aime_answer_parser.py`，支持 boxed/final/tail fallback 和 0..999 |
| DeepSeek-R1 prompt | `example.py` 有数学 boxed prompt 片段 | tokenizer chat template | AIME runner 中实现 DeepSeek-R1 推荐 prompt；明确不是论文逐字 prompt |
| 多次采样 | GSM8K runner 主要是单样本/shard | `do_sample` 参数模式可复用 | `num_samples_per_problem` 和 sample_id 维度 |
| paired seeds | 无 AIME paired seed | `random/numpy/torch` seed 设置 | `BASE_SEED + problem_id * 1000 + sample_id` |
| FP16 runner | LongBench/GSM8K 均有 | Llama load path | AIME 专用 runner |
| KIVI runner | LongBench 已有 `kivi_paper_g128` | `models/llama_kivi.py`, `bench.paper_config` | 接入 AIME runner |
| PatternKV runner | LongBench 已有 `patternkv_paper` | `models/llama_patternkv.py`, `bench.paper_config` | 接入 AIME runner |
| max_new_tokens=32768 | 无 AIME 长 CoT runner | generate 参数 | AIME 默认 32768，启动时检查模型上下文 |
| stop reason | GSM8K 有 `compute_stop_state` | `bench.gsm8k_utils` | AIME 记录 eos/length/oom/error/unknown |
| 断点续跑 | LongBench JSONL 跳过；GSM8K shard 跳过 | JSON 工具 | 单任务 JSON 原子写入，按 config_hash 校验 |
| 8GPU分片 | LongBench 6GPU 脚本 | 分阶段脚本模式 | AIME 8GPU 三阶段脚本 |
| Avg@N汇总 | 无 AIME 汇总 | GSM8K summary 思路 | Avg@N, strict_avg, majority, paired comparison |
| majority vote | 无 | 无 | 支持 N=8 时 Maj@8；N=2 只诊断 tie |

Initial HEAD: `ddd5a0097ecd8484074890254dfe0866235ed979`

Model scan: `MODEL_PATH` was not set, and no DeepSeek-R1-Distill-Llama-8B candidate was found in the requested local directories. Llama-3.1-8B-Instruct was found but intentionally not used as a substitute.
