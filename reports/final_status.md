# PatternKV 单卡 RTX 4090 第一阶段复现报告

## 1. 最终状态

PARTIAL PASS

原因：官方代码、独立环境、CUDA extension、量化单测、FP16/PatternKV 端到端 smoke 和 packed KV 路径均在当前服务器跑通；但当前服务器可见 GPU 是 RTX 3090 / SM86，不是请求的 RTX 4090 / SM89，因此不能声明 4090 验收 PASS。

## 2. 硬件与软件环境

- GPU: NVIDIA GeForce RTX 3090
- 显存: 24576 MiB
- Driver: 580.173.02
- CUDA runtime: PyTorch CUDA 12.4
- nvcc: /usr/local/cuda-12.4/bin/nvcc
- Python: 3.10.20
- PyTorch: 2.4.1+cu124
- Transformers: 4.43.1
- GCC/G++: 13.3.0
- PatternKV upstream commit: aba09a82e14732f6a0ed1f2b133925ff368d0538
- PatternKV repro commit: f3b63d311fd1ccf440655a6b7216cbd21c49ec7a on branch repro/patternkv-4090-smoke

## 3. 官方代码改动

- `.gitignore`: 忽略本地下载 wheel、build、cache；不影响算法。
- `patternkv.py`: 增加包导入锚点，满足 `import patternkv`；不影响算法。
- `models/llama_patternkv.py`: `cuml/cupy` 改可选导入；真实路径使用 PyTorch kmeans helper，未替换算法。
- `quant/setup.py`: 构建时加入 torch lib rpath，修复 `import patternkv_gemv`；不影响 kernel 数学。
- `scripts/run_smoke.py`: 统一 smoke runner 和只读统计。
- `tests/test_quant_extension.py`: 增加量化/CUDA 单测。

## 4. CUDA extension

- 编译状态: 成功
- 编译架构: SM86 (`TORCH_CUDA_ARCH_LIST=8.6`)，不是 SM89
- 导入状态: `import patternkv_gemv` 成功
- 单元测试: PASS
- 真实调用: 单测调用 `gemv_forward_cuda_outer_dim`；Smoke B cache 显示 packed K/V 被后续 decode 使用。

## 5. Smoke A / FP16 与 PatternKV 对照

当前保留的是长 prompt/160 token 对照结果：

- FP16: 成功，input=296，output=160，latency=7.072s，peak_reserved=16.54GB
- PatternKV: 成功，input=296，output=160，latency=9.703s，peak_reserved=16.53GB

## 6. Smoke B 路径覆盖

- Prefill Pattern mining: 执行，K/V centroids 初始 32 个，decode 后 layer0 为 33 / 33
- K residual quantization: 执行，`key_states_quant_trans` {'shape': [1, 8, 128, 24], 'dtype': 'torch.int32'}
- V residual quantization: 执行，`value_states_quant` {'shape': [1, 8, 384, 8], 'dtype': 'torch.int32'}
- V threshold: 执行，`v_mask` {'shape': [1, 8, 384], 'dtype': 'torch.uint8', 'mean': 0.9762369990348816}
- min-max assignment: decode 更新路径执行，K assignment max=32
- Decode Chebyshev update: 执行，Pattern 数从 32 增加到 33
- CUDA fused attention: PatternKV decode 调用 fused-with-base wrapper；量化单测直接验证 CUDA GEMV
- packed KV: 真实存在，K/V 历史 cache dtype 均为 int32

## 7. 统计信息

- K Pattern shape layer0: [8, 33, 128]
- V Pattern shape layer0: [8, 33, 128]
- V mask 利用率 layer0: 0.9762
- Quantized K tokens: 384; residual K tokens: 71; total KV len: 455
- Quantized V tokens: 384; residual V tokens: 71
- K/V packed dtype: torch.int32 / torch.int32
- Scale/min dtype: torch.float16 / torch.float16

## 8. 已知问题

- 官方仓库问题: `cuml/cupy` 未声明且未使用；`patternkv` 包名无导入锚点；`flash-attn` 依赖在 requirements 中被注释。
- RTX 4090 兼容问题: 当前无 4090，不能验证 SM89 kernel image、非法访存或性能。
- 软件版本问题: flash-attn 安装脚本跨文件系统 rename 失败，已手动安装 release wheel。
- 模型问题: 本地模型可用，路径 `/data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct`。
- 尚未验证内容: LongBench/GSM8K/AIME、大 batch、SM89 编译和真实性能。

## 9. 精确复现命令

见 `README_REPRO.md`。核心命令：

```bash
CUDA_VISIBLE_DEVICES=0 /data/zypan/kvarn-repro/tools/bin/micromamba run -n patternkv python tests/test_quant_extension.py
CUDA_VISIBLE_DEVICES=0 PATTERNKV_DEBUG_STATS=1 /data/zypan/kvarn-repro/tools/bin/micromamba run -n patternkv python scripts/run_smoke.py --model-path /data/zypan/blockgtq-repro/models/Llama-3.1-8B-Instruct --method patternkv --device cuda:0 --dtype float16 --k-bits 2 --v-bits 2 --group-size 128 --residual-length 128 --num-k-base 32 --num-v-base 32 --max-new-tokens 160 --output-json results/smoke_patternkv.json
```

## 10. 下一阶段建议

在真实 RTX 4090 上重建/重编译 SM89 后，做 LongBench 4任务 x 5条 smoke，比较 FP16、KIVI、PatternKV 三方法的质量、峰值显存、prefill/decode latency、TTFT/TPOT 和 Pattern 统计。
