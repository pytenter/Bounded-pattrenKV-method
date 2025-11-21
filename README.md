# PatternKV: Flattening KV Representation Expands Quantization Headroom

Implementation of [PatternKV: Flattening KV Representation Expands Quantization Headroom](https://arxiv.org/abs/2510.05176)


## Setup


Rackages required:

```bash
conda create -n patternkv python=3.10
conda activate patternkv
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

CUDA implementation:

```bash
cd quant && pip install -e .
```

## Example

Load model with PatternKV: 

```python

import torch
import random
from models.llama_patternkv import LlamaForCausalLM_PatternKV
from transformers import LlamaConfig, AutoTokenizer, Qwen2Config, MistralConfig
from datasets import load_dataset

model_name = "your path to Llama3"

config = LlamaConfig.from_pretrained(model_name)

config.k_bits = 2 
config.v_bits = 2
config.group_size = 128 
config.residual_length = 128 
config.use_flash = True
config.num_k_base = 32
config.num_v_base = 32


model = LlamaForCausalLM_PatternKV.from_pretrained(
    pretrained_model_name_or_path=model_name,
    config=config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
).cuda()

enc = AutoTokenizer.from_pretrained(
    model_name, 
    use_fast=False, 
    trust_remote_code=True)


```




