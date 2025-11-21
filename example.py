
import warnings
warnings.filterwarnings("ignore")
import torch
import random
from models.llama_patternkv import LlamaForCausalLM_PatternKV
from transformers import LlamaConfig, AutoTokenizer
from datasets import load_dataset

model_name = "your path to Llama3"

random.seed(0)
torch.manual_seed(0)

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


prompt = ''
prompt = "John takes care of 10 dogs. Each dog takes .5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of dogs?"
prompt += "\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n"
inputs = enc(prompt, return_tensors="pt").input_ids.cuda()

output = model.generate(inputs, max_new_tokens=512)
config_str = f"# prompt tokens: {inputs.shape[1]}, K bit: {config.k_bits}, v_bits: {config.v_bits}, group_size: {config.group_size}, residual_length: {config.residual_length}"

print(prompt + "\n" + "=" * 10 + f'\n{config_str}\n' + "=" * 10 + "\nKiVi Output:")
print(enc.decode(output[0].tolist()[inputs.shape[1]:], skip_special_tokens=True))