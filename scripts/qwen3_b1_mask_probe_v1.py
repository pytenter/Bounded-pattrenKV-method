from __future__ import annotations
import json, os, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
VENDOR=os.environ.get('QWEN3_TRANSFORMERS_VENDOR')
if VENDOR and VENDOR not in sys.path: sys.path.insert(0,VENDOR)
from transformers import AutoConfig, AutoTokenizer
import models.qwen3_patternkv_system as sysmod
from models.qwen3_patternkv_system import Qwen3ForCausalLM_PatternKVCompressed
OUT=ROOT/'reports/qwen3_v100_system_generalization_v1'; MODEL='/home/qinch2023/modelscope_models/Qwen3-8B'; CONTEXT=512
BASE='Mathematics proof sketch: define a sequence and reason about modular arithmetic carefully. '
CFG=dict(k_bits=2,v_bits=2,group_size=128,sink_length=16,recent_length=128,residual_length=128,num_k_base=32,num_v_base=32,patternkv_cache_mode='segmented_rolling',patternkv_value_objective='base',patternkv_v_precision_selector='causal_v4',patternkv_v4_budget_fraction=0.25,patternkv_random_selector_seed=20260809,patternkv_selector_task_key='b1-mask-probe')
def config():
 c=AutoConfig.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,attn_implementation='eager')
 for k,v in CFG.items(): setattr(c,k,v)
 return c
def metrics(a,b):
 af=a.detach().float(); bf=b.detach().float(); d=af-bf; an=af.norm().clamp_min(1e-12); bn=bf.norm().clamp_min(1e-12)
 return {'shape_a':list(a.shape),'shape_b':list(b.shape),'max_abs':float(d.abs().max().item()),'mean_abs':float(d.abs().mean().item()),'rel_l2':float(d.norm().div(an).item()),'cosine':float((af.flatten()*bf.flatten()).sum().div(an*bn).item())}
orig=sysmod._compressed_attention
capture={}
def wrapped(module, query_states, cache, attention_mask):
 out, probs = orig(module, query_states, cache, attention_mask)
 if int(module.layer_idx)==0 and 'query' not in capture:
  capture['module']=module; capture['query']=query_states.detach(); capture['cache']=cache; capture['mask']=attention_mask.detach().clone() if torch.is_tensor(attention_mask) else None; capture['out_actual']=out.detach(); capture['probs_actual']=probs.detach()
 return out, probs
sysmod._compressed_attention=wrapped
def summarize_mask(m):
 if m is None: return {'present':False}
 mf=m.detach().float(); return {'present':True,'shape':list(m.shape),'dtype':str(m.dtype),'min':float(mf.min().item()),'max':float(mf.max().item()),'zero_count':int((mf==0).sum().item()),'neg_inf_like_count':int((mf < -1e20).sum().item()),'tail_values':mf.flatten()[-8:].tolist()}
def main():
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
 ids=tok(BASE*160,return_tensors='pt',add_special_tokens=False).input_ids[:,:CONTEXT].to('cuda:0')
 m=Qwen3ForCausalLM_PatternKVCompressed.from_pretrained(MODEL,local_files_only=True,config=config(),torch_dtype=torch.float16,low_cpu_mem_usage=True).to('cuda:0').eval()
 with torch.no_grad():
  pre=m(input_ids=ids,use_cache=True,return_dict=True)
  nxt=pre.logits[:,-1,:].argmax(dim=-1)
  dec=m(input_ids=nxt.view(1,1),past_key_values=pre.past_key_values,use_cache=True,return_dict=True)
  torch.cuda.synchronize()
  out_none, probs_none = orig(capture['module'], capture['query'], capture['cache'], None)
  out_mask, probs_mask = orig(capture['module'], capture['query'], capture['cache'], capture['mask'])
 payload={'status':'DONE','decode_token':int(nxt.item()),'mask':summarize_mask(capture.get('mask')),'out_actual_vs_none':metrics(capture['out_actual'],out_none),'out_actual_vs_mask_replay':metrics(capture['out_actual'],out_mask),'probs_actual_vs_none':metrics(capture['probs_actual'],probs_none),'probs_actual_vs_mask_replay':metrics(capture['probs_actual'],probs_mask)}
 OUT.mkdir(parents=True,exist_ok=True)
 (OUT/'first_decode_mask_probe_v1.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
 (OUT/'first_decode_mask_probe_v1.md').write_text('# first_decode_mask_probe_v1\n\n```json\n'+json.dumps(payload,indent=2,sort_keys=True)+'\n```\n')
 print(json.dumps(payload,indent=2,sort_keys=True))
if __name__=='__main__': main()
