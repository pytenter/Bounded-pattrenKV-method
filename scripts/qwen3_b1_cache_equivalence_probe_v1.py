from __future__ import annotations
import gc, json, os, sys, time
from pathlib import Path
from typing import Any
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
VENDOR=os.environ.get('QWEN3_TRANSFORMERS_VENDOR')
if VENDOR and VENDOR not in sys.path: sys.path.insert(0,VENDOR)
from transformers import AutoConfig, AutoTokenizer
from models.qwen3_patternkv import Qwen3ForCausalLM_PatternKV
from models.qwen3_patternkv_system import Qwen3ForCausalLM_PatternKVCompressed
from models.segmented_cache import cache_segment_stats, reconstruct_full_k, reconstruct_full_v
OUT=ROOT/'reports/qwen3_v100_system_generalization_v1'
MODEL='/home/qinch2023/modelscope_models/Qwen3-8B'
CONTEXT=512
BASE='Mathematics proof sketch: define a sequence and reason about modular arithmetic carefully. '
CFG=dict(k_bits=2,v_bits=2,group_size=128,sink_length=16,recent_length=128,residual_length=128,num_k_base=32,num_v_base=32,patternkv_cache_mode='segmented_rolling',patternkv_value_objective='base',patternkv_v_precision_selector='causal_v4',patternkv_v4_budget_fraction=0.25,patternkv_random_selector_seed=20260809,patternkv_selector_task_key='b1-cache-equivalence')
def config():
 c=AutoConfig.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,attn_implementation='eager')
 for k,v in CFG.items(): setattr(c,k,v)
 return c
def metrics(a,b):
 if a is None or b is None: return {'present':False}
 af=a.detach().float(); bf=b.detach().float(); d=af-bf; an=af.norm().clamp_min(1e-12); bn=bf.norm().clamp_min(1e-12)
 return {'present':True,'shape_a':list(a.shape),'shape_b':list(b.shape),'dtype_a':str(a.dtype),'dtype_b':str(b.dtype),'max_abs':float(d.abs().max().item()) if d.numel() else 0.0,'mean_abs':float(d.abs().mean().item()) if d.numel() else 0.0,'rel_l2':float(d.norm().div(an).item()) if d.numel() else 0.0,'cosine':float((af.flatten()*bf.flatten()).sum().div(an*bn).item()) if d.numel() else 1.0}
def summarize(cache):
 return {'stats':cache_segment_stats(cache),'k_centroids_shape':list(cache.k_centroids.shape) if torch.is_tensor(cache.k_centroids) else None,'v_centroids_shape':list(cache.v_centroids.shape) if torch.is_tensor(cache.v_centroids) else None,'v_precision_true':int(cache.v_precision_mask.bool().sum().item()) if torch.is_tensor(cache.v_precision_mask) else None,'v_precision_shape':list(cache.v_precision_mask.shape) if torch.is_tensor(cache.v_precision_mask) else None,'k_assign_shape':list(cache.k_assignments.shape) if torch.is_tensor(cache.k_assignments) else None,'v_idx_shape':list(cache.v_assignment_idx.shape) if torch.is_tensor(cache.v_assignment_idx) else None,'v_mask_shape':list(cache.v_pattern_mask.shape) if torch.is_tensor(cache.v_pattern_mask) else None}
def run(cls):
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
 ids=tok(BASE*160,return_tensors='pt',add_special_tokens=False).input_ids[:,:CONTEXT].to('cuda:0')
 m=cls.from_pretrained(MODEL,local_files_only=True,config=config(),torch_dtype=torch.float16,low_cpu_mem_usage=True).to('cuda:0').eval()
 with torch.no_grad():
  pre=m(input_ids=ids,use_cache=True,return_dict=True)
  nxt=pre.logits[:,-1,:].argmax(dim=-1)
  dec=m(input_ids=nxt.view(1,1),past_key_values=pre.past_key_values,use_cache=True,return_dict=True)
  torch.cuda.synchronize()
  c=dec.past_key_values.layer_caches[0]
  out={'token':int(nxt.item()),'logits':dec.logits[:,-1,:].detach().cpu(),'summary':summarize(c),'full_k':reconstruct_full_k(c).detach().cpu(),'full_v':reconstruct_full_v(c).detach().cpu(),'v_precision_mask':c.v_precision_mask.detach().cpu() if torch.is_tensor(c.v_precision_mask) else None,'k_assignments':c.k_assignments.detach().cpu() if torch.is_tensor(c.k_assignments) else None,'v_assignment_idx':c.v_assignment_idx.detach().cpu() if torch.is_tensor(c.v_assignment_idx) else None,'v_pattern_mask':c.v_pattern_mask.detach().cpu() if torch.is_tensor(c.v_pattern_mask) else None}
 del m; torch.cuda.empty_cache(); gc.collect(); time.sleep(2)
 return out
def main():
 ref=run(Qwen3ForCausalLM_PatternKV)
 comp=run(Qwen3ForCausalLM_PatternKVCompressed)
 payload={'status':'DONE','context':CONTEXT,'reference_token':ref['token'],'compressed_token':comp['token'],'logits':metrics(ref['logits'],comp['logits']),'reference_summary':ref['summary'],'compressed_summary':comp['summary'],'full_k':metrics(ref['full_k'],comp['full_k']),'full_v':metrics(ref['full_v'],comp['full_v']),'v_precision_mask_equal':bool(torch.equal(ref['v_precision_mask'],comp['v_precision_mask'])) if ref['v_precision_mask'] is not None and comp['v_precision_mask'] is not None else None,'k_assignments_equal':bool(torch.equal(ref['k_assignments'],comp['k_assignments'])) if ref['k_assignments'] is not None and comp['k_assignments'] is not None else None,'v_assignment_idx_equal':bool(torch.equal(ref['v_assignment_idx'],comp['v_assignment_idx'])) if ref['v_assignment_idx'] is not None and comp['v_assignment_idx'] is not None else None,'v_pattern_mask_equal':bool(torch.equal(ref['v_pattern_mask'],comp['v_pattern_mask'])) if ref['v_pattern_mask'] is not None and comp['v_pattern_mask'] is not None else None}
 OUT.mkdir(parents=True,exist_ok=True)
 for name in ['first_decode_cache_equivalence_probe_v1']:
  (OUT/(name+'.json')).write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
  (OUT/(name+'.md')).write_text('# '+name+'\n\n```json\n'+json.dumps(payload,indent=2,sort_keys=True)+'\n```\n')
 print(json.dumps(payload,indent=2,sort_keys=True))
 return 0
if __name__=='__main__': raise SystemExit(main())
