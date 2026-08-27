from __future__ import annotations
import json, os, sys, gc, time, traceback
from pathlib import Path
from typing import Any
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
VENDOR=os.environ.get('QWEN3_TRANSFORMERS_VENDOR')
if VENDOR and VENDOR not in sys.path: sys.path.insert(0,VENDOR)
from transformers import AutoConfig, AutoTokenizer
from models.qwen3_patternkv_system import Qwen3ForCausalLM_PatternKVCompressed, get_qwen3_compressed_counters, reset_qwen3_compressed_counters, collect_qwen3_compressed_dynamic_stats
OUT=ROOT/'reports/qwen3_v100_system_generalization_v1'; MODEL='/home/qinch2023/modelscope_models/Qwen3-8B'; CONTEXT=512; DECODE=int(os.environ.get('QWEN3_TRUE_BATCH_DECODE','8'))
BASES=['Mathematics proof sketch: define a sequence and reason about modular arithmetic carefully. ','System benchmark prompt: explain cache compression, attention, and deterministic greedy decoding. ','Long context QA: a researcher compares two inference backends and records every generated token. ','Hardware note: compare vectorized batch execution with serial request dispatch in a deterministic backend. ']
CFG=dict(k_bits=2,v_bits=2,group_size=128,sink_length=16,recent_length=128,residual_length=128,num_k_base=32,num_v_base=32,patternkv_cache_mode='segmented_rolling',patternkv_value_objective='base',patternkv_v_precision_selector='causal_v4',patternkv_v4_budget_fraction=0.25,patternkv_random_selector_seed=20260809,patternkv_selector_task_key='true-batch-reclosure')
def config():
 c=AutoConfig.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,attn_implementation='eager')
 for k,v in CFG.items(): setattr(c,k,v)
 return c
def write(name,payload):
 OUT.mkdir(parents=True,exist_ok=True); (OUT/(name+'.json')).write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); (OUT/(name+'.md')).write_text('# '+name+'\n\n```json\n'+json.dumps(payload,indent=2,sort_keys=True)+'\n```\n')
def prompts(tok,n): return torch.cat([tok(BASES[i]*160,return_tensors='pt',add_special_tokens=False).input_ids[:,:CONTEXT] for i in range(n)],dim=0)
def run_batch(n:int):
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
 if tok.pad_token_id is None: tok.pad_token_id=tok.eos_token_id
 ids=prompts(tok,n).to('cuda:0')
 model=Qwen3ForCausalLM_PatternKVCompressed.from_pretrained(MODEL,local_files_only=True,config=config(),torch_dtype=torch.float16,low_cpu_mem_usage=True).to('cuda:0').eval()
 reset_qwen3_compressed_counters(); tokens=[]; nan_inf=False
 try:
  with torch.no_grad():
   out=model(input_ids=ids,use_cache=True,return_dict=True)
   for _ in range(DECODE):
    logits=out.logits[:,-1,:].detach().float(); nan_inf = nan_inf or (not bool(torch.isfinite(logits).all()))
    nxt=logits.argmax(dim=-1); tokens.append([int(x) for x in nxt.detach().cpu().tolist()])
    out=model(input_ids=nxt.view(n,1),past_key_values=out.past_key_values,use_cache=True,return_dict=True)
   torch.cuda.synchronize()
  stats=collect_qwen3_compressed_dynamic_stats(model,out.past_key_values)
  result={'status':'PASS','batch':n,'decode':DECODE,'tokens':tokens,'no_nan_inf':not nan_inf,'counters':get_qwen3_compressed_counters(),'stats0':stats.get('cache_segment_stats_per_layer',[{}])[0]}
 except Exception as e:
  result={'status':'FAIL','batch':n,'decode':DECODE,'error':repr(e),'traceback_tail':traceback.format_exc().splitlines()[-8:],'counters':get_qwen3_compressed_counters()}
 del model; torch.cuda.empty_cache(); gc.collect(); time.sleep(2)
 return result
def main():
 independent=[]
 for _i in range(2):
  independent.append(run_batch(1))
 b2=run_batch(2)
 payload={'status':'PASS' if b2.get('status')=='PASS' else 'FAIL','decode':DECODE,'independent_b1':independent,'b2':b2,'serial_request_forward_dispatches':b2.get('counters',{}).get('serial_request_forward_dispatches'),'serial_attention_dispatches':b2.get('counters',{}).get('serial_attention_dispatches'),'classification':'TRUE_BATCH_B2_PASS' if b2.get('status')=='PASS' else 'TRUE_BATCH_B2_RUNTIME_FAIL'}
 write('true_batch_b2_reclosure',payload)
 if payload['status']!='PASS':
  write('true_batch_b4_reclosure',{'status':'NOT_RUN','reason':'B2 true-batch failed'})
  write('timed_window_reclosure',{'status':'NOT_RUN','reason':'B2 true-batch failed'})
 print(json.dumps(payload,indent=2,sort_keys=True)); return 0 if payload['status']=='PASS' else 2
if __name__=='__main__': raise SystemExit(main())
