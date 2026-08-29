#!/usr/bin/env python
from __future__ import annotations

import argparse, gzip, hashlib, json, os, random, sys, time, traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / 'vendor/transformers_4_51_runtime'
if str(VENDOR) not in sys.path: sys.path.insert(0, str(VENDOR))
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from bench.aime_answer_parser import normalize_aime_answer, parse_aime_answer
from models.qwen3_patternkv import Qwen3ForCausalLM_PatternKV, collect_qwen3_patternkv_dynamic_stats
from models.qwen3_kivi import Qwen3ForCausalLM_KIVI, collect_qwen3_kivi_dynamic_stats

EXP='qwen3_8b_aime24_native_generalization_v1'
MODEL_PATH=Path('/home/qinch2023/modelscope_models/Qwen3-8B')
DATASET=ROOT/'datasets/aime/aime24.jsonl'
RESULT_ROOT=ROOT/'results'/EXP/'formal'
RUN_ROOT=ROOT/'run'/EXP
REPORT_ROOT=ROOT/'reports'/EXP
SEEDS=(42,43,44)
METHODS={
 'FP16': {},
 'PATTERN_BASE': {'patternkv_v_precision_selector':'base_v2','patternkv_v4_budget_fraction':0.0},
 'CAUSAL_V4_25': {'patternkv_v_precision_selector':'causal_v4','patternkv_v4_budget_fraction':0.25},
 'KIVI_PAPER_G128': {'k_bits':2,'v_bits':2,'group_size':128,'sink_length':0,'recent_length':128,'residual_length':128,'kivi_cache_mode':'segmented_rolling'},
}
BASE_CFG=dict(k_bits=2,v_bits=2,group_size=128,sink_length=16,recent_length=128,residual_length=128,num_k_base=32,num_v_base=32,patternkv_cache_mode='segmented_rolling',patternkv_value_objective='base',patternkv_random_selector_seed=20260809)
GEN_CFG=dict(do_sample=True,temperature=0.6,top_p=0.95,max_new_tokens=32768,repetition_penalty=1.0,num_return_sequences=1,use_cache=True)


def sha(x: bytes)->str: return hashlib.sha256(x).hexdigest()
def stable(o: Any)->str: return sha(json.dumps(o,sort_keys=True,ensure_ascii=False).encode())
def effective_seed(base:int,pid:int,sample:int=0)->int: return int(base)+int(pid)*1000+int(sample)
def task_id(method:str,pid:int,base:int)->str: return f'{method}__p{pid:02d}__seed{base}__sample0'
def rows(): return [json.loads(l) for l in DATASET.read_text().splitlines() if l.strip()]
def result_path(method:str,pid:int,base:int)->Path: return RESULT_ROOT/method/f'seed{base}'/f'p{pid:02d}.json'
def render(tok, problem:str):
    user=problem+'\n\nPlease reason step by step, and put your final answer within \\boxed{}.'
    text=tok.apply_chat_template([{'role':'user','content':user}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    if 'deepseek' in text.lower() or '<|eot_id|>' in text: raise RuntimeError('non-Qwen prompt artifact detected')
    return user,text

def build_manifest(methods=None, problem_ids=None, seeds=None):
    methods=methods or tuple(METHODS); problem_ids=problem_ids or range(30); seeds=seeds or SEEDS
    return [{'method':m,'problem_id':p,'base_seed':s,'sample_id':0,'effective_seed':effective_seed(s,p)} for m in methods for p in problem_ids for s in seeds]

def configure(method:str, taskkey:str):
    cfg=AutoConfig.from_pretrained(str(MODEL_PATH), local_files_only=True, trust_remote_code=False, attn_implementation='eager')
    if cfg.model_type!='qwen3': raise RuntimeError(f'not qwen3: {cfg.model_type}')
    if method != 'FP16':
        for k,v in BASE_CFG.items(): setattr(cfg,k,v)
        for k,v in METHODS[method].items(): setattr(cfg,k,v)
        if method != 'KIVI_PAPER_G128':
            setattr(cfg,'patternkv_selector_task_key',taskkey)
    return cfg

def load_model(method:str, taskkey:str):
    cfg=configure(method, taskkey)
    if method == 'FP16':
        cls = AutoModelForCausalLM
    elif method == 'KIVI_PAPER_G128':
        cls = Qwen3ForCausalLM_KIVI
    else:
        cls = Qwen3ForCausalLM_PatternKV
    model=cls.from_pretrained(str(MODEL_PATH), local_files_only=True, trust_remote_code=False, config=cfg, torch_dtype=torch.float16, low_cpu_mem_usage=True).to('cuda:0')
    model.eval(); return model

def run_one(method:str, row:dict, base_seed:int, max_new_tokens:int|None=None, phase='formal'):
    pid=int(row['problem_id']); seed=effective_seed(base_seed,pid); random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    tok=AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None: tok.pad_token_id=tok.eos_token_id
    tid=task_id(method,pid,base_seed); model=load_model(method, tid)
    user,prompt=render(tok,row['problem']); enc=tok(prompt,return_tensors='pt',add_special_tokens=False).to('cuda:0')
    gen=dict(GEN_CFG);
    if max_new_tokens is not None: gen['max_new_tokens']=max_new_tokens
    t0=time.perf_counter(); err=None; out=None
    try:
        with torch.no_grad(): out=model.generate(**enc, **gen, return_dict_in_generate=True, output_scores=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        torch.cuda.synchronize(); seq=out.sequences; gids=seq[0,enc.input_ids.shape[1]:].detach().cpu().tolist(); text=tok.decode(gids,skip_special_tokens=True)
        parsed=parse_aime_answer(text); ref=normalize_aime_answer(row['answer']); correct=parsed['parsed_answer']==ref; stop='length' if len(gids)>=gen['max_new_tokens'] else 'eos'
        if method == 'FP16':
            stats = {}
            method_cfg = {}
        elif method == 'KIVI_PAPER_G128':
            stats = collect_qwen3_kivi_dynamic_stats(model, out.past_key_values)
            method_cfg = {**BASE_CFG, **METHODS[method]}
        else:
            stats = collect_qwen3_patternkv_dynamic_stats(model, out.past_key_values)
            method_cfg = {**BASE_CFG, **METHODS[method]}
        rec={'experiment_id':EXP,'phase':phase,'dataset':'aime24','dataset_sha256':sha(DATASET.read_bytes()),'model_path':str(MODEL_PATH),'model_name':MODEL_PATH.name,'model_type':'qwen3','model_architecture':'Qwen3ForCausalLM','backend_class':model.__class__.__name__,'attention_class':model.model.layers[0].self_attn.__class__.__name__,'method':method,'display_method':method,'method_config':method_cfg,'method_config_hash':stable(method_cfg),'problem_id':pid,'base_seed':base_seed,'sample_id':0,'effective_seed':seed,'task_key':tid,'prompt_protocol':'qwen3_native_thinking_v1','rendered_prompt':prompt,'prompt_hash':stable({'prompt':prompt}),'input_token_hash':stable(enc.input_ids.detach().cpu().tolist()),'generation_config':gen,'generation_config_hash':stable(gen),'generated_text':text,'generated_token_hash':stable(gids),'generated_tokens':len(gids),'parsed_answer':parsed['parsed_answer'],'reference_answer':ref,'is_correct':correct,'parser_strategy':parsed['parser_strategy'],'parser_error':parsed['parser_error'],'stop_reason':stop,'wall_time_seconds':round(time.perf_counter()-t0,4),'gpu_physical_id':os.environ.get('CUDA_VISIBLE_DEVICES'),'git_commit':os.popen('git rev-parse HEAD').read().strip(),'timestamp':time.strftime('%Y-%m-%d %H:%M:%S %z'),'cache_statistics':stats}
    except Exception as e:
        rec={'experiment_id':EXP,'phase':phase,'method':method,'problem_id':pid,'base_seed':base_seed,'task_key':tid,'runtime_error':repr(e),'traceback':traceback.format_exc(),'timestamp':time.strftime('%Y-%m-%d %H:%M:%S %z')}
    del model; torch.cuda.empty_cache()
    return rec

def write_atomic(path:Path, rec:dict):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(rec,ensure_ascii=False,sort_keys=True)+'\n'); tmp.replace(path)

def worker(args):
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if gpu not in {'0','1','2','3','4','5','6','7'}: raise SystemExit(f'refuse GPU {gpu}')
    RUN_ROOT.mkdir(parents=True,exist_ok=True); (RUN_ROOT/'claims').mkdir(exist_ok=True); (RUN_ROOT/'logs').mkdir(exist_ok=True)
    rs={int(r['problem_id']):r for r in rows()}; manifest=build_manifest(methods=args.methods.split(','))
    for item in manifest:
        path=result_path(item['method'],item['problem_id'],item['base_seed'])
        if path.exists(): continue
        claim=RUN_ROOT/'claims'/(task_id(item['method'],item['problem_id'],item['base_seed'])+'.claim')
        try:
            fd=os.open(claim, os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.write(fd, f"pid={os.getpid()} gpu={gpu}\n".encode()); os.close(fd)
        except FileExistsError: continue
        rec=run_one(item['method'], rs[item['problem_id']], item['base_seed'])
        write_atomic(path, rec)
        print(json.dumps({'event':'wrote','task_key':item,'correct':rec.get('is_correct'),'tokens':rec.get('generated_tokens'),'error':rec.get('runtime_error')},ensure_ascii=False), flush=True)

def status():
    payload={}
    total=0
    for m in METHODS:
        done=correct=err=0
        for p in range(30):
            for s in SEEDS:
                fp=result_path(m,p,s)
                if fp.exists():
                    done+=1; rec=json.loads(fp.read_text()); correct+=int(bool(rec.get('is_correct'))); err+=int('runtime_error' in rec)
        payload[m]={'done':done,'expected':90,'correct':correct,'errors':err,'accuracy':correct/done if done else None}; total+=done
    payload['TOTAL']={'done':total,'expected':90*len(METHODS)}
    print(json.dumps(payload,indent=2,sort_keys=True))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--worker',action='store_true'); ap.add_argument('--methods',default='PATTERN_BASE,CAUSAL_V4_25'); ap.add_argument('--status',action='store_true')
    a=ap.parse_args();
    if a.status: return status()
    if a.worker: return worker(a)
    print(json.dumps({'manifest_count':len(build_manifest())}))
if __name__=='__main__': main()
