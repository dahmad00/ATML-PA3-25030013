import json,random
from pathlib import Path
import numpy as np,torch
from config import SEED,MODEL

def seed():
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
def save_json(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,default=str))
def append_jsonl(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a') as f:f.write(json.dumps(obj,default=str)+'\n')
def rng_state():return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'numpy':np.random.get_state(),'python':random.getstate()}
def set_rng(s):
 torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda']);np.random.set_state(s['numpy']);random.setstate(s['python'])
def load_model(fp32=False):
 from transformers import AutoModelForCausalLM,AutoTokenizer
 tok=AutoTokenizer.from_pretrained(MODEL)
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float32 if fp32 else torch.float16,attn_implementation='sdpa').cuda()
 return model,tok

def unique_params(model):return list(dict.fromkeys(model.parameters()))

def summarize():
 import csv,matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from config import OUT,TASKS
 root=OUT/'analysis';root.mkdir(parents=True,exist_ok=True)
 conditions=['baseline_final','full_midpoint','full_final','l2sp_final','lora4_final','lora16_final','oplora_final']
 metrics=[]
 for cond in conditions:
  path=OUT/'evaluations'/cond/'summary.json'
  if not path.exists():continue
  data=json.loads(path.read_text())
  for task in TASKS:
   if task not in data:continue
   for key,value in data[task].items():
    if isinstance(value,(int,float)) and not isinstance(value,bool):metrics.append({'condition':cond,'task':task,'metric':key,'value':value})
 def csvout(name,rows):
  if not rows:return
  with (root/name).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 csvout('benchmark_metrics.csv',metrics)
 baseline={(r['task'],r['metric']):r['value'] for r in metrics if r['condition']=='baseline_final'}
 csvout('benchmark_deltas.csv',[{**r,'delta_percentage_points':100*(r['value']-baseline[(r['task'],r['metric'])])} for r in metrics if r['condition']!='baseline_final' and (r['task'],r['metric']) in baseline])
 training=[];resources=[]
 for method in ('full','l2sp','lora4','lora16','oplora'):
  log=OUT/method/'train.jsonl'
  if log.exists():training.extend([{'method':method,**json.loads(line)} for line in log.read_text().splitlines()])
  resource=OUT/method/'resources.json'
  if resource.exists():resources.append({'method':method,**json.loads(resource.read_text())})
 csvout('training_curves.csv',training);csvout('resources.csv',resources)
 if training:
  for method in ('full','l2sp','lora4','lora16','oplora'):
   rows=[r for r in training if r['method']==method]
   if rows:plt.plot([r['step'] for r in rows],[r['sft_loss'] for r in rows],label=method)
  plt.xlabel('Optimizer update');plt.ylabel('SFT loss');plt.legend();plt.tight_layout();plt.savefig(root/'training_curves.png',dpi=160);plt.close()
 diagnostics=[]
 for path in (OUT/'diagnostics').glob('*.json') if (OUT/'diagnostics').exists() else []:
  data=json.loads(path.read_text())
  if isinstance(data,list):diagnostics.extend([{'condition':path.stem,**r} for r in data])
 csvout('diagnostics.csv',diagnostics)
 qual=[]
 for path in (OUT/'qualitative').glob('*.json') if (OUT/'qualitative').exists() else []:
  qual.extend([{'condition':path.stem,**r} for r in json.loads(path.read_text())])
 save_json(root/'qualitative_all.json',qual)
 ids={r['id'] for r in qual};chosen={min(ids),max(ids)} if ids else set()
 csvout('annotation_template.csv',[{'condition':r['condition'],'id':r['id'],'prompt':r['prompt'],'response':r['response'],'truncated':r['truncated'],'format_compliance':'','coherence':'','repetition':'','math_framing':'','inappropriate':'','notes':''} for r in qual if r['id'] in chosen])

if __name__=='__main__':summarize()
