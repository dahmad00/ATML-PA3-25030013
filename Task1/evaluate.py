import argparse,torch
from tqdm import tqdm
from config import OUT,TASKS
from utils import load_model,save_json
from models import make_adapter,merge_adapter

def get_model(method,stage='final',merge=True):
 model,tok=load_model(fp32=False)
 if method!='baseline':
  if method.startswith('lora') or method=='oplora':model=make_adapter(model,method)
  state=torch.load(OUT/method/'recovery.pt',map_location='cpu',weights_only=False)
  expected=625 if stage=='midpoint' else 1250
  if state['step']!=expected:raise ValueError(f'Checkpoint at update {state["step"]}; expected {expected}')
  model.load_state_dict(state['model']);del state
  if merge and (method.startswith('lora') or method=='oplora'):
   model.eval();ids=tok('The answer is',return_tensors='pt').to('cuda')
   with torch.inference_mode():original=model(**ids).logits.float()
   model=merge_adapter(model,method)
   with torch.inference_mode():merged=model(**ids).logits.float()
   save_json(OUT/'evaluations'/f'{method}_{stage}'/'merge_check.json',{'max_abs_logit_difference':(original-merged).abs().max().item()})
 model.eval();model.config.use_cache=True
 return model,tok

def run(method,stage):
 from lm_eval import evaluator
 from lm_eval.models.huggingface import HFLM
 model,tok=get_model(method,stage)
 root=OUT/'evaluations'/f'{method}_{stage}';root.mkdir(parents=True,exist_ok=True)
 lm=HFLM(pretrained=model,tokenizer=tok,batch_size=1,max_length=2048)
 summary={}
 for task in tqdm(TASKS,desc=f'Evaluate {method} {stage}'):
  kwargs={'max_gen_toks':1024,'do_sample':False} if task=='gsm8k_cot' else None
  result=evaluator.simple_evaluate(model=lm,tasks=[task],num_fewshot=0,apply_chat_template=(task=='gsm8k_cot'),gen_kwargs=kwargs,log_samples=True)
  summary[task]=result['results'][task]
  samples=result.get('samples',{}).get(task,[])
  for s in samples:
   if task=='gsm8k_cot':
    outputs=s.get('resps',[])
    text=str(outputs[0][0]) if outputs and isinstance(outputs[0],list) and outputs[0] else ''
    s['generation_text']=text;s['truncated_1024']=len(tok.encode(text,add_special_tokens=False))>=1024
  save_json(root/f'{task}_samples.json',samples)
  save_json(root/f'{task}_metrics.json',summary[task])
  save_json(root/'summary.json',summary)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('method',choices=['baseline','full','l2sp','lora4','lora16','oplora']);p.add_argument('--stage',choices=['final','midpoint'],default='final');a=p.parse_args();run(a.method,a.stage)
