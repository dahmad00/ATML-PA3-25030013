import argparse,yaml,torch,csv
from tqdm import tqdm
from config import OUT
from evaluate import get_model
from utils import save_json

def parse_prompts(path):
 data=yaml.safe_load(open(path))
 results=[]
 def walk(x,keys=()):
  if isinstance(x,str):results.append({'path':list(keys),'prompt':x})
  elif isinstance(x,list):
   for i,v in enumerate(x):walk(v,keys+(i,))
  elif isinstance(x,dict):
   for k,v in x.items():walk(v,keys+(k,))
 walk(data)
 return results

def run(method,stage,file):
 prompts=parse_prompts(file)
 model,tok=get_model(method,stage)
 results=[]
 for i,item in enumerate(tqdm(prompts,desc='Qualitative')):
  ids=tok.apply_chat_template([{'role':'user','content':item['prompt']}],return_tensors='pt',add_generation_prompt=True,enable_thinking=False).to('cuda')
  with torch.inference_mode():out=model.generate(ids,max_new_tokens=512,do_sample=False,pad_token_id=tok.pad_token_id)
  generated=out[0,ids.shape[1]:].tolist()
  results.append({'id':i,**item,'response':tok.decode(generated,skip_special_tokens=True),'tokens':len(generated),'truncated':len(generated)==512})
 save_json(OUT/'qualitative'/f'{method}_{stage}.json',results)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('method');p.add_argument('--stage',default='final');p.add_argument('--file',default='first_plot_questions.yaml');a=p.parse_args();run(a.method,a.stage,a.file)
