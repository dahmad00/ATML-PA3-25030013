import re
from datasets import load_dataset,Dataset
from tqdm import tqdm
from config import MODEL,OUT,SEED
from utils import save_json
from transformers import AutoTokenizer
import numpy as np

def norm(s):return re.sub(r'\s+',' ',str(s).strip().lower())
def prepare():
 tok=AutoTokenizer.from_pretrained(MODEL)
 gsm=load_dataset('openai/gsm8k','main',split='test');excluded={norm(x['question']) for x in gsm}
 hs=load_dataset('Rowan/hellaswag',split='validation')
 rng=np.random.default_rng(SEED)
 probes={'gsm8k':[int(i) for i in rng.choice(len(gsm),32,replace=False)],'hellaswag':[int(i) for i in rng.choice(len(hs),32,replace=False)]}
 save_json(OUT/'probe_ids.json',probes)
 records=[]
 for i,x in enumerate(tqdm(load_dataset('meta-math/MetaMathQA',split='train',streaming=True),desc='Select 40k')):
  q=x.get('query') or '';a=x.get('response') or ''
  if not q.strip() or not a.strip() or norm(q) in excluded or norm(x.get('original_question') or '') in excluded:continue
  prompt=tok.apply_chat_template([{'role':'user','content':q}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
  full=tok.apply_chat_template([{'role':'user','content':q},{'role':'assistant','content':a}],tokenize=True,add_generation_prompt=False,enable_thinking=False)
  if len(full)>2048 or full[:len(prompt)]!=prompt:continue
  records.append({'id':i,'input_ids':full,'labels':[-100]*len(prompt)+full[len(prompt):]})
  if len(records)==40000:break
 Dataset.from_list(records).save_to_disk(str(OUT/'train_data'))
 save_json(OUT/'training_ids.json',[x['id'] for x in records])
if __name__=='__main__':prepare()
