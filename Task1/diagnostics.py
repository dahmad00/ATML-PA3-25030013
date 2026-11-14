import argparse,torch,numpy as np
from tqdm import tqdm
from datasets import load_dataset
from config import OUT,SEED
from utils import load_model,save_json
from evaluate import get_model
from models import delta_weight

def run(method,stage):
    base,tok=load_model();model,_=get_model(method,stage,merge=False)
    gsm=load_dataset('openai/gsm8k','main',split='test');hs=load_dataset('Rowan/hellaswag',split='validation')
    import json
    ids=json.loads((OUT/'probe_ids.json').read_text())
    probes=[('math',i,gsm[i]['question']) for i in ids['gsm8k']]+[('nonmath',i,hs[i]['ctx']) for i in ids['hellaswag']]
    rows=[];displacement=0.
    if method in ('full','l2sp'):
        for p,q in zip(model.parameters(),base.parameters()):
            displacement+=(p.detach().float()-q.detach().float()).square().sum().item()
        save_json(OUT/'diagnostics'/f'{method}_{stage}_displacement.json',{'l2_norm':displacement**.5})
    for layer in tqdm([0,13,27],desc='Diagnostic layers'):
        m0=base.model.layers[layer].mlp.down_proj
        w0=m0.weight.detach().float()
        delta=(model.model.layers[layer].mlp.down_proj.weight.detach().float()-w0) if method in ('full','l2sp') else delta_weight(model,method,layer)
        u,_,_=torch.linalg.svd(w0,full_matrices=False);uk=u[:,:128]
        rho=((uk.T@delta).square().sum()/delta.square().sum().clamp_min(1e-20)).item()
        eta=(delta.norm()/w0.norm()).item()
        cached=[]
        def hook(_,inputs):
            cached.append(inputs[0][:,-1].detach().float())
        h=m0.register_forward_pre_hook(hook)
        for kind,i,question in tqdm(probes,desc=f'Layer {layer}',leave=False):
            if kind=='math':
                prompt=tok.apply_chat_template([{'role':'user','content':question}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
            else:
                prompt=tok.encode(question,add_special_tokens=False)
            x=torch.tensor([prompt],device='cuda')
            with torch.inference_mode():
                base(input_ids=x)
            z=cached.pop().flatten();ratio=((delta@z).norm()/((w0@z).norm()+1e-8)).item()
            rows.append({'layer':layer,'kind':kind,'example_id':i,'activation_ratio':ratio,'rho':rho,'eta':eta})
        h.remove()
    save_json(OUT/'diagnostics'/f'{method}_{stage}.json',rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('method');p.add_argument('--stage',default='final');a=p.parse_args();run(a.method,a.stage)
