import time
import torch
from torch import nn
from peft import LoraConfig,get_peft_model
from config import TARGETS

class OrthogonalLinear(nn.Module):
 def __init__(self,base,r=16,k=128):
  super().__init__();self.base=base
  w=base.weight.detach().float();u,s,vh=torch.linalg.svd(w,full_matrices=False)
  k=min(k,len(s));self.register_buffer('uk',u[:,:k].contiguous());self.register_buffer('vk',vh[:k].T.contiguous())
  self.A=nn.Parameter(torch.empty(r,w.shape[1],device=w.device));self.B=nn.Parameter(torch.zeros(w.shape[0],r,device=w.device))
  nn.init.kaiming_uniform_(self.A,a=5**.5);self.dropout=nn.Dropout(.05)
 def factors(self):
  a=self.A;b=self.B
  return a-(a@self.vk)@self.vk.T,b-self.uk@(self.uk.T@b)
 def delta(self):
  a,b=self.factors();return 2*(b@a)
 def forward(self,x):
  a,b=self.factors()
  return self.base(x)+2*nn.functional.linear(nn.functional.linear(self.dropout(x),a.to(x.dtype)),b.to(x.dtype))

def make_adapter(model,method):
 model.requires_grad_(False)
 if method=='oplora':
  for layer in model.model.layers:
   for block in (layer.self_attn,layer.mlp):
    for name in TARGETS:
     if hasattr(block,name):setattr(block,name,OrthogonalLinear(getattr(block,name)))
  return model
 r=4 if method=='lora4' else 16
 return get_peft_model(model,LoraConfig(r=r,lora_alpha=2*r,lora_dropout=.05,target_modules=TARGETS,bias='none',task_type='CAUSAL_LM'))

def module_at(model,method,layer):
 root=model.base_model.model if method.startswith('lora') else model
 return root.model.layers[layer].mlp.down_proj

def delta_weight(model,method,layer):
 m=module_at(model,method,layer)
 if method=='oplora':return m.delta().detach().float()
 a=m.lora_A['default'].weight.float();b=m.lora_B['default'].weight.float()
 return (b@a*m.scaling['default']).detach()

def merge_adapter(model,method):
 if method.startswith('lora'):return model.merge_and_unload()
 if method=='oplora':
  for layer in model.model.layers:
   for block in (layer.self_attn,layer.mlp):
    for name in TARGETS:
     m=getattr(block,name,None)
     if isinstance(m,OrthogonalLinear):
      with torch.no_grad():m.base.weight.add_(m.delta().to(m.base.weight.dtype))
      setattr(block,name,m.base)
 return model
