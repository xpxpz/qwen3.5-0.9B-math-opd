"""Generate and validate Qwen teacher solutions on training questions only."""
import argparse,json,os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
from math_verify import parse,verify
p=argparse.ArgumentParser();p.add_argument('--input',default='data/v2/train.jsonl');p.add_argument('--output',default='data/v2/teacher_sft.jsonl');p.add_argument('--raw',default='data/v2/teacher_raw.jsonl');p.add_argument('--batch-size',type=int,default=8);p.add_argument('--max-new-tokens',type=int,default=1024);a=p.parse_args()
model_path=os.environ['TEACHER_MODEL'];tok=AutoTokenizer.from_pretrained(model_path,padding_side='left');m=AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.bfloat16,device_map='auto');m.eval();set_seed(42)
with open(a.input,encoding='utf8') as f:rows=[json.loads(x) for x in f]
Path(a.output).parent.mkdir(parents=True,exist_ok=True); accepted=0
with open(a.output,'w',encoding='utf8') as out,open(a.raw,'w',encoding='utf8') as raw:
 for i in range(0,len(rows),a.batch_size):
  batch=rows[i:i+a.batch_size]; prompts=[tok.apply_chat_template([{'role':'user','content':'Solve the following math problem step by step. Put the final answer in \\boxed{}.\n\n'+r['problem']}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for r in batch]
  inp=tok(prompts,return_tensors='pt',padding=True).to(m.device)
  with torch.inference_mode(): seq=m.generate(**inp,max_new_tokens=a.max_new_tokens,do_sample=True,temperature=0.6,top_p=0.95,top_k=20,pad_token_id=tok.pad_token_id)
  for r,s in zip(batch,seq):
   ids=s[inp['input_ids'].shape[1]:].tolist();ended=tok.eos_token_id in ids
   if ended:ids=ids[:ids.index(tok.eos_token_id)+1]
   text=tok.decode(ids,skip_special_tokens=True);final=text.rsplit('</think>',1)[-1]
   try:
    g=parse('\\boxed{'+r['answer']+'}');pred=parse(final) if r'\boxed{' in final else [];ok=bool(ended and g and pred and verify(g,pred))
   except Exception:ok=False
   raw.write(json.dumps({'id':r['id'],'response':text,'correct':ok,'ended':ended,'tokens':len(ids)},ensure_ascii=False)+'\n')
   if ok:
    item=dict(r);item['solution']=final;item['teacher_generated']=True;out.write(json.dumps(item,ensure_ascii=False)+'\n');accepted+=1
  out.flush();raw.flush();print(f'processed {min(i+len(batch),len(rows))}/{len(rows)} accepted {accepted}',flush=True)
