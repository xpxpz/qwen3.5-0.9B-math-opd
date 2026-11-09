"""Fixed-protocol batched evaluator for official and LoRA-adapted students."""
import argparse, json, os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import PeftModel
from math_verify import parse, verify
STUDENT=os.environ.get('STUDENT_MODEL','Qwen/Qwen3.5-0.8B'); REV='2fc06364715b967f1860aea9cf38778875588b17'
def main():
    p=argparse.ArgumentParser();p.add_argument('--adapter');p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--max-new-tokens',type=int,default=768);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('Run evaluation on the CUDA machine')
    set_seed(a.seed);tok=AutoTokenizer.from_pretrained(STUDENT,revision=REV,padding_side='left')
    model=AutoModelForCausalLM.from_pretrained(STUDENT,revision=REV,dtype=torch.bfloat16,device_map='auto')
    if a.adapter: model=PeftModel.from_pretrained(model,a.adapter)
    model.eval()
    with Path(a.data).open(encoding='utf8') as f: rows=[json.loads(x) for x in f]
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf8') as f:
      for offset in range(0,len(rows),a.batch_size):
        batch=rows[offset:offset+a.batch_size]; texts=[]
        for row in batch:
            prompt='Solve the following math problem step by step. Put the final answer in \\boxed{}.\n\n'+row['problem']
            texts.append(tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False))
        inp=tok(texts,return_tensors='pt',padding=True).to(model.device)
        with torch.inference_mode():
            seq=model.generate(**inp,max_new_tokens=a.max_new_tokens,do_sample=True,temperature=0.6,top_p=0.95,top_k=20,pad_token_id=tok.pad_token_id)
        for row,all_ids in zip(batch,seq):
            ids=all_ids[inp['input_ids'].shape[1]:].tolist(); eos=tok.eos_token_id
            ended=eos in ids
            if ended: ids=ids[:ids.index(eos)+1]
            response=tok.decode(ids,skip_special_tokens=True)
            try:
                gold=parse('\\boxed{'+row['answer']+'}'); final=response.rsplit('</think>',1)[-1]; pred=parse(final) if r'\boxed{' in final else []
                ok=bool(gold and pred and verify(gold,pred)); error=None
            except Exception as e: gold=pred=[];ok=False;error=str(e)
            result={'id':row['id'],'answer':row['answer'],'response':response,'correct':ok,'gold_parsed':bool(gold),'pred_parsed':bool(pred),'error':error,'tokens':len(ids),'truncated':not ended}
            f.write(json.dumps(result,ensure_ascii=False)+'\n')
        f.flush();print('completed',min(offset+len(batch),len(rows)),'/',len(rows),flush=True)
    with out.open(encoding='utf8') as f: results=[json.loads(x) for x in f]
    print(json.dumps({'n':len(results),'accuracy':sum(x['correct'] for x in results)/len(results),'unparsed':sum(not x['pred_parsed'] for x in results),'truncated':sum(x['truncated'] for x in results),'mean_tokens':sum(x['tokens'] for x in results)/len(results)},indent=2))
if __name__=='__main__': main()
