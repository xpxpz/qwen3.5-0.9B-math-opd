"""GPU draft: validate with a short smoke run before committing compute."""
import os
import argparse, json
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig, DistillationTrainer, DistillationConfig

STUDENT=os.environ.get('STUDENT_MODEL','Qwen/Qwen3.5-0.8B')
STUDENT_REV='2fc06364715b967f1860aea9cf38778875588b17'
TEACHER=os.environ.get('TEACHER_MODEL','Qwen/Qwen3.5-4B')
TEACHER_REV='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['sft','opd']);p.add_argument('--start-adapter');p.add_argument('--train-data',default='data/train.jsonl');p.add_argument('--lr',type=float);p.add_argument('--save-steps',type=int,default=25);p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=50);p.add_argument('--max-completion',type=int,default=512);a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('Requires CUDA; run smoke test on rented GPU.')
    tok=AutoTokenizer.from_pretrained(STUDENT,revision=STUDENT_REV,padding_side='left')
    prompt=lambda r: tok.apply_chat_template([{'role':'user','content': 'Solve the following math problem step by step. Put the final answer in \\boxed{}.\n\n'+r['problem']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    rows=[json.loads(x) for x in Path(a.train_data).open(encoding='utf-8')]
    model=AutoModelForCausalLM.from_pretrained(STUDENT,revision=STUDENT_REV,dtype=torch.bfloat16)
    lora=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.0,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')
    if a.start_adapter:
        model=PeftModel.from_pretrained(model,a.start_adapter,is_trainable=True); lora=None
    if a.stage=='sft':
        if a.start_adapter: raise ValueError('SFT must start from the official model')
        items=[{'prompt':prompt(r),'completion':r['solution']+( '' if r.get('teacher_generated') else '\nFinal answer: \\boxed{'+r['answer']+'}')+tok.eos_token} for r in rows]
        items=[x for x in items if len(tok(x['prompt']+x['completion'],add_special_tokens=False)['input_ids'])<=1024]
        print('SFT examples within length budget:',len(items),flush=True)
        if not items: raise ValueError('No SFT examples fit the length budget')
        ds=Dataset.from_list(items)
        config=SFTConfig(output_dir=a.output,max_steps=a.steps,per_device_train_batch_size=1,gradient_accumulation_steps=4,learning_rate=a.lr or 2e-4,bf16=True,max_length=1024,completion_only_loss=True,logging_steps=1,save_strategy='steps',save_steps=a.save_steps,report_to='none',seed=42)
        trainer=SFTTrainer(model=model,args=config,train_dataset=ds,processing_class=tok,peft_config=lora)
    else:
        ds=Dataset.from_list([{'prompt':prompt(r)} for r in rows])
        config=DistillationConfig(output_dir=a.output,max_steps=a.steps,per_device_train_batch_size=1,gradient_accumulation_steps=4,learning_rate=a.lr or 1e-5,bf16=True,max_completion_length=a.max_completion,beta=1.0,temperature=0.7,top_p=0.95,logging_steps=1,save_strategy='steps',save_steps=a.save_steps,report_to='none',seed=42,teacher_model_init_kwargs={'revision':TEACHER_REV,'dtype':'bfloat16'})
        trainer=DistillationTrainer(model=model,teacher_model=TEACHER,args=config,train_dataset=ds,processing_class=tok,peft_config=lora)
    trainer.train();trainer.save_model(a.output);tok.save_pretrained(a.output)
if __name__=='__main__': main()
