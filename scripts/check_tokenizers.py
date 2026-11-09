from transformers import AutoTokenizer
S=('Qwen/Qwen3.5-0.8B','2fc06364715b967f1860aea9cf38778875588b17')
T=('Qwen/Qwen3.5-4B','851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')
a,b=[AutoTokenizer.from_pretrained(model,revision=rev) for model,rev in (S,T)]
assert a.get_vocab()==b.get_vocab(), 'Teacher and student vocabularies differ'
assert a.special_tokens_map==b.special_tokens_map, 'Teacher and student special tokens differ'
print('OK: vocabulary and special tokens match; size:',len(a))
print('Student thinking prompt:',repr(a.apply_chat_template([{'role':'user','content':'Solve 2+2.'}],tokenize=False,add_generation_prompt=True,enable_thinking=True)))
