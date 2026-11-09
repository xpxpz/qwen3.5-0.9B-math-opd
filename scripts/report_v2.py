import json,random
from pathlib import Path
R=Path('results/v2'); groups=['base','sft','direct-opd','sft-opd','direct-opd-1024']; datasets=['gsm8k','math500']; data={}
for m in groups:
 for ds in datasets:
  p=R/f'{m}-{ds}-test.jsonl'
  with p.open(encoding='utf8') as f:r=[json.loads(x) for x in f]
  if len(r)!=200:raise ValueError(f'{p}: {len(r)} rows, expected 200')
  data[m,ds]=r
for ds in datasets:
 ids=[x['id'] for x in data['base',ds]]
 for m in groups:assert ids==[x['id'] for x in data[m,ds]],(ds,m)
print('| 模型 | GSM8K 正确率 | MATH 正确率 | GSM 截断率 | MATH 截断率 | GSM 平均 token | MATH 平均 token |')
print('|---|---:|---:|---:|---:|---:|---:|')
for m in groups:
 a,b=data[m,'gsm8k'],data[m,'math500']; print('| %s | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f | %.1f |'%(m,100*sum(x['correct'] for x in a)/200,100*sum(x['correct'] for x in b)/200,100*sum(x['truncated'] for x in a)/200,100*sum(x['truncated'] for x in b)/200,sum(x['tokens'] for x in a)/200,sum(x['tokens'] for x in b)/200))
for ds in datasets:
 print('\n'+ds+' 配对差异（百分点，题目 bootstrap 95% CI）')
 for left,right in [('base','direct-opd'),('base','sft'),('sft','sft-opd'),('direct-opd','sft-opd'),('direct-opd','direct-opd-1024')]:
  d=[int(b['correct'])-int(a['correct']) for a,b in zip(data[left,ds],data[right,ds])]; rng=random.Random(42);n=len(d); bs=sorted(100*sum(rng.choices(d,k=n))/n for _ in range(5000)); print(f'{right} - {left}: {100*sum(d)/n:+.1f} [{bs[125]:+.1f},{bs[4874]:+.1f}]')
