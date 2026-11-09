"""Generate the figures from saved per-question predictions; no GPU required."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
res = root / 'results' / 'v2'
figures = root / 'figures'
figures.mkdir(exist_ok=True)
models = ['base', 'sft', 'direct-opd', 'sft-opd', 'direct-opd-1024']
labels = ['Base', 'SFT', 'Direct OPD', 'SFT → OPD', 'OPD 1024']
rows = {(m, d): [json.loads(s) for s in (res / f'{m}-{d}-test.jsonl').read_text().splitlines()] for m in models for d in ('gsm8k', 'math500')}
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 180})
x = np.arange(len(models)); width = .36
fig, ax = plt.subplots(figsize=(10, 4.7), layout='constrained')
for k, (ds, color) in enumerate([('gsm8k','#3469b0'), ('math500','#e39b38')]):
    vals = [100 * sum(r['correct'] for r in rows[m,ds]) / len(rows[m,ds]) for m in models]
    bars = ax.bar(x+(k-.5)*width, vals, width, label='GSM8K' if k==0 else 'MATH-500', color=color)
    ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=9)
ax.set_xticks(x, labels); ax.set_ylim(0,70); ax.set_ylabel('Accuracy (%)'); ax.set_title('Held-out test accuracy (200 problems each)'); ax.legend(frameon=False); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
fig.savefig(figures/'test_accuracy.png'); plt.close(fig)
fig, ax = plt.subplots(figsize=(10, 4.7), layout='constrained')
for k, (ds, color) in enumerate([('gsm8k','#3469b0'), ('math500','#e39b38')]):
    vals = [100 * sum(r['truncated'] for r in rows[m,ds]) / len(rows[m,ds]) for m in models]
    bars = ax.bar(x+(k-.5)*width, vals, width, label='GSM8K' if k==0 else 'MATH-500', color=color)
    ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=9)
ax.set_xticks(x, labels); ax.set_ylim(0,60); ax.set_ylabel('No EOS by 2048 tokens (%)'); ax.set_title('Generation truncation on held-out tests'); ax.legend(frameon=False); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
fig.savefig(figures/'test_truncation.png'); plt.close(fig)
fig, ax = plt.subplots(figsize=(7.5,4.5), layout='constrained')
for m,label,color in [('direct-opd','Direct OPD','#3469b0'),('sft-opd','SFT → OPD','#d47b32')]:
    vals=[]
    for step in (50,125,250):
        r=[json.loads(s) for s in (res/f'{m}-{step}-val.jsonl').read_text().splitlines()]
        vals.append(100*sum(z['correct'] for z in r)/len(r))
    ax.plot([50,125,250],vals,'o-',label=label,color=color)
    for step,v in zip((50,125,250),vals): ax.annotate(f'{v:.0f}%',(step,v),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
ax.set_xticks([50,125,250]); ax.set_ylim(45,72); ax.set_xlabel('Training steps'); ax.set_ylabel('GSM8K validation accuracy (%)'); ax.set_title('Checkpoint selection (100 validation problems)'); ax.legend(frameon=False); ax.grid(alpha=.2)
fig.savefig(figures/'validation_checkpoints.png'); plt.close(fig)
