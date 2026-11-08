"""Figures 1-3 of the paper."""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
out = sys.argv[1]; days = [1, 2, 3, 4, 5]
C = {'claude': '#c0763a', 'gpt': '#2f6f8f', 'mixed': '#6b5aa6', 'good': '#2f6f8f', 'bad': '#b3263a', 'grey': '#8a93a3'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e3e6eb', 'axes.axisbelow': True})
i = pd.read_csv('indep.csv'); t = i[i.day <= 5].groupby(['fam', 'day']).neff.mean().unstack()
f, ax = plt.subplots(figsize=(6.2, 2.6))
for k, lab in [('claude', 'Claude towns'), ('gpt', 'GPT towns'), ('mixed', 'Mixed Claude + GPT towns')]:
    ax.plot(days, t.loc[k], marker='o', color=C[k], label=lab, lw=2)
ax.set_xticks(days, [f'Day {d}' for d in days]); ax.set_ylabel('Independent forecasters'); ax.set_ylim(1, 2.4); ax.legend(frameon=False); f.tight_layout(); f.savefig(f'{out}/fig1.png', dpi=220)
a = pd.read_csv('angle1.csv'); x = a[a.ex_top & a.executed.notna()].copy(); x['right'] = x.ex_evil.astype(str) == 'True'
x['band'] = pd.cut(x.agree, [0, .75, .9, 1.01], labels=['Under 75%', '75 to 90%', '90% and up']); bb = x.groupby('band', observed=True).right.agg(['sum', 'size'])
f, ax = plt.subplots(figsize=(6.2, 2.2))
for j, (lab, row) in enumerate(bb.iterrows()):
    r, w = row['sum'], row['size'] - row['sum']
    ax.barh(j, r, color=C['good'], label='Executed an evil player' if j == 0 else None); ax.barh(j, w, left=r, color=C['bad'], label='Executed a good player' if j == 0 else None)
    ax.text(r + w + 6, j, f'{round(100 * w / (r + w))}% wrong', va='center')
ax.set_yticks(range(3), bb.index); ax.invert_yaxis(); ax.set_xlim(0, 420); ax.set_xlabel("Executions of the town's top suspect"); ax.set_ylabel('Town agreement')
ax.legend(frameon=False, loc='lower right'); f.tight_layout(); f.savefig(f'{out}/fig2.png', dpi=220)
p = pd.read_csv('angle2.csv'); g = p.groupby(['day', 'same']).r.mean().unstack()
f, ax = plt.subplots(figsize=(6.2, 2.6))
ax.plot(days, g[1].iloc[:5], marker='o', color=C['mixed'], lw=2, label='Same-model pairs'); ax.plot(days, g[0].iloc[:5], marker='o', color=C['grey'], lw=2, label='Claude-GPT pairs')
ax.set_xticks(days, [f'Day {d}' for d in days]); ax.set_ylabel('Belief agreement (r)'); ax.set_ylim(0.35, 1); ax.legend(frameon=False); f.tight_layout(); f.savefig(f'{out}/fig3.png', dpi=220)
print('figures written to', out)
