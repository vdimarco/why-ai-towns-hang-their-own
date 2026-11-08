"""Robustness checks from the external audit.
1. Game-level (cluster) bootstrap: days and executions in one game are not independent, so resample whole games.
2. Tie-aware top suspect: a player whose highest probability is shared by k players gives 1/k credit to each."""
import json, gzip, glob, sys
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
def cboot(df, stat, B=2000):
    """95% interval for stat(df), resampling games with replacement."""
    games = df.game.unique(); groups = {g: s for g, s in df.groupby('game')}
    vals = []
    for _ in range(B):
        pick = rng.choice(games, len(games))
        vals.append(stat(pd.concat([groups[g] for g in pick])))
    return f'{stat(df):.3f} [{np.percentile(vals, 2.5):.3f}, {np.percentile(vals, 97.5):.3f}]'

print('== 1. Game-level bootstrap ==')
i = pd.read_csv('indep.csv')
for fam in ['claude', 'gpt', 'mixed']:
    print(' effective number', fam, {d: cboot(i[(i.fam == fam) & (i.day == d)], lambda s: s.neff.mean()) for d in (1, 5)})
a = pd.read_csv('angle1.csv'); x = a[a.ex_top & a.executed.notna()].copy(); x['wrong'] = x.ex_evil.astype(str) != 'True'
print(' wrong rate, agreement below 90%:', cboot(x[x.agree < .9], lambda s: s.wrong.mean()), 'executions', (x.agree < .9).sum())
print(' wrong rate, agreement 90%+:     ', cboot(x[x.agree >= .9], lambda s: s.wrong.mean()))
s = x[x.nominator_evil.notna()].copy(); s['ev'] = s.nominator_evil.astype(str) == 'True'
print(' nominated by evil, wrong:', cboot(s[s.wrong], lambda q: q.ev.mean()), ' right:', cboot(s[~s.wrong], lambda q: q.ev.mean()))
p = pd.read_csv('angle2.csv')
print(' lineage gap (pairs pooled, games resampled):', cboot(p, lambda q: q[q.same == 1].r.mean() - q[q.same == 0].r.mean()))

print('\n== 2. Tie-aware top suspect ==')
rows = []; tied_players = tot_players = 0
for f in sorted(glob.glob(sys.argv[1] + '/*.json.gz')):
    g = json.loads(gzip.open(f).read()); seats = {q['n']: q for q in g['seats']}
    evil = {n for n, q in seats.items() if q['e']}; imp = next((n for n, q in seats.items() if q['c'] == 'Imp'), None)
    imp_dead, ex = None, {}
    for day in g['days_']:
        for sec in day['secs']:
            for it in sec['it']:
                if it['k'] == 'death' and it['p'] == imp and imp_dead is None: imp_dead = day['d']
                if it['k'] == 'ex' and it.get('died'): ex[day['d']] = it['p']
    for b in g['belief']:
        d = b['d']; by = b['by']
        if imp_dead is not None and d > imp_dead: continue
        T = sorted({t for q in by for t in by[q]})
        if imp not in T or len(by) < 3 or d not in ex: continue
        V = {}
        for q in by:
            v = np.array([by[q].get(t, np.nan) for t in T], float); v = np.nan_to_num(v, nan=np.nanmean(v)); V[q] = v / v.sum()
        crowd = np.mean(list(V.values()), 0)
        top = [T[j] for j in np.flatnonzero(np.isclose(crowd, crowd.max()))]
        e = ex[d]
        if e not in top: continue
        share = 0
        for q, v in V.items():
            tops = [T[j] for j in np.flatnonzero(np.isclose(v, v.max()))]
            tot_players += 1; tied_players += len(tops) > 1
            share += (e in tops) / len(tops)
        rows.append(dict(game=g['id'], day=d, agree=share / len(V), wrong=e not in evil, crowd_tie=len(top) > 1))
t = pd.DataFrame(rows)
print(f' players whose top suspect is tied: {tied_players / tot_players:.1%}; town top suspect tied: {t.crowd_tie.mean():.1%} of {len(t)} executions')
t['band'] = pd.cut(t.agree, [0, .75, .9, 1.01], labels=['under 75%', '75 to 90%', '90% and up'], include_lowest=True)
for k, q in t.groupby('band', observed=True):
    print(f' {k:11s} executions {len(q):4d}  wrong {cboot(q, lambda z: z.wrong.mean())}')
print(' below 90%:', cboot(t[t.agree < .9], lambda z: z.wrong.mean()), 'n', (t.agree < .9).sum())
t.to_csv('ties.csv', index=False)
