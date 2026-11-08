"""Three angles on Clocktower Bench belief probes: who starts a wrong consensus, lineage cliques, better aggregation."""
import json, gzip, glob, sys, itertools, re
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

INFO = {'Washerwoman', 'Librarian', 'Investigator', 'Chef', 'Empath', 'Fortune Teller', 'Undertaker', 'Ravenkeeper'}
A, B, R3 = [], [], []
for f in sorted(glob.glob(sys.argv[1] + '/*.json.gz')):
    g = json.loads(gzip.open(f).read())
    seats = {s['n']: s for s in g['seats']}
    names = list(seats)
    evil = {n for n, s in seats.items() if s['e']}
    imp = next((n for n, s in seats.items() if s['c'] == 'Imp'), None)
    mixed = g['good'].startswith('mixed')
    # per day: nominations, executions, public mention order, whisper counts
    nom, ex, first_push, wh = {}, {}, {}, {}
    for day in g['days_']:
        d = day['d']; nom[d] = []; first_push[d] = {}
        for sec in day['secs']:
            for it in sec['it']:
                k = it['k']
                if k == 'nom':
                    nom[d].append((it.get('p'), it.get('n') or it.get('t') or it.get('x')))
                if k == 'ex' and it.get('died'):
                    ex[d] = it['p']
                if k in ('say', 'w') and it.get('x'):
                    sp = it['p']
                    for n in names:
                        if n != sp and n not in first_push[d] and re.search(r'\b(execute|nominate|vote|suspect|demon|imp)\b', it['x'], re.I) and re.search(r'\b' + n + r'\b', it['x']):
                            first_push[d][n] = sp
                if k == 'w':
                    for t in (it['to'] if isinstance(it['to'], list) else [it['to']]):
                        key = frozenset((it['p'], t)); wh[key] = wh.get(key, 0) + 1
    for b in g['belief']:
        d = b['d']; by = b['by']
        T = sorted({t for q in by for t in by[q]})
        if imp not in T or len(by) < 3:
            continue
        y = np.array([t == imp for t in T], float)
        V = {}
        for q in by:
            v = np.array([by[q].get(t, np.nan) for t in T], float); v = np.nan_to_num(v, nan=np.nanmean(v)); V[q] = v / v.sum()
        M = np.array(list(V.values())); crowd = M.mean(0); top = T[int(crowd.argmax())]
        agree = np.mean([T[int(v.argmax())] == top for v in M])
        # angle 1: who first pushed the town's top suspect, and who nominated
        e = ex.get(d)
        noms = [n for n in nom.get(d, []) if n[1] == e] if e else []
        A.append(dict(game=g['id'], good=g['good'], day=d, agree=agree, top=top, top_evil=top in evil, executed=e,
                      ex_top=e == top, ex_evil=(e in evil) if e else None,
                      pusher=first_push[d].get(top), pusher_evil=(first_push[d].get(top) in evil) if first_push[d].get(top) else None,
                      nominator_evil=(noms[0][0] in evil) if noms and noms[0][0] else None, won=g['winner'] == 'good'))
        # angle 2: lineage cliques with whisper control
        if mixed:
            for a, c in itertools.combinations(by, 2):
                keep = [i for i, t in enumerate(T) if t not in (a, c)]
                x, z = V[a][keep], V[c][keep]
                if x.std() and z.std():
                    ma, mc = g['models'][seats[a]['m']], g['models'][seats[c]['m']]
                    B.append(dict(game=g['id'], day=d, same=int(ma == mc), r=np.corrcoef(x, z)[0, 1],
                                  whispers=wh.get(frozenset((a, c)), 0), both_info=int(seats[a]['c'] in INFO and seats[c]['c'] in INFO)))
        # angle 3: is information concentrated in role-holders, and does weighting by claimed info help
        for q in by:
            br = ((V[q] - y) ** 2).sum()
            R3.append(dict(game=g['id'], day=d, prober=q, role=seats[q]['c'], info=seats[q]['c'] in INFO,
                          drunk=seats[q]['c'] == 'Drunk', brier=br, dev=((V[q] - crowd) ** 2).sum(),
                          p_imp=V[q][T.index(imp)], crowd_p=crowd[T.index(imp)]))

a = pd.DataFrame(A); b = pd.DataFrame(B); c = pd.DataFrame(R3)
a.to_csv('angle1.csv', index=False); b.to_csv('angle2.csv', index=False); c.to_csv('angle3.csv', index=False)

print('=== ANGLE 1: who starts the wrong consensus ===')
x = a[a.ex_top & a.executed.notna()].copy()
x['band'] = pd.cut(x.agree, [0, .75, .9, 1.01], labels=['<75%', '75-90%', '90%+'])
x['right'] = x.ex_evil.astype(bool)
print(x.groupby(['band', 'right'], observed=True).agg(n=('right', 'size'),
      first_pusher_evil=('pusher_evil', lambda s: s.dropna().astype(bool).mean()),
      pushers_found=('pusher_evil', lambda s: s.notna().mean()),
      nominator_evil=('nominator_evil', lambda s: s.dropna().astype(bool).mean())).round(3))
wrong = x[~x.right]
print('wrong executions where an evil player first pushed the target:', round(wrong.pusher_evil.dropna().astype(bool).mean(), 3),
      'n', wrong.pusher_evil.notna().sum())

print('\n=== ANGLE 2: lineage cliques, controlling for private talk ===')
m = smf.ols('r ~ same + np.log1p(whispers) + both_info + C(day)', b).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(b.game)[0]})
print(m.params[['same', 'np.log1p(whispers)', 'both_info']].round(3).to_dict())
print('p-values', m.pvalues[['same', 'np.log1p(whispers)', 'both_info']].round(4).to_dict(), 'pairs', len(b))
print('same-model pairs whisper more?', b.groupby('same').whispers.mean().round(2).to_dict())
print('gap by day:', b.groupby(['day', 'same']).r.mean().unstack().assign(gap=lambda t: t[1] - t[0]).gap.round(3).iloc[:5].to_dict())

print('\n=== ANGLE 3: where the information sits ===')
print(c.groupby(['info']).agg(n=('brier', 'size'), brier=('brier', 'mean'), p_imp=('p_imp', 'mean')).round(3))
print('Drunk seats (fake info):', c[c.drunk].brier.mean().round(3), 'n', c.drunk.sum())
c['contrarian'] = c.groupby(['game', 'day']).dev.rank(pct=True) > .75
print('contrarians (top quarter distance from crowd) vs rest, brier:', c.groupby('contrarian').brier.mean().round(3).to_dict())
c['beats_crowd'] = c.p_imp > c.crowd_p
print('contrarians who put MORE on the real Demon than the crowd:', c[c.contrarian].beats_crowd.mean().round(3),
      'vs others', c[~c.contrarian].beats_crowd.mean().round(3))
