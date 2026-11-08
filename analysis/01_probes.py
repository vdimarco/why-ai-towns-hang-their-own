"""Build a per (game, day) and per (game, day, prober) table from Clocktower Bench belief probes."""
import json, gzip, glob, sys, itertools
import numpy as np, pandas as pd

src = sys.argv[1]
rows, prow, calib = [], [], []
for f in sorted(glob.glob(f'{src}/*.json.gz')):
    g = json.loads(gzip.open(f).read())
    seats = {s['n']: s for s in g['seats']}
    imp = next((s['n'] for s in g['seats'] if s['c'] == 'Imp'), None)
    evil = {s['n'] for s in g['seats'] if s['e']}
    # day the starting Imp leaves the game; after that the Demon may have moved, so stop there
    imp_dead, execs, deaths_by_day = None, {}, {}
    for day in g['days_']:
        for sec in day['secs']:
            for it in sec['it']:
                if it['k'] == 'death' and it['p'] == imp and imp_dead is None:
                    imp_dead = day['d']
                if it['k'] == 'ex' and it.get('died'):
                    execs[day['d']] = it['p']
    gm = g['models']
    for b in g['belief']:
        d = b['d']
        if imp is None or (imp_dead is not None and d > imp_dead):
            continue
        by = b['by']
        probers = list(by)
        targets = sorted({t for p in probers for t in by[p]})
        if imp not in targets or len(probers) < 2:
            continue
        V, models = [], []
        for p in probers:
            v = np.array([by[p].get(t, np.nan) for t in targets], float)
            v = np.nan_to_num(v, nan=np.nanmean(v))
            v = v / v.sum() if v.sum() > 0 else np.full(len(v), 1 / len(v))
            V.append(v); models.append(gm[seats[p]['m']] if 'm' in seats[p] else g['good'])
        V = np.array(V)
        y = np.array([t == imp for t in targets], float)
        ind = ((V - y) ** 2).sum(1)
        crowd = V.mean(0)
        cb = ((crowd - y) ** 2).sum()
        # mean pairwise correlation of error vectors (errors, not beliefs: what matters for averaging)
        E = V - y
        cors = [np.corrcoef(E[i], E[j])[0, 1] for i, j in itertools.combinations(range(len(V)), 2)
                if E[i].std() > 0 and E[j].std() > 0]
        rho = float(np.nanmean(cors)) if cors else np.nan
        n = len(V)
        top = targets[int(crowd.argmax())]
        ex = execs.get(d)
        for i, p in enumerate(probers):
            prow.append(dict(game=g['id'], day=d, prober=p, model=models[i], brier=ind[i],
                             p_demon=V[i][targets.index(imp)], top_right=targets[int(V[i].argmax())] == imp,
                             dev=float(((V[i] - crowd) ** 2).sum())))
            for t, pr in zip(targets, V[i]):
                if t != p:
                    calib.append((models[i], pr, t == imp))
        rows.append(dict(game=g['id'], batch=g['batch'], group=g['group'], good=g['good'], evil=g['evil'],
                         winner=g['winner'], day=d, n=n, ntarg=len(targets),
                         ind_brier=ind.mean(), crowd_brier=cb, div_gain=ind.mean() - cb, rho=rho,
                         neff=n / (1 + (n - 1) * rho) if rho == rho else np.nan,
                         crowd_p_demon=crowd[targets.index(imp)], top=top, top_right=top == imp,
                         top_evil=top in evil, top_share=crowd.max(),
                         agree_top=float(np.mean([targets[int(v.argmax())] == top for v in V])),
                         executed=ex, ex_is_top=ex == top if ex else None,
                         ex_evil=(ex in evil) if ex else None, ex_demon=(ex == imp) if ex else None,
                         prober_models=','.join(sorted(set(models)))))
pd.DataFrame(rows).to_csv('days.csv', index=False)
pd.DataFrame(prow).to_csv('probers.csv', index=False)
pd.DataFrame(calib, columns=['model', 'p', 'hit']).to_csv('calib.csv', index=False)
print(len(rows), 'game-days;', len(prow), 'prober-days;', len(calib), 'probabilities')
