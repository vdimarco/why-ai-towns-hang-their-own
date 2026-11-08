"""Compare ways to combine the good team's beliefs (Brier, lower is better)."""
import json, gzip, glob, sys, numpy as np, pandas as pd
R = []
for f in sorted(glob.glob(sys.argv[1] + '/*.json.gz')):
    g = json.loads(gzip.open(f).read()); imp = next((s['n'] for s in g['seats'] if s['c'] == 'Imp'), None)
    for b in g['belief']:
        by = b['by']; T = sorted({t for q in by for t in by[q]})
        if imp not in T or len(by) < 3: continue
        y = np.array([t == imp for t in T], float); V = []
        for q in by:
            v = np.array([by[q].get(t, np.nan) for t in T], float); v = np.nan_to_num(v, nan=np.nanmean(v)); V.append(np.clip(v / v.sum(), 1e-3, 1))
        V = np.array(V); bri = lambda x: ((x / x.sum() - y) ** 2).sum(); ent = -(V * np.log(V)).sum(1); w = np.exp(-3 * ent); lp = np.exp(np.log(V).mean(0))
        R.append(dict(uniform=bri(np.ones(len(T))), avg_member=np.mean([bri(v) for v in V]), best_member=min(bri(v) for v in V),
                      mean=bri(V.mean(0)), median=bri(np.median(V, 0)), logpool=bri(lp), extremized=bri(lp ** 2), confidence_weighted=bri((V * w[:, None]).sum(0))))
r = pd.DataFrame(R); print(r.mean().round(3).to_string()); print('best member beats mean on', round((r.best_member < r['mean']).mean(), 3), 'of game-days')
