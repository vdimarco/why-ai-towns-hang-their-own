"""Confidence intervals and tests reported in the paper. Run in the folder holding the CSVs from steps 01-03."""
import sys, json, gzip, glob, itertools
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint as pc
from scipy.stats import fisher_exact
rb = np.random.default_rng(0)
def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; m = [rb.choice(x, len(x)).mean() for _ in range(2000)]
    return f'{x.mean():.3f} [{np.percentile(m, 2.5):.3f}, {np.percentile(m, 97.5):.3f}]'
a = pd.read_csv('angle1.csv'); x = a[a.ex_top & a.executed.notna()].copy(); x['right'] = x.ex_evil.astype(str) == 'True'
x['band'] = pd.cut(x.agree, [0, .75, .9, 1.01], labels=['<75', '75-90', '90+'])
print('== Executions by town agreement (Wilson 95%) ==')
for k, s in x.groupby('band', observed=True):
    w = (~s.right).sum(); lo, hi = pc(w, len(s), method='wilson'); print(k, f'{w}/{len(s)} wrong = {w/len(s):.2f} [{lo:.2f}, {hi:.2f}]')
b = x[x.band != '90+']; w = (~b.right).sum(); lo, hi = pc(w, len(b), method='wilson'); print('below 90%', f'{w}/{len(b)} = {w/len(b):.2f} [{lo:.2f}, {hi:.2f}]')
s = x[x.nominator_evil.notna()].copy(); s['evnom'] = s.nominator_evil.astype(str) == 'True'
print('== Nominated by evil ==  Fisher p', fisher_exact(pd.crosstab(s.right, s.evnom).values)[1])
for k, g in s.groupby('right'):
    lo, hi = pc(g.evnom.sum(), len(g), method='wilson'); print('right' if k else 'wrong', f'{g.evnom.mean():.2f} [{lo:.2f}, {hi:.2f}] n={len(g)}')
cw = x[(~x.right) & (x.agree >= .75)].drop_duplicates('game'); base = a.drop_duplicates('game')
lo, hi = pc(cw.won.sum(), len(cw), method='wilson'); print('good win after confident wrong execution', f'{cw.won.mean():.2f} [{lo:.2f}, {hi:.2f}] games={len(cw)}; all games {base.won.mean():.2f}')
p = pd.read_csv('angle2.csv')
m = smf.ols('r ~ same + np.log1p(whispers) + both_info + C(day)', p).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(p.game)[0]})
print('== Lineage regression ==', m.params[['same', 'np.log1p(whispers)', 'both_info']].round(3).to_dict(), m.pvalues[['same', 'np.log1p(whispers)', 'both_info']].round(4).to_dict(), 'pairs', len(p))
g = p.groupby(['game', 'same']).r.mean().unstack().dropna(); d = g[1] - g[0]
print('per-game gap', ci(d), 'games', len(d), 'share>0', round((d > 0).mean(), 2))
q = p[p.day > 1].groupby(['game', 'same']).r.mean().unstack().dropna(); print('gap excluding day 1', ci(q[1] - q[0]))
print('gap by day', p.groupby(['day', 'same']).r.mean().unstack().pipe(lambda t: t[1] - t[0]).round(3).iloc[:5].to_dict())
print('whispers by pair type', p.groupby('same').whispers.mean().round(2).to_dict())
i = pd.read_csv('indep.csv'); i['won'] = i.game.map(base.set_index('game').won)
meta = {}
for f in glob.glob(sys.argv[1] + '/*.json.gz'):
    gg = json.loads(gzip.open(f).read()); meta[gg['id']] = (gg['good'], gg['evil'])
gl = i[i.day <= 2].groupby('game').r.mean().to_frame().dropna()
gl['good'] = [meta[k][0] for k in gl.index]; gl['evil'] = [meta[k][1] for k in gl.index]; gl['won'] = base.set_index('game').won.reindex(gl.index).astype(int)
gl['z'] = (gl.r - gl.r.mean()) / gl.r.std()
mm = smf.logit('won ~ z + C(good) + C(evil)', gl).fit(disp=0); print('== Agreement days 1-2 -> good win ==', round(mm.params.z, 3), 'p', round(mm.pvalues.z, 4), 'games', len(gl))
c = pd.read_csv('calib.csv'); c['fam'] = np.where(c.model.str.startswith('gpt'), 'gpt', 'claude'); top = c[c.p > .75]
print('== Calibration at 75-100% ==', top.groupby('fam').hit.mean().round(3).to_dict())
pr = pd.read_csv('angle3.csv'); print('== Information roles ==', pr.groupby('info').brier.mean().round(3).to_dict())
