"""Independence check. Agreement is the mean pairwise correlation of beliefs over innocent players only
(the real Demon and the pair's own seats are left out), so that agreeing on the right answer does not count.
Controls: simulated forecasters we know are independent must score near n."""
import json, gzip, glob, itertools, sys
import numpy as np, pandas as pd

def agree(V, T, imp, names):
    rs = []
    for a, c in itertools.combinations(range(len(V)), 2):
        keep = [i for i, t in enumerate(T) if t not in (names[a], names[c], imp)]
        x, y = V[a][keep], V[c][keep]
        if x.std() > 0 and y.std() > 0:
            rs.append(np.corrcoef(x, y)[0, 1])
    return np.mean(rs) if rs else np.nan

neff = lambda n, r: n / (1 + (n - 1) * min(max(r, 0), 1))

# controls
rng = np.random.default_rng(0); n, k = 8, 11
T = [f'p{i}' for i in range(k)]; names = [f'g{i}' for i in range(n)]; imp = 'p0'
def sim(kind):
    out = []
    for _ in range(500):
        if kind == 'random':          # each agent guesses at random
            V = rng.dirichlet(np.ones(k), n)
        elif kind == 'noisy_truth':   # each agent sees the truth through its own private noise
            V = np.array([np.exp(np.r_[2.0, np.zeros(k - 1)] + rng.normal(0, 1, k)) for _ in range(n)]); V /= V.sum(1, keepdims=True)
        elif kind == 'shared_prior':  # same prior over innocents (shared public evidence), private noise on top
            base = rng.normal(0, 1, k)
            V = np.array([np.exp(np.r_[2.0, np.zeros(k - 1)] + base + rng.normal(0, 1, k)) for _ in range(n)]); V /= V.sum(1, keepdims=True)
        elif kind == 'copy':          # one agent's view copied with tiny noise
            v = rng.dirichlet(np.ones(k)); V = np.clip(v + rng.normal(0, .005, (n, k)), 1e-4, 1); V /= V.sum(1, keepdims=True)
        out.append(neff(n, agree(V, T, imp, names)))
    return np.mean(out)
print('CONTROLS (n = 8): effective number our metric reports')
for kd in ['random', 'noisy_truth', 'shared_prior', 'copy']:
    print(f'  {kd:13s} {sim(kd):.2f}')

# real data, innocent-only agreement
rows = []
for f in sorted(glob.glob(sys.argv[1] + '/*.json.gz')):
    g = json.loads(gzip.open(f).read())
    imp = next((s['n'] for s in g['seats'] if s['c'] == 'Imp'), None)
    fam = 'mixed' if g['good'].startswith('mixed') else ('gpt' if g['good'].startswith('gpt') else 'claude')
    for b in g['belief']:
        by = b['by']; T = sorted({t for q in by for t in by[q]})
        if imp not in T or len(by) < 3: continue
        names = list(by); V = []
        for q in names:
            v = np.array([by[q].get(t, np.nan) for t in T], float); v = np.nan_to_num(v, nan=np.nanmean(v)); V.append(v / v.sum())
        r = agree(np.array(V), T, imp, names)
        rows.append(dict(game=g['id'], fam=fam, day=b['d'], n=len(names), r=r, neff=neff(len(names), r) if r == r else np.nan))
d = pd.DataFrame(rows); d.to_csv('indep.csv', index=False)
rb = np.random.default_rng(1)
def ci(x):
    x = x.dropna().values; m = [rb.choice(x, len(x)).mean() for _ in range(2000)]
    return f'{x.mean():.2f} [{np.percentile(m, 2.5):.2f}, {np.percentile(m, 97.5):.2f}]'
print('\nREAL DATA, innocent-only agreement r by lab and day:')
print(d[d.day <= 5].groupby(['fam', 'day']).r.mean().unstack().round(2))
print('\neffective number (95% CI):')
for fm in ['claude', 'gpt', 'mixed']:
    print(' ', fm, {dd: ci(d[(d.fam == fm) & (d.day == dd)].neff) for dd in (1, 2, 5)})
print('mean n', round(d.n.mean(), 1))
