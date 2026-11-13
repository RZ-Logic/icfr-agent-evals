"""Fabrication merge gate: pooled-count exact conditional binomial (pass-6 C6) vs per-case exact McNemar.
World model from pass-6 lane 7 (s03_ci_gate.py): 146 clean items (fab prob 0.0002 per generation x scale)
and 54 hard items (Beta(0.5, 12) x scale). E = 4 epochs. Two epoch-dependence regimes:
  indep: each epoch fabricates independently with the item's probability
  corr : the item fabricates in all 4 epochs or none, with probability 1-(1-p)^4 (same marginal any-fab rate)
Gate fires at one-sided alpha 0.025. Composition = CI split (146 standard + 54 generated hard)."""
import sys, numpy as np
from scipy import stats
rng = np.random.default_rng(23)
NC, NH, E = 146, 54, 4
def world(scale):
    return np.r_[np.full(NC, 0.0002), rng.beta(0.5, 12, NH)] * scale
def run(p, regime, mult=1.0):
    q = np.clip(p * mult, 0, 1)
    if regime == "indep":
        x = (rng.random((len(q), E)) < q[:, None]).sum(1)
    else:
        anyp = 1 - (1 - q) ** E
        x = np.where(rng.random(len(q)) < anyp, E, 0)
    return x                                   # per-case fabrication count over 4 epochs
def binom_gate(xb, xc):
    n = xb.sum() + xc.sum()
    return n > 0 and stats.binom.sf(xc.sum() - 1, n, 0.5) <= 0.025
def mcnemar_gate(xb, xc):
    b = int(((xb > 0) & (xc == 0)).sum()); c = int(((xb == 0) & (xc > 0)).sum())
    return (b + c) > 0 and stats.binom.sf(c - 1, b + c, 0.5) <= 0.025
sims = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
for scale in (0.25, 1.0):
    for regime in ("indep", "corr"):
        out = {}
        for label, mult in (("null", 1.0), ("fab x2", 2.0)):
            fb = fm = 0
            for _ in range(sims):
                p = world(scale); xb = run(p, regime); xc = run(p, regime, mult)
                fb += binom_gate(xb, xc); fm += mcnemar_gate(xb, xc)
            out[label] = (fb / sims, fm / sims)
        print(f"scale {scale:4} {regime:5}  null: pooled-binomial {out['null'][0]:.3f}  per-case McNemar {out['null'][1]:.3f}   "
              f"| fab x2: pooled-binomial {out['fab x2'][0]:.3f}  per-case McNemar {out['fab x2'][1]:.3f}")
