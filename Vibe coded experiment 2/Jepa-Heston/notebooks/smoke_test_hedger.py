# Usage: python smoke_test_hedger.py notebooks/06_two_instrument_hedging_fixed.ipynb
# Trains the hedger on synthetic GBM data. Expect PASS (std drops far below unhedged). Takes ~1-2 min on CPU.
import sys, json, numpy as np, torch, torch.nn as nn
nbk = json.load(open(sys.argv[1]))
src = {c["source"][0][:30]: "".join(c["source"]) for c in nbk["cells"] if c["cell_type"]=="code"}
net_cell = [s for s in ("".join(c["source"]) for c in nbk["cells"] if c["cell_type"]=="code") if "class Hedger" in s][0]

# synthetic GBM market (stock-only is hedgeable; dL is pure noise so the d2 leg is useless)
from math import erf, sqrt, exp, pi
rng = np.random.default_rng(0)
N, T, S0, K, sig, Ttot = 20000, 30, 100.0, 100.0, 0.2, 30/365
dt = Ttot/T
z = rng.standard_normal((N, T))
logS = np.log(S0) + np.cumsum(-0.5*sig**2*dt + sig*np.sqrt(dt)*z, axis=1)
S = np.concatenate([np.full((N,1), S0), np.exp(logS)], axis=1).astype(np.float32)
payoff = np.maximum(S[:,-1]-K, 0).astype(np.float32)
d1 = (0.5*sig**2*Ttot)/(sig*sqrt(Ttot)); Phi = lambda x: 0.5*(1+erf(x/sqrt(2)))
PREMIUM = S0*Phi(d1) - K*Phi(d1 - sig*sqrt(Ttot))
dS = (S[:,1:]-S[:,:-1]).astype(np.float32)
dL = (0.001*rng.standard_normal((N,T))).astype(np.float32)
tau = np.broadcast_to((Ttot - dt*np.arange(T)).astype(np.float32), (N,T))
F = np.stack([np.log(S[:,:T]/S0), tau], -1).astype(np.float32)
F = (F - F.reshape(-1,2).mean(0))/F.reshape(-1,2).std(0)

DEVICE='cpu'; ALPHA=0.5; N_STEPS=1500; BATCH=1024; LR=1e-3; N_UNITS=56; N_LAYERS=2
D2_SCALE=float(dS.std()/dL.std())
class _P:
    def __init__(s,it): s.it=it
    def __iter__(s): return iter(s.it)
    def set_postfix(s,d): pass
tqdm=lambda it, desc='': _P(it)
exec(net_cell)
t=lambda a: torch.tensor(a, dtype=torch.float32)
W0 = PREMIUM - payoff
cv=lambda W: np.sort(-W)[int(ALPHA*len(W)):].mean()
print(f"unhedged   std={W0.std():.3f} CVaR={cv(W0):.3f}  (premium {PREMIUM:.3f})")
net,_ = train_one(t(F), t(dS), t(dL), t(payoff), 'smoke', 0)
W = eval_W(net, t(F), t(dS), t(dL), t(payoff))
print(f"trained    std={W.std():.3f} CVaR={cv(W):.3f}  mean={W.mean():+.3f}")
print("PASS" if W.std() < 0.5*W0.std() else "FAIL: hedger did not learn")
