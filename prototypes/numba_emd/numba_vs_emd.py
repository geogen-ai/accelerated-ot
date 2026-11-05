import sys, time, numpy as np, ot, numba as nb
from numba import njit, prange

@njit(parallel=True, fastmath=True, cache=False)
def lse_update(A, pot, logm, eps, out):
    n, m = A.shape
    for i in prange(n):
        hi = -np.inf
        for j in range(m):
            v = (pot[j] - A[i, j]) / eps
            if v > hi: hi = v
        s = 0.0
        for j in range(m):
            s += np.exp((pot[j] - A[i, j]) / eps - hi)
        out[i] = eps * (logm[i] - hi - np.log(s))

@njit(parallel=True, fastmath=True, cache=False)
def col_err(Mt, f, g, b, eps):
    m = Mt.shape[0]; e = np.zeros(m)
    for j in prange(m):
        s = 0.0
        for i in range(Mt.shape[1]):
            s += np.exp((f[i] + g[j] - Mt[j, i]) / eps)
        e[j] = abs(s - b[j])
    return e.sum()

@njit(cache=False)
def sinkhorn(M, Mt, a, b, eps, thr, maxit):
    n, m = M.shape
    f = np.zeros(n); g = np.zeros(m)
    la, lb = np.log(a), np.log(b)
    it = 0; conv = False
    while it < maxit:
        for _ in range(10):
            lse_update(Mt, f, lb, eps, g)
            lse_update(M, g, la, eps, f)
        it += 10
        if col_err(Mt, f, g, b, eps) < thr:
            conv = True; break
    return f, g, it, conv

def med(fn, r=3):
    t = time.perf_counter(); v = fn(); t0 = time.perf_counter() - t
    if t0 > 5: return t0, v
    ts = [t0]
    for _ in range(r - 1):
        t = time.perf_counter(); v = fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts)), v

nb.set_num_threads(int(sys.argv[1]))
print(f"threads={nb.get_num_threads()}")
print(f"{'n':>5} {'solver':>24} {'time_s':>9} {'x_emd':>8} {'cost_gap%':>10} {'row_err':>9} {'col_err':>9} {'iters':>6} conv", flush=True)
for n in [100, 300, 1000, 3000]:
    rng = np.random.default_rng(0)
    a = np.ones(n)/n; b = np.ones(n)/n
    M = ot.dist(rng.normal(size=(n, 2)), rng.normal(size=(n, 2)) + 1.0); M /= M.max()
    Mt = M.T.copy()
    tE, G = med(lambda: ot.emd(a, b, M, numItermax=10**7))
    cE = float((G*M).sum())
    print(f"{n:5d} {'POT emd (exact)':>24} {tE:9.4f} {1:8.1f} {0:10.3f}", flush=True)
    for eps, thr in [(1e-2, 1e-3), (1e-3, 1e-3), (1e-3, 1e-6), (1e-4, 1e-3)]:
        sinkhorn(M[:5, :5].copy(), Mt[:5, :5].copy(), np.ones(5)/5, np.ones(5)/5, eps, thr, 20)
        tS, (f, g, it, c) = med(lambda: sinkhorn(M, Mt, a, b, eps, thr, 20000))
        P = np.exp((f[:, None] + g[None, :] - M) / eps)
        gap = 100*((P*M).sum() - cE)/cE
        lbl = f"numba eps={eps:g} thr={thr:g}"
        print(f"{n:5d} {lbl:>24} {tS:9.4f} {tS/tE:8.1f} {gap:10.3f} {abs(P.sum(1)-a).max():9.1e} {abs(P.sum(0)-b).max():9.1e} {it:6d} {c}", flush=True)
