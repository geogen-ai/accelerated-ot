# Compare the original LOPT/LOT code (lib_lopt + POT emd_c) with an all-Numba
# version built on the prototype Numba network simplex.
# usage: python lopt_bench.py <path to fused-pgw/lib> <threads> <orig|orig_pot1|numba>
# Run each mode in its own process: Numba's OpenMP worker threads keep spinning
# after a prange call and slow POT's solver when both share a process.
import sys, os, time
import numpy as np
import numba as nb
from numba import njit, prange

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_lopt
from nb_simplex import emd

nb.set_num_threads(int(sys.argv[2]))

# lib_lopt passes numThreads=10 to POT; optionally force POT's serial solver.
mode = sys.argv[3]
if mode == 'orig_pot1':
    _emd_c, _emd = lib_lopt.emd_c, lib_lopt.ot.lp.emd
    lib_lopt.emd_c = lambda a, b, M, it, th: _emd_c(a, b, M, it, 1)
    lib_lopt.ot.lp.emd = lambda a, b, M, numItermax=100000, numThreads=1: _emd(a, b, M, numItermax=numItermax, numThreads=1)
    print("POT emd forced to numThreads=1", flush=True)


@njit
def cost_sq(X, Y):
    n, d = X.shape
    m = Y.shape[0]
    C = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(d):
                t = X[i, k] - Y[j, k]
                s += t * t
            C[i, j] = s
    return C


@njit
def lot_embedding_nb(X0, X1, p0, p1):
    G = emd(p0, p1, cost_sq(X0, X1))[0]
    return G @ X1 / p0[:, None] - X0


@njit
def lopt_embedding_nb(X0, X1, p0, p1, Lambda):
    n, m = X0.shape[0], X1.shape[0]
    C1 = np.zeros((n + 1, m + 1))
    C1[:n, :m] = cost_sq(X0, X1) - 2 * Lambda
    mu1 = np.zeros(n + 1)
    nu1 = np.zeros(m + 1)
    mu1[:n] = p0
    nu1[:m] = p1
    mu1[n] = p1.sum()
    nu1[m] = p0.sum()
    gamma = np.ascontiguousarray(emd(mu1, nu1, C1)[0][:n, :m])
    p1_hat = gamma.sum(axis=1)
    GX = gamma @ X1
    U1 = np.zeros_like(X0)
    for i in range(n):
        if p1_hat[i] > 1e-10:
            U1[i] = GX[i] / p1_hat[i] - X0[i]
    M1 = p1.sum() - p1_hat.sum()
    p1_perp = p1 - gamma.sum(axis=0)
    return U1, p1_hat, M1, p1_perp


@njit(parallel=True)
def lopt_embedding_batch(X0, Xs, p0, Ps, Lambda):
    K = Xs.shape[0]
    U = np.empty((K,) + X0.shape)
    for k in prange(K):
        U[k] = lopt_embedding_nb(X0, Xs[k], p0, Ps[k], Lambda)[0]
    return U


@njit(parallel=True)
def lot_barycenter_nb(Xs, Ps, X0, p0, w, numItermax, stopThr):
    K = Xs.shape[0]
    U = np.empty((K,) + X0.shape)
    for _ in range(numItermax):
        for k in prange(K):
            U[k] = lot_embedding_nb(X0, Xs[k], p0, Ps[k])
        Ubar = np.zeros_like(X0)
        for k in range(K):
            Ubar += w[k] * U[k]
        X0 = X0 + Ubar
        if (Ubar ** 2).sum() <= stopThr:
            break
    return X0


def timed(f, reps=3):
    ts = []
    for _ in range(reps):
        s = time.perf_counter()
        out = f()
        ts.append(time.perf_counter() - s)
    return float(np.median(ts)), out


rng = np.random.default_rng(0)
d, Lambda = 2, 0.5
print(f"mode={mode} numba_threads={nb.get_num_threads()} (JIT compile excluded)", flush=True)
if mode == 'numba':
    Xw = np.random.default_rng(9).normal(size=(8, d)); pw = np.ones(8) / 8
    lopt_embedding_batch(Xw, Xw[None].copy(), pw, pw[None].copy(), Lambda)
    lot_barycenter_nb(Xw[None].copy(), pw[None].copy(), Xw, pw, np.ones(1), 2, 1e-7)
res = {}

print("LOPT embedding of K measures (targets have mass 0.6..1.0)", flush=True)
for n, K in [(100, 200), (300, 64), (1000, 16), (2000, 4)]:
    X0 = rng.normal(size=(n, d)); p0 = np.ones(n) / n
    Xs = rng.normal(size=(K, n, d)) + 0.5
    Ps = rng.dirichlet(np.ones(n), size=K) * rng.uniform(0.6, 1.0, size=(K, 1))
    if mode == 'numba':
        t, U = timed(lambda: lopt_embedding_batch(X0, Xs, p0, Ps, Lambda))
    else:
        t, U = timed(lambda: np.stack([lib_lopt.lopt_embedding(X0, Xs[k], p0, Ps[k], Lambda)[0] for k in range(K)]))
    res[f"lopt_{n}_{K}"] = U; res[f"t_lopt_{n}_{K}"] = t
    print(f"  n={n:5d} K={K:4d} {t:8.3f} s", flush=True)

print("LOT barycenter, 10 outer iterations", flush=True)
for n, K in [(100, 32), (300, 16), (1000, 8)]:
    X0 = rng.normal(size=(n, d)); p0 = np.ones(n) / n
    Xs = rng.normal(size=(K, n, d)) * rng.uniform(0.5, 2, size=(K, 1, 1))
    Ps = np.ones((K, n)) / n
    w = np.ones(K) / K
    if mode == 'numba':
        t, B = timed(lambda: lot_barycenter_nb(Xs, Ps, X0, p0, w, 10, 0.0), reps=1)
    else:
        t, B = timed(lambda: lib_lopt.lot_barycenter(list(Xs), list(Ps), X0, p0, w, numItermax=10, stopThr=0.0), reps=1)
    res[f"bary_{n}_{K}"] = B; res[f"t_bary_{n}_{K}"] = t
    print(f"  n={n:5d} K={K:4d} {t:8.3f} s", flush=True)
np.savez(f"lopt_bench_{mode}_t{nb.get_num_threads()}.npz", **res)
