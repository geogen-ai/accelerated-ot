import sys, time, numpy as np, ot, numba as nb
from numba import njit, prange
sys.path.insert(0, sys.argv[1])
from nb_simplex import emd_batch
@njit(parallel=True)
def burn(k, reps):
    out = np.zeros(k)
    for t in prange(k):
        x = np.zeros(1000); s = 0.0
        for rep in range(reps):
            j = (rep * 7) % 1000
            x[j] += 1.0; s += x[(j * 3) % 1000]
        out[t] = s
    return out
burn(2, 10)
base = None
print("sustained burn (fixed total work, ~3 cpu-seconds):", flush=True)
for th in [1, 2, 4, 8]:
    nb.set_num_threads(th)
    s = time.perf_counter(); burn(th*8, 800000000//(th*8)); w = time.perf_counter() - s
    base = base or w
    print(f"  t={th} wall {w:.2f}s  speedup {base/w:.2f}x", flush=True)
r = np.random.default_rng(0); n = 300; k = 16
A = np.ones((k, n))/n; B = np.ones((k, n))/n
Ms = np.stack([ot.dist(r.normal(size=(n, 2)), r.normal(size=(n, 2)) + 1.0) for _ in range(k)])
Ms /= Ms.max(axis=(1, 2), keepdims=True)
def med(f, m=9, gap=0.0):
    ts = []
    for _ in range(m):
        time.sleep(gap)
        s = time.perf_counter(); f(); ts.append(time.perf_counter() - s)
    return float(np.median(ts))
emd_batch(A, B, Ms)
print("spaced bursts (1s idle before each run):", flush=True)
tS = med(lambda: [ot.emd2(A[i], B[i], Ms[i], numItermax=10**8) for i in range(k)], gap=1.0)
print(f"  POT serial {tS:.4f}s", flush=True)
for th in [1, 2, 4, 8]:
    nb.set_num_threads(th)
    t = med(lambda: emd_batch(A, B, Ms), gap=1.0)
    print(f"  numba prange t={th} {t:.4f}s  vs POT serial {tS/t:.2f}x", flush=True)
