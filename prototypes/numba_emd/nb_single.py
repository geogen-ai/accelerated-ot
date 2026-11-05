import sys, time, numpy as np, ot
sys.path.insert(0, sys.argv[1])
from nb_simplex import emd
emd(np.ones(3)/3, np.ones(3)/3, np.random.default_rng(0).random((3, 3)))
print(f"{'n':>5} {'emd_s':>9} {'numba_s':>9} {'ratio':>6} {'max_rel':>9}", flush=True)
for n in [100, 300, 1000, 3000, 5000]:
    E, N, rel = [], [], 0
    for seed in range(3):
        r = np.random.default_rng(seed)
        a = np.ones(n)/n; b = np.ones(n)/n
        M = ot.dist(r.normal(size=(n, 2)), r.normal(size=(n, 2)) + 1.0); M /= M.max()
        for _ in range(3):
            s = time.perf_counter(); G = ot.emd(a, b, M, numItermax=10**8); E.append(time.perf_counter() - s)
            s = time.perf_counter(); c = emd(a, b, M)[3]; N.append(time.perf_counter() - s)
        ce = float((G*M).sum()); rel = max(rel, abs(c-ce)/ce)
    e, m = float(np.median(E)), float(np.median(N))
    print(f"{n:5d} {e:9.4f} {m:9.4f} {m/e:6.2f} {rel:9.1e}", flush=True)
