import sys, time, numpy as np, ot, numba as nb
sys.path.insert(0, sys.argv[1])
from nb_simplex import emd, emd_batch

def check(name, a, b, M):
    r = ot.emd(a, b, M, numItermax=10**8, log=True)
    ref = r[1]['cost']
    G, al, be, c, st, it = emd(a, b, M)
    rel = abs(c - ref) / abs(ref) if ref != 0 else abs(c)
    marg = max(abs(G.sum(1) - a).max(), abs(G.sum(0) - b).max())
    mask_a, mask_b = a > 0, b > 0
    slack = (M - al[:, None] - be[None, :])[np.ix_(mask_a, mask_b)].min()
    dual = (a * al).sum() + (b * be).sum()
    gapd = abs(dual - ref) / abs(ref) if ref != 0 else abs(dual)
    ok = st == 1 and rel < 1e-9 and marg < 1e-9 and slack > -1e-9 * max(M.max(), 1e-300) and gapd < 1e-9 + 1e-12/max(abs(ref),1e-300)*0 and G.min() >= -1e-12
    print(f"{'OK ' if ok else 'BAD'} {name:<30} ref={ref:.6g} rel={rel:.1e} marg={marg:.1e} slack={slack:.1e} dualgap={gapd:.1e} iters={it}", flush=True)
    return ok

rng = np.random.default_rng(1)
emd(np.ones(3)/3, np.ones(3)/3, rng.random((3, 3)))
res = []
for s in range(20):
    r = np.random.default_rng(s)
    n, m = r.integers(2, 80, 2)
    x, y = r.normal(size=(n, 2)), r.normal(size=(m, 2))
    a, b = r.dirichlet(np.ones(n)), r.dirichlet(np.ones(m))
    res.append(check(f"random rect {n}x{m} s{s}", a, b, ot.dist(x, y)))
n = 60; u = np.ones(n)/n
x = rng.normal(size=(n, 2))
res.append(check("identical points (cost 0 diag)", u, u, ot.dist(x, x)))
res.append(check("all-zero cost", u, u, np.zeros((n, n))))
res.append(check("constant cost", u, u, np.ones((n, n))))
g = np.array([(i, j) for i in range(8) for j in range(8)], float); w = np.ones(64)/64
res.append(check("integer grid ties", w, w, ot.dist(g, g, 'cityblock')))
res.append(check("integer random cost", u, u, rng.integers(0, 5, (n, n)).astype(float)))
res.append(check("1 x n", np.ones(1), u, rng.random((1, n))))
res.append(check("n x 1", u, np.ones(1), rng.random((n, 1))))
a0 = u.copy(); a0[:10] = 0; a0 /= a0.sum(); b0 = u.copy(); b0[-7:] = 0; b0 /= b0.sum()
res.append(check("zero-mass atoms", a0, b0, ot.dist(x, x[::-1])))
y = rng.normal(size=(n, 2))
res.append(check("huge cost scale", u, u, ot.dist(x, y) * 1e8))
res.append(check("tiny cost scale", u, u, ot.dist(x, y) * 1e-8))
print(f"validation: {sum(res)}/{len(res)} OK", flush=True)
if len(sys.argv) > 2 and sys.argv[2] == 'quick':
    sys.exit()

print(f"\nsingle solve, 1 thread, median of runs\n{'n':>5} {'seed':>4} {'emd_s':>9} {'numba_s':>9} {'ratio':>6} {'rel_diff':>9} {'iters':>8}", flush=True)
for n in [100, 300, 1000, 3000, 5000]:
    rs = []
    for seed in range(3):
        r = np.random.default_rng(seed)
        a = np.ones(n)/n; b = np.ones(n)/n
        M = ot.dist(r.normal(size=(n, 2)), r.normal(size=(n, 2)) + 1.0); M /= M.max()
        def t(f, k=3):
            ts = []
            for _ in range(k):
                s = time.perf_counter(); v = f(); ts.append(time.perf_counter() - s)
            return float(np.median(ts)), v
        tE, Ge = t(lambda: ot.emd(a, b, M, numItermax=10**8))
        tN, (G, al, be, c, st, it) = t(lambda: emd(a, b, M))
        cE = float((Ge*M).sum())
        rs.append(tN/tE)
        print(f"{n:5d} {seed:4d} {tE:9.4f} {tN:9.4f} {tN/tE:6.2f} {abs(c-cE)/cE:9.1e} {it:8d}", flush=True)
    print(f"{n:5d} mean ratio {np.mean(rs):.2f}", flush=True)

print("\nbatch: 16 problems n=300", flush=True)
r = np.random.default_rng(0); n = 300; k = 16
A = np.ones((k, n))/n; B = np.ones((k, n))/n
Ms = np.stack([ot.dist(r.normal(size=(n, 2)), r.normal(size=(n, 2)) + 1.0) for _ in range(k)])
Ms /= Ms.max(axis=(1, 2), keepdims=True)
s = time.perf_counter(); ref = np.array([ot.emd2(A[i], B[i], Ms[i], numItermax=10**8) for i in range(k)]); tS = time.perf_counter() - s
for th in [1, 2, 4]:
    nb.set_num_threads(th)
    emd_batch(A, B, Ms)
    s = time.perf_counter(); out = emd_batch(A, B, Ms); tP = time.perf_counter() - s
    print(f"POT serial {tS:.4f}s  numba prange threads={th} {tP:.4f}s  speedup {tS/tP:.2f}x  max_rel {np.max(abs(out-ref)/ref):.1e}", flush=True)
