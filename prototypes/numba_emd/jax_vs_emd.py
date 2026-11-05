import time, numpy as np, ot, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from ott.geometry import geometry
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

def solve(M, a, b, eps, thr):
    g = geometry.Geometry(cost_matrix=M, epsilon=eps)
    o = sinkhorn.Sinkhorn(threshold=thr, max_iterations=20000)(linear_problem.LinearProblem(g, a=a, b=b))
    return o.matrix, o.converged, o.n_iters
jsolve = jax.jit(solve, static_argnames=("thr",))

def med(fn, r=3):
    ts = []
    for _ in range(r):
        t = time.perf_counter(); v = fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts)), v

print(f"{'n':>5} {'solver':>22} {'time_s':>9} {'x_emd':>8} {'cost_gap%':>10} {'marg_err':>9} {'iters':>6} conv")
for n in [100, 300, 1000, 3000]:
    rng = np.random.default_rng(0)
    a = np.ones(n)/n; b = np.ones(n)/n
    M = ot.dist(rng.normal(size=(n, 2)), rng.normal(size=(n, 2)) + 1.0); M /= M.max()
    tE, G = med(lambda: ot.emd(a, b, M, numItermax=10**7))
    cE = float((G*M).sum())
    print(f"{n:5d} {'POT emd (exact)':>22} {tE:9.4f} {1:8.1f} {0:10.3f} {abs(G.sum(1)-a).max():9.1e}")
    Mj, aj, bj = map(jnp.asarray, (M, a, b))
    for eps, thr in [(1e-2, 1e-3), (1e-3, 1e-3), (1e-3, 1e-6), (1e-4, 1e-3)]:
        jsolve(Mj, aj, bj, eps, thr)[0].block_until_ready()
        def run():
            P, c, it = jsolve(Mj, aj, bj, eps, thr); P.block_until_ready(); return P, c, it
        tS, (P, c, it) = med(run)
        P = np.asarray(P)
        gap = 100*((P*M).sum() - cE)/cE
        lbl = f"ott eps={eps:g} thr={thr:g}"
        print(f"{n:5d} {lbl:>22} {tS:9.4f} {tS/tE:8.1f} {gap:10.3f} {abs(P.sum(1)-a).max():9.1e} {int(it):6d} {bool(c)}", flush=True)
