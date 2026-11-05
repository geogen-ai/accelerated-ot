# Prototype Numba port of the network simplex in POT's network_simplex_simple.h,
# which Nicolas Bonneel adapted from LEMON (Copyright (C) 2003-2010 EGRES).
import numpy as np
from numba import njit, prange

TREE, LOWER, UPPER = 0, 1, -1
EPS = 2.2204460492503131e-15
EPS_FEAS = 1e-8


@njit
def _entering(next_arc, in_arc, m, block, src, tgt, cost, state, pi):
    mn = 0.0
    cnt = block
    e = next_arc
    for _ in range(m):
        c = state[e] * (cost[e] + pi[src[e]] - pi[tgt[e]])
        if c < mn:
            mn = c
            in_arc = e
        cnt -= 1
        if cnt == 0:
            a = max(abs(pi[src[in_arc]]), abs(pi[tgt[in_arc]]), abs(cost[in_arc]))
            if mn < -EPS * a:
                return True, in_arc, e
            cnt = block
        e += 1
        if e == m:
            e = 0
    a = max(abs(pi[src[in_arc]]), abs(pi[tgt[in_arc]]), abs(cost[in_arc]))
    if mn >= -EPS * a:
        return False, in_arc, e
    return True, in_arc, e


@njit
def _pivot(in_arc, src, tgt, cost, flow, state, pi, parent, pred, forward,
           thread, rev_thread, succ_num, last_succ, dirty):
    # findJoinNode
    u = src[in_arc]
    v = tgt[in_arc]
    while u != v:
        if succ_num[u] < succ_num[v]:
            u = parent[u]
        else:
            v = parent[v]
    join = u

    # findLeavingArc
    if state[in_arc] == LOWER:
        first, second = src[in_arc], tgt[in_arc]
    else:
        first, second = tgt[in_arc], src[in_arc]
    delta = np.inf
    result = 0
    u_out = -1
    u = first
    while u != join:
        d = flow[pred[u]] if forward[u] else np.inf
        if d < delta:
            delta = d
            u_out = u
            result = 1
        u = parent[u]
    u = second
    while u != join:
        d = np.inf if forward[u] else flow[pred[u]]
        if d <= delta:
            delta = d
            u_out = u
            result = 2
        u = parent[u]
    if result == 1:
        u_in, v_in = first, second
    else:
        u_in, v_in = second, first
    change = result != 0
    if delta == np.inf:
        return False

    # changeFlow
    if delta > 0:
        val = state[in_arc] * delta
        flow[in_arc] += val
        u = src[in_arc]
        while u != join:
            flow[pred[u]] += -val if forward[u] else val
            u = parent[u]
        u = tgt[in_arc]
        while u != join:
            flow[pred[u]] += val if forward[u] else -val
            u = parent[u]
    if not change:
        state[in_arc] = -state[in_arc]
        return True
    state[in_arc] = TREE
    state[pred[u_out]] = LOWER if flow[pred[u_out]] == 0 else UPPER

    # updateTreeStructure
    old_rev_thread = rev_thread[u_out]
    old_succ_num = succ_num[u_out]
    old_last_succ = last_succ[u_out]
    v_out = parent[u_out]
    u = last_succ[u_in]
    right = thread[u]
    if old_rev_thread == v_in:
        last = thread[last_succ[u_out]]
    else:
        last = thread[v_in]
    stem = u_in
    thread[v_in] = stem
    dirty[0] = v_in
    nd = 1
    par_stem = v_in
    while stem != u_out:
        new_stem = parent[stem]
        thread[u] = new_stem
        dirty[nd] = u
        nd += 1
        w = rev_thread[stem]
        thread[w] = right
        rev_thread[right] = w
        parent[stem] = par_stem
        par_stem = stem
        stem = new_stem
        u = rev_thread[par_stem] if last_succ[stem] == last_succ[par_stem] else last_succ[stem]
        right = thread[u]
    parent[u_out] = par_stem
    thread[u] = last
    rev_thread[last] = u
    last_succ[u_out] = u
    if old_rev_thread != v_in:
        thread[old_rev_thread] = right
        rev_thread[right] = old_rev_thread
    for i in range(nd):
        x = dirty[i]
        rev_thread[thread[x]] = x

    tmp_sc = 0
    tmp_ls = last_succ[u_out]
    u = u_out
    while u != u_in:
        w = parent[u]
        pred[u] = pred[w]
        forward[u] = not forward[w]
        tmp_sc += succ_num[u] - succ_num[w]
        succ_num[u] = tmp_sc
        last_succ[w] = tmp_ls
        u = w
    pred[u_in] = in_arc
    forward[u_in] = u_in == src[in_arc]
    succ_num[u_in] = old_succ_num

    up_limit_in = -1
    up_limit_out = -1
    if last_succ[join] == v_in:
        up_limit_out = join
    else:
        up_limit_in = join
    u = v_in
    while u != up_limit_in and last_succ[u] == v_in:
        last_succ[u] = last_succ[u_out]
        u = parent[u]
    if join != old_rev_thread and v_in != old_rev_thread:
        u = v_out
        while u != up_limit_out and last_succ[u] == old_last_succ:
            last_succ[u] = old_rev_thread
            u = parent[u]
    else:
        u = v_out
        while u != up_limit_out and last_succ[u] == old_last_succ:
            last_succ[u] = last_succ[u_out]
            u = parent[u]
    u = v_in
    while u != join:
        succ_num[u] += old_succ_num
        u = parent[u]
    u = v_out
    while u != join:
        succ_num[u] -= old_succ_num
        u = parent[u]

    # updatePotential
    if forward[u_in]:
        sigma = pi[v_in] - pi[u_in] - cost[pred[u_in]]
    else:
        sigma = pi[v_in] - pi[u_in] + cost[pred[u_in]]
    end = thread[last_succ[u_in]]
    u = u_in
    while u != end:
        pi[u] += sigma
        u = thread[u]
    return True


@njit
def network_simplex(a, b, M, max_iter):
    """Return (G, alpha, beta, cost, status, n_iter). a and b must be positive.

    status: 1 optimal, 0 infeasible, -1 unbounded, 2 max_iter reached.
    """
    n1, n2 = a.size, b.size
    N = n1 + n2
    m = n1 * n2
    src = np.empty(m + N, np.int32)
    tgt = np.empty(m + N, np.int32)
    cost = np.empty(m + N, np.float64)
    flow = np.zeros(m + N, np.float64)
    state = np.ones(m + N, np.int8)
    supply = np.empty(N + 1, np.float64)
    for i in range(n1):
        supply[N - 1 - i] = a[i]
    for j in range(n2):
        supply[n2 - 1 - j] = -b[j]
    if abs(supply[:N].sum()) > EPS_FEAS:
        return np.zeros((n1, n2)), np.zeros(n1), np.zeros(n2), 0.0, 0, 0

    k = max(int(np.sqrt(m)), 10)
    block = max(int(np.sqrt(m + 0.0)), 10)
    p = 0
    q = 0
    max_c = 0.0
    best_c = np.full(n2, np.inf)
    best_arc = np.full(n2, -1, np.int64)
    for g in range(m - 1, -1, -1):
        i = g // n2
        j = g - i * n2
        c = M[i, j]
        src[p] = N - 1 - i
        tgt[p] = n2 - 1 - j
        cost[p] = c
        if c > max_c:
            max_c = c
        if c < best_c[j]:
            best_c[j] = c
            best_arc[j] = p
        p += k
        if p >= m:
            q += 1
            p = q

    ART = (max_c + 1.0) * N
    root = N
    pi = np.empty(N + 1, np.float64)
    parent = np.empty(N + 1, np.int32)
    pred = np.empty(N + 1, np.int64)
    forward = np.empty(N + 1, np.bool_)
    thread = np.empty(N + 1, np.int32)
    rev_thread = np.empty(N + 1, np.int32)
    succ_num = np.empty(N + 1, np.int32)
    last_succ = np.empty(N + 1, np.int32)
    dirty = np.empty(N + 2, np.int32)

    parent[root] = -1
    pred[root] = -1
    thread[root] = 0
    rev_thread[0] = root
    succ_num[root] = N + 1
    last_succ[root] = root - 1
    pi[root] = 0.0
    for u in range(N):
        e = m + u
        parent[u] = root
        pred[u] = e
        thread[u] = u + 1
        rev_thread[u + 1] = u
        succ_num[u] = 1
        last_succ[u] = u
        state[e] = TREE
        if supply[u] >= 0:
            forward[u] = True
            pi[u] = 0.0
            src[e], tgt[e] = u, root
            flow[e] = supply[u]
            cost[e] = 0.0
        else:
            forward[u] = False
            pi[u] = ART
            src[e], tgt[e] = root, u
            flow[e] = -supply[u]
            cost[e] = ART

    for j in range(n2):
        e = best_arc[j]
        if state[e] * (cost[e] + pi[src[e]] - pi[tgt[e]]) >= 0:
            continue
        if not _pivot(e, src, tgt, cost, flow, state, pi, parent, pred, forward,
                      thread, rev_thread, succ_num, last_succ, dirty):
            return np.zeros((n1, n2)), np.zeros(n1), np.zeros(n2), 0.0, -1, 0

    status = 1
    it = 0
    next_arc = 0
    in_arc = 0
    while True:
        found, in_arc, next_arc = _entering(next_arc, in_arc, m, block, src, tgt, cost, state, pi)
        if not found:
            break
        it += 1
        if it >= max_iter:
            status = 2
            break
        if not _pivot(in_arc, src, tgt, cost, flow, state, pi, parent, pred, forward,
                      thread, rev_thread, succ_num, last_succ, dirty):
            return np.zeros((n1, n2)), np.zeros(n1), np.zeros(n2), 0.0, -1, it

    if status == 1:
        for e in range(m, m + N):
            if abs(flow[e]) > EPS_FEAS:
                return np.zeros((n1, n2)), np.zeros(n1), np.zeros(n2), 0.0, 0, it

    G = np.zeros((n1, n2))
    total = 0.0
    for e in range(m):
        if flow[e] != 0.0:
            i = N - 1 - src[e]
            j = n2 - 1 - tgt[e]
            G[i, j] = flow[e]
            total += flow[e] * cost[e]
    alpha = np.empty(n1)
    beta = np.empty(n2)
    for i in range(n1):
        alpha[i] = -pi[N - 1 - i]
    for j in range(n2):
        beta[j] = pi[n2 - 1 - j]
    return G, alpha, beta, total, status, it


@njit
def emd(a, b, M, max_iter=100000000):
    """Exact OT. Zero-mass atoms are removed before the solve."""
    ia = np.flatnonzero(a)
    ib = np.flatnonzero(b)
    Ms = np.empty((ia.size, ib.size))
    for i in range(ia.size):
        for j in range(ib.size):
            Ms[i, j] = M[ia[i], ib[j]]
    Gs, al, be, total, status, it = network_simplex(a[ia], b[ib], Ms, max_iter)
    G = np.zeros(M.shape)
    alpha = np.zeros(a.size)
    beta = np.zeros(b.size)
    for i in range(ia.size):
        alpha[ia[i]] = al[i]
        for j in range(ib.size):
            G[ia[i], ib[j]] = Gs[i, j]
    for j in range(ib.size):
        beta[ib[j]] = be[j]
    return G, alpha, beta, total, status, it


@njit(parallel=True)
def emd_batch(A, B, Ms):
    k = Ms.shape[0]
    out = np.empty(k)
    for t in prange(k):
        out[t] = network_simplex(A[t], B[t], Ms[t], 100000000)[3]
    return out
