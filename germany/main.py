import itertools
import math

import numpy as np

D = 5
NAMES = ["x1", "x2", "x3", "x4", "x5"]
PAIRS = list(itertools.combinations(range(D), 2))
TOL = 1e-4


class _Stop(Exception):
    pass


def _features(X, pair):
    cols = [np.ones(len(X))] + [X[:, k] for k in range(D)]
    if pair is not None:
        cols.append(X[:, pair[0]] * X[:, pair[1]])
    return np.column_stack(cols)


def _ransac(X, y, pair, rng, trials=400):
    F = _features(X, pair)
    n, p = F.shape
    m = p + 2
    best = None
    for _ in range(trials):
        idx = rng.choice(n, m, replace=False)
        coef = np.linalg.lstsq(F[idx], y[idx], rcond=None)[0]
        inl = np.abs(F @ coef - y) < TOL
        c = int(inl.sum())
        if best is None or c > best[0]:
            best = (c, inl)
    inl = best[1]
    coef = np.linalg.lstsq(F[inl], y[inl], rcond=None)[0]
    for _ in range(3):
        inl = np.abs(F @ coef - y) < TOL
        if inl.sum() < p:
            break
        coef = np.linalg.lstsq(F[inl], y[inl], rcond=None)[0]
    inl = np.abs(F @ coef - y) < TOL
    return coef, inl


def _fit_smooth(X, y, rng):
    cands = []
    coef0, inl0 = _ransac(X, y, None, rng)
    cands.append((int(inl0.sum()), 0.0, None, coef0, inl0))
    for pair in PAIRS:
        coef, inl = _ransac(X, y, pair, rng)
        cands.append((int(inl.sum()), -abs(coef[-1]), pair, coef, inl))
    cands.sort(key=lambda t: (t[0], t[1]), reverse=True)
    top = cands[0]
    if cands[0][0] <= coef0.size + 0 or top[2] is None:
        pass
    none_model = next(c for c in cands if c[2] is None)
    if none_model[0] >= top[0]:
        top = none_model
    return top[2], top[3], top[4]


def _predict_smooth(X, pair, coef):
    return _features(np.atleast_2d(X), pair) @ coef


def _fit_plane(B, idx):
    sub = B[:, idx]
    mu = sub.mean(axis=0)
    u, s, vt = np.linalg.svd(sub - mu)
    a = vt[-1]
    t = float(a @ mu)
    res = np.abs(sub @ a - t)
    full = np.zeros(D)
    full[list(idx)] = a
    return full, t, float(res.max())


def _solve(oracle, points):
    rng = np.random.default_rng(20261114)
    P = np.array(points, dtype=float)
    pool = {}

    def ask(pts, phase):
        pts = [np.clip(np.asarray(p, float), 0.0, 1.0) for p in pts]
        fresh = {oracle.key(p) for p in pts if oracle.key(p) not in oracle.cache}
        if len(fresh) > oracle.remaining - 2:
            raise _Stop()
        ys = oracle.ask_many([[float(v) for v in p] for p in pts], phase=phase)
        for p, yv in zip(pts, ys):
            pool[tuple(round(float(v), 9) for v in p)] = float(yv)
        return ys

    def arrays():
        keys = list(pool.keys())
        return np.array(keys, float), np.array([pool[k] for k in keys], float)

    verts = [list(v) for v in itertools.product([0.0, 1.0], repeat=D) if sum(v) % 2 == 0]
    interior = [list(rng.uniform(0.05, 0.95, D)) for _ in range(26)]
    ask(verts + [[0.5] * D] + interior, "design")

    def fit_all():
        X, y = arrays()
        mid = np.array([q.min() > 0.01 and q.max() < 0.99 for q in X])
        pair, coef, _ = _fit_smooth(X[mid], y[mid], rng)
        res = y - _predict_smooth(X, pair, coef)
        good = np.abs(res) < TOL
        F = _features(X, pair)
        if good.sum() >= F.shape[1] + 2:
            coef = np.linalg.lstsq(F[good], y[good], rcond=None)[0]
            res = y - F @ coef
        return X, y, pair, coef, res

    X, y, pair, coef, resid = fit_all()
    inl = np.abs(resid) <= 1e-3
    out = ~inl

    tries = 0
    while not out.any() and oracle.remaining > 60 and tries < 40:
        extra = [list(np.where(rng.random(D) < 0.5, rng.uniform(0.0, 0.25, D), rng.uniform(0.75, 1.0, D))) for _ in range(5)]
        ask(extra, "search_for_jump")
        X, y, pair, coef, resid = fit_all()
        inl = np.abs(resid) <= 1e-3
        out = ~inl
        tries += 1

    sub = {"relevant_variables": [], "irrelevant_variables": [], "interactions": [],
           "discontinuities": [], "model_expression": "", "predictions": [], "notes": ""}

    if not out.any():
        pred = _predict_smooth(P, pair, coef)
        return _finish(sub, pair, coef, None, None, None, None, P, pred, [], oracle,
                       "no jump found in the budget spent")

    J = float(np.median(resid[out]))
    sgn = 1.0 if J > 0 else -1.0

    def is_active(x, yv):
        r = yv - float(_predict_smooth(x, pair, coef)[0])
        return r * sgn > abs(J) / 2.0

    act_pts = [X[i] for i in np.where(out)[0]]
    ina_pts = [X[i] for i in np.where(inl)[0]]

    def bisect(u, v, steps):
        lo, hi = 0.0, 1.0
        for _ in range(steps):
            if oracle.remaining < 1:
                break
            mid = (lo + hi) / 2.0
            x = u + mid * (v - u)
            yv = ask([x], "bisect")[0]
            if is_active(x, yv):
                hi = mid
            else:
                lo = mid
        return u + (lo + hi) / 2.0 * (v - u), u + lo * (v - u)

    interior_ina = [q for q in ina_pts if q.min() > 0.02 and q.max() < 0.98]

    def pick_inactive():
        src = interior_ina if interior_ina else ina_pts
        return src[int(rng.integers(len(src)))]

    def steps_for(u, v, res=0.003):
        length = float(np.linalg.norm(v - u))
        return max(6, int(math.ceil(math.log2(max(length, 1e-6) / res))))

    flips = {}
    seen_room = [[False, False] for _ in range(D)]
    for attempt in range(3):
        if oracle.remaining < 40:
            break
        v = act_pts[int(rng.integers(len(act_pts)))]
        u = pick_inactive()
        est, xlo = bisect(u, v, steps_for(u, v))
        for k in range(D):
            for d, sg in enumerate((1.0, -1.0)):
                room = (1.0 - xlo[k]) if sg > 0 else xlo[k]
                m = min(0.25, room)
                if m < 0.02:
                    continue
                x = xlo.copy()
                x[k] += sg * m
                yv = ask([x], "flip_test")[0]
                seen_room[k][d] = True
                if is_active(x, yv):
                    flips[k] = sg
        if len(flips) == 2:
            break

    plane = None
    if len(flips) == 2:
        gi, gj = sorted(flips)
        si = 1.0 if flips[gi] > 0 else 0.0
        sj = 1.0 if flips[gj] > 0 else 0.0
        Xall, yall = arrays()
        corner = {}
        for i in range(len(yall)):
            x = Xall[i]
            if x[gi] in (0.0, 1.0) and x[gj] in (0.0, 1.0):
                corner[(x[gi], x[gj])] = is_active(x, yall[i])
        act_c = [c for c, v_ in corner.items() if v_]
        ina_c = [c for c, v_ in corner.items() if not v_]
        base = np.full(D, 0.5)

        def lift(c):
            x = base.copy()
            x[gi], x[gj] = c
            return x

        pairs_ = []
        for ca in act_c:
            for ci in ina_c:
                adj = (ca[0] == ci[0]) != (ca[1] == ci[1])
                pairs_.append((0 if adj else 1, ca, ci))
        pairs_.sort(key=lambda p_: p_[0])
        B2 = []
        for _, ca, ci in pairs_[:3]:
            if oracle.remaining < 12:
                break
            est, _xlo = bisect(lift(ci), lift(ca), 9)
            B2.append(est[[gi, gj]])
        if len(B2) >= 2:
            B2 = np.array(B2)
            mu = B2.mean(axis=0)
            nrm = np.linalg.svd(B2 - mu)[2][-1]
            tt = float(nrm @ mu)
            err = float(np.abs((B2 - mu) @ nrm).max())
            ca = act_c[0]
            if float(nrm @ np.array(ca)) < tt:
                nrm, tt = -nrm, -tt
            a_ = np.zeros(D)
            a_[gi], a_[gj] = nrm
            plane = (a_, tt, (gi, gj), err)

    ok = False
    if plane is not None and plane[3] < 0.012:
        a, t, idx, _err = plane
        mism = 0
        tests = 0
        for _ in range(10):
            if oracle.remaining < 4:
                break
            x = rng.uniform(0.05, 0.95, D)
            gi, gj = idx
            cur = float(a @ x - t)
            x = x - (cur - rng.choice([-0.03, 0.03])) * a / float(a @ a)
            if x.min() < 0.0 or x.max() > 1.0:
                continue
            yv = ask([x], "verify")[0]
            tests += 1
            pred_active = float(a @ x - t) >= 0.0
            if pred_active != is_active(x, yv):
                mism += 1
        ok = mism == 0

    if not ok:
        boundary = []
        order = rng.permutation(len(act_pts))
        s = 0
        while oracle.remaining >= 12 and s < 10:
            v = act_pts[order[s % len(act_pts)]]
            u = pick_inactive()
            est, _xlo = bisect(u, v, steps_for(u, v))
            boundary.append(est)
            s += 1
        B = np.array(boundary)
        Xall, yall = arrays()
        best = None
        for k in range(2, D + 1):
            stats = []
            for idx in itertools.combinations(range(D), k):
                a_, t_, err = _fit_plane(B, idx)
                stats.append((err, idx, a_, t_))
            stats.sort(key=lambda s_: s_[0])
            err, idx, a_, t_ = stats[0]
            if err < 0.012:
                best = (a_, t_, idx)
                break
        if best is None:
            a_, t_, err = _fit_plane(B, tuple(range(D)))
            best = (a_, t_, tuple(range(D)))
        a, t, idx = best
        labels = np.array([is_active(Xall[i], yall[i]) for i in range(len(yall))])
        margin = Xall @ a - t
        if (margin[labels] < 0).mean() > 0.5:
            a, t = -a, -t

    scale = float(np.abs(a).max())
    a, t = a / scale, t / scale

    smooth = _predict_smooth(P, pair, coef)
    gate = (P @ a) >= t
    pred = smooth + J * gate
    return _finish(sub, pair, coef, J, a, t, idx, P, pred, [], oracle,
                   "located the switching surface by bisection")


def _num(v):
    return f"{v:+.6f}"


def _finish(sub, pair, coef, J, a, t, gate_idx, P, pred, boundary, oracle, how):
    lin = coef[1:1 + D]
    relevant = set(k for k in range(D) if abs(lin[k]) > 1e-3)
    inter = []
    terms = [f"{coef[0]:.6f}"]
    for k in range(D):
        if abs(lin[k]) > 1e-3:
            terms.append(f"{_num(lin[k])}*{NAMES[k]}")
    if pair is not None and abs(coef[-1]) > 1e-3:
        relevant.update(pair)
        inter.append({"variables": [NAMES[pair[0]], NAMES[pair[1]]], "type": "product",
                      "coefficient": float(coef[-1])})
        terms.append(f"{_num(coef[-1])}*{NAMES[pair[0]]}*{NAMES[pair[1]]}")
    disc = []
    if J is not None:
        gv = sorted(gate_idx, key=lambda k: -abs(a[k]))[:2] if len(gate_idx) > 2 else list(gate_idx)
        gv = sorted(gv)
        relevant.update(gv)
        cond = " + ".join(f"{a[k]:.6f}*{NAMES[k]}" for k in range(D) if abs(a[k]) > 1e-9)
        disc.append({"gate_variables": [NAMES[k] for k in gv],
                     "boundary": {"form": "linear",
                                  "coefficients": {NAMES[k]: float(a[k]) for k in range(D)},
                                  "threshold": float(t), "active_when": "greater_equal"},
                     "jump": float(abs(J)), "condition": f"{cond} >= {t:.6f}"})
        terms.append(f"({J:.6f} if {cond} >= {t:.6f} else 0.0)")
    sub["relevant_variables"] = [NAMES[k] for k in sorted(relevant)]
    sub["irrelevant_variables"] = [NAMES[k] for k in range(D) if k not in relevant]
    sub["interactions"] = inter
    sub["discontinuities"] = disc
    sub["model_expression"] = " + ".join(terms).replace("+ -", "- ")
    sub["predictions"] = [float(v) for v in pred]
    sub["notes"] = (f"Queried 16 cube corners, the centre and 26 interior points; fitted a robust model on the interior points "
                    f"(linear terms plus one product term) and read the jump from the points that deviate by "
                    f"a constant. Then {how}; {oracle.spent} queries used in total.")
    return sub


def _emergency(oracle, points):
    P = np.array(points, dtype=float)
    keys = list(oracle.cache.keys())
    X = np.array(keys, float)
    y = np.array([oracle.cache[k] for k in keys], float)
    rng = np.random.default_rng(1)
    pair, coef, inl = _fit_smooth(X, y, rng)
    resid = y - _predict_smooth(X, pair, coef)
    out = np.abs(resid) > 1e-3
    sub = {}
    if not out.any() or out.all():
        pred = _predict_smooth(P, pair, coef)
        return _finish(sub, pair, coef, None, None, None, None, P, pred, [], oracle, "stopped early")
    J = float(np.median(resid[out]))
    a = X[out].mean(axis=0) - X[~out].mean(axis=0)
    a = a / float(np.abs(a).max())
    t = float((a @ X[out].min(axis=0) * 0 + np.max(X[~out] @ a) + np.min(X[out] @ a)) / 2.0)
    pred = _predict_smooth(P, pair, coef) + J * ((P @ a) >= t)
    return _finish(sub, pair, coef, J, a, t, tuple(range(D)), P, pred, [], oracle, "stopped early on budget guard")


def solve(oracle, points):
    try:
        return _solve(oracle, points)
    except _Stop:
        pass
    except Exception as exc:
        print("solve failed, using fallback:", repr(exc))
    try:
        return _emergency(oracle, points)
    except Exception as exc:
        print("fallback failed:", repr(exc))
        mean = float(np.mean(list(oracle.cache.values()))) if oracle.cache else 0.0
        return {"relevant_variables": [], "irrelevant_variables": list(NAMES), "interactions": [],
                "discontinuities": [], "model_expression": f"{mean:.6f}",
                "predictions": [mean] * len(points), "notes": "fallback"}
