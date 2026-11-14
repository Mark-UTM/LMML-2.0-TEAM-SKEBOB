import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, load_reads, load_register
from mask_model import assign_by_signature, deviations

NAMES = ["RHGLWQ", "XWBPHD", "ANSEMO", "KMZFNC"]

def dm_loglik(c, alpha):
    a0, n = alpha.sum(), c.sum()
    return math.lgamma(a0) - math.lgamma(a0 + n) + sum(math.lgamma(a + x) - math.lgamma(a) for a, x in zip(alpha, c))

def fit_alpha(counts):
    pooled = np.sum(counts, axis=0) + 0.5
    mu = pooled / pooled.sum()
    best = max((sum(dm_loglik(c, s * mu) for c in counts), s) for s in np.geomspace(1, 3000, 60))
    return best[1] * mu

def frame_posterior(c, types, prior=None):
    ll = []
    for m in range(4):
        if m not in types:
            ll.append(-np.inf)
            continue
        cm = np.array([c[k ^ m] for k in range(4)])
        ll.append(dm_loglik(cm, types[m]))
    ll = np.array(ll)
    if prior is not None:
        ll = ll + np.log(prior)
    p = np.exp(ll - ll.max())
    return p / p.sum()

def main():
    reg = load_register()
    reads = assign_by_signature(load_reads(), reg)
    known = reads[reads.cands.map(len) == 1].assign(ring=lambda d: d.cands.str[0])
    rows = []
    for loc, g in known.groupby("locality_id"):
        c = np.bincount([x for r, t in zip(g.ring_read, g.ring) for x in deviations(r, t)], minlength=4)
        if c.sum() >= 20 and c.max() / c.sum() >= 0.75:
            rows.append(dict(loc=loc, lat=g.latitude.mean(), mode=int(c.argmax()), c=c))
    T = pd.DataFrame(rows)
    print("чистые точки по типу:", T["mode"].value_counts().sort_index().to_dict())

    def types_from(df):
        return {m: fit_alpha(list(g.c)) for m, g in df.groupby("mode") if len(g) >= 3}

    types = types_from(T)
    for m, a in types.items():
        print(f"  тип {m}: средний профиль {np.round(a / a.sum(), 3)}, концентрация {a.sum():.0f}")

    ok, conf = [], []
    for i, row in T.iterrows():
        tp = types_from(T.drop(i))
        p = frame_posterior(row.c, tp)
        ok.append(p.argmax() == 0)
        conf.append(p[0])
    T["ok"], T["p_true"] = ok, conf
    print(f"\nвалидация на {len(T)} чистых точках: верная рамка выбрана в {np.mean(ok):.1%}")
    for m, g in T.groupby("mode"):
        print(f"  тип {m}: {g.ok.mean():.1%} из {len(g)}, средняя p(верной) {g.p_true.mean():.2f}")

    cand = pd.read_csv(OUT / "stork_candidates.csv", dtype=str)
    cah = reads[reads.record_id.astype(str).isin(cand.record_id)]
    cc = np.bincount([x for r in cah.ring_read for x in deviations(r, "RHGLWQ")], minlength=4)
    print(f"\nКагул, отклонения относительно RHGLWQ: {cc.tolist()}")
    p = frame_posterior(cc, types)
    print("  только состав ошибок (равный prior):", "  ".join(f"{n} {v:.3f}" for n, v in zip(NAMES, p)))
    south = T[T.lat < 46.45]
    pr = np.array([(south["mode"] == m).sum() + 0.5 for m in range(4)])
    pr = pr / pr.sum()
    p2 = frame_posterior(cc, types, prior=pr)
    print(f"  + prior по чистым точкам юга {np.round(pr, 3)}:", "  ".join(f"{n} {v:.3f}" for n, v in zip(NAMES, p2)))

    t0 = T[T["mode"] == 0]
    shifted = T[T["mode"] != 0]
    p_cahul = p[0]
    fa_strong = ((t0.p_true <= p_cahul * 2).sum() + 1) / (len(t0) + 2)
    fa_any = ((t0.p_true < 0.5).sum() + 1) / (len(t0) + 2)
    hit = ((shifted.p_true >= 0.5).sum() + 1) / (len(shifted) + 2)
    prior0 = (south["mode"] == 0).sum() + 1
    prior0 = prior0 / (len(south) + 2)
    print(f"\nкалибровка по известным точкам: ложная тревога на верной точке {fa_any:.2f} "
          f"(не слабее Кагула: {fa_strong:.2f}), срабатывание на сдвинутой {hit:.2f}; prior юга за верную {prior0:.2f}")
    for name, fa in [("учитывая силу сигнала", fa_strong), ("только «сработал / нет»", fa_any)]:
        post0 = prior0 * fa / (prior0 * fa + (1 - prior0) * hit)
        print(f"  P(RHGLWQ) {name}: {post0:.2f}; остальное делят XWBPHD и ANSEMO")
    T.drop(columns="c").to_csv(OUT / "noise_test_validation.csv", index=False)

if __name__ == "__main__":
    main()
