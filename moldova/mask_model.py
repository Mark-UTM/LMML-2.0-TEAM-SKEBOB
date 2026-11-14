import collections
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, UNREAD, km, load_reads, load_register

GROUPS = ["AKRX", "BGZS", "CODQ", "EFLP", "HWNM", "IYJT"]
GROUP = {c: g for g, s in enumerate(GROUPS) for c in s}
LABEL = {c: i for s in GROUPS for i, c in enumerate(s)}

def xor(letter: str, d: int) -> str:
    return GROUPS[GROUP[letter]][LABEL[letter] ^ d]

def signature(s: str) -> str:
    return "".join("?" if c == UNREAD else str(GROUP[c]) for c in s)

def assign_by_signature(reads: pd.DataFrame, reg: pd.DataFrame) -> pd.DataFrame:
    sigs = list(zip(reg.ring, reg.ring.map(signature)))

    def cands(s):
        sg = signature(s)
        return [r for r, g in sigs if all(a == "?" or a == b for a, b in zip(sg, g))]

    reads = reads.copy()
    reads["cands"] = reads.ring_read.map(cands)
    return reads

def deviations(read: str, ring: str):
    return [LABEL[a] ^ LABEL[b] for a, b in zip(read, ring) if a != UNREAD]

def ring_table(known: pd.DataFrame, reg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ring, d in known.groupby("ring"):
        c = collections.Counter(x for r in d.ring_read for x in deviations(r, ring))
        r = reg.loc[reg.ring == ring].iloc[0]
        rows.append(dict(ring=ring, n_reads=len(d), mode=max(range(4), key=lambda k: c[k]),
                         lat=r.ringed_latitude, lon=r.ringed_longitude, counts=[c[k] for k in range(4)]))
    return pd.DataFrame(rows)

def fit_Q(rings: pd.DataFrame) -> np.ndarray:
    Q = np.ones((4, 4))
    for m, cnt in zip(rings["mode"], rings["counts"]):
        Q[m] += cnt
    return Q / Q.sum(axis=1, keepdims=True)

def spatial_prior(lat, lon, rings: pd.DataFrame, k: int = 10, scale_km: float = 30.0, smooth: float = 0.5) -> np.ndarray:
    d = km(lat, lon, rings.lat.values, rings.lon.values)
    idx = np.argsort(d)[:k]
    w = np.exp(-d[idx] / scale_km)
    p = np.full(4, smooth)
    for i, wi in zip(idx, w):
        p[rings["mode"].iloc[i]] += wi
    return p / p.sum()

def decode(reads_list, sig: str, Q: np.ndarray, prior: np.ndarray):
    ll = np.zeros((4, 6, 4))
    for r in reads_list:
        for pos, a in enumerate(r):
            if a == UNREAD:
                continue
            for m in range(4):
                for t in range(4):
                    ll[m, pos, t] += np.log(Q[m][LABEL[a] ^ t])
    logp = {}
    for labels in itertools.product(range(4), repeat=6):
        per_m = [np.log(prior[m]) + sum(ll[m, p, labels[p]] for p in range(6)) for m in range(4)]
        logp[labels] = np.logaddexp.reduce(per_m)
    z = np.logaddexp.reduce(list(logp.values()))
    out = sorted(((np.exp(v - z), "".join(GROUPS[int(sig[p])][labels[p]] for p in range(6))) for labels, v in logp.items()),
                 reverse=True)
    return out

def majority(reads_list):
    s = ""
    for pos in range(6):
        c = collections.Counter(r[pos] for r in reads_list if r[pos] != UNREAD)
        s += c.most_common(1)[0][0] if c else UNREAD
    return s

def main():
    reg = load_register()
    reads = assign_by_signature(load_reads(), reg)
    known = reads[reads.cands.map(len) == 1].assign(ring=lambda d: d.cands.str[0])
    stork = reads[reads.cands.map(len) == 0]
    print(f"чтений с однозначной подписью: {len(known)}, без совпадения в реестре: {len(stork)} "
          f"(виды: {stork.species.value_counts().to_dict()})")

    rings = ring_table(known, reg)
    print("моды колец (по всем известным):", rings["mode"].value_counts().sort_index().to_dict())

    test = rings[rings.n_reads >= 5].reset_index(drop=True)
    res = []
    for i, row in test.iterrows():
        others = rings[rings.ring != row.ring]
        Q = fit_Q(others)
        rl = known[known.ring == row.ring].ring_read.tolist()
        sig = signature(row.ring)
        p_sp = spatial_prior(row.lat, row.lon, others)
        p_un = np.full(4, 0.25)
        p_glob = np.bincount(others["mode"], minlength=4) + 0.5
        p_glob = p_glob / p_glob.sum()
        best_sp = decode(rl, sig, Q, p_sp)[0]
        best_un = decode(rl, sig, Q, p_un)[0]
        best_gl = decode(rl, sig, Q, p_glob)[0]
        res.append(dict(ring=row.ring, n=row.n_reads, mode=row["mode"],
                        majority=majority(rl) == row.ring,
                        uniform=best_un[1] == row.ring, glob=best_gl[1] == row.ring,
                        spatial=best_sp[1] == row.ring, p_spatial=best_sp[0],
                        prior0=p_sp[row["mode"]]))
    R = pd.DataFrame(res)
    R.to_csv(OUT / "mask_model_check_known_rings.csv", index=False)
    print(f"\nleave-one-ring-out, колец с >=5 чтениями: {len(R)}")
    for c in ["majority", "uniform", "glob", "spatial"]:
        print(f"  {c:9s} точно восстановлено колец: {R[c].mean():.1%}")
    print("  по моде кольца (доля верных, spatial / majority):")
    for m, g in R.groupby("mode"):
        print(f"    mode {m}: колец {len(g):3d}  spatial {g.spatial.mean():.1%}  majority {g.majority.mean():.1%}")
    conf = R[R.p_spatial >= 0.9]
    print(f"  когда модель уверена (p>=0.9): колец {len(conf)}, верно {conf.spatial.mean():.1%}")

    Q = fit_Q(rings)
    cahul = stork.ring_read.tolist()
    sig = collections.Counter(signature(r) for r in cahul)
    full_sig = "".join(collections.Counter(s[p] for s in sig.elements() if s[p] != "?").most_common(1)[0][0] for p in range(6))
    lat, lon = stork.latitude.median(), stork.longitude.median()
    print(f"\nаист: {len(cahul)} чтений, подпись групп {full_sig}, точка {lat:.4f}, {lon:.4f}")
    print("  ближайшие известные кольца:",
          rings.assign(d=km(lat, lon, rings.lat.values, rings.lon.values)).nsmallest(8, "d")[["ring", "mode", "d"]]
          .round(1).values.tolist())
    for name, prior in [("spatial", spatial_prior(lat, lon, rings)),
                        ("global", (np.bincount(rings["mode"], minlength=4) + .5) / (len(rings) + 2)),
                        ("uniform", np.full(4, .25))]:
        post = decode(cahul, full_sig, Q, prior)
        print(f"  prior {name:8s} {np.round(prior, 2)} -> " + ", ".join(f"{s} {p:.3f}" for p, s in post[:4]))
    print("  голосование:", majority(cahul))

if __name__ == "__main__":
    main()
