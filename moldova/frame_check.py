import collections
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, UNREAD, load_reads, load_register
from mask_model import LABEL, assign_by_signature, ring_table, xor

CANDIDATE = "RHGLWQ"
SOUTH_LAT = 46.45

def letter_profile(read_ring_pairs):
    c = collections.Counter(LABEL[a] ^ LABEL[b] for r, t in read_ring_pairs for a, b in zip(r, t) if a != UNREAD)
    n = sum(c.values())
    return np.array([c[k] for k in range(4)], float), n

def main():
    reg = load_register()
    reads = assign_by_signature(load_reads(), reg)
    known = reads[reads.cands.map(len) == 1].assign(ring=lambda d: d.cands.str[0])
    known = known.merge(reg[["ring", "ringed_latitude", "ringed_longitude"]], on="ring")
    known["site"] = list(zip(known.ringed_latitude.round(3), known.ringed_longitude.round(3)))
    cand = pd.read_csv(OUT / "stork_candidates.csv", dtype=str)
    cahul = reads[reads.record_id.astype(str).isin(cand.record_id)].ring_read.tolist()

    sites = {}
    for site, d in known.groupby("site"):
        cnt, n = letter_profile(zip(d.ring_read, d.ring))
        if n >= 60:
            sites[site] = (cnt + 0.5) / (n + 2)
    lls = []
    for m in range(4):
        truth = "".join(xor(c, m) for c in CANDIDATE)
        cnt, n = letter_profile((r, truth) for r in cahul)
        best = max(sites.items(), key=lambda kv: float((cnt * np.log(kv[1])).sum()))
        ll = float((cnt * np.log(best[1])).sum())
        lls.append(ll)
        print(f"mask {m}: кольцо {truth}, профиль Кагула {np.round(cnt / n, 3)}, "
              f"больше всего похоже на место {best[0]} (log-lik {ll:.1f})")

    rings = ring_table(known, reg)
    rings = rings[rings.n_reads >= 5]
    rings["band"] = pd.cut(rings.lat, [45, SOUTH_LAT, 46.75, 47.05, 47.35, 49])
    print("\nмоды известных колец по широте:")
    print(pd.crosstab(rings.band, rings["mode"]).to_string())
    south = rings[rings.lat < SOUTH_LAT]
    prior = np.bincount(south["mode"], minlength=4) + 0.5
    prior = prior / prior.sum()

    post = prior * np.exp(np.array(lls) - max(lls))
    post = post / post.sum()
    print(f"\nprior (кольца южнее {SOUTH_LAT}N, n={len(south)}) и апостериорная вероятность:")
    for m in range(4):
        print(f"  {''.join(xor(c, m) for c in CANDIDATE)}  prior {prior[m]:.3f}  posterior {post[m]:.3f}")

    clean_places_by_latitude(known)

def clean_places_by_latitude(known: pd.DataFrame, thr: float = 0.75, min_letters: int = 20):
    print(f"\nчистые места (>= {thr:.0%} букв на одной маске, >= {min_letters} букв): какая у них главная маска")
    for key in ["ring", "locality_id"]:
        rows = []
        for k, d in known.groupby(key):
            c, n = letter_profile(zip(d.ring_read, d.ring))
            if n >= min_letters and c.max() / n >= thr:
                rows.append((int(c.argmax()), d.ringed_latitude.mean()))
        t = pd.DataFrame(rows, columns=["top", "lat"])
        for name, sub in [(f"южнее {SOUTH_LAT}N", t[t.lat < SOUTH_LAT]), ("севернее 46.6N", t[t.lat >= 46.6])]:
            v = sub.top.value_counts().reindex(range(4), fill_value=0).tolist()
            k0 = v[0]
            print(f"  {key:11s} {name:15s}: {len(sub):3d} шт., маски 0/1/2/3 = {v}, верное чтение у {k0}/{len(sub)}"
                  f"  (rule of succession {(k0 + 1) / (len(sub) + 2):.2f})")

if __name__ == "__main__":
    main()
