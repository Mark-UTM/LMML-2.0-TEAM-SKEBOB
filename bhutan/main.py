import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
CANDIDATES = 10
RIDGE = 1.0
TIME_BUDGET = 9.0


def find_input_dir():
    for base in (HERE, Path.cwd()):
        folder = base / "input"
        if folder.is_dir():
            return folder
    raise SystemExit("input folder not found")


def load_strips(folder):
    names = sorted(p.name for p in folder.iterdir() if p.suffix.lower() == ".png")
    strips = []
    for name in names:
        im = cv2.imread(str(folder / name), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise SystemExit(f"cannot read {name}")
        if im.ndim == 2:
            im = np.dstack([im] * 3)
        strips.append(im)
    return names, strips


def cheap_cost(rgb):
    n, h, _, _ = rgb.shape
    k = h * 3
    r1, r2, l1, l2 = rgb[:, :, -1], rgb[:, :, -2], rgb[:, :, 0], rgb[:, :, 1]
    r1f, l1f = r1.reshape(n, k), l1.reshape(n, k)
    d = cdist((2 * r1 - r2).reshape(n, k), l1f, "cityblock")
    d += cdist(r1f, (2 * l1 - l2).reshape(n, k), "cityblock")
    d += 0.5 * cdist(r1f, l1f, "cityblock")
    d /= k
    np.fill_diagonal(d, np.inf)
    return d


def gradient_side(a1, a2, b, pairs):
    n, h, _ = a1.shape
    out = np.full((n, n), np.nan)
    for a in range(n):
        js = np.where(pairs[a])[0]
        if len(js) == 0:
            continue
        g = a1[a] - a2[a]
        mu = g.mean(0)
        cov = np.cov(g.T) + np.eye(3) * RIDGE
        whiten = np.linalg.cholesky(np.linalg.inv(cov))
        e = (b[js] - (a1[a] + mu)[None]).reshape(len(js) * h, 3) @ whiten
        out[a, js] = np.sqrt((e * e).sum(1)).reshape(len(js), h).sum(1)
    return out


def edge_cost(strips):
    rgb = np.stack([s[:, :, :3].astype(np.float32) for s in strips])
    n, h, _, _ = rgb.shape
    r1, r2, l1, l2 = rgb[:, :, -1], rgb[:, :, -2], rgb[:, :, 0], rgb[:, :, 1]
    d0 = cheap_cost(rgb)
    k = min(CANDIDATES, n - 1)
    mask = np.zeros((n, n), bool)
    rows = np.argsort(d0, axis=1)[:, :k]
    cols = np.argsort(d0, axis=0)[:k, :]
    mask[np.repeat(np.arange(n), k), rows.ravel()] = True
    mask[cols.ravel(), np.tile(np.arange(n), k)] = True
    np.fill_diagonal(mask, False)
    d = (gradient_side(r1, r2, l1, mask) + gradient_side(l1, l2, r1, mask.T).T) / (h * 3)
    finite = d0[np.isfinite(d0)]
    fallback = np.nanmax(d) * 3 + d0 / max(finite.min(), 1e-6)
    d = np.where(mask, d, fallback)
    np.fill_diagonal(d, np.inf)
    ds = np.where(np.isfinite(d), d, np.inf)
    best_row = np.sort(ds, axis=1)[:, :2].mean(1)
    best_col = np.sort(ds, axis=0)[:2, :].mean(0)
    d = d / np.sqrt(best_row[:, None] * best_col[None, :] + 1e-9)
    np.fill_diagonal(d, np.inf)
    return d


def greedy_chain(d):
    n = len(d)
    order = np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]
    nxt = [-1] * n
    prv = [-1] * n
    head = list(range(n))

    def root(a):
        while head[a] != a:
            head[a] = head[head[a]]
            a = head[a]
        return a

    edges = 0
    for i, j in order:
        if edges == n - 1:
            break
        if not np.isfinite(d[i, j]) or nxt[i] != -1 or prv[j] != -1:
            continue
        ri, rj = root(i), root(j)
        if ri == rj:
            continue
        nxt[i], prv[j] = j, i
        head[rj] = ri
        edges += 1
    start = next(i for i in range(n) if prv[i] == -1)
    chain = [start]
    while nxt[chain[-1]] != -1:
        chain.append(nxt[chain[-1]])
    return chain


def path_cost(chain, d):
    return float(sum(d[a, b] for a, b in zip(chain[:-1], chain[1:])))


def refine(chain, d, deadline):
    n = len(chain)
    pad = np.zeros((n + 1, n + 1))
    pad[:n, :n] = np.where(np.isfinite(d), d, 1e9)
    ch = np.array(chain)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for seg in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
            if seg >= n:
                break
            i = 0
            while i <= n - seg:
                if time.time() > deadline:
                    return [int(x) for x in ch]
                block = ch[i:i + seg]
                rest = np.concatenate([ch[:i], ch[i + seg:]])
                a = rest[i - 1] if i > 0 else n
                b = rest[i] if i < len(rest) else n
                removed = pad[a, block[0]] + pad[block[-1], b] - pad[a, b]
                pp = np.concatenate([[n], rest])
                qq = np.concatenate([rest, [n]])
                inner_fwd = pad[block[:-1], block[1:]].sum()
                inner_rev = pad[block[1:], block[:-1]].sum()
                add_f = pad[pp, block[0]] + pad[block[-1], qq] - pad[pp, qq]
                add_r = pad[pp, block[-1]] + pad[block[0], qq] - pad[pp, qq]
                gain_f = removed - add_f
                gain_f[i] = -1.0
                gain_r = removed - add_r + inner_fwd - inner_rev
                kf, kr = int(np.argmax(gain_f)), int(np.argmax(gain_r))
                if max(gain_f[kf], gain_r[kr]) > 1e-9:
                    if gain_f[kf] >= gain_r[kr]:
                        ch = np.concatenate([rest[:kf], block, rest[kf:]])
                    else:
                        ch = np.concatenate([rest[:kr], block[::-1], rest[kr:]])
                    improved = True
                i += 1
    return [int(x) for x in ch]


def orient(chain, strips):
    if len(chain) < 2:
        return chain
    half = len(chain) // 2
    gray = [cv2.cvtColor(strips[i][:, :, :3], cv2.COLOR_BGR2GRAY).mean() for i in chain]
    return chain if np.mean(gray[:half]) <= np.mean(gray[len(chain) - half:]) else chain[::-1]


def main():
    t0 = time.time()
    names, strips = load_strips(find_input_dir())
    chain = list(range(len(strips)))
    if len(strips) > 2:
        d = edge_cost(strips)
        chain = refine(greedy_chain(d), d, t0 + TIME_BUDGET)
    chain = orient(chain, strips)
    image = np.concatenate([strips[i] for i in chain], axis=1)
    cv2.imwrite(str(HERE / "reconstructed_image.png"), image)
    with open(HERE / "reconstructed_order.json", "w") as f:
        json.dump([names[i] for i in chain], f, indent=4)
    print(f"{len(names)} strips in {time.time() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
