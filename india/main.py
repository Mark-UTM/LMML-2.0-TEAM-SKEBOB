import csv
import os
import zipfile
from multiprocessing import Pool

import cv2
import numpy as np

ARCHIVE = "test-input.zip"
OUTPUT = "output.csv"
SIDE = 384
SMALL = 192
NEIGHBOURS = 20
THRESHOLD = 0.5
MODEL = {"names": ["good", "inl", "ratio", "cover", "scale", "ang", "ar", "tx", "ty", "res", "mad", "mad_p90", "mad_b", "mad_g", "mad_r", "ncc", "ncc_e", "dmean", "dstd", "dh", "ds", "dv", "sat_a", "sat_b", "szratio", "nka", "nkb"], "mean": [137.25575, 114.58877, 0.64819, 0.88634, -6.54888, 16.6025, 0.16, 0.19241, 0.21075, 0.39415, 41.25695, 79.94515, 42.31253, 39.65158, 41.80673, 0.59135, 0.47744, 30.71389, 21.25107, 29.80713, 42.55484, 44.03198, 84.62669, 84.08146, 0.05569, 531.37213, 489.95971], "scale": [139.48168, 142.49558, 0.24685, 0.15784, 14.32926, 40.54236, 0.25809, 0.26017, 0.2776, 0.4091, 33.59631, 60.75135, 37.8837, 33.35128, 33.76687, 0.40406, 0.38638, 32.89091, 24.08123, 27.13671, 35.8376, 34.54379, 37.73989, 48.7108, 0.22978, 115.79457, 153.11021], "coef": [0.2774, 1.0636, 0.7387, -1.3291, 0.4185, 0.6583, -6.5389, -0.566, -0.5472, -2.9285, -0.417, -1.6384, -0.2599, -0.4392, -0.5195, 2.4576, -0.3996, 1.3713, 1.6476, -0.5054, -1.2538, 0.1177, 0.5132, -0.4341, 0.0791, -0.4258, -0.7123], "bias": -9.2998}

IMAGES = None


def read_archive():
    items = []
    with zipfile.ZipFile(ARCHIVE) as z:
        for info in sorted(z.infolist(), key=lambda i: i.filename):
            name = os.path.basename(info.filename)
            if info.is_dir() or not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            items.append((name, z.read(info)))
    return items


def decode(data):
    buf = np.frombuffer(data, np.uint8)
    im = cv2.imdecode(buf, cv2.IMREAD_REDUCED_COLOR_4)
    if im is None or max(im.shape[:2]) < 200:
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return im


def describe(item):
    cv2.setNumThreads(1)
    name, data = item
    im = decode(data)
    if im is None:
        return name, (1, 1), np.zeros(768, np.float32), np.zeros((0, 2), np.float32), None, np.zeros((8, 8, 3), np.uint8)
    s = SIDE / max(im.shape[:2])
    if s < 1:
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    kp, des = cv2.SIFT_create(nfeatures=600).detectAndCompute(gray, None)
    pts = np.array([k.pt for k in kp], np.float32).reshape(-1, 2)
    thumb = cv2.resize(gray, (32, 24), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    k = SMALL / max(im.shape[:2])
    small = cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    return name, im.shape[:2], thumb, pts, des, small


def candidates(feats):
    th = np.stack([f[2] for f in feats])
    th = th - th.mean(1, keepdims=True)
    th /= np.linalg.norm(th, axis=1, keepdims=True) + 1e-6
    sim = th @ th.T
    np.fill_diagonal(sim, -9)
    k = min(NEIGHBOURS, len(feats) - 1)
    nn = np.argsort(-sim, axis=1)[:, :k]
    pairs = set()
    for i in range(len(feats)):
        for j in nn[i]:
            pairs.add((min(i, int(j)), max(i, int(j))))
    return sorted(pairs)


def init(feats):
    global IMAGES
    IMAGES = feats
    cv2.setNumThreads(1)


def ncc(a, b, m):
    a = a[m] - a[m].mean()
    b = b[m] - b[m].mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))


def pair_features(pair):
    i, j = pair
    _, shape_a, _, pa, da, sa = IMAGES[i]
    _, shape_b, _, pb, db, sb = IMAGES[j]
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return None
    matches = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
    if len(good) < 6:
        return None
    src = pa[[g.queryIdx for g in good]]
    dst = pb[[g.trainIdx for g in good]]
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if M is None:
        return None
    n = int(inl.sum())
    if n < 12:
        return None
    hA, wA = shape_a
    hB, wB = shape_b
    ka = sa.shape[1] / wA
    kb = sb.shape[1] / wB
    Ms = np.diag([kb, kb, 1.0]) @ np.vstack([M, [0, 0, 1]]) @ np.linalg.inv(np.diag([ka, ka, 1.0]))
    inv = cv2.invertAffineTransform(Ms[:2])
    dsize = (sa.shape[1], sa.shape[0])
    warped = cv2.warpAffine(sb, inv, dsize, flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpAffine(np.ones(sb.shape[:2], np.uint8), inv, dsize, flags=cv2.INTER_NEAREST) > 0
    mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if mask.sum() < 500:
        return None
    A = cv2.GaussianBlur(sa, (0, 0), 1.2).astype(np.float32)
    B = cv2.GaussianBlur(warped, (0, 0), 1.2).astype(np.float32)
    d = np.abs(A - B)[mask]
    ga = cv2.cvtColor(sa, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ea = cv2.magnitude(cv2.Sobel(ga, cv2.CV_32F, 1, 0), cv2.Sobel(ga, cv2.CV_32F, 0, 1))
    eb = cv2.magnitude(cv2.Sobel(gb, cv2.CV_32F, 1, 0), cv2.Sobel(gb, cv2.CV_32F, 0, 1))
    hsv_a = cv2.cvtColor(sa, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_b = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV).astype(np.float32)
    scale = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
    angle = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    mean_a, mean_b = A[mask].mean(0), B[mask].mean(0)
    std_a, std_b = A[mask].std(0), B[mask].std(0)
    res = np.abs((src @ M[:, :2].T + M[:, 2]) - dst)[inl.ravel() > 0]
    values = dict(
        good=len(good), inl=n, ratio=n / len(good), cover=float(mask.mean()),
        scale=float(np.log(scale)) if scale > 0 else -9.0, ang=abs(angle),
        ar=abs(np.log((hA / wA) / (hB / wB))), tx=abs(M[0, 2]) / wB, ty=abs(M[1, 2]) / hB, res=float(res.mean()),
        mad=float(d.mean()), mad_p90=float(np.percentile(d, 90)),
        mad_b=float(d[:, 0].mean()), mad_g=float(d[:, 1].mean()), mad_r=float(d[:, 2].mean()),
        ncc=ncc(ga, gb, mask), ncc_e=ncc(ea, eb, mask),
        dmean=float(np.abs(mean_a - mean_b).mean()), dstd=float(np.abs(std_a - std_b).mean()),
        dh=float(np.abs(hsv_a[..., 0][mask] - hsv_b[..., 0][mask]).mean()),
        ds=float(np.abs(hsv_a[..., 1][mask] - hsv_b[..., 1][mask]).mean()),
        dv=float(np.abs(hsv_a[..., 2][mask] - hsv_b[..., 2][mask]).mean()),
        sat_a=float(hsv_a[..., 1][mask].mean()), sat_b=float(hsv_b[..., 1][mask].mean()),
        szratio=float(np.log(max(hA, wA) / max(hB, wB))), nka=len(pa), nkb=len(pb),
    )
    x = (np.array([values[k] for k in MODEL["names"]]) - np.array(MODEL["mean"])) / np.array(MODEL["scale"])
    z = float(x @ np.array(MODEL["coef"]) + MODEL["bias"])
    return i, j, 1.0 / (1.0 + np.exp(-z))


def main():
    items = read_archive()
    with Pool(os.cpu_count() or 1) as pool:
        feats = pool.map(describe, items, chunksize=16)
    names = [f[0] for f in feats]
    rows = []
    if len(feats) > 1:
        pairs = candidates(feats)
        with Pool(os.cpu_count() or 1, initializer=init, initargs=(feats,)) as pool:
            for r in pool.imap_unordered(pair_features, pairs, chunksize=100):
                if r is not None and r[2] > THRESHOLD:
                    rows.append((names[r[0]], names[r[1]]))
    rows.sort()
    with open(OUTPUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_a", "image_b"])
        w.writerows(rows)
    print(len(names), "images,", len(rows), "pairs")


if __name__ == "__main__":
    main()
