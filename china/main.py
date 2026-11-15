import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

MAXS = 512
CROP = 352
STRIDE = 8
BIT_THRESHOLD = 5
MEAN, STD = 0.449, 0.226
torch.set_num_threads(os.cpu_count() or 4)


class Net(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        r = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.l1, self.l2, self.l3 = r.layer1, r.layer2, r.layer3
        self.red3 = nn.Conv2d(256, 128, 1)
        self.head = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, n_classes, 1))
        self.head[-1].bias.data.fill_(-2.5)

    def forward(self, x):
        f2 = self.l2(self.l1(self.stem(x)))
        f3 = self.l3(f2)
        up = F.interpolate(self.red3(f3), size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(torch.cat([f2, up], 1))


def prep(im):
    x = (im.astype(np.float32) / 255.0 - MEAN) / STD
    return np.repeat(x[None], 3, 0)


def heatmap(boxes, cls, n_classes, size, scale):
    hm = np.zeros((n_classes, size, size), np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    for (x, y, w, h), c in zip(boxes, cls):
        cx, cy = (x + w / 2) * scale / STRIDE, (y + h / 2) * scale / STRIDE
        if not (0 <= cx < size and 0 <= cy < size):
            continue
        sg = max(0.8, (w + h) * scale / STRIDE / 14.0)
        g = np.exp(-((xx - int(cx)) ** 2 + (yy - int(cy)) ** 2) / (2 * sg * sg))
        hm[c] = np.maximum(hm[c], g)
        hm[c, int(cy), int(cx)] = 1.0
    return hm


def focal_loss(pred, gt):
    p = torch.sigmoid(pred).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    pl = torch.log(p) * (1 - p) ** 2 * pos
    nl = torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * (1 - pos)
    return -(pl.sum() + nl.sum()) / pos.sum().clamp(min=1)


def sample(name, imgs, labels, class_of, n_classes, rng):
    im = imgs[name]
    s = MAXS / max(im.shape) * rng.uniform(0.8, 1.3)
    r = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    ph, pw = max(CROP, r.shape[0]), max(CROP, r.shape[1])
    canvas = np.full((ph, pw), int(np.median(r)), np.uint8)
    canvas[:r.shape[0], :r.shape[1]] = r
    y0, x0 = rng.integers(0, ph - CROP + 1), rng.integers(0, pw - CROP + 1)
    crop = canvas[y0:y0 + CROP, x0:x0 + CROP].astype(np.float32)
    crop = np.clip(crop * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20), 0, 255).astype(np.uint8)
    d = labels[labels.image == name]
    boxes = d[["x", "y", "w", "h"]].values.copy()
    boxes[:, 0] -= x0 / s
    boxes[:, 1] -= y0 / s
    cls = [class_of.get(g, n_classes - 1) for g in d.glyph_id]
    return prep(crop), heatmap(boxes, cls, n_classes, CROP // STRIDE, s)


def infer(net, im):
    s = MAXS / max(im.shape)
    r = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    ph, pw = (r.shape[0] + 15) // 16 * 16, (r.shape[1] + 15) // 16 * 16
    canvas = np.full((ph, pw), int(np.median(r)), np.uint8)
    canvas[:r.shape[0], :r.shape[1]] = r
    with torch.no_grad():
        out = net(torch.from_numpy(prep(canvas))[None])[0]
    return torch.sigmoid(out)


def peak_scores(net, im, n_glyphs):
    p = infer(net, im)
    keep = p == F.max_pool2d(p[None], 3, 1, 1)[0]
    return [p[c][keep[c]].numpy() for c in range(n_glyphs)]


def train(net, imgs, labels, names, class_of, n_classes, epochs):
    for p in net.stem.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW([p for n, p in net.named_parameters() if not n.startswith("stem")], lr=1e-3, weight_decay=1e-4)
    batch = 12
    per_epoch = len(names) // batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * per_epoch, pct_start=0.15)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for ep in range(epochs):
        order = rng.permutation(names)
        for b in range(per_epoch):
            items = [sample(n, imgs, labels, class_of, n_classes, rng) for n in order[b * batch:(b + 1) * batch]]
            x = torch.from_numpy(np.stack([a for a, _ in items]))
            y = torch.from_numpy(np.stack([h for _, h in items]))
            loss = focal_loss(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        print("epoch %d loss %.3f %.0fs" % (ep, loss.item(), time.time() - t0), file=sys.stderr, flush=True)


def calibrate(net, imgs, labels, held, glyphs, class_of):
    scores = {n: peak_scores(net, imgs[n], len(glyphs)) for n in held}
    truth = np.zeros((len(held), len(glyphs)))
    for i, n in enumerate(held):
        for g in labels[labels.image == n].glyph_id:
            if g in class_of:
                truth[i, class_of[g]] += 1
    thresholds = {}
    for c, g in enumerate(glyphs):
        best = None
        for t in np.arange(0.15, 0.71, 0.05):
            pred = np.array([(scores[n][c] > t).sum() for n in held])
            err = np.abs(pred - truth[:, c]).sum() + abs(pred.sum() - truth[:, c].sum())
            if best is None or err < best[0]:
                best = (err, t, pred.sum(), truth[:, c].sum())
        thresholds[g] = best[1]
        print("%s threshold %.2f held-out predicted %d true %d" % (g, best[1], best[2], best[3]), file=sys.stderr)
    return thresholds


def decode(counts):
    letters = []
    for _, grp in counts.groupby("crate"):
        bits = (grp.sort_values("set")["count"].values >= BIT_THRESHOLD).astype(int)
        value = int("".join(map(str, bits)), 2)
        letters.append(chr(ord("A") + value - 1))
    return "".join(letters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default="counts.csv")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--weights", default="")
    args = ap.parse_args()

    labels = pd.read_csv(os.path.join(args.data, "train_labels.csv"))
    crates = pd.read_csv(os.path.join(args.data, "crates.csv"), encoding="utf-8-sig")
    glyphs = list(crates.glyph_id)
    class_of = {g: i for i, g in enumerate(glyphs)}
    n_classes = len(glyphs) + 1
    names = sorted(labels.image.unique())
    train_names, held = names[:1650], names[1650:]
    imgs = {n: cv2.imread(os.path.join(args.data, "train", "images", n), 0) for n in names}

    net = Net(n_classes, pretrained=not args.weights)
    if args.weights:
        net.load_state_dict(torch.load(args.weights, weights_only=True))
    else:
        train(net, imgs, labels, train_names, class_of, n_classes, args.epochs)
        torch.save(net.state_dict(), "weights.pt")
    net.eval()

    thresholds = calibrate(net, imgs, labels, held, glyphs, class_of)

    rows = []
    for crate, g in zip(crates.crate, glyphs):
        for s in range(1, 6):
            files = sorted(glob.glob(os.path.join(args.data, "crates", crate, "set_%d" % s, "*.jpg")))
            count = sum(int((peak_scores(net, cv2.imread(f, 0), len(glyphs))[class_of[g]] > thresholds[g]).sum()) for f in files)
            rows.append((crate, "set_%d" % s, g, count))
    counts = pd.DataFrame(rows, columns=["crate", "set", "glyph_id", "count"])
    counts.to_csv(args.out, index=False)
    print(counts.pivot(index="crate", columns="set", values="count"))
    print("SIGMOID_" + decode(counts))


if __name__ == "__main__":
    main()
