import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from detector import Detector

def evaluate(params=None, inp=HERE / "train-input.csv", out=HERE / "train-output.csv", verbose=True):
    x = pd.read_csv(inp)
    y = pd.read_csv(out)
    det = Detector(**(params or {}))
    scores = []
    for row in x.to_dict("records"):
        s = det.update(row)
        assert 0.0 <= s <= 1.0 and s == s
        scores.append(s)
    pred = pd.Series(scores) >= 0.5
    truth = y.is_anomaly.astype(bool)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    rec = {t: float(pred[y.anomaly_type == t].mean()) for t in ["point", "drift", "collective"]}
    score = 100 * (0.80 * f1 + 0.05 * rec["point"] + 0.075 * rec["drift"] + 0.075 * rec["collective"])
    score = min(100.0, max(1.0, score))
    if verbose:
        print(f"score {score:.2f} | F1 {f1:.3f} (tp {tp}, fp {fp}, fn {fn}) | recall point {rec['point']:.2f} "
              f"drift {rec['drift']:.2f} collective {rec['collective']:.2f}")
    return score, pred

if __name__ == "__main__":
    evaluate()
