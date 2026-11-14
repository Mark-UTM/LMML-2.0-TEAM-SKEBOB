from pathlib import Path
import numpy as np
import pandas as pd
import torch
from model import FEATURE_COLUMNS, MythicThreatClassifier

HERE = Path(__file__).resolve().parent
X = pd.read_csv(HERE / "test_dataset.csv")
y = pd.read_csv(HERE / "test_labels.csv").set_index("encounter_id").loc[X.encounter_id, "is_catastrophic_threat"].values
xt = torch.tensor(X[FEATURE_COLUMNS].values, dtype=torch.float32)

rows = {}
for name in ["ares", "athena", "apollo"]:
    m = MythicThreatClassifier()
    m.load_state_dict(torch.load(HERE / "models" / f"model_{name}.pth", weights_only=True))
    m.eval()
    with torch.no_grad():
        p = torch.sigmoid(m(xt)).numpy()
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f = lambda b: (1 + b * b) * prec * rec / max(b * b * prec + rec, 1e-12)
    rows[name.upper()] = dict(accuracy=(tp + tn) / len(y), TP=tp, FP=fp, FN=fn, TN=tn,
                              precision=prec, recall=rec, F1=f(1), F2=f(2))
r = pd.DataFrame(rows).T
print(r.round(4).to_string())
for ratio in [2, 5, 10, 20]:
    cost = r.FN * ratio + r.FP
    print(f"cost FN={ratio}xFP -> best {cost.idxmin()}  ({cost.to_dict()})")
best = r.recall.astype(float).idxmax()
print(f"\nHighest recall / fewest missed threats: {best}  ->  SIGMOID_{best}")
