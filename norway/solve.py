from itertools import combinations
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
P = [f"pixel_{i}" for i in range(64)]
MIN_HITS, MAX_GAP, MIN_SHARE = 5, 60, 0.6
svc = lambda: SVC(C=10, gamma=0.001)

def repair(df, pred):
    df = df.assign(n=df.sample_id.str[4:].astype(int), pred=pred).sort_values("n")
    fixed = df.given_label.copy()
    for a, b in combinations(range(10), 2):
        hits = df[(df.given_label.isin([a, b])) & (df.pred.isin([a, b])) & (df.given_label != df.pred)].n.tolist()
        runs, cur = [], []
        for n in hits:
            if cur and n - cur[-1] > MAX_GAP:
                runs.append(cur); cur = []
            cur.append(n)
        runs.append(cur)
        for run in filter(None, runs):
            inside = (df.n >= run[0]) & (df.n <= run[-1]) & df.given_label.isin([a, b])
            share = len(run) / max(int(inside.sum()), 1)
            if len(run) >= MIN_HITS and share >= MIN_SHARE:
                fixed[inside] = df.given_label[inside].map({a: b, b: a})
                print(f"  bad night ids {run[0]}-{run[-1]}: swap {a}<->{b}, {int(inside.sum())} labels repaired")
    return fixed.reindex(df.index).loc[df.sort_index().index]

tr = pd.read_csv(HERE / "train-input.csv").merge(pd.read_csv(HERE / "train-output.csv"), on="sample_id")
te = pd.read_csv(HERE / "test-input.csv")

print("Train check (out-of-fold predictions on given labels):")
oof = cross_val_predict(svc(), tr[P], tr.given_label, cv=5)
rep = repair(tr[["sample_id", "given_label"]], oof)
wrong = tr.given_label != tr.corrected_label; changed = rep != tr.given_label
good = (changed & (rep == tr.corrected_label)).sum()
p, r = good / max(changed.sum(), 1), good / wrong.sum()
print(f"  train repair F1 = {2 * p * r / (p + r):.3f}")

print("Test:")
clf = svc().fit(tr[P], tr.corrected_label)
te["corrected_label"] = repair(te[["sample_id", "given_label"]], clf.predict(te[P])).values
te[["sample_id", "corrected_label"]].to_csv(HERE / "test-output.csv", index=False)
print(f"  changed {(te.corrected_label != te.given_label).sum()} labels -> test-output.csv")

final = svc().fit(pd.concat([tr[P], te[P]]), pd.concat([tr.corrected_label, te.corrected_label]))
joblib.dump(final, HERE / "model.joblib")
print("  retrained model -> model.joblib")
