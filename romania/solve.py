import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
CH = [f"c{j + 1:02d}" for j in range(48)]
IDX = {c: i for i, c in enumerate(CH)}
RADIAL = [IDX[c] for c in ["c03", "c09", "c12", "c14", "c15", "c19", "c27", "c28", "c30", "c36", "c39", "c40"]]
NONRADIAL = [j for j in range(48) if j not in RADIAL]
C13_CUT = -0.6
E_CUT = 12.0
SEGMENTS = ["A-high", "B-high", "low"]


class Scaler:
    def __init__(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0)

    def z(self, X):
        return (X - self.mu) / self.sd

    def quiet_energy(self, X):
        return (self.z(X)[:, NONRADIAL] ** 2).sum(1)


def segment_of(X, sc):
    narrow_band = sc.quiet_energy(X) < E_CUT
    c13 = X[:, IDX["c13"]]
    return np.where(narrow_band, "low", np.where(c13 >= C13_CUT, "A-high", "B-high"))


def make_experts():
    return {
        "A-high": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)),
        "B-high": HistGradientBoostingClassifier(random_state=0),
        "low": make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(),
                             LogisticRegression(C=0.03, max_iter=6000)),
    }


def expert_input(name, X, sc):
    return sc.z(X)[:, RADIAL] if name == "low" else X


class Mixture:
    def fit(self, X, y):
        self.sc = Scaler(X)
        seg = segment_of(X, self.sc)
        self.experts = make_experts()
        for name, model in self.experts.items():
            m = seg == name
            model.fit(expert_input(name, X[m], self.sc), y[m])
        return self

    def predict(self, X):
        seg = segment_of(X, self.sc)
        pred = np.zeros(len(X), dtype=int)
        for name, model in self.experts.items():
            m = seg == name
            if m.any():
                pred[m] = model.predict(expert_input(name, X[m], self.sc))
        return pred


def main():
    train = pd.read_csv(HERE / "dataset" / "train.csv")
    sealed = pd.read_csv(HERE / "dataset" / "sealed.csv")
    X, y, S = train[CH].to_numpy(), train.vrancea.to_numpy(), sealed[CH].to_numpy()

    print("1. single models, 5-fold CV accuracy on train")
    singles = {
        "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "SVM (rbf)": make_pipeline(StandardScaler(), SVC(C=3.0)),
        "random forest": RandomForestClassifier(300, n_jobs=-1, random_state=0),
        "gradient boosting": HistGradientBoostingClassifier(random_state=0),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for name, model in singles.items():
        print(f"   {name:20s} {cross_val_score(model, X, y, cv=cv, n_jobs=-1).mean():.3f}")

    sc = Scaler(X)
    seg_train, seg_sealed = segment_of(X, sc), segment_of(S, sc)
    print("\n2. the three populations (gate: energy of the 36 non-radial channels, then c13)")
    for k in SEGMENTS:
        print(f"   {k:7s} share of train {np.mean(seg_train == k):.3f}   share of sealed {np.mean(seg_sealed == k):.3f}")

    print("\n3. mixture of experts, 5-fold CV accuracy per population")
    weights = {k: np.mean(seg_sealed == k) for k in SEGMENTS}
    expected = []
    for seed in (1, 2, 3):
        oof = np.zeros(len(y), dtype=int)
        for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            oof[b] = Mixture().fit(X[a], y[a]).predict(X[b])
        acc = {k: np.mean(oof[seg_train == k] == y[seg_train == k]) for k in SEGMENTS}
        expected.append(sum(weights[k] * acc[k] for k in SEGMENTS))
        print(f"   seed {seed}: " + "  ".join(f"{k} {v:.3f}" for k, v in acc.items())
              + f"   train-weighted {np.mean(oof == y):.3f}   sealed-weighted {expected[-1]:.3f}")
    print(f"   expected accuracy on the sealed set (populations weighted as in sealed): {np.mean(expected):.3f}")

    pred = Mixture().fit(X, y).predict(S)
    pd.DataFrame({"record_id": sealed.record_id, "vrancea": pred}).to_csv(HERE / "predictions.csv", index=False)
    print(f"\n4. wrote predictions.csv ({len(pred)} records, {pred.mean():.3f} predicted Vrancea)")


if __name__ == "__main__":
    main()
