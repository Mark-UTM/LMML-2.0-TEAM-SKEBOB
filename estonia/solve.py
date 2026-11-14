from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

d = pd.read_csv(Path(__file__).resolve().parent / "applications.csv")
X = d.drop(columns=["submission_id", "date_of_birth", "citizenship", "label"]); y = d.label
person = d.date_of_birth + "|" + d.citizenship
knn = lambda: KNeighborsClassifier(n_neighbors=1)
pipe = lambda: make_pipeline(StandardScaler(), knn())

def shipped(scale_before_split=True, seed=1137):
    if scale_before_split:
        Xs = StandardScaler().fit_transform(X)
        a, b, ya, yb = train_test_split(Xs, y, test_size=0.2, random_state=seed)
        return accuracy_score(yb, knn().fit(a, ya).predict(b))
    a, b, ya, yb = train_test_split(X, y, test_size=0.2, random_state=seed)
    return accuracy_score(yb, pipe().fit(a, ya).predict(b))

print(f"{len(d)} applications, {person.nunique()} distinct people; label constant per person: "
      f"{(d.groupby(person).label.nunique() == 1).all()}")
print(f"Reported pipeline                               : {shipped():.4f}")
print(f"Decoy A fixed (scaler fit on train only)        : {shipped(scale_before_split=False):.4f}")
seeds = [shipped(seed=s) for s in range(30)]
print(f"Decoy B fixed (30 random splits, mean +- sd)    : {np.mean(seeds):.4f} +- {np.std(seeds):.4f}")
loo = cross_val_predict(pipe(), X, y, cv=LeaveOneOut())
print(f"Decoys A+B fixed (LeaveOneOut, scaler in fold)  : {accuracy_score(y, loo):.4f}  <- still inflated")
logo = cross_val_predict(pipe(), X, y, cv=LeaveOneGroupOut(), groups=person)
acc = accuracy_score(y, logo)
print(f"Real bug fixed (leave-one-PERSON-out)           : {acc:.4f}")
print(f"FLAG: SIGMOID_{round(acc * 100)}")
