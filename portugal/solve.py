from pathlib import Path
import pandas as pd

d = pd.read_csv(Path(__file__).resolve().parent / "douro_vineyard_data.csv")
rate = lambda g: pd.crosstab(g, d.treatment, values=d.harvest_success, aggfunc="mean").round(3)

print("Pooled success rate:\n", d.groupby("treatment").harvest_success.mean().round(3), "\n")
print("Candidate columns (natural / categorical / quantile groups) - any full reversal?")
for c in d.columns.drop(["plot_id", "treatment", "harvest_success"]):
    s = d[c]
    g = s.astype(str) if (s.dtype == object or s.nunique() <= 15) else pd.qcut(s, 3).astype(str)
    t = rate(g)
    print(f"  {c:24} full reversal: {(t['treatment'] < t['control']).all()}")

group = pd.cut(d.distance_from_porto_km, [0, 109, 1000], labels=["near (<=104 km)", "far (>=114 km)"])
print("\ndistance_from_porto_km split at its natural gap (104-114 km):")
print(rate(group))
print(pd.crosstab(group, d.treatment))
t = rate(group)
assert (t["treatment"] < t["control"]).all()
print("\nControl ahead in every group -> SIGMOID_DISTANCE_FROM_PORTO_KM")
