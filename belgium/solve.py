from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / "potato_fields.csv")
d["perimeter"] = 2 * (d.long_side_m + d.area_ha * 1e4 / d.long_side_m)

def ols(y, cols):
    X = np.column_stack([np.ones(len(d))] + [d[c] for c in cols])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    print(f"{y.name} ~ {' + '.join(cols)}: coef {np.round(b, 4)}, R2 {1 - r.var() / y.var():.6f}, resid sd {r.std():.3f}")
    return r

r_edge = ols(d.edge_m, ["perimeter", "ditch_m", "hedge_m"])
r_hedge = ols(d.hedge_m, ["perimeter", "ditch_m"])
r_ditch = ols(d.ditch_m, ["perimeter"])

fig, ax = plt.subplots(3, 1, figsize=(16, 11))
for a, r, title in [(ax[0], r_hedge, "hedge_m (noted by Mariette on site) - linear model  ->  SUPERVISION  (flag)"),
                    (ax[1], r_edge, "edge_m (measured by Mariette) - full model  ->  PREDICTION"),
                    (ax[2], r_ditch, "ditch_m (farmer's declaration) - 0.5 * perimeter  ->  REGRESSION")]:
    a.scatter(d.perimeter, r, s=3)
    a.set_title(title); a.set_xlabel("perimeter 2(L + A/L), m"); a.set_ylabel("residual, m")
fig.tight_layout()
fig.savefig(HERE / "fry_ndship.png", dpi=110)
print("saved fry_ndship.png  ->  SIGMOID_SUPERVISION")
