import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = ["temperature_c", "humidity_pct", "vibration_g", "acoustic_db", "air_pressure_kpa", "co2_ppm"]
AR = ["temperature_c", "air_pressure_kpa"]

def healed():
    x = pd.read_csv(HERE / "train-input.csv")
    y = pd.read_csv(HERE / "train-output.csv")
    bad = y.is_anomaly.astype(bool).values
    rng = np.random.default_rng(0)
    for zone, g in x.groupby("bunker_zone"):
        normal = g[~bad[g.index]]
        for s in S:
            idx = g.index[bad[g.index]]
            if s in AR:
                col = x.loc[g.index, s].copy()
                col[bad[g.index]] = np.nan
                x.loc[g.index, s] = col.interpolate(limit_direction="both").values
            else:
                x.loc[idx, s] = rng.normal(normal[s].mean(), normal[s].std(), len(idx))
    return x

def make(seed, base):
    rng = np.random.default_rng(seed)
    x = base.copy()
    n = len(x)
    lab = np.array(["normal"] * n, dtype=object)
    std = {(z, s): base.loc[base.bunker_zone == z, s].std() for z in base.bunker_zone.unique() for s in S}
    taken = np.zeros(n, bool)

    def place(length):
        for _ in range(1000):
            i = int(rng.integers(100, n - length - 10))
            if not taken[i - 20:i + length + 20].any() and x.bunker_zone[i] == x.bunker_zone[i + length - 1]:
                taken[i:i + length] = True
                return i
        return None

    for _ in range(3):
        L = int(rng.integers(45, 75))
        i = place(L)
        if i is None:
            continue
        s = rng.choice(S)
        sign = rng.choice([-1, 1])
        top = rng.uniform(7, 13)
        z = x.bunker_zone[i]
        x.loc[i:i + L - 1, s] += sign * top * std[(z, s)] * np.linspace(0, 1, L)
        lab[i:i + L] = "drift"
    for _ in range(3):
        L = int(rng.integers(25, 40))
        i = place(L)
        if i is None:
            continue
        z = x.bunker_zone[i]
        sens = rng.choice(S, size=int(rng.integers(2, 4)), replace=False)
        for s in sens:
            x.loc[i:i + L - 1, s] += rng.choice([-1, 1]) * rng.uniform(3.5, 5.5) * std[(z, s)]
        lab[i:i + L] = "collective"
    for _ in range(7):
        i = place(1)
        if i is None:
            continue
        z = x.bunker_zone[i]
        s = rng.choice(S)
        x.loc[i, s] += rng.choice([-1, 1]) * rng.uniform(12, 25) * std[(z, s)]
        lab[i] = "point"
    y = pd.DataFrame({"timestamp": x.timestamp, "is_anomaly": (lab != "normal").astype(int), "anomaly_type": lab})
    return x, y
