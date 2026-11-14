import math

SENSORS = ["temperature_c", "humidity_pct", "vibration_g", "acoustic_db", "air_pressure_kpa", "co2_ppm"]

P = dict(
    warmup=60,
    base_alpha=0.01,
    var_alpha=0.01,
    point_z=7.0,
    clip_z=6.0,
    ew_alpha=0.25,
    drift_z=2.1,
    cusum_k=0.5,
    cusum_h=9.0,
    coll_z=1.8,
    coll_n=2,
    end_z=1.8,
    trend_w=20,
    trend_t=5.5,
    trend_m=1.2,
)

class _Zone:
    def __init__(self):
        self.n = 0
        self.mean = {s: 0.0 for s in SENSORS}
        self.var = {s: 0.0 for s in SENSORS}
        self.ew = {s: 0.0 for s in SENSORS}
        self.cp = {s: 0.0 for s in SENSORS}
        self.cn = {s: 0.0 for s in SENSORS}
        self.ewvar = {s: None for s in SENSORS}
        self.hist = {s: [] for s in SENSORS}

class Detector:
    def __init__(self, **params):
        self.p = dict(P)
        self.p.update(params)
        self.zones = {}

    def _values(self, reading):
        out = {}
        for s in SENSORS:
            try:
                v = float(reading.get(s))
            except (TypeError, ValueError):
                v = float("nan")
            out[s] = v
        return out

    def update(self, reading: dict) -> float:
        p = self.p
        zone = self.zones.setdefault(str(reading.get("bunker_zone", "")), _Zone())
        v = self._values(reading)

        if zone.n < p["warmup"]:
            zone.n += 1
            for s in SENSORS:
                if math.isfinite(v[s]):
                    d = v[s] - zone.mean[s]
                    zone.mean[s] += d / zone.n
                    zone.var[s] += (d * (v[s] - zone.mean[s]) - zone.var[s]) / zone.n
            return 0.0

        z = {}
        for s in SENSORS:
            sd = math.sqrt(max(zone.var[s], 1e-12))
            z[s] = (v[s] - zone.mean[s]) / sd if math.isfinite(v[s]) else 0.0

        point = max(abs(z[s]) for s in SENSORS) >= p["point_z"]

        a = p["ew_alpha"]
        white = a / (2 - a)
        ez = {}
        for s in SENSORS:
            zc = max(-p["clip_z"], min(p["clip_z"], z[s])) if not point else 0.0
            zone.ew[s] += a * (zc - zone.ew[s])
            r = max(1.0, (zone.ewvar[s] if zone.ewvar[s] is not None else white) / white)
            ez[s] = zone.ew[s] / math.sqrt(r)
            zs = zc / math.sqrt(r)
            zone.cp[s] = max(0.0, zone.cp[s] + zs - p["cusum_k"])
            zone.cn[s] = max(0.0, zone.cn[s] - zs - p["cusum_k"])

        drift = any(abs(ez[s]) >= p["drift_z"] for s in SENSORS) or \
            any(max(zone.cp[s], zone.cn[s]) >= p["cusum_h"] for s in SENSORS)

        w = int(p["trend_w"])
        if w > 2:
            tx = [i - (w - 1) / 2 for i in range(w)]
            sxx = sum(t * t for t in tx)
            for s in SENSORS:
                h = zone.hist[s]
                h.append(0.0 if point else max(-p["clip_z"], min(p["clip_z"], z[s])))
                if len(h) > w:
                    h.pop(0)
                if len(h) == w and not drift:
                    m = sum(h) / w
                    b = sum(t * (v - m) for t, v in zip(tx, h)) / sxx
                    res = sum((v - m - b * t) ** 2 for t, v in zip(tx, h)) / (w - 2)
                    tstat = b / math.sqrt(max(res, 1e-9) / sxx)
                    recent = sum(h[-5:]) / 5
                    if abs(tstat) >= p["trend_t"] and abs(recent) >= p["trend_m"] and tstat * recent > 0:
                        drift = True
        coll = sum(1 for s in SENSORS if abs(ez[s]) >= p["coll_z"]) >= p["coll_n"]

        if (drift or coll) and max(abs(z[s]) for s in SENSORS) < p["end_z"]:
            for s in SENSORS:
                zone.cp[s] = zone.cn[s] = 0.0
                zone.ew[s] *= 0.3
            drift = coll = False

        anomalous = point or drift or coll
        if not anomalous:
            for s in SENSORS:
                if math.isfinite(v[s]):
                    d = v[s] - zone.mean[s]
                    zone.mean[s] += p["base_alpha"] * d
                    zone.var[s] += p["var_alpha"] * (d * d - zone.var[s])
                e2 = zone.ew[s] * zone.ew[s]
                zone.ewvar[s] = e2 if zone.ewvar[s] is None else zone.ewvar[s] + p["var_alpha"] * (e2 - zone.ewvar[s])
            return 0.0
        return 1.0
