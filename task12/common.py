from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TASKS = ROOT / "data"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

UNREAD = "?"
STORK = "Ciconia ciconia"

def load_observations() -> pd.DataFrame:
    return pd.read_csv(TASKS / "observations.csv", dtype={"ring_read": str, "observers": str})

def load_register() -> pd.DataFrame:
    return pd.read_csv(TASKS / "ring_register.csv")

def load_reads() -> pd.DataFrame:
    o = load_observations()
    return o[o.ring_read.notna()].copy()

def mismatches(read: str, ring: str) -> int:
    return sum(1 for a, b in zip(read, ring) if a != UNREAD and a != b)

def km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))
