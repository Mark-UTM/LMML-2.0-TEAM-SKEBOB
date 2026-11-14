from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
guide = pd.read_csv(HERE / "office_price_guide.csv")
buys = pd.read_csv(HERE / "purchases.csv").merge(guide, on="Product")

buys["unit_price"] = buys.Total_paid_eur / buys.Quantity
buys["verdict"] = np.select(
    [buys.unit_price < buys.Fair_min_eur, buys.unit_price > buys.Fair_max_eur],
    ["fake", "tourist trap"], "fair")
print(buys.groupby(["Product", "verdict"]).size().unstack(fill_value=0), "\n")

fair = buys[buys.verdict == "fair"]
board = np.zeros((6, 21), dtype=int)
for _, r in fair.iterrows():
    board[r.List_no - 1, r.Pilot_no - 1] = 1
for row in board:
    print("".join("#" if v else "." for v in row))

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.imshow(board, cmap="Greens", vmin=0, vmax=1.3, aspect="equal")
ax.set_xticks(range(21), [str(i) for i in range(1, 22)])
ax.set_yticks(range(6), [f"{n}. {p}" for n, p in zip(guide.List_no, guide.Product)])
ax.set_xticks(np.arange(-.5, 21), minor=True)
ax.set_yticks(np.arange(-.5, 6), minor=True)
ax.grid(which="minor", color="white", linewidth=1.5)
ax.tick_params(which="minor", length=0)
ax.set_xlabel("Pilot_no (one column per pilot)")
ax.set_title(f"Christmas board: fair-price souvenirs ({len(fair)} of {len(buys)} receipts)  ->  CRUCIO")
fig.tight_layout()
fig.savefig(HERE / "board.png", dpi=150)
print("\nsaved", HERE / "board.png")
