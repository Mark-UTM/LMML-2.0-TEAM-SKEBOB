from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from bleu import corpus_bleu
from facts import card_mr, check

HERE = Path(__file__).resolve().parent
refs = pd.read_csv(HERE / "dataset" / "references.csv")
panel = pd.read_csv(HERE / "dataset" / "panel.csv")
ref_map = refs.groupby(["round", "card"]).reference.apply(list).to_dict()

rows = []
for _, x in panel.iterrows():
    c = check(x.text, card_mr(ref_map[(x["round"], x.card)]))
    rows.append(dict(round=x["round"], seat=x.seat, **c))
cards = pd.DataFrame(rows)
deck = cards.groupby(["round", "seat"])[["missing", "repeated", "wrong"]].sum()
deck["dishonesty"] = deck.sum(axis=1)
deck["bleu"] = [corpus_bleu(list(g.text), [ref_map[(r, c)] for c in g.card])["bleu"]
                for (r, s), g in panel.groupby(["round", "seat"])]
deck.to_csv(HERE / "honesty_scores.csv")

def decode(seats):
    return "".join(chr(ord("A") + 9 * (seats[i] - 1) + 3 * (seats[i + 1] - 1) + (seats[i + 2] - 1))
                   for i in range(0, len(seats), 3))

d = deck.reset_index()
honest = d.loc[d.groupby("round").dishonesty.idxmin()].sort_values("round").seat.tolist()
crowned = d.loc[d.groupby("round").bleu.idxmax()].sort_values("round").seat.tolist()
print(deck.dishonesty.unstack().T.to_string())
print("BLEU crown :", "".join(map(str, crowned)), "->", decode(crowned))
print("Honest     :", "".join(map(str, honest)), "->", decode(honest))

fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for seat, g in d.groupby("seat"):
    ax[0].scatter(g["round"], g.bleu, label=f"seat {seat}", s=30)
    ax[1].scatter(g["round"], g.dishonesty, label=f"seat {seat}", s=30)
ax[1].scatter(range(1, 25), [d[(d["round"] == r) & (d.seat == s)].dishonesty.iloc[0] for r, s in zip(range(1, 25), honest)],
              s=140, facecolors="none", edgecolors="k", label="honest (min)")
ax[0].set_ylabel("deck BLEU (house scorer)"); ax[1].set_ylabel("missing + repeated + wrong facts")
ax[1].set_xlabel("round"); ax[1].set_xticks(range(1, 25)); ax[0].legend(ncol=3, frameon=False); ax[1].legend(ncol=4, frameon=False)
ax[0].set_title(f"BLEU crowns {decode(crowned)};  fact check names the honest panellist -> {decode(honest)}")
fig.tight_layout(); fig.savefig(HERE / "honesty_vs_bleu.png", dpi=120)
print("FLAG: SIGMOID_" + decode(honest))
