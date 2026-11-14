# Feather Stories (task 12)

**Flag: `SIGMOID_RHGLWQ`**

The unregistered White Stork lives at Lake Manta in the Lower Prut, near Cahul (45.6061 N, 28.1209 E, locality `L5819951`).
It was read there 24 times between 2016 and 2025, by more than 20 different observers. The reads agree on **RHGLWQ**,
and that ring is not in the register.

## 1. What a misread looks like

- An unread letter is written `?` in the data. The statement says `.`, but the data uses `?`.
- Only 24 letters ever occur (no U, no V), and misreads never leave a group of four look-alike letters:
  `{A K R X} {B G S Z} {C D O Q} {E F L P} {H M N W} {I J T Y}`.
- Errors are heavy: only 95 of the 3023 ring reads match a registered ring exactly.

## 2. Finding the stork's reads

The two approaches below were built independently and agree.

**Group signature (`mask_model.py`).** A misread keeps every letter in its group, so every read keeps the
*group signature* of the true ring: the group index of each letter. 287 of the 289 register rings have a unique signature.
Matching reads by signature, with `?` as a wildcard, assigns 2544 reads to exactly one registered bird, and the species agrees in 99.9% of them.
**Exactly 21 reads match no registered ring. All of them are `Ciconia ciconia`, and all were made at one point, L5819951 near Cahul.**

**Likelihood and nest distance (`assign.py`).** Each of the 320 White Stork reads is scored against the 30 registered storks
with a letter-confusion likelihood plus the distance to the stork's nest. 296 reads are explained by registered storks near their nests.
24 reads are explained by none, and all of them are at the same Cahul point. The nearest registered stork lives 74 km away.
The 21 signature-orphans are among these 24. The other 3 reads are at the same point and match another bird's signature only through `?`.

## 3. Decoding the ring (`decode.py`)

| position | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| letter | R | H | G | L | W | Q |
| reads | R 19/23 | H 19/22 | G 13/16 | L 15/19 | W 18/22 | Q 17/19 |

Maximum likelihood with a learned confusion matrix, a plain majority vote and leave-one-out (24 of 24) all give `RHGLWQ`.

## 4. The trap we ruled out: whole places that misread the same way

Inside a group, give each letter a 2-bit label:
`A0 K1 R2 X3 | B0 G1 Z2 S3 | C0 O1 D2 Q3 | E0 F1 L2 P3 | H0 W1 N2 M3 | I0 Y1 J2 T3`.
Every misread is then `true XOR mask`, with mask 1, 2 or 3 (the three ways to pair up four letters).
The mask **belongs to the observation point**:

- At some points almost everyone reads the same wrong letters. At Șoldănești (47.94 N), FAGQKC is read as EKBDAO,
  and 86% of all letters there carry mask 1.
- A bird read away from its nest takes the local point's mask, not its nest's mask (75 cases against 19).
- Observer, year, month, hour, species, ring letters, locality id and coordinates do not predict the mask
  (`mask_bird.py`, `frame_check.py`).

So reads that agree with each other are not proof on their own. If Cahul were a shifted point, the ring would be
`RHGLWQ XOR m`, giving `XWBPHD`, `ANSEMO` or `KMZFNC`. Why we answer RHGLWQ:

- **KMZFNC is excluded.** No clean point anywhere is dominated by mask 3, and the profile likelihood is far lower (`frame_check.py`).
- **Latitude.** South of 46.45 N, every clean point (at least 75% of letters on one mask) is read correctly:
  11 of 11 rings and 6 of 6 observation points. North of 46.6 N it is only 20 of 63 and 15 of 33.
  The share of shifted points grows steadily towards the north. Cahul (45.6 N) is the southernmost point in the data (`frame_check.py`).
- **Counter-evidence.** The rare errors at Cahul (relative to RHGLWQ: 4 × mask 1, 5 × mask 2, 11 × mask 3) are spread more like
  a shifted point than a typical correct one (`noise_test.py`; this test picks the right frame for 87.5% of known points).
  The test also raises equally confident false alarms on correct points, including the southern point L7813627, whose profile is close to Cahul's.
  Calibrated on its error rate and combined with the latitude evidence, it still leaves RHGLWQ as the single most likely ring,
  at about 0.4–0.65. Each alternative is lower.
- **Design.** Nothing in the data determines a point's mask. If the stork's own point were shifted, its ring could not be derived at all.

## Run

Put the two task files, `observations.csv` and `ring_register.csv`, into `moldova/data/`. Then:

```
pip install numpy==2.5.3 pandas==3.0.6
python moldova/assign.py       # stork reads -> moldova/out/stork_candidates.csv   (stdlib only)
python moldova/decode.py       # ML + majority vote + leave-one-out -> RHGLWQ       (stdlib only)
python moldova/mask_model.py   # group signatures, masks, check on known rings
python moldova/frame_check.py  # the four frame hypotheses, latitude evidence
python moldova/noise_test.py   # error-composition test, its validation and calibration
python moldova/mask_bird.py    # masks per place / bird / observer                 (stdlib only)
```

Run `assign.py` first: `decode.py`, `frame_check.py` and `noise_test.py` read its output.

`mask_model.py` also prints the result of a full Bayesian model: a prior on the mask times a per-mask error distribution.
In leave-one-ring-out on known rings it recovers only 25.5% of rings exactly, worse than a plain majority vote for rings that are read correctly.
That is why we do not rely on its stork output, which points to XWBPHD. The frame argument is in section 4.
