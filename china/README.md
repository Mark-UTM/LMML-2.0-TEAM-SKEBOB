# Carved in bones, engraved in the heart (China)

**Flag: `SIGMOID_FENGSHUI`**

`main.py` trains a glyph detector on the labelled rubbings, counts each crate's glyph in its five sets, writes `counts.csv` and prints the decoded flag.

## Counting
The glyph appears only a handful of times per set in natural images, so counting needs a real detector. Template matching and HOG sliding windows (tried first) could not tell the glyphs apart: counts were off by dozens per set.

The detector is a CenterNet-style heatmap model: an ImageNet resnet18 (stem, layers 1-3) with a small head that predicts one heatmap per crate glyph plus one for all other glyphs.
It is trained on 1,650 of the 1,800 labelled images (random crop and scale, brightness jitter, focal loss, 6 epochs on CPU).
Each peak of a heatmap is one glyph, so a count is the number of local maxima above a threshold.
The threshold is picked per glyph on the 150 held-out images, where predicted and true counts per glyph agree closely (for example 37 vs 37, 16 vs 16, 19 vs 19, 71 vs 74 occurrences).

## Decoding
The counts of every crate fall into two clearly separated groups, 0-4 and 6-19:

| crate | glyph | counts in sets 1..5 | bits | value | letter |
|---|---|---|---|---|---|
| crate_01 | g015 | 0 1 12 10 1 | 00110 | 6 | F |
| crate_02 | g028 | 2 0 14 1 14 | 00101 | 5 | E |
| crate_03 | g005 | 1 17 14 15 4 | 01110 | 14 | N |
| crate_04 | g018 | 2 1 11 13 16 | 00111 | 7 | G |
| crate_05 | g003 | 19 1 0 13 13 | 10011 | 19 | S |
| crate_06 | g011 | 1 8 3 2 2 | 01000 | 8 | H |
| crate_07 | g001 | 13 2 14 2 14 | 10101 | 21 | U |
| crate_08 | g010 | 2 6 1 1 15 | 01001 | 9 | I |

A set with 5 or more glyphs is a 1, otherwise 0. The five sets read in order are a 5-bit number, which is the letter's position in the alphabet (A = 1). Crates 01 to 08 spell FENGSHUI.

## Run
```
pip install -r requirements.txt
python main.py --data path/to/dataset --out counts.csv
```
`--data` holds `train/`, `train_labels.csv`, `crates.csv` and `crates/`. Training takes about 15 minutes on a 12-core CPU. `--weights weights.pt` skips training and reuses saved weights.
