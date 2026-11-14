# Fry-ndship (Belgium)

**Flag: `SIGMOID_PREDICTION`**

![residuals](fry_ndship.png)

## How it was found
1. A field's edge is its geometric perimeter plus its ditch and hedge. For a rectangle,
   `P = 2 * (long_side + area / long_side)` with `area = area_ha * 1e4`.
   Fitting `edge_m ~ 0.41 * P + 1.00 * ditch_m + 1.00 * hedge_m` gives a residual sd of only 0.05 m.
2. Each column hides a word in its noise. Plotting the residual against the perimeter makes it readable:
   - `ditch_m` (the farmer's declaration) minus `0.5 * P` spells **REGRESSION**;
   - `hedge_m` (noted by Mariette) minus a linear term in `P` spells **SUPERVISOR**;
   - `edge_m` (measured by Mariette) minus the full linear model spells **PREDICTION**.
3. The flag is the word in Mariette's own measurements: **PREDICTION**.

## Run
Put `potato_fields.csv` from the task next to `solve.py`.
```
pip install numpy pandas matplotlib
python belgium/solve.py      # prints the fits and writes fry_ndship.png
```
