# The Douro Paradox (Portugal)

**Flag: `SIGMOID_DISTANCE_FROM_PORTO_KM`**

## How it was found
1. Pooled over all plots, the treatment succeeds more often than the control.
2. Splitting every column into quantile or categorical groups never reverses the effect in *every* group.
3. `distance_from_porto_km` is bimodal, with an empty gap between 104 and 114 km (the lower vs the upper Douro).
   Splitting at that natural gap, the control beats the treatment in **both** groups: a complete Simpson's reversal.

Why it happens: the new method was mostly trialled far from Porto (1746 of the 1890 treatment plots), where success is higher
for every method. Pooling mixes that location advantage into the treatment's success rate.

## Run
Put `douro_vineyard_data.csv` from the task next to `solve.py`.
```
pip install pandas
python portugal/solve.py
```
