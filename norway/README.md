# The Nisse's Bad Shift (Norway)

The repaired labels are in `test-output.csv` (`sample_id,corrected_label`), and the model retrained on them is `model.joblib`.

## How the labels were repaired
**Damage model.** On a bad night (a contiguous run of `sample_id`s), the nisse swaps one digit pair `a<->b`
for **every** sample of `a` or `b` written that night. Different nights may swap different pairs.

**Repair:**
1. Train an RBF SVC on clean labels. On the train set itself, use out-of-fold predictions.
2. Take the disagreements (given label ≠ predicted) that form a swap pair `{a, b}`, and cluster them along `sample_id`.
   A cluster counts as a bad night spanning `[first, last]` if it has at least 5 disagreements and they make up
   at least 60% of all `a`/`b`-labelled samples in that span. On a real bad night, nearly all of them are swapped.
3. Inside that night, swap `a<->b` for every sample labelled `a` or `b`. This also catches samples the classifier missed.
4. Retrain the classifier on the repaired labels and save it to `model.joblib`.

The script prints the repair F1 on the train set, where the corrected labels are known.

## Run
Put `train-input.csv`, `train-output.csv` and `test-input.csv` from the task next to `solve.py`.
```
pip install -r norway/requirements.txt
python norway/solve.py       # -> test-output.csv, model.joblib
```
