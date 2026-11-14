# Shaken, Still Standing (Romania)

**Flag: `SIGMOID_CONVOLUTION`**. It is printed by the task's evaluator for `predictions.csv`.

`solve.py` fits the final model, prints the cross-validated scores below and writes `predictions.csv`.

## Single models are not enough
5-fold cross-validation on the 7,500 training records:

| model | accuracy |
|---|---|
| logistic regression | 0.601 |
| SVM (rbf) | 0.676 |
| random forest | 0.760 |
| gradient boosting | 0.847 |

The bar set by the evaluator is about 92.5% on the sealed set. The big gap between linear models (60%) and trees (85%) says the label does not follow one rule.

## What the data is made of
The sealed set is not distributed like the training set: a classifier tells them apart with AUC 0.69. Only `c13` shifts (median -1.1 in train, -0.6 in sealed), so the sealed set contains the training populations in different proportions.
Splitting the records by simple summaries reveals three populations, each with its own rule:

1. **Narrow-band records (25% of train, 40% of sealed).** 36 of the 48 channels are almost silent (variance 0.1 against 1.3 elsewhere); only 12 channels
   (`c03 c09 c12 c14 c15 c19 c27 c28 c30 c36 c39 c40`) carry signal. The summed energy of the 36 quiet channels separates these records from all others with an empty gap (below 6 against above 20), so the gate is exact.
   The label is a smooth radial rule in the 12 active channels: the share of y=1 grows from about 5% to 99% with the distance from the center, and a full quadratic form over those 12 channels is the best fit.
2. **Loud records with `c13 >= -0.6` (26% of train, 48% of sealed).** The label is a linear rule over many channels. The switch at `c13 = -0.6` is sharp:
   the linear expert is right 91-94% above it and 50% below it.
3. **Loud records with `c13 < -0.6` (49% of train, 12% of sealed).** The label is a parity (XOR) of three thresholds, on `c35`, `c43` and `c38`
   (thresholds about -0.25, -0.6 and 0.75). A parity of three signs is invisible to correlations and to linear models, and it needs a model that handles interactions.

## The combination: one expert per population
| population | expert | CV accuracy |
|---|---|---|
| narrow-band | logistic regression on all quadratic terms of the 12 active channels | 0.92 |
| loud, `c13 >= -0.6` | logistic regression on all channels (C = 0.1) | 0.92-0.93 |
| loud, `c13 < -0.6` | gradient boosting | 0.969 |

Each record is routed by the two gates above and answered by its expert. Cross-validated accuracy is 0.945 on the training mix and **0.928 when the populations are weighted as in the sealed set** (three CV seeds: 0.927, 0.929, 0.928).
The best single model reaches 0.847 on the training mix, and the same model would do worse on the sealed set, where the populations it handles badly are more common.

## What did not help
Gradient boosting with more trees, an MLP, k-nearest neighbours and a radial-basis SVM on all channels (0.53-0.68); parity with fitted thresholds instead of boosting (0.948 against 0.966);
extra channels or terms in the radial rule; an L1 or relaxed-lasso linear expert (no gain); self-training with the unlabeled sealed records (+0.25 points); a mixture-of-experts fitted by EM.

## Run
Put the task's `dataset/train.csv` and `dataset/sealed.csv` into `dataset/` next to `solve.py`.
```
pip install numpy pandas scikit-learn
python solve.py
```
