# Just a siblings' thing (Greece)

**Flag: `SIGMOID_ARES`**

## How it was found
- The three checkpoints (`ares`, `athena`, `apollo`) share `model.MythicThreatClassifier`, a single linear layer producing a logit,
  with `P(threat) = sigmoid(logit)` and a 0.5 threshold. Weights are loaded with `weights_only=True`, so no pickled code runs.
- All three have the same accuracy, so accuracy cannot pick the oracle.
- The setting is one where a missed catastrophic threat (a false negative) costs far more than a false alarm.
  We compared recall, FN count, precision, F1 and F2, and swept the expected cost for FN = 2, 5, 10 and 20 × FP.
- **ARES** has the highest recall and the fewest missed threats, and it wins on F2 and in every cost sweep.

## Run
Put the task files next to `solve.py`: `model.py`, `models/model_*.pth`, `test_dataset.csv` and `test_labels.csv`.
```
pip install -r greece/requirements.txt
python greece/solve.py
```
