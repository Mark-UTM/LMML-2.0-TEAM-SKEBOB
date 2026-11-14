# This Data Is Perfect! (Estonia)

**Flag: `SIGMOID_77`**

## How it was found
The shipped fraud screen (`project_pipeline.py`, StandardScaler + 1-NN) reports 93.42% accuracy. It makes three questionable choices:
- **A.** The scaler is fit on all rows before the split (preprocessing leakage). Fixing it barely changes the score, so it is a decoy.
- **B.** It uses a single 80/20 split with one fixed `random_state`. Averaging 30 splits gives the same level, so it is a decoy too.
- **C. The real bug: repeat applicants.** The 1971 applications come from only 900 people
  (`date_of_birth` + `citizenship`), and each person always has the same label. Identity fields are dropped before modelling,
  so the same person lands in both train and test, and 1-NN simply finds their other application.
  Even plain leave-one-out (with A and B fixed) stays inflated.

Fix: keep the model and evaluate leave-one-**person**-out
(`LeaveOneGroupOut`, groups = `date_of_birth|citizenship`), the grouped equivalent of leave-one-out.
The honest accuracy is 0.77, which gives `SIGMOID_77`.

## Run
Put `applications.csv` from the task next to `solve.py`.
```
pip install -r estonia/requirements.txt
python estonia/solve.py
```
