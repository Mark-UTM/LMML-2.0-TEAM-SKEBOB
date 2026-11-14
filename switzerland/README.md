# The Weak Layer (Switzerland)

`main.py` fits a `DecisionTreeClassifier` with default hyperparameters (integer `random_state`) on `test-input.csv`,
with `slab_released` as the target. The word is spelled by the fitted tree's questions in order:
the first letter of the feature tested at each split node, visiting split nodes in the tree's own node order
(scikit-learn numbers nodes depth-first, root first). It writes `SIGMOID_<WORD>` to `test-output.txt`.

On the practice table it gives `SIGMOID_DEMO`, matching `train-output.txt`.

## Run
```
cd switzerland
cp train-input.csv test-input.csv   # or the real test input
pip install pandas scikit-learn
python main.py                      # -> test-output.txt
```
