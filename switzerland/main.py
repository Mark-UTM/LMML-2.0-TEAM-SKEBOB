import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

TARGET = "slab_released"

def word_from_tree(clf, names):
    t = clf.tree_
    return "".join(names[f][0].upper() for node, f in enumerate(t.feature) if t.children_left[node] != -1)

def main():
    d = pd.read_csv("test-input.csv")
    y = d[TARGET] if TARGET in d.columns else d.iloc[:, -1]
    X = d.drop(columns=[y.name])
    clf = DecisionTreeClassifier(random_state=0).fit(X, y)
    names = list(X.columns)
    print(export_text(clf, feature_names=names))
    with open("test-output.txt", "w") as f:
        f.write(f"SIGMOID_{word_from_tree(clf, names)}\n")

if __name__ == "__main__":
    main()
