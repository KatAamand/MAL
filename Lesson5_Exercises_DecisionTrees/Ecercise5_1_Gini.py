import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn import tree

data = pd.DataFrame({
    "x1": [3, 3, 3, 4, 5],
    "x2": [1, 2, 1, 2, 1], 
    "y": ["A", "B", "B", "B", "A"]
})

# Calculating Gini-koefficienten
def gini_impurity(y): 
    classes, counts = np.unique(y, return_counts=True)
    prob_sq = (counts / counts.sum()) ** 2
    return 1 - prob_sq.sum()

gini_dataset = gini_impurity(data["y"])
print(f"Gini-koefficienten for hele datasættet: {gini_dataset}")

X = data[["x1", "x2"]]
y = data["y"]
clf = DecisionTreeClassifier(max_depth=1, criterion="gini")
clf.fit(X, y)

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=["x1", "x2"], class_names=["A", "B"], filled = True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree")
graph