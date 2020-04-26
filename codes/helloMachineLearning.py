import numpy as np
import pandas as pd
from sklearn import tree
import graphviz
from sklearn.model_selection import cross_val_score

bank_data = pd.read_csv("../datas/train_set.csv")
first_set = bank_data[['balance', 'duration']]
labels = bank_data['y']
# print(first_set)
# decision tree begins
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=4)
clf = clf.fit(first_set, labels)
scores_treeBD_d4 = cross_val_score(clf, first_set, labels, cv=10)  # 10-means cross validate
print(scores_treeBD_d4)
# visualize
feature_name = ['balance', 'duration']
firstTree_dot = tree.export_graphviz(
    clf
    , out_file=None
    , feature_names=feature_name
    , class_names=['not_buy', 'buy']
)
graph = graphviz.Source(firstTree_dot)
graph.render("../output/TreeForBalanceAndDuration")

# write scores into file
with open("../output/scoresOfTrees.txt", "w") as scoreFile:
    scoreFile.write("Scores of TreeForBalanceAndDuration, depth=4\n")
    scoreFile.write(str(scores_treeBD_d4))
    scoreFile.write("\n")
