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
# decision tree max_depth=4
clf_treeBD_d4 = tree.DecisionTreeClassifier(criterion="entropy", max_depth=4)
clf_treeBD_d4 = clf_treeBD_d4.fit(first_set, labels)
scores_treeBD_d4 = cross_val_score(clf_treeBD_d4, first_set, labels, cv=10)  # 10-means cross validate
print(scores_treeBD_d4)
# visualize
feature_name = ['balance', 'duration']
class_name = ['not_buy','buy']
treeBD_d4_dot = tree.export_graphviz(
    clf_treeBD_d4
    , out_file=None
    , feature_names=feature_name
    , class_names=class_name
)
graph = graphviz.Source(treeBD_d4_dot)
graph.render("../output/TreeForBalanceAndDurationD4")
# decision tree max_depth=8
clf_treeBD_d8 = tree.DecisionTreeClassifier(criterion="entropy", max_depth=8)
clf_treeBD_d8 = clf_treeBD_d8.fit(first_set, labels)
scores_treeBD_d8 = cross_val_score(clf_treeBD_d8, first_set, labels, cv=10)  # 10-means cross validate
print(scores_treeBD_d8)
# visualize
treeBD_d8_dot = tree.export_graphviz(
    clf_treeBD_d8
    , out_file=None
    , feature_names=feature_name
    , class_names=class_name
)
graph = graphviz.Source(treeBD_d8_dot)
graph.render("../output/TreeForBalanceAndDurationD8")

# write scores into file
with open("../output/scoresOfTrees.txt", "w") as scoreFile:
    scoreFile.write("Scores of TreeForBalanceAndDuration, depth=4\n")
    scoreFile.write(str(scores_treeBD_d4))
    scoreFile.write("\n")
    scoreFile.write("Scores of TreeForBalanceAndDuration, depth=8\n")
    scoreFile.write(str(scores_treeBD_d8))
    scoreFile.write("\n")
