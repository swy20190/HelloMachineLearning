import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import preprocessing
import graphviz
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model

bank_data = pd.read_csv("../datas/train_set.csv")
# first set of balance and duration
first_set = bank_data[['age', 'balance']]
labels = bank_data['y']
# print(first_set)
# decision tree begins
# decision tree max_depth=4
clf_treeAB_d4 = tree.DecisionTreeClassifier(criterion="entropy", max_depth=4)
clf_treeAB_d4 = clf_treeAB_d4.fit(first_set, labels)
scores_treeAB_d4 = cross_val_score(clf_treeAB_d4, first_set, labels, cv=10)  # 10-means cross validate
print(scores_treeAB_d4)
# visualize
feature_name = ['age', 'balance']
class_name = ['not_buy', 'buy']
treeAB_d4_dot = tree.export_graphviz(
    clf_treeAB_d4
    , out_file=None
    , feature_names=feature_name
    , class_names=class_name
)
graph = graphviz.Source(treeAB_d4_dot)
graph.render("../output/TreeForAgeAndBalanceD4")
# decision tree max_depth=8
clf_treeAB_d8 = tree.DecisionTreeClassifier(criterion="entropy", max_depth=8)
clf_treeAB_d8 = clf_treeAB_d8.fit(first_set, labels)
scores_treeAB_d8 = cross_val_score(clf_treeAB_d8, first_set, labels, cv=10)  # 10-means cross validate
print(scores_treeAB_d8)
# visualize
treeAB_d8_dot = tree.export_graphviz(
    clf_treeAB_d8
    , out_file=None
    , feature_names=feature_name
    , class_names=class_name
)
graph = graphviz.Source(treeAB_d8_dot)
graph.render("../output/TreeForAgeAndBalanceD8")

# second set of duration, campaign, pdays, as well as previous
# performs much more better than the 1st one
second_set = bank_data[['duration', 'campaign', 'pdays', 'previous']]
# decision tree max_depth=4
clf_treeDrtCpnPdsPrev_d4 = tree.DecisionTreeClassifier(criterion="entropy", max_depth=4)
clf_treeDrtCpnPdsPrev_d4 = clf_treeDrtCpnPdsPrev_d4.fit(second_set, labels)
scores_treeDrtCpnPdsPrev_d4 = cross_val_score(
    clf_treeDrtCpnPdsPrev_d4, second_set, labels, cv=10
)  # 10-means cross validate
print(scores_treeDrtCpnPdsPrev_d4)
# visualize
feature_name_2 = ['duration', 'campaign', 'pdays', 'previous']
treeDrtCpnPdsPrev_d4_dot = tree.export_graphviz(
    clf_treeDrtCpnPdsPrev_d4
    , out_file=None
    , feature_names=feature_name_2
    , class_names=class_name
)
graph = graphviz.Source(treeDrtCpnPdsPrev_d4_dot)
graph.render("../output/TreeForDrtCpnPdsPrevD4")
# decision tree max_depth=8
clf_treeDrtCpnPdsPrev_d8 = tree.DecisionTreeClassifier(criterion="entropy", max_depth=8)
clf_treeDrtCpnPdsPrev_d8 = clf_treeDrtCpnPdsPrev_d8.fit(second_set, labels)
scores_treeDrtCpnPdsPrev_d8 = cross_val_score(
    clf_treeDrtCpnPdsPrev_d8, second_set, labels, cv=10
) # 10-means cross validate
print(scores_treeDrtCpnPdsPrev_d8)
# visualize
treeDrtCpnPdsPrev_d8_dot = tree.export_graphviz(
    clf_treeDrtCpnPdsPrev_d8
    , out_file=None
    , feature_names=feature_name_2
    , class_names=class_name
)
graph = graphviz.Source(treeDrtCpnPdsPrev_d8_dot)
graph.render("../output/TreeForDrtCpnPdsPrevD8")
# decision tree ends
# naive bayes begins
gnb_1 = GaussianNB()
gnb_1.fit(first_set, labels)
scores_GNB_1 = cross_val_score(gnb_1, first_set, labels, cv=10)  # 10-means cross validate
print(scores_GNB_1)

gnb_2 = GaussianNB()
gnb_2.fit(second_set, labels)
scores_GNB_2 = cross_val_score(gnb_2, second_set, labels, cv=10)  # 10-means cross validate
print(scores_GNB_2)
# naive bayes ends
# Ridge begins
# performs very bad
reg_1_1 = linear_model.Ridge(alpha=1)
reg_1_1.fit(first_set, labels)
print(reg_1_1.coef_)
scores_Reg_1_1 = cross_val_score(reg_1_1, first_set, labels, cv=10)
print(scores_Reg_1_1)
reg_1_5 = linear_model.Ridge(alpha=.5)
reg_1_5.fit(first_set, labels)
print(reg_1_5.coef_)
scores_Reg_1_5 = cross_val_score(reg_1_5, first_set, labels, cv=10)
print(scores_Reg_1_5)

# write scores into file
with open("../output/scoresOfTrees.txt", "w") as scoreFile:
    scoreFile.write("Scores of TreeForAgeAndBalance, depth=4\n")
    scoreFile.write(str(scores_treeAB_d4))
    scoreFile.write("\n")
    scoreFile.write("Scores of TreeForAgeAndBalance, depth=8\n")
    scoreFile.write(str(scores_treeAB_d8))
    scoreFile.write("\n")
    scoreFile.write("Scores of TreeForDrtCpnPdsPrev, depth=4\n")
    scoreFile.write(str(scores_treeDrtCpnPdsPrev_d4))
    scoreFile.write("\n")
    scoreFile.write("Scores of TreeForDrtCpnPdsPrev, depth=8\n")
    scoreFile.write("\n")

with open("../output/scoresOfBayes.txt", "w") as ya_scoreFile:
    ya_scoreFile.write("Scores of GausNBForAgeAndBalance\n")
    ya_scoreFile.write(str(scores_GNB_1))
    ya_scoreFile.write("\n")
    ya_scoreFile.write("Scores of GausNBForDrtCpnPdsPrev\n")
    ya_scoreFile.write(str(scores_GNB_2))
    ya_scoreFile.write("\n")

with open("../output/scoresOfReg.txt", "w") as sor:
    sor.write("Coef of set 1 alpha=1 is:\n")
    sor.write(str(reg_1_1.coef_)+'\n')
    sor.write("Scores of RidgeForAgeAndBalance alpha=1\n")
    sor.write(str(scores_Reg_1_1))
    sor.write("\n")
    sor.write("Coef of set 1 alpha=0.5 is:\n")
    sor.write(str(reg_1_5.coef_)+'\n')
    sor.write("Scores of RidgeForAgeAndBalance alpha=0.5\n")
    sor.write(str(scores_Reg_1_5))
    sor.write('\n')
