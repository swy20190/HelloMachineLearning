import numpy as np
from sklearn import tree

first_set = np.loadtxt(
    "../datas/train_set.csv", str, delimiter=",", usecols=(17, 6, 12), skiprows=1
)  # balance & duration
first_set_label = np.loadtxt(
    "../datas/train_set.csv", str, delimiter=",", usecols=17, skiprows=1
)  # 1st labels
# print(first_set)
# decision tree begins
clf = tree.DecisionTreeClassifier()
clf = clf.fit(first_set, first_set_label)
