import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def distEclud(vec_a, vec_b):
    return sum((vec_a - vec_b)**2)**0.5


def randCent(dataset, k):
    n = np.shape(dataset)[1]
    centroids = np.mat(np.zeros([k, n]))
    for i in range(n):
        maxi = max(dataset[:, i])
        mini = min(dataset[:, i])
        centroids[:, i] = mini + (maxi - mini) * np.random.random([k, 1])
    return centroids

frog_data = pd.read_csv("../datas/Frogs_MFCCs.csv")
first_set = frog_data[['MFCCs_ 1', 'MFCCs_ 5', 'MFCCs_ 9', 'MFCCs_13', 'MFCCs_17', 'MFCCs_21']]
# print(first_set)
first_set = first_set.values
# print(first_set)
jj = randCent(first_set, 4)
print(jj)
