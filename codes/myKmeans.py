import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def distEclud(vec_a, vec_b):
    sum = 0.0
    # print(len(vec_a))
    vec_a = vec_a.tolist()
    vec_b = vec_b.tolist()[0]
    # print(vec_a)
    # print(vec_b)
    for i in range(0, len(vec_a)):
        sum += (vec_a[i]-vec_b[i])**2
    return np.sqrt(sum)


def randCent(dataset, k):
    n = np.shape(dataset)[1]
    centroids = np.mat(np.zeros([k, n]))
    for i in range(n):
        maxi = max(dataset[:, i])
        mini = min(dataset[:, i])
        centroids[:, i] = mini + (maxi - mini) * np.random.random([k, 1])
    return centroids


def my_k_means(data_set, k):
    m = np.shape(data_set)[0]  # 样本数量
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroids = randCent(data_set, k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            mindist = np.inf
            for j in range(k):
                distj = distEclud(data_set[i, :], centroids[j, :])
                if distj < mindist:
                    mindist = distj
                    minj = j
            if clusterAssment[i, 0] != minj:
                cluster_changed = True
            clusterAssment[i, :] = minj, mindist**2
        for cent in range(k):
            data_cent = data_set[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(data_cent, axis=0)
    return centroids, clusterAssment

frog_data = pd.read_csv("../datas/Frogs_MFCCs.csv")
first_set = frog_data[['MFCCs_ 1', 'MFCCs_ 5', 'MFCCs_ 9', 'MFCCs_13', 'MFCCs_17', 'MFCCs_21']]
# print(first_set)
first_set = first_set.values
# print(first_set)
# rc = randCent(first_set, 4)
# print(rc)
t1, t2 = my_k_means(first_set, 4)
print(t2)
