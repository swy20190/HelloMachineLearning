import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.manifold import TSNE


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
original_first_set = first_set
first_set = first_set.values
# clustering
t1, t2 = my_k_means(first_set, 4)
t2 = t2.tolist()
print(t2)
p_labels = []
for i in t2:
    p_labels.append(i[0])
print(p_labels)
r = pd.concat([original_first_set, pd.Series(p_labels, index=original_first_set.index)], axis=1)
r.columns = list(original_first_set.columns) + [u'聚类类别']
r.to_excel("../output/myKMeansSet1.xlsx")
# scoring
tLabel = []
for family in frog_data['Family']:
    if family == "Leptodactylidae":
        tLabel.append(0)
    elif family == "Dendrobatidae":
        tLabel.append(1)
    elif family == "Hylidae":
        tLabel.append(2)
    else:
        tLabel.append(3)
with open("../output/scoreOfMyKMeans.txt", "a") as sf:
    sf.write("By myKMeans, the f-m_score of Set_1 is: " + str(
        metrics.fowlkes_mallows_score(tLabel, p_labels)) + "\n")
    sf.write("By myKMeans, the rand_score of Set_1 is: " + str(
        metrics.adjusted_rand_score(tLabel, p_labels)) + "\n")
# visualize
t_sne_db_1 = TSNE()
t_sne_db_1.fit(original_first_set)
t_sne_db_1 = pd.DataFrame(t_sne_db_1.embedding_, index=original_first_set.index)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
dd = t_sne_db_1[r[u'聚类类别'] == 0]
plt.plot(dd[0], dd[1], 'r.')
dd = t_sne_db_1[r[u'聚类类别'] == 1]
plt.plot(dd[0], dd[1], 'go')
dd = t_sne_db_1[r[u'聚类类别'] == 2]
plt.plot(dd[0], dd[1], 'b*')
dd = t_sne_db_1[r[u'聚类类别'] == 3]
plt.plot(dd[0], dd[1], 'o')
plt.savefig("../output/myKMeansSet_1.png")
plt.clf()
