import pandas as pd
from sklearn import cluster
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def k_means(data_set, output_file, png_file, t_labels, score_file, set_name):
    model = cluster.KMeans(n_clusters=4, max_iter=100, n_jobs=4, init="k-means++")
    model.fit(data_set)
    # print(list(model.labels_))
    p_labels = list(model.labels_)
    r = pd.concat([data_set, pd.Series(model.labels_, index=data_set.index)], axis=1)
    r.columns = list(data_set.columns) + [u'聚类类别']
    print(r)
    r.to_excel(output_file)
    with open(score_file, "a") as sf:
        sf.write("By k-means, the f-m_score of " + set_name + " is: " + str(metrics.fowlkes_mallows_score(t_labels, p_labels))+"\n")
        sf.write("By k-means, the rand_score of " + set_name + " is: " + str(metrics.adjusted_rand_score(t_labels, p_labels))+"\n")
    t_sne = TSNE()
    t_sne.fit(data_set)
    t_sne = pd.DataFrame(t_sne.embedding_, index=data_set.index)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    dd = t_sne[r[u'聚类类别'] == 0]
    plt.plot(dd[0], dd[1], 'r.')
    dd = t_sne[r[u'聚类类别'] == 1]
    plt.plot(dd[0], dd[1], 'go')
    dd = t_sne[r[u'聚类类别'] == 2]
    plt.plot(dd[0], dd[1], 'b*')
    dd = t_sne[r[u'聚类类别'] == 3]
    plt.plot(dd[0], dd[1], 'o')
    plt.savefig(png_file)
    plt.clf()
    # plt.show()


frog_data = pd.read_csv("../datas/Frogs_MFCCs.csv")
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

scoreFile = "../output/scoreOfClustering.txt"
first_set = frog_data[['MFCCs_ 1', 'MFCCs_ 5', 'MFCCs_ 9', 'MFCCs_13', 'MFCCs_17', 'MFCCs_21']]
k_means(first_set, "../output/kMeansSet_1.xlsx", "../output/kMeansSet_1.png", tLabel, scoreFile, "Set_1")

second_set = frog_data[['MFCCs_ 3', 'MFCCs_ 7', 'MFCCs_11', 'MFCCs_15', 'MFCCs_19']]
k_means(second_set, "../output/kMeansSet_2.xlsx", "../output/kMeansSet_2.png", tLabel, scoreFile, "Set_2")

# DBSCAN begins
db = cluster.DBSCAN(eps=0.1011, min_samples=115, n_jobs=-1)
db.fit(first_set)
r = pd.concat([first_set, pd.Series(db.labels_, index=first_set.index)], axis=1)
r.columns = list(first_set.columns) + [u'聚类类别']
# print(r)
r.to_excel("../output/dbscanSet_1.xlsx")
p_labels = list(db.labels_)
with open(scoreFile, "a") as sf:
    sf.write("By DBSCAN, the f-m_score of Set_1 is: " + str(
        metrics.fowlkes_mallows_score(tLabel, p_labels)) + "\n")
    sf.write("By DBSCAN, the rand_score of Set_1 is: " + str(
        metrics.adjusted_rand_score(tLabel, p_labels)) + "\n")

t_sne_db_1 = TSNE()
t_sne_db_1.fit(first_set)
t_sne_db_1 = pd.DataFrame(t_sne_db_1.embedding_, index=first_set.index)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
dd = t_sne_db_1[r[u'聚类类别'] == 0]
plt.plot(dd[0], dd[1], 'r.')
dd = t_sne_db_1[r[u'聚类类别'] == 1]
plt.plot(dd[0], dd[1], 'go')
dd = t_sne_db_1[r[u'聚类类别'] == 2]
plt.plot(dd[0], dd[1], 'b*')
dd = t_sne_db_1[r[u'聚类类别'] == -1]
plt.plot(dd[0], dd[1], 'o')
plt.savefig("../output/dbscanSet_1.png")
plt.clf()

# dbscan of set 2
db = cluster.DBSCAN(eps=0.1020, min_samples=90, n_jobs=-1)
db.fit(second_set)
r = pd.concat([second_set, pd.Series(db.labels_, index=second_set.index)], axis=1)
r.columns = list(second_set.columns) + [u'聚类类别']
# print(r)
r.to_excel("../output/dbscanSet_2.xlsx")
p_labels = list(db.labels_)
with open(scoreFile, "a") as sf:
    sf.write("By DBSCAN, the f-m_score of Set_2 is: " + str(
        metrics.fowlkes_mallows_score(tLabel, p_labels)) + "\n")
    sf.write("By DBSCAN, the rand_score of Set_2 is: " + str(
        metrics.adjusted_rand_score(tLabel, p_labels)) + "\n")

t_sne_db_2 = TSNE()
t_sne_db_2.fit(second_set)
t_sne_db_2 = pd.DataFrame(t_sne_db_2.embedding_, index=second_set.index)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
dd = t_sne_db_2[r[u'聚类类别'] == 0]
plt.plot(dd[0], dd[1], 'r.')
dd = t_sne_db_2[r[u'聚类类别'] == 1]
plt.plot(dd[0], dd[1], 'go')
dd = t_sne_db_2[r[u'聚类类别'] == 2]
plt.plot(dd[0], dd[1], 'b*')
dd = t_sne_db_2[r[u'聚类类别'] == -1]
plt.plot(dd[0], dd[1], 'o')
plt.savefig("../output/dbscanSet_2.png")
plt.clf()
