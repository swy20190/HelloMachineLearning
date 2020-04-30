import pandas as pd
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

frog_data = pd.read_csv("../datas/Frogs_MFCCs.csv")
first_set = frog_data[['MFCCs_ 1', 'MFCCs_ 5', 'MFCCs_ 9', 'MFCCs_13', 'MFCCs_17', 'MFCCs_21']]
outputFile_1 = "../output/clusterResult_1.xls"
model_1 = cluster.KMeans(n_clusters=4, max_iter=10, n_jobs=4)
model_1.fit(first_set)

# print clustering result
r = pd.concat([first_set, pd.Series(model_1.labels_, index=first_set.index)], axis=1)
r.columns = list(first_set.columns) + [u'聚类类别']
print(r)
r.to_excel(outputFile_1)


tsne = TSNE()
tsne.fit(first_set)
tsne = pd.DataFrame(tsne.embedding_, index=first_set.index)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
d = tsne[r[u'聚类类别'] == 0]
plt.plot(d[0], d[1], 'r.')
d = tsne[r[u'聚类类别'] == 1]
plt.plot(d[0], d[1], 'go')
d = tsne[r[u'聚类类别'] == 2]
plt.plot(d[0], d[1], 'b*')
d = tsne[r[u'聚类类别'] == 3]
plt.plot(d[0], d[1], 'o')
plt.savefig("../output/KMeans_clusteringOfSet1.png")
plt.show()
