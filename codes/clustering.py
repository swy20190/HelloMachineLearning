import pandas as pd
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def k_means(data_set, output_file, png_file):
    model = cluster.KMeans(n_clusters=4, max_iter=100, n_jobs=4)
    model.fit(data_set)
    r = pd.concat([data_set, pd.Series(model.labels_, index=data_set.index)], axis=1)
    r.columns = list(data_set.columns) + [u'聚类类别']
    print(r)
    r.to_excel(output_file)
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
    plt.show()


frog_data = pd.read_csv("../datas/Frogs_MFCCs.csv")
first_set = frog_data[['MFCCs_ 1', 'MFCCs_ 5', 'MFCCs_ 9', 'MFCCs_13', 'MFCCs_17', 'MFCCs_21']]
k_means(first_set, "../output/kMeansSet_1.xlsx", "../output/kMeansSet_1.png")
