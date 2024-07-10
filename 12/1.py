import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage

iris = load_iris()
X = iris.data

km = KMeans(n_clusters=5, random_state=90)
km.fit(X)
labels_ = km.labels_

# plt.scatter(X[:, 0], X[:, 1], c = labels_)
# plt.show()

centroids = km.cluster_centers_
Z = linkage(centroids, method = "ward")
dendrogram(Z, labels=['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'])
plt.title("Hierachical")
plt.xlabel("cluster")
plt.ylabel("Distance")
plt.show()