from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


X, y = make_blobs(n_samples = 100, centers = 3)


kmeans = KMeans(n_clusters=3)
kmeans.fit(X) 
print(kmeans.labels_)

plt.scatter(X[:,0], X[:,1], c = kmeans.labels_)
plt.show() 
