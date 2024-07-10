import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
clusters_range = range(1, 10)
inertia_values = []

for n in clusters_range:
    kmeans = KMeans(n_clusters=n,random_state=42)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)
    
plt.plot(clusters_range, inertia_values, marker = 'o')
plt.xlabel('n_clusters')
plt.ylabel('intertia_values')
plt.show()