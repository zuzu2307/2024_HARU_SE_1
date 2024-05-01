from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer 
import matplotlib.pyplot as plt

data = load_breast_cancer() 

X = data.data
y = data.target

kmeans = KMeans(n_clusters=2)
kmeans.fit(X) 

print(y)
print(kmeans.labels_)

t = f = 0
for a,b in zip(y, kmeans.labels_):
    if a == b:
        t += 1
    else:
        f += 1
print(t, f)
print(round(100 * f / (t + f), 2))

plt.scatter(X[:,0], X[:,1], c = kmeans.labels_)
plt.show() 
