from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import numpy as np

epsilon = 0.1
points = np.array([
    [1 + 2 * epsilon, 1],
    [4, 1],
    [5 + 2 * epsilon, 1],
    [6,1],
    [7 - epsilon, 1] 
])

linked = linkage(points, method='complete')

plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', labels = ['d1','d2','d3','d4','d5'], distance_sort = 'descending', show_leaf_counts = True)

plt.title('Dendrogram')
plt.xlabel('Data Point')
plt.ylabel('Euclidean Distance')
plt.show()