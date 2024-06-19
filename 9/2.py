import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
import mglearn

iris = load_iris()
X = iris.data[:100,2:]
Y = iris.target[:100]

# mglearn.discrete_scatter(X[:,0],X[:,1],Y)
# plt.legend(['setosa','versicolor'], loc='best')
# plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, random_state=0)
svm = LinearSVC()
svm.fit(X_train, Y_train)
plt.figure(figsize=(10,6))
mglearn.plots.plot_2d_separator(svm, X)
plt.xlabel('petal lenght')
plt.ylabel('petal width')
plt.legend(['setosa','versicolor'], loc='best')
plt.show()