import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

X_train = np.array([[1,1,0,0,0,0,0],[0,1,1,1,0,0,0],
                   [0,0,0,0,1,1,0],[0,1,0,0,0,1,1]])

test = np.array([[0,2,1,0,0,1,0]])
y_train = np.array([1,1,0,0])

model1 = MultinomialNB()
model1.fit(X_train,y_train)

print(model1.predict(test))

model2 = BernoulliNB()
model2.fit(X_train,y_train)

print(model2.predict(test))