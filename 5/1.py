import numpy as np

A = np.matrix([[1, 2, 3], [4, 5, 6]])

print(A)
print("-------------")
print(A.T) 
print("-------------")
print(A.I)

print("-------------")

B = np.matrix([[0, 1, 2], [3, 4, 5]])
print(A + B," \n-----------------\n" ,A - B) 
print("-------------")

C = np.matrix([[1, 2], [3, 4]])
print(A * C) 