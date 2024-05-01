import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 4, 3, 2, 1])

norm_x = np.linalg.norm(x)
print("Euclidean norm of vector x:", norm_x)

distance = np.linalg.norm(x - y)
print("Euclidean distance between x and y:", distance)

std = np.std(x)
print("Standard deviation of values:", std)

corr_matrix = np.corrcoef(x, y)
corr_coefficient = corr_matrix[0, 1]
print("Correlation coefficient between x and y:", corr_coefficient)
