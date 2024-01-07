import numpy as np

a = [[1,2,3],[4,5,6],[7,8,9]]
b = np.array(a, dtype=float)

print(b)

b[1,:2] = 0.

print(b)

print(type(b))

c = np.where(b < 5, -1.0, b)

print(c)