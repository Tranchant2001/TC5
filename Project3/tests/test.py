import numpy as np
import os

print("Test 1")
a = np.array([[0,1,2],[3,4,5],[6,7,8]])
print(a)

print("Test 2")
b = a[:,2]
print(b)
print(b.shape)

print("Test 3")
c = np.tile(b, (1,5))
print(c)
print(c.shape)

print("Test 4")
d = np.tile(b, (5,1)).transpose()
print(d)
print(d.shape)

print("Test 5")
e = d[0,1]
e = 33
print(d)

print("Test 6")
f = np.array([0,2,5,6,87,4,1,3])
print(f)
g = np.delete(f, np.argwhere(f>5))
print(g)

print("Test 7")
h = np.array([[0,2,5],[6, np.NaN,4],[1,3,9]])
print(h)
i = np.isnan(h[1,1])
print(i)
print(type(i))
j = bool(i)
print(type(j))

print("Test 8")
#Chemin absolu du fichier .py qu'on execute
fullpath = os.path.abspath(__file__)
#Chemin absolu du dossier contenant le .py qu'on execute
dirpath = os.path.dirname(fullpath)
print(fullpath)
print(dirpath)
print(dirpath+"\\outputs")
projectpath = os.path.dirname(dirpath)
print(projectpath)

"Test 9"
print(type(fullpath))
print(type(dirpath))
print(type(projectpath))

def testing_docstring(input1:int, input2:float, input3:int)->float:
    """_summary_

    Args:
        input1 (int): _description_
        input2 (float): _description_
        input3 (int): _description_

    Returns:
        float: _description_
    """
    q = input1 // input3
    r = input1 % input3

    return q*input2 + r



