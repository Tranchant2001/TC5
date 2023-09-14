# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 16:46:13 2023

@author: jules
"""

import numpy as np

N = 5

X = np.zeros((N, N))
k = 0

for i in range(N):
    for j in range(N):
        k += 1
        X[i,j] = k

print(X)

X = X.reshape(N*N)


A = np.zeros((N*N, N*N))

for i in range(N):
    for j in range(N):
        
        idx = i*N + j
        
        if j<(N-1):
            
            A[idx ,idx + 1] = 1
            
        else :

            A[idx ,idx - j] = 1
            


B = np.zeros((N*N, N*N))

for i in range(N):
    for j in range(N):
        
        idx = i*N + j
        
        if j>0:
            
            B[idx ,idx - 1] = 1
            
        else :

            B[idx, idx + (N-1)] = 1
            


C = np.zeros((N*N, N*N))

for i in range(N):
    for j in range(N):
        
        idx = i*N + j
        
        C[idx, (idx + N)%(N*N)] = 1
            


D = np.zeros((N*N, N*N))

for i in range(N):
    for j in range(N):
        
        idx = i*N + j
        
        D[idx, (idx+(N*N - N))%(N*N)] = 1
            


test = A + B + C + D


M = np.zeros((N*N, N*N))
for i in range(N):
    for j in range(N):
        
        idx = i*N + j
        
        # Treat x-rotation
        if j<(N-1):
            
            M[idx ,idx + 1] = 1
            
        else :

            M[idx ,idx - j] = 1
            
        if j>0:
            
            M[idx ,idx - 1] = 1
            
        else :

            M[idx, idx + (N-1)] = 1
        
        # Treat y-rotation
        M[idx, (idx + N)%(N*N)] = 1
        M[idx, (idx+(N*N - N))%(N*N)] = 1
        

res = np.dot(D,X)
res = res.reshape((N,N))

print(res)
        
        
        
        
        
        
