# -*- coding: utf-8 -*-
#%%

"""
Projet n°2 du TC5: Numerical Methods du Master PPF.
Travail en binôme:
    Jules Chaveyriat
    Martin Guillon

Créé le 11/10/2023.
Mis à jour le 11/10/2023.
v1.1

DESCRIPTION:
Version answering to the laplacian operator discussion.

"""
### PACKAGES    ###

import os
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from numba import njit


### FONCTIONS   ###

def get_velocity_field(X, Y):
    
    u = np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)
    
    return u    


def laplacian_operator(u):
        
    u_xx = (np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)) / dx**2
    u_yy = (np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)) / dx**2

    laplacian_u = u_xx + u_yy

    return laplacian_u


def error_estimation_laplacian(u, A2U):
    
### MAIN   ###

#Chemin absolu du fichier .py qu'on execute
fullpath = os.path.abspath(__file__)
#Chemin absolu du dossier contenant le .py qu'on execute
dirpath = os.path.dirname(fullpath)

# Parameters of the problem
L = 1.0     # Length of the (square shaped) domain (m)

# Initial Parameters of the simulation
N = 400    # Number of steps for each space axis

# Space step is then:
dx = L/N

# Put here the maximum time you want to spend on the computation.
max_time_computation = dt.timedelta(hours=6)

# Create mesh grid
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y)

# Initial velocity field
U = get_velocity_field(X, Y)
