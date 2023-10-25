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
Version answering to the Poisson equation.

"""
### PACKAGES    ###

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from numba import njit


### FONCTIONS   ###

def initialization_Uz(x, y):

    return (x - L)*x*(y - L)*y  


@njit
def update_SOR(u, b, w):

    uk1 = np.zeros((N, N), dtype=float)

    for i in range(1,N-1):
        for j in range(1, N-1):
            uk1[i][j] = (1 - w)*u[i][j] +  w*0.25*(uk1[i-1][j] +uk1[i][j-1] + u[i+1][j] + u[i][j+1] - b[i][j])

    return uk1
    

@njit
def boundary_cond(U_field):
    Uf = np.zeros(U_field.shape, dtype=float)
    Uf[1:-1,1:-1] = U_field[1:-1,1:-1]
    return Uf


def convergence_error(un, un1):

    minus_arr = un1 - un
    error = np.sqrt(np.sum(minus_arr*minus_arr))
    return error


def plot_scalar_field(some_array, **kwargs):
    """
    Plot ta color map of a scalar field contained in a Numpy array.
    """    
    # Create a figure and axis for the animation
    fig, ax = plt.subplots()
    
    # Plot the scalar field
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    if 'title' in kwargs.keys():
        ax.set_title(kwargs.get('title'))
    
    image = ax.imshow(some_array, extent=(0, L, 0, L), origin='lower', cmap='viridis')
    fig.colorbar(image)

    if 'saveaspng' in kwargs.keys():
        plt.savefig(dirpath+"/results_3_SOR/"+kwargs.get('saveaspng'), dpi=128, bbox_inches="tight")
    
    if 'pause' in kwargs.keys():
        plt.pause(kwargs.get("pause"))

    plt.close(fig)


### MAIN   ###

#Chemin absolu du fichier .py qu'on execute
fullpath = os.path.abspath(__file__)
#Chemin absolu du dossier contenant le .py qu'on execute
dirpath = os.path.dirname(fullpath)

# Parameters of the problem
L = 1.0     # Length of the (square shaped) domain (m)

# Initial Parameters of the simulation
#N_list = np.array([4,5,6,7,8,12,16,24,32,64,128,256,512]) # Number of steps for each space axis
N_list = np.array([20])
len_Nl = N_list.shape[0]

# Useful constants
dpdz = 1000
mu = 1e-3
beta = dpdz/mu
piL_p2 = (np.pi/L)**2 #pi over L squared

# Maximum number of iterations
niter = 100
# Norm threshold
norm_th = 1e-3


for k in range(len_Nl):

    N = N_list[k]

    # Space step is then:
    dx = L/N

    # omega parameter evaluation of the SOR method
    omega = 2*(1 - math.pi/N - (math.pi/N)**2)

    # Create mesh grid
    x = np.linspace(0, L, N, endpoint=True)
    y = np.linspace(0, L, N, endpoint=True)
    X, Y = np.meshgrid(x, y)


    # Initial Uz field
    U = initialization_Uz(X, Y)
    Ushape = U.shape

    # beta in a horizontal vector
    beta_arr = np.full((N,N), beta)

    # Plot of the initial U field. I crop the borders because they are source of peak instability.
    #U_crop = np.delete(np.delete(U, [0], 0), [0], 1)
    plot_scalar_field(U, title="$U_{0}$", saveaspng="N"+str(N)+"_Uindex-0.png", pause=3)
    
    eps = 1.
    kiter = 1
    while kiter < niter+1 and eps > norm_th:

        Up1 = update_SOR(U, beta_arr, omega)
        Up1 = boundary_cond(Up1)
        eps = convergence_error(U, Up1)/abs(np.sum(Up1))
        #U_crop = np.delete(np.delete(np.reshape(vecUp1, Ushape), [0], 0), [0], 1)
        plot_scalar_field(Up1, title=f"U n°{kiter}\n normalized err={eps:3f}", saveaspng="N"+str(N)+"_Uindex-"+str(kiter)+".png", pause=3)
        U = Up1
        kiter += 1