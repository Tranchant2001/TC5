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
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from numba import njit


### FONCTIONS   ###

def get_Uz_field(X, Y):
    
    u = np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)
    
    return u    


def snd_x_deriv(u):
    
    return (np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)) / (dx**2)


def snd_y_deriv(u):
    
    return (np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)) / (dx**2)


def laplacian_operator(u):

    return snd_x_deriv(u) + snd_y_deriv(u)


def analy_laplacian(u):
    
    return -2*(piL_p2)*u


def update_jacobi(vecu, b_vec):

    return -0.25*(-np.roll(vecu, -N) - np.roll(vecu, -1) - np.roll(vecu, 1) - np.roll(vecu, N) + b_vec)


def error_estimation_laplacian(u, ux4_plus_uy4, A2u, a_A2u):

    rel_diff = (a_A2u - A2u).sum() / (N**2)
    leading_error = ((dx*piL_p2)**2/6)*(u.sum()) / (N**2)

    # here calculus of the numerical leading term
    num_lead_term_sum = (dx**2/12)*(ux4_plus_uy4.sum()) / (N**2)

    return rel_diff, leading_error, num_lead_term_sum


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
    ax.imshow(some_array, extent=(0, L, 0, L), origin='lower', cmap='viridis')

    if 'saveaspng' in kwargs.keys():
        plt.savefig(dirpath+"/results_3-solution-poisson/"+kwargs.get('saveaspng'), dpi=128, bbox_inches="tight")
    
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

#Jacobi number of iterations
n_jacobi = 10


for k in range(len_Nl):

    N = N_list[k]
    print(N)

    # Space step is then:
    dx = L/N

    # Create mesh grid
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y)


    # Initial Uz field
    U = get_Uz_field(X, Y)
    Ushape = U.shape

    # vecteur linearized
    vecU = U.flatten()
    len_vecU = vecU.shape[0]
    vec_beta = np.full(len_vecU, beta)

    # Plot of the initial U field. I crop the borders because they are source of peak instability.
    U_crop = np.delete(np.delete(U, [0], 0), [0], 1)
    plot_scalar_field(U_crop, title="$U_{0}$", saveaspng="N"+str(N)+"_Uindex-0.png", pause=3)
    
    for k in range(1,n_jacobi+1):
        
        vecU = update_jacobi(vecU, vec_beta)
        U_crop = np.delete(np.delete(np.reshape(vecU, Ushape), [0], 0), [0], 1)
        plot_scalar_field(U_crop, title="$U_{"+str(k)+"}$", saveaspng="N"+str(N)+"_Uindex-"+str(k)+".png", pause=3)

    #plot_scalar_field(U_crop, title="U field (N="+str(N)+")", saveaspng="N"+str(N)+"_U_field.png")

"""
dx_list = L/N_list
plt.loglog(dx_list, error_list, label="Difference $|\Delta u - \Delta u^{num}|\slash N^2$")
plt.loglog(dx_list, leading_error_list, label="Analytical $sum(\epsilon_{ij})\slash N^2$")
plt.loglog(dx_list, num_lead_list, label="Numerical $sum(\epsilon_{ij}^{num})\slash N^2$")
plt.xlabel("$log(\Delta x)$")
plt.ylabel("Error (a.u.)")
plt.legend()
plt.savefig(dirpath+"/results_2-laplacian-operator/erros_logf-of-logdx.png", dpi=128, bbox_inches="tight")
plt.show()
"""