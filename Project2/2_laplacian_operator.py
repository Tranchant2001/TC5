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

def get_Uz_field(X, Y):
    
    u = np.sin(np.pi * X / L) * np.sin(np.pi * Y / L)
    
    return u    


def laplacian_operator(u):
        
    u_xx = np.zeros_like(u)
    u_yy = np.zeros_like(u)

    u_xx = (np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)) / (dx**2)
    u_yy = (np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)) / (dx**2)

    laplacian_u = u_xx + u_yy

    return laplacian_u


def analy_laplacian(u):
    
    return -2*(piL_p2)*u


def error_estimation_laplacian(u, A2u, a_A2u):

    rel_diff = (a_A2u - A2u).sum() / u.sum()
    leading_error = ((dx*piL_p2)**2/3)

    return rel_diff, leading_error


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
        plt.savefig(dirpath+"/results_2-laplacian-operator/"+kwargs.get('saveaspng'), dpi=128, bbox_inches="tight")
    plt.close(fig)


### MAIN   ###

#Chemin absolu du fichier .py qu'on execute
fullpath = os.path.abspath(__file__)
#Chemin absolu du dossier contenant le .py qu'on execute
dirpath = os.path.dirname(fullpath)

# Parameters of the problem
L = 1.0     # Length of the (square shaped) domain (m)

# Initial Parameters of the simulation
N_list = np.array([4,5,6,7,8,12,16,24,32,64,128,256,512]) # Number of steps for each space axis
len_Nl = N_list.shape[0]

# Useful constants
piL_p2 = (np.pi/L)**2

# Put here the maximum time you want to spend on the computation.
max_time_computation = dt.timedelta(hours=6)

error_list = np.zeros(len_Nl, dtype=float)
leading_error_list = np.zeros(len_Nl, dtype=float)

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

    # Laplacian estimated
    A2U = laplacian_operator(U)

    # Analytical Laplacian
    analy_A2U = analy_laplacian(U)

    # I crop the borders because they are source of peak instability.
    U_crop = np.delete(np.delete(U, [0], 0), [0], 1)
    A2U_crop = np.delete(np.delete(A2U, [0], 0), [0], 1)
    a_A2U_crop = np.delete(np.delete(analy_A2U, [0], 0), [0], 1)
    plot_scalar_field(U_crop, title="U field (N="+str(N)+")", saveaspng="N"+str(N)+"_U_field.png")
    plot_scalar_field(A2U_crop, title="Estimated LU field (N="+str(N)+")", saveaspng="N"+str(N)+"_estLU_field.png")
    plot_scalar_field(a_A2U_crop, title="Analytic LU field (N="+str(N)+")", saveaspng="N"+str(N)+"_analyLU_field.png")
    plot_scalar_field(a_A2U_crop-A2U_crop, title="Diffference error (N="+str(N)+")", saveaspng="N"+str(N)+"_diff-error_field.png")

    diff_error, leading_error = error_estimation_laplacian(U_crop, A2U_crop, a_A2U_crop)

    error_list[k] = abs(diff_error)
    leading_error_list[k] = leading_error

dx_list = L/N_list
plt.plot(dx_list, error_list, label="Difference")
plt.plot(dx_list, leading_error_list, label="Leading term of the error")
plt.xlabel("$\Delta x$")
plt.ylabel("Error (a.u.)")
plt.legend()
plt.savefig(dirpath+"/results_2-laplacian-operator/erros_f-of-dx.png", dpi=128, bbox_inches="tight")
plt.show()



plt.plot(N_list, error_list, label="Difference")
plt.plot(N_list, leading_error_list, label="Leading term of the error")
plt.xlabel("N")
plt.ylabel("Error (a.u.)")
plt.legend()
plt.savefig(dirpath+"/results_2-laplacian-operator/erros_f-of-N.png", dpi=64, bbox_inches="tight")
plt.show()