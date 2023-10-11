# -*- coding: utf-8 -*-
#%%

"""
Projet n°1 du TC5: Numerical Methods du Master PPF.
Travail en binôme:
    Jules Chaveyriat
    Martin Guillon

Created on Thu Sep 14 12:26:59 2023
Last updated 21/09/2023
v1.1

DESCRIPTION:
Version with the explicit method of Backward Euler.
"""

### PACKAGES    ###

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import datetime as dt
from numba import jit


### FUNCTIONS    ###

def set_velocity_field(L, N, X, Y):
    
    u = np.cos(4 * np.pi * X) * np.sin(4 * np.pi * Y)
    v = -np.sin(4 * np.pi * X) * np.cos(4 * np.pi * Y)
    
    return u, v

def set_initial_potential(X, Y):
    
    phi = np.where(np.sqrt((X - 0.5)**2 + (Y - 0.5)**2) < 0.3, 1.0, 0.0)
    
    return phi


def get_metric(phi):
    
    # Calculate standard deviation and average of phi
    std_phi = np.std(phi)
    avg_phi = np.mean(phi)
    
    # Calculate the metric
    metric = std_phi / avg_phi
    
    return metric


def plot_potential_field(phi, time, metric, **kwargs):
    
    # Create a figure and axis for the animation
    fig, ax = plt.subplots()
    
    # Plot the scalar field
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Diffusion of Scalar Field \n Time: {time:.2f} s \n Metric: {metric:.5f}')
    ax.imshow(phi, extent=(0, L, 0, L), origin='lower', cmap='viridis')

    if 'saveaspng' in kwargs.keys():
        plt.savefig(kwargs.get('saveaspng'), dpi=108, bbox_inches="tight")
    plt.pause(0.4)
    plt.close(fig)


def assemble_linear_system(phi):
    # Initialize matrix A as a sparse matrix
    A = lil_matrix((Nx * Ny, Nx * Ny))

    # Initialize vector b
    b = np.zeros(Nx * Ny)

    Dx_term = D * (2 / dx**2)
    Dy_term = D * (2 / dy**2)
     
    # Loop over grid points
    for i in range(Nx):
        for j in range(Ny):
            # Calculate index
            
            idx = i*Nx + j
            
            # Calculate coefficients for spatial derivatives
            Vx_term = u[i, j] / (2 * dx)
            Vy_term = v[i, j] / (2 * dy)
    
            # Diagonal (self) entry
            A[idx, idx] = 1 + delta_t * (Dx_term + Dy_term)
            
            
            # Treat x-rotation
            
            if j<(N-1):
                
                A[idx ,idx + 1] =  - delta_t * (Dx_term - Vx_term)   # i + 1
                
            else :
 
                A[idx ,idx - j] =  - delta_t * (Dx_term - Vx_term)   
            
            if j>0:
                
                A[idx ,idx - 1] = - delta_t * (Dx_term + Vx_term)    # i - 1
                
            else :

                A[idx, idx + (N-1)] = - delta_t * (Dx_term + Vx_term)
            
            # Treat y-rotation
            A[idx, (idx + N)%(N*N)] = - delta_t * (Dy_term + Vy_term)    # j + 1
            A[idx, (idx + (N*N - N))%(N*N)] = - delta_t * (Dy_term - Vy_term) # j - 1
         
        
            # Set the right-hand side vector
            b[idx] = phi[i, j]  # phi is the solution at the previous time step
    
    return A, b


def simulation(N, delta_t, phi, T_comput:dt.timedelta=dt.timedelta(days=7), showAndSave=True):
    
    metric = get_metric(phi)
    
    # Set time to zero
    time = 0
    
    # Set frame counting to zero
    frame = 0

    start_datetime = dt.datetime.now()
    step_datetime = dt.datetime.now()    
    
    while metric >= 0.05 and (metric < 3 or np.mean(phi) < 1) and step_datetime - start_datetime < T_comput:
        
        time = frame * delta_t
        

        
        if frame%50 == 0:
            print(frame)
            if showAndSave:
                plot_potential_field(phi, time, metric, saveaspng=pictures_save_path+'/'+str(frame)+"_phi_field.png")

        # Assemble the linear system (A) and right-hand side (rhs)
        A, rhs = assemble_linear_system(phi)

        # Solve the linear system using an efficient solver
        phi_np1 = spsolve(A, rhs)

        # Update phi for the next time step
        phin = phi_np1
        
        phi = phin.reshape((Nx, Ny))

        step_datetime = dt.datetime.now()
        metric = get_metric(phi)
        frame += 1
        
    if metric >= 3 or np.mean(phi) > 1:        
        print("Warning: The simulation stopped running because a divergence was detected (metric >= 3 or np.mean(phi) > 1).")
        print(f"\tParameters: (L, D, N, $\Delta t$)=({L}, {D}, {N}, {delta_t})")
        print("\tSimulation duration: "+str(step_datetime - start_datetime))
        print(f"\tVirtual stop time: {time:.2f} s")        
        print(f"\tVirtual stop frame: {frame}")
        print(f"\tMetric: {metric:5f}")

    elif step_datetime - start_datetime >= T_comput:
        print("Warning: The simulation stopped running because the max duration of simulation ("+str(T_comput)+") was reached.")
        print(f"\tParameters: (L, D, N, $\Delta t$)=({L}, {D}, {N}, {delta_t})")
        print("\tSimulation duration: "+str(step_datetime - start_datetime))
        print(f"\tVirtual stop time: {time:.2f} s")
        print(f"\tVirtual stop frame: {frame}")
        print(f"\tMetric: {metric:5f}")

    else:
        print("Success: The simulation stopped running because the field was homogeneous enough (metric < 0.05).")
        print(f"\tParameters: (L, D, N, $\Delta t$)=({L}, {D}, {N}, {delta_t})")
        print("\tSimulation duration: "+str(step_datetime - start_datetime))
        print(f"\tVirtual stop time: {time:.2f} s")        
        print(f"\tVirtual stop frame: {frame}")
        print(f"\tMetric: {metric:5f}")
    
    return time


### MAIN    ###

#Chemin absolu du fichier .py qu'on execute
fullpath = os.path.abspath(__file__)
#Chemin absolu du dossier contenant le .py qu'on execute
dirpath = os.path.dirname(fullpath)

# Parameters of the problem
L = 1.0     # Length of the (square shaped) domain (m)
D = 0.001   # Diffusion coeffiscient

# Initial Parameters of the simulation
Lx = L
Ly = L
N = 60   # Number of steps for each space axis
delta_t = 0.001   # Time step
Nx = N
Ny = N
dx = Lx / Nx # Spatial step size in the x-direction
dy = Ly / Ny  # Spatial step size in the y-direction

# Put here the maximum time you want to spend on the computation.
max_time_computation = dt.timedelta(hours=6)

# Display and save parameters
show_and_save = True
pictures_save_path = dirpath+'/outputs_program_backwardEuler'

# Create mesh grid
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y)

# Velocity field
u, v = set_velocity_field(L, N, X, Y)

# Initial scalar field
phi = set_initial_potential(X, Y)

result = simulation(N, delta_t, phi, max_time_computation, show_and_save)
print(f"Total time passed in the simulation: {result:.3f} seconds ")

