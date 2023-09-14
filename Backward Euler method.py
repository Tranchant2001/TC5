# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:26:59 2023

@author: jules
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Parameters of the problem
L = 1.0     # Length of the (square shaped) domain (m)
D = 0.001   # Diffusion coeffiscient

# Initial Parameters of the simulation
Lx = L
Ly = L
N = 60   # Number of steps for each space axis
dt = 0.001   # Time step
Nx = N
Ny = N
dx = Lx / Nx # Spatial step size in the x-direction
dy = Ly / Ny  # Spatial step size in the y-direction

# Create mesh grid
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y)

def set_velocity_field(L, N, X, Y):
    
    u = np.cos(4 * np.pi * X) * np.sin(4 * np.pi * Y)
    v = -np.sin(4 * np.pi * X) * np.cos(4 * np.pi * Y)
    
    return u, v

def set_initial_potential(X, Y):
    
    phi = np.where(np.sqrt((X - 0.5)**2 + (Y - 0.5)**2) < 0.3, 1.0, 0.0)
    
    return phi

u, v = set_velocity_field(L, N, X, Y)

phi = set_initial_potential(X, Y)



def get_metric(phi):
    
    # Calculate standard deviation and average of phi
    std_phi = np.std(phi)
    avg_phi = np.mean(phi)
    
    # Calculate the metric
    metric = std_phi / avg_phi
    
    return metric

def plot_potential_field(phi, time, metric):
    
    # Create a figure and axis for the animation
    fig, ax = plt.subplots()
    
    # Plot the scalar field
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Diffusion of Scalar Field \n Time: {time:.2f} s \n Metric: {metric:.5f}')
    ax.imshow(phi, extent=(0, L, 0, L), origin='lower', cmap='viridis')
    
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
            A[idx, idx] = 1 + dt * (Dx_term + Dy_term)
            
            
            # Treat x-rotation
            
            if j<(N-1):
                
                A[idx ,idx + 1] =  - dt * (Dx_term - Vx_term)   # i + 1
                
            else :
 
                A[idx ,idx - j] =  - dt * (Dx_term - Vx_term)   
            
            if j>0:
                
                A[idx ,idx - 1] = - dt * (Dx_term + Vx_term)    # i - 1
                
            else :

                A[idx, idx + (N-1)] = - dt * (Dx_term + Vx_term)
            
            # Treat y-rotation
            A[idx, (idx + N)%(N*N)] = - dt * (Dy_term + Vy_term)    # j + 1
            A[idx, (idx + (N*N - N))%(N*N)] = - dt * (Dy_term - Vy_term) # j - 1
         
        
            # Set the right-hand side vector
            b[idx] = phi[i, j]  # phi is the solution at the previous time step
    
    return A, b

def simulation(N, dt, phi):
    
    metric = get_metric(phi)
    
    # Set time to zero
    time = 0
    
    plot_potential_field(phi, time, metric)
    
    # Set frame counting to zero
    frame = 0
    
    
    while metric >= 0.05 and frame < 100000:
        
        # Assemble the linear system (A) and right-hand side (rhs)
        A, rhs = assemble_linear_system(phi)

        # Solve the linear system using an efficient solver
        phi_np1 = spsolve(A, rhs)

        # Update phi for the next time step
        phin = phi_np1
        
        phi = phin.reshape((Nx, Ny))
        
        frame += 1
        time = frame * dt
        
        metric = get_metric(phi)
        
        if frame%1 == 0:
            print(frame)
            plot_potential_field(phi, time, metric)
            plt.pause(0.001)
            
        if metric > 3 or np.mean(phi) > 1 :
            
            print("divergence \n")
            time = 0
            break
        
    plot_potential_field(phi, time, metric)
    
    return time

result = simulation(N, dt, phi)
print(f"Total time passed in the simulation: {result:.3f} seconds ")

