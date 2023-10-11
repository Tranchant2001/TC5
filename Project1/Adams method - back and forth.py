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
N = 128  # Number of steps for each space axis
dt = 0.001   # Time step
h = L/N  # Spatial step size

# Create mesh grid
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y)

def set_velocity_field(X, Y):
    
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

def plot_potential_field(phi, time, metric):
    
    # Create a figure and axis for the animation
    fig, ax = plt.subplots()
    
    # Plot the scalar field
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Diffusion of Scalar Field \n Time: {time:.2f} s \n Metric: {metric:.5f}')
    ax.imshow(phi, extent=(0, L, 0, L), origin='lower', cmap='viridis')
    
def first_step(phi) :
    
    
    phi_x = np.zeros_like(phi)
    phi_y = np.zeros_like(phi)
    phi_xx = np.zeros_like(phi)
    phi_yy = np.zeros_like(phi)

    # Calculate first derivatives based on the signs of u and v
    phi_x = np.where(u <= 0,
                     (np.roll(phi, -1, axis=0) - phi) / h,
                     (phi - np.roll(phi, 1, axis=0)) / h)

    phi_y = np.where(v <= 0,
                     (np.roll(phi, -1, axis=1) - phi) / h,
                     (phi - np.roll(phi, 1, axis=1)) / h)

    # Calculate second derivatives
    phi_xx = (np.roll(phi, -1, axis=0) - 2 * phi + np.roll(phi, 1, axis=0)) / h**2
    phi_yy = (np.roll(phi, -1, axis=1) - 2 * phi + np.roll(phi, 1, axis=1)) / h**2

    laplacian_phi = phi_xx + phi_yy
    phi += dt * (D * laplacian_phi - u * phi_x - v * phi_y)
    
    return phi

def get_adv_term(phi):
    

    adv_x = u * np.where(u <= 0,
                     (np.roll(phi, -1, axis=0) - phi) / h,
                     (phi - np.roll(phi, 1, axis=0)) / h)
    adv_y = v * np.where(v <= 0,
                     (np.roll(phi, -1, axis=1) - phi) / h,
                     (phi - np.roll(phi, 1, axis=1)) / h)
    
    return adv_x, adv_y
    
def get_diff_term(phi):
      
    phi_x_plus_1 = np.roll(phi, - 1, axis=0)
    phi_x_minus_1 = np.roll(phi, 1, axis=0)
    phi_y_plus_1 = np.roll(phi, - 1, axis=1)
    phi_y_minus_1 = np.roll(phi, 1, axis=1)
    
    diff = D * ((phi_x_plus_1 - 2 * phi + phi_x_minus_1) / h**2 +
                  (phi_y_plus_1 - 2 * phi + phi_y_minus_1) / h**2)
    
    return diff
    

def update(phi, phi_prev):
    
    # Compute the advection term using the velocity field at time step n
    adv_x_n, adv_y_n = get_adv_term(phi)
    
    # Compute the diffusion term at time step n
    diff_n = get_diff_term(phi)
    
    # Compute the advection term using the velocity field at time step n-1
    adv_x_n_minus_1, adv_y_n_minus_1 = get_adv_term(phi_prev)
    
    # Compute the diffusion term at time step n-1
    diff_n_minus_1 = get_diff_term(phi_prev)
    
    # Update phi using the second-order Adams-Bashforth method
    phi_new = phi - dt * (1.5 * (adv_x_n + adv_y_n - diff_n) - 0.5 * (adv_x_n_minus_1 + adv_y_n_minus_1 - diff_n_minus_1))
        
    
    return phi_new
    
    

def simulation(N, dt, phi):
    
    metric = get_metric(phi)
    
    # Set time to zero
    time = 0
    
    plot_potential_field(phi, time, metric)
    
    # Set frame counting to zero
    frame = 0
    
    # Get phi(n= 0) and phi(n = 1)
    phi_0 = phi
    phi_1 = first_step(phi)
       
    phi_new = update(phi_1, phi_0)
    phi_prev = phi_1
    phi = phi_new
    
    while metric >= 0.05 and frame < 1000000:
            
        phi_new = update(phi, phi_prev)
        phi_prev = phi
        phi = phi_new
        
        frame += 1
        time = frame * dt
        
        metric = get_metric(phi)
        
        if frame%100 == 0:
            print(frame)
            plot_potential_field(phi, time, metric)
            plt.pause(0.001)
            
        if metric > 3 or np.mean(phi) > 1 :
            
            print("divergence \n")
            time = 0
            break
        
    plot_potential_field(phi, time, metric)
    
    return time

u, v = set_velocity_field(X, Y)

phi = set_initial_potential(X, Y)

result = simulation(N, dt, phi)
print(f"Total time passed in the simulation: {result:.3f} seconds ")

