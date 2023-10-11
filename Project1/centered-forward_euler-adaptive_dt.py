# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 22:17:44 2023

@author: PC
"""

"""
HI, this is the first version of the project : resolution with explicit Euler method
The second version of the project will involve implicit Euler method
"""

import numpy as np
import matplotlib.pyplot as plt


# Parameters of the problem
L = 1.0     # Length of the (square shaped) domain (m)
D = 0.001   # Diffusion coeffiscient

def create_mesh_grid(N):
    
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y)
    
    return X, Y
def set_velocity_field(X, Y):
    
    u = np.cos(4 * np.pi * X) * np.sin(4 * np.pi * Y)
    v = -np.sin(4 * np.pi * X) * np.cos(4 * np.pi * Y)
    
    return u, v

def set_initial_potential(X, Y):
    
    phi = np.where(np.sqrt((X - 0.5)**2 + (Y - 0.5)**2) < 0.3, 1.0, 0.0)
    
    return phi


# Function to update the simulation at each step
def update(phi, N, dt):
    
    h = L/N
    
    phi_x = np.zeros_like(phi)
    phi_y = np.zeros_like(phi)
    phi_xx = np.zeros_like(phi)
    phi_yy = np.zeros_like(phi)

    # Calculate first derivatives based on the signs of u and v
    phi_x = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2*h)
                     

    phi_y = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2*h)

    # Calculate second derivatives
    phi_xx = (np.roll(phi, -1, axis=0) - 2 * phi + np.roll(phi, 1, axis=0)) / h**2
    phi_yy = (np.roll(phi, -1, axis=1) - 2 * phi + np.roll(phi, 1, axis=1)) / h**2

    laplacian_phi = phi_xx + phi_yy
    phi += dt * (D * laplacian_phi - u * phi_x - v * phi_y)
    
    return phi

def get_metric(phi):
    
    # Calculate standard deviation and average of phi
    std_phi = np.std(phi)
    avg_phi = np.mean(phi)
    
    # Calculate the metric
    metric = std_phi / avg_phi
    
    return metric

def get_mass(phi, N):
    
    h = L/N
    
    return np.sum(phi) * h**2

def get_new_dt(phi, N, dt):
    
    h = L/N
    
    # Calculate the maximum gradient of the potential field
    max_gradient = np.max(np.abs(np.gradient(phi, h, axis=(0, 1))))
    
    # Calculate the Fourier and CFL stability criteria
    fourier_criterion = D * dt / (h**2) <= 0.25
    cfl_criterion = (dt/h)**2 + (dt/h)**2 <= 1.0
    
    # Calculate a safety factor (you can adjust this as needed)
    safety_factor = 0.9
    
    # Calculate a new time step based on the maximum gradient
    new_dt = safety_factor * h**2 / (4 * D * max_gradient)
    
    # Ensure that the new time step is not too large
    if new_dt > 2 * dt:
        new_dt = 2 * dt

    
    # Ensure that the new time step satisfies the stability criteria
    if not (fourier_criterion and cfl_criterion):
        new_dt = min(new_dt, (0.25 * h**2) / D, h / np.sqrt(2))
    
        
    return new_dt

def plot_potential_field(phi, time, metric):
    
    # Create a figure and axis for the plot
    fig, ax = plt.subplots()
    
    # Plot the scalar field
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Diffusion of a scalar field \n Time: {time:.2f} s \n Metric: {metric:.5f}')
    
    # Plot the scalar field with a color scale (colorbar)
    im = ax.imshow(phi, extent=(0, L, 0, L), origin='lower', cmap='viridis')

    # Add a colorbar
    cbar = plt.colorbar(im, ax=ax)

def simulation(N, dt, phi):
    
    metric = get_metric(phi)
    
    # Set time to zero
    time = 0
    
    # Set frame counting to zero
    frame = 0
    
    
    while metric >= 0.05 and frame < 1000000:
        
        if frame%100 == 0:
            
            # Update the time step
            dt = get_new_dt(phi, N, dt)
            
            plot_potential_field(phi, time, metric)
            plt.pause(0.001)
        
        # Update the simulation
        phi = update(phi, N, dt)
        
        frame += 1
        time += dt
        
        metric = get_metric(phi)
         
        if metric > 3  :
            
            print("divergence \n")
            time = 0
            break
        
    plot_potential_field(phi, time, metric)
    
    return time

    





###MAIN###

dt_values = [0.0001]
N_values = [400]

result_hist = []

for value in dt_values :
    
    dt = value
    
    result_t = []

    for N in N_values:
        
        print("N = ", N)
        
        # Create mesh grid, velocity field and initial potential
        X, Y = create_mesh_grid(N)
    
        u, v = set_velocity_field(X, Y)
    
        phi = set_initial_potential(X, Y)
        
        # Start simulation
        total_time_passed = simulation(N, dt, phi)
    
        result_t.append(total_time_passed)
        
        print(f"Total time to reach homogeneity: {total_time_passed:.3f} seconds ")
        

    result_hist.append(result_t)

for tests in result_hist:
    plt.close()
    plt.plot(N_values, tests)
    
    # Add a title
    plt.title("time to reach homogeneity with varying N")
    
    # Add axis labels
    plt.xlabel("N")
    plt.ylabel("time (s)")
    
    # Add a grid
    plt.grid(True)

    plt.show()
