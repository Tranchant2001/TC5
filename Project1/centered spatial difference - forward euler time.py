# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 11:45:48 2023

@author: jules
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 22:11:52 2023

@author: jules
"""

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

# Initial Parameters of the simulation
N = 64    # Number of steps for each space axis
dt = 0.001   # Time step
T = 10      # Total time

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

# Function to update the simulation at each step
def update(phi, N, dt):
    
    # Precising the parameters for the program
    Nx = N
    Ny = N
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

def plot_potential_field(phi, time, metric):
    
    # Create a figure and axis for the animation
    fig, ax = plt.subplots()
    
    # Plot the scalar field
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Diffusion of Scalar Field \n Time: {time:.2f} s \n Metric: {metric:.5f}')
    ax.imshow(phi, extent=(0, L, 0, L), origin='lower', cmap='viridis')

def simulation(N, dt, T, phi):
    
    metric = get_metric(phi)
    
    # Set time to zero
    time = 0
    
    plot_potential_field(phi, time, metric)
    
    # Set frame counting to zero
    frame = 0
    
    
    while metric >= 0.05 and frame < 100000:
        
        # Update the simulation
        phi = update(phi, N, dt)
        
        frame += 1
        time = frame * dt
        
        metric = get_metric(phi)
        
        if frame%100 == 0:
            print(frame)
            plot_potential_field(phi, time, metric)
            plt.pause(0.001)
            
        if metric > 3 :
            
            print("divergence \n")
            time = 0
            break
        
    plot_potential_field(phi, time, metric)
    
    return time


result_hist = []

    
dt_values = [0.0001]
N_values = [300]

for value in dt_values :
    
    dt = value
    
    result_t = []

    for N in N_values:
        
        
        print(f"N = {N}, dt = {dt}")
        
        # Create mesh grid
        x = np.linspace(0, L, N, endpoint=False)
        y = np.linspace(0, L, N, endpoint=False)
        X, Y = np.meshgrid(x, y)
    
        u, v = set_velocity_field(L, N, X, Y)
    
        phi = set_initial_potential(X, Y)
        
        total_time_passed = simulation(N, dt, T, phi)
    
        result_t.append(total_time_passed)
        
        
    
        print(f"Total time passed in the simulation: {total_time_passed:.3f} seconds ")
        
        phi = set_initial_potential(X, Y)

    result_hist.append(result_t)

for tests in result_hist:
    plt.close()
    plt.plot(N_values, tests)
    plt.show()
    print("N = ", N, " t = ", tests)

