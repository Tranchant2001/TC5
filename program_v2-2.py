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

### PACKAGES    ###

import os
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from numba import jit


### FUNCTIONS   ###

@jit(nopython=True)
def set_velocity_field(L, N, X, Y):
    
    u = np.cos(4 * np.pi * X) * np.sin(4 * np.pi * Y)
    v = -np.sin(4 * np.pi * X) * np.cos(4 * np.pi * Y)
    
    return u, v


@jit(nopython=True)
def set_initial_potential(X, Y):
    
    phi = np.where(np.sqrt((X - 0.5)**2 + (Y - 0.5)**2) < 0.3, 1.0, 0.0)
    
    return phi


def update(phi, N, delta_t):
    """
    Function to update the simulation at each step    
    """
    # Precising the parameters for the program
    Nx = N
    Ny = N
    h = L/N
    
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
    phi += delta_t * (D * laplacian_phi - u * phi_x - v * phi_y)
    
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
        plt.savefig(dirpath+"/outputs_program_RK/"+kwargs.get('saveaspng'), dpi=108, bbox_inches="tight")
    plt.pause(1)
    plt.close(fig)


def simulation(N, delta_t, T, phi, T_comput:dt.timedelta=dt.timedelta(days=7), showAndSave=True):
    
    metric = get_metric(phi)
    
    # Set time to zero
    time = 0
    
    # Set frame counting to zero
    frame = 0
    
    start_datetime = dt.datetime.now()
    step_datetime = dt.datetime.now()

    while metric >= 0.05 and metric < 3 and step_datetime - start_datetime < T_comput:
        
        time = frame * delta_t
        
        metric = get_metric(phi)
        

        if frame%100 == 0:
            print(frame)
            if showAndSave:
                plot_potential_field(phi, time, metric, saveaspng=str(frame)+"_phi_field.png")

        phi = update(phi, N, delta_t)
        step_datetime = dt.datetime.now()
        frame += 1

    if metric >= 3 :        
        print("Warning: The simulation stopped running because a divergence was detected (metric >= 3).")
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


# Parameters of the problem
L = 1.0     # Length of the (square shaped) domain (m)
D = 0.001   # Diffusion coeffiscient

# Initial Parameters of the simulation
N = 64    # Number of steps for each space axis
delta_t = 0.001   # Time step
T = 10      # Total time

# Create mesh grid
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y)
u, v = set_velocity_field(L, N, X, Y)

phi = set_initial_potential(X, Y)




# J'ai mis l'iteration en commentaire car je ne voulais pas la lancer.
"""
result_hist = []
dt_values = [0.001]
N_values = [300]

for value in dt_values :
    
    delta_t = value
    
    result_t = []

    for N in N_values:
        
        
        print(f"N = {N}, delta_t = {delta_t}")
        
        # Create mesh grid
        x = np.linspace(0, L, N, endpoint=False)
        y = np.linspace(0, L, N, endpoint=False)
        X, Y = np.meshgrid(x, y)
    
        u, v = set_velocity_field(L, N, X, Y)
    
        phi = set_initial_potential(X, Y)
        
        total_time_passed = simulation(N, delta_t, T, phi)
    
        result_t.append(total_time_passed)
        
        
    
        print(f"Total time passed in the simulation: {total_time_passed:.3f} seconds ")
        
        phi = set_initial_potential(X, Y)

    result_hist.append(result_t)

for tests in result_hist:
    plt.close()
    plt.plot(N_values, tests)
    plt.show()
    print("N = ", N, " t = ", tests)

"""    
