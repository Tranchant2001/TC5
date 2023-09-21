# -*- coding: utf-8 -*-
#%%

"""
Projet n°1 du TC5: Numerical Methods du Master PPF.
Travail en binôme:
    Jules Chaveyriat
    Martin Guillon

Créé le 06/09/2023.
Mis à jour le 17/09/2023.
v1.1

DESCRIPTION:
Version with the explicit method of RK3-Heun.

"""
### PACKAGES    ###

import os
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from numba import jit


### FONCTIONS   ###

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
    Function to update the simulation at each step.
    Here follows the RK3-heun method.  
    """
    # Precising the parameters for the program
    dx = L/N
    
    def f(phi):
        """
        Function f as defined at the beginning of the Section 3 of the handout.
        Here f = [D*Laplacian -V.gradient]
        """
        phi_x = np.zeros_like(phi)
        phi_y = np.zeros_like(phi)
        phi_xx = np.zeros_like(phi)
        phi_yy = np.zeros_like(phi)

        # Calculate first derivatives based on the signs of u and v
        phi_x = np.where(u <= 0,
                        (np.roll(phi, -1, axis=0) - phi) / dx,
                        (phi - np.roll(phi, 1, axis=0)) / dx)

        phi_y = np.where(v <= 0,
                        (np.roll(phi, -1, axis=1) - phi) / dx,
                        (phi - np.roll(phi, 1, axis=1)) / dx)

        # Calculate second derivatives
        phi_xx = (np.roll(phi, -1, axis=0) - 2 * phi + np.roll(phi, 1, axis=0)) / dx**2
        phi_yy = (np.roll(phi, -1, axis=1) - 2 * phi + np.roll(phi, 1, axis=1)) / dx**2

        laplacian_phi = phi_xx + phi_yy
        
        return D*laplacian_phi - u*phi_x - v*phi_y        

    # Processes the intermediate coefficients of the RK method, as specified at the page 42 of the handout.
    k1 = f(phi)
    k2 = f(phi + delta_t*k1/3)
    k3 = f(phi + delta_t*2*k2/3)
    
    # Finally processes the phi at next step with the weight sum of defined k1 and k3 defined above.
    phi_plus_1 = phi + delta_t*k1/4 + 3*delta_t*k3/4 
    
    return phi_plus_1


def get_metric(phi):
    """
    Calculate standard deviation and average of phi
    """
    std_phi = np.std(phi)
    avg_phi = np.mean(phi)
    
    # Calculate the metric
    metric = std_phi / avg_phi
    
    return metric


def plot_potential_field(phi, time, metric, **kwargs):
    """
    Plot the state of the scalar field.
    Specifies in the title the time and the metric
    """    
    # Create a figure and axis for the animation
    fig, ax = plt.subplots()
    
    # Plot the scalar field
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Diffusion of Scalar Field ($\Delta t=${delta_t}, N={N}) \n Time: {time:.2f} s \n Metric: {metric:.5f}')
    ax.imshow(phi, extent=(0, L, 0, L), origin='lower', cmap='viridis')

    if 'saveaspng' in kwargs.keys():
        plt.savefig(dirpath+"/outputs_program_RK/"+kwargs.get('saveaspng'), dpi=108, bbox_inches="tight")
    plt.pause(1)
    plt.close(fig)


def simulation(N, delta_t, phi, T_comput:dt.timedelta=dt.timedelta(days=7), showAndSave=True):
    """
    Function taking the parameter of the problem as time and space smpling csts N and dt.
    Takes the initial scalar field phi.
    Takes T_comput, the maximum reel time you want this simulation to run. Default is set to 7 days.
    Processes frame by frame the deduced field with the scheme used in the above function update(phi, N, dt)
    Plots one frame per 100.
    Returns the time it took to reach metric < 0.05
    """
    
    metric = get_metric(phi)
    
    # Set time to zero
    time = 0
    
    # Set frame counting to zero
    frame = 0
    
    start_datetime = dt.datetime.now()
    step_datetime = dt.datetime.now()
    
    while metric >= 0.05 and metric < 3 and step_datetime - start_datetime < T_comput:
        
        # Update the simulation

        
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


### MAIN   ###


#Chemin absolu du fichier .py qu'on execute
fullpath = os.path.abspath(__file__)
#Chemin absolu du dossier contenant le .py qu'on execute
dirpath = os.path.dirname(fullpath)

# Parameters of the problem
L = 1.0     # Length of the (square shaped) domain (m)
D = 0.001   # Diffusion coefficient

# Initial Parameters of the simulation
N = 400    # Number of steps for each space axis
delta_t = 0.00005   # Time step

# Put here the maximum time you want to spend on the computation.
max_time_computation = dt.timedelta(hours=6)

# Create mesh grid
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y)

# Initial velocity field
u, v = set_velocity_field(L, N, X, Y)

# Initial scalar field
phi = set_initial_potential(X, Y)

# Display and save parameters
show_and_save = True
pictures_save_path = dirpath+'/outputs_program_backwardEuler'

total_time_passed = simulation(N, delta_t, phi, max_time_computation, show_and_save)