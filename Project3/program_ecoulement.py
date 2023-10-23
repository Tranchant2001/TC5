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
from numba import njit


### FUNCTIONS   ###

def set_velocity_field():

    u = np.zeros((N,N), dtype=float)
    v = np.zeros((N,N), dtype=float)

    ii = 0
    while ii*dx >= 0. and ii*dx < L_slot:
        v[0][ii] = 1.
        v[N-1][ii] = -1.
        ii += 1

    while ii*dx >= L_slot and ii*dx < L_slot + L_coflow:
        v[0][ii] = 0.2
        v[N-1][ii] = -0.2
        ii += 1        
    
    return u, v

@njit
def get_vector_metric(u, v):

    norm_array = np.sqrt(u*u + v*v)
    return norm_array, np.std(norm_array)/np.mean(norm_array)

@njit
def get_scalar_metric(phi):
    """
    Calculate standard deviation and average of phi
    """
    return np.std(phi)/np.mean(phi)

@njit
def field_x(field, u):

    field_x = np.zeros((N,N), dtype=float)
    for i in range(N):
        for j in range(1, N-1):
            if u[i][j] == 0.:
                field_x[i][j] = (field[i][j+1] - field[i][j-1])/(2*dx)
            elif u[i][j] > 0.:
                field_x[i][j] = (field[i][j] - field[i][j-1])/dx
            else:
                field_x[i][j] = (field[i][j+1] - field[i][j])/dx

    return field_x

@njit
def field_y(field, v):

    field_y = np.zeros((N,N), dtype=float)
    for i in range(1, N-1):
        for j in range(N):
            if v[i][j] == 0.:
                field_y[i][j] = (field[i+1][j] - field[i-1][j])/(2*dx)
            elif v[i][j] > 0.:
                field_y[i][j] = (field[i][j] - field[i-1][j])/dx
            else:
                field_y[i][j] = (field[i+1][j] - field[i][j])/dx

    return field_y


@njit
def boundary_conditions(v, u_new, v_new):
    # inlet and walls conditions for u
    u_new[:,0] = np.zeros(N, dtype=float)
    u_new[0,:] = np.zeros(N, dtype=float)
    u_new[N-1,:] = np.zeros(N, dtype=float)

    # outlet condition
    u_new[:,N-1] = u_new[:,N-2]
    v_new[:,N-1] = v_new[:,N-2]

    # inlet conditions for v
    for jj in range(N):
        if jj*dx >= 0. and jj*dx < L_slot:
            v_new[0][jj] = 1.
            v_new[N-1][jj] = -1.

        elif jj*dx >= L_slot and jj*dx < L_slot+L_coflow:
            v_new[0][jj] = 0.2
            v_new[N-1][jj] = -0.2

        else:
            v_new[0][jj] = 0.
            v_new[N-1][jj] = 0.
    
    # slipping wall condition on v
    for ii in range(1,N-1):
        if v[ii,0] > 0:
            v_new[ii][0] = -v[ii][0]*(v[ii][0] - v[ii-1][0])/dx + D*(v[ii+1][0] - v[ii][0] + v[ii-1][0] + v[ii][2] -2*v[ii][1])/(dx**2)
        if v[ii,0] < 0:
            v_new[ii][0] = -v[ii][0]*(v[ii+1][0] - v[ii][0])/dx + D*(v[ii+1][0] - v[ii][0] + v[ii-1][0] + v[ii][2] -2*v[ii][1])/(dx**2)
        else:
            v_new[ii][0] = D*(v[ii+1][0] - v[ii][0] + v[ii-1][0] + v[ii][2] -2*v[ii][1])/(dx**2)

    return u_new, v_new


def update(u, v):
    """
    Function to update the simulation at each step    
    """

    # Calculate first derivatives based on the signs of u and v
    u_x = field_x(u, u)
    u_y = field_y(u, v)
    v_x = field_x(v, u)
    v_y = field_y(v, v)    


    # Calculate second derivatives
    u_xx = (np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)) / dx**2
    u_yy = (np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)) / dx**2
    v_xx = (np.roll(v, -1, axis=1) - 2 * v + np.roll(v, 1, axis=1)) / dx**2
    v_yy = (np.roll(v, -1, axis=0) - 2 * v + np.roll(v, 1, axis=0)) / dx**2

    laplacian_u = u_xx + u_yy
    laplacian_v = v_xx + v_yy

    u_new = u + delta_t*(D*laplacian_u - u*u_x - v*u_y)
    v_new = v + delta_t*(D*laplacian_v - u*v_x - v*v_y)

    u_new, v_new = boundary_conditions(v, u_new, v_new)

    return u_new, v_new


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


def plot_vector_field(x_field, y_field, time, metric, **kwargs):

    # Create a figure and axis for the animation
    fig = plt.figure() 
    
    # Plot the scalar field
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Vector Field \n Time: {time:.2f} s \n Metric: {metric:.5f}')
    plt.quiver(X, Y, x_field, y_field)
    if 'saveaspng' in kwargs.keys():
        plt.savefig(dirpath+"/outputs_program_ecoulement/"+kwargs.get('saveaspng'), dpi=108, bbox_inches="tight")
    plt.show()
    plt.pause(1)
    plt.close(fig)


def plot_scalar_field(s_field, time, metric, **kwargs):
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
    ax.set_title(f'Scalar Field ($\Delta t=${delta_t}, N={N}) \n Time: {time:.2f} s \n Metric: {metric:.5f}')
    image = ax.imshow(s_field, extent=(0, L, 0, L), origin='lower', cmap='viridis')
    fig.colorbar(image, ax=ax)
    if 'saveaspng' in kwargs.keys():
        plt.savefig(dirpath+"/outputs_program_ecoulement/"+kwargs.get('saveaspng'), dpi=108, bbox_inches="tight")
    plt.pause(1)
    plt.close(fig)


def simulation(u, v, T_comput:dt.timedelta=dt.timedelta(days=7), showAndSave=True):
    
    # Get the norm field of velocities
    norm_arr, vel_metric = get_vector_metric(u, v)

    # Set time to zero
    time = 0
    
    # Set frame counting to zero
    frame = 0
    
    start_datetime = dt.datetime.now()
    step_datetime = dt.datetime.now()

    while vel_metric >= 0.05 and vel_metric < divergence_threshold and step_datetime - start_datetime < T_comput:
        
        time = frame * delta_t
        
        norm_arr, vel_metric = get_vector_metric(u, v)
        

        if frame%1 == 0:
            print(frame)
            if showAndSave:
                plot_scalar_field(norm_arr, time, vel_metric, saveaspng=str(frame)+"_velocity_field.png")

        u, v = update(u, v)
        step_datetime = dt.datetime.now()
        frame += 1

    if vel_metric >= divergence_threshold :        
        print(f"Warning: The simulation stopped running because a divergence was detected (vel_metric >= {divergence_threshold}).")
        print(f"\tParameters: (L, D, N, $\Delta t$)=({L}, {D}, {N}, {delta_t})")
        print("\tSimulation duration: "+str(step_datetime - start_datetime))
        print(f"\tVirtual stop time: {time:.2f} s")        
        print(f"\tVirtual stop frame: {frame}")
        print(f"\tVelocity norm: {vel_metric:5f}")

    elif step_datetime - start_datetime >= T_comput:
        print("Warning: The simulation stopped running because the max duration of simulation ("+str(T_comput)+") was reached.")
        print(f"\tParameters: (L, D, N, $\Delta t$)=({L}, {D}, {N}, {delta_t})")
        print("\tSimulation duration: "+str(step_datetime - start_datetime))
        print(f"\tVirtual stop time: {time:.2f} s")
        print(f"\tVirtual stop frame: {frame}")
        print(f"\tVelocity norm: {vel_metric:5f}")

    else:
        print("Success: The simulation stopped running because the field was homogeneous enough (vel_metric < 0.05).")
        print(f"\tParameters: (L, D, N, $\Delta t$)=({L}, {D}, {N}, {delta_t})")
        print("\tSimulation duration: "+str(step_datetime - start_datetime))
        print(f"\tVirtual stop time: {time:.2f} s")        
        print(f"\tVirtual stop frame: {frame}")
        print(f"\tVelocity norm: {vel_metric:5f}")


### MAIN    ###

#Chemin absolu du fichier .py qu'on execute
fullpath = os.path.abspath(__file__)
#Chemin absolu du dossier contenant le .py qu'on execute
dirpath = os.path.dirname(fullpath)

# Parameters of the problem
L = 2e-3     # Length in the square shape domain.
D = 15e-6   # Diffusion coeffiscient
L_slot = 5e-4 # length of the inlet slot.
L_coflow = 5e-4 # length of the inlet coflow.

# Initial Parameters of the simulation
N = 128    # Number of steps for each space axis
dx = L/N
max_u = 1. # because the max inlet velocity is 1 m/s.
delta_t = 0.90*min((dx)**2/D, dx/max_u)   # Time step
divergence_threshold = 1000

# Create mesh grid
x = np.linspace(0, L, N, endpoint=True)
y = np.linspace(0, L, N, endpoint=True)
i = np.linspace(0, N, N, endpoint=True)
j = np.linspace(0, N, N, endpoint=True)
X, Y = np.meshgrid(x, y)
I, J = np.meshgrid(i, j)

U, V = set_velocity_field()

# Put here the maximum time you want to spend on the computation.
max_time_computation = dt.timedelta(hours=0, minutes=2)

simulation(U, V, max_time_computation, True)
 
