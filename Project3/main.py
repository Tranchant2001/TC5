# -*- coding: utf-8 -*-

"""
Projet n°3 du TC5: Numerical Methods du Master PPF.
Travail en binôme:
    Jules Chaveyriat
    Martin Guillon

Créé le 24/10/2023.
Mis à jour le 14/11/2023.
v1.1

DESCRIPTION:
Propagating the Fluid flow in the chamber.

"""


import datetime
from counter_flow_combustion import CounterFlowCombustion
from fluid_flow import FluidFlow



# Parameters of the problem
L = 2e-3    # Length in the square shape domain.
nu = 15e-6  # Velocity Diffusion coefficient.
D = 15e-6   # Species Diffusion coefficient.
a = 15e-6   # Temperature Diffusion coefficient.
L_slot = 5e-4 # length of the inlet slot.
L_coflow = 5e-4 # length of the inlet coflow.
rho = 1.1614 # Fluid density.
Temp_a = 10000 # Temperature in Arhenus' law.
time_before_ignit = 0.01 # Time before the ignition is triggered in seconds.
c_p = 1200.


# Initial Parameters of the simulation
physN = 64 # Number of steps for each space axis. "Physical N" in opposition with the size accounting for ghost cells which is N.


# Put here the maximum time you want to spend on the computation.
max_time_computation = datetime.timedelta(hours=1, minutes=0)
# Show and register plots ?
show_and_save = True
register_period = 561
# Coordinates of the pixel to observe to check chemistry well functioning.
i_reactor = 33 # Enter the coordinate, not accounting for ghost cells.
j_reactor = 0

# Stop threshold of elliptic solver
ell_crit = 1e-4
# Divergence stop cirterion
div_crit = 1e6
conv_crit = 0.01

mysimu = CounterFlowCombustion(L, physN, L_slot, L_coflow, nu, D, a, rho, c_p, Temp_a, time_before_ignit, max_time_computation, show_and_save, register_period, i_reactor, j_reactor, ell_crit, div_crit, conv_crit)
mysimu.compute()