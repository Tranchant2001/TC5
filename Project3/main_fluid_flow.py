# -*- coding: utf-8 -*-

"""
Projet n°3 du TC5: Numerical Methods du Master PPF.
Travail en binôme:
    Martin Guillon

Créé le 16/01/2024.
Mis à jour le 16/01/2024.
v1.1

DESCRIPTION:
Propagating the Fluid flow in the chamber.

"""


import datetime
from fluid_flow import FluidFlow



# Parameters of the problem
L = 2e-3    # Length in the square shape domain.
nu = 15e-6  # Velocity Diffusion coefficient.
L_slot = 5e-4 # length of the inlet slot.
L_coflow = 5e-4 # length of the inlet coflow.
rho = 1.1614 # Fluid density.

# Initial Parameters of the simulation
physN = 132 # Number of steps for each space axis. "Physical N" in opposition with the size accounting for ghost cells which is N.

# Put here the maximum time you want to spend on the computation.
max_time_computation = datetime.timedelta(hours=1, minutes=0)
# Show and register plots ?
show_and_save = True
register_period = 3000
# Maximum size of diagnostics data lists whose new elements are appended every frame:
lists_max_size = 1024 # = 2**10. 

# Stop threshold of elliptic solver
ell_crit = 2e-4
# Divergence stop cirterion
div_crit = 1e6
conv_crit = 2.5

mysimu = FluidFlow(L, physN, L_slot, L_coflow, nu, rho, max_time_computation, show_and_save, register_period ,lists_max_size, ell_crit, div_crit, conv_crit)
mysimu.compute()