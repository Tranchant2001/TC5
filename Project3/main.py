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



# Parameters of the problem
L = 2e-3     # Length in the square shape domain.
D = 15e-6   # Diffusion coefficient
L_slot = 5e-4 # length of the inlet slot.
L_coflow = 5e-4 # length of the inlet coflow.
pho = 1.1614 # Fluid density.

# Initial Parameters of the simulation
N = 1024    # Number of steps for each space axis


# Put here the maximum time you want to spend on the computation.
max_time_computation = datetime.timedelta(hours=1, minutes=0)
# Show and register plots ?
show_and_save = True
register_period = 10000

# Stop threshold of elliptic solver
ell_crit = 2e-4
# Divergence stop cirterion
div_crit = 100.
conv_crit = 5e-6

mysimu = CounterFlowCombustion(L, N, L_slot, L_coflow, D, pho, max_time_computation, show_and_save, register_period, ell_crit, div_crit, conv_crit)
mysimu.compute()