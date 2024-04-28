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
import os
import glob
import atexit

from modules.counter_flow_combustion import CounterFlowCombustion
from modules.misc import print_write_end_message
from modules.simu_parameters import SimuParameters



#Chemin absolu du fichier .py qu'on execute
fullpath = os.path.abspath(__file__)
#Chemin absolu du dossier contenant le .py qu'on execute
projectpath = os.path.dirname(fullpath)
# Chemin des data
datapath = projectpath + "\\outputs\\Data"
# Chemin des figures
figpath = projectpath + "\\outputs\\Figures"



### PHYSICAL PARAMETERS ###
L = 2e-3    # Length in the square shape domain.
nu = 15e-6  # Velocity Diffusion coefficient.
D = 15e-6   # Species Diffusion coefficient.
a = 15e-6   # Temperature Diffusion coefficient.
L_slot = 5e-4 # length of the inlet slot.
L_coflow = 5e-4 # length of the inlet coflow.
rho = 1.1614 # Fluid density.
Temp_a = 10000 # Temperature in Arhenus' law.
time_before_ignit = 0.02 # Time before the ignition is triggered in seconds.
c_p = 1200.


### NUMERICAL PARAMETERS ###
physN = 51 # Number of points for each space axis. "Physical N" in opposition with the size accounting for ghost cells which is N = physN + ghost_thickness.
ell_crit = 1e-4 # Stop threshold of elliptic solver
div_crit = 1e6 # Divergence stop cirterion
uv_conv_crit = 0.05 # Velocity field convergence threshold
temp_conv_crit = 117 # Temperature convergence threshold
max_time_computation = datetime.timedelta(hours=1, minutes=0) # Put here the maximum time you want to spend on the computation.


### OTHER PARAMETERS ###
show_and_save = True # Show and register plots ?
register_period = 5 # Period in frames of registeration of plots.
# Coordinates of the pixel to observe to check chemistry well functioning. Enter the coordinate, not accounting for ghost cells. 
i_reactor = physN//2 # Here its in the middle of the left wall.
j_reactor = 0


### FUNCTIONS ###

def delete_existing_datapath():

    data_files = glob.glob(datapath+"\\*")
    suppressed_things = False
    for f in data_files:
        try:
            os.remove(f)
        except:
            pass
        else:
            suppressed_things = True
    
    if suppressed_things:
        print("Warning: any file located in "+datapath+" was removed.")


def end_message(asimu:CounterFlowCombustion):
    
    div_crit = asimu.prms.div_crit
    uv_conv_crit = asimu.prms.uv_conv_crit
    temp_conv_crit = asimu.prms.temp_conv_crit
    L = asimu.prms.L
    D = asimu.prms.D
    N = asimu.prms.N
    dt = asimu.prms.dt
    max_t_comput = asimu.prms.max_t_comput    

    stop_datetime = asimu.stop_datetime
    start_datetime = asimu.start_datetime
    time = asimu.time
    frame = asimu.frame
    uv_consecutive_diff = asimu.uv_consecutive_diff
    temp_consecutive_diff = asimu.temp_consecutive_diff
    uv_max_strain_rate = asimu.uv_max_strain_rate
    n2_diff_zone_thick = asimu.n2_diff_zone_thick
    Temp_maxT_flame = asimu.Temp_maxT_flame
    endcode = asimu.endcode


    print_write_end_message(endcode, div_crit, max_t_comput, uv_conv_crit, temp_conv_crit, L, D, N, dt, stop_datetime - start_datetime, time, frame, uv_consecutive_diff, temp_consecutive_diff, uv_max_strain_rate, n2_diff_zone_thick, Temp_maxT_flame)



### MAIN ###
    
if __name__ == "__main__":

    if show_and_save:
        delete_existing_datapath()

    myparams = SimuParameters(L, physN, L_slot, L_coflow, nu, D, a, rho, c_p, Temp_a, time_before_ignit, register_period, i_reactor, j_reactor, ell_crit, div_crit, uv_conv_crit, temp_conv_crit, max_time_computation, show_and_save, datapath)
    myparams.save_as_pickle()

    mysimu = CounterFlowCombustion(myparams)

    atexit.register(end_message, mysimu)

    mysimu.compute()

