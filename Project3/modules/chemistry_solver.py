# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
from numba import njit
import datetime


from modules.field import Field
from modules.velocity_field import VelocityField
from modules.pressure_field import PressureField
from modules.species import Species, Dioxygen, Dinitrogen, Methane, Water, CarbonDioxide
from modules.temperature_field import TemperatureField
import modules.misc as misc

#Chemin absolu du fichier .py qu'on execute
fullpath = os.path.abspath(__file__)
#Chemin absolu du dossier contenant le .py qu'on execute
modulepath = os.path.dirname(fullpath)
# Chemin absolu du dossier du Projet
projectpath = os.path.dirname(modulepath)
# Chemin absolu du dossier de figures
figpath = projectpath + "\\outputs\\Figures"

@njit
def numba_RK2(physN:int, dt_transport:float, o2_arr:np.ndarray, W_o2, st_o2, Dhf_o2, ch4_arr:np.ndarray, W_ch4, st_ch4, Dhf_ch4, h2o_arr:np.ndarray, W_h2o, st_h2o, Dhf_h2o, co2_arr:np.ndarray, W_co2, st_co2, Dhf_co2, Temp_arr:np.ndarray, Ta, rho, c_p, i_reactor, j_reactor, thick:int, zero_th:float, suivitime, suivio2, suivich4, suivih2o, suivico2, suiviT):
    
    nsuivi = suivio2.shape[0]
    
    for i in range(physN):
        for j in range(physN):
            # lines useful for debugging
            if i == i_reactor and j == j_reactor:
                nothing = True
            isworth = True
            frame_chem = 0
            time_chem = 0.
            o20 = o2_arr[i,j]
            ch40 = ch4_arr[i,j]
            h2o0 = h2o_arr[i,j]
            co20 = co2_arr[i,j]
            Temp0 = Temp_arr[i,j]
            while time_chem < dt_transport and isworth:

                # lines useful for debugging
                #if bool(np.isnan(o2_arr[i,j])) or bool(np.isnan(ch4_arr[i,j])):
                #    pass 

                Q = numba_progress_rate(o2_arr[i,j], W_o2, ch4_arr[i,j], W_ch4, Temp_arr[i,j], Ta, rho, zero_th)
                #print(o2_arr[i,j])

                timescale = get_timescale_problem(Q, rho, c_p, o2_arr[i,j], W_o2, st_o2, ch4_arr[i,j], W_ch4, st_ch4, Dhf_ch4, st_h2o, Dhf_h2o, st_co2, Dhf_co2, Temp_arr[i,j])
                if time_chem + timescale >= dt_transport:
                    dtchem = dt_transport - time_chem
                    time_chem = dt_transport
                else:
                    dtchem = timescale
                    time_chem += timescale

                o2_arr[i,j] = o20 + (0.5*dtchem*st_o2*W_o2/rho)*Q
                #print(o2_arr[i,j])
                ch4_arr[i,j] = ch40 + (0.5*dtchem*st_ch4*W_ch4/rho)*Q
                Temp_arr[i,j] = Temp0 - (0.5*dtchem/(rho*c_p))*(st_ch4*Dhf_ch4 + st_co2*Dhf_co2 + st_h2o*Dhf_h2o)*Q

                Q = numba_progress_rate(o2_arr[i,j], W_o2, ch4_arr[i,j], W_ch4, Temp_arr[i,j], Ta, rho, zero_th)
                o2_arr[i,j] = o20 + (dtchem*st_o2*W_o2/rho)*Q
                ch4_arr[i,j] = ch40 + (dtchem*st_ch4*W_ch4/rho)*Q
                h2o_arr[i,j] = h2o0 + (dtchem*st_h2o*W_h2o/rho)*Q
                co2_arr[i,j] = co20 + (dtchem*st_co2*W_co2/rho)*Q
                Temp_arr[i,j] = Temp0 - (dtchem/(rho*c_p))*(st_ch4*Dhf_ch4 + st_co2*Dhf_co2 + st_h2o*Dhf_h2o)*Q

                isworth = is_worth_continue(o20, o2_arr[i,j], Temp0, Temp_arr[i,j], dtchem)

                o20 = o2_arr[i,j]
                ch40 = ch4_arr[i,j]
                h2o0 = h2o_arr[i,j]
                co20 = co2_arr[i,j]
                Temp0 = Temp_arr[i,j]

                if i == i_reactor and j == j_reactor and frame_chem+1 < nsuivi:
                    suivitime[frame_chem+1] = time_chem
                    suivio2[frame_chem+1] = o20
                    suivich4[frame_chem+1] = ch40
                    suivih2o[frame_chem+1] = h2o0
                    suivico2[frame_chem+1] = co20
                    suiviT[frame_chem+1] = Temp0

                frame_chem += 1

"""
TO DO: implementer un methode implicite
@njit
def numba_Linearized_Trapezoidal(physN:int, dtchemlist:np.ndarray, o2_arr:np.ndarray, W_o2, st_o2, Dhf_o2, ch4_arr:np.ndarray, W_ch4, st_ch4, Dhf_ch4, h2o_arr:np.ndarray, W_h2o, st_h2o, Dhf_h2o, co2_arr:np.ndarray, W_co2, st_co2, Dhf_co2, Temp_arr:np.ndarray, Ta, rho, c_p, i_reactor, j_reactor, suivitime, suivio2, suivich4, suivih2o, suivico2, suiviT):

    nchem = dtchemlist.shape[0]

    for i in range(physN):
        for j in range(physN):
            isworth = True
            frame_chem = 0
            o20 = o2_arr[i,j]
            ch40 = ch4_arr[i,j]
            h2o0 = h2o_arr[i,j]
            co20 = co2_arr[i,j]
            Temp0 = Temp_arr[i,j]
            while frame_chem < nchem and isworth:
                
                dtchem = dtchemlist[frame_chem]

                Q = numba_progress_rate(o2_arr[i,j], W_o2, ch4_arr[i,j], W_ch4, Temp_arr[i,j], Ta, rho)
                #print(o2_arr[i,j])
                o2_arr[i,j] = o20 + (0.5*dtchem*st_o2*W_o2/rho)*Q
                #print(o2_arr[i,j])
                ch4_arr[i,j] = ch40 + (0.5*dtchem*st_ch4*W_ch4/rho)*Q
                Temp_arr[i,j] = Temp0 - (0.5*dtchem/(rho*c_p))*(st_ch4*Dhf_ch4 + st_co2*Dhf_co2 + st_h2o*Dhf_h2o)*Q

                Q = numba_progress_rate(o2_arr[i,j], W_o2, ch4_arr[i,j], W_ch4, Temp_arr[i,j], Ta, rho)
                o2_arr[i,j] = o20 + (dtchem*st_o2*W_o2/rho)*Q
                ch4_arr[i,j] = ch40 + (dtchem*st_ch4*W_ch4/rho)*Q
                h2o_arr[i,j] = h2o0 + (dtchem*st_h2o*W_h2o/rho)*Q
                co2_arr[i,j] = co20 + (dtchem*st_co2*W_co2/rho)*Q
                Temp_arr[i,j] = Temp0 - (dtchem/(rho*c_p))*(st_ch4*Dhf_ch4 + st_co2*Dhf_co2 + st_h2o*Dhf_h2o)*Q

                isworth = is_worth_continue(o20, o2_arr[i,j], Temp0, Temp_arr[i,j], dtchem)

                o20 = o2_arr[i,j]
                ch40 = ch4_arr[i,j]
                h2o0 = h2o_arr[i,j]
                co20 = co2_arr[i,j]
                Temp0 = Temp_arr[i,j]

                if i == i_reactor and j == j_reactor:
                    suivitime[frame_chem+1] = suivitime[frame_chem]+dtchem
                    suivio2[frame_chem+1] = o20
                    suivich4[frame_chem+1] = ch40
                    suivih2o[frame_chem+1] = h2o0
                    suivico2[frame_chem+1] = co20
                    suiviT[frame_chem+1] = Temp0

                frame_chem += 1"""

@njit
def numba_progress_rate(o2_val, W_o2, ch4_val, W_ch4, T_val, Ta, rho, zero_th:float):

        A = 1.1e8

        ch4_conc = (rho/W_ch4)*ch4_val
        o2_conc = (rho/W_o2)*o2_val

        Q = 0.

        if o2_conc > zero_th:

            Q = A*(o2_conc**2)*ch4_conc*np.exp(-Ta/T_val)

        return Q


def plot_chemistry(suivi_time, suivi_o2, suivi_ch4, suivi_h2o, suivi_co2, suivi_T, i, j, frame):

    fig1, ax1 = plt.subplots()
    ax1.clear()
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Massic fraction')
    ax1.plot(suivi_time, suivi_o2, label="O2", linestyle="-", marker=".")
    ax1.plot(suivi_time, suivi_ch4, label="CH4", linestyle="-", marker=".")
    ax1.plot(suivi_time, suivi_h2o, label="H2O", linestyle="-", marker=".")
    ax1.plot(suivi_time, suivi_co2, label="CO2", linestyle="-", marker=".")
    plt.legend()
    savename1 = f"{frame}_i{i}j{j}_species.png"
    plt.savefig(figpath+"\\"+savename1, dpi=108, bbox_inches="tight")
    plt.close(fig1)

    fig2 = plt.figure()
    plt.plot(suivi_time, suivi_T, linestyle="-", marker=".")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    savename2 = f"{frame}_i{i}j{j}_temperature.png"
    plt.savefig(figpath+"\\"+savename2, dpi=108, bbox_inches="tight")
    plt.close(fig2)

    fig3, ax3 = plt.subplots()
    ax3.clear()
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Absolute massic fraction derivative (/s)')
    ax3.semilogy(suivi_time, np.abs(np.gradient(suivi_o2, suivi_time))  , label="dO2/dt",   linestyle="-", marker=".")
    ax3.semilogy(suivi_time, np.abs(np.gradient(suivi_ch4, suivi_time)) , label="dCH4/dt",  linestyle="-", marker=".")
    ax3.semilogy(suivi_time, np.abs(np.gradient(suivi_h2o, suivi_time)) , label="dH2O/dt",  linestyle="-", marker=".")
    ax3.semilogy(suivi_time, np.abs(np.gradient(suivi_co2, suivi_time)) , label="dCO2/dt",  linestyle="-", marker=".")
    plt.legend()
    savename3 = f"{frame}_i{i}j{j}_species_deriv.png"
    plt.savefig(figpath+"\\"+savename3, dpi=108, bbox_inches="tight")
    plt.close(fig3)

    fig4 = plt.figure()
    plt.semilogy(suivi_time, np.abs(np.gradient(suivi_T, suivi_time)), linestyle="-", marker=".")
    plt.xlabel("Time (s)")
    plt.ylabel("Absolute temperature derivative (K/s)")
    savename4 = f"{frame}_i{i}j{j}_temperature_deriv.png"
    plt.savefig(figpath+"\\"+savename4, dpi=108, bbox_inches="tight")
    plt.close(fig2)


def chemistry_loop(o2:Dioxygen, ch4:Methane, h2o:Water,co2:CarbonDioxide, Temp:TemperatureField, L:float, frame:int, dt_transport:float, rho, c_p, Ta:float, i_reactor, j_reactor, showAndSave):
    
    thick = o2.ghost_thick
    physN = o2.N - 2*thick

    dx = L/(physN-1)
    Na = 6.0221408e+23 # Avogadro number
    zero_threshold = 1/(Na*dx**3) # if the density number gets under this threshold, it means that there is < 1 particle per mesh volume so its physically 0. 
    o2_arr = o2.values[thick:-thick,thick:-thick]
    ch4_arr = ch4.values[thick:-thick,thick:-thick]
    h2o_arr = h2o.values[thick:-thick,thick:-thick]
    co2_arr = co2.values[thick:-thick,thick:-thick]
    Temp_arr = Temp.values[thick:-thick,thick:-thick]

    # We dont know in advance how much time step there will be but 200 is a good number.
    suivitime = np.full(200, -1.0)
    suivio2 = np.full(200, -1.0)
    suivich4 = np.full(200, -1.0)
    suivih2o = np.full(200, -1.0)
    suivico2 = np.full(200, -1.0)
    suiviT = np.full(200, -1.0)

    # inserting ther first value inside.
    suivitime[0] = 0.
    # Les indices i_reactor et j_reactor ne prennent pas en compte la taille des ghost_cells. Donc ici pas besoin de retirer/ajouter thick.
    suivio2[0] = o2_arr[i_reactor, j_reactor]
    suivich4[0] = ch4_arr[i_reactor, j_reactor]
    suivih2o[0] = h2o_arr[i_reactor, j_reactor]
    suivico2[0] = co2_arr[i_reactor, j_reactor]
    suiviT[0] = Temp_arr[i_reactor, j_reactor]

    numba_RK2(physN, dt_transport, o2_arr, o2.W, o2.stoech, o2.Dhf, ch4_arr, ch4.W, ch4.stoech, ch4.Dhf, h2o_arr, h2o.W, h2o.stoech, h2o.Dhf, co2_arr, co2.W, co2.stoech, co2.Dhf, Temp_arr, Ta, rho, c_p, i_reactor, j_reactor, thick, zero_threshold,suivitime, suivio2, suivich4, suivih2o, suivico2, suiviT)

    o2.values[thick:-thick,thick:-thick]    =   o2_arr
    ch4.values[thick:-thick,thick:-thick]   =   ch4_arr
    h2o.values[thick:-thick,thick:-thick]   =   h2o_arr
    co2.values[thick:-thick,thick:-thick]   =   co2_arr
    Temp.values[thick:-thick,thick:-thick]  =   Temp_arr

    if showAndSave:
        suivitime = np.delete(suivitime, np.argwhere(suivitime == -1.0))
        suivio2 = np.delete(suivio2, np.argwhere(suivio2 == -1.0))
        suivich4 = np.delete(suivich4, np.argwhere(suivich4 == -1.0))
        suivih2o = np.delete(suivih2o, np.argwhere(suivih2o == -1.0))
        suivico2 = np.delete(suivico2, np.argwhere(suivico2 == -1.0))
        suiviT = np.delete(suiviT, np.argwhere(suiviT == -1.0))

        plot_chemistry(suivitime, suivio2, suivich4, suivih2o, suivico2, suiviT, i_reactor, j_reactor, frame)


@njit
def is_worth_continue(o2_previous:float, o2_current:float, Temp_previous:float, Temp_current:float, dtchem:float):
    max_derive_o2 = abs((o2_current - o2_previous)/dtchem)
    max_deriv_T = abs((Temp_current - Temp_previous)/dtchem)

    return (max_derive_o2 > 50. or max_deriv_T > 5e5)


@njit
def get_timescale_problem(Q, rho, c_p, o2_val, W_o2, st_o2, ch4_val, W_ch4, st_ch4, Dhf_ch4, st_h2o, Dhf_h2o, st_co2, Dhf_co2, T_val)->float:
    
    timescale = 1.
    if Q != 0.:
        timescales_arr = np.zeros(3, dtype=float)
        timescales_arr[0] = abs(ch4_val*rho/(st_ch4*W_ch4*Q)) #timescale of ch4 variation.
        timescales_arr[1] = abs(o2_val*rho/(st_o2*W_o2*Q)) #timescale of o2 variation.
        timescales_arr[2] = abs(T_val*rho*c_p/(-(Dhf_ch4*st_ch4 + Dhf_h2o*st_h2o + Dhf_co2*st_co2)*Q)) #timescale of T° variation.
        
        timescale = 0.2*np.min(timescales_arr)

    return timescale


def test():
    # Constants
    L = 2e-3
    physN = 45
    dx = L/(physN-1)
    D = 15e-6
    L_slot = 5e-4
    L_coflow = 5e-4
    rho = 1.1614
    c_p = 1200.
    thick = 2
    Ta = 10000.

    frame = 15 # whatever here

    max_u = 1. # because the max inlet velocity is 1 m/s.    
    max_Fo = 0.25 # Fourier threshold in 2D
    max_CFL = 0.5 # CFL limit thresholf in 2D
    dt = 0.4*min(max_Fo*(dx**2)/D, max_CFL*dx/max_u)   # Time step
    print(dt)

    # Initializing the species
    o2 = Dioxygen(np.full((physN, physN), 0.2, dtype=float), dx, L_slot, L_coflow, rho, False, thick)
    ch4 = Methane(np.full((physN, physN), 0.2, dtype=float), dx, L_slot, L_coflow, rho, False, thick)
    h2o = Water(np.zeros((physN, physN), dtype=float), dx, L_slot, L_coflow, rho, False, thick)
    co2 = CarbonDioxide(np.zeros((physN, physN), dtype=float), dx, L_slot, L_coflow, rho, False, thick)

    # Initializing the temperature
    Temp = TemperatureField(np.full((physN, physN), 300), dx, False, thick)

    # Ignition of the chamber
    Temp.ignite_or_not()

    # Coordinates to observe the reaction
    i_reactor = 22
    j_reactor = 0

    # Initial Picture
    misc.plot_field(o2, True, title=f'O2 Population k=0 ($\Delta t=${dt})', saveaspng="0_O2.png")
    misc.plot_field(ch4, True, title=f'CH4 Population k=0 ($\Delta t=${dt})', saveaspng="0_CH4.png")
    misc.plot_field(h2o, True, title=f'H2O Population k=0 ($\Delta t=${dt})', saveaspng="0_H2O.png")
    misc.plot_field(co2, True, title=f'CO2 Population k=0 ($\Delta t=${dt})', saveaspng="0_CO2.png")
    misc.plot_field(Temp, True, title=f'T° Field k=0 ($\Delta t=${dt})',saveaspng="0_temp_field.png")

    start_datetime = datetime.datetime.now()
    
    chemistry_loop(o2, ch4, h2o, co2, Temp, L, frame, dt, rho, c_p, Ta, i_reactor, j_reactor, True)

    stop_datetime = datetime.datetime.now()

    print(stop_datetime - start_datetime)

    # Final Picture
    misc.plot_field(o2, True, title=f'O2 Population k=0 ($\Delta t=${dt})', saveaspng="1_O2.png")
    misc.plot_field(ch4, True, title=f'CH4 Population k=0 ($\Delta t=${dt})', saveaspng="1_CH4.png")
    misc.plot_field(h2o, True, title=f'H2O Population k=0 ($\Delta t=${dt})', saveaspng="1_H2O.png")
    misc.plot_field(co2, True, title=f'CO2 Population k=0 ($\Delta t=${dt})', saveaspng="1_CO2.png")
    misc.plot_field(Temp, True, title=f'T° Field k=0 ($\Delta t=${dt})',saveaspng="1_temp_field.png")
