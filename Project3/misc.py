# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator

from velocity_field import VelocityField
from field import Field
from species import Dinitrogen
from species import Dioxygen
from temperature_field import TemperatureField



#Chemin absolu du fichier .py qu'on execute
fullpath = os.path.abspath(__file__)
#Chemin absolu du dossier contenant le .py qu'on execute
dirpath = os.path.dirname(fullpath)


def plot_field(fi:Field, display_ghost=False, **kwargs):
    """
    Plot the state of the scalar field.
    Specifies in the title the time and the metric
    """
    if fi.got_ghost_cells and not display_ghost:
        thick = fi.ghost_thick
        phi_copy = np.copy(fi.values[thick : -thick , thick : -thick])
    else: 
        phi_copy = np.copy(fi.values)

    # Create a figure and axis for the animation
    fig, ax = plt.subplots()
    
    # Plot the scalar field
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if 'title' in kwargs.keys():
        ax.set_title(kwargs.get('title'))

    image = ax.imshow(phi_copy, origin='lower', cmap='viridis')
    fig.colorbar(image, ax=ax)
    if 'saveaspng' in kwargs.keys():
        plt.savefig(dirpath+"/outputs_program_ecoulement/"+kwargs.get('saveaspng'), dpi=108, bbox_inches="tight")
    if 'pause' in kwargs.keys():
        plt.pause(kwargs.get("pause"))
    plt.close(fig)


def uv_copy(uv:VelocityField) -> VelocityField:
    N = uv.N
    new = VelocityField(np.zeros((N, N), dtype=np.float32), np.zeros((N, N), dtype=np.float32), uv.dx, uv.L_slot, uv.L_slot, uv.got_ghost_cells, uv.ghost_thick)
    new.v.values = np.copy(uv.v.values)
    new.u.values = np.copy(uv.u.values)
    new.update_metric()

    return new


def velocity_derivative_norm(uv0:VelocityField, uv1:VelocityField, dt:float) -> np.float32:
    thick0 = uv0.ghost_thick
    u0 = uv0.u.values[thick0:-thick0 , thick0:-thick0]
    v0 = uv0.v.values[thick0:-thick0 , thick0:-thick0]
    
    thick1 = uv1.ghost_thick
    u1 = uv1.u.values[thick1:-thick1 , thick1:-thick1]
    v1 = uv1.v.values[thick1:-thick1 , thick1:-thick1]

    return np.mean(np.sqrt( ((u0 - u1)/dt)**2 + ((v0 - v1)/dt)**2))


def array_residual(f0:np.ndarray, thick0:int, f1:np.ndarray, thick1:int) -> np.float32:
    
    v0 = np.copy(f0)[thick0:-thick0 , thick0:-thick0]

    v1 = np.copy(f1)[thick1:-thick1 , thick1:-thick1]

    return np.mean(np.sqrt((v0 - v1)**2))


def plot_uv(uv:VelocityField, X, Y, **kwargs):

    if uv.got_ghost_cells:
        N = uv.N
        thick = uv.ghost_thick
        u = uv.u.values[thick : thick, thick : thick]
        v = uv.v.values[thick : thick, thick : thick]
    
    else:
        u = uv.u.values
        v = uv.v.values

    # Create a figure and axis for the animation
    fig1, ax1 = plt.subplots() 
    
    # Plot the scalar field
    ax1.clear()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    if 'title' in kwargs.keys():
        ax1.set_title(kwargs.get('title'))
    ax1.quiver(X, Y, u, v, scale=5)
    if 'saveaspng' in kwargs.keys():
        plt.savefig(dirpath+"/outputs_program_ecoulement/"+kwargs.get('saveaspng'), dpi=108, bbox_inches="tight")
    if 'pause' in kwargs.keys():
        plt.pause(kwargs.get('pause'))
    plt.close(fig1)


def plot_strain_rate(uv:VelocityField, y, **kwargs):

    fig2, ax2 = plt.subplots()
    ax2.clear()
    ax2.set_xlabel('Y (mm)')
    ax2.set_ylabel('Strain rate (Hz)')
    if 'title' in kwargs.keys():
        ax2.set_title(kwargs.get('title'))
    ax2.plot(1000*y, uv.strain_rate)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if 'saveaspng' in kwargs.keys():
        plt.savefig(dirpath+"/outputs_program_ecoulement/"+kwargs.get('saveaspng'), dpi=108, bbox_inches="tight")
    if 'pause' in kwargs.keys():
        plt.pause(kwargs.get('pause'))
    plt.close(fig2)


def plot_diffusive_zone(n2:Dinitrogen, y, frame:int, dt, time, **kwargs) -> float:

    # Plot the diffusive zone in the whole engine area.
    # Then plots the Dinitrogen population on the left wal and determines the diffusive region.
    
    dx = n2.dx
    thick = n2.ghost_thick

    diffusive_zone = np.copy(n2.values[thick : -thick , thick : -thick])
    diffusive_zone = np.where((diffusive_zone > 0.08)&(diffusive_zone < 0.72), diffusive_zone, 0.)

    # Create a figure and axis for the animation
    fig1, ax1 = plt.subplots()
    
    # Plot the scalar field
    ax1.clear()
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title(f"Diffusive zone k={frame} ($\Delta t=${dt}, N={n2.N}) \n Time: {time:.6f} s")

    image = ax1.imshow(diffusive_zone, origin='lower', cmap='viridis')
    fig1.colorbar(image, ax=ax1)

    plt.savefig(dirpath+"/outputs_program_ecoulement/"+str(frame)+"_diff_zone.png", dpi=108, bbox_inches="tight")
    if 'pause' in kwargs.keys():
        plt.pause(kwargs.get("pause"))
    plt.close(fig1)

    del diffusive_zone

    left_wall_n2 = np.copy(n2.values[thick : -thick, thick])
    fig2 = plt.figure()

    # Plot the N2 population on the left wall.
    pix_L = 0
    k = 0
    while left_wall_n2[k] <= 0.72:
        if left_wall_n2[k] >= 0.08:
            pix_L += 1
        k += 1

    length = (pix_L*dx)*1000 # thickness of the diffusive zone in mm.
    n2.diff_zone_thick = length

    ysup = (y[k] - dx/2)*1000
    yinf = (y[k] - dx/2)*1000 - length

    max_n2 = np.max(left_wall_n2)
    min_n2 = np.min(left_wall_n2)

    fig2, ax2 = plt.subplots()
    ax2.set_xlabel('Y (mm)')
    ax2.set_ylabel('Dinitrogen massic fraction')
    ax2.plot(1000*y, left_wall_n2, label = "$N_2$ mass fraction")
    ax2.plot([yinf, yinf], [min_n2, max_n2], linestyle='dashed', color="black", label = f"Diffusive zone\n(Thickness={length:.3f} mm)")
    ax2.plot([ysup, ysup], [min_n2, max_n2], linestyle='dashed', color="black")
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.set_title("$Y_{N_2}$ on the left wall"+ f" k={frame} ($\Delta t=${dt}, N={n2.N}) \n Time: {time:.6f} s")
    plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1.))
    plt.savefig(dirpath+"/outputs_program_ecoulement/"+str(frame)+"_N2_left_wall.png", dpi=108, bbox_inches="tight")
    if 'pause' in kwargs.keys():
        plt.pause(kwargs.get('pause'))
    plt.close(fig2)


def print_write_end_message(code, div_crit, max_t_comput, conv_crit, L, D, N, dt, duration_delta, time, frame, uv_consecutive_diff, max_strain_rate, diff_zone_thick):
    
    assert(code in ["divergence", "timeout", "success"])
    
    first_line = ""
    
    if code == "divergence":
        first_line = f"Warning: The simulation stopped running because a divergence was detected (vel_metric >= {div_crit})."
    elif code == "timeout":
        first_line = "Warning: The simulation stopped running because the max duration of simulation ("+str(max_t_comput)+") was reached."
    elif code == "success":
        first_line = f"Success: The simulation stopped running because the velocity field was stable enough (uv_consecutive_difference < {conv_crit:.2e})."

    parameters =   f"\tParameters: (L, D, N, $\Delta t$)=({L}, {D}, {N}, {dt})"
    simu_duration = "\tSimulation duration: "+str(duration_delta)
    vtime =        f"\tVirtual stop time: {time} s"
    vframe =       f"\tVirtual stop frame: {frame}"
    uv_difference =f"\tVelocity consecutive difference: {uv_consecutive_diff:.2e}"
    max_srate =    f"\tMaximum strain rate on the left wall: {max_strain_rate} Hz"
    diff_zone =    f"\tThickness of the diffusive zone: {diff_zone_thick} mm"

    message = first_line + "\n" + parameters + "\n" + simu_duration + "\n" + vtime + "\n" + vframe + "\n" + uv_difference + "\n" + max_srate+ "\n" + diff_zone + "\n"

    print(message)

    endfile = open(dirpath+"/outputs_program_ecoulement/simulation_report.txt", "w")
    endfile.write(message)


def plot_chemistry(suivi_time, suivi_o2, suivi_n2, suivi_ch4, suivi_h2o, suivi_co2, suivi_T, i, j, frame):

    fig1, ax1 = plt.subplots()
    ax1.clear()
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Massic fraction')
    ax1.plot(suivi_time, suivi_o2, label="O2", linestyle="-", marker=".")
    ax1.plot(suivi_time, suivi_n2, label="N2", linestyle="-", marker=".")
    ax1.plot(suivi_time, suivi_ch4, label="CH4", linestyle="-", marker=".")
    ax1.plot(suivi_time, suivi_h2o, label="H2O", linestyle="-", marker=".")
    ax1.plot(suivi_time, suivi_co2, label="CO2", linestyle="-", marker=".")
    plt.legend()
    savename1 = f"{frame}_i{i}j{j}_species.png"
    plt.savefig(dirpath+"/outputs_program_ecoulement/"+savename1, dpi=108, bbox_inches="tight")
    plt.close(fig1)

    fig2 = plt.figure()
    plt.plot(suivi_time, suivi_T, linestyle="-", marker=".")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    savename2 = f"{frame}_i{i}j{j}_temperature.png"
    plt.savefig(dirpath+"/outputs_program_ecoulement/"+savename2, dpi=108, bbox_inches="tight")
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
    plt.savefig(dirpath+"/outputs_program_ecoulement/"+savename3, dpi=108, bbox_inches="tight")
    plt.close(fig3)

    fig4 = plt.figure()
    plt.semilogy(suivi_time, np.abs(np.gradient(suivi_T, suivi_time)), linestyle="-", marker=".")
    plt.xlabel("Time (s)")
    plt.ylabel("Absolute temperature derivative (K/s)")
    savename4 = f"{frame}_i{i}j{j}_temperature_deriv.png"
    plt.savefig(dirpath+"/outputs_program_ecoulement/"+savename4, dpi=108, bbox_inches="tight")
    plt.close(fig2)


def plot_consecutive_deriv(analysis_array:np.ndarray, **kwargs):

    fig, ax1 = plt.subplots(layout="constrained")

    frame_arr = analysis_array[0,:]
    data_arr = analysis_array[2,:]

    ax1.semilogy(frame_arr, data_arr)
    ax1.set_xlabel("Frame index")
    ax1.set_ylabel("Consecutive Velocity difference")
    if 'title' in kwargs.keys():
        ax1.set_title(kwargs.get('title'))

    if 'saveaspng' in kwargs.keys():
        plt.savefig(dirpath+"/outputs_program_ecoulement/"+kwargs.get('saveaspng'), dpi=108, bbox_inches="tight")
    if 'pause' in kwargs.keys():
        plt.pause(kwargs.get('pause'))
    plt.close(fig)


def plot_array(array, **kwargs):

    """
    Plot the state of the scalar field.
    Specifies in the title the time and the metric
    """

    # Create a figure and axis for the animation
    fig, ax = plt.subplots()
    
    # Plot the scalar field
    ax.clear()
    ax.set_xlabel('Column index')
    ax.set_ylabel('Row index')
    if 'title' in kwargs.keys():
        ax.set_title(kwargs.get('title'))

    image = ax.imshow(array, origin='lower', cmap='viridis')
    fig.colorbar(image, ax=ax)
    if 'saveaspng' in kwargs.keys():
        plt.savefig(dirpath+"/outputs_program_ecoulement/"+kwargs.get('saveaspng'), dpi=108, bbox_inches="tight")
    if 'pause' in kwargs.keys():
        plt.show()
        plt.pause(kwargs.get("pause"))
    plt.close(fig)


def worth_continue_chemistry(o2_0:np.ndarray, o2_1:np.ndarray, T_0:np.ndarray, T_1:np.ndarray, thick:int, dt:float):
    
    o2_0 = o2_0[thick:-thick , thick:-thick]
    o2_1 = o2_1[thick:-thick , thick:-thick]
    T_0 = T_0[thick:-thick , thick:-thick]
    T_1 = T_1[thick:-thick , thick:-thick]

    max_derive_o2 = np.max(np.abs((o2_1 - o2_0)/dt))
    max_deriv_T = np.max(np.abs((T_1 - T_0)/dt))

    return (max_derive_o2 > 50. or max_deriv_T > 5e5)


def two_arrays_derivative(arr0:np.ndarray, arr1:np.ndarray, dt:float) -> np.float32:
    pass

def register_array_csv(filename:str, arr:np.ndarray):

    np.savetxt(dirpath+"/outputs_program_ecoulement/"+filename, arr, delimiter="\t", fmt="%.3e")

