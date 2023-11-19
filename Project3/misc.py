# -*- coding: utf-8 -*-

import numpy as np
from velocity_field import VelocityField
from field import Field
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FormatStrFormatter



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


def velocity_residual(uv0:VelocityField, uv1:VelocityField) -> np.float32:
    thick0 = uv0.ghost_thick
    u0 = uv0.u.values[thick0:-thick0 , thick0:-thick0]
    v0 = uv0.v.values[thick0:-thick0 , thick0:-thick0]
    thick1 = uv1.ghost_thick
    u1 = uv1.u.values[thick1:-thick1 , thick1:-thick1]
    v1 = uv1.v.values[thick1:-thick1 , thick1:-thick1]

    return np.mean(np.sqrt((u0 - u1)**2 + (v0 - v1)**2))


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


def plot_strain_rate(uv, y, **kwargs):

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