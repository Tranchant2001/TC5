# -*- coding: utf-8 -*-
"""
Projet n°1 du TC5: Numerical Methods du Master PPF.
Travail en binôme:
    Jules Chaveyriat
    Martin Guillon

Créé le 06/09/2023.
Mis à jour le 14/09/2023.
v1.0

DESCRIPTION:
"""


### PACKAGES    ###

import math
import numpy as np
import matplotlib.pyplot as plt


### FONCTIONS   ###

def grid():
    # initialisation de mon espace x, y    
    x =  np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    # fonction numpy voir doc pour fonctionnement exact
    return np.meshgrid(x, y)


def input_velocity_field(X, Y, optional_display=False):
    
    # fonction renvoyant la composante u de la velocite
    def input_u_field(x, y):
        u = np.cos(4*math.pi*x)*np.sin(4*np.pi*y)
        return u
    
    # fonction renvoyant la composante v de la velocite    
    def input_v_field(x, y):
        v = -np.sin(4*math.pi*x)*np.cos(4*np.pi*y)
        return v
    

    
    # Numpy peut appliquer des scalaires ou des tableaux à  ses fonctions natives. Pas besoin de map
    U = input_u_field(X, Y)
    V = input_v_field(X, Y)
    
    if optional_display:
        # fonction de matplotlib pour représenter un champ de vecteurs.
        plt.quiver(X, Y, U, V)
        plt.show()
    
    return U, V


def input_phi_field(X, Y, optional_display=False):
    x0 = 0.5
    y0 = 0.5
    
    def input_phi_field_point(x, y):
        r = np.sqrt((x-x0)**2+(y-y0)**2)
        condition = r <= 0.3
        result = np.zeros_like(r, dtype=float)
        result[condition] = 1.
        return result
        
    phi_field = input_phi_field_point(X, Y)
    
    if optional_display:
        plt.imshow(phi_field, cmap='viridis')
    
    return phi_field


### MAIN    ###

X, Y = grid()
U, V = input_velocity_field(X, Y, True)
PHI = input_phi_field(X, Y, True)
