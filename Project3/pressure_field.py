# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
from numba import njit

from field import Field
#from numba_functions import numba_update_SOR_forward, numba_update_SOR_backward


### NUMBA FUNCTIONS ###

@njit
def numba_update_SOR_forward(p:np.ndarray, b:np.ndarray, N:int, thick:int, w:float) -> np.ndarray:

    pk1 = np.copy(p)

    for i in range(thick, N-thick):
        for j in range(thick, N-thick):
            pk1[i][j] = (1 - w)*p[i][j] +  w*0.25*(pk1[i-1][j] +pk1[i][j-1] + p[i+1][j] + p[i][j+1] - b[i][j])

    return pk1


@njit
def numba_update_SOR_backward(p:np.ndarray, b:np.ndarray, N:int, thick:int, w:float) -> np.ndarray:

    pk1 = np.copy(p)

    for i in range(N-thick-1, thick-1, -1):
        for j in range(thick, N-thick):
            pk1[i][j] = (1 - w)*p[i][j] +  w*0.25*(pk1[i+1][j] +pk1[i][j-1] + p[i-1][j] + p[i][j+1] - b[i][j])

    return pk1 

### CLASS   ###

class PressureField(Field):

    #__slots__ = "values", "dx", "got_ghost_cells", "ghost_thick", "shape", "N", "last_nb_iter"

    def __init__(self, array:np.ndarray, dx:float, got_ghost_cells=False, ghost_thick=0):

        assert(array.shape[0] == array.shape[1])

        super().__init__(array, dx, got_ghost_cells, ghost_thick)

        if got_ghost_cells and ghost_thick > 0:
            self.ghost_thick = ghost_thick
            self.got_ghost_cells = True
            self._fillGhosts()

        elif not got_ghost_cells and ghost_thick > 0:
            self.ghost_thick = 0
            self.got_ghost_cells = False          
            self._addGhosts(ghost_thick)
            self._fillGhosts()

        else:
            self.ghost_thick = 0
            self.got_ghost_cells = False

        self.N = self.shape[0]
        self.last_nb_iter = -1


    def _fillGhosts(self):

        if self.got_ghost_cells:    
            thick = self.ghost_thick
            N = self.shape[0]
            P_new = np.copy(self.values)

            P_new[:,-thick:] = np.full((N, thick), 1.0)
            
            for i in range(thick):

                P_new[:,i] = P_new[:,thick]
                P_new[i,:] = P_new[thick,:]
                P_new[-i-1,:] = P_new[-thick-1,:]

            # Dead corner values filled with -1.0
            P_new[:thick, :thick] = -1.0
            P_new[:thick, -thick:] = -1.0
            P_new[-thick:, :thick] = -1.0
            P_new[-thick:, -thick:] = -1.0

            self.values = P_new
             

    def update_forward_SOR(self, b, w):

        assert(self.got_ghost_cells == True)

        self.values = numba_update_SOR_forward(self.values, b, self.N, self.ghost_thick, w)
        self._fillGhosts()


    def update_backward_SOR(self, b, w):

        assert(self.got_ghost_cells == True)
        
        self.values = numba_update_SOR_backward(self.values, b, self.N, self.ghost_thick, w)
        self._fillGhosts()

    
    def residual_error(self, b) -> float:

        assert(self.got_ghost_cells == True)

        p = self.values
        thick = self.ghost_thick

        p_xx = np.roll(p, 1, axis=1) - 2*p + np.roll(p, -1, axis=1)
        p_yy = np.roll(p, 1, axis=0) - 2*p + np.roll(p, -1, axis=0)
        Ap = p_xx + p_yy
        residu = np.sqrt((b - Ap)*(b - Ap))

        # there may be problems at the edges so they are not considered to process the residual norm
        residu = residu[thick : -thick, thick : -thick]

        return np.mean(residu)
