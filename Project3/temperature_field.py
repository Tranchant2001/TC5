# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
from numba import njit

from field import Field
#from numba_functions import numba_update_SOR_forward, numba_update_SOR_backward


### CLASS   ###

class TemperatureField(Field):

    #__slots__ = "values", "dx", "got_ghost_cells", "ghost_thick", "shape", "N", "last_nb_iter"

    def __init__(self, array:np.ndarray, dx:float, got_ghost_cells=False, ghost_thick=0):

        super().__init__(array, dx, got_ghost_cells, ghost_thick)

        assert(array.shape[0] == array.shape[1])

        if got_ghost_cells and ghost_thick > 0:
            self.ghost_thick = ghost_thick
            self.got_ghost_cells = True
            self.fillGhosts()

        elif not got_ghost_cells and ghost_thick > 0:
            self.ghost_thick = 0
            self.got_ghost_cells = False          
            self._addGhosts(ghost_thick)
            self.fillGhosts()

        else:
            self.ghost_thick = 0
            self.got_ghost_cells = False

        self.N = self.shape[0]
        self.last_nb_iter = -1

    def fillGhosts(self):

        if self.got_ghost_cells:    
            thick = self.ghost_thick
            N = self.shape[0]

            T = self.values

            T[ : , : thick] = np.transpose(np.tile(T[:,thick], (thick,1)))
            T[ : thick, : ] = np.tile(T[thick,:], (thick,1))
            T[-thick : , : ] = np.tile(T[-thick-1,:], (thick,1))
            T[:,-thick:] = np.transpose(np.tile(T[-thick-1,:], (thick,1)))
            for i in range(thick):
                T[ : ,-i-1] = T[:,-thick-1]

            # Dead corner values filled with -1.0
            T[:thick, :thick] = 666.
            T[:thick, -thick:] = 666.
            T[-thick:, :thick] = 666.
            T[-thick:, -thick:] = 666.

    def ignite_or_not(self):
        T0 = np.copy(self.values)
        physN = self.N - 2*self.ghost_thick
        y_arr = np.linspace(0, self.dx*(physN-1), physN, endpoint=True, dtype=float)*1000 # axes des Y en mm.
        
        a = False
        b = False
        for k in range(physN):
            if not a and y_arr[k] >= 0.750:
                a = True
                binf = k
            elif a and not b and y_arr[k] >= 1.250:
                bsup = k
                b = True

        binf = binf + self.ghost_thick
        bsup = bsup + self.ghost_thick
        T0 = np.copy(self.values)[binf:bsup+1, :]
        self.values[binf:bsup+1, :] = np.where(T0 >= 1000., T0, 1000.)
        

             
