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

            T = self.values

            T[ : , : thick] = np.full((N, thick), 298.15)
            T[ : thick, : ] = np.full((thick, N), 298.15)
            T[-thick : , : ] = np.full((thick, N), 298.15)
            
            for i in range(thick):
                T[ : ,-i-1] = T[:,-thick-1]

            # Dead corner values filled with -1.0
            T[:thick, :thick] = -1.0
            T[:thick, -thick:] = -1.0
            T[-thick:, :thick] = -1.0
            T[-thick:, -thick:] = -1.0


             
