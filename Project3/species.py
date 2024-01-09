# -*- coding: utf-8 -*-

import numpy as np

from field import Field
from velocity_field import VelocityField

### CLASS   ###

class Species(Field):

    def __init__(self, array: np.ndarray, dx: float, L_slot:float, L_coflow:float, W:float, stoech:int, pho:float, symbol:str, got_ghost_cells=False, ghost_thick=0):
        
        super().__init__(array, dx, got_ghost_cells, ghost_thick)

        assert(array.shape[0] == array.shape[1])

        self.L_slot = L_slot
        self.L_coflow = L_coflow
        self.W = W
        self.stoech = stoech
        self.pho = pho
        self.symbol = symbol

        self.reaction_rate = np.full((array.shape[0], array.shape[0]), 666.0) # Initial weird values
        self.concentration = np.full((array.shape[0], array.shape[0]), 666.0)

    def update_reaction_rate(self, Q:np.ndarray):

        reaction_rate = self.reaction_rate
        reaction_rate = self.W*self.stoech*Q

        if self.got_ghost_cells:
            thick = self.ghost_thick
            # Dead corner values filled with -1.0
            reaction_rate[:thick, :thick] = -1.0
            reaction_rate[:thick, -thick:] = -1.0
            reaction_rate[-thick:, :thick] = -1.0
            reaction_rate[-thick:, -thick:] = -1.0

    def update_concentration(self):
        
        concentration_arr = self.concentration
        concentration_arr = self.pho*self.values/self.w
        
        if self.got_ghost_cells:
            thick = self.ghost_thick
            # Dead corner values filled with -1.0
            concentration_arr[:thick, :thick] = -1.0
            concentration_arr[:thick, -thick:] = -1.0
            concentration_arr[-thick:, :thick] = -1.0
            concentration_arr[-thick:, -thick:] = -1.0



class Dioxygen(Species):

    #__slots__ = "values", "dx", "L_slot", "L_coflow", "got_ghost_cells", "ghost_thick"

    def __init__(self, array:np.ndarray, dx:float, L_slot:float, L_coflow:float, pho:float, got_ghost_cells=False, ghost_thick=0):

        super().__init__(array, dx, L_slot, L_coflow, 31.999e-3, -2, pho, "O_{2}", got_ghost_cells, ghost_thick)

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
                   
    def fillGhosts(self):
    
        Y = self.values
        thick = self.ghost_thick
        dx = self.dx
        L_slot = self.L_slot
        L_coflow = self.L_coflow
        
        # length in nb of cells of the co_flow and the slot.
        N_inlet = int(L_slot/dx)
        N_coflow = int(L_coflow/dx)
        
        # Walls Neumann boundary condition.
        for i in range(thick):
            Y[:, i] = Y[:, thick]
            Y[:, -i-1] = Y[:, -thick-1]
            Y[i, :] = Y[thick, :]
            Y[-i-1, :] = Y[-1-thick, :]

        # Bottom slot    
        Y[-thick : , thick : thick + N_inlet] = 0.2
        # Top slot
        Y[ : thick, thick : thick + N_inlet] = 0.0
        # Bottom coflow
        Y[-thick : , thick + N_inlet : thick + N_inlet + N_coflow] = 0.0
        # Top coflow
        Y[ : thick, thick + N_inlet : thick + N_inlet + N_coflow] = 0.0

        # Dead corner values filled with -1.0
        Y[:thick, :thick] = -1.0
        Y[:thick, -thick:] = -1.0
        Y[-thick:, :thick] = -1.0
        Y[-thick:, -thick:] = -1.0



class Dinitrogen(Species):

    #__slots__ = "values", "dx", "L_slot", "L_coflow", "got_ghost_cells", "ghost_thick"

    def __init__(self, array:np.ndarray, dx:float, L_slot:float, L_coflow:float, pho:float, got_ghost_cells=False, ghost_thick=0):

        super().__init__(array, dx, L_slot, L_coflow, 28.01e-3, 0, pho, "N_{2}", got_ghost_cells, ghost_thick)

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

        # Initializing the thickness of the diffusive zone in mm.
        self.diff_zone_thick = 0.
    
    def fillGhosts(self):
            
        Y = self.values
        thick = self.ghost_thick
        dx = self.dx
        L_slot = self.L_slot
        L_coflow = self.L_coflow
        
        # length in nb of cells of the co_flow and the slot.
        N_inlet = int(L_slot/dx)
        N_coflow = int(L_coflow/dx)
        
        # Walls Neumann boundary condition.
        for i in range(thick):
            Y[:, i] = Y[:, thick]
            Y[:, -i-1] = Y[:, -thick-1]
            Y[i, :] = Y[thick, :]
            Y[-i-1, :] = Y[-1-thick, :]

        # Bottom slot    
        Y[-thick : , thick : thick + N_inlet] = 0.8
        # Top slot
        Y[ : thick, thick : thick + N_inlet] = 0.0
        # Bottom coflow
        Y[-thick : , thick + N_inlet : thick + N_inlet + N_coflow] = 1.0
        # Top coflow
        Y[ : thick, thick + N_inlet : thick + N_inlet + N_coflow] = 1.0

        # Dead corner values filled with -1.0
        Y[:thick, :thick] = -1.0
        Y[:thick, -thick:] = -1.0
        Y[-thick:, :thick] = -1.0
        Y[-thick:, -thick:] = -1.0



class Methane(Species):

    #__slots__ = "values", "dx", "L_slot", "L_coflow", "got_ghost_cells", "ghost_thick"

    def __init__(self, array:np.ndarray, dx:float, L_slot:float, L_coflow:float, pho:float, got_ghost_cells=False, ghost_thick=0):

        super().__init__(array, dx, L_slot, L_coflow, 16.04e-3, -1, pho, "CH_{4}", got_ghost_cells, ghost_thick)

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

    def fillGhosts(self):

        Y = self.values
        thick = self.ghost_thick
        dx = self.dx
        L_slot = self.L_slot
        L_coflow = self.L_coflow
        
        # length in nb of cells of the co_flow and the slot.
        N_inlet = int(L_slot/dx)
        N_coflow = int(L_coflow/dx)
        
        # Walls Neumann boundary condition.
        for i in range(thick):
            Y[:, i] = Y[:, thick]
            Y[:, -i-1] = Y[:, -thick-1]
            Y[i, :] = Y[thick, :]
            Y[-i-1, :] = Y[-1-thick, :]

        # Bottom slot    
        Y[-thick : , thick : thick + N_inlet] = 0.
        # Top slot
        Y[ : thick, thick : thick + N_inlet] = 1.
        # Bottom coflow
        Y[-thick : , thick + N_inlet : thick + N_inlet + N_coflow] = 0.
        # Top coflow
        Y[ : thick, thick + N_inlet : thick + N_inlet + N_coflow] = 0.

        # Dead corner values filled with -1.0
        Y[:thick, :thick] = -1.0
        Y[:thick, -thick:] = -1.0
        Y[-thick:, :thick] = -1.0
        Y[-thick:, -thick:] = -1.0



class Water(Species):

    #__slots__ = "values", "dx", "L_slot", "L_coflow", "got_ghost_cells", "ghost_thick"

    def __init__(self, array:np.ndarray, dx:float, L_slot:float, L_coflow:float, pho:float, got_ghost_cells=False, ghost_thick=0):

        super().__init__(array, dx, L_slot, L_coflow, 18.01528e-3, 2, pho, "H_{2}O", got_ghost_cells, ghost_thick)

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

    def fillGhosts(self):

        Y = self.values
        thick = self.ghost_thick
        dx = self.dx
        L_slot = self.L_slot
        L_coflow = self.L_coflow
        
        # length in nb of cells of the co_flow and the slot.
        N_inlet = int(L_slot/dx)
        N_coflow = int(L_coflow/dx)
        
        # Walls Neumann boundary condition.
        for i in range(thick):
            Y[:, i] = Y[:, thick]
            Y[:, -i-1] = Y[:, -thick-1]
            Y[i, :] = Y[thick, :]
            Y[-i-1, :] = Y[-1-thick, :]

        # Bottom slot    
        Y[-thick : , thick : thick + N_inlet] = 0.
        # Top slot
        Y[ : thick, thick : thick + N_inlet] = 0.
        # Bottom coflow
        Y[-thick : , thick + N_inlet : thick + N_inlet + N_coflow] = 0.
        # Top coflow
        Y[ : thick, thick + N_inlet : thick + N_inlet + N_coflow] = 0.

        # Dead corner values filled with -1.0
        Y[:thick, :thick] = -1.0
        Y[:thick, -thick:] = -1.0
        Y[-thick:, :thick] = -1.0
        Y[-thick:, -thick:] = -1.0



class CarbonDioxide(Species):

    #__slots__ = "values", "dx", "L_slot", "L_coflow", "got_ghost_cells", "ghost_thick"

    def __init__(self, array:np.ndarray, dx:float, L_slot:float, L_coflow:float, pho:float, got_ghost_cells=False, ghost_thick=0):

        super().__init__(array, dx, L_slot, L_coflow, 44.01e-3, 1, pho, "CO_{2}", got_ghost_cells, ghost_thick)

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

    def fillGhosts(self):

        Y = self.values
        thick = self.ghost_thick
        dx = self.dx
        L_slot = self.L_slot
        L_coflow = self.L_coflow
        
        # length in nb of cells of the co_flow and the slot.
        N_inlet = int(L_slot/dx)
        N_coflow = int(L_coflow/dx)
        
        # Walls Neumann boundary condition.
        for i in range(thick):
            Y[:, i] = Y[:, thick]
            Y[:, -i-1] = Y[:, -thick-1]
            Y[i, :] = Y[thick, :]
            Y[-i-1, :] = Y[-1-thick, :]

        # Bottom slot    
        Y[-thick : , thick : thick + N_inlet] = 0.
        # Top slot
        Y[ : thick, thick : thick + N_inlet] = 0.
        # Bottom coflow
        Y[-thick : , thick + N_inlet : thick + N_inlet + N_coflow] = 0.
        # Top coflow
        Y[ : thick, thick + N_inlet : thick + N_inlet + N_coflow] = 0.

        # Dead corner values filled with -1.0
        Y[:thick, :thick] = -1.0
        Y[:thick, -thick:] = -1.0
        Y[-thick:, :thick] = -1.0
        Y[-thick:, -thick:] = -1.0

