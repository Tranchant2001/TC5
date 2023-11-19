# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os



epsilon = 1e-9 # valeur en-dessous de laquelle on considère avoir 0.



class Field():

   #__slots__ = "values", "dx", "got_ghost_cells", "ghost_thick", "shape"

    def __init__(self, array:np.ndarray, dx:float, got_ghost_cells=False, ghost_thick=0):
        
        self.values = array
        self.dx = dx
        self.got_ghost_cells = got_ghost_cells
        self.ghost_thick = ghost_thick
        
        self.shape = array.shape
    
    
    def _addGhosts(self, thick:int):

        if not self.got_ghost_cells:
            x_wide = np.full((self.shape[0]+2*thick, self.shape[1]+2*thick), -1.0, dtype=np.float32)
            x_wide[thick:-thick, thick:-thick] = self.values
            self.values = x_wide

            self.shape = self.values.shape
            self.got_ghost_cells = True
            self.ghost_thick = thick


    def threshold_zeros(self):

        self.values = np.where(np.absolute(self.values) < epsilon, 0., self.values)

    
    def _stripGhosts(self):

        if self.got_ghost_cells:
            
            thick = self.ghost_thick
            self.values = self.values[thick:-thick, thick:-thick]

            self.shape = self.values.shape
            self.got_ghost_cells = False
            self.ghost_thick = 0

    
    def _get_metric(self):
        """
        Calculate standard deviation and average of phi
        """
        values_copy = np.copy(self.values)
        if self.got_ghost_cells:
            thick = self.ghost_thick
            values_copy = values_copy[thick : -thick , thick : -thick]

        # Calculate the metric
        metric = np.std(values_copy) / np.mean(values_copy)

        return metric


    def derivative_x(self):
        
        assert(self.got_ghost_cells == True)

        d_x = (0.5/self.dx)*(np.roll(self.values, -1, axis=1) - np.roll(self.values, 1, axis=1))

        return d_x
    

    def derivative_y(self):
        
        assert(self.got_ghost_cells == True)

        d_y = (0.5/self.dx)*(np.roll(self.values, -1, axis=0) - np.roll(self.values, 1, axis=0))

        return d_y
