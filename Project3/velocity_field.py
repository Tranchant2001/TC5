# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
from field import Field



epsilon = 1e-9 # valeur en-dessous de laquelle on considère avoir 0.



class VelocityField():

    #__slots__ = "u", "v", "dx", "L_slot", "L_coflow", "got_ghost_cells", "ghost_thick", "shape", "N", "norm", "metric", "strain_rate", "max_strain_rate"

    def __init__(self, u_array:np.ndarray, v_array:np.ndarray, dx:float, L_slot:float, L_coflow:float, got_ghost_cells=False, ghost_thick=0):

        assert(u_array.shape == v_array.shape)
        assert(u_array.shape[0] == u_array.shape[1])

        self.shape = u_array.shape
        self.N = u_array.shape[0]

        self.L_slot = L_slot
        self.L_coflow = L_coflow
        self.dx = dx

        
        if got_ghost_cells and ghost_thick > 0:
            self.ghost_thick = ghost_thick
            self.got_ghost_cells = True
            self.u = Field(u_array, dx, True, ghost_thick)
            self.v = Field(u_array, dx, True, ghost_thick)
            self._fillGhosts()

        elif not got_ghost_cells and ghost_thick > 0:
            self.ghost_thick = 0
            self.got_ghost_cells = False
            self.u = Field(u_array, dx, False, 0)
            self.v = Field(u_array, dx, False, 0)            
            self._addGhosts(ghost_thick)
            self._fillGhosts()

        else:
            self.ghost_thick = 0
            self.got_ghost_cells = False
            self.u = Field(u_array, dx)
            self.v = Field(v_array, dx)


        # Initialisation du champ de norme. Ce champ n'a pas de ghost cells
        self.norm = Field(np.empty((self.N , self.N)), dx, self.got_ghost_cells, self.ghost_thick)
        self.norm._stripGhosts()
        self.metric = 1.
        self.strain_rate = np.zeros(self.N-2*self.ghost_thick, dtype=np.float32)
        self.max_strain_rate = 0.


    def _addGhosts(self, thick):

        if not self.got_ghost_cells:
            self.u._addGhosts(thick)
            self.v._addGhosts(thick)

            self.shape = self.u.shape
            self.N = self.shape[0]
            
            self.got_ghost_cells = True
            self.ghost_thick = thick


    def _stripGhosts(self):

        if self.got_ghost_cells:
            
            thick = self.ghost_thick
            
            self.u = self.u._stripGhosts()
            self.v = self.v._stripGhosts()

            self.L = self.L - 2*thick*self.dx
            self.N = self.N - 2*thick
            self.shape = (self.N, self.N)
            self.got_ghost_cells = False
            self.ghost_thick = 0

    
    def _fillGhosts(self):

        if self.got_ghost_cells:    
            thick = self.ghost_thick
            u_new = np.copy(self.u.values)
            v_new = np.copy(self.v.values)
            N = self.N
            dx = self.dx
            L_slot = self.L_slot
            L_coflow = self.L_coflow

            # inlet and walls conditions for u
            u_new[:,0:thick+1] = 0.
            u_new[0:thick+1,:] = 0.
            u_new[-thick-1:,:] = 0.

            # outlet condition
            u_new[:, N-thick-1] = np.where(self.u.values[:, N-thick-1] >= 0, (4*u_new[:, N-thick-2] - u_new[:, N-thick-3])/3, u_new[:, N-thick-1]) # ensure derivative is zero on the first ghost cell.
            v_new[:, N-thick-1] = np.where(self.u.values[:, N-thick-1] >= 0, (4*v_new[:, N-thick-2] - v_new[:, N-thick-3])/3, v_new[:, N-thick-1])
            u_new[:, N-thick:] = np.tile(u_new[:, N-thick-1], (thick, 1)).transpose()
            v_new[:, N-thick:] = np.tile(v_new[:, N-thick-1], (thick, 1)).transpose()

            # inlet conditions for v
            N_inlet = int(L_slot/dx)
            N_coflow = int(L_coflow/dx)

            v_new[:thick+1, thick : thick + N_inlet] = 1.0
            v_new[-thick-1:, thick : thick + N_inlet] = -1.0

            v_new[:thick+1, thick + N_inlet : thick + N_inlet + N_coflow] = 0.2
            v_new[-thick-1:, thick + N_inlet : thick + N_inlet + N_coflow] = -0.2

            v_new[:thick+1, thick + N_inlet + N_coflow:] = 0.
            v_new[-thick-1:, thick + N_inlet + N_coflow:] =  0.

            # slipping wall simple
            v_new[:, thick] = np.where(self.u.values[:, thick] <= 0, (4*v_new[:, thick+1] - v_new[:, thick+2])/3, v_new[:, thick])
            v_new[:, :thick] = np.tile(v_new[:, thick], (thick, 1)).transpose()

            """
            # Other slipping wall conditions
            for ii in range(1,N-1):
                if v[ii,0] > 0:
                    #v_new[ii][0] = v[ii][0] + delta_t*D*(v[ii+1][0] - v[ii][0] + v[ii-1][0] + v[ii][2] -2*v[ii][1])/(dx**2)
                    v_new[ii][0] = v[ii][0] - delta_t*v[ii][0]*(v[ii][0] - v[ii-1][0])/dx + delta_t*D*(v[ii+1][0] - v[ii][0] + v[ii-1][0] + v[ii][2] -2*v[ii][1])/(dx**2)
                    #v_new[ii][0] = v[ii][0] - delta_t*v[ii][0]*(v[ii][0] - v[ii-1][0])/dx

                elif v[ii,0] < 0:
                    #v_new[ii][0] = v[ii][0] + delta_t*D*(v[ii+1][0] - v[ii][0] + v[ii-1][0] + v[ii][2] -2*v[ii][1])/(dx**2)
                    v_new[ii][0] = v[ii][0] - delta_t*v[ii][0]*(v[ii+1][0] - v[ii][0])/dx + delta_t*D*(v[ii+1][0] - v[ii][0] + v[ii-1][0] + v[ii][2] -2*v[ii][1])/(dx**2)
                    #v_new[ii][0] = v[ii][0] - delta_t*v[ii][0]*(v[ii+1][0] - v[ii][0])/dx
                else:
                    v_new[ii][0] = delta_t*D*(v[ii+1][0] - v[ii][0] + v[ii-1][0] + v[ii][2] -2*v[ii][1])/(dx**2)
                    #v_new[ii][0] = D*(v[ii+1][0] - v[ii][0] + v[ii-1][0] + v[ii][2] -2*v[ii][1])/(dx**2)
            """

            # Dead corner values filled with -1.0
            u_new[:thick, :thick] = -1.0
            u_new[:thick, -thick:] = -1.0
            u_new[-thick:, :thick] = -1.0
            u_new[-thick:, -thick:] = -1.0
            
            v_new[:thick, :thick] = -1.0
            v_new[:thick, -thick:] = -1.0
            v_new[-thick:, :thick] = -1.0
            v_new[-thick:, -thick:] = -1.0 

            self.u.values = u_new
            self.v.values = v_new


    def update_metric(self):
        
        u = np.copy(self.u.values)
        v = np.copy(self.v.values)
        
        norm_array = np.sqrt(u*u + v*v)

        if self.got_ghost_cells:
            thick = self.ghost_thick
            N = self.N
            norm_array = norm_array[thick : N-thick , thick : N-thick]

        self.norm.values = norm_array

        mean = np.mean(norm_array)
        std = np.std(norm_array)
        
        if abs(mean) < epsilon and abs(std) < epsilon:
            self.metric = 1.

        elif abs(mean) < epsilon:
            self.metric = 1e3

        else:
            self.metric = std / mean


        # Process the strain rate on the slipping wall:
        slipping_wall_v = self.v.values[:, thick]
        self.strain_rate = np.absolute(np.gradient(slipping_wall_v, self.dx)[thick:-thick])
        self.max_strain_rate = np.max(self.strain_rate)
