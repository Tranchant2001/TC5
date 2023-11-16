# -*- coding: utf-8 -*-
"""
Projet n°3 du TC5: Numerical Methods du Master PPF.
Travail en binôme:
    Jules Chaveyriat
    Martin Guillon

Créé le 24/10/2023.
Mis à jour le 14/11/2023.
v1.1

DESCRIPTION:
Propagating the Fluid flow in the chamber.

"""



### PACKAGES    ###

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import datetime
from numba import float32, int8, boolean, double, int16
from numba import njit
from numba.experimental import jitclass


### GENERAL FUNCTIONS   ###

class Field():

    def __init__(self, array:np.ndarray, dx:float, got_ghost_cells=False, ghost_thick=0):
        
        self.values = array
        self.dx = dx
        self.shape = array.shape
        self.got_ghost_cells = got_ghost_cells
        self.ghost_thick = ghost_thick
        
    
    
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


    def plot(self, display_ghost=False, **kwargs):
        """
        Plot the state of the scalar field.
        Specifies in the title the time and the metric
        """
        if self.got_ghost_cells and not display_ghost:
            thick = self.ghost_thick
            phi_copy = np.copy(self.values[thick : -thick , thick : -thick])
        else: 
            phi_copy = np.copy(self.values)

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



class VelocityField():

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
        # self.update_metric()


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
            u_new[:,0:thick] = 0.
            u_new[0:thick,:] = 0.
            u_new[-thick:,:] = 0.

            # outlet condition
            u_new[: , -thick:] = u_new[: , -2*thick : -thick]
            v_new[: , -thick:] = v_new[: , -2*thick : -thick]

            # inlet conditions for v
            N_inlet = int(L_slot/dx)
            N_coflow = int(L_coflow/dx)

            v_new[:thick, thick : thick + N_inlet] = np.full((thick, N_inlet), 1.0)
            v_new[-thick:, thick : thick + N_inlet] = np.full((thick, N_inlet), -1.0)

            v_new[:thick, thick + N_inlet : thick + N_inlet + N_coflow] = np.full((thick, N_coflow), 0.2)
            v_new[-thick:, thick + N_inlet : thick + N_inlet + N_coflow] = np.full((thick, N_coflow), -0.2)

            v_new[:thick, thick + N_inlet + N_coflow:] = 0.
            v_new[-thick:, thick + N_inlet + N_coflow:] =  0.

            # slipping wall simple
            for j in range(thick):
                v_new[:,j] = v_new[:,thick]

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

            self.u.threshold_zeros()
            self.v.threshold_zeros()   


    def update_metric(self):
        
        u = np.copy(self.u.values)
        v = np.copy(self.v.values)
        
        norm_array = np.sqrt(u*u + v*v)

        if self.got_ghost_cells:
            thick = self.ghost_thick
            N = self.N
            norm_array = norm_array[thick : N-thick , thick : N-thick]

        self.norm.values = norm_array
        self.norm.threshold_zeros()
        mean = np.mean(norm_array)
        std = np.std(norm_array)
        
        if abs(mean) < epsilon and abs(std) < epsilon:
            self.metric = 1.

        elif abs(mean) < epsilon:
            self.metric = 1e3

        else:
            self.metric = std / mean


    def plot(self, X, Y, **kwargs):

        if self.got_ghost_cells:
            N = self.N
            thick = self.ghost_thick
            u = self.u.values[thick : thick, thick : thick]
            v = self.v.values[thick : thick, thick : thick]
        
        else:
            u = self.u.values
            v = self.v.values

        # Create a figure and axis for the animation
        fig, ax = plt.subplots() 
        
        # Plot the scalar field
        ax.clear()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if 'title' in kwargs.keys():
            ax.set_title(kwargs.get('title'))
        ax.quiver(X, Y, u, v, scale=5)
        if 'saveaspng' in kwargs.keys():
            plt.savefig(dirpath+"/outputs_program_ecoulement/"+kwargs.get('saveaspng'), dpi=108, bbox_inches="tight")
        if 'pause' in kwargs.keys():
            plt.pause(kwargs.get('pause'))
        plt.close(fig)



class PressureField(Field):

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

            self.threshold_zeros()              


    def update_forward_SOR(self, b, w):

        assert(self.got_ghost_cells == True)
        p = self.values
        N = self.N
        thick = self.ghost_thick

        pk1 = np.copy(p)

        for i in range(thick, N-thick):
            for j in range(thick, N-thick):
                pk1[i][j] = (1 - w)*p[i][j] +  w*0.25*(pk1[i-1][j] +pk1[i][j-1] + p[i+1][j] + p[i][j+1] - b[i][j])

        self.values = pk1
        self._fillGhosts()


    def update_backward_SOR(self, b, w):

        assert(self.got_ghost_cells == True)
        p = self.values
        N = self.N
        thick = self.ghost_thick
        pk1 = np.copy(p)

        for i in range(N-thick-1, thick-1, -1):
            for j in range(thick, N-thick):
                pk1[i][j] = (1 - w)*p[i][j] +  w*0.25*(pk1[i+1][j] +pk1[i][j-1] + p[i-1][j] + p[i][j+1] - b[i][j])
        
        self.values = pk1
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

        return np.std(residu) / np.mean(residu)



class CounterFlowCombustion():

    def __init__(self, L:float, physN:float, L_slot:float, L_coflow:float, D:float, pho:float, max_t_comput:datetime.timedelta, show_save:bool, ell_crit:float, div_crit:float):

        self.L = L
        self.physN = physN
        self.L_slot = L_slot
        self.L_coflow = L_coflow
        self.D = D
        self.pho = pho
        self.max_t_comput = max_t_comput
        self.show_save = show_save
        self.ell_crit = ell_crit
        self.div_crit = div_crit

        self.dx = L/physN
        max_u = 1. # because the max inlet velocity is 1 m/s.

        # Choice of dt function of limits
        max_Fo = 0.25 # Fourier threshold in 2D
        max_CFL = 0.5 # CFL limit thresholf in 2D
        self.dt = 0.2*min(max_Fo*self.dx**2/D, max_CFL*self.dx/max_u)   # Time step
        #self.dt = 4e-6
        
        self.omega = 2*(1 - math.pi/physN - (math.pi/physN)**2)
        #self.omega = 1.5    # omega parameter evaluation of the SOR method

        # Create mesh grid
        #self.X, self.Y = np.meshgrid(np.linspace(0, L, physN, endpoint=True) , np.linspace(0, L, N, endpoint=True))
        #self.I, self.J = np.meshgrid(np.linspace(0, N, N, endpoint=True, dtype=int) , np.linspace(0, N, N, endpoint=True, dtype=int))

        self.ghost_thick = 2
        self.N = physN + 2*self.ghost_thick


    def get_beta(self, uvet:VelocityField):

        ##f function using 2nd-order Forward Difference 2nd order standard Laplacian.
        dx = self.dx
        u = uvet.u.values
        v = uvet.v.values

        forward_x =     0.5*(-3*u     + 4*np.roll(u, -1, axis=1)    - np.roll(u, -2, axis=1))/dx
        backward_x =    0.5*( 3*u     - 4*np.roll(u, 1, axis=1)     + np.roll(u, 2, axis=1))/dx

        forward_y =     0.5*(-3*v   + 4*np.roll(v, -1, axis=0) - np.roll(v, -2, axis=0))/dx
        backward_y =    0.5*( 3*v   - 4*np.roll(v, 1, axis=0)  + np.roll(v, 2, axis=0))/dx

        u_x = np.where(u < 0, forward_x, backward_x)

        v_y = np.where(v < 0, forward_y, backward_y)

        return (dx**2*self.pho/self.dt)*(u_x + v_y)


    def ell_solver(self, uvet:VelocityField, P:PressureField):

        # omega parameter evaluation of the SOR method
        omega = self.omega
        dx =self.dx
        dt = self.dt
        pho = self.pho

        # beta Laplcaian P converges to beta_arr
        beta_arr =self.get_beta(uvet)
    
        eps = 1.
        kiter = 1
        maxiter = 50
        while eps > self.ell_crit and kiter < maxiter:
            if kiter%1002==1000:
                print(f"\tElliptic Solver\ti={kiter}")
                P.plot(title=f"Elliptic solver\nIter={kiter}\nresidual err={eps:.3f}\n $\omega={omega}$", saveaspng=f"{kiter}_pressure_convergance.png", pause=2)
            # Realizes a backward and then a forward SOR to symmetrizes error.
            P.update_forward_SOR(beta_arr, omega)
            P.update_backward_SOR(beta_arr, omega)

            eps = P.residual_error(beta_arr)

            kiter += 1

        return P


    def _f(self, uval, vval,  phi):

        ##f function using 2nd-order Forward Difference 2nd order standard Laplacian.
        dx = self.dx

        # get the translation arrays to calculate further the derivatives estimates.
        # Translation along x axis, indexed by i.
        phi_ip1 = np.roll(phi, -1, axis=1)
        phi_ip2 = np.roll(phi, -2, axis=1)
        phi_im1 = np.roll(phi, 1, axis=1)
        phi_im2 = np.roll(phi, 2, axis=1)
        
        # Translation along y axis, indexed by j.
        phi_jp1 = np.roll(phi, -1, axis=0)
        phi_jp2 = np.roll(phi, -2, axis=0)
        phi_jm1 = np.roll(phi, 1, axis=0)
        phi_jm2 = np.roll(phi, 2, axis=0)

        forward_x =     0.5*(-3*phi     + 4*phi_ip1 - phi_ip2)/dx
        backward_x =    0.5*( 3*phi     - 4*phi_im1 + phi_im2)/dx

        forward_y =     0.5*(-3*phi     + 4*phi_jp1 - phi_jp2)/dx
        backward_y =    0.5*( 3*phi     - 4*phi_jm1 + phi_jm2)/dx

        phi_x = np.where(uval < 0, forward_x, backward_x)

        phi_y = np.where(vval < 0, forward_y, backward_y)

        # Calculate second derivatives
        phi_xx = (phi_ip1 - 2 * phi + phi_im1) / dx**2
        phi_yy = (phi_jp1 - 2 * phi + phi_jm1) / dx**2

        laplacian_phi = phi_xx + phi_yy
        
        return self.D*laplacian_phi - uval*phi_x - vval*phi_y
    

    def _update_RK2(self, uv:VelocityField, uvet:VelocityField):
        """
        Function to update the simulation at each step.
        Here follows the RK2 method.  
        """
        L = self.L
        N = self.N
        L_slot = self.L_slot
        L_coflow = self.L_coflow
        thick = uv.ghost_thick

        u0 = np.copy(uv.u.values)
        v0 = np.copy(uv.v.values)
        # Processes the intermediate coefficients of the RK method, as specified at the page 42 of the handout.
        uvet.u.values = u0 + 0.5*self.dt*self._f(u0, v0, u0)
        uvet.v.values = v0 + 0.5*self.dt*self._f(u0, v0, v0)
        uvet._fillGhosts()
 
        u12 = np.copy(uvet.u.values)
        v12 = np.copy(uvet.v.values)
        # Finally processes the phi at next step with the weight sum of defined k1 and k3 defined above.
        uvet.u.values = u0 + self.dt*self._f(u12, v12, u12)
        uvet.v.values = v0 + self.dt*self._f(u12, v12, v12)
        uvet._fillGhosts()


    def compute(self):
        
        N = self.N
        dt = self.dt
        pho = self.pho
        dx = self.dx
        L_slot = self.L_slot
        L_coflow = self.L_coflow
        thick = self.ghost_thick

        # Initialize Velocity field
        uv = VelocityField(np.zeros((self.physN, self.physN), dtype=float), np.zeros((self.physN, self.physN), dtype=float) , dx, L_slot, L_coflow, False, thick)
        uv.update_metric()
        
        # Initializing the V* field
        uvet = VelocityField(np.zeros((self.physN, self.physN), dtype=float), np.zeros((self.physN, self.physN), dtype=float) , dx, L_slot, L_coflow, False, thick)
        uvet.update_metric()

        # Initializing the P field
        Press = PressureField(np.full((self.physN, self.physN), 1.0), dx, False, thick)

        # Set time to zero
        time = 0
        
        # Set frame counting to zero
        frame = 0
        
        start_datetime = datetime.datetime.now()
        stop_datetime = datetime.datetime.now()


        while uv.metric >= 0.05 and uv.metric < self.div_crit and stop_datetime - start_datetime < self.max_t_comput:
            
            time = frame * dt
            
            if frame%50 == 0:
                print(f"Frame=\t{frame}\t;\tVirtual time={time}")
                if self.show_save:

                    uv.v.plot(True, title=f'V Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s \n Metric: {uv.metric:.5f}',saveaspng=str(frame)+"_v_field.png")
                    uv.u.plot(True, title=f'U Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s \n Metric: {uv.metric:.5f}',saveaspng=str(frame)+"_u_field.png")
                    Press.plot(True, title=f'Pressure Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(frame)+"_press_field.png")

                    stop_datetime = datetime.datetime.now()
            # uvet = self._update_RK2(uv)
            self._update_RK2(uv, uvet)
            uvet.update_metric()
            # uvet.norm.plot(title=f'V* norm Field k={frame} ($\Delta t=${self.dt}, N={N}) \n Time: {time:.6f} s \n Metric: {uvet.metric}',saveaspng=str(frame)+"_vetnorm_field.png")
            self.ell_solver(uvet, Press)
            #Press.plot(True, title=f'Pressure Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(frame)+"_press_field.png")
            uv.u.values = uvet.u.values - (dt/pho)*Press.derivative_x()
            uv.v.values = uvet.v.values - (dt/pho)*Press.derivative_y()
            # uv = VelocityField(up1_arr, vp1_arr, dx, L_slot, L_coflow, True, thick)
            uv._fillGhosts()
            uv.update_metric()
            #uv.norm.plot(title=f'V* norm Field k={frame} ($\Delta t=${self.dt}, N={N}) \n Time: {time:.6f} s \n Metric: {uv.metric}',saveaspng=str(frame)+"_vetnorm_field.png")
            
            frame += 1

        uv.v.plot(True, title=f'V Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s \n Metric: {uv.metric:.5f}',saveaspng=str(frame)+"_v_field.png")
        uv.u.plot(True, title=f'U Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s \n Metric: {uv.metric:.5f}',saveaspng=str(frame)+"_u_field.png")
        Press.plot(True, title=f'Pressure Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(frame)+"_press_field.png")

        if uv.metric >= self.div_crit :        
            print(f"Warning: The simulation stopped running because a divergence was detected (vel_metric >= {self.div_crit}).")
            print(f"\tParameters: (L, D, N, $\Delta t$)=({self.L}, {self.D}, {N}, {dt})")
            print("\tSimulation duration: "+str(stop_datetime - start_datetime))
            print(f"\tVirtual stop time: {time} s")   
            print(f"\tVirtual stop frame: {frame}")
            print(f"\tVelocity norm: {uv.metric:5f}")

        elif stop_datetime - start_datetime >= self.max_t_comput:
            print("Warning: The simulation stopped running because the max duration of simulation ("+str(self.max_t_comput)+") was reached.")
            print(f"\tParameters: (L, D, N, $\Delta t$)=({self.L}, {self.D}, {N}, {dt})")
            print("\tSimulation duration: "+str(stop_datetime - start_datetime))
            print(f"\tVirtual stop time: {time} s")
            print(f"\tVirtual stop frame: {frame}")
            print(f"\tVelocity norm: {uv.metric:5f}")

        else:
            print("Success: The simulation stopped running because the field was homogeneous enough (vel_metric < 0.05).")
            print(f"\tParameters: (L, D, N, $\Delta t$)=({self.L}, {self.D}, {N}, {dt})")
            print("\tSimulation duration: "+str(stop_datetime - start_datetime))
            print(f"\tVirtual stop time: {time} s")        
            print(f"\tVirtual stop frame: {frame}")
            print(f"\tVelocity norm: {uv.metric:5f}")




### MAIN    ###

def main():

    # Definition of all global variables:
    global fullpath, dirpath, epsilon

    epsilon = 1e-9 # valeur en-dessous de laquelle on considère avoir 0.

    #Chemin absolu du fichier .py qu'on execute
    fullpath = os.path.abspath(__file__)
    #Chemin absolu du dossier contenant le .py qu'on execute
    dirpath = os.path.dirname(fullpath)

    # Parameters of the problem
    L = 2e-3     # Length in the square shape domain.
    D = 15e-6   # Diffusion coefficient
    L_slot = 5e-4 # length of the inlet slot.
    L_coflow = 5e-4 # length of the inlet coflow.
    pho = 1.1614 # Fluid density.

    # Initial Parameters of the simulation
    N = 64    # Number of steps for each space axis


    # Put here the maximum time you want to spend on the computation.
    max_time_computation = datetime.timedelta(hours=1, minutes=0)
    # Show and register plots ?
    show_and_save = True

    # Stop threshold of elliptic solver
    ell_crit = 1e-2
    # Divergence stop cirterion
    div_crit = 100.

    mysimu = CounterFlowCombustion(L, N, L_slot, L_coflow, D, pho, max_time_computation, show_and_save, ell_crit, div_crit)
    mysimu.compute()
 

main()