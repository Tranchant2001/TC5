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
from numba import njit



### GENERAL FUNCTIONS   ###

class Field():

    def __init__(self, array, dx, got_ghost_cells=False, ghost_thick=0):
        
        self.values = array
        self.dx = dx
        self.shape = array.shape
        self.got_ghost_cells = got_ghost_cells
        self.ghost_thick = ghost_thick
        
    
    @njit
    def _addGhosts(self, thick:int):

        if not self.got_ghost_cells:
            x_wide = np.full((self.shape[0]+2*thick, self.shape[1]+2*thick), -1.0, dtype=float)
            x_wide[thick:-thick, thick:-thick] = self.values
            self.values = x_wide

            self.shape = self.values.shape
            self.got_ghost_cells = True
            self.ghost_thick = thick


    @njit
    def _stripGhosts(self):

        if self.got_ghost_cells:
            
            thick = self.ghost_thick
            self.values = self.values[thick:-thick, thick:-thick]

            self.shape = self.values.shape
            self.got_ghost_cells = False
            self.ghost_thick = 0


    @njit
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
        
        assert(self.got_ghost_cells)

        d_x = (0.5/self.dx)*(np.roll(self.values, -1, axis=1) - np.roll(self.values, 1, axis=1))

        return d_x
    

    def derivative_y(self):
        
        assert(self.got_ghost_cells)

        d_y = (0.5/self.dx)*(np.roll(self.values, -1, axis=0) - np.roll(self.values, 1, axis=0))

        return d_y


    def plot(self, **kwargs):
        """
        Plot the state of the scalar field.
        Specifies in the title the time and the metric
        """
        if self.got_ghost_cells:
            thick = self.ghost_thick
            phi_copy = self.values[thick : thick , thick : thick]
        else: 
            phi_copy = self.values

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

    def __init__(self, u_array, v_array, dx, L_slot, L_coflow, got_ghost_cells=False, ghost_thick=0):

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
        self.norm = Field(np.empty((self.simulation.N , self.simulation.N)), self.L, self.N)
        self.metric = 1.
        # self.update_metric()


    @njit
    def _addGhosts(self, thick):

        if not self.got_ghost_cells:
            self.u._addGhosts(thick)
            self.v._addGhosts(thick)

            self.shape = self.u.shape
            self.N = self.shape[0]
            
            self.got_ghost_cells = True
            self.ghost_thick = thick


    @njit
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


    @njit
    def _fillGhosts(self):

        if self.got_ghost_cells:    
            thick = self.ghost_thick
            u_new = self.u.values
            v_new = self.v.values
            N = self.N
            dx = self.dx
            L_slot = self.L_slot
            L_coflow = self.L_coflow

            # inlet and walls conditions for u
            u_new[:,0:thick] = np.zeros((N, thick), dtype=float)
            u_new[0:thick,:] = np.zeros((thick, N), dtype=float)
            u_new[N-thick:,:] = np.zeros((thick, N), dtype=float)

            # outlet condition
            u_new[: , N-thick:] = u_new[: , N-2*thick : N-thick]
            v_new[: , N-thick:] = v_new[: , N-2*thick : N-thick]

            # inlet conditions for v
            N_inlet = int(L_slot/dx)
            N_coflow = int(L_coflow/dx)

            v_new[:thick, thick : thick + N_inlet] = np.full((thick, N_inlet), 1.0)
            v_new[N-thick:, thick : thick + N_inlet] = np.full((thick, N_inlet), -1.0)

            v_new[:thick, thick + N_inlet : thick + N_inlet + N_coflow] = np.full((thick, N_coflow), 0.2)
            v_new[N-thick:, thick + N_inlet : thick + N_inlet + N_coflow] = np.full((thick, N_coflow), -0.2)

            v_new[:thick, thick + N_inlet + N_coflow:] = 0.
            v_new[N-thick:, thick + N_inlet + N_coflow:] =  0.

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
            u_new[:thick, :thick] = np.full((thick, thick), -1.0, dtype=float)
            u_new[:thick, -thick:] = np.full((thick, thick), -1.0, dtype=float)
            u_new[-thick:, :thick] = np.full((thick, thick), -1.0, dtype=float)
            u_new[-thick:, -thick:] = np.full((thick, thick), -1.0, dtype=float)
            
            v_new[:thick, :thick] = np.full((thick, thick), -1.0, dtype=float)
            v_new[:thick, -thick:] = np.full((thick, thick), -1.0, dtype=float)
            v_new[-thick:, :thick] = np.full((thick, thick), -1.0, dtype=float)
            v_new[-thick:, -thick:] = np.full((thick, thick), -1.0, dtype=float)   

            self.u.values = u_new
            self.v.values = v_new   


    @njit
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


    def plot(self, X, Y, **kwargs):

        if self.got_ghost_cells:
            N = self.N
            thick = self.ghost_thick
            u = self.u.values[thick : N-thick, thick : N-thick]
            v = self.v.values[thick : N-thick, thick : N-thick]
        
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

    def __init__(self, L, N, array=None, got_ghost_cells=False, ghost_thick=0):

        if array == None:
            array = np.full((N, N), 1.0)

        super().__init__(array, L, N, got_ghost_cells, ghost_thick)




    def _fillGhosts(self):

        if self.got_ghost_cells:    
            thick = self.ghost_thick
            N = self.N
            P_new = np.copy(self.values)

            P_new[:,N-thick] = np.full((N, thick), 1.0)
            
            for i in range(thick):

                P_new[:,i] = P_new[:,thick]
                P_new[i,:] = P_new[thick,:]
                P_new[N-i,:] = P_new[N-thick,:]

            # Dead corner values filled with -1.0
            P_new[:thick, :thick] = np.full((thick, thick), -1.0, dtype=float)
            P_new[:thick, -thick:] = np.full((thick, thick), -1.0, dtype=float)
            P_new[-thick:, :thick] = np.full((thick, thick), -1.0, dtype=float)
            P_new[-thick:, -thick:] = np.full((thick, thick), -1.0, dtype=float)                


    @njit
    def update_forward_SOR(self, b, w):

        assert(self.got_ghost_cells == True)
        p = self.values
        N = self.N
        thick = self.ghost_thick

        pk1 = np.zeros((N, N), dtype=float)

        for i in range(thick, N-thick):
            for j in range(thick, N-thick):
                pk1[i][j] = (1 - w)*p[i][j] +  w*0.25*(pk1[i-1][j] +pk1[i][j-1] + p[i+1][j] + p[i][j+1] - b[i][j])

        pk1 = pk1[thick : N-thick , thick : N-thick]

        Pfield_new = PressureField(self.L, N, pk1)

        return Pfield_new


    @njit
    def update_backward_SOR(self, b, w):

        assert(self.got_ghost_cells == True)
        p = self.values
        N = self.N
        thick = self.ghost_thick
        pk1 = np.zeros((N, N), dtype=float)

        for i in range(N-thick-1, thick-1, -1):
            for j in range(N-thick-1, thick-1, -1):
                pk1[i][j] = (1 - w)*p[i][j] +  w*0.25*(pk1[i+1][j] +pk1[i][j-1] + p[i-1][j] + p[i][j+1] - b[i][j])
        
        pk1 = pk1[thick : N-thick , thick : N-thick]

        Pfield_new = PressureField(self.L, N, pk1)

        return Pfield_new
    

    def residual_error(self, b):

        assert(self.got_ghost_cells == True)
        N = self.N
        p = self.values
        thick = self.ghost_thick

        p_xx = np.roll(p, 1, axis=1) - 2*p + np.roll(p, -1, axis=1)
        p_yy = np.roll(p, 1, axis=0) - 2*p + np.roll(p, -1, axis=0)
        Ap = p_xx + p_yy
        residu = b - Ap

        # there may be problems at the edges so they are not considered to process the residual norm
        residu = residu[thick : N-thick, thick : N-thick]

        return np.std(residu) / np.mean(residu)



class CounterFlowCombustion():

    def __init__(self, L, N, L_slot, L_coflow, D, pho, max_t_comput, show_save, ell_crit, div_crit):

        self.L = L
        self.N = N
        self.L_slot = L_slot
        self.L_coflow = L_coflow
        self.D = D
        self.pho = pho
        self.max_t_comput = max_t_comput
        self.show_save = show_save
        self.ell_crit = ell_crit
        self.div_crit = div_crit

        self.dx = self.L/self.N
        max_u = 1. # because the max inlet velocity is 1 m/s.

        # Choice of dt function of limits
        max_Fo = 0.25 # Fourier threshold in 2D
        max_CFL = 0.5 # CFL limit thresholf in 2D
        self.dt = 0.5*min(max_Fo*self.dx**2/D, max_CFL*self.dx/max_u)   # Time step

        self.omega = 1.5    # omega parameter evaluation of the SOR method

        # Create mesh grid
        self.X, self.Y = np.meshgrid(np.linspace(0, L, N, endpoint=True) , np.linspace(0, L, N, endpoint=True))
        #self.I, self.J = np.meshgrid(np.linspace(0, N, N, endpoint=True, dtype=int) , np.linspace(0, N, N, endpoint=True, dtype=int))

        self.ghost_thick = 2


    def ell_solver(self, uvet:VelocityField):

        # omega parameter evaluation of the SOR method
        #omega = 2*(1 - math.pi/N - (math.pi/N)**2)
        omega = self.omega
        dx =self.dx
        dt = self.dt
        pho = self.pho
        
        # Initial Uz field
        P = PressureField(self.L, self.N)

        # beta Laplcaian P converges to beta_arr
        beta_arr = (dx**2*pho/dt)*(uvet.u.derivative_x() + uvet.v.derivative_y())
    
        eps = 1.
        kiter = 1
        maxiter = 100000
        while eps > self.ell_crit and kiter < maxiter:
            if kiter%1000==1:
                P.plot(title=f"Elliptic solver\nIter={kiter}\nresidual err={eps:.3f}\n $\omega={omega}$", saveaspng=f"{kiter}_pressure_convergance.png", pause=2)
            # Realizes a backward and then a forward SOR to symmetrizes error.
            Pp1 = P.update_forward_SOR(beta_arr, omega)
            Pp1 = Pp1.update_backward_SOR(beta_arr, omega)
            #Pp1_b = update_backward_SOR(P, beta_arr, omega)
            #Pp1 = (Pp1_f + Pp1_b)/2
            #Pp1 = update_jacobi(P, beta_arr)

            eps = Pp1.residual_error(beta_arr)

            P = Pp1
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

        forward_x = 0.5*(-3*phi + 4*phi_ip1 - phi_ip2)/dx
        backward_x = 0.5*(3*phi + -4*phi_im1 + phi_im2)/dx

        forward_y = 0.5*(-3*phi + 4*phi_jp1 - phi_jp2)/dx
        backward_y = 0.5*(3*phi + -4*phi_jm1 + phi_jm2)/dx

        phi_x = np.where(uval < 0, forward_x, backward_x)

        phi_y = np.where(vval < 0, forward_y, backward_y)

        # Calculate second derivatives
        phi_xx = (phi_ip1 - 2 * phi + phi_im1) / dx**2
        phi_yy = (phi_jp1 - 2 * phi + phi_jm1) / dx**2

        laplacian_phi = phi_xx + phi_yy
        
        return self.D*laplacian_phi - uval*phi_x - vval*phi_y
    

    def _update_RK2(self, uv:VelocityField):
        """
        Function to update the simulation at each step.
        Here follows the RK2 method.  
        """
        L = self.L
        N = self.N
        L_slot = self.L_slot
        L_coflow = self.L_coflow
        thick = uv.ghost_thick

        ucopy = np.copy(uv.u.values)
        vcopy = np.copy(uv.v.values)
        # Processes the intermediate coefficients of the RK method, as specified at the page 42 of the handout.
        demi_u_arr = uv.u.values + 0.5*self.dt*self._f(uv.u.values, uv.v.values, ucopy)
        demi_v_arr = uv.v.values + 0.5*self.dt*self._f(uv.u.values, uv.v.values, vcopy)
        uv_new = VelocityField(demi_u_arr , demi_v_arr, self, True, thick)

        ucopy = np.copy(uv_new.u.values)
        vcopy = np.copy(uv_new.v.values)  
        # Finally processes the phi at next step with the weight sum of defined k1 and k3 defined above.
        u1_arr = uv.u.values + self.dt*self._f(uv_new.u.values, uv_new.v.values, ucopy)
        v1_arr = uv.v.values + self.dt*self._f(uv_new.u.values, uv_new.v.values, vcopy)
        uv_new = VelocityField(demi_u_arr , demi_v_arr, self, True, thick)
        
        return uv_new


    def compute(self):
        
        N = self.N
        dt = self.dt
        pho = self.pho

        # Initialize Velocity field
        uv = VelocityField(np.zeros((N, N), dtype=float), np.zeros((N, N), dtype=float) , self, False, self.ghost_thick)
        uv.update_metric()

        # Set time to zero
        time = 0
        
        # Set frame counting to zero
        frame = 0
        
        start_datetime = datetime.datetime.now()
        stop_datetime = datetime.datetime.now()


        while uv.metric >= 0.05 and uv.metric < self.div_crit and stop_datetime - start_datetime < self.max_t_comput:
            
            time = frame * dt
            
            if frame%1 == 0:
                print(frame)
                if self.show_save:

                    uv.norm.plot(title=f'V norm Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s \n Metric: {uv.metric:.5f}',saveaspng=str(frame)+"_vnorm_field.png")
                    #plot_vector_field(u, v, X, Y, frame, time, vel_metric, saveaspng=str(frame)+"_velocity_field.png")
                    """             
                    ax1.clear()
                    ax1.set_title(f'Vector Field k={frame}\n Time: {time:.3f} s \n Metric: {vel_metric:.5f}')
                    image = ax1.imshow(norm_arr, extent=(0, L, 0, L), origin='lower', cmap='viridis')
                    fig1.colorbar(image, ax=ax1)
                    """
                    stop_datetime = datetime.datetime.now()
            uvet = self._update_RK2(uv)
            uvet.update_metric()
            uvet.norm.plot(title=f'V* norm Field k={frame} ($\Delta t=${self.dt}, N={N}) \n Time: {time:.6f} s \n Metric: {uvet.metric}',saveaspng=str(frame)+"_vetnorm_field.png")
            Press = self.ell_solver(uvet)
            Press.plot(title=f'Pressure Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(frame)+"_press_field.png")
            up1_arr = uvet.u.values - (dt/pho)*Press.derivative_x()
            vp1_arr = uvet.v.values - (dt/pho)*Press.derivative_y()
            uv = VelocityField(up1_arr, vp1_arr, self, True, self.ghost_thick)
            uv.update_metric()
            
            frame += 1



        if uv.metric >= self.div_crit :        
            print(f"Warning: The simulation stopped running because a divergence was detected (vel_metric >= {self.div_crit}).")
            print(f"\tParameters: (L, D, N, $\Delta t$)=({self.L}, {self.D}, {N}, {dt})")
            print("\tSimulation duration: "+str(stop_datetime - start_datetime))
            print(f"\tVirtual stop time: {time} s")   
            print(f"\tVirtual stop frame: {frame}")
            print(f"\tVelocity norm: {uv.metric:5f}")

        elif stop_datetime - start_datetime >= self.max_t_comput:
            print("Warning: The simulation stopped running because the max duration of simulation ("+str(self.max_t_comput)+") was reached.")
            print(f"\tParameters: (L, D, N, $\Delta t$)=({self.L}, {self.D}, {N}, {delta_t})")
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
    max_time_computation = datetime.timedelta(hours=0, minutes=2)
    # Show and register plots ?
    show_and_save = True

    # Stop threshold of elliptic solver
    ell_crit = 1e-3
    # Divergence stop cirterion
    div_crit = 10

    mysimu = CounterFlowCombustion(L, N, L_slot, L_coflow, D, pho, max_time_computation, show_and_save, ell_crit, div_crit)
    mysimu.compute()
 

main()