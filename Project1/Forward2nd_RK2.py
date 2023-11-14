# -*- coding: utf-8 -*-
#%%

"""
Projet n°1 du TC5: Numerical Methods du Master PPF.
Travail en binôme:
    Jules Chaveyriat
    Martin Guillon

Créé le 13/11/2023.
Mis à jour le 13/11/2023.
v1.1

DESCRIPTION:
Version with the explicit method of RK3-Heun.

"""
### PACKAGES    ###

import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import cmath
from numba import njit
import warnings


### FONCTIONS   ###


class FluidFlow():

    def __init__(self, L, N, D, dt, showAndSave, max_t_comput):
        self.L = L
        self.N = N
        self.D = D
        self.max_t_comput = max_t_comput
        self.showAndSave = showAndSave

        self.ghost_thick = 2
        self.dx = self.L/self.N
        self.dt = dt
        self.nu = self.dt/self.dx

        self.register_period = int(15/(self.dt*100))

        self.x_array = np.linspace(0, self.L, self.N, endpoint=False)
        self.y_array = np.linspace(0, self.L, self.N, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x_array, self.y_array)
        self.X = self._addGhosts(self.X)
        self.X = self._fillGhosts(self.X, True)
        self.Y = self._addGhosts(self.Y)
        self.Y = self._fillGhosts(self.Y, True)
        
        self.U0, self.V0 = self._set_velocity_field()
        self.U0 = self._fillGhosts(self.U0, True)
        self.V0 = self._fillGhosts(self.V0, True)

        self.Phi0 = self._set_initial_potential()
        self.Phi0 = self._fillGhosts(self.Phi0, True)
        
        self.tFinal = -1.0


    def _addGhosts(self, x):
        thick = self.ghost_thick
        x_wide = np.full((x.shape[0]+2*thick, x.shape[1]+2*thick), -1.0, dtype=float)
        x_wide[thick:-thick, thick:-thick] = x
        return x_wide


    def _fillGhosts(self, x, periodic=True):
        thick = self.ghost_thick
        if periodic:
            x[:, :thick] = x[:, -2*thick:-thick]
            x[:, -thick:] = x[:, thick:2*thick]
            x[:thick, :] = x[-2*thick:-thick, :]
            x[-thick:, :] = x[thick:2*thick, :]

            # filling dead corners with -1.0
            x[:thick, :thick] = np.full((thick, thick), -1.0, dtype=float)
            x[:thick, -thick:] = np.full((thick, thick), -1.0, dtype=float)
            x[-thick:, :thick] = np.full((thick, thick), -1.0, dtype=float)
            x[-thick:, -thick:] = np.full((thick, thick), -1.0, dtype=float)

        return x
    

    def _stripGhosts(self, x):

        thick = self.ghost_thick
        x_stripped = np.empty((x.shape[0]-2*thick, x.shape[1]-2*thick), dtype=float)
        x_stripped = x[thick:-thick, thick:-thick]

        return x_stripped


    def _get_lambda(self):
        # the maximum amplification factor mofulus is assumed to be achieved for the maximum values of U and V.
        umax = np.max(self.U0)
        vmax = np.max(self.V0)

        kxdx = np.linspace(0, 2*np.pi, 128, endpoint=False)
        kydx = np.linspace(0, 2*np.pi, 128, endpoint=False)
        Kx, Ky = np.meshgrid(kxdx, kydx)

        c_kx = np.cos(Kx)
        s_kx = np.sin(Kx)
        c_ky = np.cos(Ky)
        s_ky = np.sin(Ky)

        real_part = 1/(self.dx)*(1.5*(umax + vmax) - 4*self.D/self.dx + self.D*(c_kx + c_ky)/self.dx + umax*(c_kx**2 - 0.5 -2*c_kx) + vmax*(c_ky**2 - 0.5 -2*c_ky))
        imag_part = 1/(self.dx)*(umax*(c_kx*s_kx - 2*s_kx) + vmax*(c_ky*s_ky - 2*s_ky))

        #lambda_arr = np.empty(Kx.shape, dtype=np.complex_)

        lambda_arr = real_part + 1j*imag_part

        return lambda_arr


    def amplification_factor(self):

        lambda_arr = self._get_lambda()
        self.sigma = 0.5*(self.dt*lambda_arr)**2 + self.dt*lambda_arr + 1

        sigma_mod = np.absolute(self.sigma)

        sigma_mod = 0.5*sigma_mod

        sigma_maxmod = np.max(sigma_mod)

        if sigma_maxmod > 1.:
            print(f"\nWarning:\n\tThe amplification modulus is strictly greater (={sigma_maxmod:.3f}) than 1 and your simulation may diverge.")
        else:
            print(f"\nSuccess:\n\tThe amplification modulus is lower (={sigma_maxmod:.3f}) than 1 and your simulation may be stable.")

        # Plot the scalar field
        fig, ax = plt.subplots()
        ax.set_xlabel('$k_x \Delta x$')
        ax.set_ylabel('$k_y \Delta x$')
        ax.set_title(f'Modulus of amplification factor for Lax-Wendroff + RK2 scheme\n($\Delta t=${self.dt}, N={self.N})')
        image = ax.imshow(sigma_mod, extent=(0, 2*np.pi, 0, 2*np.pi), origin='lower', cmap='viridis')
        fig.colorbar(image)
        fig.savefig(dirpath+"/outputs_Forward2ndRK2/amplfication_factor_map.png", dpi=108, bbox_inches="tight")
        plt.show()
        

    def _set_velocity_field(self):
        
        u = np.cos((4 * np.pi * self.X) / self.L) * np.sin((4 * np.pi * self.Y) / self.L)
        v = -np.sin((4 * np.pi * self.X) / self.L) * np.cos((4 * np.pi * self.Y) / self.L)
        
        return u, v


    def _set_initial_potential(self):
        
        phi = np.where(np.sqrt((self.X - 0.5)**2 + (self.Y - 0.5)**2) < 0.3, 1.0, 0.0)
        
        return phi


        
    def _f(self, phi):

        ##f function using 2nd-order Forward Difference 2nd order standard Laplacian.
        

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

        forward_x = 0.5*(-3*phi + 4*phi_ip1 - phi_ip2)/self.dx
        backward_x = 0.5*(3*phi + -4*phi_im1 + phi_im2)/self.dx

        forward_y = 0.5*(-3*phi + 4*phi_jp1 - phi_jp2)/self.dx
        backward_y = 0.5*(3*phi + -4*phi_jm1 + phi_jm2)/self.dx

        phi_x = np.where(self.U0 < 0, forward_x, backward_x)

        phi_y = np.where(self.V0 < 0, forward_y, backward_y)

        # Calculate second derivatives
        phi_xx = (phi_ip1 - 2 * phi + phi_im1) / self.dx**2
        phi_yy = (phi_jp1 - 2 * phi + phi_jm1) / self.dx**2

        laplacian_phi = phi_xx + phi_yy
        
        return self.D*laplacian_phi - self.U0*phi_x - self.V0*phi_y
    

    """
    def _f(self, phi):

        ##f function using 2nd-order Centered Difference 2nd order standard Laplacian.
        

        # get the translation arrays to calculate further the derivatives estimates.
        # Translation along x axis, indexed by i.
        phi_ip1 = np.roll(phi, -1, axis=1)
        phi_im1 = np.roll(phi, 1, axis=1)
        
        # Translation along y axis, indexed by j.
        phi_jp1 = np.roll(phi, -1, axis=0)
        phi_jm1 = np.roll(phi, 1, axis=0)

        phi_x = (phi_ip1 - phi_im1)/self.dx

        phi_y = (phi_jp1 - phi_jm1)/self.dx

        # Calculate second derivatives
        phi_xx = (phi_ip1 - 2 * phi + phi_im1) / self.dx**2
        phi_yy = (phi_jp1 - 2 * phi + phi_jm1) / self.dx**2

        laplacian_phi = phi_xx + phi_yy
        
        return self.D*laplacian_phi - self.U0*phi_x - self.V0*phi_y
    """

    def _update_RK2(self, phi):
        """
        Function to update the simulation at each step.
        Here follows the RK2 method.  
        """   

        # Processes the intermediate coefficients of the RK method, as specified at the page 42 of the handout.
        k1 = self._f(phi)
        k1 = self._fillGhosts(k1, True)
        k2 = self._f(phi + 0.5*self.dt*k1)
        k2 = self._fillGhosts(k2, True)
        
        # Finally processes the phi at next step with the weight sum of defined k1 and k3 defined above.
        phi_plus_1 = phi + self.dt*k2
        phi_plus_1 = self._fillGhosts(phi_plus_1, True)
        
        return phi_plus_1


    def _get_metric(self, phi, stripped=False):
        """
        Calculate standard deviation and average of phi
        """
        phi_s = np.copy(phi)
        if not stripped:
            phi_s = self._stripGhosts(phi_s)
        std_phi = np.std(phi_s)
        avg_phi = np.mean(phi_s)

        # Calculate the metric
        metric = std_phi / avg_phi

        return metric


    def _register_scalar_field(self, phi, stripped=False, **kwargs):
        """
        Plot the state of the scalar field.
        Specifies in the title the time and the metric
        """
        phi_s = np.copy(phi)
        if not stripped:
            phi_s = self._stripGhosts(phi_s)

        # Create a figure and axis for the animation
        fig, ax = plt.subplots()
        
        # Plot the scalar field
        ax.clear()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if "title" in kwargs.keys():
            ax.set_title(kwargs.get('title', None))
        scale_plot = max(np.max(phi_s), 1.0)
        image = ax.imshow(phi_s, extent=(0, self.L, 0, self.L), vmax=scale_plot, origin='lower', cmap='viridis')
        fig.colorbar(image)

        if 'saveaspng' in kwargs.keys():
            plt.savefig(dirpath+"/outputs_Forward2ndRK2/"+kwargs.get('saveaspng', None), dpi=108, bbox_inches="tight")
        plt.pause(1)
        plt.close(fig)


    def compute(self):
        """
        Function taking the parameter of the problem as time and space smpling csts N and dt.
        Takes the initial scalar field phi.
        Takes T_comput, the maximum reel time you want this simulation to run. Default is set to 7 days.
        Processes frame by frame the deduced field with the scheme used in the above function update(phi, N, dt)
        Plots one frame per 100.
        Returns the time it took to reach metric < 0.05
        """

        # Initialisation of the variable metric.
        metric = 1.
        
        Phi = np.copy(self.Phi0)
        
        # Set time to zero
        time = 0
        
        # Set frame counting to zero
        frame = 0
        
        start_datetime = datetime.datetime.now()
        step_datetime = datetime.datetime.now()
        
        while metric >= 0.05 and metric < 3 and step_datetime - start_datetime < self.max_t_comput:

            time = frame * self.dt
            
            metric = self._get_metric(Phi, False)
            
            if frame%self.register_period == 0:
                print(frame)
                if self.showAndSave:
                    self._register_scalar_field(Phi, False, title=f'Diffusion of Scalar Field ($\Delta t=${self.dt}, N={self.N})\nt={time:.3f} and n={frame}\nMetric: {metric:.3f}',saveaspng=str(frame)+"_phi_field.png")

                
            Phi = self._update_RK2(Phi)
            step_datetime = datetime.datetime.now()
            frame += 1

        if metric >= 3 :        
            print("Warning: The simulation stopped running because a divergence was detected (metric >= 3).")
            print(f"\tParameters: (L, D, N, $\Delta t$)=({self.L}, {self.D}, {self.N}, {self.dt})")
            print("\tSimulation duration: "+str(step_datetime - start_datetime))
            print(f"\tVirtual stop time: {time:.2f} s")        
            print(f"\tVirtual stop frame: {frame}")
            print(f"\tMetric: {metric:5f}")

        elif step_datetime - start_datetime >= self.max_t_comput:
            print("Warning: The simulation stopped running because the max duration of simulation ("+str(self.max_t_comput)+") was reached.")
            print(f"\tParameters: (L, D, N, $\Delta t$)=({self.L}, {self.D}, {self.N}, {self.dt})")
            print("\tSimulation duration: "+str(step_datetime - start_datetime))
            print(f"\tVirtual stop time: {time:.2f} s")
            print(f"\tVirtual stop frame: {frame}")
            print(f"\tMetric: {metric:5f}")

        else:
            print("Success: The simulation stopped running because the field was homogeneous enough (metric < 0.05).")
            print(f"\tParameters: (L, D, N, $\Delta t$)=({self.L}, {self.D}, {self.N}, {self.dt})")
            print("\tSimulation duration: "+str(step_datetime - start_datetime))
            print(f"\tVirtual stop time: {time:.2f} s")        
            print(f"\tVirtual stop frame: {frame}")
            print(f"\tMetric: {metric:5f}")

            self.tFinal = time





### MAIN   ###



def main():

    global fullpath, dirpath
    #Chemin absolu du fichier .py qu'on execute
    fullpath = os.path.abspath(__file__)
    #Chemin absolu du dossier contenant le .py qu'on execute
    dirpath = os.path.dirname(fullpath)

    # Parameters of the problem
    L = 1.0     # Length of the (square shaped) domain (m)
    D = 0.001   # Diffusion coefficient
    # Initial Parameters of the simulation
    N = 512    # Number of steps for each space axis
    # Time resolution
    dt = 3e-4
    # Display and save parameters
    show_and_save = True
    max_time_computation = datetime.timedelta(hours=6)

    mysimu = FluidFlow(L, N, D, dt, show_and_save, max_time_computation)
    #mysimu.amplification_factor()
    mysimu.compute()

main()