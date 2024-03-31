# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
from field import Field
from velocity_field import VelocityField
from pressure_field import PressureField
import misc



#Chemin absolu du fichier .py qu'on execute
fullpath = os.path.abspath(__file__)
#Chemin absolu du dossier contenant le .py qu'on execute
dirpath = os.path.dirname(fullpath)



class FluidFlow():

    #__slots__ = "L", "physN", "L_slot", "L_coflow", "D", "rho", "max_t_comput", "show_save", "register_period", "ell_crit", "div_crit", "conv_crit", "dx", "dt", "omega", "y", "ghost_thick", "N"
    
    def __init__(self, L:float, physN:float, L_slot:float, L_coflow:float, nu:float, rho:float, max_t_comput:datetime.timedelta, show_save:bool, register_period:int, lists_max_size:int, ell_crit:float, div_crit:float, conv_crit:float):

        self.L = L
        self.physN = physN
        self.L_slot = L_slot
        self.L_coflow = L_coflow
        self.nu = nu
        self.rho = rho
        self.max_t_comput = max_t_comput
        self.show_save = show_save
        self.register_period = register_period
        self.lists_max_size = lists_max_size
        self.ell_crit = ell_crit
        self.div_crit = div_crit
        self.conv_crit = conv_crit

        self.dx = L/physN
        max_u = 1. # because the max inlet velocity is 1 m/s.

        # Choice of dt function of limits
        max_Fo = 0.25 # Fourier threshold in 2D
        max_CFL = 0.5 # CFL limit thresholf in 2D
        self.dt = 0.4*min(max_Fo*self.dx**2/nu, max_CFL*self.dx/max_u)   # Time step
        #self.dt = 4e-6
        
        self.omega = 2*(1 - math.pi/physN - (math.pi/physN)**2)
        #self.omega = 1.5    # omega parameter evaluation of the SOR method

        self.y = np.linspace(0, L, physN, endpoint=True, dtype=np.float32)
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

        return (dx**2*self.rho/self.dt)*(u_x + v_y)


    def ell_solver(self, uvet:VelocityField, P:PressureField):

        # omega parameter evaluation of the SOR method
        omega = self.omega
        dx =self.dx
        dt = self.dt
        rho = self.rho

        # beta Laplcaian P converges to beta_arr
        beta_arr =self.get_beta(uvet)
    
        eps = 1.
        kiter = 1
        maxiter = 999
        while eps > self.ell_crit and kiter < maxiter:
            if kiter%1002==1000:
                print(f"\tElliptic Solver\ti={kiter}")
                misc.plot_field(P, title=f"Elliptic solver\nIter={kiter}\nresidual err={eps:.3f}\n $\omega={omega}$", saveaspng=f"{kiter}_pressure_convergance.png", pause=2)
            # Realizes a backward and then a forward SOR to symmetrizes error.
            P.update_forward_SOR(beta_arr, omega)
            P.update_backward_SOR(beta_arr, omega)

            eps = P.residual_error(beta_arr)

            kiter += 1
        
        P.last_nb_iter = kiter-1

        return P


    def _f_velocities(self, uval, vval,  phi):

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

        phi_x = np.where(uval < 0, forward_x, backward_x) #### bon sens, test !

        phi_y = np.where(vval < 0, forward_y, backward_y) ####

        # Calculate second derivatives
        phi_xx = (phi_ip1 - 2 * phi + phi_im1) / dx**2
        phi_yy = (phi_jp1 - 2 * phi + phi_jm1) / dx**2

        laplacian_phi = phi_xx + phi_yy
        
        return self.nu*laplacian_phi - uval*phi_x - vval*phi_y
    

    def _update_RK2_velocity(self, uv:VelocityField, uvet:VelocityField):
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
        uvet.u.values = u0 + 0.5*self.dt*self._f_velocities(u0, v0, u0)
        uvet.v.values = v0 + 0.5*self.dt*self._f_velocities(u0, v0, v0)
        uvet._fillGhosts()
        #misc.plot_field(uvet.v, True, title=f'V* Field',saveaspng="_v-et00_field.png")
        #misc.plot_field(uvet.u, True, title=f'U* Field',saveaspng="_u-et00_field.png")            

 
        u12 = np.copy(uvet.u.values)
        v12 = np.copy(uvet.v.values)
        # Finally processes the phi at next step with the weight sum of defined k1 and k3 defined above.
        uvet.u.values = u0 + self.dt*self._f_velocities(u12, v12, u12)
        uvet.v.values = v0 + self.dt*self._f_velocities(u12, v12, v12)
        uvet._fillGhosts()
        #misc.plot_field(uvet.v, True, title=f'V* Field',saveaspng="_v-et01_field.png")
        #misc.plot_field(uvet.u, True, title=f'U* Field',saveaspng="_u-et01_field.png")            


    def print_write_end_message(self, code, div_crit, max_t_comput, conv_crit, L, nu, N, dt, duration_delta, time, frame, uv_consecutiv_deriv, max_strain_rate):
        
        assert(code in ["divergence", "timeout", "success"])
        
        first_line = ""
        
        if code == "divergence":
            first_line = f"Warning: The simulation stopped running because a divergence was detected (uv_consecutive_deriv >= {div_crit:.2e})."
        elif code == "timeout":
            first_line = "Warning: The simulation stopped running because the max duration of simulation ("+str(max_t_comput)+") was reached."
        elif code == "success":
            first_line = f"Success: The simulation stopped running because the velocity field was stable enough (uv_consecutive_deriv < {conv_crit:.2e})."

        parameters =   f"\tParameters: (L, nu, N, $\Delta t$)=({L}, {nu}, {N}, {dt})"
        simu_duration = "\tSimulation duration: "+str(duration_delta)
        vtime =        f"\tVirtual stop time: {time} s"
        vframe =       f"\tVirtual stop frame: {frame}"
        uv_difference =f"\tVelocity consecutive derivative: {uv_consecutiv_deriv:.2e}"
        max_srate =    f"\tMaximum strain rate on the left wall: {max_strain_rate} Hz"

        message = first_line + "\n" + parameters + "\n" + simu_duration + "\n" + vtime + "\n" + vframe + "\n" + uv_difference + "\n" + max_srate+ "\n"

        print(message)

        endfile = open(dirpath+"/outputs_program_ecoulement/simulation_report.txt", "w")
        endfile.write(message)


    def compute(self):
        
        # Bring attributes closer as variables.
        N = self.N
        physN = self.physN
        dt = self.dt
        rho = self.rho
        dx = self.dx
        L_slot = self.L_slot
        L_coflow = self.L_coflow
        thick = self.ghost_thick
        frame_period = self.register_period

        # Initialize Velocity field
        uv = VelocityField(np.zeros((physN, physN), dtype=float), np.zeros((physN, physN), dtype=float) , dx, L_slot, L_coflow, False, thick)
        uv.update_metric()

        # Initalize uvcopy
        uvcopy = VelocityField(np.zeros((physN, physN), dtype=float), np.zeros((physN, physN), dtype=float) , dx, L_slot, L_coflow, False, thick)
        uv_consecutiv_deriv = self.conv_crit + 1.
        cons_deriv_analysis = np.zeros((3,10), dtype=float)

        # Initializing the V* field
        uvet = VelocityField(np.zeros((physN, physN), dtype=float), np.zeros((physN, physN), dtype=float) , dx, L_slot, L_coflow, False, thick)
        uvet.update_metric()

        # Initializing the P field
        Press = PressureField(np.full((physN, physN), 1.0), dx, False, thick)

        # Set time to zero
        time = 0.
        
        # Set frame counting to zero
        frame = 0
        
        # Variables that will track the computation time.
        start_datetime = datetime.datetime.now()
        stop_datetime = datetime.datetime.now()

        while uv_consecutiv_deriv >= self.conv_crit and uv_consecutiv_deriv < self.div_crit and stop_datetime - start_datetime < self.max_t_comput:
            
            time = frame * dt

            if frame%frame_period == 0:
                
                print(f"Frame=\t{frame:06}\t ; \tVirtual time={time:.2e} s\t;\tLast SOR nb iter={Press.last_nb_iter}\t;\tVelocity Cons. Deriv.={uv_consecutiv_deriv:.2e}")
                stop_datetime = datetime.datetime.now()

                if self.show_save:

                    misc.plot_field(uv.v, True, title=f'V Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(frame)+"_v_field.png")
                    misc.plot_field(uv.u, True, title=f'U Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(frame)+"_u_field.png")
                    misc.plot_field(uvet.v, True, title=f'V* Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(frame)+"_v-et_field.png")
                    misc.plot_field(uvet.u, True, title=f'U* Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(frame)+"_u-et_field.png")
                    misc.plot_strain_rate(uv, self.y, title="Spatial representation of th strain rate along the slipping wall", saveaspng=str(frame)+"_strain_rate.png")

            if frame%frame_period >= 0 and frame%frame_period <= 9:
                # Copying the uv field at the start of the frame.
                uvcopy = misc.uv_copy(uv)

            # Fluid Flow
            self._update_RK2_velocity(uv, uvet)
            uvet.update_metric()
            #misc.plot_field(uvet.v, True, title=f'V* Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(frame)+"_v-et0_field.png")
            #misc.plot_field(uvet.u, True, title=f'U* Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(frame)+"_u-et0_field.png")            
            self.ell_solver(uvet, Press)
            #misc.plot_field(uvet.v, True, title=f'V* Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(frame)+"_v-et1_field.png")
            #misc.plot_field(uvet.u, True, title=f'U* Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(frame)+"_u-et1_field.png")            
            uv.u.values = uvet.u.values - (dt/rho)*Press.derivative_x()
            uv.v.values = uvet.v.values - (dt/rho)*Press.derivative_y()
            #misc.plot_field(uvet.v, True, title=f'V* Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(frame)+"_v-et2_field.png")
            #misc.plot_field(uvet.u, True, title=f'U* Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(frame)+"_u-et2_field.png")            
            uv._fillGhosts()
            uv.update_metric()

            frame += 1

            if frame%frame_period >= 1 and frame%frame_period <= 10:
                uv_consecutiv_deriv = misc.velocity_derivative_norm(uv, uvcopy, dt)
                cons_deriv_analysis = np.roll(cons_deriv_analysis, -1, axis=1)
                cons_deriv_analysis[0,-1] = frame
                cons_deriv_analysis[1,-1] = time
                cons_deriv_analysis[2, -1] = uv_consecutiv_deriv
            
            if frame%frame_period == 10:
                misc.plot_consecutive_deriv(cons_deriv_analysis, title=f"Convergence plot k from {frame-9} to {frame} \n ($\Delta t=${dt}, N={N})", saveaspng=str(frame)+"_convergence.png")

        # Plots of the final frame
        misc.plot_field(uv.v, True, title=f'V Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(frame)+"_v_field.png")
        misc.plot_field(uv.u, True, title=f'U Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(frame)+"_u_field.png")
        misc.plot_strain_rate(uv, self.y, title=f"Spatial representation of th strain rate along the slipping wall\nMax is {uv.max_strain_rate:.4e} Hz", saveaspng=str(frame)+"_strain_rate.png")
        misc.plot_field(Press, True, title=f'Pressure Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(frame)+"_press_field.png")

        # About to print and write in final file the report of the simulation.
        if uv_consecutiv_deriv >= self.div_crit :
            code = "divergence"        
        elif stop_datetime - start_datetime >= self.max_t_comput:
            code = "timeout"
        else:
            code = "success"

        self.print_write_end_message(code, self.div_crit, self.max_t_comput, self.conv_crit, self.L, self.nu, N, dt, stop_datetime-start_datetime, time, frame, uv_consecutiv_deriv, uv.max_strain_rate)
    