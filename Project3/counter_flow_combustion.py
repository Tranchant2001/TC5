# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import datetime
import math
from field import Field
from velocity_field import VelocityField
from pressure_field import PressureField
from species import Species, Dioxygen, Dinitrogen, Methane, Water, CarbonDioxide
from temperature_field import TemperatureField
import misc



class CounterFlowCombustion():

    #__slots__ = "L", "physN", "L_slot", "L_coflow", "D", "pho", "max_t_comput", "show_save", "register_period", "ell_crit", "div_crit", "conv_crit", "dx", "dt", "omega", "y", "ghost_thick", "N"
    
    def __init__(self, L:float, physN:float, L_slot:float, L_coflow:float, nu:float, D:float, a:float, pho:float, Ta:float, max_t_comput:datetime.timedelta, show_save:bool, register_period:int, ell_crit:float, div_crit:float, conv_crit:float):

        self.L = L
        self.physN = physN
        self.L_slot = L_slot
        self.L_coflow = L_coflow
        self.nu = nu
        self.D = D
        self.a = a
        self.pho = pho
        self.Ta = Ta
        self.max_t_comput = max_t_comput
        self.show_save = show_save
        self.register_period = register_period
        self.ell_crit = ell_crit
        self.div_crit = div_crit
        self.conv_crit = conv_crit

        self.dx = L/physN
        max_u = 1. # because the max inlet velocity is 1 m/s.

        # Choice of dt function of limits
        max_Fo = 0.25 # Fourier threshold in 2D
        max_CFL = 0.5 # CFL limit thresholf in 2D
        self.dt = 0.2*min(max_Fo*self.dx**2/D, max_CFL*self.dx/max_u)   # Time step
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
        maxiter = 100
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

        phi_x = np.where(uval < 0, forward_x, backward_x)

        phi_y = np.where(vval < 0, forward_y, backward_y)

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
 
        u12 = np.copy(uvet.u.values)
        v12 = np.copy(uvet.v.values)
        # Finally processes the phi at next step with the weight sum of defined k1 and k3 defined above.
        uvet.u.values = u0 + self.dt*self._f_velocities(u12, v12, u12)
        uvet.v.values = v0 + self.dt*self._f_velocities(u12, v12, v12)
        uvet._fillGhosts()


    def _f_species(self, uval, vval, phi, rr_arr):

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
        
        #return self.D*laplacian_phi - uval*phi_x - vval*phi_y + rr_arr/self.pho
        return self.D*laplacian_phi - uval*phi_x - vval*phi_y


    def _update_RK2_species(self, uv:VelocityField, species_field:Species):
        """
        Function to update the simulation at each step.
        Here follows the RK2 method.  
        """

        u_arr = uv.u.values
        v_arr = uv.v.values

        y0 = np.copy(species_field.values)
        # Processes the intermediate coefficients of the RK method, as specified at the page 42 of the handout.
        #Q = self.progress_rate(ch4_field, o2_field, T_field)
        #species_field.update_reaction_rate(Q)
        species_field.values = y0 + 0.5*self.dt*self._f_species(u_arr, v_arr, species_field.values, species_field.reaction_rate)
        species_field.fillGhosts()

        species_field.values = y0 + self.dt*self._f_species(u_arr, v_arr, species_field.values, species_field.reaction_rate)
        species_field.fillGhosts()


    def progress_rate(self, ch4_field:Methane, o2_field:Dioxygen, T_field:TemperatureField):
        
        ch4_field.update_concentration()
        o2_field.update_concentration()
        ch4_conc = ch4_field.concentration
        o2_conc = o2_field.concentration

        Q = self.A*o2_conc*ch4_conc*np.exp(-self.Ta/T_field)
        
        return Q
    

    def compute(self):
        
        # Bring attributes closer as variables.
        N = self.N
        physN = self.physN
        dt = self.dt
        pho = self.pho
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
        uv_consecutive_diff = 1.0

        # Initializing the V* field
        uvet = VelocityField(np.zeros((physN, physN), dtype=float), np.zeros((physN, physN), dtype=float) , dx, L_slot, L_coflow, False, thick)
        uvet.update_metric()

        # Initializing the P field
        Press = PressureField(np.full((physN, physN), 1.0), dx, False, thick)

        # Initializing the species
        o2 = Dioxygen(np.full((physN, physN), 0.2, dtype=float), dx, L_slot, L_coflow, pho, False, thick)
        n2 = Dinitrogen(np.full((physN, physN), 0.8 ,dtype=float), dx, L_slot, L_coflow, pho, False, thick)
        ch4 = Methane(np.zeros((physN, physN), dtype=float), dx, L_slot, L_coflow, pho, False, thick)
        h2o = Water(np.zeros((physN, physN), dtype=float), dx, L_slot, L_coflow, pho, False, thick)
        co2 = CarbonDioxide(np.zeros((physN, physN), dtype=float), dx, L_slot, L_coflow, pho, False, thick)

        # Initializing the temperature
        #Temp = TemperatureField(np.full((physN, physN), 298.15), dx, False, thick)

        # Set time to zero
        time = 0.
        
        # Set frame counting to zero
        frame = 0
        
        start_datetime = datetime.datetime.now()
        stop_datetime = datetime.datetime.now()


        while uv_consecutive_diff >= self.conv_crit and uv.metric < self.div_crit and stop_datetime - start_datetime < self.max_t_comput:
            
            time = frame * dt

            if frame%frame_period == 0:
                
                if frame != 0:
                    uv_consecutive_diff = misc.velocity_residual(uvcopy, uv)

                print(f"Frame=\t{frame:06}\t ; \tVirtual time={time:.2e} s\t;\tLast SOR nb iter={Press.last_nb_iter}\t;\tVelocity Residual={uv_consecutive_diff:.2e}")
                stop_datetime = datetime.datetime.now()

                if self.show_save:

                    misc.plot_field(uv.v, True, title=f'V Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(frame)+"_v_field.png")
                    misc.plot_field(uv.u, True, title=f'U Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(frame)+"_u_field.png")
                    misc.plot_strain_rate(uv, self.y, title="Spatial representation of th strain rate along the slipping wall", saveaspng=str(frame)+"_strain_rate.png")
                    misc.plot_field(Press, True, title=f'Pressure Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(frame)+"_press_field.png")
                    misc.plot_field(o2, True, title=f'O2 Population k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(frame)+"_O2.png")
                    misc.plot_field(n2, True, title=f'N2 Population k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(frame)+"_N2.png")
                    misc.plot_field(ch4, True, title=f'CH4 Population k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(frame)+"_CH4.png")
                    misc.plot_field(h2o, True, title=f'H2O Population k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(frame)+"_H2O.png")
                    misc.plot_field(co2, True, title=f'CO2 Population k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(frame)+"_CO2.png")
                    misc.plot_diffusive_zone(n2, self.y, frame, dt, time)

            elif frame%frame_period == frame_period-1 :
                uvcopy = misc.uv_copy(uv)

            # Temperature transport

            # Population transport
            self._update_RK2_species(uv, o2)
            self._update_RK2_species(uv, n2)
            self._update_RK2_species(uv, ch4)
            self._update_RK2_species(uv, h2o)
            self._update_RK2_species(uv, co2)

            # Fluid Flow
            self._update_RK2_velocity(uv, uvet)
            uvet.update_metric()
            self.ell_solver(uvet, Press)
            uv.u.values = uvet.u.values - (dt/pho)*Press.derivative_x()
            uv.v.values = uvet.v.values - (dt/pho)*Press.derivative_y()
            uv._fillGhosts()
            uv.update_metric()
            
            frame += 1

        # Plots of the final frame
        misc.plot_field(uv.v, True, title=f'V Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s \n Metric: {uv.metric:.5f}',saveaspng=str(frame)+"_v_field.png")
        misc.plot_field(uv.u, True, title=f'U Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s \n Metric: {uv.metric:.5f}',saveaspng=str(frame)+"_u_field.png")
        misc.plot_strain_rate(uv, self.y, title=f"Spatial representation of th strain rate along the slipping wall\nMax is {uv.max_strain_rate:.2e} Hz", saveaspng=str(frame)+"_strain_rate.png")
        misc.plot_field(Press, True, title=f'Pressure Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(frame)+"_press_field.png")
        misc.plot_field(o2, True, title=f'O2 Population k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(frame+"_O2.png"))
        misc.plot_field(n2, True, title=f'N2 Population k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(frame+"_N2.png"))
        misc.plot_field(ch4, True, title=f'CH4 Population k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(frame+"_CH4.png"))
        misc.plot_field(h2o, True, title=f'H2O Population k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(frame+"_H2O.png"))
        misc.plot_field(co2, True, title=f'CO2 Population k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(frame+"_CO2.png"))
        misc.plot_diffusive_zone(n2, self.y, frame, dt, time)

        if uv.metric >= self.div_crit :        
            print(f"Warning: The simulation stopped running because a divergence was detected (vel_metric >= {self.div_crit}).")
            print(f"\tParameters: (L, D, N, $\Delta t$)=({self.L}, {self.D}, {N}, {dt})")
            print("\tSimulation duration: "+str(stop_datetime - start_datetime))
            print(f"\tVirtual stop time: {time} s")   
            print(f"\tVirtual stop frame: {frame}")
            print(f"\tVelocity consecutive difference: {uv_consecutive_diff:.2e}")

        elif stop_datetime - start_datetime >= self.max_t_comput:
            print("Warning: The simulation stopped running because the max duration of simulation ("+str(self.max_t_comput)+") was reached.")
            print(f"\tParameters: (L, D, N, $\Delta t$)=({self.L}, {self.D}, {N}, {dt})")
            print("\tSimulation duration: "+str(stop_datetime - start_datetime))
            print(f"\tVirtual stop time: {time} s")
            print(f"\tVirtual stop frame: {frame}")
            print(f"\tVelocity consecutive difference: {uv_consecutive_diff:.2e}")

        else:
            print(f"Success: The simulation stopped running because the velocity field was stable enough (uv_consecutive_difference < {self.conv_crit:.2e}).")
            print(f"\tParameters: (L, D, N, $\Delta t$)=({self.L}, {self.D}, {N}, {dt})")
            print("\tSimulation duration: "+str(stop_datetime - start_datetime))
            print(f"\tVirtual stop time: {time} s")        
            print(f"\tVirtual stop frame: {frame}")
            print(f"\tVelocity consecutive difference: {uv_consecutive_diff:.2e}")

