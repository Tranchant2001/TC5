# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime
import math

from modules.field import Field
from modules.velocity_field import VelocityField
from modules.pressure_field import PressureField
from modules.species import Species, Dioxygen, Dinitrogen, Methane, Water, CarbonDioxide
from modules.temperature_field import TemperatureField
import modules.misc as misc
import modules.chemistry_solver as chemistry_solver



class CounterFlowCombustion():

    #__slots__ = "L", "physN", "L_slot", "L_coflow", "D", "rho", "max_t_comput", "show_save", "register_period", "ell_crit", "div_crit", "conv_crit", "dx", "dt", "omega", "y", "ghost_thick", "N"
    
    def __init__(self, L:float, physN:float, L_slot:float, L_coflow:float, nu:float, D:float, a:float, rho:float, c_p:float, Ta:float, time_ignite:float, max_t_comput:datetime.timedelta, show_save:bool, register_period:int, i_reactor:int, j_reactor:int, ell_crit:float, div_crit:float, uv_conv_crit:float, temp_conv_crit:float):

        self.L = L
        self.physN = physN
        self.L_slot = L_slot
        self.L_coflow = L_coflow
        self.nu = nu
        self.D = D
        self.a = a
        self.rho = rho
        self.c_p = c_p
        self.Ta = Ta
        self.time_ignite = time_ignite
        self.max_t_comput = max_t_comput
        self.show_save = show_save
        self.register_period = register_period
        self.ell_crit = ell_crit
        self.div_crit = div_crit
        self.uv_conv_crit = uv_conv_crit
        self.temp_conv_crit = temp_conv_crit

        self.dx = L/(physN-1)
        max_u = 1. # because the max inlet velocity is 1 m/s.

        # Choice of dt function of limits
        max_Fo = 0.25 # Fourier threshold in 2D
        max_CFL = 0.5 # CFL limit thresholf in 2D
        self.dt = 0.4*min(max_Fo*self.dx**2/D, max_CFL*self.dx/max_u)   # Time step
        self.dtchem_list = self.set_dtchem_list(self.dt)
        #self.dt = 4e-6
        
        self.omega = 2*(1 - math.pi/physN - (math.pi/physN)**2)
        #self.omega = 1.5    # omega parameter evaluation of the SOR method

        self.y = np.linspace(0, L, physN, endpoint=True, dtype=np.float32)
        # Create mesh grid
        #self.X, self.Y = np.meshgrid(np.linspace(0, L, physN, endpoint=True) , np.linspace(0, L, N, endpoint=True))
        #self.I, self.J = np.meshgrid(np.linspace(0, N, N, endpoint=True, dtype=int) , np.linspace(0, N, N, endpoint=True, dtype=int))

        self.ghost_thick = 2
        self.N = physN + 2*self.ghost_thick
        self.i_reactor = i_reactor + self.ghost_thick 
        self.j_reactor = j_reactor + self.ghost_thick

        # I want the variable counting the nb of frames realized to be a class attribute.
        self.frame = 0 
        self.chem_frame = 0     

        
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
        warningiter = 1500
        while eps > self.ell_crit and kiter < warningiter*3:
            if kiter%warningiter == warningiter-1:
                print(f"\tWarning: Elliptic Solver has reached more than {kiter}\t iterations !")
                misc.plot_field(P, title=f"Elliptic solver\nIter={kiter}\nresidual err={eps:.3e}\n $\omega={omega}$", saveaspng=f"{kiter}_pressure_convergance.png", pause=2)
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


    def _f_species(self, uval, vval, phi):

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
        
        #return self.D*laplacian_phi - uval*phi_x - vval*phi_y + rr_arr/self.rho
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
        species_field.values = y0 + 0.5*self.dt*self._f_species(u_arr, v_arr, species_field.values)
        species_field.fillGhosts()

        species_field.values = y0 + self.dt*self._f_species(u_arr, v_arr, species_field.values)
        species_field.fillGhosts()


    def _f_temperature(self, uval, vval, phi):
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
        
        #return self.D*laplacian_phi - uval*phi_x - vval*phi_y + rr_arr/self.rho
        return self.a*laplacian_phi - uval*phi_x - vval*phi_y


    def _update_RK2_temperature(self, uv:VelocityField, temp_field:TemperatureField):
        """
        Function to update the simulation at each step.
        Here follows the RK2 method.  
        """

        u_arr = uv.u.values
        v_arr = uv.v.values

        T0 = np.copy(temp_field.values)
        # Processes the intermediate coefficients of the RK method, as specified at the page 42 of the handout.
        temp_field.values = T0 + 0.5*self.dt*self._f_temperature(u_arr, v_arr, temp_field.values)
        temp_field.fillGhosts()

        temp_field.values = T0 + self.dt*self._f_temperature(u_arr, v_arr, temp_field.values)
        temp_field.fillGhosts()


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
        uv_consecutive_diff = 1000.

        # Initializing the V* field
        uvet = VelocityField(np.zeros((physN, physN), dtype=float), np.zeros((physN, physN), dtype=float) , dx, L_slot, L_coflow, False, thick)
        uvet.update_metric()

        # Initializing the P field
        Press = PressureField(np.full((physN, physN), 1.0), dx, False, thick)

        # Initializing the species
        o2 = Dioxygen(np.full((physN, physN), 0.2, dtype=float), dx, L_slot, L_coflow, rho, False, thick)
        n2 = Dinitrogen(np.full((physN, physN), 0.8 ,dtype=float), dx, L_slot, L_coflow, rho, False, thick)
        ch4 = Methane(np.zeros((physN, physN), dtype=float), dx, L_slot, L_coflow, rho, False, thick)
        h2o = Water(np.zeros((physN, physN), dtype=float), dx, L_slot, L_coflow, rho, False, thick)
        co2 = CarbonDioxide(np.zeros((physN, physN), dtype=float), dx, L_slot, L_coflow, rho, False, thick)

        # Initializing the temperature
        Temp = TemperatureField(np.full((physN, physN), 300), dx, False, thick)
        
        # Initalize TempCopy
        TempCopy = TemperatureField(np.full((physN, physN), 300), dx, False, thick)
        temp_consecutive_diff = 1000.
        
        # Set time to zero
        time = 0.
        
        start_datetime = datetime.datetime.now()
        stop_datetime = datetime.datetime.now()

        # Settings
        hasIgnitionStarted = False

        while uv_consecutive_diff >= self.uv_conv_crit and temp_consecutive_diff >= self.temp_conv_crit and uv.metric < self.div_crit and stop_datetime - start_datetime < self.max_t_comput:
            
            time = self.frame * dt

            isDisplayFrame = (self.frame%frame_period == 0)

            if isDisplayFrame:
                
                if self.frame != 0:
                    uv_consecutive_diff = misc.velocity_derivative_norm(uvcopy, uv, dt)
                    if hasIgnitionStarted:
                        temp_consecutive_diff = misc.temperature_derivative(TempCopy, Temp, dt)

                print(f"Frame=\t{self.frame:06}\t ; \tVirtual time={time:.2e} s\t;\tLast SOR nb iter={Press.last_nb_iter}\t;\tVelocity Cons. Diff.={uv_consecutive_diff:.2e}\t;\tT° Cons. Diff.={temp_consecutive_diff:.2e}")
                stop_datetime = datetime.datetime.now()

                if self.show_save:
                    #misc.register_array_csv(f"{self.frame}_ch4.csv", ch4.values)
                    #misc.plot_field(uv.v, True, title=f'V Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_v_field.png")
                    #misc.plot_field(uv.u, True, title=f'U Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_u_field.png")
                    #misc.plot_field(uvet.v, True, title=f'V* Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_v-et_field.png")
                    #misc.plot_field(uvet.u, True, title=f'U* Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_u-et_field.png")
                    #misc.plot_strain_rate(uv, self.y, title="Spatial representation of th strain rate along the slipping wall", saveaspng=str(self.frame)+"_strain_rate.png")
                    #misc.plot_field(Press, True, title=f'Pressure Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_press_field.png")
                    misc.plot_field(o2, False, vmin=0.0, vmax=1.01, title=f'O2 Population k={self.frame} ($\Delta t=${dt:.3e}, N={N}) \n Time: {time:.3e} s', saveaspng=str(self.frame)+"_O2.png")
                    #misc.plot_field(n2, False, vmin=0.0, vmax=1.01, title=f'N2 Population k={self.frame} ($\Delta t=${dt:.3e}, N={N}) \n Time: {time:.3e} s', saveaspng=str(self.frame)+"_N2.png")
                    misc.plot_field(ch4, False, vmin=0.0, vmax=1.01, title=f'CH4 Population k={self.frame} ($\Delta t=${dt:.3e}, N={N}) \n Time: {time:.3e} s', saveaspng=str(self.frame)+"_CH4.png")
                    misc.plot_field(h2o, False, vmin=0.0, vmax=1.01, title=f'H2O Population k={self.frame} ($\Delta t=${dt:.3e}, N={N}) \n Time: {time:.3e} s', saveaspng=str(self.frame)+"_H2O.png")
                    misc.plot_field(co2, False, vmin=0.0, vmax=1.01, title=f'CO2 Population k={self.frame} ($\Delta t=${dt:.3e}, N={N}) \n Time: {time:.3e} s', saveaspng=str(self.frame)+"_CO2.png")
                    misc.plot_field(Temp, False, title=f'T° Field k={self.frame} ($\Delta t=${dt:.3e}, N={N}) \n Time: {time:.3e} s',saveaspng=str(self.frame)+"_temp_field.png")
                    #misc.plot_diffusive_zone(n2, self.y, self.frame, dt, time)

            elif self.frame%frame_period == frame_period-1 :
                uvcopy = misc.uv_copy(uv)
                if hasIgnitionStarted:
                    TempCopy = misc.temp_copy(Temp)

       
            if time >= self.time_ignite:
                # This function ignites the middle of the box to 1000 K or more.
                Temp.ignite_or_not()


            # Temperature transport
            self._update_RK2_temperature(uv, Temp)
                
            # Population transport
            self._update_RK2_species(uv, o2)
            #self._update_RK2_species(uv, n2)
            self._update_RK2_species(uv, ch4)
            self._update_RK2_species(uv, h2o)
            self._update_RK2_species(uv, co2)

            #if hasIgnitionStarted:
            #    misc.plot_field(ch4, False, title=f'CH4 Population just before chemistry k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(self.frame)+"_CH4_beforeChem.png")


            if time >= self.time_ignite:
                if not hasIgnitionStarted:
                    print(f"\tIgnition started at {time:.3e} s\t, frame {self.frame}.")
                    frame_igni = self.frame
                    chemistry_solver.chemistry_loop(o2, ch4, h2o, co2, Temp, self.L, self.frame, dt, rho, self.c_p, self.Ta, self.i_reactor, self.j_reactor, True)
                    hasIgnitionStarted = True
                else:
                    chemistry_solver.chemistry_loop(o2, ch4, h2o, co2, Temp, self.L, self.frame, dt, rho, self.c_p, self.Ta, self.i_reactor, self.j_reactor, False)

                #misc.plot_field(o2, False, title=f'O2 Population k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(self.frame)+"_O2.png")
                #misc.plot_field(ch4, False, title=f'CH4 Population k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(self.frame)+"_CH4.png")
                #misc.plot_field(h2o, False, title=f'H2O Population k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(self.frame)+"_H2O.png")
                #misc.plot_field(co2, False, title=f'CO2 Population k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(self.frame)+"_CO2.png")
                #misc.plot_field(Temp, False, title=f'T° Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_temp_field.png")                



            o2.fillGhosts()
            n2.fillGhosts()
            ch4.fillGhosts()
            h2o.fillGhosts()
            co2.fillGhosts()
            Temp.fillGhosts()


            #misc.register_array_csv(f"{self.frame}_ch4_endofframe.csv", ch4.values)

            #sys.exit()

            # Fluid Flow
            self._update_RK2_velocity(uv, uvet)
            uvet.update_metric()
            #misc.plot_field(uvet.v, True, title=f'V* Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_v-et0_field.png")
            #misc.plot_field(uvet.u, True, title=f'U* Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_u-et0_field.png")            
            self.ell_solver(uvet, Press)
            #misc.plot_field(uvet.v, True, title=f'V* Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_v-et1_field.png")
            #misc.plot_field(uvet.u, True, title=f'U* Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_u-et1_field.png")            
            uv.u.values = uvet.u.values - (dt/rho)*Press.derivative_x()
            uv.v.values = uvet.v.values - (dt/rho)*Press.derivative_y()
            #misc.plot_field(uvet.v, True, title=f'V* Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_v-et2_field.png")
            #misc.plot_field(uvet.u, True, title=f'U* Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_u-et2_field.png")            
            uv._fillGhosts()
            uv.update_metric()
            
            self.frame += 1

        # Determines the maximum
        Temp.maxT_flame = np.max(Temp.values[thick:-thick , thick:-thick])

        # Plots of the final frame
        #misc.plot_field(uv.v, True, title=f'V Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s \n Metric: {uv.metric:.5f}',saveaspng=str(self.frame)+"_v_field.png")
        #misc.plot_field(uv.u, True, title=f'U Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s \n Metric: {uv.metric:.5f}',saveaspng=str(self.frame)+"_u_field.png")
        #misc.plot_strain_rate(uv, self.y, title=f"Spatial representation of th strain rate along the slipping wall\nMax is {uv.max_strain_rate:.4e} Hz", saveaspng=str(self.frame)+"_strain_rate.png")
        #misc.plot_field(Press, True, title=f'Pressure Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_press_field.png")
        misc.plot_field(o2, False, vmin=0.0, vmax=1.01, title=f'O2 Population k={self.frame} ($\Delta t=${dt:.3e}, N={N}) \n Time: {time:.3e} s', saveaspng=str(self.frame)+"_O2.png")
        #misc.plot_field(n2, True, title=f'N2 Population k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(self.frame)+"_N2.png")        
        misc.plot_field(ch4, False, vmin=0.0, vmax=1.01, title=f'CH4 Population k={self.frame} ($\Delta t=${dt:.3e}, N={N}) \n Time: {time:.3e} s', saveaspng=str(self.frame)+"_CH4.png")
        misc.plot_field(h2o, False, vmin=0.0, vmax=1.01, title=f'H2O Population k={self.frame} ($\Delta t=${dt:.3e}, N={N}) \n Time: {time:.3e} s', saveaspng=str(self.frame)+"_H2O.png")
        misc.plot_field(co2, False, vmin=0.0, vmax=1.01, title=f'CO2 Population k={self.frame} ($\Delta t=${dt:.3e}, N={N}) \n Time: {time:.3e} s', saveaspng=str(self.frame)+"_CO2.png")
        misc.plot_field(Temp, False, title=f'T° Field k={self.frame} ($\Delta t=${dt:.3e}, N={N}) \n Time: {time:.3e} s',saveaspng=str(self.frame)+"_temp_field.png")

        #misc.plot_diffusive_zone(n2, self.y, self.frame, dt, time)

        # About to print and write in final file the report of the simulation.
        if uv.metric >= self.div_crit :
            code = "divergence"        
        elif stop_datetime - start_datetime >= self.max_t_comput:
            code = "timeout"
        elif uv_consecutive_diff < self.uv_conv_crit:
            code = "success_velocity"
        else:
            code = "success_temp"

        misc.print_write_end_message(code, self.div_crit, self.max_t_comput, self.uv_conv_crit, self.temp_conv_crit, self.L, self.D, N, dt, stop_datetime-start_datetime, time, self.frame, uv_consecutive_diff, temp_consecutive_diff, uv.max_strain_rate, n2.diff_zone_thick, Temp.maxT_flame)

