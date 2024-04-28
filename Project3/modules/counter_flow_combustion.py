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
from modules.simu_parameters import SimuParameters



class CounterFlowCombustion():
    
    def __init__(self, myparams:SimuParameters):

        self.prms = myparams

        self.y = np.linspace(0, myparams.L, myparams.physN, endpoint=True, dtype=np.float32)

        # I want the variable counting the nb of frames realized to be a class attribute.
        self.frame = 0 
        self.chem_frame = 0
        self.time = 0.
        self.start_datetime = datetime.datetime.now()
        self.stop_datetime = datetime.datetime.now()

        # Diagnostics variables that have to be initialized now.
        self.uv_consecutive_diff = 1000.
        self.temp_consecutive_diff = 1000.
        self.Temp_maxT_flame = None
        self.uv_max_strain_rate = None
        self.n2_diff_zone_thick = None

        # End interrupt key code. Initialized at "manual interrupt" because for a success or divergence end, the code is changed.
        self.endcode = "manual_interrupt"

        
    def get_beta(self, uvet:VelocityField):

        ##f function using 2nd-order Forward Difference 2nd order standard Laplacian.
        dx = self.prms.dx
        rho = self.prms.rho
        dt = self.prms.dt

        u = uvet.u.values
        v = uvet.v.values

        forward_x =     0.5*(-3*u     + 4*np.roll(u, -1, axis=1)    - np.roll(u, -2, axis=1))/dx
        backward_x =    0.5*( 3*u     - 4*np.roll(u, 1, axis=1)     + np.roll(u, 2, axis=1))/dx

        forward_y =     0.5*(-3*v   + 4*np.roll(v, -1, axis=0) - np.roll(v, -2, axis=0))/dx
        backward_y =    0.5*( 3*v   - 4*np.roll(v, 1, axis=0)  + np.roll(v, 2, axis=0))/dx

        u_x = np.where(u < 0, forward_x, backward_x)

        v_y = np.where(v < 0, forward_y, backward_y)

        return (dx**2*rho/dt)*(u_x + v_y)


    def ell_solver(self, uvet:VelocityField, P:PressureField):

        # omega parameter evaluation of the SOR method
        omega = self.prms.omega
        dx =self.prms.dx
        dt = self.prms.dt
        rho = self.prms.rho

        # beta Laplcaian P converges to beta_arr
        beta_arr =self.get_beta(uvet)
    
        eps = 1.
        kiter = 1
        warningiter = 1500
        while eps > self.prms.ell_crit and kiter < warningiter*3:
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
        dx = self.prms.dx
        nu = self.prms.nu

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
        
        return nu*laplacian_phi - uval*phi_x - vval*phi_y
    

    def _update_RK2_velocity(self, uv:VelocityField, uvet:VelocityField):
        """
        Function to update the simulation at each step.
        Here follows the RK2 method.  
        """
        dt = self.prms.dt

        u0 = np.copy(uv.u.values)
        v0 = np.copy(uv.v.values)
        # Processes the intermediate coefficients of the RK method, as specified at the page 42 of the handout.
        uvet.u.values = u0 + 0.5*dt*self._f_velocities(u0, v0, u0)
        uvet.v.values = v0 + 0.5*dt*self._f_velocities(u0, v0, v0)
        uvet._fillGhosts()
 
        u12 = np.copy(uvet.u.values)
        v12 = np.copy(uvet.v.values)
        # Finally processes the phi at next step with the weight sum of defined k1 and k3 defined above.
        uvet.u.values = u0 + dt*self._f_velocities(u12, v12, u12)
        uvet.v.values = v0 + dt*self._f_velocities(u12, v12, v12)
        uvet._fillGhosts()          


    def _f_species(self, uval, vval, phi):

        ##f function using 2nd-order Forward Difference 2nd order standard Laplacian.
        dx = self.prms.dx
        D = self.prms.D

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
        return D*laplacian_phi - uval*phi_x - vval*phi_y


    def _update_RK2_species(self, uv:VelocityField, species_field:Species):
        """
        Function to update the simulation at each step.
        Here follows the RK2 method.  
        """
        dt = self.prms.dt

        u_arr = uv.u.values
        v_arr = uv.v.values

        y0 = np.copy(species_field.values)
        # Processes the intermediate coefficients of the RK method, as specified at the page 42 of the handout.
        species_field.values = y0 + 0.5*dt*self._f_species(u_arr, v_arr, species_field.values)
        species_field.fillGhosts()

        species_field.values = y0 + dt*self._f_species(u_arr, v_arr, species_field.values)
        species_field.fillGhosts()


    def _f_temperature(self, uval, vval, phi):
        ##f function using 2nd-order Forward Difference 2nd order standard Laplacian.
        dx = self.prms.dx
        a = self.prms.a

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
        return a*laplacian_phi - uval*phi_x - vval*phi_y


    def _update_RK2_temperature(self, uv:VelocityField, temp_field:TemperatureField):
        """
        Function to update the simulation at each step.
        Here follows the RK2 method.  
        """
        dt = self.prms.dt

        u_arr = uv.u.values
        v_arr = uv.v.values

        T0 = np.copy(temp_field.values)
        # Processes the intermediate coefficients of the RK method, as specified at the page 42 of the handout.
        temp_field.values = T0 + 0.5*dt*self._f_temperature(u_arr, v_arr, temp_field.values)
        temp_field.fillGhosts()

        temp_field.values = T0 + dt*self._f_temperature(u_arr, v_arr, temp_field.values)
        temp_field.fillGhosts()


    def compute(self):
        
        # Bring attributes closer as variables.
        L = self.prms.L
        c_p = self.prms.c_p
        Ta = self.prms.Ta
        i_reactor = self.prms.i_reactor
        j_reactor = self.prms.j_reactor
        N = self.prms.N
        physN = self.prms.physN
        dt = self.prms.dt
        rho = self.prms.rho
        dx = self.prms.dx
        L_slot = self.prms.L_slot
        L_coflow = self.prms.L_coflow
        thick = self.prms.ghost_thick
        frame_period = self.prms.register_period

        # Initialize Velocity field
        uv = VelocityField(np.zeros((physN, physN), dtype=float), np.zeros((physN, physN), dtype=float) , dx, L_slot, L_coflow, False, thick)
        uv.update_metric()

        # Initalize uvcopy
        uvcopy = VelocityField(np.zeros((physN, physN), dtype=float), np.zeros((physN, physN), dtype=float) , dx, L_slot, L_coflow, False, thick)
        self.uv_consecutive_diff = 1000.

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
        self.temp_consecutive_diff = 1000.
        
        # Set time to zero
        self.time = 0.
        
        self.start_datetime = datetime.datetime.now()
        self.stop_datetime = datetime.datetime.now()

        # Settings
        hasIgnitionStarted = False
        already_saved_X_times = 0

        while self.uv_consecutive_diff >= self.prms.uv_conv_crit and self.temp_consecutive_diff >= self.prms.temp_conv_crit and uv.metric < self.prms.div_crit and self.stop_datetime - self.start_datetime < self.prms.max_t_comput:
            
            self.time = self.frame * dt

            isDisplayFrame = (self.frame%frame_period == 0)

            if isDisplayFrame:
                
                if self.frame != 0:
                    self.uv_consecutive_diff = misc.velocity_derivative_norm(uvcopy, uv, dt)
                    if hasIgnitionStarted:
                        self.temp_consecutive_diff = misc.temperature_derivative(TempCopy, Temp, dt)

                print(f"Frame=\t{self.frame:06}\t ; \tVirtual time={self.time:.2e} s\t;\tLast SOR nb iter={Press.last_nb_iter}\t;\tVelocity Cons. Diff.={self.uv_consecutive_diff:.2e}\t;\tT° Cons. Diff.={self.temp_consecutive_diff:.2e}")
                self.stop_datetime = datetime.datetime.now()

                if self.prms.show_save:
                    """misc.register_array_csv(f"{self.frame}_ch4.csv", ch4.values)
                    misc.plot_field(uv.v, True, title=f'V Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_v_field.png")
                    misc.plot_field(uv.u, True, title=f'U Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_u_field.png")
                    misc.plot_field(uvet.v, True, title=f'V* Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_v-et_field.png")
                    misc.plot_field(uvet.u, True, title=f'U* Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_u-et_field.png")
                    misc.plot_strain_rate(uv, self.y, title="Spatial representation of th strain rate along the slipping wall", saveaspng=str(self.frame)+"_strain_rate.png")
                    misc.plot_field(Press, True, title=f'Pressure Field k={self.frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(self.frame)+"_press_field.png")
                    misc.plot_field(o2, False, vmin=0.0, vmax=1.01, title=f'O2 Population k={self.frame} ($\Delta t=${dt:.3e}, N={N}) \n Time: {time:.3e} s', saveaspng=str(self.frame)+"_O2.png")
                    misc.plot_field(n2, False, vmin=0.0, vmax=1.01, title=f'N2 Population k={self.frame} ($\Delta t=${dt:.3e}, N={N}) \n Time: {time:.3e} s', saveaspng=str(self.frame)+"_N2.png")
                    misc.plot_field(ch4, False, vmin=0.0, vmax=1.01, title=f'CH4 Population k={self.frame} ($\Delta t=${dt:.3e}, N={N}) \n Time: {time:.3e} s', saveaspng=str(self.frame)+"_CH4.png")
                    misc.plot_field(h2o, False, vmin=0.0, vmax=1.01, title=f'H2O Population k={self.frame} ($\Delta t=${dt:.3e}, N={N}) \n Time: {time:.3e} s', saveaspng=str(self.frame)+"_H2O.png")
                    misc.plot_field(co2, False, vmin=0.0, vmax=1.01, title=f'CO2 Population k={self.frame} ($\Delta t=${dt:.3e}, N={N}) \n Time: {time:.3e} s', saveaspng=str(self.frame)+"_CO2.png")
                    misc.plot_field(Temp, False, title=f'T° Field k={self.frame} ($\Delta t=${dt:.3e}, N={N}) \n Time: {time:.3e} s',saveaspng=str(self.frame)+"_temp_field.png")
                    misc.plot_diffusive_zone(n2, self.y, self.frame, dt, time)"""
                    self.n2_diff_zone_thick = n2.diff_zone_thick 
                    misc.register_frame_hdf5(uv, Press, o2, n2, ch4, h2o, co2, Temp, self.frame, already_saved_X_times, f"{already_saved_X_times}_all_fields.h5")
                    already_saved_X_times += 1

            elif self.frame%frame_period == frame_period-1 :
                uvcopy = misc.uv_copy(uv)
                if hasIgnitionStarted:
                    TempCopy = misc.temp_copy(Temp)

       
            if self.time >= self.prms.time_ignite:
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


            if self.time >= self.prms.time_ignite:
                if not hasIgnitionStarted:
                    print(f"\tIgnition started at {self.time:.3e} s\t, frame {self.frame}.")
                    frame_igni = self.frame
                    chemistry_solver.chemistry_loop(o2, ch4, h2o, co2, Temp, L, self.frame, dt, rho, c_p, Ta, i_reactor, j_reactor, True)
                    hasIgnitionStarted = True
                else:
                    chemistry_solver.chemistry_loop(o2, ch4, h2o, co2, Temp, L, self.frame, dt, rho, c_p, Ta, i_reactor, j_reactor, False)

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
            self.ell_solver(uvet, Press)
            uv.u.values = uvet.u.values - (dt/rho)*Press.derivative_x()
            uv.v.values = uvet.v.values - (dt/rho)*Press.derivative_y()
            uv._fillGhosts()
            uv.update_metric()
            self.uv_max_strain_rate = uv.max_strain_rate
            
            self.frame += 1

        # Determines the maximum
        Temp.maxT_flame = np.max(Temp.values[thick:-thick , thick:-thick])
        self.Temp_maxT_flame = Temp.maxT_flame

        # Plots of the final frame
        misc.plot_field(o2, False, vmin=0.0, vmax=1.01, title=f'O2 Population k={self.frame} ($\Delta t=${dt:.3e}, N={N}) \n Time: {self.time:.3e} s', saveaspng=str(self.frame)+"_O2.png")
        misc.plot_field(ch4, False, vmin=0.0, vmax=1.01, title=f'CH4 Population k={self.frame} ($\Delta t=${dt:.3e}, N={N}) \n Time: {self.time:.3e} s', saveaspng=str(self.frame)+"_CH4.png")
        misc.plot_field(h2o, False, vmin=0.0, vmax=1.01, title=f'H2O Population k={self.frame} ($\Delta t=${dt:.3e}, N={N}) \n Time: {self.time:.3e} s', saveaspng=str(self.frame)+"_H2O.png")
        misc.plot_field(co2, False, vmin=0.0, vmax=1.01, title=f'CO2 Population k={self.frame} ($\Delta t=${dt:.3e}, N={N}) \n Time: {self.time:.3e} s', saveaspng=str(self.frame)+"_CO2.png")
        misc.plot_field(Temp, False, title=f'T° Field k={self.frame} ($\Delta t=${dt:.3e}, N={N}) \n Time: {self.time:.3e} s',saveaspng=str(self.frame)+"_temp_field.png")

        # About to print and write in final file the report of the simulation.
        if uv.metric >= self.prms.div_crit :
            self.endcode = "divergence"        
        elif self.stop_datetime - self.start_datetime >= self.prms.max_t_comput:
            self.endcode = "timeout"
        elif self.uv_consecutive_diff < self.prms.uv_conv_crit:
            self.endcode = "success_velocity"
        elif self.temp_consecutive_diff < self.prms.temp_conv_crit:
            self.endcode = "success_temp"


"""    def end_message(self):

        div_crit = self.prms.div_crit
        uv_conv_crit = self.prms.uv_conv_crit
        temp_conv_crit = self.prms.temp_conv_crit
        L = self.prms.L
        D = self.prms.D
        N = self.prms.N
        dt = self.prms.dt

        stop_datetime = self.stop_datetime
        start_datetime = self.start_datetime
        time = self.time
        frame = self.frame
        uv_consecutive_diff = self.uv_consecutive_diff
        temp_consecutive_diff = self.temp_consecutive_diff
        uv_max_strain_rate = self.uv_max_strain_rate
        n2_diff_zone_thick = self.n2_diff_zone_thick
        Temp_maxT_flame = self.Temp_maxT_flame
        endcode = self.endcode
        max_t_comput = self.max_t_comput

        misc.print_write_end_message(endcode, div_crit, max_t_comput, uv_conv_crit, temp_conv_crit, L, D, N, dt, stop_datetime - start_datetime, time, frame, uv_consecutive_diff, temp_consecutive_diff, uv_max_strain_rate, n2_diff_zone_thick, Temp_maxT_flame)

"""