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

    #__slots__ = "L", "physN", "L_slot", "L_coflow", "D", "rho", "max_t_comput", "show_save", "register_period", "ell_crit", "div_crit", "conv_crit", "dx", "dt", "omega", "y", "ghost_thick", "N"
    
    def __init__(self, L:float, physN:float, L_slot:float, L_coflow:float, nu:float, D:float, a:float, rho:float, c_p:float, Ta:float, time_ignite:float, max_t_comput:datetime.timedelta, show_save:bool, register_period:int, ell_crit:float, div_crit:float, conv_crit:float):

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
        self.conv_crit = conv_crit

        self.dx = L/physN
        max_u = 1. # because the max inlet velocity is 1 m/s.

        # Choice of dt function of limits
        max_Fo = 0.25 # Fourier threshold in 2D
        max_CFL = 0.5 # CFL limit thresholf in 2D
        self.dt = 0.2*min(max_Fo*self.dx**2/D, max_CFL*self.dx/max_u)   # Time step
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


    def set_dtchem_list(self, dt):
        
        l0 = np.linspace(0, 1, 20, endpoint=False)
        l1 = np.flip(1 - l0)
        l2 = l1*l1*l1*l1
        l3 = l2*dt
        l4 = np.zeros(20, dtype=float)
        l4[0] = l3[0]
        for k in range(1, 20):
            l4[k] = l3[k] - l3[k-1]

        fig1 = plt.figure()
        plt.plot(l3, label="Time piled up")
        plt.plot(l4, label="$\Delta t$")
        plt.xlabel("Index")
        plt.ylabel("Time s")
        plt.legend()
        plt.show()

        return l4


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
        species_field.values = y0 + 0.5*self.dt*self._f_species(u_arr, v_arr, species_field.values, species_field.reaction_rate)
        species_field.fillGhosts()

        species_field.values = y0 + self.dt*self._f_species(u_arr, v_arr, species_field.values, species_field.reaction_rate)
        species_field.fillGhosts()


    def _converge_all_species(self, o2:Dioxygen, n2:Dinitrogen, ch4:Methane, h2o:Water, co2:CarbonDioxide, T_field:TemperatureField, dtchem:float): #, conv_crit_chem):

        rho = self.rho
        c_p = self.c_p
        #thick = self.ghost_thick

        #all_converg = np.full(5, 1.)
        #while np.sum(np.where(all_converg > conv_crit_chem, False, True)) < 5:
        #o20 = np.copy(o2.values)
        #n20 = np.copy(n20.values)
        #ch40 = np.copy(ch40.values)
        #h2o0 = np.copy(h2o0.values)
        #co20 = np.copy(co20.values)

        Q = self.progress_rate(ch4, o2, T_field)

        o2.values = o2.values + (dtchem*o2.stoech*o2.W/rho)*Q
        n2.values = n2.values + (dtchem*n2.stoech*n2.W/rho)*Q
        ch4.values = ch4.values + (dtchem*ch4.stoech*ch4.W/rho)*Q
        h2o.values = h2o.values + (dtchem*h2o.stoech*h2o.W/rho)*Q
        co2.values = co2.values + (dtchem*co2.stoech*co2.W/rho)*Q

        omega_T = -(o2.Dhf*o2.stoech + n2.Dhf*n2.stoech + ch4.Dhf*ch4.stoech + h2o.Dhf*h2o.stoech + co2.Dhf*co2.stoech)*Q
        T_field.values = T_field.values + (dtchem/(rho*c_p))*omega_T

        #all_converg[0] = misc.array_residual(o20, thick, o2.values, thick)
        #all_converg[1] = misc.array_residual(n20, thick, n2.values, thick)
        #all_converg[2] = misc.array_residual(ch40, thick, ch4.values, thick)
        #all_converg[3] = misc.array_residual(h2o0, thick, h2o.values, thick)
        #all_converg[4] = misc.array_residual(co20, thick, co2.values, thick)


    def progress_rate(self, ch4_field:Methane, o2_field:Dioxygen, T_field:TemperatureField):
        
        N = self.N
        thick = self.ghost_thick

        ch4_field.update_concentration()
        o2_field.update_concentration()
        ch4_conc = ch4_field.concentration
        o2_conc = o2_field.concentration

        T_arr = T_field.values[thick:-thick, thick:-thick]
        exp_array = np.zeros((N,N), dtype=float)
        exp_array[thick:-thick, thick:-thick] = np.where(T_arr >= 100, np.exp(-self.Ta/T_arr), 0.) # Caution to counter from RuntimeWarning: overflow encountered in exp
        Q = 1.1e8*o2_conc*ch4_conc*exp_array
        
        return Q
    

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
        uv_consecutive_diff = 1.0

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
        Temp = TemperatureField(np.full((physN, physN), 1500), dx, False, thick)

        #Initializing the thickness of the diffusive zone.
        diffz_thick = 0.

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
                    misc.plot_field(Temp, True, title=f'T° Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(frame)+"_temp_field.png")
                    misc.plot_diffusive_zone(n2, self.y, frame, dt, time)

            elif frame%frame_period == frame_period-1 :
                uvcopy = misc.uv_copy(uv)

            # Temperature transport
            #   Temperature constant in time for the moment.
                
            # Population transport
            self._update_RK2_species(uv, o2)
            self._update_RK2_species(uv, n2)
            self._update_RK2_species(uv, ch4)
            self._update_RK2_species(uv, h2o)
            self._update_RK2_species(uv, co2)

            # Chemistry, impacting all species populations and temperature.
            mid_coord = N//2
            suivitime = [0.]
            suivio2 = [o2.values[mid_coord, mid_coord]]
            suivin2 = [n2.values[mid_coord, mid_coord]]
            suivich4 = [ch4.values[mid_coord, mid_coord]]
            suivih2o = [h2o.values[mid_coord, mid_coord]]
            suivico2 = [co2.values[mid_coord, mid_coord]]
            suiviT = [Temp.values[mid_coord, mid_coord]]

            for dtchem in self.dtchem_list:

                self._converge_all_species(o2, n2, ch4, h2o, co2, Temp, dtchem)
                if frame%frame_period == 0:
                    suivitime.append(suivitime[-1]+dtchem)
                    suivio2.append(o2.values[mid_coord, mid_coord])
                    suivin2.append(n2.values[mid_coord, mid_coord])
                    suivich4.append(ch4.values[mid_coord, mid_coord])
                    suivih2o.append(h2o.values[mid_coord, mid_coord])
                    suivico2.append(co2.values[mid_coord, mid_coord])
                    suiviT.append(Temp.values[mid_coord, mid_coord])

            if frame%frame_period == 0:
                fig1 = plt.figure()
                plt.plot(suivitime, suivio2, label="O2")
                plt.plot(suivitime, suivio2, label="O2")
                plt.plot(suivitime, suivio2, label="O2")
                plt.plot(suivitime, suivio2, label="O2")
                plt.plot(suivitime, suivio2, label="O2")
                plt.xlabel("Time (s)")
                plt.ylabel("Massic fraction")
                plt.legend()
                plt.show()

                fig2 = plt.figure()
                plt.plot(suivitime, suiviT)
                plt.xlabel("Time (s)")
                plt.ylabel("Temperature (K)")
                plt.show()

            o2.fillGhosts()
            n2.fillGhosts()
            ch4.fillGhosts()
            h2o.fillGhosts()
            co2.fillGhosts()
            Temp._fillGhosts()               

            # Fluid Flow
            self._update_RK2_velocity(uv, uvet)
            uvet.update_metric()
            self.ell_solver(uvet, Press)
            uv.u.values = uvet.u.values - (dt/rho)*Press.derivative_x()
            uv.v.values = uvet.v.values - (dt/rho)*Press.derivative_y()
            uv._fillGhosts()
            uv.update_metric()
            
            frame += 1

        # Plots of the final frame
        misc.plot_field(uv.v, True, title=f'V Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s \n Metric: {uv.metric:.5f}',saveaspng=str(frame)+"_v_field.png")
        misc.plot_field(uv.u, True, title=f'U Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s \n Metric: {uv.metric:.5f}',saveaspng=str(frame)+"_u_field.png")
        misc.plot_strain_rate(uv, self.y, title=f"Spatial representation of th strain rate along the slipping wall\nMax is {uv.max_strain_rate:.4e} Hz", saveaspng=str(frame)+"_strain_rate.png")
        misc.plot_field(Press, True, title=f'Pressure Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(frame)+"_press_field.png")
        misc.plot_field(o2, True, title=f'O2 Population k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(frame)+"_O2.png")
        misc.plot_field(n2, True, title=f'N2 Population k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(frame)+"_N2.png")
        misc.plot_field(ch4, True, title=f'CH4 Population k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(frame)+"_CH4.png")
        misc.plot_field(h2o, True, title=f'H2O Population k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(frame)+"_H2O.png")
        misc.plot_field(co2, True, title=f'CO2 Population k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s', saveaspng=str(frame)+"_CO2.png")
        misc.plot_field(Temp, True, title=f'T° Field k={frame} ($\Delta t=${dt}, N={N}) \n Time: {time:.6f} s',saveaspng=str(frame)+"_temp_field.png")
        misc.plot_diffusive_zone(n2, self.y, frame, dt, time)

        # About to print and write in final file the report of the simulation.
        if uv.metric >= self.div_crit :
            code = "divergence"        
        elif stop_datetime - start_datetime >= self.max_t_comput:
            code = "timeout"
        else:
            code = "success"

        misc.print_write_end_message(code, self.div_crit, self.max_t_comput, self.conv_crit, self.L, self.D, N, dt, stop_datetime-start_datetime, time, frame, uv_consecutive_diff, uv.max_strain_rate, n2.diff_zone_thick)

